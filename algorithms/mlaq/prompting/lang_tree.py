import os, sys, time, re

# Fix Windows SSL cert store corruption (affects torch/aiohttp imports)
try:
    import ssl, certifi
    ssl.create_default_context = (
        lambda purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None:
        ssl._create_default_https_context(purpose, cafile=certifi.where())
    )
except Exception:
    pass

from pathlib import Path
_DEFAULT_TREE = str(Path(__file__).resolve().parents[3] / "data" / "_scratch" / 'mlaq' / "lang_tree.pth")

import torch
import numpy as np


def preprocess_action(action):
    """
    格式化 LLM 输出的 action
    """
    action = action.replace("'''", "")
    action = action.replace("```", "")

    # 逐行去除行尾空格
    action = '\n'.join(line.rstrip() for line in action.split('\n'))

    for i in action:
        if i == '\n':
            action = action[1:]
        else:
            break
    for i in reversed(action):
        if i == '\n':
            action = action[:-1]
        else:
            break

    action = action + '\n'

    return action


def preprocess_state(state):
    """
    格式化 LLM 输出的 state
    """
    state = state.replace("'''", "")
    state = state.replace("```", "")
    state = state.replace('[Prediction] Begin', '')
    state = state.replace('[Prediction] End', '')

    # 逐行去除行尾空格
    state = '\n'.join(line.rstrip() for line in state.split('\n'))

    for i in state:
        if i == '\n':
            state = state[1:]
        else:
            break

    for i in reversed(state):
        if i == '\n':
            state = state[:-1]
        else:
            break

    state = state + '\n'

    return state


_JOINT_ACTION_RE = re.compile(r'^NAME\s+(\S+)\s+ACTION\s+(.*)$')


def parse_joint_action(action, agent_names):
    """
    把联合动作文本拆成 {agent: 个体动作}
    """
    out = {}
    for line in action.split('\n'):
        line = line.rstrip()
        if not line or line.upper().startswith('EXECUTE'):
            continue
        m = _JOINT_ACTION_RE.match(line)
        if not m:
            return None
        out[m.group(1)] = m.group(2).strip()

    if not out or set(out) != set(agent_names):
        return None

    return out


class Nodes:
    def __init__(self, state):
        """
        Nodes 类的成员属性
        """
        state = preprocess_state(state)

        self.state = state
        self.value = -1
        self.children = []
        self.avail_actions = []
        self.gamma = 0.995

    @property
    def num_children(self):
        return len(self.children)

    @property
    def num_avail_actions(self):
        return len(self.avail_actions)

    def add_child(self, action, child, virtual=False):
        if not virtual:
            action = preprocess_action(action)

        if action not in self.avail_actions:
            # 新动作则直接添加新边和子节点
            self.avail_actions.append(action)
            self.children.append({
                'action': action,
                'child': child
            })
        else:
            # 已有动作则更新子节点
            for temp_child in self.children:
                if temp_child['action'] == action:
                    temp_child['child'] = child
                    break

    def remove_child(self, action, virtual=False):
        if not virtual:
            action = preprocess_action(action)
        else:
            action = 'action_virtual'

        # 找到对应的子节点并删除
        for i in range(len(self.children)):
            if self.children[i]['action'] == action:
                self.children.pop(i)
                break

        # 找到对应的动作并删除
        for i in range(len(self.avail_actions)):
            if self.avail_actions[i] == action:
                self.avail_actions.pop(i)
                break

    def update_child(self, action, child):
        action = preprocess_action(action)

        for i in range(len(self.children)):
            if self.children[i]['action'] == action:
                self.children[i]['child'] = child
                break

    def get_children(self):
        # 获取所有边及其对应的子节点
        return self.children

    def get_child(self, action):
        # 获取指定边对应的子节点
        for child in self.children:
            if child['action'] == action:
                return child['child']

        return None

    def get_children_actions(self, without_virtual=False):
        """
        获取当前节点所有已记录过的动作
        """
        actions = []
        for child in self.children:
            if without_virtual and child['action'] == 'action_virtual':
                continue
            else:
                actions.append(child['action'])

        return actions

    def compute_value(self, with_virtual=False):
        """
        Nodes 类的成员函数
        计算当前 node 自己的 value 值
        """
        if with_virtual:
            if self.num_children == 1:
                return 0
            else:
                value = -1
                for child in self.children:
                    if child['action'] == 'action_virtual':
                        continue
                    else:
                        temp_value = child['child'].value * 0.995
                        if temp_value > value:
                            value = temp_value

                return value
        else:
            if self.num_children == 0:
                return None
            else:
                value = -1
                for child in self.children:
                    temp_value = child['child'].value * 0.995
                    if temp_value > value:
                        value = temp_value

                return value

    def get_best(self, with_ucb=False, without_virtual=False):
        """
        获取价值最大的动作及其对应的子节点
        """
        values = {}
        for child in self.children:
            if with_ucb:
                assert hasattr(child['child'], 'ucb_value'), 'The child node does not have ucb_value attribute'
                values[child['action']] = child['child'].ucb_value
            else:
                values[child['action']] = child['child'].value

        if without_virtual:
            values.pop('action_virtual')

        best_action = max(values, key=values.get)
        for child in self.children:
            if child['action'] == best_action:
                best_child = child['child']
                break

        return best_action, best_child


class LangTree:
    def __init__(self, load=False, if_temp_tree=False, tree_path=_DEFAULT_TREE,
                num_agents=1, q_alpha=None, q_gamma=0.995, agent_names=None,
                q_loop=1000, q_tol=1e-10):
        self.nodes = []
        self.if_temp_tree = if_temp_tree
        self.tree_path = tree_path

        # VDN 相关参数
        self.agent_names = list(agent_names) if agent_names else [f'agent{i}' for i in range(num_agents)]
        self.num_agents = len(self.agent_names)
        self.q_gamma = q_gamma
        # 写回时按 1/k 归一化，等效步长恒为 alpha，故 alpha 不再受 k_max 限制
        # （实测：归一化后 alpha <= 0.4 稳定、>= 0.5 发散，且与 k_max 无关）
        self.q_alpha = q_alpha if q_alpha is not None else 0.3
        self.q_loop = q_loop      # 迭代轮数上限（归一化 + alpha=0.3 下实测需 200~350 轮）
        self.q_tol = q_tol        # 收敛判据：max|delta| 小于此值即提前退出
        self.q_tables = [dict() for _ in range(self.num_agents)]   # 每个 agent 一张 Q 表 {(s, a_i): value}
        self.q_time = {'total': 0.0, 'calls': 0}   # 统计 Q 或 V 值的计算耗时
        self._joint_cache = {}   # 联合动作文本 -> {agent: 个体动作} 或 None，避免重复解析

        if load:
            self.load(tree_path)

        self.refine_tree()

    def refine_tree(self):
        """
        遍历所有节点的子节点，把不在树中的子节点都加入到树中
        修复引用断裂问题
        """
        for node in self.nodes:
            for child in node.children:
                if child['child'] not in self.nodes:
                    self.nodes.append(child['child'])
                    print(f'Node {child["child"].state} added to the tree')

        self.save()

    def save(self):
        Path(self.tree_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, self.tree_path)

    def load(self, path=_DEFAULT_TREE):
        try:
            temp_tree = torch.load(path, weights_only=False)
            self.nodes = temp_tree.nodes
        except:
            print('No saved tree found')

    def build_node(self, state):
        """
        根据 state 创建新节点或返回已存在的节点
        """
        state = preprocess_state(state)

        state_exist = False
        for node in self.nodes:
            if node.state == state:
                state_exist = True
                break

        if not state_exist:
            node = Nodes(state)
            if self.if_temp_tree:
                node.action_exhausted = False
                node.value = 0
                node.ucb_value = 0
                node.visit_num = 0

            self.nodes.append(node)

        return node

    def build_virtual_node(self, state):
        """
        为 state 节点创建一个虚拟子节点
        """
        assert self.if_temp_tree, 'This function is only for temporary tree'

        # 创建或获得当前状态对应的节点
        state_node = self.build_node(state)

        # 若当前状态节点已探索完所有动作
        if state_node.action_exhausted:
            return  # 则不再创建

        # 若当前状态节点已有虚拟子节点
        for child in state_node.children:
            if child['action'] == 'action_virtual':
                return  # 则不再创建

        # 创建虚拟子节点并添加到当前状态节点的子节点列表中
        action = 'action_virtual'
        next_state = state + '_sub_virtual'
        next_state_node = self.build_node(next_state)
        next_state_node.visit_num = 2  # 略低的探索偏好
        next_state_node.value = -0.1  # 略低的探索价值
        state_node.add_child(action, next_state_node, virtual=True)

    def update(self, transition, *, edge_metadata=None):
        state, action, next_state = transition
        state_node = self.build_node(state)
        next_state_node = self.build_node(next_state)
        state_node.add_child(action, next_state_node)
        if edge_metadata is not None:
            action_key = preprocess_action(action)
            edge = next(edge for edge in state_node.children if edge["action"] == action_key)
            edge.update(edge_metadata)
        if not self.if_temp_tree:
            self.save()

    def remove_node(self, state, action, virtual=False):
        state_node = self.get_node(state)
        next_state_node = state_node.get_child(action)
        state_node.remove_child(action, virtual=virtual)  # 删除 state 节点的 action 边及其对应的子节点 (删除关系)

        if next_state_node is not None:
            # 若仍有其他节点引用 next_state_node，则不删除 next_state_node (保留节点)
            remove_flag = True
            for node in self.nodes:
                for child in node.children:
                    if child['child'] == next_state_node:
                        remove_flag = False
                        break

            if remove_flag:  # 若没有其他节点引用 next_state_node，则删除 next_state_node (删除节点)
                self.nodes.pop(self.nodes.index(next_state_node))

        if not self.if_temp_tree:
            self.save()

    def get_node(self, state):
        state = preprocess_state(state)

        for node in self.nodes:
            if node.state == state:
                return node

        return None

    def check_transition_existence(self, transition):
        """
        检查 (s, a) 是否存在于树中
        """
        state, action, _ = transition
        state_node = self.build_node(state)
        for i in range(len(state_node.children)):
            if state_node.children[i]['action'] == action:
                return True

        return False

    def check(self, transition):
        """
        检查 (s, a, s') 中的 a 和 s' 是否预测错误
        transition 是与环境交互得到的真实值
        树中 node 的 action 和 child 等是 LLM 的预测值
        """
        state, action, next_state = transition
        state_node = self.build_node(state)
        next_state_node = self.build_node(next_state)
        for i in range(len(state_node.children)):
            if state_node.children[i]['action'] == action:
                # 若树中 LLM 预测的 s' 与真实的 s' 不同，则修正并记录错误
                if state_node.children[i]['child'].state != next_state:
                    state_node.children[i]['child'] = next_state_node
                    return False, {
                        'predicted_transition': [state, action, state_node.children[i]['child'].state],
                        'true_transition': transition
                    }
                else:
                    return True, None

        # if the program reaches here, it means that the action does not exist in the tree
        raise ValueError('Action does not exist in the tree')

    def compute_values(self, target_state, loop=5):
        _t0 = time.perf_counter()  # —— 计时：算价值总耗时/次数（对比用）——
        # initial the valuse of all nodes to 0 except target state
        target_state = preprocess_state(target_state)  # 先在树中找到目标状态节点并设置 value 为 1
        state_exist = False
        for node in self.nodes:
            if node.state == target_state:
                state_exist = True
                node.value = 1
            else:
                node.value = 0

        if not state_exist:
            raise ValueError('Target state does not exist in the tree')

        for _ in range(loop):  # 迭代更新 loop 次
            for node in self.nodes:
                if node.state != target_state:
                    temp_value = node.compute_value()  # 计算每个节点的 value 值
                    if temp_value is not None:
                        node.value = temp_value  # 更新每个节点的 value 值

        self.q_time['total'] += time.perf_counter() - _t0
        self.q_time['calls'] += 1

    def compute_ucb_values(self, loop=5):
        """
        计算每个节点的 UCB 值
        """
        # 计算每个节点的 value 值
        for _ in range(loop):
            for node in self.nodes:
                temp_value = node.compute_value()
                if temp_value is not None:
                    node.value = temp_value

        # 计算每个节点的 UCB 值
        for node in self.nodes:
            if node.visit_num == 0:
                raise ValueError('The visit number of the node should not be 0')

            # 如果当前节点有子节点
            if node.num_children > 0:
                # 获取当前节点的总被访问数
                total_num = 0
                for child in node.children:
                    child_node = child['child']
                    total_num += child_node.visit_num

                # 计算每个子节点的 UCB 值
                for child in node.children:
                    child_node = child['child']
                    child_node.ucb_value = child_node.value / child_node.visit_num + 2 * np.sqrt(
                                            np.log(total_num) / child_node.visit_num)
                    # node.ucb_value = node.value / node.visit_num + 2 * np.sqrt(np.log(node.visit_num) / node.visit_num)


    """
    --------------------------------------- VDN (Q_i(s, a_i)) ---------------------------------------
    """
    def _parse_joint(self, action):
        """
        把联合动作文本拆成 {agent: 个体动作} (带缓存版)
        """
        # 把 cache 别名到 self._joint_cache
        cache = getattr(self, '_joint_cache', None)
        if cache is None:
            cache = self._joint_cache = {}
        # 若 cache 中没有 action 则解析并添加到 cache
        if action not in cache:
            cache[action] = parse_joint_action(action, self.agent_names)
        # 返回拆解结果
        return cache[action]

    def get_individual_actions(self, state, agent):
        """
        根据状态 s 查询 agent 的已见个体动作
        """
        # 校验 agent 是否在 self.agent_names 中
        if agent not in self.agent_names:
            raise ValueError(f'Unknown agent: {agent}')

        # 获取状态 s 对应的节点
        node = self.get_node(state)
        if node is None:
            return []

        actions = []
        seen = set()
        # 遍历所有子节点以收集 agent 的个体动作
        for child in node.children:
            # 排除虚拟动作
            if child['action'] == 'action_virtual':
                continue

            # 拆解联合动作文本以获取 agent 的个体动作
            parts = self._parse_joint(child['action'])
            if parts is None:
                continue
            action = parts[agent]

            # (去重) 若该个体动作尚未见过则添加到 actions 列表
            if action not in seen:
                seen.add(action)
                actions.append(action)
        return actions

    def get_q_i(self, state, agent, individual_action):
        """
        获取 Q_i(s, a_i) 值
        """
        i = self.agent_names.index(agent)
        key = (preprocess_state(state), individual_action.strip())
        return self.q_tables[i].get(key)

    def get_q_tot(self, s, a):
        """
        获取 Q_tot(s,a) = Σ_i Q_i(s, a_i) 值
        """
        # 将联合动作文本拆成 {agent: 个体动作}
        pa = self._parse_joint(a)
        if pa is None:
            return None
        # 获取每个 agent 的 Q_i(s, a_i) 值并求和
        return sum(self.q_tables[i].get((s, pa[ag]), 0.0)
                    for i, ag in enumerate(self.agent_names))

    def get_max_q_tot(self, state):
        """
        获取状态 s 所有出边的最大 Q_tot 值
        """
        # 获取状态 s 对应的节点
        state_node = self.get_node(state)
        if state_node is None:
            return 0.0
        # 获取所有出边的 Q_tot 值
        vals = [q for q in (self.get_q_tot(state_node.state, child['action'])
                            for child in state_node.children
                            if child['action'] != 'action_virtual')
                if q is not None]
        # 返回最大 Q_tot 值
        return max(vals) if vals else 0.0

    def vdn_compute_q_values(self, target_state, loop=None, tol=None):
        """
        遍历式 VDN 计算 Q_i
        """
        loop = self.q_loop if loop is None else loop
        tol = self.q_tol if tol is None else tol
        _t0 = time.perf_counter()  # —— 计时：算 Q 值总耗时/次数（对比用）——
        target_state = preprocess_state(target_state)

        # 预计算每个条目 (i, s, a_i) 被几条边共用 —— 写回时的归一化因子。
        # 条目被 k 条边共用时一轮会收到 k 个 delta，按 1/k 归一化使等效步长恒为 alpha，
        # 否则等效步长是 k·alpha，alpha 就被 k_max 卡死（实测 k_max=8 时 1/N 直接发散）。
        share = {}
        for node in self.nodes:
            s = node.state
            for child in node.children:
                if child['action'] == 'action_virtual':
                    continue
                pa = self._parse_joint(child['action'])
                if pa is None:
                    continue
                for i, ag in enumerate(self.agent_names):
                    key = (i, s, pa[ag])
                    share[key] = share.get(key, 0) + 1

        for _ in range(loop):
            # 1) 计算每个节点的 max Q_tot
            maxq = {}
            # 遍历所有全局状态节点
            for node in self.nodes:
                # 获取所有出边的 Q_tot 值
                vals = [q for q in (self.get_q_tot(node.state, child['action'])
                                    for child in node.children
                                    if child['action'] != 'action_virtual')
                        if q is not None]
                # 记录最大 Q_tot 值
                maxq[node.state] = max(vals) if vals else 0.0

            # 2) 计算每条边的 delta
            updates = []
            # 遍历所有节点
            for node in self.nodes:
                s = node.state
                # 遍历所有子节点，计算每条边的 delta
                for child in node.children:
                    # 获取 a
                    a = child['action']
                    if a == 'action_virtual':
                        continue

                    # 拆解 a 得到 a_i
                    pa = self._parse_joint(a)
                    if pa is None:
                        continue

                    # 计算 qt = Σ_i Q_i(s, a_i)
                    qt = sum(self.q_tables[i].get((s, pa[ag]), 0.0)
                                for i, ag in enumerate(self.agent_names))

                    # 获取 s'
                    s_next = child['child'].state
                    # 计算 tgt = r + γ(1-d) * max Q_tot(s', a')
                    r = 1.0 if s_next == target_state else 0.0  # 稀疏奖励
                    d = 1.0 if s_next == target_state else 0.0  # 目标状态终止
                    tgt = r + self.q_gamma * (1 - d) * maxq.get(s_next, 0.0)

                    # 计算 delta = tgt - qt
                    updates.append((s, pa, tgt - qt))

            # 3) 同步写回各个 agent 的 Q_i 表
            # 先把每个条目本轮收到的 delta 求和，再按 1/k 归一化后一次性写回
            max_delta = 0.0
            acc = {}
            for s, pa, delta in updates:
                # 获取本轮最大更新幅度
                max_delta = max(max_delta, abs(delta))

                for i, ag in enumerate(self.agent_names):
                    key = (i, s, pa[ag])
                    acc[key] = acc.get(key, 0.0) + delta

            # 更新每个 agent 的 Q_i(s, a_i) 值（除以共用边数 k，使等效步长恒为 alpha）
            for (i, s, a_i), total in acc.items():
                k = share[(i, s, a_i)]
                key = (s, a_i)
                self.q_tables[i][key] = self.q_tables[i].get(key, 0.0) + (self.q_alpha / k) * total

            # 判断是否收敛
            if max_delta < tol:
                break

        self.q_time['total'] += time.perf_counter() - _t0
        self.q_time['calls'] += 1

    def vdn_compute_ucb_values(self, target_state, loop=None):
        """
        计算 UCB 值
        """
        # 计算 Q_i 值
        self.vdn_compute_q_values(target_state, loop=loop)

        # 计算 UCB 值
        for node in self.nodes:
            if node.num_children == 0:
                continue

            total_num = sum(max(1, child['child'].visit_num) for child in node.children)
            for child in node.children:
                child_node = child['child']
                visit = max(1, child_node.visit_num)

                # 计算 Q_tot 值
                q = self.get_q_tot(node.state, child['action'])

                # 拆不出 per-agent 动作的边 (只有虚拟边会走到)
                if q is None:
                    q = child_node.value

                child_node.ucb_value = q / visit + 2 * np.sqrt(np.log(max(1, total_num)) / visit)

    def vdn_get_best(self, state, with_ucb=False, without_virtual=False):
        """
        选择最优动作
        with_ucb=False  按照 Q_tot 选择最优动作
        with_ucb=True   按照 ucb_value 选择最优动作
        """
        state_node = self.get_node(state)
        if state_node is None:
            return None, None

        best_action, best_child, best_score = None, None, float('-inf')
        for child in state_node.children:
            action = child['action']

            # without_virtual 参数指明是否需要排除虚拟动作
            if without_virtual and action == 'action_virtual':
                continue

            if with_ucb:
                # 根据 UCB 值选择最优动作 —— LLM 想象阶段
                sc = getattr(child['child'], 'ucb_value', float('-inf'))
            else:
                # 根据 Q_tot 值选择最优动作 —— 环境交互阶段
                sc = self.get_q_tot(state_node.state, action)
                if sc is None:  # 拆不出 per-agent 动作的边不参与选择
                    continue

            if sc > best_score:
                best_score, best_action, best_child = sc, action, child['child']

        return best_action, best_child

    def if_same(self, state, target_state):
        state = preprocess_state(state)
        target_state = preprocess_state(target_state)

        return state == target_state


class Tree2HtmlTranslator:
    """
    HTML 可视化语言树
    """
    def __init__(self, tree: LangTree):
        self.tree = tree
        self.html = ''
        self.nodes_str = ''
        self.edges_str = ''
        self.edges = []

    def get_default_header(self):
        self.html += (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '  <head>\n'
            '    <meta charset="UTF-8" />\n'
            '    <title>Tutorial Demo</title>\n'
            '  </head>\n'
            '  <body>\n'
            '    <div id="mountNode"></div>\n'
            '    <script src="https://gw.alipayobjects.com/os/antv/pkg/_antv.g6-3.7.1/dist/g6.min.js"></script>\n'
            '    <!-- 4.x and later versions -->\n'
            '    <!-- <script src="https://gw.alipayobjects.com/os/lib/antv/g6/4.3.11/dist/g6.min.js"></script> -->\n'
            '    <script>\n'
            '      const graph = new G6.Graph({\n'
            '        container: \'mountNode\',\n'
            '        width: 1600,\n'
            '        height: 3200,\n'
            '        defaultNode: {\n'
            '          labelCfg: {\n'
            '            style: {\n'
            '              fill: \'#fff\',\n'
            '            },\n'
            '          },\n'
            '        },\n'
            '        defaultEdge: {\n'
            '          labelCfg: {\n'
            '            autoRotate: true,\n'
            '          },\n'
            '        },\n'
            '        nodeStateStyles: {\n'
            '          hover: {\n'
            '            fill: \'lightsteelblue\',\n'
            '          },\n'
            '          click: {\n'
            '            stroke: \'#000\',\n'
            '            lineWidth: 3,\n'
            '          },\n'
            '        },\n'
            '        edgeStateStyles: {\n'
            '          click: {\n'
            '            stroke: \'steelblue\',\n'
            '          },\n'
            '        },\n'
            '        layout: {\n'
            '          type: \'force\',\n'
            '          linkDistance: 100,\n'
            '          preventOverlap: true,\n'
            '          nodeStrength: -30,\n'
            '          edgeStrength: 0.1,\n'
            '        },\n'
            '        modes: {\n'
            '          default: [\'drag-canvas\', \'zoom-canvas\', \'drag-node\'],\n'
            '        },\n'
            '      });\n'
            '\n'
            '      const main = async () => {\n'
            '        const remoteData = {\n'
        )

    def get_default_footer(self):
        self.html += (
            "\n        const nodes = remoteData.nodes;\n"
            "        const edges = remoteData.edges;\n"
            "        nodes.forEach((node) => {\n"
            "          if (!node.style) {\n"
            "            node.style = {};\n"
            "          }\n"
            "          node.style.lineWidth = 1;\n"
            "          node.style.stroke = '#666';\n"
            "          node.style.fill = 'steelblue';\n"
            "          switch (node.class) {\n"
            "            case 'state': {\n"
            "              node.type = 'rect';\n"
            "              node.size = [180, 200];\n"
            "              break;\n"
            "            }\n"
            "            case 'c1': {\n"
            "              node.type = 'rect';\n"
            "              node.size = [35, 20];\n"
            "              break;\n"
            "            }\n"
            "          }\n"
            "        });\n"
            "        edges.forEach((edge) => {\n"
            "          if (!edge.style) {\n"
            "            edge.style = {};\n"
            "          }\n"
            "          edge.style.lineWidth = edge.weight;\n"
            "          edge.style.opacity = 0.6;\n"
            "          edge.style.stroke = 'grey';\n"
            "        });\n"
            "\n"
            "        graph.data(remoteData);\n"
            "        graph.render();\n"
            "\n"
            "        graph.on('node:mouseenter', (e) => {\n"
            "          const nodeItem = e.item;\n"
            "          graph.setItemState(nodeItem, 'hover', true);\n"
            "        });\n"
            "        graph.on('node:mouseleave', (e) => {\n"
            "          const nodeItem = e.item;\n"
            "          graph.setItemState(nodeItem, 'hover', false);\n"
            "        });\n"
            "        graph.on('node:click', (e) => {\n"
            "          const clickNodes = graph.findAllByState('node', 'click');\n"
            "          clickNodes.forEach((cn) => {\n"
            "            graph.setItemState(cn, 'click', false);\n"
            "          });\n"
            "          const nodeItem = e.item;\n"
            "          graph.setItemState(nodeItem, 'click', true);\n"
            "        });\n"
            "        graph.on('edge:click', (e) => {\n"
            "          const clickEdges = graph.findAllByState('edge', 'click');\n"
            "          clickEdges.forEach((ce) => {\n"
            "            graph.setItemState(ce, 'click', false);\n"
            "          });\n"
            "          const edgeItem = e.item;\n"
            "          graph.setItemState(edgeItem, 'click', true);\n"
            "        });\n"
            "      };\n"
            "      main();\n"
            "    </script>\n"
            "  </body>\n"
            "</html>\n"
        )

    def process_tree(self):
        for i, node in enumerate(self.tree.nodes):
            node.id = i

        for node in self.tree.nodes:
            state = node.state.replace('\n', '\\n')
            self.nodes_str += f"""    {{"id": "{node.id}", "label": "{state}", "class": "state"}},\n"""
            for child in node.children:
                self.edges.append((node.id, child['child'].id))
                action = child['action'].replace('\n', '\\n')
                self.edges_str += f"""    {{"source": "{node.id}", "target": "{child['child'].id}", "label": "{action}", "weight": 1.0}},\n"""

        self.nodes_str = '    "nodes": [\n' + self.nodes_str + '    ],\n'
        self.edges_str = '    "edges": [\n' + self.edges_str + '    ]\n'

    def translate(self):
        self.get_default_header()
        self.process_tree()
        self.html += self.nodes_str
        self.html += self.edges_str
        self.html += '}\n'
        self.get_default_footer()
        with open('tree.html', 'w') as f:
            f.write(self.html)


if __name__ == "__main__":
    # ===== 在此处填写要绘制的树文件路径（.pth），然后直接运行 =====
    # 例如: "data/sort_dialog/mlaq/20260729_1906/run_2/case_1/lang_tree.pth"
    tree_path = "data\\sort_dialog\\mlaq\\20260729_1906\\run_2\\case_1\\lang_tree.pth"
    tree = LangTree(load=False)
    try:
        # torch.load 需要找到 prompting.lang_tree.LangTree, 确保父目录在 sys.path
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        raw = torch.load(tree_path, map_location='cpu')
        tree.nodes = raw.nodes
        tree.if_temp_tree = raw.if_temp_tree
        print(f"Loaded {len(tree.nodes)} nodes from {tree_path}")
    except Exception as e:
        print(f"Warning: {e}")
    Tree2HtmlTranslator(tree).translate()
    print(f"Done: tree.html generated")
