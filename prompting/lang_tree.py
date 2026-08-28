import os, sys

# Fix Windows SSL cert store corruption (affects torch/aiohttp imports)
try:
    import ssl, certifi
    ssl.create_default_context = (
        lambda purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None:
        ssl._create_default_https_context(purpose, cafile=certifi.where())
    )
except Exception:
    pass

import torch
import numpy as np


def preprocess_action(action):
    """
    格式化 LLM 输出的 action
    """
    action = action.replace("'''", "")
    action = action.replace("```", "")
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

    def compute_distance(self, target_state):
        pass

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


class LangTree:
    def __init__(self, load=False, if_temp_tree=False, tree_path='data/lang_tree.pth'):
        self.nodes = []
        self.if_temp_tree = if_temp_tree
        self.tree_path = tree_path
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

    def init_values(self):
        """
        初始化各个节点的统计数值
        """
        for node in self.nodes:
            node.value = 0
            node.ucb_value = 0
            node.visit_num = 0

    def save(self):
        torch.save(self, self.tree_path)

    def load(self, path='data/lang_tree.pth'):
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

    def update(self, transition):
        state, action, next_state = transition
        state_node = self.build_node(state)
        next_state_node = self.build_node(next_state)
        state_node.add_child(action, next_state_node)
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
    tree_path = "data\\sort_dialog\\run_2\\case_1\\lang_tree.pth"
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
