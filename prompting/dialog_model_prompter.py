import os
import json
import copy
import time
import torch
import shutil
import openai
import tiktoken
import os.path as path
from typing import Dict
from datetime import datetime

from prompting.lang_tree import LangTree, preprocess_state, preprocess_action, parse_joint_action
from prompting.paths import exp_output_dir
from tools.minimax_quota import handle_api_error
from tools.q_consistency import QConsistencyRecorder
from rocobench.envs import EnvState

# assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
# OPENAI_KEY = str(json.load(open("openai_key.json")))
# openai.api_key = OPENAI_KEY


PATH_PLAN_INSTRUCTION = """
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incorporate [Environment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""


class DialogModelPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent
    """
    def __init__(
            self,
            env, parser, feedback_manager,
            epoch_id=0,
            data_dir: str = "data",
            max_tokens: int = 512,
            debug_mode: bool = False,
            use_waypoints: bool = False,
            robot_name_map: Dict[str, str] = {"panda": "Bob"},
            num_replans: int = 3,
            max_calls_per_round: int = 10,
            use_history: bool = True,
            use_feedback: bool = True,
            temperature: float = 0,
            llm_source: str = "gpt-4",
            load: bool = False,
            skip_check_memory: bool = False,
            algo: str = "mlaq",     # "mlaq"=值迭代 / "vdn-mlaq"=遍历式 VDN（精确匹配 == 'vdn-mlaq'）
            compare_q: bool = False,  # 旁路记录两套 Q 与动作选择的一致性（见 tools/q_consistency.py）
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.algo = algo   # "mlaq"(值迭代) / "vdn-mlaq"(遍历式 VDN)
        # 旁路记录器：只读树、不写节点属性，关闭时为 None（不产生任何开销）
        self.q_recorder = QConsistencyRecorder() if compare_q else None
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = []
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round
        self.temperature = temperature
        self.llm_source = llm_source
        self.memory_path = path.join(exp_output_dir(self.env.name, epoch_id, data_dir), 'memory.pth')

        self.skip_check_memory = skip_check_memory
        self.state_desp = None

        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.last_end_step = 100

        # add the assumed tokens of each round into this list
        self.consumed_token = {
            'input_token': [],
            'output_token': []
        }
        # this memory ratio is used to statistics the calling frequency of the memory's transition data
        self.memory_ratio = {
            'llm_count': [],
            'memory_count': []
        }
        # this dict is used for temporary storage of the assumed tokens of each round
        self.temp_consumed_token = {
            'input_token': 0,
            'output_token': 0
        }
        self.temp_memory_ratio = {
            'llm_count': 0,
            'memory_count': 0
        }

        if load:
            try:
                self.memory = torch.load(self.memory_path)
            except:
                self.memory = {
                    'prediction': {},
                    'policy': {},
                }
        else:
            self.memory = {
                'prediction': {},
                'policy': {},
            }

        self.temporary_memory = {
            'prediction': {},
            'policy': {},
        }

        self.epoch_id = epoch_id
        self.state = None
        self.action = None

        # 实例化永久语言树
        self.lang_tree = LangTree(
            load=load,
            if_temp_tree=False,
            tree_path=path.join(exp_output_dir(self.env.name, epoch_id, data_dir), 'lang_tree.pth'),
            agent_names=self.robot_agent_names,
            algo=self.algo,
        )

        self.temp_lang_tree = LangTree(load=False, if_temp_tree=True, agent_names=self.robot_agent_names, algo=self.algo)
        self.case_dir, self.simulate_save_dir, self.interact_save_dir = None, None, None
        self.round_save_dir, self.env_round_save_dir = None, None
        self.round_index = 1
        self.env_round_index = 1
        self.interaction_failure = 0
        self.imagination_failure = 0
        self.env_replans = 0

        assert llm_source in ["gpt-4", "gpt-3.5-turbo", 'gpt-4-1106-preview', 'gpt-4-0125-preview',  "claude", "deepseek-v4-flash", "deepseek-v4-pro", "MiniMax-M3"], \
            f"llm_source must be one of [gpt4, gpt-3.5-turbo, gpt-4-1106-preview, gpt-4-0125-preview, claude, deepseek-v4-flash, deepseek-v4-pro, MiniMax-M3], got {llm_source}"

    """
    --------------------------------------- 语言树管理 (起) ---------------------------------------
    """

    # ===== Q 一致性旁路记录（--compare_q 开启时才生效；逻辑在 tools/q_consistency.py）=====
    def _qcompare(self, tree, *, owner=None, ucb=False, loop=20):
        """主算法算完后，只在副本上计算对照算法；不计入主算法耗时。"""
        if self.q_recorder is None:
            return
        try:
            self.q_recorder.observe_compute(
                tree, self.algo, self.env.target_state_desp,
                owner=owner, ucb=ucb, loop=loop)
        except Exception as e:
            self.log(f'[q_consistency] comparison failed: {e}')

    def _qrec(self, tree, state, phase, actual_action, site=None, ucb=False,
              without_virtual=False, owner=None):
        """记录一次 get_best：两套 Q/UCB + 各自选中的动作。记录失败不影响实验。"""
        if self.q_recorder is None:
            return
        if site is None:
            import sys as _sys
            site = _sys._getframe(1).f_code.co_name   # 调用点所在函数名
        try:
            self.q_recorder.record_get_best(tree, state, phase, site,
                                            algo=self.algo, actual_action=actual_action,
                                            owner=owner, ucb=ucb,
                                            without_virtual=without_virtual)
        except Exception as e:
            self.log(f'[q_consistency] record_get_best failed: {e}')

    def finish_q_consistency(self):
        """case 收尾：整棵树的 Q 表 → <case_dir>/q_consistency.xlsx"""
        if self.q_recorder is None:
            return None
        try:
            self.q_recorder.record_case_end(self.lang_tree, algo=self.algo)
        except Exception as e:
            self.log(f'[q_consistency] case_end snapshot failed: {e}')
        try:
            return self.q_recorder.finish_case(self.lang_tree)
        except Exception as e:
            self.log(f'[q_consistency] export failed: {e}')
            return None

    def check_path_exist(self, state_desp, target_state_desp):
        """
        判断永久树中是否存在从当前状态到目标状态的最优轨迹
        algo='vdn-mlaq'：Q_tot > 阈值；algo='mlaq'：值迭代 value > 0.5
        """
        if self.algo == 'vdn-mlaq':
            # VDN：get_max_q_tot 对不在树的状态返回 0 → False，等价原 None 判断
            self.lang_tree.vdn_compute_q_values(target_state_desp)
            self._qcompare(self.lang_tree, loop=10)
            return self.lang_tree.get_max_q_tot(state_desp) > 0.5
        # MLAQ（值迭代）
        lang_tree = copy.deepcopy(self.lang_tree)  # 深拷贝 (避免影响永久树上的 value 值)
        state_node = lang_tree.get_node(state_desp)  # 根据状态描述获取当前节点
        target_state_desp_node = lang_tree.get_node(target_state_desp)  # 根据状态描述获取目标节点
        if target_state_desp_node is None or state_node is None:
            return False
        lang_tree.compute_values(target_state=target_state_desp, loop=10)
        self._qcompare(lang_tree, owner=self.lang_tree, loop=10)
        return state_node.value > 0.5

    def _build_temp_virtual_node(self, state_desp):
        """VDN 的目标/失败状态不提供新的全局或个体想象入口。"""
        if self.algo == 'vdn-mlaq':
            state = preprocess_state(state_desp)
            if (state in (preprocess_state(self.env.target_state_desp),
                          preprocess_state('Success'), preprocess_state('Failure'))
                    or self.env.if_fail(state_desp)):
                self.temp_lang_tree.build_node(state_desp)
                self.temp_lang_tree.remove_node(state_desp, 'action_virtual', virtual=True)
                return
        self.temp_lang_tree.build_virtual_node(state_desp)

    def initial_temp_lang_tree(self, state_desp, target_state_desp):
        """
        初始化临时语言树
        检查永久树中是否存在最优轨迹，若有则在临时树里添加，并计算 UCB 值
        """
        self.temp_lang_tree = LangTree(load=False, if_temp_tree=True, agent_names=self.robot_agent_names, algo=self.algo)  # 创建临时语言树

        self._build_temp_virtual_node(state_desp)  # 为初始状态节点创建虚拟子节点
        self.temp_lang_tree.get_node(state_desp).visit_num += 1

        # 判断是否存在最优轨迹
        path_exist = self.check_path_exist(state_desp, target_state_desp=target_state_desp)  # 在语言树上粗略计算状态价值
        
        # 若从初始状态到目标状态存在最优轨迹，则把该轨迹复制到临时语言树上
        if path_exist:
            if self.algo == 'vdn-mlaq':
                self.lang_tree.vdn_compute_q_values(target_state_desp)
            else:
                self.lang_tree.compute_values(target_state=preprocess_state(target_state_desp), loop=20)

            self._qcompare(self.lang_tree)

            # 把永久树上的最优轨迹复制到临时树上
            reach_target = self.algo == 'vdn-mlaq' and preprocess_state(state_desp) == preprocess_state(target_state_desp)
            visited_states = set() if self.algo == 'vdn-mlaq' else None
            while not reach_target:
                state_node = self.lang_tree.get_node(preprocess_state(state_desp))  # 在永久树上找到初始状态对应的节点
                if self.algo == 'vdn-mlaq' and (state_node is None or state_node.state in visited_states):
                    break
                if self.algo == 'vdn-mlaq':
                    visited_states.add(state_node.state)
                if self.algo == 'vdn-mlaq':
                    action, next_state_node = self.lang_tree.vdn_get_best(state_node.state)
                else:
                    action, next_state_node = state_node.get_best()
                if self.algo == 'vdn-mlaq' and next_state_node is None:
                    break
                self._qrec(self.lang_tree, state_node.state, 'imagine', action)

                # 把 (s, a, s') 复制到临时树上
                self.temp_lang_tree.update(transition=[state_node.state, action, next_state_node.state])  # 把 (s, a, s') 加入到临时树上
                self._build_temp_virtual_node(next_state_node.state)  # 为 s' 创建虚拟子节点
                self.temp_lang_tree.get_node(next_state_node.state).visit_num += 1

                state_desp = next_state_node.state  # 指针移动到下一节点

                # 判断是否到达目标状态节点
                if preprocess_state(state_desp) == preprocess_state(target_state_desp):
                    reach_target = True

                    self.temp_lang_tree.update(transition=[state_desp, 'nothing', 'Success'])
                    self.temp_lang_tree.remove_node(state_desp, 'action_virtual', virtual=True)  # 移除目标状态节点的虚拟子节点
                    success_node = self.temp_lang_tree.get_node('Success')
                    success_node.visit_num += 1
                    success_node.value = 1

            if self.algo == 'vdn-mlaq':
                self.temp_lang_tree.vdn_compute_ucb_values(target_state_desp)  # 计算临时树上每个节点的 UCB 值（VDN 版）
            else:
                self.temp_lang_tree.compute_ucb_values(loop=20)
            self._qcompare(self.temp_lang_tree, ucb=True)
        elif self.algo == 'vdn-mlaq':
            self.temp_lang_tree.vdn_compute_ucb_values(target_state_desp)

    def fill_temp_lang_tree(self, state_desp, target_state_desp):
        """
        填充临时语言树
        检查永久树中是否存在新发现的最优轨迹，若有则在临时树里覆盖或添加，并刷新 UCB 值
        """
        self._build_temp_virtual_node(state_desp)
        # 若对临时树中来说是新节点，则将其 visit_num 置为 1
        if self.temp_lang_tree.get_node(state_desp).visit_num == 0:
            self.temp_lang_tree.get_node(state_desp).visit_num = 1

        # 判断是否存在最优轨迹
        path_exist = self.check_path_exist(state_desp, target_state_desp=target_state_desp)

        # 若从当前状态到目标状态存在最优轨迹，则把该轨迹复制到临时语言树上
        if path_exist:
            if self.algo == 'vdn-mlaq':
                self.lang_tree.vdn_compute_q_values(target_state_desp)
            else:
                self.lang_tree.compute_values(target_state=preprocess_state(target_state_desp), loop=20)

            self._qcompare(self.lang_tree)

            # 把永久树上的最优轨迹复制到临时树上
            reach_target = self.algo == 'vdn-mlaq' and preprocess_state(state_desp) == preprocess_state(target_state_desp)
            visited_states = set() if self.algo == 'vdn-mlaq' else None
            while not reach_target:
                state_node = self.lang_tree.get_node(preprocess_state(state_desp))  # 在永久树上找到当前状态对应的节点
                if self.algo == 'vdn-mlaq' and (state_node is None or state_node.state in visited_states):
                    break
                if self.algo == 'vdn-mlaq':
                    visited_states.add(state_node.state)
                if self.algo == 'vdn-mlaq':
                    action, next_state_node = self.lang_tree.vdn_get_best(state_node.state)
                else:
                    action, next_state_node = state_node.get_best()
                if self.algo == 'vdn-mlaq' and next_state_node is None:
                    break
                self._qrec(self.lang_tree, state_node.state, 'imagine', action)

                self.temp_lang_tree.update(transition=[state_node.state, action, next_state_node.state])  # 把 (s, a, s') 加入到临时树上
                self._build_temp_virtual_node(next_state_node.state)  # 为 s' 创建虚拟子节点

                # 若对临时树中来说是新节点，则将其 visit_num 置为 1
                if self.temp_lang_tree.get_node(state_node.state).visit_num == 0:
                    self.temp_lang_tree.get_node(state_node.state).visit_num = 1
                if self.temp_lang_tree.get_node(next_state_node.state).visit_num == 0:
                    self.temp_lang_tree.get_node(next_state_node.state).visit_num = 1

                state_desp = next_state_node.state  # 指针移动到下一节点

                self.temp_memory_ratio['memory_count'] += 1  # 统计使用已知 transition 的次数

                if preprocess_state(state_desp) == preprocess_state(target_state_desp):
                    reach_target = True

                    self.temp_lang_tree.update(transition=[state_desp, 'nothing', 'Success'])
                    self.temp_lang_tree.remove_node(state_desp, 'action_virtual', virtual=True)
                    success_node = self.temp_lang_tree.get_node('Success')  # 移除目标状态节点的虚拟子节点
                    # 若对临时树中来说是新节点，则将其 visit_num 置为 1
                    if success_node.visit_num == 0:
                        success_node.visit_num += 1
                    success_node.value = 1

        if self.algo == 'vdn-mlaq':
            self.temp_lang_tree.vdn_compute_ucb_values(target_state_desp)
        else:
            self.temp_lang_tree.compute_ucb_values(loop=20)
        self._qcompare(self.temp_lang_tree, ucb=True)

    """
    --------------------------------------- 语言树管理 (终) ---------------------------------------
    """

    def get_end_step(self, target_state_desp):
        """
        计算最优轨迹的步数，若不存在最优轨迹则返回 100
        仅用于效率评估
        """
        self._build_temp_virtual_node(self.state_desp)
        path_exist = self.check_path_exist(self.state_desp, target_state_desp=target_state_desp)
        end_step = 0
        if path_exist:
            if self.algo == 'vdn-mlaq':
                self.lang_tree.vdn_compute_q_values(target_state_desp)
            else:
                self.lang_tree.compute_values(target_state=preprocess_state(target_state_desp), loop=20)
            self._qcompare(self.lang_tree)
            reach_target = False
            state_desp = self.state_desp
            if self.algo == 'vdn-mlaq' and preprocess_state(state_desp) == preprocess_state(target_state_desp):
                return 0
            visited_states = set() if self.algo == 'vdn-mlaq' else None
            while not reach_target:
                state_node = self.lang_tree.get_node(preprocess_state(state_desp))
                if self.algo == 'vdn-mlaq' and (state_node is None or state_node.state in visited_states):
                    return 100
                if self.algo == 'vdn-mlaq':
                    visited_states.add(state_node.state)
                if self.algo == 'vdn-mlaq':
                    action, next_state_node = self.lang_tree.vdn_get_best(state_node.state)
                else:
                    action, next_state_node = state_node.get_best()
                if self.algo == 'vdn-mlaq' and next_state_node is None:
                    return 100
                self._qrec(self.lang_tree, state_node.state, 'evaluation', action)
                end_step += 1
                state_desp = next_state_node.state

                if preprocess_state(state_desp) == preprocess_state(target_state_desp):
                    reach_target = True

        if end_step == 0:
            # MLAQ 保留原先无路径时隐式返回 None 的行为。
            return 100 if self.algo == 'vdn-mlaq' else None
        return end_step

    def round_reset(self):
        self.state = None
        self.action = None
        self.temporary_memory = copy.deepcopy(self.memory)  # 重置临时 memory

    def case_reset(self):
        self.interaction_failure = 0
        self.imagination_failure = 0
        self.env_replans = 0
        self.last_end_step = 100
        self.lang_tree.q_time = {'total': 0.0, 'calls': 0}   # 清零本 case 的"算 Q/V 值"计时

        # 获取永久记忆 (包含 policy 和 prediction 错误历史) 的副本 (避免在测试用例中修改永久记忆)
        self.temporary_memory = copy.deepcopy(self.memory)

        # 用于统计每个用例中 LLM 想象所耗 token 数量
        self.consumed_token = {
            'input_token': [],
            'output_token': []
        }
        # 用于统计每个用例中 LLM 想象次数和复用 memory 的次数
        self.memory_ratio = {
            'llm_count': [],
            'memory_count': []
        }

        self.round_index = 1  # 控制 imagination 的次数
        self.state_desp = None

        # 旁路记录：开启一个新 case 的 Q 一致性记录
        if self.q_recorder is not None:
            self.q_recorder.start_case(self.case_dir, self.env.target_state_desp)

    def get_summary(self):
        return copy.deepcopy(self.consumed_token), copy.deepcopy(self.memory_ratio)

    """
    --------------------------------------- 设置输出目录相关 ---------------------------------------
    """

    def set_case_dir(self, save_dir):
        assert save_dir is not None, "save_dir cannot be None"
        
        # 创建用例目录
        self.case_dir = save_dir
        if not os.path.exists(self.case_dir):
            os.makedirs(self.case_dir)

        # 在用例目录下创建 simulate 目录
        self.simulate_save_dir = path.join(self.case_dir, "simulate")
        if not os.path.exists(self.simulate_save_dir):
            os.makedirs(self.simulate_save_dir)

        # 在用例目录下创建 interact 目录
        self.interact_save_dir = path.join(self.case_dir, "interact")
        if not os.path.exists(self.interact_save_dir):
            os.makedirs(self.interact_save_dir)

    def set_round_dir(self, round_save_dir):
        assert round_save_dir is not None, "round_save_dir cannot be None"
        self.round_save_dir = round_save_dir
        if not os.path.exists(self.round_save_dir):
            os.makedirs(self.round_save_dir)

    def set_env_round_dir(self, env_round_save_dir):
        assert env_round_save_dir is not None, "env_round_save_dir cannot be None"
        self.env_round_save_dir = env_round_save_dir
        if not os.path.exists(self.env_round_save_dir):
            os.makedirs(self.env_round_save_dir)

    """
    --------------------------------------- 记录日志相关 ---------------------------------------
    """

    def log(self, message, file_name="/log.txt", if_test=False):
        if if_test:
            save_file_name = self.env_round_save_dir + file_name
        else:
            save_file_name = self.round_save_dir + file_name

        with open(save_file_name, "a") as f:
            f.write(message + "\n")

        print(message)

    """
    --------------------------------------- LLM 提示词相关 ---------------------------------------
    """
    def get_role_prompt(self, role: str, state=None, action=None, next_state=None, info=None) -> dict:
        # role: predictor, action_checker, prediction_checker, policy
        if role == 'predictor':
            prompt = self._get_predictor_prompt(state, action, info)
        elif role == 'action_checker':
            prompt = self._get_action_checker_prompt(state, action)
        elif role == 'prediction_checker':
            prompt = self._get_prediction_checker_prompt(state, action, next_state)
        else:
            raise ValueError(
                f"role must be one of ['predictor', 'action_checker', 'prediction_checker', 'policy'], got {role}"
            )

        return prompt

    def _get_predictor_prompt(self, state, action, info) -> dict:
        """
        LLM 预测下一状态 s' 的提示词
        """
        state_space_desp = self.env.get_state_space_prompt()
        memory_prompt = '[Prediction Mistakes]\n'
        if preprocess_state(state) in self.memory['prediction'].keys():
            for memory in self.memory['prediction'][preprocess_state(state)]:
                wrong_prediction, true_prediction = memory
                memory_prompt += (
                    f"In the current state and action, you have outputted a wrong prediction. The prediction is as follows:\n"
                    f"{wrong_prediction}\n"
                    f"The true (correct) prediction is:\n"
                    f"{true_prediction}\n"
                )
        else:
            memory_prompt += 'No prediction mistakes.'

        system_prompt = (
            f"You are a state predictor. You will be given a [State] and an [Action] as input. should predict the [State] after the [Action] is applied to the given [State]. You should follow the [Predicting Instruction] to predict the [State].\n"
            f"{self.env.get_predicting_prompt()}\n"
        )
        system_prompt = (
            f"{system_prompt}\n"
            f"The following lists the [State Space Definition].\n"
            f"{state_space_desp}\n"
            f"{memory_prompt}\n"
            f"{info['feedbacks']}\n"
            f"Follow the [Predicting Instruction] strictly and think step-by-step."
        )
        user_prompt = (
            f"Now, you should predict the next [State] after the following [Action] is applied to the given [State]. You have to follow the instructions strictly rather than your own thoughts.\n"
            f"{state}\n"
            f"{action}\n"
        )
        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def _get_action_checker_prompt(self, state: str, action: str) -> dict:
        """
        LLM 动作检查器
        """
        action_constrains = self.env.get_action_constrains(state)
        task_prompt = self.env.get_task_prompt()
        memory_prompt = '[Action Mistakes]\n'
        if preprocess_state(state) in self.memory['policy'].keys():
            for memory in self.memory['policy'][preprocess_state(state)]:
                wrong_action, feedback = memory
                memory_prompt += (
                    f"In the current state, you mistakenly believed that a wrong action was the correct one. The action is as follows:\n"
                    f"{wrong_action}\n"
                    f"The environment provides the [Feedback] as follows:\n"
                    f"{feedback}\n"
                )
        else:
            memory_prompt += 'No action mistakes.'

        system_prompt = (
            f"You are an action checker. You should check if the [Action] is valid based on the given [State].\n"
            f"You should follow the check instructions in [Detailed Constraints for Action] to check the [Action] step-by-step. Take the mistakes in the [Action Mistakes] into consideration.\n"
        )
        system_prompt = (
            f"{system_prompt}\n"
            f"The following lists the [Task Description], and [Detailed Constraints for Action]. \n"
            f"{task_prompt}\n"
            f"{action_constrains}\n"
            f"{memory_prompt}\n"
        )
        system_prompt += f"Follow the [Detailed Constraints for Action] strictly and think step-by-step, then give the [Conclusion]\n"

        user_prompt = (
            f"Now you should check the following [Action] based on the given [State]\n"
            f"{state}\n"
            f"{action}\n"
            f"{self.env.get_check_tail()['action']}\n"
        )
        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def _get_prediction_checker_prompt(self, state: str, action: str, prediction: str) -> dict:
        """
        LLM 预测检查器
        """
        memory_prompt = '[Prediction Mistakes]\n'
        search_feature = preprocess_state(state) + preprocess_action(action)
        if search_feature in self.memory['prediction'].keys():
            for memory in self.memory['prediction'][search_feature]:
                wrong_prediction, true_prediction = memory
                memory_prompt += (
                    f"In the current state and action, you mistakenly believed that a wrong prediction was the correct one. The prediction is as follows:\n"
                    f"{wrong_prediction}\n"
                    f"The true (correct) prediction is:\n"
                    f"{true_prediction}\n"
                )
        else:
            memory_prompt += 'No prediction mistakes.'

        system_prompt = (
            f"You are a prediction checker. You will receive a [State], an [Action], and the subsequent [Prediction].\n"
            f"You should follow the instructions in [Detailed Constraints for Prediction] to check if the [Prediction] is valid based on the given [State] and [Action]. Take the mistakes in the [Prediction Mistakes] into consideration.\n"
        )
        system_prompt = (
            f"{system_prompt}\n"
            f"{self.env.get_state_template()}\n"
            f"{self.env.get_prediction_constrains()}\n"
            f"{memory_prompt}\n"
        )
        system_prompt += f"Follow the [Detailed Constraints for Prediction] strictly and think step-by-step, then give the [Conclusion]\n"

        user_prompt = (
            f"Now you should check the following [Action] based on the given [State] \n"
            f"The original state is:\n"
            f"{state}\n"
            f"The action is:\n"
            f"{action}\n"
            f"The prediction is:\n"
            f"{prediction}\n"
            f"{self.env.get_check_tail()['predicting']}\n"
        )

        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def get_forbidden_action_prompt(self, agent: str, forbidden_actions: list, fixed_actions=None):
        if fixed_actions is not None and self.algo != 'vdn-mlaq':
            raise ValueError('Fixed-agent dialogue requires algo="vdn-mlaq"')
        prompt = '[Forbidden Actions]\n'
        if len(forbidden_actions) == 0:
            prompt += 'No forbidden actions.'
        else:
            prompt += 'The following joint actions are forbidden:\n'
            for i, action in enumerate(forbidden_actions):
                prompt += '[Forbidden Action {}]\n If '.format(i + 1)
                agent_names = self.env.agent_names
                agent_action = action.split('\n')
                action_dict = {}
                for temp_agent in agent_names:
                    for temp_action in agent_action:
                        if temp_agent in temp_action:
                            action_dict[temp_agent] = temp_action

                logged_agens = []
                for temp_agent in agent_names:
                    if temp_agent == agent:
                        continue
                    else:
                        if len(logged_agens) != 0:
                            prompt += ', '
                        prompt += f'{temp_agent} chooses {action_dict[temp_agent]}'
                        logged_agens.append(temp_agent)

                prompt += f' simultaneously! Then, you cannot choose {action_dict[agent]}.\n'

        if fixed_actions:
            prompt += (
                '1. Choose a joint action outside the list above. Only participating agents may '
                'change their actions; all actions in [Fixed Actions] must remain unchanged.\n'
            )
        else:
            prompt += '1. You should try your best to output an action that is not in the list above. However, if you insist on choosing a forbidden action, you can persuade other agents to modify their actions, but you need to pay attention to whether there is still an available action\n'
        prompt += (
            '2. IMPORTANT: If there is no available action after 2 rounds of chat with other agents, you need to output an [ERROR]!!\n'
        )

        return prompt

    def _extract_action(self, action_prompt: str, info=None) -> str:
        prompt = (
            f"There is a description of an action space description and the action space definition as follows:\n"
            f"[----- Action Space Description Begin -----]\n"
            f"{self.env.get_action_space_prompt()}.\n"
            f"[----- Action Space Description End -----]\n"
            f"Now, I will provide you the decision process of the agent as follows:\n"
            f"[----- Decision Process Begin -----]\n"
            f"{action_prompt}\n"
            f"[----- Decision Process End -----]\n"
            f"Now, read the above paragraphs especially the conclusion, and help me to extract the [Action] in the format of [Action Space Definition]. If there is no action, please output [ERROR].\n"
            f"The only thing you should output is the [Action] and do not include any other information. Please note that do not include any marks like quotations."
        )
        response = self.new_query_once(prompt, use_stream=True, info=info)

        return response

    def _extract_state(self, state_prompt: str, info=None) -> str:
        prompt = (
            f"There is a description of a state space as follows:\n"
            f"[----- State Space Description Begin -----]\n"
            f"{self.env.get_state_space_prompt()}.\n"
            f"[----- State Space Description End -----]\n"
            f"Now, I will provide you the predicting process of the agent as follows:\n"
            f"[----- Predicting Process Begin -----]\n"
            f"{state_prompt}\n"
            f"[----- Predicting Process End -----]\n"
            f"Now, read the above paragraphs especially the conclusion, and help me to extract the [State] in the format of [State Space Definition].\n"
            f"The only thing you should output is the [State] and do not include any other information. Please note that do not include any marks like quotations."
        )
        response = self.new_query_once(prompt, use_stream=True, info=info)

        return response

    """
    --------------------------------------- LLM 调用相关 (起) ---------------------------------------
    """

    def compose_system_prompt(
            self,
            state_desp: str,
            agent_name: str,
            action_history = [],
            forbidden_actions = [],
            chat_history = [],  # chat from previous replan rounds
            current_chat = [],  # chat from current round, this comes AFTER env feedback
            feedback_history = [],
            active_agents=None,
            fixed_actions=None
    ) -> str:
        """
        组装系统提示词
        """
        if active_agents is not None or fixed_actions is not None:
            active_agents, fixed_actions = self._get_dialog_agents(active_agents, fixed_actions)
        # 获取动作空间描述
        action_desp = self.env.get_action_prompt()
        # 若使用中间路径点则需加入路径规划指令
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        # 设置环境反馈作为提示词
        execute_feedback = ""
        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"

        # 获取角色提示词
        agent_prompt = self.env.get_agent_prompt(state_desp, agent_name)

        # 设置对话历史作为提示词
        chat_history = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""

        # 设置系统提示词 = 动作空间描述 + 环境反馈 + 角色提示词 + 对话历史
        system_prompt = f"{action_desp}\n{execute_feedback}{agent_prompt}\n{chat_history}\n"

        # 设置 LLM Checker 反馈作为提示词
        if self.use_feedback and len(feedback_history) > 0:
            for i, feedback in enumerate(feedback_history):
                system_prompt += f"[Feedback {i + 1}]" + '\n'
                system_prompt += feedback['action'] + 'fails, and the feedback is:' + '\n'
                system_prompt += feedback['feedback'] + '\n'

        # 设置动作历史作为提示词
        if len(action_history) > 0:
            system_prompt += f"[Action History]\n"
            for i, action in enumerate(action_history):
                system_prompt += f"Round {i + 1}:\n{action}\n"

        # 设置禁用动作作为提示词
        if len(forbidden_actions) > 0:
            forbidden_action_prompt = self.get_forbidden_action_prompt(agent_name, forbidden_actions, fixed_actions)
            system_prompt += forbidden_action_prompt

        if active_agents is not None:
            system_prompt += (
                '[Participating Agents]\n' + ', '.join(active_agents) + '\n'
                'Only these agents take turns discussing and choosing their actions.\n'
                '[Fixed Actions]\n'
            )
            for name, action in (fixed_actions or {}).items():
                system_prompt += f'NAME {name} ACTION {action}\n'
            system_prompt += (
                'Fixed agents do not speak. Do not change or ask to change their actions. '
                'Coordinate your choices with these fixed actions.\n'
                'For this round, dialogue order and approval requirements apply only to participating agents. '
                'Fixed actions are already selected. Do not wait for a response or approval from fixed agents.\n'
                'The final EXECUTE plan must contain exactly one action line for EVERY agent, '
                'including every fixed action copied unchanged. Do not output only participating agents.\n'
            )

        # 设置当前对话作为提示词
        if len(current_chat) > 0:
            system_prompt += "[Current Chat]\n" + "\n".join(current_chat) + "\n"

        return system_prompt

    def new_query_once(self, prompt, max_query=10, use_stream=False, info=None):
        """
        LLM 调用接口
        """
        if type(prompt) == str:
            # 注：MiniMax 等厂商要求 messages 里至少有一条 user 消息，
            # 只发 system 会报 "chat content is empty (2013)"；故 str 提示词按 user 发送
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_str = prompt
        else:
            messages = [
                {"role": "system", "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['user_prompt']},
            ]
            input_str = prompt['system_prompt'] + prompt['user_prompt']

        est_input_token = len(self.token_encoder.encode(input_str))  # API 未返回 usage 时的估算兜底

        start_time = time.time()
        usage = None
        if use_stream:
            for _ in range(max_query):
                response, usage = '', None
                try:
                    stream = openai.ChatCompletion.create(
                        model=self.llm_source,
                        messages=messages,
                        temperature=0.0,
                        stream=True,
                        stream_options={'include_usage': True})  # 让流式也返回真实 usage
                    for chunk in stream:
                        if 'usage' in chunk and chunk['usage']:
                            usage = chunk['usage']  # 末个 choices 为空的 chunk 携带 usage
                        if len(chunk['choices']) > 0:
                            if chunk['choices'][0]['delta'].get('content') is not None:
                                response += chunk['choices'][0]['delta']['content']
                    break
                except Exception as e:
                    handle_api_error(e)
                    continue
        else:
            response, usage = None, None
            for _ in range(max_query):
                try:
                    resp = openai.ChatCompletion.create(
                        model=self.llm_source,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    response = resp['choices'][0]['message']["content"]
                    usage = resp['usage'] if 'usage' in resp else None
                    break
                except Exception as e:
                    handle_api_error(e)
                continue

        # token 统计：优先 API 真实 usage，缺失则回退 tiktoken 估算
        if usage:
            input_token = usage.get('prompt_tokens', est_input_token)
            output_token = usage.get('completion_tokens', len(self.token_encoder.encode(response)))
        else:
            input_token = est_input_token
            output_token = len(self.token_encoder.encode(response))

        end_time = time.time()
        print("Time elapsed: {:.2f}".format(end_time - start_time))

        if info is not None:
            assert 'role' in info.keys(), "info must have key 'role'"
            assert 'timestep' in info.keys(), "info must have key 'timestep'"
            assert 'id' in info.keys(), "info must have key 'id'"
            assert 'property' in info.keys(), "info must have key 'property'"
            save_dir = path.join(self.round_save_dir, f"timestep_{info['timestep']}")
            save_dir = path.join(save_dir, f"{info['role']}_query")
            save_dir = path.join(save_dir, f"trial_{info['id']}")
            if not path.exists(save_dir):
                os.makedirs(save_dir)

            with open(path.join(save_dir, "{}_prompt.txt".format(info['property'])), "w") as f:
                if type(prompt) == str:
                    f.write(prompt)
                else:
                    f.write(prompt['system_prompt'] + prompt['user_prompt'])

            with open(path.join(save_dir, "{}_response.txt".format(info['property'])), "w") as f:
                f.write(response)

        if info['property'] == 'main':
            self.temp_consumed_token['input_token'] += input_token
            self.temp_consumed_token['output_token'] += output_token

        return response

    def _get_dialog_agents(self, active_agents=None, fixed_actions=None):
        """校验参与者与固定动作覆盖全部 agent，并按原有发言顺序排列。"""
        if active_agents is None and fixed_actions is None:
            return list(self.robot_agent_names), {}
        if self.algo != 'vdn-mlaq':
            raise ValueError('Fixed-agent dialogue requires algo="vdn-mlaq"')
        if active_agents is None:
            raise ValueError('active_agents is required when fixed_actions is provided')
        active_agents = list(active_agents)
        fixed_actions = dict(fixed_actions or {})
        active_set = set(active_agents)
        if not active_agents or len(active_set) != len(active_agents):
            raise ValueError('active_agents must be nonempty and contain no duplicates')
        if active_set & set(fixed_actions) or active_set | set(fixed_actions) != set(self.robot_agent_names):
            raise ValueError('Participating and fixed agents must partition all agents')
        if any(not isinstance(action, str) or not action.strip() or '\n' in action.strip()
                or action.strip() == 'action_virtual' for action in fixed_actions.values()):
            raise ValueError('Fixed actions must be concrete individual actions on one line')
        if self.max_calls_per_round < len(active_agents):
            raise ValueError('Dialogue budget must allow every participating agent to speak')
        return ([agent for agent in self.robot_agent_names if agent in active_set],
                {agent: fixed_actions[agent].strip() for agent in self.robot_agent_names if agent in fixed_actions})

    def _check_dialog_action(self, action, fixed_actions):
        """在环境自动补全动作前检查完整性与固定动作；返回错误说明或 None。"""
        if self.algo != 'vdn-mlaq':
            raise ValueError('Fixed-agent dialogue requires algo="vdn-mlaq"')
        parts = parse_joint_action(action, self.robot_agent_names)
        action_lines = [line for line in action.splitlines() if line.startswith('NAME ')]
        if (parts is None or len(action_lines) != len(self.robot_agent_names)
                or any(not value or value == 'action_virtual' for value in parts.values())):
            return 'Output exactly one NAME ... ACTION ... line for every agent in the final EXECUTE plan.'
        changed = [agent for agent, fixed in fixed_actions.items() if parts[agent] != fixed]
        if changed:
            return 'Keep the fixed actions unchanged for: ' + ', '.join(changed) + '.'
        return None

    def prompt_one_dialog_round(
            self,
            obs,
            feedback_history,
            action_history=[],
            forbidden_actions=[],
            replan_idx=0,
            timestep=0,
            active_agents=None,
            fixed_actions=None
    ):
        """
        多智能体协商得到联合动作
        持续提问直至输出 EXECUTE 或达到单轮最大调用次数上限
        """
        selected_dialog = active_agents is not None or fixed_actions is not None
        dialog_agents, fixed_actions = self._get_dialog_agents(active_agents, fixed_actions)
        chat_history = []
        agent_responses = []
        dialog_done = False
        num_responses = {agent_name: 0 for agent_name in dialog_agents}
        if self.algo == 'vdn-mlaq':
            agent_name, response = None, 'ERROR'

        # 循环调用多智能体
        n_calls = 0  # 调用 LLM 的次数
        while n_calls < self.max_calls_per_round:
            # 依次提问每个智能体
            for next_agent in dialog_agents:
                if self.algo == 'vdn-mlaq' and n_calls >= self.max_calls_per_round:
                    break
                agent_name = next_agent
                # 设置系统提示词
                system_prompt = self.compose_system_prompt(
                    obs,
                    agent_name,
                    action_history=action_history,
                    forbidden_actions=forbidden_actions,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,
                    active_agents=dialog_agents if selected_dialog else None,
                    fixed_actions=fixed_actions if selected_dialog else None,
                )

                # 设置用户提示词
                if n_calls < self.max_calls_per_round - 1:
                    # 非最后一轮
                    agent_prompt = (
                        f"You are {agent_name}, before you make decisions, you should check "
                        f"if the [Action] is violating the [Forbidden Actions].\nYour response is:\n"
                    )
                else:
                    # 最后一轮
                    agent_prompt = (
                        f"You are {agent_name}, before you make decisions, you should check "
                        f"if the [Action] is violating the [Forbidden Actions].\n"
                        f"This is the last call, you must end your response by incorporating "
                        f"all previous discussions and output the best plan via EXECUTE.\n"
                        f"Your response is:\n"
                    )

                # 调用 LLM
                response = self.new_query_once(
                    prompt={
                        "system_prompt": system_prompt,
                        "user_prompt": agent_prompt + self.env.get_policy_tail(),
                    },
                    info={'role': agent_name + '{}'.format(replan_idx), 'timestep': timestep, 'id': n_calls,
                        'property': 'main'},
                    use_stream=True
                )

                num_responses[agent_name] += 1  # 调用当前智能体的次数加一
                pruned_response = response.strip()  # 去掉 response 中多余的换行和空格
                agent_responses.append(f"[{agent_name}]:\n{pruned_response}")  # 将回复添加到 agent_responses 中
                n_calls += 1  # 调用 LLM 的次数加一

                # 判断是否结束
                if 'EXECUTE' in response:
                    # 要求第一轮对话 replan_idx = 0 时每个智能体都必须发言一次
                    # 重试 replan_idx > 0 时有智能体输出 EXECUTE 即可结束
                    # 指定参与者时，每次尝试（包括重试）都要求所有参与者发言。
                    if (not selected_dialog and replan_idx > 0) or all(v > 0 for v in num_responses.values()):
                        dialog_done = True
                        break

            if dialog_done:
                break

        if selected_dialog and not dialog_done:
            response = 'ERROR'
        return agent_name, response, agent_responses

    """
    --------------------------------------- memory 更新相关 ---------------------------------------
    """

    def update_memory_policy(self, state: str, action: str, feedback: str, temporary: bool = False):
        """
        更新 memory 中的动作 a 预测错误记录
        注：动作 a 由 basic policy 预测
        """
        if not temporary:  # 永久记住
            if preprocess_state(state) not in self.memory['policy'].keys():
                self.memory['policy'][preprocess_state(state)] = [[action, feedback]]
                self.temporary_memory['policy'][preprocess_state(state)] = [[action, feedback]]
            else:
                self.memory['policy'][preprocess_state(state)].append([action, feedback])
                self.temporary_memory['policy'][preprocess_state(state)].append([action, feedback])

            torch.save(self.memory, self.memory_path)
        else:  # 临时记住
            if preprocess_state(state) not in self.temporary_memory['policy'].keys():
                self.temporary_memory['policy'][preprocess_state(state)] = [[action, feedback]]
            else:
                self.temporary_memory['policy'][preprocess_state(state)].append([action, feedback])

    def update_memory_prediction(self, transition, true_transition, temporary: bool = False):
        """
        更新 memory 中的下一状态 s' 预测错误记录
        注：下一状态 s' 由 world model 预测
        """
        if not temporary:  # 永久记住
            state, action, next_state = transition
            true_next_state = true_transition[-1]
            search_feature = preprocess_state(state) + preprocess_action(action)
            if search_feature not in self.memory['prediction'].keys():
                self.memory['prediction'][search_feature] = [[next_state, true_next_state]]
                self.temporary_memory['prediction'][search_feature] = [[next_state, true_next_state]]
            else:  # 临时记住
                self.memory['prediction'][search_feature].append([next_state, true_next_state])
                self.temporary_memory['prediction'][search_feature].append([next_state, true_next_state])

            torch.save(self.memory, self.memory_path)
        else:
            state, action, next_state = transition
            true_next_state = true_transition[-1]
            search_feature = preprocess_state(state) + preprocess_action(action)
            if search_feature not in self.temporary_memory['prediction'].keys():
                self.temporary_memory['prediction'][search_feature] = [[next_state, true_next_state]]
            else:
                self.temporary_memory['prediction'][search_feature].append([next_state, true_next_state])

    """
    --------------------------------------- 辅助一次动作选择 ---------------------------------------
    """
    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def get_imagine_flag(self, state_desp):
        """
        判断是否需要通过 LLM 想象来探索新轨迹
        """
        if self.algo == 'vdn-mlaq':
            node = self.temp_lang_tree.get_node(state_desp)
            return node is None or not any(
                parse_joint_action(child['action'], self.robot_agent_names) is not None
                for child in node.children if child['action'] != 'action_virtual')
        result = False
        # if the current state is not in the temporary language tree
        if self.temp_lang_tree.get_node(state_desp) is None:  # 临时树中没有该状态节点
            result = True
        else:
            # if the current state has no children (only the virtual action)
            if self.temp_lang_tree.get_node(state_desp).num_children == 1:  # 临时树中该状态节点只有虚拟子节点
                result = True

        return result

    """
    --------------------------------------- 一次动作选择 ---------------------------------------
    """
    def decision_with_model(self, obs: EnvState, path_exist: bool = False):
        """
        根据 value 值为当前状态选择一个动作
        并通过环境检测该动作是否合理
        """
        # 根据当前观测 obs 获取当前状态的描述 state_desp
        state_desp = self.env.get_state_prompt(obs)

        # 若当前状态不是初始状态，则会检查上次预测的 s' 是否预测正确
        if self.state is not None:
            # 检查 LLM 预测的 (s, a, s') 是否正确
            # 若错误则会修正永久树中的 transition
            check_pass, transitions = self.lang_tree.check([self.state, self.action, state_desp])
            if not check_pass:
                # 同时添加到 memory 的 prediction 错误记录中
                self.update_memory_prediction(transitions['predicted_transition'], transitions['true_transition'])

        # 循环尝试直至找到一个 ready_to_execute 的动作
        trial = 0
        ready_to_execute, failed = False, False
        while not ready_to_execute:
            if self.algo == 'vdn-mlaq' and trial >= self.num_replans:
                self.log('Reached the environment-action replanning limit.', if_test=True)
                return False, None, path_exist
            """
            --------------------------------------- 上一轮动作校验失败则尝试 LLM 想象探索 ---------------------------------------
            """
            if failed or self.get_imagine_flag(state_desp):  # 当前状态未知或之前动作失败 → 尝试 LLM 想象探索
                path_exist = self.check_path_exist(state_desp, target_state_desp=self.env.target_state_desp)

                if not path_exist:
                    self.interaction_failure += 1
                    if self.interaction_failure > 3 or len(self.temp_lang_tree.nodes) > 40:
                        self.log("-------------- The temporary imagination fails more than 3 times. The decision process is terminated. --------------", if_test=True)
                        if self.algo == 'vdn-mlaq':
                            return False, None, path_exist
                        break

                self.imagine_with_ucb(obs=obs, horizon=12, interrupt_flag=True)

            """
            --------------------------------------- 根据 value 值为当前状态选择一个动作 ---------------------------------------
            """
            # 重新检查从当前状态到目标状态的路径（无论是否触发了想象，都基于最新树状态）
            path_exist = self.check_path_exist(state_desp, target_state_desp=self.env.target_state_desp)

            if path_exist:
                # 若存在候选最优轨迹则直接选择最优动作
                if self.algo == 'vdn-mlaq':
                    self.lang_tree.vdn_compute_q_values(self.env.target_state_desp)
                    decision_tree = self.lang_tree
                    self._qcompare(decision_tree, loop=100)
                    action, _ = self.lang_tree.vdn_get_best(state_desp)
                else:
                    lang_tree = copy.deepcopy(self.lang_tree)
                    lang_tree.compute_values(target_state=self.env.target_state_desp, loop=100)
                    decision_tree = lang_tree
                    self._qcompare(decision_tree, owner=self.lang_tree, loop=100)
                    action, _ = lang_tree.get_node(state_desp).get_best()
                self._qrec(decision_tree, state_desp, 'interact', action, owner=self.lang_tree)
            else:
                # 若不存在候选最优轨迹
                if self.get_imagine_flag(state_desp):
                    # 若仍然值得 LLM 想象探索则继续
                    self.log("After the imagination, the current state still has no available children.", if_test=True)
                    continue
                else:
                    # 若不值得 LLM 想象探索则随便选一个动作
                    state_node = self.temp_lang_tree.get_node(state_desp)
                    if self.algo == 'vdn-mlaq':
                        action, _ = self.temp_lang_tree.vdn_get_best(state_desp, without_virtual=True)
                        if action is None:
                            self.log('No concrete joint action is available for environment execution.', if_test=True)
                            return None, None, None
                    else:
                        action = state_node.children[-1]['action']

            """
            --------------------------------------- 通过环境检测动作是否合理 ---------------------------------------
            """
            # 检查动作的格式是否满足要求
            if self.parser is not None:
                parse_succ, parsed_str, llm_plans = self.parser.parse(obs, action)  # llm_plans 指明每个机械臂在每一步需要执行什么动作
                assert type(llm_plans) == list, "llm_plans must be a list when the parser is not none"
            else:
                parse_succ, parsed_str, llm_plans = True, '', action
                assert type(llm_plans) == str, "llm_plans must be a string when the parser is none"

            error_info = {
                'task_error': False,        # 是否违反任务约束
                'dynamics_error': False,    # 是否违反动力学约束
                'IK_error': False           # 是否违反运动学约束
            }
            action_feedback = None
            if not parse_succ:
                # 若格式不满足要求
                self.log('Failed to parse action: {}'.format(action), if_test=True)
                error_info['dynamics_error'] = True
                action_feedback = (
                    f"This previous [Action] failed to parse!: '{action}'\n"
                    f"{parsed_str} Re-format to strictly follow [Action Output Instruction]!"
                )
                ready_to_execute = False
                failed = True  # 继续进行 LLM 想象探索
            else:
                # 若格式满足要求
                ready_to_execute = True
                if self.parser is not None:
                    for _, llm_plan in enumerate(llm_plans):
                        # 使用 MuJoCo 仿真器中的快速物理模拟来验证 LLM 想象的动作是否可执行
                        ready_to_execute, env_feedback, error_info = self.feedback_manager.give_feedback(obs, llm_plan)
                        if not ready_to_execute:
                            self.env_replans += 1
                            self.log('Failed to execute action: {}'.format(action), if_test=True)
                            action_feedback = env_feedback
                            if 'IK' in env_feedback:
                                error_info['IK_error'] = True
                            failed = True
                            break
                else:
                    # 使用 MuJoCo 仿真器中的快速物理模拟来验证 LLM 想象的动作是否可执行
                    ready_to_execute, env_feedback, error_info = self.feedback_manager.give_feedback(obs, llm_plans)
                    if not ready_to_execute:
                        self.env_replans += 1
                        self.log('Failed to execute action: {}'.format(action), if_test=True)
                        action_feedback = env_feedback
                        failed = True

            if not ready_to_execute:
                self.fill_temp_lang_tree(state_desp=state_desp, target_state_desp=self.env.target_state_desp)
                self.log(action_feedback, if_test=True)
                if error_info['task_error'] and error_info['dynamics_error']:
                    raise ValueError("Both task error and dynamics error occur.")

                if 'IK_error' in error_info.keys():
                    if error_info['IK_error']:
                        return None, None, None

                # 若动作 a 违反任务约束
                if error_info['task_error']:
                    self.update_memory_policy(state_desp, action, action_feedback, temporary=True)

                # 若动作 a 违反动力学约束
                if error_info['dynamics_error']:
                    self.log("Remove the current state and its children from the language tree.", if_test=True)
                    self.lang_tree.remove_node(state_desp, action)  # 从永久树中删除 (s, a)
                    self.temp_lang_tree.remove_node(state_desp, action)  # 从临时树中删除 (s, a)
                    self.update_memory_policy(state_desp, action, action_feedback, temporary=False)

            trial += 1

        self.state = self.env.get_state_prompt(obs)
        self.action = action

        return ready_to_execute, llm_plans, path_exist

    """
    --------------------------------------- LLM 想象 (始) ---------------------------------------
    """
    def imagine_with_ucb(self, obs, horizon: int = 10, interrupt_flag=False):
        """
        在临时树上做 UCB 树搜索
        遇到未知节点就调 LLM 扩展树
        多轮迭代直到找到最优轨迹或确认找不到
        """
        initial_state_desp = self.env.get_state_prompt(obs)
        target_state_desp = self.env.target_state_desp
        if self.algo == 'vdn-mlaq' and (
                preprocess_state(initial_state_desp) in (preprocess_state(target_state_desp),
                                                        preprocess_state('Success'), preprocess_state('Failure'))
                or self.env.if_fail(initial_state_desp)):
            return
        self.fill_temp_lang_tree(state_desp=initial_state_desp, target_state_desp=target_state_desp)

        stop_flag = False
        while not stop_flag:
            self.set_round_dir(path.join(self.simulate_save_dir, f"round_{self.round_index}"))

            # 设置循环条件
            final, failed, restart = False, False, False
            t, llm_count = 0, 0

            # 初始化统计量
            self.temp_consumed_token = {'input_token': 0, 'output_token': 0}
            self.temp_memory_ratio = {'llm_count': 0, 'memory_count': 0}
            action_history = []

            # 打印初始状态
            self.log(f"-------------- Start {self.round_index}th round imagination --------------")
            self.log('Original state:\n' + initial_state_desp)

            # 每次 LLM 想象都从初始状态开始
            state_desp = initial_state_desp
            self.temp_lang_tree.get_node(state_desp).visit_num += 1
            # 图可能有环；horizon 仅限制 LLM 请求，另限纯记忆遍历避免死循环。
            if self.algo == 'vdn-mlaq':
                search_steps = 0
                max_search_steps = max(1, horizon) * max(1, len(self.temp_lang_tree.nodes))
            # 从初始状态开始探索直至到达目标状态 或 LLM 调用达到最大次数上限
            while not final and llm_count < horizon:
                if self.algo == 'vdn-mlaq' and search_steps >= max_search_steps:
                    self.log('Reached the per-round graph traversal limit.')
                    break
                if self.algo == 'vdn-mlaq':
                    search_steps += 1
                self.log('---------- Time step {} ----------'.format(t))

                """
                --------------------------------------- 想象 transition (根据 s 预测 a 和 s') ---------------------------------------
                """
                state_node = self.temp_lang_tree.get_node(state_desp)  # 在临时树上找到当前状态节点
                assert state_node is not None, f"state_node is None, state_desp:{state_desp}"
                if self.algo == 'vdn-mlaq':
                    # with_ucb=True；不排除 virtual（想象需要可能选中 virtual 来调 LLM）
                    action, best_child = self.temp_lang_tree.vdn_get_best(state_desp, with_ucb=True)
                else:
                    action, best_child = state_node.get_best(with_ucb=True)  # 通过 UCB 选择最优动作和对应的子节点
                self._qrec(self.temp_lang_tree, state_desp, 'imagine', action, ucb=True)

                if self.algo == 'vdn-mlaq' and action is None:
                    self.log('No available search edge at the current state.')
                    self.imagination_failure += 1
                    break

                dialog_options = {}
                if action == 'action_virtual':
                    best_child.visit_num += 1  # 记录全局虚拟入口被选择，包括全员固定的情况。
                    if self.algo == 'vdn-mlaq':
                        anchor_action, active_agents = self.temp_lang_tree.select_imagination_agents(state_desp)
                        anchor_parts = (parse_joint_action(anchor_action, self.robot_agent_names)
                                        if anchor_action is not None else {})
                        fixed_actions = {agent: anchor_parts[agent] for agent in self.robot_agent_names
                                         if agent not in active_agents}
                        self.log(f'Local UCB participants: {active_agents}; fixed actions: {fixed_actions}')
                        if active_agents:
                            dialog_options = dict(active_agents=active_agents, fixed_actions=fixed_actions)
                        else:
                            # 所有人选已知动作时沿基准继续，不强制选人、不消耗 LLM 请求。
                            action = anchor_action
                            best_child = state_node.get_child(action)
                            self.log('All agents keep the anchor joint action; no LLM request.')

                if action == 'action_virtual':
                    # 若选到虚拟子节点
                    self.log('----- Use LLM -----')
                    llm_count += 1

                    # 禁用已知动作，即已探索过的动作
                    forbidden_actions = [
                        child['action'] for child in state_node.children
                        if child['action'] not in ('action_virtual', preprocess_action('ERROR'))
                    ]

                    # 调用 LLM 想象动作 a 和下一状态 s'
                    imagine_results = self.imagine_transition(
                        state_desp=state_desp, forbidden_actions=forbidden_actions, action_history=action_history, timestep=t,
                        **dialog_options
                    )

                    # imagine_results 包含 'transition' 和 'imagine_info'
                    imagine_info = imagine_results['imagine_info']
                    assert imagine_info in ('success', 'failure', 'exhausted', 'action_failed', 'prediction_failed'), \
                        f"Unexpected imagine_info: {imagine_info}"

                    if imagine_info == 'success':
                        action = imagine_results['transition'][1]  # 获取动作 a
                        next_state_desp  = imagine_results['transition'][2]  # 获取下一状态 s'
                        action_history.append(action)  # 把动作 a 加入到动作历史中
                        # 根据下一状态 s' 判断是否到达目标状态
                        final = (preprocess_state(next_state_desp) == preprocess_state(target_state_desp))
                    elif imagine_info == 'failure':
                        failed = True
                    elif imagine_info == 'exhausted':
                        # 若 LLM 想象力耗尽，则根据 UCB 值从已知动作中选择最优动作 (排除掉了虚拟子节点)
                        if self.algo == 'vdn-mlaq':
                            action, best_child = self.temp_lang_tree.vdn_get_best(state_desp, with_ucb=True, without_virtual=True)
                        else:
                            action, best_child = state_node.get_best(with_ucb=True, without_virtual=True)
                        self._qrec(self.temp_lang_tree, state_desp, 'imagine',
                                    action, ucb=True, without_virtual=True)
                        action_history.append(action)
                        assert best_child is not None, f"child is None, action: {action}"
                        next_state_desp = best_child.state
                        # 根据下一状态 s' 判断是否到达目标状态
                        final = (preprocess_state(next_state_desp) == preprocess_state(target_state_desp))
                    elif imagine_info in ('action_failed', 'prediction_failed'):
                        # 若 LLM 想象失败则重试
                        restart = True
                        self.imagination_failure += 1
                else:
                    # 若选到普通子节点
                    self.log('----- Use memory -----')
                    action_history.append(action)
                    child = state_node.get_child(action)
                    assert child is not None, f"child is None, action: {action}"
                    next_state_desp = child.state
                    if self.algo == 'vdn-mlaq' and parse_joint_action(action, self.robot_agent_names) is not None:
                        self.temp_lang_tree.add_joint_visit_num(state_desp, action)
                    # 根据下一状态 s' 判断是否到达目标状态
                    final = (preprocess_state(next_state_desp) == preprocess_state(target_state_desp))
                    failed = (next_state_desp == preprocess_state('Failure')
                              or self.algo == 'vdn-mlaq' and self.env.if_fail(next_state_desp))

                """
                --------------------------------------- 判断内循环是终止还是继续 ---------------------------------------
                """
                if final:
                    # (后续) 终止内循环
                    self.log(f"Imagination step {t}:\n{action}\n{next_state_desp}")
                    self.log(f"Imagination reaches the target state:\n{target_state_desp}")
                    self.temp_lang_tree.update(transition=[next_state_desp, 'nothing', 'Success'])
                    self.temp_lang_tree.get_node(next_state_desp).visit_num += 1
                    self.temp_lang_tree.remove_node(next_state_desp, 'action_virtual', virtual=True)
                    success_node = self.temp_lang_tree.get_node('Success')
                    success_node.visit_num += 1
                    success_node.value = 1
                elif failed:
                    # (后续) 留在当前状态重试
                    self.log(f"Imagination fails at state:\n{state_desp}")
                    failure_node = self.temp_lang_tree.get_node('Failure')
                    if failure_node is not None:
                        failure_node.visit_num += 1
                        failure_node.value = -1
                elif restart:
                    # (后续) 留在当前状态重试
                    self.log('------- Reach the action failed or prediction failed terminal state -------')
                    self.log('------- Restart the imagination -------')
                else:
                    # 正常完成 LLM 想象 transition
                    # (后续) 指针移动到下一状态
                    self.log(f"Imagination step {t}:\n{action}\n{next_state_desp}")  # 打印本轮想象的动作 a 和下一状态 s'
                    state_desp = next_state_desp  # 指针移动到下一状态 s'
                    self.temp_lang_tree.get_node(next_state_desp).visit_num += 1  # 下一状态 s' 的访问次数 + 1
                    t += 1
                    self.fill_temp_lang_tree(state_desp=initial_state_desp, target_state_desp=target_state_desp)
                    # 若临时树的节点个数 > 40 则停止想象
                    if len(self.temp_lang_tree.nodes) > 40:
                        self.log('The temporary language tree is too large. The imagination process is terminated.')
                        break
                    continue

                # final / failed / restart 情况的共享收尾
                self.fill_temp_lang_tree(state_desp=initial_state_desp, target_state_desp=target_state_desp)
                self.log('-------------- End {}th round imagination --------------\n'.format(self.round_index))
                break

            # summarize the data for this round
            stop_flag = self._summarize_this_round(
                initial_state_desp=initial_state_desp, target_state_desp=target_state_desp, interrupt_flag=interrupt_flag
            )

    def _summarize_this_round(self, initial_state_desp, target_state_desp, interrupt_flag=False):
        # summarize the token and memory ratio of this round
        # return if the test should be conducted after this round

        if self.algo == 'vdn-mlaq':
            # 无论本轮成功、失败或纯记忆遍历，都先记成本，避免提前停止漏记。
            self.consumed_token['input_token'].append(self.temp_consumed_token['input_token'])
            self.consumed_token['output_token'].append(self.temp_consumed_token['output_token'])
            self.memory_ratio['llm_count'].append(self.temp_memory_ratio['llm_count'])
            self.memory_ratio['memory_count'].append(self.temp_memory_ratio['memory_count'])
            self.round_index += 1

        if len(self.temp_lang_tree.nodes) > 40:
            return True

        if self.imagination_failure >= 3:
            return True

        path_exist = self.check_path_exist(initial_state_desp, target_state_desp)
        current_end_step = self.get_end_step(target_state_desp)
        if path_exist:
            optimized_flag = False
            if current_end_step < self.last_end_step:
                optimized_flag = True

            self.last_end_step = current_end_step

        if interrupt_flag and path_exist:
            return True

        if self.algo == 'vdn-mlaq':
            # 全员固定仍改变次数，纯记忆轮可以继续；但同样服从轮数上限。
            return (self.round_index >= 8 or path_exist and (
                current_end_step <= self.env.optimal_steps
                or optimized_flag and self.temp_consumed_token['input_token'] > 0))

        if self.temp_consumed_token['input_token'] == 0:
            # if the assumed token is 0, it means that the LLM has not been called in this round
            # therefore, we should not add the assumed token of this round into the list
            if self.round_index == 1:
                if len(self.consumed_token['input_token']) == 0:
                    self.consumed_token['input_token'].append(0)
                    self.consumed_token['output_token'].append(0)
                    self.memory_ratio['llm_count'].append(0)
                    self.memory_ratio['memory_count'].append(0)
                    return True
                else:
                    self.round_index += 1
                    return False
            else:
                shutil.rmtree(self.round_save_dir)
                return False
        else:
            # if the assumed token is not 0, it means that the LLM has been called in this round
            self.consumed_token['input_token'].append(self.temp_consumed_token['input_token'])
            self.consumed_token['output_token'].append(self.temp_consumed_token['output_token'])
            self.memory_ratio['llm_count'].append(self.temp_memory_ratio['llm_count'])
            self.memory_ratio['memory_count'].append(self.temp_memory_ratio['memory_count'])
            self.round_index += 1
            if path_exist:
                # if the current end step is the optimal steps, we should conduct the test
                if current_end_step <= self.env.optimal_steps:
                    return True

                if optimized_flag:
                    return True
                else:
                    if self.round_index >= 8:
                        return True
                    else:
                        return False
            else:
                return False

    def imagine_transition(self, state_desp: str, forbidden_actions=None,
                            action_history=None, timestep=0, active_agents=None, fixed_actions=None):
        """
        想象一条转移

        1. predict_action：预测动作、检查动作及重试
        2. predict_next_state：复用记忆，或预测下一状态、检查下一状态及重试
        3. 本函数统一更新树；预测函数不写树
        """
        selected_dialog = active_agents is not None or fixed_actions is not None
        dialog_options = {}
        if selected_dialog:
            active_agents, fixed_actions = self._get_dialog_agents(active_agents, fixed_actions)
            self.temp_lang_tree.init_individual_actions(state_desp, with_virtual=True)
            for agent in active_agents:
                # 一次扩展请求记一次；内部最多三次提案重试，不按发言条数重复计数。
                self.temp_lang_tree.add_individual_visit_num(state_desp, agent, 'action_virtual')
            dialog_options = dict(active_agents=active_agents, fixed_actions=fixed_actions)
        # 预测动作 a
        action, status = self.predict_action(
            state_desp, forbidden_actions or [], action_history or [], timestep, **dialog_options)
        if status != 'success':
            if status == 'failure':
                # 保留原完整联合提案的 ERROR 行为。
                self.temp_lang_tree.update(transition=[state_desp, 'ERROR', 'Failure'])
            return {'transition': None, 'imagine_info': status}
        action = preprocess_action(action)

        # 预测下一状态 s'，并标明是复用记忆还是新预测的
        next_state, source = self.predict_next_state(state_desp, action, timestep)
        if next_state is None:
            return {'transition': None, 'imagine_info': 'prediction_failed'}

        # 拼接得到 transition
        transition = [state_desp, action, next_state]

        # 更新统计信息并写入永久树
        if source == 'memory':
            self.temp_memory_ratio['memory_count'] += 1
        else:
            self.lang_tree.update(transition)
            self.temp_memory_ratio['llm_count'] += 1

        if selected_dialog:
            # 只有得到转移才计具体动作一次；失败状态的已评价转移也保留供 TD 使用。
            self.temp_lang_tree.update(transition)
            self.temp_lang_tree.add_joint_visit_num(state_desp, action)
        if self.env.if_fail(next_state) or (selected_dialog and preprocess_state(next_state) == preprocess_state('Failure')):
            if selected_dialog:
                self._build_temp_virtual_node(next_state)
            return {'transition': transition, 'imagine_info': 'failure'}

        # 写入临时树并挂上虚拟子节点
        if not selected_dialog:
            self.temp_lang_tree.update(transition)
        self._build_temp_virtual_node(next_state)
        return {'transition': transition, 'imagine_info': 'success'}

    def predict_action(self, state_desp, forbidden_actions=None,
                        action_history=None, timestep=0, active_agents=None, fixed_actions=None):
        """
        预测并检查完整联合动作；可指定参与讨论的 agent 和其余固定个体动作。
        指定参与者时最多尝试三次，固定动作约束在所有尝试中保持不变。
        """
        forbidden_actions = forbidden_actions or []
        action_history = action_history or []
        selected_dialog = active_agents is not None or fixed_actions is not None
        if selected_dialog:
            active_agents, fixed_actions = self._get_dialog_agents(active_agents, fixed_actions)

        action, feedbacks = None, []
        trial, action_success = 0, False

        # 获取动作 a
        while trial < 3:  # 最多尝试 3 次
            # 调用 LLM 预测动作 a
            dialog_options = dict(active_agents=active_agents, fixed_actions=fixed_actions) if selected_dialog else {}
            action, _ = self.get_llm_action(state_desp, feedbacks, forbidden_actions, action_history,
                                            timestep+1, trial+1, **dialog_options)
            action = action.replace('Error', 'ERROR')

            if selected_dialog:
                # 先校验原始完整提案，避免环境补 WAIT 掩盖缺失动作。
                error = self._check_dialog_action(action, fixed_actions)
                if error is None:
                    action = self.env.process_action(action)
                    error = self._check_dialog_action(action, fixed_actions)
                if error is None and preprocess_action(action) in {
                    preprocess_action(forbidden) for forbidden in forbidden_actions
                }:
                    error = 'This joint action has already been explored. Choose a different joint action while preserving fixed actions.'
                if error is not None:
                    feedbacks.append({'action': action, 'feedback': error})
                    trial += 1
                    continue
                action_success, feedback = self.check_action(state_desp, action, timestep+1, trial+1)
                if action_success:
                    break
                feedbacks.append(feedback)
                trial += 1
                self.log('----- Re-plan the action -----')
                continue

            if 'ERROR' in action:  # 若直接预测出错
                trial += 2 if len(forbidden_actions) == 0 else 3  # 若有禁用动作则快速耗尽重试次数
            else:  # 若预测正常则调用 LLM 检查预测结果是否合理
                action_success, feedback = self.check_action(state_desp, action, timestep+1, trial+1)
                if action_success:  # 若动作合理则跳出循环
                    break
                else:  # 若动作不合理则记录错误反馈并重新尝试预测动作
                    feedbacks.append(feedback)
                    trial += 1 if len(forbidden_actions) == 0 else 2
                    self.log('----- Re-plan the action -----')

        # (兜底) 若尝试次数达到上限但动作 a 仍不合理，则直接返回
        if not action_success:
            if selected_dialog:
                # 固定部分队友的提案失败，不代表整个状态的动作已穷尽。
                return None, 'action_failed'
            if len(forbidden_actions) == 0:
                if 'ERROR' in action:
                    return None, 'failure'
                else:
                    return None, 'action_failed'
            else:  # 除禁用动作外 LLM 想不出任何可用动作
                return None, 'exhausted'
        return action, 'success'

    def get_llm_action(self, obs, feedbacks: list = [], forbidden_actions: list = [], action_history: list = [],
                    timestep=None, trial_id=None, active_agents=None, fixed_actions=None):
        """
        让 agent 轮流讨论。指定参与者时返回待校验的完整提案，
        由 predict_action 检查固定动作后再进行环境格式处理。
        """
        _, final_response, agent_responses = self.prompt_one_dialog_round(
            obs,
            feedbacks,
            action_history=action_history,
            forbidden_actions=forbidden_actions,
            replan_idx=trial_id,
            timestep=timestep,
            active_agents=active_agents,
            fixed_actions=fixed_actions
        )

        # action = self._extract_action(final_response, info={
        #     'role': 'policy', 'timestep': timestep, 'id': trial_id, 'property': 'extract_action'
        # })

        if active_agents is not None or fixed_actions is not None:
            # 仅提取最终方案，交给 predict_action 校验后再进行环境格式处理。
            action = ('EXECUTE\n' + final_response.rsplit('EXECUTE', 1)[1].strip()
                        if 'EXECUTE' in final_response else 'ERROR')
        else:
            action = self.env.process_action(final_response)

        return action, agent_responses

    def predict_next_state(self, state: str, action: str, timestep=0):
        """
        复用已有下一状态或预测新的下一状态
        """
        # 若永久树中存在 (s, a) 则直接复用 s'
        action = preprocess_action(action)
        state_node = self.lang_tree.get_node(state)
        if state_node:
            for child in state_node.children:
                if action == child['action']:
                    self.log('Transition exists. Use the existing transition.\n')
                    return child['child'].state, 'memory'

        # 若永久树中不存在 (s, a) 则预测 s'
        self.log('----- Imagine the transition -----')

        prediction_passed = False
        trial = 0
        predictor_feedback = ''

        while trial < 3:
            predictor_prompt = self.get_role_prompt(
                'predictor', state, action, next_state=None,
                info={'feedbacks': predictor_feedback})
            predictor_response = self.new_query_once(predictor_prompt, use_stream=True, info={
                'role': 'world_model', 'timestep': timestep + 1,
                'id': trial + 1, 'property': 'main'})
            predicted_next_state = preprocess_state(self.env.process_state(predictor_response))

            if 'ERROR' in predicted_next_state:  # 若直接预测出错
                prediction_passed = False
                feedback = 'The prediction is invalid.'
            else:  # 若预测正常则调用 LLM 检查预测结果是否合理
                prediction_passed, feedback = self.check_next_state(
                    state, action, predicted_next_state, timestep + 1, trial + 1
                )

            if prediction_passed:  # 若预测合理则跳出循环
                break

            self.log('----- Re-predict the next state -----')
            trial += 1
            predictor_feedback = self._process_prediction_feedback(feedback)  # 把 LLM (检测) 的反馈包装成 prompt 片段以注入到对 LLM (预测) 的下一次请求中

        if not prediction_passed:
            return None, None

        return predicted_next_state, 'llm'

    def check_action(self, state: str, action: str, timestep=None, trial_id=None):
        """
        调用 LLM 判断 LLM 预测的动作是否合理
        """
        self.log('----- Check the action -----')

        # 检查动作是否在永久树中存在
        state_node = self.lang_tree.get_node(state)  # 在永久树上寻找当前状态 s
        if state_node and not self.skip_check_memory:  # 若存在且需要复用永久树中的 transition
            children_actions = state_node.get_children_actions(without_virtual=True)  # 获取当前状态 s 的所有子节点动作
            if preprocess_action(action) in children_actions:  # 若动作 a 存在
                self.log('The action is valid because it is in the full memory.')
                return True, None  # 直接认为动作 a 合理并返回 True

        # 若动作 a 不在永久树中则调用 LLM 检查动作 a 是否合理
        # 第一次调用 LLM 检查动作 a 是否合理
        action_checker_prompt = self.get_role_prompt('action_checker', state, action)
        action_checker_response = self.new_query_once(action_checker_prompt, use_stream=True, info={
            'role': 'action_checker', 'timestep': timestep, 'id': trial_id, 'property': 'main'
        })

        # 第二次调用 LLM 对第一次的检查结果进行总结，输出 [VALID] 或 [ERROR]
        feedback_prompt = (
            f"Here is the feedback from an action checker. \n"
            f"[--- Begin ---]\n"
            f"{action_checker_response}\n"
            f"[--- End ---]\n"
            f"Please read the feedback carefully and help to summarize it. Please provide the feedback in a more concise manner. \n"
            f"If the feedback says that the action is valid, please output [VALID]; otherwise, please output [ERROR] and provide the reason. "
        )
        feedback = self.new_query_once(feedback_prompt, use_stream=True, info={
            'role': 'policy', 'timestep': timestep, 'id': trial_id, 'property': 'feedback'
        })

        # 处理第二次 LLM 的输出，统一为 [VALID] 或 [ERROR]
        feedback = feedback.replace('Invalid', 'ERROR')
        feedback = feedback.replace('invalid', 'ERROR')
        feedback = feedback.replace('Valid', 'VALID')
        feedback = feedback.replace('valid', 'VALID')

        # 根据第二次 LLM 的输出判断动作 a 是否合理
        if 'VALID' not in feedback or 'ERROR' in feedback:  # 若不合理则返回 False 和反馈信息
            return False, {'feedback': feedback, 'action': action}
        else:  # 若合理则返回 True 和 None
            return True, None

    def check_next_state(self, state: str, action: str, prediction: str, timestep=None, trial_id=None):
        """
        调用 LLM 判断 LLM 预测的下一状态是否合理
        """
        self.log('----- Check the prediction -----')

        # 第一次调用 LLM 检查下一状态 s' 是否合理
        checker_prompt = self.get_role_prompt('prediction_checker', state, action, next_state=prediction)
        checker_response = self.new_query_once(checker_prompt, use_stream=True, info={
            'role': 'prediction_checker', 'timestep': timestep, 'id': trial_id, 'property': 'main'
        })

        # 第二次调用 LLM 对第一次的检查结果进行总结，输出 [VALID] 或 [ERROR]
        feedback_prompt = (
            f"Here is the feedback from a prediction checker. \n"
            f"[--- Begin ---]\n"
            f"{checker_response}\n"
            f"[--- End ---]\n"
            f"Please read the feedback carefully and help to summarize it. Please provide the feedback in a more concise manner. \n"
            f"If the feedback says that the prediction is correct, please output [VALID]; otherwise, please output [ERROR] and provide the reason. "
        )
        feedback = self.new_query_once(feedback_prompt, use_stream=True, info={
            'role': 'prediction_checker', 'timestep': timestep, 'id': trial_id, 'property': 'feedback'
        })

        # 处理第二次 LLM 的输出，统一为 [VALID] 或 [ERROR]
        feedback = feedback.replace('Invalid', 'ERROR')
        feedback = feedback.replace('invalid', 'ERROR')
        feedback = feedback.replace('INVALID', 'ERROR')
        feedback = feedback.replace('Valid', 'VALID')
        feedback = feedback.replace('valid', 'VALID')

        # 根据第二次 LLM 的输出判断动作 a 是否合理
        if 'VALID' not in feedback:
            return False, feedback  # 若不合理则返回 False 和反馈信息
        else:
            return True, None  # 若合理则返回 True 和 None

    def _process_prediction_feedback(self, feedback: str):
        """
        当前 s' 未通过检测时供 predict_next_state 使用
        把 LLM (检测) 的反馈包装成 prompt 片段以注入到对 LLM (预测) 的下一次请求中
        """
        if feedback is not None:
            prediction_feedback = '[Prediction Feedback]\nI have send your prediction to a GPT-based checker to check whether your prediction is correct. The checker said that your prediction is incorrect: ' + feedback
            prediction_feedback += 'The action has been checked before so that you do not need to check it again. '
        else:
            prediction_feedback = ''

        return prediction_feedback
    """
    --------------------------------------- LLM 想象 (终) ---------------------------------------
    """
