import copy
import os
import sys
import json
import random
import pickle
import shutil
import logging
from glob import glob
import os.path as path
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional, Any
from rocobench import PlannedPathPolicy, LLMPathPlan, MultiArmRRT
from prompting import LLMResponseParser, FeedbackManager, ModelPrompter
from prompting.paths import exp_output_dir
from tools.minimax_quota import LLMQuotaExceeded
from rocobench.envs import SortOneBlockTask, MakeSandwichTask, MujocoSimEnv, SimRobot, visualize_voxel_scene
from rocobench.dialog_envs import SortOneBlockTask as SortOneBlockDialogTask
from rocobench.dialog_envs import MakeSandwichTask as MakeSandwichDialogTask
from prompting.dialog_model_prompter import DialogModelPrompter


TASK_NAME_MAP = {
    "sort_central": SortOneBlockTask,
    'sort_dialog': SortOneBlockDialogTask,
    "sandwich_central": MakeSandwichTask,
    'sandwich_dialog': MakeSandwichDialogTask,
}


class LLMRunner:
    def __init__(
            self,
            env: MujocoSimEnv,
            robots: Dict[str, SimRobot],
            max_runner_steps: int = 50,
            video_format: str = "mp4",
            epoch_id: int = 0,
            num_runs: int = 1,
            verbose: bool = False,
            np_seed: int = 0,
            start_seed: int = 0,
            run_name: str = "run",
            data_dir: str = "data",
            overwrite: bool = False,
            llm_output_mode="action_only",  # "action_only" or "action_and_path"
            llm_comm_mode="chat",
            llm_num_replans=1,
            give_env_feedback=True,
            skip_display=False,
            policy_kwargs: Dict[str, Any] = dict(control_freq=50),
            direct_waypoints: int = 0,
            max_failed_waypoints: int = 0,
            debug_mode: bool = False,
            split_parsed_plans: bool = False,
            use_history: bool = False,
            use_feedback: bool = False,
            temperature: float = 0.0,
            llm_source: str = "gpt4",
            tree_load: bool = False,
            skip_check_memory: bool = False,
            algo: str = 'vdn-mlaq',     # "mlaq"=值迭代 / "vdn-mlaq"=遍历式 VDN
            compare_q: bool = False,  # 旁路记录两套 Q 与动作选择的一致性
    ):
        self.env = env
        self.env.reset()
        self.robots = robots
        self.robot_agent_names = list(robots.keys())  # ['Alice', etc.]
        self.data_dir = data_dir
        self.run_name = run_name
        self.algo = 'vdn-mlaq'   # 透传给 prompter：mlaq / vdn
        self.compare_q = compare_q

        # NOTE: the run directory is data/env_name, or data/env_name/<epoch_id> (epoch_id = "<算法>/<YYYYMMDD_HHMM>")
        run_dir = exp_output_dir(self.env.name, epoch_id, self.data_dir)
        os.makedirs(run_dir, exist_ok=overwrite)
        self.run_dir = run_dir
        self.verbose = verbose
        self.np_seed = np_seed
        self.start_seed = start_seed
        self.num_runs = num_runs
        self.overwrite = overwrite
        self.direct_waypoints = direct_waypoints
        self.max_failed_waypoints = max_failed_waypoints
        self.max_runner_steps = max_runner_steps
        self.give_env_feedback = give_env_feedback
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.tree_load = tree_load
        self.skip_check_memory = skip_check_memory

        self.llm_output_mode = llm_output_mode
        self.debug_mode = debug_mode  # useful for debug

        self.llm_num_replans = llm_num_replans
        self.llm_comm_mode = llm_comm_mode
        self.response_keywords = ['NAME', 'ACTION']
        if llm_output_mode == "action_and_path":
            self.response_keywords.append('PATH')
        self.planner = MultiArmRRT(
            self.env.physics,
            robots=robots,
            graspable_object_names=self.env.get_graspable_objects(),
            allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )
        self.episode_id = 0
        self.policy_kwargs = policy_kwargs
        self.video_format = video_format
        self.skip_display = skip_display
        self.split_parsed_plans = split_parsed_plans
        self.temperature = temperature
        self.parser = LLMResponseParser(
            self.env,
            llm_output_mode,
            self.env.robot_name_map,
            self.response_keywords,
            self.direct_waypoints,
            use_prepick=self.env.use_prepick,
            use_preplace=self.env.use_preplace,  # NOTE: should be custom defined in each task env
            split_parsed_plans=False,  # self.split_parsed_plans,
        )
        self.feedback_manager = FeedbackManager(
            env=self.env,
            planner=self.planner,
            llm_output_mode=self.llm_output_mode,
            robot_name_map=self.env.robot_name_map,
            step_std_threshold=self.env.waypoint_std_threshold,
            max_failed_waypoints=self.max_failed_waypoints,
        )
        if 'dialog' in self.env.name:
            self.prompter = DialogModelPrompter(
                env=self.env,
                epoch_id=epoch_id,
                data_dir=self.data_dir,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=3000,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
                load=self.tree_load,
                skip_check_memory=skip_check_memory,
                algo=self.algo,
                compare_q=self.compare_q,
            )
        else:
            self.prompter = ModelPrompter(
                env=self.env,
                epoch_id=epoch_id,
                data_dir=self.data_dir,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=3000,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
                load=self.tree_load
            )

    def get_case_end(self, end_step: int):
        # if the optimal steps are reached, we consider the case ends
        if end_step <= self.env.optimal_steps:
            return True, 'success'

        # if the interaction fails for more than 3 times (end_step = -1), we consider the case ends
        if end_step == -1:
            return True, 'failure'

        # if the language tree is too large, we consider the case ends
        if len(self.prompter.temp_lang_tree.nodes) > 40:
            return True, 'failure'

        # if the episode id is larger than 3, we consider the case ends
        if self.episode_id >= 3:
            return True, 'failure'

        return False, None

    def reset_env(self, case_id):
        # seed = case_id + start_id
        seed = 0
        self.env.seed(np_seed=seed)
        random.seed(seed)

        obs = self.env.reset(case_id=case_id, reload=True)
        self.env.get_target_prompt(obs=obs)

        return obs

    def display_plan(self, plan: LLMPathPlan, save_name="vis_plan", save_dir=None):
        """ Display the plan in the open3d viewer """
        env = deepcopy(self.env)
        env.physics.data.qpos[:] = self.env.physics.data.qpos[:].copy()
        env.physics.forward()
        env.render_point_cloud = True
        obs = env.get_obs()
        path_ls = plan.path_3d_list
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_name}.jpg")
        visualize_voxel_scene(
            obs.scene,
            path_pts=path_ls,
            save_img=(save_dir is not None),
            img_path=save_path
        )

    def roco_env_step(self, llm_plan):
        obs = self.env.get_obs()
        reward, done, info = 0, False, {}
        for i, plan in enumerate(llm_plan):
            policy = PlannedPathPolicy(
                physics=self.env.physics,
                robots=self.robots,
                path_plan=plan,
                graspable_object_names=self.env.get_graspable_objects(),
                allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                plan_splitted=self.split_parsed_plans,
                **self.policy_kwargs,
            )

            num_sim_steps = 0
            plan_success, reason = policy.plan(self.env)
            if plan_success:
                while not policy.plan_exhausted:
                    sim_action = policy.act(obs, self.env.physics)
                    obs, reward, done, info = self.env.step(sim_action, verbose=False)
                    num_sim_steps += 1

        try:
            vid_name = path.join(self.prompter.env_round_save_dir, f"replay_video.mp4")
            self.env.export_render_to_video(vid_name, out_type=self.video_format, fps=50)
        except:
            pass

        return obs, reward, done, info

    def test_one_episode(self, initial_obs):
        """
        按 LLM 想象的轨迹与环境交互一个 episode
        """
        initial_state = self.env.get_state_prompt(initial_obs)  # 获取初始状态的文本描述
        path_exist = self.prompter.check_path_exist(initial_state, self.env.target_state_desp)  # 判断是否存在候选最优轨迹

        # 设置当前轮次环境交互的保存目录 interact/round_{episode_id}
        env_round_save_dir = path.join(self.prompter.interact_save_dir, f"round_{self.episode_id}")
        self.prompter.set_env_round_dir(env_round_save_dir)

        self.prompter.log(f"-------------- Start testing case {self.episode_id} --------------", if_test=True)
        self.prompter.log(f"Initial state: {initial_state}", if_test=True)
        self.prompter.log(f"Target state: {self.env.target_state_desp}", if_test=True)

        IK_failed = False
        step, failed = 0, False
        obs = initial_obs
        state = initial_state
        # 与环境交互一个 episode
        while step < self.max_runner_steps:
            self.prompter.log(f"------ Step {step} ------", if_test=True)

            # 更新临时树及其 UCB 值
            self.prompter.fill_temp_lang_tree(state, self.env.target_state_desp)

            # 获取动作 a 即 llm_plan
            ready_to_execute, current_llm_plan, path_exist = self.prompter.decision_with_model(
                obs, path_exist=path_exist
            )

            if ready_to_execute is None and current_llm_plan is None and path_exist is None:
                self.prompter.log("The run failed due to IK failure. Restart the interaction test.", if_test=True)
                IK_failed = True
                break

            if not ready_to_execute:
                failed = True
                break

            # 在环境中执行动作 a
            # obs, reward, done, info = self.roco_env_step(current_llm_plan)
            obs, _, done, _ = self.roco_env_step(current_llm_plan)
            state = self.env.get_state_prompt(obs)

            # 日志记录 transition
            self.prompter.log(f"Action: \n{current_llm_plan[0].parsed_proposal}", if_test=True)
            self.prompter.log(f"State: \n{state}", if_test=True)

            step += 1

            if done:
                self.episode_id += 1
                break

        self.prompter.log("Run finished after {} timesteps\n".format(step), if_test=True)

        if IK_failed:
            return None

        if failed:
            self.prompter.log("The run failed", if_test=True)
            return -1

        return step

    def one_run_with_model(self, case_id):
        """
        完成一个测试用例
        """
        case_result = {
            'success_token': None,               # 首次成功时的 token 消耗
            'success_memory_ratio': None,        # 首次成功时的 memory 使用比例
            'converge_token': None,              # 收敛时的 token 消耗
            'converge_memory_ratio': None,       # 收敛时的 memory 使用比例
            'end_step': None,                    # 最终执行步数
            'success_end_step': None,            # 首次成功时的执行步数
            'optimal_steps': self.env.optimal_steps,  # 该 case 的最优步数
            'env_replans': None,                 # 环境反馈导致的重规划次数
        }

        obs = self.reset_env(case_id)  # 从环境中获得初始观测
        state = self.env.get_state_prompt(obs)  # 将初始观测转换为文本描述的初始状态
        self.prompter.state_desp = copy.deepcopy(state)
        
        # 初始化临时树 (根据初始状态和目标状态)
        self.prompter.initial_temp_lang_tree(
            state_desp=state, target_state_desp=self.env.target_state_desp
        )

        # (调试用) 将初始状态、目标状态和初始信息写入 Task.txt 文件中
        with open(path.join(self.prompter.case_dir, "Task.txt"), "a") as f:
            f.write(f"Original state:\n{state}\n")
            f.write(f"Target state:\n{self.env.target_state_desp}\n")
            f.write(f"Initial Information:\n{str(self.env.get_initial_information())}\n")

        print('---------- Task Information ----------')
        print(f"Original state:\n{state}\n")
        print(f"Target state:\n{self.env.target_state_desp}\n")
        print(f"Initial Information:\n{str(self.env.get_initial_information())}\n")
        print('-------------------------------------')

        # 循环尝试完成测试用例
        first_success, case_end = False, False
        while not case_end:
            obs = self.reset_env(case_id)

            # 重置当前轮次 state、action、memory
            self.prompter.round_reset()

            # 调用 LLM 进行想象
            # 直到存在一条从初始状态到目标状态的可行轨迹 或 尝试次数达到上限
            self.prompter.imagine_with_ucb(obs=obs, horizon=12, interrupt_flag=False)  # 限制 LLM 的想象次数

            if self.prompter.imagination_failure >= 3:  # 若尝试次数达到上限则直接跳过当前测试用例
                print("Imagination failure. Pass to the next case.")
                break

            # 与环境交互以进行测试
            while True:
                end_step = self.test_one_episode(initial_obs=obs)
                if end_step is not None:  # 若执行步数不为 None 则退出循环
                    break

            # 根据执行步数判断是否结束、结束原因 (成功/失败)
            case_end, result = self.get_case_end(end_step)

            if result == 'success':  # 结束且步数最优
                # if the case ends with reaching the optimal steps, we record the token and memory ratio
                assert case_end is True, "The case should end if it succeeds"

                # converge_memory_ratio 是 LLM 收敛时的效率 (LLM 和 M 使用次数)
                case_result['converge_token'], case_result['converge_memory_ratio'] = self.prompter.get_summary()
                llm_counts = case_result['converge_memory_ratio']['llm_count']
                memory_counts = case_result['converge_memory_ratio']['memory_count']
                if len(llm_counts) == 1 and (llm_counts[0] + memory_counts[0]) == 0:
                    case_result['converge_memory_ratio']['memory_count'] = [end_step]
                if case_result['success_memory_ratio'] is None:  # success_memory_ratio 是 LLM 首次成功时的效率
                    case_result['success_memory_ratio'] = case_result['converge_memory_ratio']
                    case_result['success_token'] = case_result['converge_token']
                    case_result['success_end_step'] = end_step
            elif result is None:  # 结束但步数不是最优
                if first_success is False:
                    # if the test first reaches the target state, we record the token and memory ratio
                    first_success = True
                    case_result['success_token'], case_result['success_memory_ratio'] = self.prompter.get_summary()
                    case_result['success_end_step'] = end_step
                    if len(case_result['success_memory_ratio']) == 1 and case_result['success_memory_ratio'][0] == 0:
                        case_result['success_memory_ratio'] = [end_step]
            elif result == 'failure':  # 结束但失败
                assert case_end is True, "The case should end if it fails"
            else:
                raise ValueError(f"Unknown result {result}")

            case_result['end_step'] = end_step
            case_result['env_replans'] = self.prompter.env_replans
            # 本 case 的 Q/V 值计算统计（累计到 case 结束前，最后一次写入即最终值）
            _qt = self.prompter.lang_tree.q_time
            case_result['q_compute_time_ms'] = round(_qt['total'] * 1000, 3)
            case_result['q_compute_calls'] = _qt['calls']
            case_result['fixed_teammate'] = dict(self.prompter.fixed_teammate_counts)
            with open(path.join(self.prompter.case_dir, "case_result.json"), "w") as f:
                json.dump(case_result, f, indent=2)

            # 每个 case 完成后保存 lang_tree 快照
            self.prompter.lang_tree.save()
            shutil.copy(
                self.prompter.lang_tree.tree_path,
                path.join(self.prompter.case_dir, "lang_tree.pth")
            )

        # —— case 正常收尾：写 finished 标记 ——
        # case_result.json 在循环内每轮都会写，故用 finished 区分"跑完"与"中途中断"，
        # 供 run_benchmark 断点续跑只认已完成的 case。
        case_result['finished'] = True
        with open(path.join(self.prompter.case_dir, "case_result.json"), "w") as f:
            json.dump(case_result, f, indent=2)

    def run(self, args):
        # 创建输出目录
        start_id = args.start_run_id  # run_id = 0, 1, 2, ...
        save_dir = os.path.join(self.run_dir, f"run_{start_id}")
        os.makedirs(save_dir, exist_ok=self.overwrite)  # 创建 run_{start_id} 目录

        print(f"==== Run {start_id} starts ====")

        # 保存 args 参数到 JSON 文件中
        args_dict = vars(args)
        args_dict["env"] = self.env.__class__.__name__
        timestamp = datetime.now().strftime("%Y%m_%H%M")
        fname = os.path.join(save_dir, f"args_{timestamp}.json")  # 在 run_{start_id} 目录下
        os.makedirs(os.path.dirname(fname), exist_ok=True)  # 创建 JSON 文件
        json.dump(args_dict, open(fname, "w"), indent=2)  # 写入参数

        start_case_id = args.start_case_id  # case_id = 0, 1, 2, ...
        total_test_number = min(self.env.case_number, start_case_id + args.num_cases)  # 限制本次运行需要测试的用例个数

        # 循环完成每个测试用例
        for case_id in range(start_case_id, total_test_number):
            case_dir = os.path.join(self.run_dir, f"run_{start_id}", f"case_{case_id}")  # 用例目录路径 run_{start_id}/case_{case_id}
            self.prompter.set_case_dir(case_dir)  # 创建 case_{case_id} 目录并设置为当前用例目录

            self.prompter.lang_tree.refine_tree()  # 修复永久树中的断裂问题 (初始化永久树？)

            print(f"============ Case {case_id} starts ============")
            self.episode_id = 0
            self.prompter.case_reset()
            try:
                self.one_run_with_model(case_id)
            except LLMQuotaExceeded as qe:
                # 配额无法靠等待区间重置恢复（如周额度耗尽）→ 打印续跑命令并干净退出
                rel = os.path.relpath(self.run_dir, self.data_dir).replace(os.sep, '/')
                epoch = '/'.join(rel.split('/')[1:])   # <algo>/<YYYYMMDD_HHMM>
                print(f"\n[quota] {qe}")
                print(f"本批次输出目录: {self.run_dir}")
                print("额度恢复后用以下命令续跑（断点恢复，已完成的 case 不会重跑）：")
                print(f"    python run_benchmark.py --epoch {epoch} --llm {self.prompter.llm_source}"
                        + (' --compare_q' if self.compare_q else ''))
                sys.exit(1)
            finally:
                # case 收尾：写本 case 的 Q 一致性表（--compare_q 关闭时为空操作）
                self.prompter.finish_q_consistency()
            print(f"============ Case {case_id} ends ============\n\n")
