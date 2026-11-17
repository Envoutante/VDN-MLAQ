from runtime_layout import ALGORITHM, VALUE_MODE, PROJECT_ROOT, DATA_ROOT, bootstrap, validate_epoch
bootstrap()

import os
import ssl

# Fix Windows SSL cert store corruption issue in Python 3.8
import certifi
ssl.create_default_context = (
    lambda purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None:
    ssl._create_default_https_context(purpose, cafile=certifi.where())
)

# Fix requests/urllib3 SSL: force certifi CA bundle and disable system proxy
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import json
import openai
import logging
from datetime import datetime
import argparse

# Bypass Windows system proxy (detected from WinHTTP settings) for openai
openai.proxy = {"http": "", "https": ""}
# openai 0.27 调 session.request 时不传 proxies，系统代理会被合并到请求级并覆盖上面的
# 空字符串会话级代理，导致开启系统代理后请求仍走代理（Clash 7890），TLS 握手被截断报 EOF。
# no_proxy="*" 让 requests 对所有 URL 都绕过系统代理，强制直连。
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

# from runner.roco_runner import LLMRunner, TASK_NAME_MAP
from runner.roco_runner import LLMRunner, TASK_NAME_MAP
from runner.text_runner import LLMRunner as TextLLMRunner


def _import_blocksworld():
    from environments.blocksworld.task_blocksworld import BlocksWorldTask
    return BlocksWorldTask


# print out logging.info
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)

# 从本地 .env 读取 API 配置（.env 已被 .gitignore 忽略，不提交，避免密钥泄露）
def _load_env_file(env_path=PROJECT_ROOT / ".env"):
    """极简 .env 解析：key=value 逐行读取，忽略空行与 # 注释。"""
    if not os.path.exists(env_path):
        return {}
    env = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_env = _load_env_file()

# 按 --llm_source 选择对应的 .env 字段：(key 字段, base 字段, 默认 base)
# 与各 prompter 的 assert 白名单保持一致；未登记的模型直接报错。
_LLM_ENV_FIELDS = {
    "deepseek-v4-flash":  ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    "deepseek-v4-pro":    ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    "MiniMax-M3":         ("MINIMAX_API_KEY",  "MINIMAX_API_BASE",  "https://api.minimax.cn/v1"),
    "gpt-4":              ("OPENAI_API_KEY",   "OPENAI_API_BASE",   "https://api.openai.com/v1"),
    "gpt-3.5-turbo":      ("OPENAI_API_KEY",   "OPENAI_API_BASE",   "https://api.openai.com/v1"),
    "gpt-4-1106-preview": ("OPENAI_API_KEY",   "OPENAI_API_BASE",   "https://api.openai.com/v1"),
    "gpt-4-0125-preview": ("OPENAI_API_KEY",   "OPENAI_API_BASE",   "https://api.openai.com/v1"),
    "claude":             ("OPENAI_API_KEY",   "OPENAI_API_BASE",   "https://api.openai.com/v1"),
}


def _setup_llm_api(llm_source):
    """按模型名从 .env 读取对应的 API key/base，并写入 openai 全局配置。"""
    if llm_source not in _LLM_ENV_FIELDS:
        raise RuntimeError(
            f"未知模型 {llm_source}：请在 run_MLAQ.py 的 _LLM_ENV_FIELDS 中登记其 key/base 字段"
        )
    key_field, base_field, default_base = _LLM_ENV_FIELDS[llm_source]
    api_key = _env.get(key_field, "")
    api_base = _env.get(base_field, default_base)
    if not api_key:
        raise RuntimeError(
            f"未在 .env 中找到 {key_field}（模型 {llm_source} 对应）。请在项目根目录 .env 中写入："
            f"{key_field}=sk-xxxx 与 {base_field}={default_base}（.env 不会提交到 git）"
        )
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["LLM_API_KEY"] = api_key      # 供 tools/minimax_quota 查 Token Plan 余量
    openai.api_base = api_base
    openai.api_key = api_key


def main(args):
    if args.task != "sort_dialog":
        raise ValueError("VDN v3 local panel observations support sort_dialog only")
    if args.epoch:
        args.epoch = validate_epoch(args.epoch)
    args.data_dir = str((PROJECT_ROOT / args.data_dir).resolve())
    _setup_llm_api(args.llm_source)   # 按 --llm_source 载入对应模型的 key/base
    # 本次实验的输出标识: <算法>/<YYYYMMDD_HHMM>，落到 data/<task>/<epoch_id> 下
    # 每次启动自动追加分钟时间戳，避免不同算法/批次互相覆盖。
    # 若显式传 --epoch（如 run_benchmark 多阶段续跑要固定同一批次），则原样使用
    epoch_id = args.epoch if args.epoch else "{}/{}".format(ALGORITHM, datetime.now().strftime("%Y%m%d_%H%M"))

    if args.task in ['sort_dialog', 'sandwich_dialog', 'sort_central', 'sandwich_central']:
        assert args.task in TASK_NAME_MAP.keys(), f"Task {args.task} not supported"
        env_cl = TASK_NAME_MAP[args.task]

        render_freq = 600
        if args.control_freq == 15:
            render_freq = 1200
        elif args.control_freq == 10:
            render_freq = 2000
        elif args.control_freq == 5:
            render_freq = 3000
        env = env_cl(
            render_freq=render_freq,
            image_hw=(400, 400),
            sim_forward_steps=300,
            error_freq=30,
            error_threshold=1e-5,
            randomize_init=True,
            render_point_cloud=0,
            render_cameras=["face_panda", "face_ur5e", "teaser", ],
            one_obj_each=True,
        )
        # 为环境设置最优步数
        env.set_optimal_steps(args.optimal_steps)

        # 获取环境中的机械臂
        robots = env.get_sim_robots()
        
        # 创建任务文件夹
        if not os.path.exists(args.data_dir + "/{}".format(args.task)):
            os.makedirs(args.data_dir + "/{}".format(args.task))

        # 创建 runner
        runner = LLMRunner(
            env=env,
            data_dir=args.data_dir,
            robots=robots,
            max_runner_steps=args.tsteps,
            num_runs=args.num_runs,
            run_name=args.run_name,
            epoch_id=epoch_id,
            overwrite=True,
            skip_display=args.skip_display,
            llm_output_mode=args.output_mode,  # "action_only" or "action_and_path"
            llm_comm_mode=args.comm_mode,  # "chat" or "plan"
            llm_num_replans=args.num_replans,
            policy_kwargs=dict(
                control_freq=args.control_freq,
                use_weld=args.use_weld,
                skip_direct_path=0,
                skip_smooth_path=0,
                check_relative_pose=args.rel_pose,
            ),
            direct_waypoints=args.direct_waypoints,
            max_failed_waypoints=args.max_failed_waypoints,
            debug_mode=args.debug_mode,
            split_parsed_plans=args.split_parsed_plans,
            use_history=(not args.no_history),
            use_feedback=(not args.no_feedback),
            temperature=args.temperature,
            llm_source=args.llm_source,
            tree_load=args.tree_load,
            skip_check_memory=args.skip_check_memory,
            algo=VALUE_MODE,   # algo 名选择：mlaq=值迭代 / vdn-mlaq=遍历式 VDN
            compare_q=args.compare_q,   # 旁路记录两套 Q 与动作选择的一致性（每个 case 一个 excel）
        )

        # 运行 runner
        runner.run(args)
    elif args.task in ['blocksworld']:
        BlocksWorldTask = _import_blocksworld()
        env = BlocksWorldTask(initial_coef=0.7, target_coef=0.3)
        env.set_optimal_steps(args.optimal_steps)
        if args.no_feedback:
            assert args.num_replans == 1, "no feedback mode requires num_replans=1 but longer -tsteps"

        # 创建 runner
        runner = TextLLMRunner(
            env=env,
            data_dir=args.data_dir,
            max_runner_steps=args.tsteps,
            num_runs=args.num_runs,
            run_name=args.run_name,
            epoch_id=epoch_id,
            overwrite=True,
            skip_display=args.skip_display,
            llm_num_replans=args.num_replans,
            debug_mode=args.debug_mode,
            use_history=(not args.no_history),
            use_feedback=(not args.no_feedback),
            temperature=args.temperature,
            llm_source=args.llm_source,
            tree_load=args.tree_load,
            skip_check_memory=args.skip_check_memory,
        )

        # 运行 runner
        runner.run(args)
    else:
        raise ValueError(f"Task {args.task} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # special parameters in LAQ
    parser.add_argument("--data_dir", "-d", type=str, default=str(DATA_ROOT))
    parser.add_argument("--epoch", type=str, default="", help="显式指定批次标识（如 mlaq/20260729_1906），覆盖自动时间戳；供 run_benchmark 等多阶段续跑固定同一批次")
    parser.add_argument("--task", type=str, default="sandwich_dialog")  # "sandwich_dialog", "sort_dialog", "sandwich_central", "sort_central", "blocksworld"
    parser.add_argument("--start_run_id", "-sri", type=int, default=0)
    parser.add_argument("--start_case_id", "-sci", type=int, default=0)
    parser.add_argument("--num_cases", type=int, default=30, help="本次最多跑几个 case（从 start_case_id 起），测试用可设小值")
    parser.add_argument("--tree_load", "-tl", action="store_true")
    parser.add_argument("--llm_source", "-llm", type=str, default="gpt-4-0125-preview")
    # optimal steps for different domains
    # [2, 4, 6, 8, 10, 12] for Blocksworld
    # [1, 2, 3, 4, 5, 6] for Sort
    # [6, 8, 10] for Sandwich
    parser.add_argument("--optimal_steps", "-ostep", type=int, default=2)
    parser.add_argument("--temperature", "-temp", type=float, default=0)
    parser.add_argument("--tsteps", "-t", type=int, default=20)     # 20 for Blocksworld, 16 for Sandwich, 8 for Sort

    # parameters in RoCo codes, some is not used in LAQ. We keep them for compatibility.
    parser.add_argument("--run_name", "-rn", type=str, default="test")
    parser.add_argument("--cycle_id", "-ci", type=int, default=0)
    parser.add_argument("--start_id", "-sid", type=int, default=-1)
    parser.add_argument("--num_runs", '-nruns', type=int, default=1)
    parser.add_argument("--output_mode", type=str, default="action_only", choices=["action_only", "action_and_path"])
    parser.add_argument("--comm_mode", type=str, default="dialog", choices=["chat", "plan", "dialog"])
    parser.add_argument("--control_freq", "-cf", type=int, default=15)
    parser.add_argument("--skip_display", "-sd", action="store_true")
    parser.add_argument("--direct_waypoints", "-dw", type=int, default=5)
    parser.add_argument("--num_replans", "-nr", type=int, default=5)
    parser.add_argument("--cont", "-c", action="store_true")
    parser.add_argument("--load_run_name", "-lr", type=str, default="sort_task")
    parser.add_argument("--load_run_id", "-ld", type=int, default=0)
    parser.add_argument("--max_failed_waypoints", "-max", type=int, default=1)
    parser.add_argument("--debug_mode", "-i", action="store_true")
    parser.add_argument("--use_weld", "-w", type=int, default=1)
    parser.add_argument("--rel_pose", "-rp", action="store_true")
    parser.add_argument("--split_parsed_plans", "-sp", action="store_true")
    parser.add_argument("--no_history", "-nh", action="store_true")
    parser.add_argument("--no_feedback", "-nf", action="store_true")
    parser.add_argument("--pre_imagine", "-pi", action="store_true")
    parser.add_argument("--skip_check_memory", "-sca", action="store_true")
    parser.add_argument("--compare_q", action="store_true",
                        help="旁路记录两套 Q（MLAQ 值迭代 / VDN）与动作选择的一致性，每个 case 写一个 q_consistency.xlsx")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    main(args)
