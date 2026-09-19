"""
按 optimal_steps 从 1 到 N 依次运行实验
每步完成后自动保存 lang_tree，由下一步加载

输出遵循 data/<task>/<algo>/<批次>/ 目录约定（变体 B）：
整个 benchmark 的一次完整运行落在同一个批次里，例如
    data/sort_dialog/mlaq/20260729_1906/
        run_0/ ... run_5/        # 每个 optimal_steps 一步
        lang_tree.pth  memory.pth

用法：
    python run_benchmark.py                              # 默认配置，新建一个批次
    python run_benchmark.py --max_opt 4                  # 只跑到 optimal_steps=4
    python run_benchmark.py --llm MiniMax-M3             # 指定模型（默认取 LLM_SOURCE）
    python run_benchmark.py --epoch mlaq/20260729_1906   # 续跑/复用指定批次
"""

import subprocess
import sys
import os
import shutil
import re
import json
from datetime import datetime

# ---------- 默认配置 ----------
TASK = "sort_dialog"
ALGO = "mlaq"
LLM_SOURCE = "deepseek-v4-flash"
TSTEPS = 8
NUM_RUNS = 1
MAX_OPTIMAL_STEPS = 6
# -----------------------------


def _find_resume_point(data_root: str):
    """
    扫描 data_root 下的 run_{id} 目录，找到最大的 run_id 中"最早的未完成 case"。
    返回 (resume_run_id, resume_case_id, restored_tree)。
    若无任何 case 目录，返回 (0, 0, False)。
    """
    resume_run_id = 0
    resume_case_id = 0
    restored_tree = False

    if not os.path.isdir(data_root):
        return resume_run_id, resume_case_id, restored_tree

    # 找出所有 run_{id} 目录
    run_dirs = []
    for entry in os.listdir(data_root):
        m = re.match(r"run_(\d+)$", entry)
        if m and os.path.isdir(os.path.join(data_root, entry)):
            run_dirs.append(int(m.group(1)))

    if not run_dirs:
        return resume_run_id, resume_case_id, restored_tree

    max_run = max(run_dirs)
    run_dir = os.path.join(data_root, f"run_{max_run}")

    # 判定每个 case 是否"正常收尾"（case_result.json 内含 finished: true）
    # 注：case_result.json 每个 round 都会写，只靠"文件存在"会把中途中断的 case
    # 误判为已完成；故以 finished 标记为准。
    finished, unfinished = set(), set()
    for entry in os.listdir(run_dir):
        m = re.match(r"case_(\d+)$", entry)
        if not m or not os.path.isdir(os.path.join(run_dir, entry)):
            continue
        case_id = int(m.group(1))
        ok = False
        result_path = os.path.join(run_dir, entry, "case_result.json")
        if os.path.isfile(result_path):
            try:
                with open(result_path, encoding="utf-8") as f:
                    ok = json.load(f).get("finished") is True
            except (OSError, ValueError):
                ok = False
        (finished if ok else unfinished).add(case_id)

    if not finished and not unfinished:
        return resume_run_id, resume_case_id, restored_tree

    resume_run_id = max_run
    # 从"最早的未完成 case"续跑（避免跳过被中断的 case）；全部完成则从最大已完成 +1
    resume_case_id = min(unfinished) if unfinished else max(finished) + 1

    # 恢复 lang_tree：取"续跑点前一个 case"的快照（即进入该 case 时的树状态）
    shared_tree = os.path.join(data_root, "lang_tree.pth")
    prev_tree = os.path.join(run_dir, f"case_{resume_case_id - 1}", "lang_tree.pth")
    if resume_case_id > 0 and os.path.isfile(prev_tree):
        shutil.copy(prev_tree, shared_tree)
        restored_tree = True

    return resume_run_id, resume_case_id, restored_tree


def main():
    # 解析简易参数
    max_opt = MAX_OPTIMAL_STEPS
    algo = ALGO
    llm = LLM_SOURCE
    epoch = None
    compare_q = False
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--max_opt" and i + 1 < len(sys.argv):
            max_opt = int(sys.argv[i + 1]); i += 1
        elif arg.startswith("--max_opt="):
            max_opt = int(arg.split("=")[1])
        elif arg == "--algo" and i + 1 < len(sys.argv):
            algo = sys.argv[i + 1]; i += 1
        elif arg.startswith("--algo="):
            algo = arg.split("=")[1]
        elif arg == "--llm" and i + 1 < len(sys.argv):
            llm = sys.argv[i + 1]; i += 1
        elif arg.startswith("--llm="):
            llm = arg.split("=")[1]
        elif arg == "--epoch" and i + 1 < len(sys.argv):
            epoch = sys.argv[i + 1]; i += 1
        elif arg.startswith("--epoch="):
            epoch = arg.split("=")[1]
        elif arg == "--compare_q":
            compare_q = True
        i += 1

    project_root = os.path.dirname(os.path.abspath(__file__))
    # 批次标识: 未显式指定时按当前时间新建一个，保证每次完整 benchmark 独立成批
    if epoch is None:
        epoch = "{}/{}".format(algo, datetime.now().strftime("%Y%m%d_%H%M"))
    else:
        algo = epoch.split("/")[0]
    # data_root = data/<task>/<algo>/<YYYYMMDD_HHMM>
    data_root = os.path.join(project_root, "data", TASK, *epoch.split("/"))
    print(f"本批次输出目录: {data_root}")
    os.makedirs(data_root, exist_ok=True)

    # 断点续跑：找到上次中断的位置
    resume_run_id, resume_case_id, restored_tree = _find_resume_point(data_root)

    base_args = [
        sys.executable, "run_MLAQ.py",
        "--task", TASK,
        "--algo", algo,
        "--epoch", epoch,
        "--llm_source", llm,
        "--tsteps", str(TSTEPS),
        "--num_runs", str(NUM_RUNS),
        "--skip_display",
    ]
    if compare_q:
        base_args.append("--compare_q")

    for opt in range(resume_run_id + 1 if resume_case_id > 0 else 1, max_opt + 1):
        run_id = opt - 1
        shared_tree = os.path.join(data_root, "lang_tree.pth")
        run_dir = os.path.join(data_root, f"run_{run_id}")

        start_case = resume_case_id if run_id == resume_run_id else 0
        need_tree_load = restored_tree or opt > 1

        args = base_args + [
            "--optimal_steps", str(opt),
            "--start_run_id", str(run_id),
            "--start_case_id", str(start_case),
        ]
        if need_tree_load:
            args.append("--tree_load")

        print(f"\n{'='*60}")
        print(f"  Running optimal_steps={opt}  run_id={run_id}  start_case={start_case}")
        print(f"{'='*60}")
        sys.stdout.flush()

        result = subprocess.run(args, cwd=project_root)
        if result.returncode != 0:
            print(f"ERROR: optimal_steps={opt} failed with code {result.returncode}")
            print(f"  Resume command: cd {project_root} && python run_benchmark.py --epoch {epoch}")
            sys.exit(1)

        print(f"  optimal_steps = {opt} completed")

        # 备份 lang_tree 到 run 目录
        if os.path.exists(shared_tree):
            shutil.copy(shared_tree, os.path.join(run_dir, "lang_tree.pth"))
            print(f"  backup saved at run_{run_id}/lang_tree.pth")

        # 第一个续跑完成后，后续 run 不再需要特判
        resume_case_id = 0


if __name__ == "__main__":
    main()
