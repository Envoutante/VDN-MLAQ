"""
按 optimal_steps 从 1 到 N 依次运行实验
每步完成后自动保存 lang_tree，由下一步加载

用法：
    python run_benchmark.py                    # 默认配置
    python run_benchmark.py --max_opt 4        # 只跑到 optimal_steps=4
"""

import subprocess
import sys
import os
import shutil
import re

# ---------- 默认配置 ----------
TASK = "sort_dialog"
LLM_SOURCE = "deepseek-v4-flash"
TSTEPS = 8
NUM_RUNS = 1
MAX_OPTIMAL_STEPS = 6
# -----------------------------


def _find_resume_point(data_root: str):
    """
    扫描 data_root 下的 run_{id} 目录，找到最大的 run_id 中已完成的最大 case_id。
    返回 (resume_run_id, resume_case_id, restored_tree)。
    若无任何已完成的 case，返回 (0, 0, False)。
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

    # 找出该 run 下所有有 case_result.json 的 case 目录
    max_case = -1
    for entry in os.listdir(run_dir):
        m = re.match(r"case_(\d+)$", entry)
        if not m:
            continue
        case_id = int(m.group(1))
        result_path = os.path.join(run_dir, entry, "case_result.json")
        if os.path.isfile(result_path) and case_id > max_case:
            max_case = case_id

    if max_case >= 0:
        resume_run_id = max_run
        resume_case_id = max_case + 1

        # 从该 case 恢复 lang_tree 到共享路径
        case_tree = os.path.join(run_dir, f"case_{max_case}", "lang_tree.pth")
        shared_tree = os.path.join(data_root, "lang_tree.pth")
        if os.path.isfile(case_tree):
            shutil.copy(case_tree, shared_tree)
            restored_tree = True

    return resume_run_id, resume_case_id, restored_tree


def main():
    # 解析简易参数
    max_opt = MAX_OPTIMAL_STEPS
    for arg in sys.argv[1:]:
        if arg.startswith("--max_opt="):
            max_opt = int(arg.split("=")[1])
        elif arg.startswith("--max_opt") and len(sys.argv) > sys.argv.index(arg) + 1:
            max_opt = int(sys.argv[sys.argv.index(arg) + 1])

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "data", TASK)

    # 断点续跑：找到上次中断的位置
    resume_run_id, resume_case_id, restored_tree = _find_resume_point(data_root)

    base_args = [
        sys.executable, "run_MLAQ.py",
        "--task", TASK,
        "--llm_source", LLM_SOURCE,
        "--tsteps", str(TSTEPS),
        "--num_runs", str(NUM_RUNS),
        "--skip_display",
    ]

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
            print(f"  Resume command: cd {project_root} && python run_benchmark.py")
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
