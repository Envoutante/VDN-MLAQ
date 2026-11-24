"""
按 optimal_steps 从 1 到 N 依次运行实验
每步完成后自动保存 lang_tree，由下一步加载

输出遵循 data/<task>/<algo>/<批次>/ 目录约定（变体 B）：
整个 benchmark 的一次完整运行落在同一个批次里，例如
    data/sort_dialog/mlaq_langmarl_v2/20260729_1906/
        run_0/ ... run_5/        # 每个 optimal_steps 一步
        lang_tree.pth  memory.pth

用法：
    python run_benchmark.py                              # 默认配置，新建一个批次
    python run_benchmark.py --max_opt 4                  # 只跑到 optimal_steps=4
    python run_benchmark.py --llm MiniMax-M3             # 指定模型（默认取 LLM_SOURCE）
    python run_benchmark.py --epoch mlaq_langmarl_v2/20260729_1906   # 续跑/复用指定批次
"""

from runtime_layout import ALGORITHM, ALGORITHM_ROOT, PROJECT_ROOT, bootstrap, validate_epoch
bootstrap()

import subprocess
import sys
import shutil
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
from datetime import datetime

# ---------- 默认配置 ----------
TASK = "sort_dialog"
ALGO = ALGORITHM
LLM_SOURCE = "deepseek-v4-flash"
TSTEPS = 8
NUM_RUNS = 1
CASES_PER_RUN = 30
MAX_OPTIMAL_STEPS = 6
# -----------------------------


def _benchmark_case_counts():
    # Keep the same case order and sizes as the Sort dialog environment.
    cases_path = ALGORITHM_ROOT / "environments" / "rocobench" / "dialog_envs" / "task_sort_cases.json"
    with cases_path.open(encoding="utf-8") as file:
        available = Counter(case["optimal_steps"] for case in json.load(file))
    return {run_id: min(CASES_PER_RUN, available[run_id + 1])
            for run_id in range(MAX_OPTIMAL_STEPS)}


def _case_finished(case_dir):
    try:
        with (case_dir / "case_result.json").open(encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, ValueError):
        return False
    return isinstance(result, dict) and result.get("finished") is True


def _pending_case_ranges(run_dir, case_count):
    """Yield half-open runs of unfinished or missing cases in the benchmark."""
    start = None
    for case_id in range(case_count):
        if _case_finished(run_dir / f"case_{case_id}"):
            if start is not None:
                yield start, case_id
                start = None
        elif start is None:
            start = case_id
    if start is not None:
        yield start, case_count


def _restore_tree_before(data_root, run_id, start_case, case_counts):
    shared_tree = data_root / "lang_tree.pth"
    for prior_run in range(run_id, -1, -1):
        end = start_case if prior_run == run_id else case_counts[prior_run]
        for case_id in range(end - 1, -1, -1):
            case = data_root / f"run_{prior_run}" / f"case_{case_id}"
            snapshot = case / "lang_tree.pth"
            if snapshot.is_file() and _case_finished(case):
                with NamedTemporaryFile(dir=data_root, delete=False) as file:
                    temporary = Path(file.name)
                try:
                    shutil.copyfile(snapshot, temporary)
                    temporary.replace(shared_tree)
                finally:
                    temporary.unlink(missing_ok=True)
                print(f"  Restored tree from {snapshot.relative_to(data_root)}")
                return True

    return (run_id > 0 or start_case > 0) and shared_tree.is_file()


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Fixed algorithm: " + ALGORITHM + "\nOptions: --max_opt N --llm MODEL --epoch " + ALGORITHM + "/BATCH --compare_q")
        return
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
        else:
            raise ValueError(f'Unknown argument: {arg}')
        i += 1

    if not 1 <= max_opt <= MAX_OPTIMAL_STEPS:
        raise ValueError(f"--max_opt must be between 1 and {MAX_OPTIMAL_STEPS}")

    algorithm_root = str(ALGORITHM_ROOT)
    # 批次标识: 未显式指定时按当前时间新建一个，保证每次完整 benchmark 独立成批
    if epoch is None:
        epoch = "{}/{}".format(algo, datetime.now().strftime("%Y%m%d_%H%M"))
    else:
        epoch = validate_epoch(epoch)
    # data_root = data/<task>/<algo>/<YYYYMMDD_HHMM>
    data_root = PROJECT_ROOT / "data" / TASK / epoch
    print(f"本批次输出目录: {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    case_counts = _benchmark_case_counts()

    base_args = [
        sys.executable, "run_MLAQ.py",
        "--task", TASK,
        "--epoch", epoch,
        "--llm_source", llm,
        "--tsteps", str(TSTEPS),
        "--num_runs", str(NUM_RUNS),
        "--skip_display",
    ]
    if compare_q:
        base_args.append("--compare_q")

    shared_tree = data_root / "lang_tree.pth"
    for run_id in range(max_opt):
        opt = run_id + 1
        run_dir = data_root / f"run_{run_id}"
        pending = list(_pending_case_ranges(run_dir, case_counts[run_id]))
        if not pending:
            print(f"  run_{run_id} already completed its {case_counts[run_id]} benchmark cases")
            continue

        for start_case, end_case in pending:
            args = base_args + [
                "--optimal_steps", str(opt),
                "--start_run_id", str(run_id),
                "--start_case_id", str(start_case),
                "--num_cases", str(end_case - start_case),
            ]
            if _restore_tree_before(data_root, run_id, start_case, case_counts):
                args.append("--tree_load")

            print(f"\n{'='*60}")
            print(f"  Running optimal_steps={opt}  run_id={run_id}  cases={start_case}..{end_case - 1}")
            print(f"{'='*60}")
            sys.stdout.flush()

            result = subprocess.run(args, cwd=algorithm_root)
            if result.returncode != 0:
                print(f"ERROR: optimal_steps={opt} failed with code {result.returncode}")
                print(f"  Resume command: python algorithms/{ALGORITHM}/run_benchmark.py --epoch {epoch} "
                      f"--llm {llm} --max_opt {max_opt}"
                      + (" --compare_q" if compare_q else ""))
                sys.exit(1)

        print(f"  optimal_steps = {opt} completed")
        if shared_tree.is_file():
            shutil.copyfile(shared_tree, run_dir / "lang_tree.pth")
            print(f"  backup saved at run_{run_id}/lang_tree.pth")


if __name__ == "__main__":
    main()
