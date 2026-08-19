"""
MLAQ / RoCo 指标汇总脚本
根据 case_result.json 计算实验指标
"""

import json
import os
import glob


def find_case_results(run_dir: str) -> list:
    """扫描 run_dir 下所有 case_*/case_result.json"""
    pattern = os.path.join(run_dir, "case_*", "case_result.json")
    return sorted(glob.glob(pattern))


def load_result(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _is_success(r: dict) -> bool:
    """判断 case 是否成功：success_token 存在且 input_token 非空"""
    st = r.get("success_token")
    if st is None:
        return False
    return len(st.get("input_token", [])) > 0


# ---------- Success Rate ----------

def compute_success_rate(results: list) -> float:
    """
    Success Rate: 至少一次到达目标状态的 case 占比
    S_i = 1 if success_token != null, else 0
    """
    if not results:
        return 0.0
    success = sum(1 for r in results if _is_success(r))
    return success / len(results)


def _get_memory_ratio_field(r: dict, field: str) -> dict:
    """获取 memory_ratio 数据，优先 converge 再 fallback 到 success"""
    val = r.get(field)
    if val is not None and val.get("llm_count") is not None:
        return val
    return None


def compute_case_memory_ratio(mr: dict) -> float:
    """
    计算单个 case 的 Memory Re-Util. Ratio
    ratio = M / (M + L)
    处理 0/0: 当成全部来自 memory (ratio=1.0)
    """
    llm = sum(mr.get("llm_count", []))
    mem = sum(mr.get("memory_count", []))
    total = llm + mem
    if total == 0:
        return 1.0
    return mem / total


# ---------- Memory Re-Util. Ratio ----------

def compute_memory_reutilization(
    results: list,
    field: str = "converge_memory_ratio"
) -> float:
    """
    Memory Re-Util. Ratio: 复用已有 memory transition 的比例
    取有数据的 case 的平均 ratio
    """
    ratios = []
    for r in results:
        mr = _get_memory_ratio_field(r, field)
        if mr is not None:
            ratios.append(compute_case_memory_ratio(mr))
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


# ---------- Optimal Rate ----------

def _get_optimal_steps(r: dict, default: int = 0) -> int:
    return r.get("optimal_steps", default)


def _is_optimal_conv(r: dict) -> bool:
    """收敛时是否达到最优步数 (end_step == optimal_steps 已蕴含 converge_token 存在)"""
    end_step = r.get("end_step")
    opt = _get_optimal_steps(r)
    return end_step is not None and end_step == opt


def _is_optimal_first(r: dict) -> bool:
    """首次成功时是否已是最优步数"""
    if not _is_success(r):
        return False
    sess = r.get("success_end_step")
    opt = _get_optimal_steps(r)
    return sess is not None and sess == opt


def compute_optimal_rate_conv(results: list) -> float:
    """Optimal Rate @ Convergence"""
    if not results:
        return 0.0
    opt = sum(1 for r in results if _is_optimal_conv(r))
    return opt / len(results)


def compute_optimal_rate_first(results: list) -> float:
    """Optimal Rate @ First Success"""
    if not results:
        return 0.0
    opt = sum(1 for r in results if _is_optimal_first(r))
    return opt / len(results)


# ---------- Average Token ----------

def _get_token_field(r: dict, field: str) -> dict:
    """获取 token 字段（如 success_token）"""
    val = r.get(field)
    if val is None:
        return None
    if len(val.get("input_token", [])) == 0:
        return None
    return val


def _sum_tokens(tokens: dict) -> int:
    return sum(tokens.get("input_token", [])) + sum(tokens.get("output_token", []))


def compute_average_token(results: list, field: str = "converge_token") -> float:
    """
    Average Token: 有数据的 case 的 token 总和的平均值
    T_i = sum(input_token) + sum(output_token)
    """
    vals = []
    for r in results:
        tf = _get_token_field(r, field)
        if tf is not None:
            vals.append(_sum_tokens(tf))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ---------- Optimal Gap ----------

def compute_optimal_gap(results: list, field: str = "converge_token") -> float:
    """
    Optimal Gap: 实际步数与最优步数的差的平均值
    G_i = end_step - optimal_steps
    与是否收敛无关，对所有有执行步数的 case 统计
    field == "success_token": 用 success_end_step (首次成功步数)
    field == "converge_token": 用 end_step (最终执行步数)
    """
    gaps = []
    for r in results:
        if field == "success_token":
            step = r.get("success_end_step")
        else:
            step = r.get("end_step")
        opt = _get_optimal_steps(r)
        if step is not None and opt is not None:
            gaps.append(step - opt)
    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)


# ---------- Avg Env Replans ----------

def compute_avg_env_replans(results: list) -> float:
    """所有 case 的环境重规划次数的平均值"""
    replans = [r.get("env_replans", 0) for r in results if r.get("env_replans") is not None]
    if not replans:
        return 0.0
    return sum(replans) / len(replans)


def _compute_all_metrics(results: list) -> dict:
    """计算单个 run 的所有指标，返回 {指标名: 值}"""
    return {
        "Success Rate": compute_success_rate(results),
        "Avg Env Replans": compute_avg_env_replans(results),
        "Optimal Rate @ First Success": compute_optimal_rate_first(results),
        "Optimal Rate @ Convergence": compute_optimal_rate_conv(results),
        "Average Token @ First Success": compute_average_token(results, "success_token"),
        "Average Token @ Convergence": compute_average_token(results, "converge_token"),
        "Optimal Gap @ First Success": compute_optimal_gap(results, "success_token"),
        "Optimal Gap @ Convergence": compute_optimal_gap(results, "converge_token"),
        "Memory Re-Util. Ratio @ First Success": compute_memory_reutilization(results, "success_memory_ratio"),
        "Memory Re-Util. Ratio @ Convergence": compute_memory_reutilization(results, "converge_memory_ratio"),
    }


# 每个指标对应的格式化函数
_FMT = {
    "Success Rate": lambda v: f"{v:.2%}",
    "Avg Env Replans": lambda v: f"{v:.2f}",
    "Optimal Rate @ First Success": lambda v: f"{v:.2%}",
    "Optimal Rate @ Convergence": lambda v: f"{v:.2%}",
    "Average Token @ First Success": lambda v: f"{v:,.0f}",
    "Average Token @ Convergence": lambda v: f"{v:,.0f}",
    "Optimal Gap @ First Success": lambda v: f"{v:.2f}",
    "Optimal Gap @ Convergence": lambda v: f"{v:.2f}",
    "Memory Re-Util. Ratio @ First Success": lambda v: f"{v:.2%}",
    "Memory Re-Util. Ratio @ Convergence": lambda v: f"{v:.2%}",
}

# 指标打印顺序
_METRIC_ORDER = [
    "Success Rate",
    "Avg Env Replans",
    "Optimal Rate @ First Success",
    "Optimal Rate @ Convergence",
    "Average Token @ First Success",
    "Average Token @ Convergence",
    "Optimal Gap @ First Success",
    "Optimal Gap @ Convergence",
    "Memory Re-Util. Ratio @ First Success",
    "Memory Re-Util. Ratio @ Convergence",
]


def print_metrics_table(run_dirs: list, header: list = None):
    """
    以表格形式打印多个 run 的指标。
    run_dirs: 每个元素是一个 run 目录（对应一个 optimal_steps）
    header: 各列标签（默认取 run 目录的 basename）
    最后一列为所有 run 的均值。
    """
    # 计算每个 run 的指标
    metrics_list = []
    for rd in run_dirs:
        results = [load_result(p) for p in find_case_results(rd)]
        metrics_list.append(_compute_all_metrics(results))

    if header is None:
        header = [os.path.basename(rd) for rd in run_dirs]
    header = header + ["Average"]

    # 计算列宽
    name_width = max(len(m) for m in _METRIC_ORDER) + 2
    col_width = max(len(h) for h in header) + 4
    col_width = max(col_width, 12)

    # 打印表头
    row = " " * name_width + "".join(f"{h:>{col_width}}" for h in header)
    print(row)
    print("-" * len(row))

    # 打印每个指标行
    for metric in _METRIC_ORDER:
        values = [m[metric] for m in metrics_list]
        avg = sum(values) / len(values) if values else 0.0
        row = f"{metric:<{name_width}}"
        for v in values:
            row += f"{_FMT[metric](v):>{col_width}}"
        row += f"{_FMT[metric](avg):>{col_width}}"
        print(row)


if __name__ == "__main__":
    RUN_DIRS = [
        "data/sort_dialog/run_0",
        "data/sort_dialog/run_1",
        "data/sort_dialog/run_2",
        "data/sort_dialog/run_3",
        "data/sort_dialog/run_4",
        "data/sort_dialog/run_5",
    ]
    project_root = os.path.dirname(os.path.dirname(__file__))
    run_dirs = [os.path.join(project_root, rd) for rd in RUN_DIRS]
    header = [f"opt={i}" for i in range(1, len(run_dirs) + 1)]
    print_metrics_table(run_dirs, header=header)
