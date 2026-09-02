"""实验输出路径的统一定位。

目录约定（变体 B：任务 → 算法 → 批次）：
    data/<task>/                        # 未指定 epoch_id 时的旧默认（不再用于新实验）
    data/<task>/<epoch_id>/             # 传 epoch_id = "<算法>/<YYYYMMDD_HHMM>" 时的实验根目录
其中 <task> 即 env.name，例如：
    data/sort_dialog/mlaq/20260902_1530/
        run_0/case_0/...                 # 详细输出
        lang_tree.pth  memory.pth        # 本批次自有的树与记忆

所有 runner / prompter 构造输出路径时都应经过这里，避免多处硬编码导致错位。
"""
import os


def exp_output_dir(env_name: str, epoch_id=None, data_dir: str = "data") -> str:
    """返回本次实验的输出根目录。epoch_id 为空时退化为 data_dir/<task>。"""
    root = os.path.join(data_dir, env_name)
    return os.path.join(root, str(epoch_id)) if epoch_id else root
