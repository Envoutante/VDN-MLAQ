# VDN-MLAQ v1.1

从 [v1](../vdn_mlaq_v1/README.md) 派生，用于单独检验全局虚拟节点初始访问次数的影响。

唯一算法改动是把新建全局虚拟节点的 `visit_num` 从 `2` 降为 `1`。虚拟节点价值仍为 `-0.1`，全局 UCB 探索系数为 `2`，折扣因子为 `0.995`。全局搜索选中虚拟动作时仍让所有智能体联合想象。

在项目根目录激活 `mlaq` 环境后运行：

```bash
nohup python -u algorithms/vdn_mlaq_v1_1/run_benchmark.py \
  --llm deepseek-v4-flash --compare_q \
  > "vdn_mlaq_v1_1_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

默认批量任务为 `sort_dialog`。结果写入 `data/sort_dialog/vdn_mlaq_v1_1/<批次>/`。续跑时增加 `--epoch vdn_mlaq_v1_1/<批次>`。与 v1 做对照时使用相同模型、任务顺序、初始记忆和推理预算。

离线检查覆盖初始化、固定图上的 UCB 选择和批量入口。在线路径步数与 token 成本待测。
