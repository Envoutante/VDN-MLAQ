# MLAQ 算法实验项目

算法代码已按完整副本隔离。先激活原 `mlaq` Python 环境，在本目录运行。

| 目录 | 定义 | 来源 |
|---|---|---|
| [mlaq](algorithms/mlaq/README.md) | MLAQ 基线，原有价值迭代与联合探索 | 3f5088a 的 MLAQ 分支 |
| [mlaq_langmarl_v1](algorithms/mlaq_langmarl_v1/README.md) | 集中评估实际轨迹，按需给个别智能体建议 | MLAQ 基线 |
| [mlaq_langmarl_v2](algorithms/mlaq_langmarl_v2/README.md) | 集中评估给每位智能体一条个体反馈 | mlaq_langmarl_v1 |
| [vdn_mlaq_v1](algorithms/vdn_mlaq_v1/README.md) | 全局状态个体 Q 表，团队 TD，Q_tot=ΣQ_i | 3f5088a 的 VDN 分支 |
| [vdn_mlaq_v1_1](algorithms/vdn_mlaq_v1_1/README.md) | v1.1，将全局虚拟节点初始访问次数从 2 降为 1 | 由隔离后的 v1 派生 |
| [vdn_mlaq_v2](algorithms/vdn_mlaq_v2/README.md) | v1 + 局部虚拟动作、个体 UCB 选人 | 8c3dcd9 的 VDN 分支 |
| [vdn_mlaq_v3](algorithms/vdn_mlaq_v3/README.md) | v1 + 局部观测个体 Q 表，跨全局状态共享价值 | 由隔离后的 v1 派生，仅支持 sort_dialog |
| [vdn_mlaq_v4](algorithms/vdn_mlaq_v4/README.md) | v2 + 固定队友比较个体动作，按历史改善率分配探索 | 由隔离后的 v2 派生 |

```bash
python algorithms/mlaq/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v1/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v1_1/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v2/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v3/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v4/run_benchmark.py --compare_q
```

不用 `--algo`；启动脚本所在目录决定算法。可用 `--llm` 指定模型，`--max_opt` 限制任务难度，`--epoch <本版本名>/<批次>` 续跑。

## 目录约定

- `algorithms/<版本>/`：独立保存搜索、执行入口、依赖及环境。环境代码和资源统一放在各版本的 `environments/`。
- `environments/rocobench/` 包含 Sort、Sandwich 及机械臂相关代码。`environments/blocksworld/` 包含 BlocksWorld 和 PDDL 案例。这两个路径均相对于所属算法目录。
- `data/`：公共输出根目录，各算法/批次隔离；历史结果不移动。
- `tools/`：公共统计与辅助工具。Q 对照只在副本上估值，不参与实际选动作；API 配额工具由各版代码调用，修改这类运行工具也需核对所有版本。
- `docs/`、`figures/`、`没用的代码/`：公共说明、图片和历史参考，不作为算法导入来源。
- `.env`：项目根本地 API 配置，不复制、不提交。环境/场景资源各自独立，防止变体修改影响基线。

VDN 的主要算法版本使用 v1、v2、v3、v4 等编号，单项参数对照可使用 v1.1 这类小版本。每版 README 记录来源、机制、与来源版本的差异及已验证范围。正式实验记录具体提交。

MLAQ 保留本项目现有模型、证书/代理、API 适配及实验记录支持，并非作者仓库的逐字副本。详情见各算法 README。v1/v2 除局部探索外还存在历史边界处理差异，二者性能差不能未经控制就全部归因于局部 UCB。

研究方案、论文调研、诊断脚本与结果仍保存在 `D:/MyDocuments/Academy/8-研二/强化学习/MLAQ改进`。原项目说明保存在 `docs/README_拆分前.md`。
