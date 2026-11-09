# MLAQ 算法实验项目

算法代码已按完整副本隔离。先激活原 `mlaq` Python 环境，在本目录运行。

| 目录 | 定义 | 来源 |
|---|---|---|
| [mlaq](algorithms/mlaq/README.md) | MLAQ 基线，原有价值迭代与联合探索 | 3f5088a 的 MLAQ 分支 |
| [vdn_mlaq_v1](algorithms/vdn_mlaq_v1/README.md) | 全局状态个体 Q 表，团队 TD，Q_tot=ΣQ_i | 3f5088a 的 VDN 分支 |
| [vdn_mlaq_v2](algorithms/vdn_mlaq_v2/README.md) | v1 + 局部虚拟动作、个体 UCB 选人 | 8c3dcd9 的 VDN 分支 |

```bash
python algorithms/mlaq/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v1/run_benchmark.py --compare_q
python algorithms/vdn_mlaq_v2/run_benchmark.py --compare_q
```

不用 `--algo`；启动脚本所在目录决定算法。可用 `--llm` 指定模型，`--max_opt` 限制任务难度，`--epoch <本版本名>/<批次>` 续跑。

## 目录约定

- `algorithms/<版本>/`：独立的 prompting、runner、rocobench、other_envs、入口、依赖声明与说明。
- `data/`：公共输出根目录，各算法/批次隔离；历史结果不移动。
- `tools/`：公共统计与辅助工具。Q 对照只在副本上估值，不参与实际选动作；API 配额工具仍由三套代码调用，修改这类运行工具也需核对所有版本。
- `docs/`、`figures/`、`没用的代码/`：公共说明、图片和历史参考，不作为算法导入来源。
- `.env`：项目根本地 API 配置，不复制、不提交。环境/场景资源各自独立，防止变体修改影响基线。

VDN 编号按算法变化递增为 v1、v2、v3；每版 README 必须写来源、机制、与上一版差异、已验证范围。小修复用 Git 记录；正式实验记录具体提交，不能只记录版本目录名。

MLAQ 保留本项目现有模型、证书/代理、API 适配及实验记录支持，并非作者仓库的逐字副本。详情见各算法 README。v1/v2 除局部探索外还存在历史边界处理差异，二者性能差不能未经控制就全部归因于局部 UCB。

研究方案、论文调研、诊断脚本与结果仍保存在 `D:/MyDocuments/Academy/8-研二/强化学习/MLAQ改进`。原项目说明保存在 `docs/README_拆分前.md`。
