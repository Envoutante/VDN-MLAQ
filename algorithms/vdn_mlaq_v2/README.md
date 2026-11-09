# VDN-MLAQ v2：个体 UCB 探索

## 算法定义

在 v1 的价值分解基础上，维护个体动作与虚拟动作的 Q/访问次数。全局选中虚拟入口后，用局部 UCB 比较基准个体动作与虚拟动作；固定未被选中的队友，只让参与者对话。虚拟 Q 初值 -0.1；沿用当前 Q/n + 2√(ln N/n) 公式，本次不修改公式。还保留 8c3dcd9 中的循环保护、重试上限、终止状态处理和加载修正。

## 代码来源

- 搜索与图实现来自 Git `8c3dcd9`，算法分支已固定展开，不是重新编写算法。
- MLAQ 的原始参考为 `d966ea7`；`3f5088a` 是本项目已适配、已用于隔离对照的基线。保留现有文本行尾空白规范化、API 请求适配、usage 统计、断点续跑完成标记、Q 对照日志等历史改动，不能声称整个项目与作者仓库逐字相同。
- 环境、解析器、运行器、模型列表、证书/代理适配取拆分前 `8c3dcd9`；两个来源间环境及运行器没有差异。
- 本次新增的是目录/路径适配、版本固定、跨版本 epoch 校验；没有修改奖励、折扣、TD 步长、UCB 或提示词内容。
- `algo` 内部字段是旧数据/记录器兼容标记，启动不接受 `--algo`。VDN 的旧标记仍为 `vdn-mlaq`，输出身份使用目录名 `vdn_mlaq_v2`。

## 运行

在项目根目录、已激活 mlaq 环境后：

```bash
python algorithms/vdn_mlaq_v2/run_benchmark.py --llm deepseek-v4-flash --compare_q
```

单独运行：

```bash
python algorithms/vdn_mlaq_v2/run_MLAQ.py --task sort_dialog --optimal_steps 2 --skip_display
```

输出到项目根目录 `data/<task>/vdn_mlaq_v2/<批次>/`；不写进算法目录。密钥读取项目根 `.env`。启动时切换到本算法目录，以保证原有场景和任务文件的相对路径有效。

续跑新批次：`--epoch vdn_mlaq_v2/YYYYMMDD_HHMM`。只允许本版本前缀，防止误覆盖其他算法。历史 `vdn-mlaq/…` 目录不自动改名或续写；复用历史数据需明确复制到新批次后再操作，不要只按旧目录名推断算法版本。

三个版本保留 `prompting.lang_tree` 类路径，各自进程可读取旧 pth。不要在同一 Python 进程中混合导入不同版本；离线工具应将选定算法目录放到 sys.path 首位。v2 加载旧图按已有实现重建 Q/个体计数，不代表精确恢复中断时的全部搜索状态。

## 支持范围与后续修改

VDN 只支持 sort_dialog / sandwich_dialog；中央提案及 BlocksWorld 尚未接入 VDN，入口会拒绝以 VDN 名义运行它们。
已做的离线验证不等于真实 LLM 效果实验。未来改变价值学习或探索规则时，从完整目录复制为下一个编号 v3、v4，并更新本文、runtime_layout.py 的 ALGORITHM 及默认临时保存目录。不要命名为 latest/final，也不要在已发表结果对应版本中悄悄更改算法。
