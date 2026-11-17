# MLAQ + LangMARL 个体反馈 v1

从本项目的原始 MLAQ 基础版派生，支持 Sort 的三智能体对话。每条联合动作继续由原 MLAQ 搜索图、全局 UCB、候选检查和价值迭代处理。集中评估和动作提案使用同一个已配置的 LLM，分开调用，模型权重固定。

1. 环境交互记录每个联合动作前后的实际状态、奖励和执行状态。
2. 有完整执行步骤的回合以当前 `interact/round_<回合号>/log.txt` 的实际执行记录为评估输入。提示词补充简短的动作规则与回复格式，评估 LLM 可以指出哪些动作能并行、哪些交接可以提前，并针对具体状态和智能体给出建议。
3. 同一 case 后续搜索触发候选动作生成时，按任务目标、动作前状态和智能体查找建议。每次对话调用最多插入该智能体的一条建议。
4. 日志记录哪版建议出现在新联合动作的生成过程中。如果该动作随后实际执行，下一次评估会收到对应的版本号和实际结果。评估器判断旧建议是否被采纳，再提出保留、修订或撤回。旧版本及修改依据保存在日志中。

建议只在同一个 case 内使用。批量续跑从未完成的 case 开始重新收集建议，原日志按执行会话保留。连续复用旧路线时可能不会调用动作 LLM。每个 case 最多记录三次环境交互回合，逆运动学失败重试也计入上限。

## 运行

在项目根目录、激活 mlaq 环境后运行一个 case。

    python -u algorithms/mlaq_langmarl_v1/run_MLAQ.py --task sort_dialog --llm_source deepseek-v4-flash --optimal_steps 2 --tsteps 8 --num_cases 1 --skip_display

批量运行。

    python -u algorithms/mlaq_langmarl_v1/run_benchmark.py --llm deepseek-v4-flash

输出到项目根目录的 data/sort_dialog/mlaq_langmarl_v1/<批次>/。续跑时传 --epoch mlaq_langmarl_v1/<批次>。

每个 case 的 trajectory_feedback.jsonl 记录实际轨迹、拒绝的动作、建议的版本历史、提示词使用以及关联的新联合边。case_result.json 中的 trajectory_feedback 记录使用次数、入图与执行次数，并分别统计规划和评估的请求及 token。API 未返回 usage 时，token 按文本估算。建议随候选生成时进入了提示词，是否按建议行动由评估器根据实际轨迹判断。

评估 LLM 复用动作生成的流式调用，使用模型默认的生成预算。完整提示词和原始回复正文保存在最后一轮想象对应的 `simulate/round_<轮次>/critic_query/trial_<序号>/main_prompt.txt` 和 `main_response.txt`。同一轮的评估调用按 `trial_1`、`trial_2` 依次保存，空建议列表也照常保存。`trajectory_feedback.jsonl` 记录目录、API 结束原因、正文字符数和 API 提供的推理 token 数。

[参考的 LangMARL 集中评估器](https://github.com/DaRL-GenAI/LangMARL/blob/28b8655b6e89c8c27eb5d3cdc212c3297465a696/langmarl/core/critic.py#L104)

