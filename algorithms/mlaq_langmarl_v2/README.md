# MLAQ + LangMARL 个体反馈 v2

从 `mlaq_langmarl_v1` 派生，支持 Sort 的三智能体对话。v1 允许评估器只给部分智能体建议，v2 要求每次有效评估分别给 Alice、Bob、Chad 一条个体反馈。联合动作继续由原 MLAQ 搜索图、全局 UCB、候选检查和价值迭代处理。模型权重固定。

1. 环境交互记录每个联合动作前后的实际状态、奖励和执行状态。
2. 评估 LLM 阅读当前回合的实际执行日志。对做得好的智能体，说清楚应保持哪项动作及其理由。涉及同一步协作时，分别写清参与者该做什么。同一份回复必须恰好包含每位智能体一条有效反馈，缺人、重复或字段无效时整轮反馈作废。
3. 同一 case 后续搜索触发候选动作生成时，按任务目标、动作前状态和智能体查找建议。每次对话调用最多插入该智能体的一条建议。
4. 日志记录哪版建议出现在新联合动作的生成过程中。评估器可以保留已有建议，也可以在有证据表明旧建议被遵循后，用新建议替换它。保留时沿用原建议和版本，修订时生成新版本。旧版本及理由保存在日志中。

建议只在同一个 case 内使用。批量续跑从未完成的 case 开始重新收集建议，原日志按执行会话保留。连续复用旧路线时可能不会调用动作 LLM。每个 case 最多记录三次环境交互回合，逆运动学失败重试也计入上限。

## 运行

在项目根目录、激活 mlaq 环境后运行一个 case。

    python -u algorithms/mlaq_langmarl_v2/run_MLAQ.py --task sort_dialog --llm_source deepseek-v4-flash --optimal_steps 2 --tsteps 8 --num_cases 1 --skip_display

批量运行。

    python -u algorithms/mlaq_langmarl_v2/run_benchmark.py --llm deepseek-v4-flash

输出到项目根目录的 `data/sort_dialog/mlaq_langmarl_v2/<批次>/`。续跑时传 `--epoch mlaq_langmarl_v2/<批次>`。
批量实验每个难度最多跑 30 个 case（编号 0–29）。续跑只补原计划内剩余的 case。

每个 case 的 trajectory_feedback.jsonl 记录实际轨迹、拒绝的动作、建议的版本历史、提示词使用以及关联的新联合边。case_result.json 中的 trajectory_feedback 记录使用次数、入图与执行次数，并分别统计规划和评估的请求及 token。API 未返回 usage 时，token 按文本估算。建议随候选生成时进入了提示词，是否按建议行动由评估器根据实际轨迹判断。

评估 LLM 复用动作生成的流式调用，使用模型默认的生成预算。完整提示词和原始回复正文保存在最后一轮想象对应的 `simulate/round_<轮次>/critic_query/trial_<序号>/main_prompt.txt` 和 `main_response.txt`。同一轮的评估调用按 `trial_1`、`trial_2` 依次保存，无效回复也按原样保存。`trajectory_feedback.jsonl` 记录目录、API 结束原因、正文字符数和 API 提供的推理 token 数。

[参考的 LangMARL 集中评估器](https://github.com/DaRL-GenAI/LangMARL/blob/28b8655b6e89c8c27eb5d3cdc212c3297465a696/langmarl/core/critic.py#L104)

