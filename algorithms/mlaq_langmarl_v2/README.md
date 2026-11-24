# MLAQ + LangMARL 个体反馈 v2

从 `mlaq_langmarl_v1` 派生，支持 Sort 三智能体对话及 Overcooked-AI 双厨师对话。v1 允许评估器只给部分智能体建议，v2 要求每次有效评估给每位智能体一条反馈。模型权重固定。

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
批量实验按 Sort 环境的数据集计数，每个难度最多选前 30 个 case。续跑检查所有计划内的编号，
跳过 `case_result.json` 已标记 `finished: true` 的 case，并补跑缺失或中断的 case。

每个 case 的 trajectory_feedback.jsonl 记录实际轨迹、拒绝的动作、建议的版本历史、提示词使用以及关联的新联合边。case_result.json 中的 trajectory_feedback 记录使用次数、入图与执行次数，并分别统计规划和评估的请求及 token。API 未返回 usage 时，token 按文本估算。建议随候选生成时进入了提示词，是否按建议行动由评估器根据实际轨迹判断。

评估 LLM 复用动作生成的流式调用，使用模型默认的生成预算。完整提示词和原始回复正文保存在最后一轮想象对应的 `simulate/round_<轮次>/critic_query/trial_<序号>/main_prompt.txt` 和 `main_response.txt`。同一轮的评估调用按 `trial_1`、`trial_2` 依次保存，无效回复也按原样保存。`trajectory_feedback.jsonl` 记录目录、API 结束原因、正文字符数和 API 提供的推理 token 数。

[参考的 LangMARL 集中评估器](https://github.com/DaRL-GenAI/LangMARL/blob/28b8655b6e89c8c27eb5d3cdc212c3297465a696/langmarl/core/critic.py#L104)

## Overcooked-AI

使用与 [MLAQ 厨房版本](../mlaq/environments/overcooked_ai/README.md) 相同的地图、初始状态和底层动作控制方式。

```bash
python -u algorithms/mlaq_langmarl_v2/run_MLAQ.py --task overcooked_dialog --llm_source deepseek-v4-flash --num_cases 1
```

评估器在每次实际执行后读取厨房日志，分别给 RightChef 和 LeftChef 一条建议。建议用执行前的状态定位，
并在搜索再次扩展该状态时进入各自提示词。日志记录建议版本、生成的联合候选和实际执行的版本。
再次遇到有待尝试建议的状态时，搜索优先生成一次联合候选。原动作保留供正常估值。
同一组建议生成合法候选后即完成尝试，失败时最多补试一次。
首次在正式执行中遇到这类状态时，会从该状态补一次想象。`case_result.json` 的 `feedback_search`
统计触发位置、候选生成结果和额外执行中搜索的模型调用次数。
结果位于 `data/overcooked_dialog/forced_coordination/mlaq_langmarl_v2/<批次>/`。
实际执行从 `interact/round_1` 编号，同一个 episode 中的补规划不会另开执行目录。
厨房视频与执行日志放在同一目录。评估器读取真实执行日志，提示词与回复保存在该 episode 的
`interact/round_<编号>/critic_query/trial_<序号>/`。评估后的有效建议保存在当前 case 的内存中，
下一次想象遇到匹配的状态时，分别写进两名厨师各自的动作提示词。

想象和实际执行都以 `--grid_horizon`（默认 400）个底层网格步为终点。每份汤的奖励记在对应转移上，
下一次决策使用送达后的厨房状态和绝对网格步。搜索按剩余时间内的预测累计奖励比较路线。
`--tsteps` 设置每个搜索批次的高层决策数，同一轮想象可连续进行多个批次。
找到完整预测路线后立即执行。`--imagination_rounds` 设定每次规划的尝试上限，只有续路还未补齐才重试。
执行中的预测偏差导致续路缺失时，从真实状态重新搜索到同一终点，并使用下一个想象轮号。
搜索预算用尽且仍找不到完整路线时，以 `plan_incomplete` 记录当时的进度。

新候选先通过可达性检查及 LLM 动作检查，再由 LLM 预测并检查下一状态，通过后写入搜索图。
想象沿预测状态展开。模拟器在实际执行阶段推进环境，并用真实后继状态修正搜索图。
动作检查失败时，把原因发给两名厨师，在原状态重提。下一状态检查失败时，在同一动作上重新预测。
两个环节各最多尝试三次。提示词、回复和拒绝原因保存在对应 `timestep_<编号>` 的角色和 `trial_<编号>` 目录中。
永久图保存通过检查的预测转移和实际执行后的修正。每个 episode 用永久图初始化临时搜索图，
虚拟动作和 UCB 访问次数留在临时图中。
`case_result.json` 额外记录每轮的送汤数量、送达步数，以及首次执行前是否已有完整预测路线。

