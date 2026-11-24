# Multi-Agent Act-only

Overcooked-AI 的直接交互基线，使用和 MLAQ 对照组相同的 Forced Coordination 地图、固定初始状态和底层控制器。
每次决策时，RightChef 和 LeftChef 看同一份真实厨房状态。RightChef 先选动作，LeftChef 再根据厨房状态和
RightChef 本次选择的动作决定自己的动作。控制器同时执行两人的命令，把新的真实状态交给下一次决策。

每位厨师每次只请求一次模型回复，输出一条 USE 命令或 WAIT。格式错误或目标不可达时，对应厨师本次执行 WAIT，
另一位厨师的有效动作照常执行。送汤后继续，默认到第 400 个底层网格步结束。每个 episode 重新从初始状态开始。

动作选择依靠固定 LLM 和当前观察。程序不保留推理历史，也不使用搜索图、世界模型或回合评估。

## 运行

在项目根目录使用已配置好 Overcooked-AI 的 Python 环境，在 Git Bash 执行。

~~~bash
nohup python -u algorithms/act_only/run_act_only.py \
  --llm_source deepseek-v4-flash --episodes 2 --grid_horizon 400 \
  > "act_only_overcooked_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
~~~

API 配置默认读取项目根目录的 .env。依赖与现有厨房实验相同。
[Overcooked-AI 环境配置](../mlaq/environments/overcooked_ai/README.md)

结果写入 data/overcooked_dialog/forced_coordination/act_only/<批次>/run_0/case_0/。
case_result.json 记录各回合的奖励、送汤时刻、无效动作数和模型用量。
interact/round_x 保存执行日志、视频回放和每次决策两名厨师的提示词及原始回复。
