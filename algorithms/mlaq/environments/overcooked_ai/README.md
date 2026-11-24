# Overcooked-AI 400 步 MLAQ 基线

上游使用 HumanCompatibleAI/overcooked_ai 的 `neurips2019` 标签，固定到提交
`6eaceb0a9a2501f1b9fccbf4c7016d6662ed1108`。经典 `Forced Coordination` 在这个标签里叫 `random0`。厨房状态、碰撞、拿放和烹饪规则都由上游代码执行。[上游版本](https://github.com/HumanCompatibleAI/overcooked_ai/tree/6eaceb0a9a2501f1b9fccbf4c7016d6662ed1108)

`OvercookedFirstSoup` 使用地图自带的固定初始状态。每次 `step()` 接收两名厨师同一网格步的动作。
MLAQ 入口关闭首汤终止，让送汤后的厨房状态接着进入下一次决策。`info["delivered"]` 标记本步是否送达，
`info["grid_steps"]` 记录当前绝对网格步。

## 复现

在工作树根目录运行以下 PowerShell 命令。示例使用已经安装的 Python 3.12，依赖装进独立的虚拟环境。

```powershell
$runtime = Join-Path $env:TEMP 'mlaq-overcooked-runtime'
$upstream = Join-Path $env:TEMP 'mlaq-overcooked-upstream'
python -m venv $runtime
git clone --depth 1 --branch neurips2019 https://github.com/HumanCompatibleAI/overcooked_ai.git $upstream
if ((git -C $upstream rev-parse HEAD) -ne '6eaceb0a9a2501f1b9fccbf4c7016d6662ed1108') {
    throw 'Unexpected Overcooked-AI source revision'
}
& "$runtime/Scripts/python.exe" -m pip install -r algorithms/mlaq/environments/overcooked_ai/requirements.txt
& "$runtime/Scripts/python.exe" -m pip install --no-deps -e $upstream
& "$runtime/Scripts/python.exe" -m algorithms.mlaq.environments.overcooked_ai.smoke_test
```

源码采用 editable 安装，以保留旧版安装包所需的地图资源。

## 从 MLAQ 入口运行

在现有 `mlaq` 的 Python 3.8 环境中，用 `python -m venv --system-site-packages <运行环境目录>`
创建独立环境，以复用 Torch 和 LLM API 依赖。用该环境安装上面的厨房依赖和上游源码。
下面的命令假定 `python` 指向这个环境，并在项目根目录执行。

```bash
nohup python -u algorithms/mlaq/run_MLAQ.py \
  --task overcooked_dialog --llm_source deepseek-v4-flash \
  --num_cases 1 --episodes 2 --grid_horizon 400 \
  > "mlaq_overcooked_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

默认读取项目根目录的 `.env`，实验结果写入项目根目录的 `data`。
默认跑固定初始状态的 `case_0`，执行 2 个 episode，每个 episode 的正常终点是第 400 个底层网格步。
`--imagination_rounds` 默认最多尝试两轮，找到完整路线后就执行。`--tsteps` 默认每 40 次高层决策重算一次想象阶段的路线价值。
结果位于 `data/overcooked_dialog/forced_coordination/mlaq/<批次>/`。旧的首汤批次应保留供历史分析，
400 步基线使用新批次和独立搜索图。

基线与 `mlaq_langmarl_v2` 的厨房版本使用相同的 UCB、候选动作校验、LLM 下一状态预测和累计奖励估值。
正式执行时才推进模拟器。需要补规划时，从真实状态接着想象到同一个第 400 步。搜索预算内找不到完整路线时，
记录 `plan_incomplete`。基线关闭回合后评估和建议驱动的探索，动作及预测校验仍会向 LLM 提供被拒绝的原因以重试。
`case_result.json` 中的 `critic_feedback` 为 `false`，每个 episode 同样记录奖励、送汤次数和送达网格步。

每个执行回合自动在 `interact/round_<编号>/replay_video.mp4` 保存厨房回放，和 `log.txt` 放在一起。
视频以 6 帧/秒播放，包含初始状态和实际执行的每个网格步，显示厨师、手持物品、锅内进度以及回合结果。
已建好的厨房运行环境需重新安装本目录的 `requirements.txt` 以补齐回放依赖。视频使用独立的渲染工具读取模拟器状态，
图像素材来自 [Overcooked-AI 官方仓库](https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/graphics)。

## 离线验证

手写路线由左侧厨师传递三个洋葱和一个盘子，右侧厨师煮汤并送餐。三个洋葱分别在第 7、14、21 步入锅，第 27 步将盘子放上共享台面，第 40 步煮好，第 44 步送达，团队稀疏奖励为 20。重置后重复同一路线，结果一致。单独用一步上限验证了超时会结束回合，送达标志为假。

手写路线验证的是适配器的首汤模式。上面的 MLAQ 实验入口使用全时长模式。
