# 经典 Overcooked-AI 首单送达

上游使用 HumanCompatibleAI/overcooked_ai 的 `neurips2019` 标签，固定到提交
`6eaceb0a9a2501f1b9fccbf4c7016d6662ed1108`。经典 `Forced Coordination` 在这个标签里叫 `random0`。厨房状态、碰撞、拿放和烹饪规则都由上游代码执行。[上游版本](https://github.com/HumanCompatibleAI/overcooked_ai/tree/6eaceb0a9a2501f1b9fccbf4c7016d6662ed1108)

`OvercookedFirstSoup` 使用地图自带的固定初始状态。每次 `step()` 接收两名厨师同一网格步的动作。首次获得送餐奖励时回合结束，同时保留上游的回合步数上限。返回的 `info["delivered"]` 区分送达与超时，`info["grid_steps"]` 记录真实网格步数。

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
python -u algorithms/mlaq/run_MLAQ.py --task overcooked_dialog --llm_source deepseek-v4-flash --num_cases 1
```

默认读取项目根目录的 `.env`，实验结果写入项目根目录的 `data`。
默认跑固定初始状态的 `case_0`。可用 `--episodes`、`--imagination_rounds`、`--tsteps`、
`--grid_horizon` 分别设置执行回合数、每回合的想象次数、高层决策上限和每回合网格步数上限。
默认值依次为 2、2、40、400。结果位于 `data/overcooked_dialog/forced_coordination/mlaq/<批次>/`。
地图在目录中位于算法名之前，批次内独立保存搜索图和地图配置。

每次高层决策由 RightChef 和 LeftChef 各选一个 `USE x,y` 或 `WAIT`。控制器在经典地图上
把两个动作并行展开为网格操作。搜索用 MLAQ 的虚拟动作和 UCB，在环境副本上推演候选；
正式执行使用真实模拟器。搜索图以首单送达为目标，并沿用每个高层决策一次的折扣。
每轮记录实际网格步数、高层决策数、送达结果和 API 返回的 token 用量。

每个执行回合自动在 `interact/round_<编号>/replay_video.mp4` 保存厨房回放，和 `log.txt` 放在一起。
视频以 6 帧/秒播放，包含初始状态和实际执行的每个网格步，显示厨师、手持物品、锅内进度以及回合结果。
已建好的厨房运行环境需重新安装本目录的 `requirements.txt` 以补齐回放依赖。视频使用独立的渲染工具读取模拟器状态，
图像素材来自 [Overcooked-AI 官方仓库](https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/graphics)。

## 离线验证

手写路线由左侧厨师传递三个洋葱和一个盘子，右侧厨师煮汤并送餐。三个洋葱分别在第 7、14、21 步入锅，第 27 步将盘子放上共享台面，第 40 步煮好，第 44 步送达，团队稀疏奖励为 20。重置后重复同一路线，结果一致。单独用一步上限验证了超时会结束回合，送达标志为假。

用固定模型回复通过 `run_MLAQ.py` 完成了完整搜索和执行。该路线经过 26 次高层决策、43 个网格步，送达奖励为 20。
