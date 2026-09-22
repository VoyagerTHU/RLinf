# Solver Kitchen × RLinf 集成计划

目标：让 RLinf 以 `../solver`（Newton + MuJoCo-Warp 的 GPU 批量厨房仿真）作为渲染仿真环境，
复现之前 RoboCasa GR1 × StarVLA 的 RL 训练形态：图像观测、分组 seed 的 GRPO、
4 个 EnvWorker × 8 环境、720 步 episode、16 步动作 chunk、固定 seed 评测。

生成信息：2026-09-21，机器 `ja100`（8× A100 80GB）。RLinf 分支 `robocasa-gr1-grpo-fix`，
solver 位于 `/home/A100/solver`（实际路径 `/data/dengyixuan/wyz/robots/solver`）。

---

## 0. 已确认的事实与约束

| 项目 | RLinf 训练环境 | solver |
|---|---|---|
| 解释器 | `/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python`，Python 3.10.18 | 要求 Python ≥ 3.11（uv 已装 3.11.14） |
| torch | 2.6.0+cu124 | 锁定 2.12.0 |
| 其它 | ray 2.49.2、flash-attn、StarVLA | newton 1.4.0、warp-lang 1.16.0、mujoco 3.10、定制 mujoco-warp 分支 |

- 两套依赖不能装进同一个 venv。RLinf 的 `python_interpreter_path` 是**按节点**生效的，
  单机上 actor/rollout/env 共用一个解释器，所以不能靠它切换 env worker 的解释器。
- 结论：solver 必须在**独立子进程**中运行（用 solver 自己的 venv），RLinf 的 EnvWorker
  通过进程间通道下发动作、回收观测。这和之前 RoboCasa GR1 用 MuJoCo 子进程的形态一致。
- solver 侧现成可用的栈：`KitchenVectorEnv → KitchenReachTask → KitchenTorchEnv`
  （16 关节双臂 PiPER，reach 任务，masked reset，CUDA graph），
  `TiledCameraViewer` 可批量出 RGB（需 CUDA）。注意 `KitchenVectorEnv.step` 只调
  `sim.step(viewer)`，不调 `render`，所以相机 mount 同步需要适配器自己调用
  `viewer.sync_camera_mounts(sim.state_0)` 再 `capture()`。
- 之前的 GR1 训练契约（适配器必须满足）：
  - 构造签名 `Env(cfg, num_envs, seed_offset, total_num_processes, worker_info)`；
  - `reset() -> (obs, info)`；`chunk_step(actions[N, chunk, D]) -> (obs_list, rewards[N, chunk], terminations, truncations, infos_list)`；
  - obs 为 dict：`main_images` uint8 `[N, H, W, 3]`（CPU tensor）、`task_descriptions` list[str]；
    本环境额外提供 `states` float `[N, obs_dim]` 给 MLP/CNN 策略；
  - `infos["episode"]` 含 `success_once / return / episode_len / reward / sample_seed / sample_group / sample_trajectory / metric_valid`；
  - `update_reset_state_ids()`、`set_seed_selection_round()`、`capture_image()`、`close()`；
  - `auto_reset: false`，episode 长度由 `max_episode_steps` 截断，rollout 结束后统一 reset；
  - 同一 seed 组（`group_size` 个环境）必须从**相同初始状态**开始。
- StarVLA 侧：`predict_action_batch` 只需要 `main_images` 与 `task_descriptions`，
  动作反归一化依赖 checkpoint 的 `norm_stats[unnorm_key]`，且非 gr1 平台会走
  LIBERO 式夹爪映射。solver 的动作是 16 维连续关节目标，需要新的 `policy_setup`。
- GPU 4–7 正被其他训练占用（约 44 GB / 60–70% util）；GPU 3 空闲，0–2 轻负载。
  本计划的所有测试使用 **GPU 3**（`CUDA_VISIBLE_DEVICES=3`）。

---

## 1. 交付物

### 1.1 solver 侧运行时（新建，位于 RLinf 仓库内，独立于 rlinf 包的依赖）

`rlinf/envs/solver_kitchen/solver_core.py`
- 只 import `solver`、`newton`、`warp`、`numpy`、solver venv 里的 torch；**不 import `rlinf`**，
  以便在 solver venv 中作为独立进程运行。
- `SolverKitchenCore(config)`：
  - 构建 `KitchenVectorEnv(world_count=num_envs, device="cuda:0", ...)`，
    挂 `TiledCameraViewer(num_envs, env_origins=0)` 并按配置注册相机
    （默认一个 224×224 的 `main` 相机，可选 wrist/extra 相机）；
  - 构建 `KitchenReachTask` + `KitchenTorchEnv(auto_reset=False, normalized_actions=True)`；
  - `reset(env_ids, seeds)`：masked reset；按 seed 覆盖每个 world 的目标偏移，使同组
    world 初始状态与目标完全一致；重新计算观测；
  - `step(actions[N, 16])`：一步物理，返回 `states / reward / success / failure / terminated / truncated`；
  - `render() -> uint8 [N, C, H, W, 3]`：同步相机 mount 后 `capture(outputs=("rgb",))`；
  - `metadata`：`action_dim=16`、`obs_dim`、关节名、关节限位、相机名与分辨率。

`rlinf/envs/solver_kitchen/bridge_server.py`
- 用 solver venv 运行的独立进程：`python bridge_server.py --address ... --authkey ...`。
- 基于 `multiprocessing.connection`（pickle + numpy），命令：`init / reset / step / render / metadata / close`。
- 任何异常都序列化回父进程并让父进程抛出，避免静默挂死。

### 1.2 RLinf 侧环境（新建）

`rlinf/envs/solver_kitchen/backend.py`
- `SolverBackend` 协议 + `InProcessBackend`（解释器已装 solver 时直接调用 core，
  用于 solver venv 内的集成测试）+ `SubprocessBackend`（默认，启动 bridge_server）。

`rlinf/envs/solver_kitchen/solver_kitchen_env.py`
- `SolverKitchenEnv`：满足第 0 节列出的 GR1 契约；
- 分组 seed：复用 `rlinf/envs/robocasa_gr1/seed_pool.py` 的
  `select_process_seed_groups`（seed 池来自 `seed_manifest` JSON 或 `seed_pool_size` 生成的范围），
  `group_size` 个环境共享一个 seed，评测模式下按轮次顺序遍历并用 `metric_valid` 屏蔽 padding；
- 渲染策略：默认每个 chunk 只在最后一步渲染（策略只消费最后观测），
  `render_every_step: true` 时每步渲染（用于视频）；
- `states` 与图像一起返回，MLP/CNN 策略直接用；
- `infos["episode"]` 指标与 GR1 对齐，另加 `distance`（末端到目标距离）方便诊断。

注册：
- `rlinf/envs/__init__.py`：`SupportedEnvType.SOLVER_KITCHEN = "solver_kitchen"` 与 `get_env_cls` 分支；
- `rlinf/envs/action_utils.py`：`prepare_actions` 直通分支；
- `rlinf/config.py`：`validate_cfg` 增加 solver_kitchen 的分组校验（`total_num_envs` 能被
  `group_size` 整除；GRPO 时 env 与 actor world size 相等，沿用 GR1 的检查）。

### 1.3 StarVLA 适配（小改）

- `rlinf/models/embodiment/starvla/utils/action_space.py`：新增 `policy_setup="solver_kitchen"`，
  走与 gr1 相同的"全通道连续 min-max"路径，不做夹爪映射；
- `resolve_action_norm_stats` 支持配置内联的 `action_norm_stats`（`q01/q99/mask`）覆盖，
  因为 checkpoint 没有该机器人的统计；环境接收 `[-1, 1]` 归一化动作时用恒等统计。

### 1.4 配置与脚本

- `examples/embodiment/config/env/solver_kitchen.yaml`：env 基础配置（相机、任务、seed、视频）；
- `examples/embodiment/config/solver_kitchen_ppo_mlp.yaml`：状态观测 PPO，单卡，验证训练闭环；
- `examples/embodiment/config/solver_kitchen_ppo_cnn.yaml`：图像观测 PPO，验证渲染链路能训练；
- `examples/embodiment/config/solver_kitchen_grpo_starvla.yaml`：镜像
  `robocasa_gr1_cup_drawer_grpo_starvla.yaml`（group_size 8、rollout_epoch 4、720 步、
  chunk 16、token-level、参考 KL 自适应），动作维 16，`policy_setup: solver_kitchen`；
- `examples/embodiment/setup_solver_kitchen_venv.sh`：在 `../solver` 用 uv + Python 3.11 建 venv；
- `examples/embodiment/run_solver_kitchen_grpo.sh` / `run_solver_kitchen_ppo_mlp.sh`：
  设置 `SOLVER_ROOT`、`SOLVER_PYTHON`、`REPO_PATH` 等并启动；
- `requirements/install.sh`：`SUPPORTED_ENVS` 增加 `solver_kitchen`，调用上面的 venv 脚本。

### 1.5 文档与记忆

- 本计划文件随进度更新"执行记录"一节；
- 完成后把非显而易见的坑写入 Claude 记忆。

---

## 2. 测试方法

### 2.1 单元测试（CPU，RLinf venv，`tests/unit_tests/envs/`）

| 文件 | 覆盖 |
|---|---|
| `test_solver_kitchen_env.py` | 用 `FakeBackend` 驱动 `SolverKitchenEnv`：obs 键/形状/dtype；`chunk_step` 堆叠；`max_episode_steps` 截断；`success` 终止与 `success_once` 累计；同组环境 seed 相同、跨组不同；评测 padding 的 `metric_valid`；`update_reset_state_ids` 推进轮次；`render_every_step` 开关的渲染次数；`capture_image` |
| `test_solver_kitchen_bridge.py` | 用当前解释器启动 `bridge_server` 并注入 `FakeCore`，验证协议往返、异常透传、`close` 后进程退出 |
| `test_solver_kitchen_registry.py` | `SupportedEnvType`、`get_env_cls`、`prepare_actions` 直通、`validate_cfg` 对分组/placement 的拒绝 |
| `test_starvla_solver_kitchen_actions.py` | `policy_setup=solver_kitchen` 的反归一化为纯 min-max、无夹爪映射；内联 norm stats 覆盖 |

运行：`cd tests/unit_tests && $TRAIN_PYTHON -m pytest envs/test_solver_kitchen_*.py -q`

### 2.2 solver 侧集成测试（GPU 3，solver venv，`tests/integration_tests/solver_kitchen/`）

1. `test_core_smoke.py`：8 worlds，reset → 渲染 → step 16 步 → 渲染；断言图像非全黑/非常数、
   states 有限；保存一张 PNG 到 scratchpad 供人工确认相机位置。
2. `test_core_determinism.py`：同一 seed 两次 reset 的 states 与图像逐 bit 一致；
   同组 8 个 world 的初始 states 一致、跨组不同。
3. `test_core_throughput.py`（记录用）：32 worlds，224×224，测 100 步的 env-step/s，
   分别统计"仅物理"与"物理 + 渲染"，写入本文档。

### 2.3 跨解释器桥接测试（GPU 3）

- 父进程 = RLinf venv（py3.10/torch2.6），子进程 = solver venv（py3.11/torch2.12）：
  `SolverKitchenEnv` 走 `SubprocessBackend`，4 环境，reset + 3 个 chunk_step + close；
  断言 obs 契约、子进程正常退出、无 zombie。

### 2.4 端到端训练（GPU 3）

1. **PPO-MLP**：`solver_kitchen_ppo_mlp.yaml`，32 环境、300 步、跑 10 个 epoch。
   验收：训练不崩，`metrics/training_step_metrics.jsonl` 中 `env/success_once` 或
   `env/return` 随步数上升（solver 文档中该任务 PPO 可在数十次迭代内到 100%）。
2. **PPO-CNN**：`solver_kitchen_ppo_cnn.yaml`，同规模，跑 5 个 epoch。
   验收：渲染链路在训练循环内稳定，吞吐可接受，视频文件生成且内容正确。
3. **GRPO-StarVLA smoke**：`solver_kitchen_grpo_starvla.yaml`，先 `runner.max_steps: 1`。
   验收：模型加载、rollout、组内优势归一化、actor 更新、权重同步全部跑通。
   已知风险：现有 OFT checkpoint 的动作头是 29 维，改为 16 维需要重初始化动作头，
   意味着策略从随机开始。管线正确性可以验证，但"和之前一样的成功率提升"需要先有
   在 solver 厨房数据上 SFT 过的 StarVLA checkpoint，这一点不在本计划内，执行时会明确报告。

---

## 3. 执行顺序

1. 建 solver venv（后台 `uv sync`），同时写 `solver_core.py`、`bridge_server.py`。
2. 在 solver venv 里跑 2.2 的 smoke，用 PNG 校准相机位姿。
3. 写 `SolverKitchenEnv` + backend + 注册 + 单元测试（2.1），在 RLinf venv 跑通。
4. 跨解释器桥接测试（2.3）。
5. 配置 + 脚本；跑 PPO-MLP（2.4.1）。
6. StarVLA 适配 + 单测；跑 PPO-CNN 与 GRPO smoke（2.4.2、2.4.3）。
7. install.sh、文档、记忆；更新本文档的执行记录。

---

## 4. 执行记录

（执行过程中填写：完成项、测试输出摘要、吞吐数据、遗留问题。）
