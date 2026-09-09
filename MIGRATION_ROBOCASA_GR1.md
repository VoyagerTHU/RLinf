# RoboCasa GR1 × StarVLA RL 环境迁移手册

本文件描述如何在一台**新的 Linux + NVIDIA 机器**上，从零复现 `robocasa-gr1-grpo-fix` 分支的
GRPO / PPO / SAC 训练环境，并用 `ja100` 上跑通的数据做对照验证。写给人和 Claude Code 看：
所有命令都可以按顺序直接执行，路径全部通过第 1 节的变量给出。

生成信息：2026-09-09，源机器 `ja100`（Ubuntu 22.04.5，NVIDIA 570.195.03，8× A100 80GB），
仓库 `VoyagerTHU/RLinf` 分支 `robocasa-gr1-grpo-fix`。

---

## 0. 需要带走的东西一览

| 类别 | 内容 | 大小 | 本仓库里是否已有 |
|---|---|---|---|
| 代码 | RLinf（本仓库，分支 `robocasa-gr1-grpo-fix`；对照用 `robocasa-gr1-grpo-baseline`） | — | 是 |
| 代码 | starVLA `dfa2189` + 本地补丁 | — | 补丁在 `examples/embodiment/migration/patches/starvla/` |
| 代码 | robocasa-gr1-tabletop-tasks `4840e67` + 本地补丁 | — | 补丁在 `examples/embodiment/migration/patches/robocasa-gr1-tabletop-tasks/` |
| 环境 | Python 3.10 + torch 2.6.0/cu124 + flash-attn 2.8.3 + 189 个锁定包 | — | `examples/embodiment/migration/requirements-gr1.lock` |
| 模型 | StarVLA OFT checkpoint `Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt` | 9.8 GB | 否，需拷贝或从 HF 下载 |
| 模型 | 基座 `Qwen/Qwen3-VL-4B-Instruct` | 17 GB | 否 |
| 数据 | RoboCasa GR1 场景/物体资产（`robocasa/models/assets/`） | 8.2 GB | 否，脚本下载 |
| 数据 | 训练/评测 seed 清单、NVIDIA EGL manifest | — | 是（`examples/embodiment/config/seeds/`、`examples/embodiment/runtime_site/`） |
| 可选 | 已训练 checkpoint（续训/评测） | 19 GB/个 | 否 |

`ja100` 上的源路径（rsync 用）：
```
/data/dengyixuan/wyz/models/StarVLA/Qwen3-VL-OFT-Robocasa/          # OFT checkpoint
/data/dengyixuan/wyz/models/Qwen/Qwen3-VL-4B-Instruct/               # 基座 VLM
/data/dengyixuan/wyz/robots/robocasa-gr1-tabletop-tasks/robocasa/models/assets/   # RoboCasa 资产（也可脚本下载）
/data/dengyixuan/wyz/experiments/rlinf_starvla_robocasa_gr1_grpo_groupfix_20260908/run/cup_to_drawer_close_grpo_groupfix_tokenlevel_lr1e6/checkpoints/global_step_*/actor/model_state_dict/full_weights.pt
```

---

## 1. 路径变量（先改这里）

```bash
export WORK=/data/$USER/gr1            # 一切放在这个目录下
export MODELS=$WORK/models
export RLINF_ROOT=$WORK/RLinf
export STARVLA_ROOT=$WORK/starVLA
export ROBOCASA_TASK_ROOT=$WORK/robocasa-gr1-tabletop-tasks
export ENV_NAME=gr1rl                  # conda 环境名
export GPUS=0,1,2,3                    # 训练用的物理 GPU，必须连续
mkdir -p $WORK $MODELS
```

硬件要求（ja100 实测）：4 张 GPU，每张训练期间占 36–45 GB（4B backbone bf16 + FP32 OFT head 的 FSDP actor ≈ 15 GB、rollout 副本 ≈ 10 GB、8 个 MuJoCo EGL 渲染子进程各 1.3–1.8 GB）；CPU 64 核（每张卡 8 个模拟器子进程 + 评测时 56 个 llvmpipe 模拟器）；SAC 配方的 replay 窗口每 rank 约 11 GB 内存；磁盘 ≥ 100 GB。

---

## 2. 系统依赖

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential git git-lfs ffmpeg \
  libegl1 libgl1 libglvnd-dev libgl1-mesa-dri libegl-mesa0 libosmesa6 \
  libglfw3 libglew-dev
```

检查两个 EGL 后端都在（训练用 NVIDIA EGL 渲染，固定 seed 评测用 Mesa llvmpipe 复现官方像素）：
```bash
ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0
ls /usr/share/glvnd/egl_vendor.d/        # ja100 上只有 50_mesa.json；NVIDIA 的 manifest 由本仓库提供
cat $RLINF_ROOT/examples/embodiment/runtime_site/10_nvidia.json   # 训练时通过 __EGL_VENDOR_LIBRARY_FILENAMES 指向它
nvidia-smi --query-gpu=driver_version --format=csv,noheader        # ≥ 550（cu124 wheel 要求），ja100 是 570.195.03
```

---

## 3. Python 环境（单一 conda 环境，合并 ja100 上的三层）

ja100 的环境是 conda `cve_env`（torch/flash-attn）+ 两个 `--system-site-packages` 的 uv venv（starvla-cve、robocasa-gr1-cve）+ 借用第三个 conda 环境里的 wandb，通过 `runtime_site/sitecustomize.py` 拼起来。新机器上**不要**复制这些目录，按下面重建一个环境即可。

```bash
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME
export TRAIN_PYTHON=$(which python)

# 3.1 torch（cu124 wheel 自带 CUDA 运行库，不需要系统 CUDA toolkit）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3.2 flash-attn 2.8.3 预编译 wheel（torch 2.6 / cu12 / cp310 / cxx11abi=FALSE，ja100 实测组合）
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "import torch, flash_attn; print(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI, flash_attn.__version__)"
# 期望: 2.6.0+cu124 False 2.8.3

# 3.3 锁定依赖（189 个 pin，含 ray 2.49.2 / transformers 4.57.0 / numpy 1.26.4 / mujoco 3.2.6 / robosuite 1.5.1 / wandb 0.27）
pip install -r $RLINF_ROOT/examples/embodiment/migration/requirements-gr1.lock
python -c "import ray, transformers, numpy, mujoco, robosuite; print(ray.__version__, transformers.__version__, numpy.__version__, mujoco.__version__, robosuite.__version__)"
# 期望: 2.49.2 4.57.0 1.26.4 3.2.6 1.5.1
```

注意事项：
- **Ray 必须是 2.49.2**。2.57.0 会在 RLinf 的 ChannelWorker 进程组初始化时触发 pybind11 释放错误而崩溃；`run_robocasa_gr1_cup_drawer_grpo.sh` 启动前会检查版本。
- 锁定文件里排除了 torch/flash-attn/starVLA/robocasa（分别在 3.1/3.2/第 4 节安装）。两个 venv 层版本不一致的包按训练解释器的版本取（transformers 4.57.0、pydantic 2.11.9）。
- 原始的两份 `pip freeze` 保留在 `examples/embodiment/migration/freeze-*.lock` 供排查。
- 如果 pip 解析器报冲突，优先保证：`ray==2.49.2 transformers==4.57.0 numpy==1.26.4 mujoco==3.2.6 robosuite==1.5.1 gymnasium==1.0.0 hydra-core==1.3.5 omegaconf==2.3.0 peft==0.17.1 accelerate==1.10.1`，其余可放宽。

---

## 4. 三个代码仓库

### 4.1 RLinf
```bash
git clone -b robocasa-gr1-grpo-fix git@github.com:VoyagerTHU/RLinf.git $RLINF_ROOT
cd $RLINF_ROOT
git fetch origin robocasa-gr1-grpo-baseline:robocasa-gr1-grpo-baseline   # 对照原版时用
git remote add upstream https://github.com/RLinf/RLinf.git
```
RLinf 不需要 `pip install`，启动脚本通过 `PYTHONPATH` 引用。

### 4.2 starVLA（上游 `dfa2189` + 补丁）
```bash
git clone https://github.com/starVLA/starVLA.git $STARVLA_ROOT
cd $STARVLA_ROOT && git checkout dfa2189e2508331fce7bc40c29fc76bba001e6c0
git checkout -b voyager/gr1-eval-tooling
git am $RLINF_ROOT/examples/embodiment/migration/patches/starvla/*.patch
pip install -e . --no-deps
mkdir -p playground/Pretrained_models
ln -s $MODELS/Qwen/Qwen3-VL-4B-Instruct playground/Pretrained_models/Qwen3-VL-4B-Instruct
```
补丁内容：固定 seed 评测脚本、文件式 policy server、RoboCasa 接口 / simulation_env / 视频 wrapper 修改、`framework/__init__.py` 的 logger 调用修正。ja100 上的分支同名（`voyager/gr1-eval-tooling` @ 0a53e04）。

> 软链的原因：OFT checkpoint 内部把基座记录为 `./playground/Pretrained_models/Qwen3-VL-4B-Instruct`（相对路径），所以启动脚本会 `cd $STARVLA_ROOT` 后再运行训练。

### 4.3 robocasa-gr1-tabletop-tasks（上游 `4840e67` + 补丁）
```bash
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git $ROBOCASA_TASK_ROOT
cd $ROBOCASA_TASK_ROOT && git checkout 4840e671596f93ca03651524b9f72ffb1aadfeff
git checkout -b voyager/gr1-rlinf-fixes
git am $RLINF_ROOT/examples/embodiment/migration/patches/robocasa-gr1-tabletop-tasks/*.patch
pip install -e . --no-deps
```
补丁内容（缺一不可）：
- `gymnasium_basic.reset(seed=...)` 重新给 robosuite 自己的 `Generator` 播种——否则第一集之后场景/物体随机化不跟 seed 走，固定 seed 评测和 GRPO 的同 seed 分组都会失效；
- `obj_in_drawer` 子任务信号改用与成功判定相同的 `obj_inside_of(partial_check=True)`；
- `download_groot_assets.py` 支持 `HF_ENDPOINT` 镜像。

（如果之后把这两个仓库 fork 到 VoyagerTHU 下，可以直接 `git clone -b voyager/... git@github.com:VoyagerTHU/<repo>.git`，省掉打补丁。）

### 4.4 RoboCasa 资产（8.2 GB：objects 6.7 GB、fixtures 1 GB、textures 0.5 GB）
```bash
cd $ROBOCASA_TASK_ROOT
python robocasa/scripts/download_groot_assets.py --help     # 看清参数名后：
python robocasa/scripts/download_groot_assets.py -y         # 国内可先 export HF_ENDPOINT=https://hf-mirror.com
du -sh robocasa/models/assets                               # 期望 ≈ 8.2G
```
或直接 rsync ja100 的 `robocasa/models/assets/`。

---

## 5. 模型权重

```bash
mkdir -p $MODELS/StarVLA $MODELS/Qwen
# 方式 A：从 ja100 拷贝
rsync -avP ja100:/data/dengyixuan/wyz/models/StarVLA/Qwen3-VL-OFT-Robocasa $MODELS/StarVLA/
rsync -avP ja100:/data/dengyixuan/wyz/models/Qwen/Qwen3-VL-4B-Instruct   $MODELS/Qwen/
# 方式 B：HuggingFace
huggingface-cli download StarVLA/Qwen3-VL-OFT-Robocasa --local-dir $MODELS/StarVLA/Qwen3-VL-OFT-Robocasa
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct     --local-dir $MODELS/Qwen/Qwen3-VL-4B-Instruct

ls -la $MODELS/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt   # 9 785 286 394 字节
```

---

## 6. 配置里需要改的路径

三个配方（`examples/embodiment/config/robocasa_gr1_cup_drawer_{grpo,ppo,sac}_starvla.yaml`）里写死了 ja100 的路径，用 Hydra 覆盖或直接编辑：

| 键 | ja100 值 | 新机器 |
|---|---|---|
| `actor.model.model_path` | `/data/dengyixuan/wyz/models/StarVLA/.../steps_90000_pytorch_model.pt` | `$MODELS/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt` |
| `runner.logger.log_path` | 由 `ROBOCASA_GR1_{GRPO,PPO,SAC}_LOG_ROOT` 环境变量给 | 启动脚本自动设为 `$EXP_ROOT/run` |
| `cluster.component_placement` | 由 `ROBOCASA_GR1_ACTOR_ROLLOUT_PLACEMENT` / `ROBOCASA_GR1_ENV_PLACEMENT` 给，默认 `4-7` | 启动脚本按 `GPUS` 自动设置 |
| `env.eval.mesa_egl_device` | `8`（= ja100 的 NVIDIA GPU 数：EGL 设备表先列 8 张 NVIDIA 卡，第 9 个才是 Mesa llvmpipe） | **改成新机器的 NVIDIA GPU 总数**（例如 4 卡机器填 4）；只影响评测渲染 |
| SAC 配方 `env.train.seed_manifest` | 指向 ja100 一个实验目录里的 50-seed 清单 | 改为 `${oc.env:REPO_PATH}/examples/embodiment/config/seeds/robocasa_gr1_cup_drawer_train_500.json`，或从 ja100 拷贝那份 |

`env` 和 `actor` 的 placement 必须相同（同一组 GPU）：GRPO 的 seed 组是在每个 actor rank 本地归一化的，env/actor world size 不同会把组拆散，`validate_embodied_cfg` 会直接报错。

---

## 7. 验证（不需要 GPU 的先做）

```bash
cd $RLINF_ROOT
pip install pytest ruff
# 7.1 单元测试（含分组校验、KL 控制器、PPO 指标、SAC mask）
PYTHONPATH=$PWD python -m pytest -q tests/unit_tests/test_rollout_grouping.py tests/unit_tests/algorithms tests/unit_tests/test_rollout_horizon.py tests/unit_tests/test_grad_clip_per_group.py tests/unit_tests/test_sac_transition_valid.py tests/unit_tests/envs/test_robocasa_gr1_seed_pool.py tests/unit_tests/envs/test_robocasa_gr1_egl.py
# 期望全部通过（ja100: 61 passed + envs 测试）

# 7.2 配置能组合
export EMBODIED_PATH=$RLINF_ROOT/examples/embodiment REPO_PATH=$RLINF_ROOT
python - <<'EOF'
import os
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=os.environ["EMBODIED_PATH"]+"/config", version_base=None):
    cfg = compose(config_name="robocasa_gr1_cup_drawer_grpo_starvla")
OmegaConf.resolve(cfg); print(cfg.actor.model.model_path, cfg.cluster.component_placement)
EOF

# 7.3 模型能加载（需要 1 张 GPU，~10 GB）
cd $STARVLA_ROOT && PYTHONPATH=$RLINF_ROOT:$STARVLA_ROOT:$ROBOCASA_TASK_ROOT python - <<'EOF'
import torch
from omegaconf import OmegaConf
from rlinf.models.embodiment.starvla import get_model
cfg = OmegaConf.create({"model_path": "$MODELS/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt".replace("$MODELS", __import__("os").environ["MODELS"]),
    "precision": "bf16", "action_model_precision": "fp32", "action_dim": 29, "num_action_chunks": 16,
    "num_executed_action_chunks": 12, "rollout_prompt_seq_len": 155, "policy_setup": "gr1", "unnorm_key": "gr1",
    "action_stats_source": "minmax", "add_value_head": False, "initial_logstd": -3.5, "trainable_logstd": False,
    "use_proprio": False, "starvla": {"ckpt_path": "${model_path}", "framework_name": "QwenOFT", "expected_action_dim": 29,
    "expected_num_action_chunks": 16, "expected_image_size": [224, 224], "enable_state_input": False}})
m = get_model(cfg); print(type(m).__name__, sum(p.numel() for p in m.parameters())/1e9, "B params")
EOF

# 7.4 MuJoCo + EGL 能渲染（在训练 GPU 上）
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPUS%%,*} __EGL_VENDOR_LIBRARY_FILENAMES=$RLINF_ROOT/examples/embodiment/runtime_site/10_nvidia.json \
PYTHONPATH=$ROBOCASA_TASK_ROOT python - <<'EOF'
import gymnasium as gym, robocasa, robocasa.utils.gym_utils.gymnasium_groot
env = gym.make("gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env", enable_render=True, disable_env_checker=True)
obs, info = env.reset(seed=1)
print(obs["video.ego_view_bg_crop_pad_res256_freq20"].shape, obs["annotation.human.coarse_action"])
EOF
# 期望: (256, 256, 3) unlocked_waist: pick up the cup, place it into the drawer and close the drawer
```

---

## 8. 启动训练

```bash
export EXP_ROOT=$WORK/experiments/gr1_grpo_$(date +%Y%m%d)
export CONFIG_NAME=robocasa_gr1_cup_drawer_grpo_starvla        # 或 ..._ppo_starvla / ..._sac_starvla
setsid nohup bash $RLINF_ROOT/examples/embodiment/migration/launch_gr1_training.sh \
    actor.model.model_path=$MODELS/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt \
    env.eval.mesa_egl_device=$(nvidia-smi -L | wc -l) \
    > /dev/null 2>&1 &
```
启动脚本会：校验 `GPUS` 空闲 → 在 `ray start` **之前**导出全部环境变量 → 起私有 Ray head（`--num-gpus=<n>`）→ 调 `run_robocasa_gr1_cup_drawer_grpo.sh`。产物：

| 文件 | 内容 |
|---|---|
| `$EXP_ROOT/logs/train.log` | 全部训练日志 |
| `$EXP_ROOT/run/metrics/training_step_metrics.jsonl` | 每步指标（actor 指标在 `train/actor/*`） |
| `$EXP_ROOT/run/rollout_tables/{train,eval}_seed_rollouts.jsonl` | 逐 seed / 逐轨迹结果 |
| `$EXP_ROOT/run/<experiment_name>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt` | 每 5 步一个，19 GB |
| `$EXP_ROOT/run/video/{train,eval}` | 视频 |
| `$EXP_ROOT/train.exit` | 训练进程退出码（出现即已结束） |

停止：`bash $RLINF_ROOT/examples/embodiment/migration/stop_gr1_training.sh $EXP_ROOT $TRAIN_PYTHON`。
续训：追加 `runner.resume_dir=$EXP_ROOT/run/<experiment_name>/checkpoints/global_step_N`。

### 冒烟对照（ja100，GRPO 配方 step 0，4×A100）

| 指标 | ja100 |
|---|---|
| `env/success_once`（128 条采样轨迹） | 0.35（后续步 0.35–0.65） |
| `env/grasped_once` | 0.91 |
| `train/actor/rollout_actor_logprob_mean_abs_delta` | 0.006 |
| `train/actor/post_update_logprob_mean_abs_delta` | 0.042 |
| `train/actor/last_global_batch_proximal_kl` | 1.8e-3 |
| `train/actor/optimizer_steps_this_update` | 15 |
| `time/generate_rollouts` / `time/actor/run_training` / `time/step` | 372 s / 382 s / 773 s |
| 固定 50-seed 评测（每 5 步） | 基线 0.48–0.54 → 0.56, 0.46, 0.64, 0.64, 0.64, 0.66, 0.58, 0.54, 0.60, 0.66, 0.48, 0.56, 0.62, 0.70 |

新机器上第一步如果 `success_once` 在 0.3–0.5、`post_update_logprob_mean_abs_delta` 在 0.03–0.05、没有 early-stop，就可以认为复现成功。

---

## 9. 已知的坑

1. **Ray 2.57** 崩溃，锁定 2.49.2。
2. **环境变量要在 `ray start` 之前导出**（PYTHONPATH、EGL manifest 等），否则 Ray worker 报 `No module named 'rlinf'`。启动脚本已处理。
3. **`env.eval.mesa_egl_device`** 是主机相关的：等于该机器 NVIDIA GPU 的数量。
4. **`rollout_prompt_seq_len: 155`** 是这个任务的 prompt token 数（同一任务、同一 tokenizer 不变）；换任务需要重新量。
5. **`num_executed_action_chunks: 12`** 与 `env.*.action_steps_per_chunk: 12` 必须一致（预测 16 执行 12，对齐官方评测）。
6. 训练渲染用 NVIDIA EGL、评测用 Mesa llvmpipe，两个库都要装；`__EGL_VENDOR_LIBRARY_FILENAMES` 指向仓库里的 `10_nvidia.json`。
7. FP32 的 OFT head 是必需的：每轮均值只动 ~1e-3，bf16 网格（|w|≈1 处 4e-3）会吞掉更新。
8. 日志默认 `logger_backends: []`（本地 jsonl）；要上 W&B 自行加 `wandb` 并配好 API key。
9. 这台机器上 ja100 用的 `sitecustomize.py` 多环境拼接不再需要，保持 `RLINF_EXTRA_SITE_PACKAGES` 为空即可。
10. 深度补丁/依赖发生冲突时，`examples/embodiment/migration/freeze-*.lock` 是 ja100 上两个 venv 的原始 `pip freeze`。

---

## 10. 目录结构速查

```
$WORK/
├── RLinf/                      # 本仓库，分支 robocasa-gr1-grpo-fix
│   ├── MIGRATION_ROBOCASA_GR1.md
│   └── examples/embodiment/
│       ├── config/robocasa_gr1_cup_drawer_{grpo,ppo,sac}_starvla.yaml
│       ├── config/seeds/robocasa_gr1_cup_drawer_{train_500,eval_50}.json
│       ├── runtime_site/10_nvidia.json
│       ├── run_robocasa_gr1_cup_drawer_grpo.sh          # 内层启动脚本（Ray 版本检查、PYTHONPATH）
│       └── migration/
│           ├── requirements-gr1.lock                     # 合并后的依赖锁定
│           ├── freeze-starvla-cve.lock / freeze-robocasa-gr1-cve.lock
│           ├── patches/starvla/*.patch
│           ├── patches/robocasa-gr1-tabletop-tasks/*.patch
│           ├── launch_gr1_training.sh                    # 外层启动脚本（GPU 校验、私有 Ray）
│           └── stop_gr1_training.sh
├── starVLA/                    # dfa2189 + 补丁，playground/Pretrained_models/Qwen3-VL-4B-Instruct -> $MODELS/Qwen/...
├── robocasa-gr1-tabletop-tasks/# 4840e67 + 补丁，robocasa/models/assets/ 8.2 GB
├── models/
│   ├── StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt
│   └── Qwen/Qwen3-VL-4B-Instruct/
└── experiments/<run>/          # EXP_ROOT
```
