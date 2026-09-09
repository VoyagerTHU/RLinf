#!/usr/bin/env bash
# Detached launcher for the RoboCasa GR1 / StarVLA recipes (GRPO, PPO, SAC).
#
# What it does (mirrors the ja100 experiment launchers, generalized):
#   * pins the run to an explicit list of physical GPUs and refuses to start if
#     they are busy;
#   * exports every environment variable BEFORE `ray start` (Ray captures the
#     environment at start time; exporting later inside the inner script is
#     too late for the workers);
#   * starts a private Ray head on a random port with `--num-gpus=<n>`;
#   * runs examples/embodiment/run_robocasa_gr1_cup_drawer_grpo.sh with
#     CONFIG_NAME and any extra Hydra overrides;
#   * writes logs/train.log, launcher.pid, ray_address.txt, train.exit.
#
# Usage:
#   EXP_ROOT=/data/exp/gr1_grpo_$(date +%Y%m%d) GPUS=4,5,6,7 \
#   CONFIG_NAME=robocasa_gr1_cup_drawer_grpo_starvla \
#   bash examples/embodiment/migration/launch_gr1_training.sh [hydra overrides...]
#
# Run it detached:  setsid nohup bash launch_gr1_training.sh > /dev/null 2>&1 &
# Stop it:          bash examples/embodiment/migration/stop_gr1_training.sh $EXP_ROOT
set -euo pipefail

# ---- required / commonly overridden -----------------------------------------
RLINF_ROOT=${RLINF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
EXP_ROOT=${EXP_ROOT:?set EXP_ROOT to the experiment directory (logs, checkpoints, videos)}
GPUS=${GPUS:?set GPUS to a comma-separated list of physical GPU indices, e.g. 4,5,6,7}
CONFIG_NAME=${CONFIG_NAME:-robocasa_gr1_cup_drawer_grpo_starvla}
TRAIN_PYTHON=${TRAIN_PYTHON:?set TRAIN_PYTHON to the python of the training environment}
STARVLA_ROOT=${STARVLA_ROOT:?set STARVLA_ROOT to the patched starVLA checkout}
ROBOCASA_TASK_ROOT=${ROBOCASA_TASK_ROOT:?set ROBOCASA_TASK_ROOT to the patched robocasa-gr1-tabletop-tasks checkout}
# Optional: extra site-packages directories appended after the interpreter's
# own (legacy multi-venv layout). Leave empty for a single consolidated env.
RLINF_EXTRA_SITE_PACKAGES=${RLINF_EXTRA_SITE_PACKAGES:-}
# Optional: EGL vendor manifest forcing NVIDIA EGL for training simulators.
EGL_VENDOR_MANIFEST=${EGL_VENDOR_MANIFEST:-$RLINF_ROOT/examples/embodiment/runtime_site/10_nvidia.json}
# Optional: caches
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
# Optional: require these exact GPU UUIDs (comma separated, same order as GPUS).
GPU_UUID_ALLOWLIST=${GPU_UUID_ALLOWLIST:-}
MAX_BUSY_MIB=${MAX_BUSY_MIB:-1024}

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/run"
exec 9>"$EXP_ROOT/train.lock"
flock -n 9 || { echo "training already running under $EXP_ROOT" >&2; exit 1; }

# ---- GPU selection -----------------------------------------------------------
IFS=',' read -r -a GPU_ARR <<<"$GPUS"
NUM_GPUS=${#GPU_ARR[@]}
FIRST=${GPU_ARR[0]}; LAST=${GPU_ARR[$((NUM_GPUS-1))]}
if (( LAST - FIRST + 1 != NUM_GPUS )); then
    echo "GPUS must be a contiguous ascending range (RLinf placement uses 'a-b'), got $GPUS" >&2
    exit 1
fi
if (( NUM_GPUS == 1 )); then PLACEMENT="$FIRST"; else PLACEMENT="$FIRST-$LAST"; fi

"$TRAIN_PYTHON" - "$GPUS" "$GPU_UUID_ALLOWLIST" "$MAX_BUSY_MIB" <<'PY'
import csv, subprocess, sys
gpus = [int(x) for x in sys.argv[1].split(",")]
allow = [u for u in sys.argv[2].split(",") if u]
max_busy = int(sys.argv[3])
rows = csv.reader(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True).splitlines())
actual = {int(i): (u.strip(), int(m)) for i, u, m in rows}
for k, i in enumerate(gpus):
    if i not in actual:
        sys.exit(f"GPU {i} does not exist")
    if actual[i][1] > max_busy:
        sys.exit(f"GPU {i} is busy ({actual[i][1]} MiB used)")
    if allow and actual[i][0] != allow[k]:
        sys.exit(f"GPU {i} has uuid {actual[i][0]}, expected {allow[k]}")
print(f"GPUs {gpus} verified idle")
PY

# ---- environment (must precede `ray start`) ----------------------------------
export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ROBOCASA_GR1_ACTOR_ROLLOUT_PLACEMENT="$PLACEMENT"
export ROBOCASA_GR1_ENV_PLACEMENT="$PLACEMENT"
export ROBOCASA_GR1_GRPO_LOG_ROOT="$EXP_ROOT/run"
export ROBOCASA_GR1_PPO_LOG_ROOT="$EXP_ROOT/run"
export ROBOCASA_GR1_SAC_LOG_ROOT="$EXP_ROOT/run"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export REPO_PATH="$RLINF_ROOT"
export PYTHONPATH="$RLINF_ROOT:$STARVLA_ROOT:$ROBOCASA_TASK_ROOT:$RLINF_ROOT/examples/embodiment/runtime_site${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_EXTRA_SITE_PACKAGES
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_MANIFEST"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_NO_CUDA_MEMORY_CACHING RAY_ADDRESS
export STARVLA_ROOT ROBOCASA_TASK_ROOT TRAIN_PYTHON CONFIG_NAME

# ---- private Ray head ----------------------------------------------------------
RAY_BIN=$(dirname "$TRAIN_PYTHON")/ray
RAY_PORT=$("$TRAIN_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
RAY_TMP=${RAY_TMP:-/tmp/ray-gr1-$RAY_PORT}
cleanup() { "$RAY_BIN" stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT
cd "$STARVLA_ROOT"   # the checkpoint references ./playground/Pretrained_models/... relative to here
"$RAY_BIN" start --head --port="$RAY_PORT" --num-gpus="$NUM_GPUS" --num-cpus="${RAY_NUM_CPUS:-$(nproc)}" \
    --include-dashboard=false --disable-usage-stats --temp-dir="$RAY_TMP" \
    > "$EXP_ROOT/logs/ray_start.log" 2>&1
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
printf '%s\n' "$RAY_ADDRESS" > "$EXP_ROOT/ray_address.txt"
printf '%s\n' "$$" > "$EXP_ROOT/launcher.pid"
date -u +%FT%TZ > "$EXP_ROOT/train_started.txt"
{
    echo "config=$CONFIG_NAME gpus=$GPUS placement=$PLACEMENT"
    echo "rlinf=$RLINF_ROOT ($(git -C "$RLINF_ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git))"
    echo "starvla=$STARVLA_ROOT ($(git -C "$STARVLA_ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git))"
    echo "robocasa=$ROBOCASA_TASK_ROOT ($(git -C "$ROBOCASA_TASK_ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git))"
    echo "overrides=$*"
} > "$EXP_ROOT/launch_info.txt"

set +e
bash "$RLINF_ROOT/examples/embodiment/run_robocasa_gr1_cup_drawer_grpo.sh" "$@" \
    > "$EXP_ROOT/logs/train.log" 2>&1
train_exit=$?
printf '%s\n' "$train_exit" > "$EXP_ROOT/train.exit"
date -u +%FT%TZ > "$EXP_ROOT/train_finished.txt"
exit "$train_exit"
