#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STARVLA_ROOT=${STARVLA_ROOT:-/data/dengyixuan/wyz/robots/starVLA}
ROBOCASA_TASK_ROOT=${ROBOCASA_TASK_ROOT:-/data/dengyixuan/wyz/robots/robocasa-gr1-tabletop-tasks}
EVAL_PYTHON=${EVAL_PYTHON:-/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python}
ROBOCASA_SITE_PACKAGES=${ROBOCASA_SITE_PACKAGES:-/data/dengyixuan/wyz/.venvs/robocasa-gr1-cve/lib/python3.10/site-packages}
WANDB_SITE_PACKAGES=${WANDB_SITE_PACKAGES:-/home/dengyixuan/anaconda3/envs/VAMPO/lib/python3.10/site-packages}
CONFIG_NAME=${CONFIG_NAME:-robocasa_gr1_cup_drawer_grpo_starvla}
EXPECTED_RAY_VERSION=${EXPECTED_RAY_VERSION:-2.49.2}

CHECKPOINT=${1:?Usage: $0 FULL_WEIGHTS_CHECKPOINT_OR_base [RUN_ROOT] [EVAL_STEP]}
RUN_ROOT=${2:-/data/dengyixuan/wyz/experiments/rlinf_starvla_robocasa_gr1_step_fixed50_eval_$(date +%Y%m%d_%H%M%S)}
EVAL_STEP=${3:-0}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-cup_to_drawer_close_step${EVAL_STEP}_fixed50_eval}

CHECKPOINT_ARGS=()
if [[ "${CHECKPOINT}" != "base" ]]; then
    CHECKPOINT_ARGS+=("runner.ckpt_path=${CHECKPOINT}")
fi

RAY_VERSION=$("${EVAL_PYTHON}" -c 'import ray; print(ray.__version__)')
if [[ "${RAY_VERSION}" != "${EXPECTED_RAY_VERSION}" ]]; then
    echo "This recipe requires Ray ${EXPECTED_RAY_VERSION}; found ${RAY_VERSION}." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}"

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export REPO_PATH="${RLINF_ROOT}"
export PYTHONPATH="${RLINF_ROOT}:${STARVLA_ROOT}:${ROBOCASA_TASK_ROOT}:${RLINF_ROOT}/examples/embodiment/runtime_site${PYTHONPATH:+:${PYTHONPATH}}"
export RLINF_EXTRA_SITE_PACKAGES="${ROBOCASA_SITE_PACKAGES}:${WANDB_SITE_PACKAGES}${RLINF_EXTRA_SITE_PACKAGES:+:${RLINF_EXTRA_SITE_PACKAGES}}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export __EGL_VENDOR_LIBRARY_FILENAMES="${RLINF_ROOT}/examples/embodiment/runtime_site/10_nvidia.json"
export TORCH_HOME=/data/dengyixuan/wyz/.cache/torch
export HF_HOME=/data/dengyixuan/wyz/huggingface
export VBENCH_CACHE_DIR=/data/dengyixuan/wyz/.cache/vbench
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=1

cd "${STARVLA_ROOT}"

exec "${EVAL_PYTHON}" "${EMBODIED_PATH}/eval_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name "${CONFIG_NAME}" \
    "${CHECKPOINT_ARGS[@]}" \
    "runner.logger.log_path=${RUN_ROOT}" \
    "runner.logger.experiment_name=${EXPERIMENT_NAME}" \
    "+runner.eval_step=${EVAL_STEP}" \
    "env.eval.video_cfg.save_video=true"
