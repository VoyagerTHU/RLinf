#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STARVLA_ROOT=${STARVLA_ROOT:-/data/dengyixuan/wyz/robots/starVLA}
ROBOCASA_TASK_ROOT=${ROBOCASA_TASK_ROOT:-/data/dengyixuan/wyz/robots/robocasa-gr1-tabletop-tasks}
TRAIN_PYTHON=${TRAIN_PYTHON:-/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python}
ROBOCASA_SITE_PACKAGES=${ROBOCASA_SITE_PACKAGES:-/data/dengyixuan/wyz/.venvs/robocasa-gr1-cve/lib/python3.10/site-packages}
WANDB_SITE_PACKAGES=${WANDB_SITE_PACKAGES:-/home/dengyixuan/anaconda3/envs/VAMPO/lib/python3.10/site-packages}
CONFIG_NAME=${CONFIG_NAME:-robocasa_gr1_cup_drawer_grpo_starvla}
EXPECTED_RAY_VERSION=${EXPECTED_RAY_VERSION:-2.49.2}

RAY_VERSION=$("${TRAIN_PYTHON}" -c 'import ray; print(ray.__version__)')
if [[ "${RAY_VERSION}" != "${EXPECTED_RAY_VERSION}" ]]; then
    echo "This recipe requires Ray ${EXPECTED_RAY_VERSION}; found ${RAY_VERSION}." >&2
    echo "Ray 2.57.0 crashes RLinf ChannelWorker process-group setup on this host." >&2
    exit 1
fi

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export REPO_PATH="${RLINF_ROOT}"
export PYTHONPATH="${RLINF_ROOT}:${STARVLA_ROOT}:${ROBOCASA_TASK_ROOT}:${RLINF_ROOT}/examples/embodiment/runtime_site${PYTHONPATH:+:${PYTHONPATH}}"
export RLINF_EXTRA_SITE_PACKAGES="${ROBOCASA_SITE_PACKAGES}:${WANDB_SITE_PACKAGES}${RLINF_EXTRA_SITE_PACKAGES:+:${RLINF_EXTRA_SITE_PACKAGES}}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
# The host GLVND directory exposes only Mesa. Point GLVND at the project-local
# NVIDIA manifest so EGL initialization fails instead of silently using swrast.
export __EGL_VENDOR_LIBRARY_FILENAMES="${RLINF_ROOT}/examples/embodiment/runtime_site/10_nvidia.json"
export TORCH_HOME=/data/dengyixuan/wyz/.cache/torch
export HF_HOME=/data/dengyixuan/wyz/huggingface
export VBENCH_CACHE_DIR=/data/dengyixuan/wyz/.cache/vbench
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# The checkpoint stores its base VLM as ./playground/Pretrained_models/....
# Match StarVLA's official launch working directory so that path resolves to
# the existing local Qwen3 symlink.
cd "${STARVLA_ROOT}"

exec "${TRAIN_PYTHON}" "${EMBODIED_PATH}/train_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name "${CONFIG_NAME}" \
    "$@"
