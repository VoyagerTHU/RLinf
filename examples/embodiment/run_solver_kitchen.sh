#!/usr/bin/env bash
# Launch an RLinf embodied recipe on the solver kitchen environment.
#
#   CONFIG_NAME=solver_kitchen_ppo_mlp CUDA_VISIBLE_DEVICES=3 \
#     bash examples/embodiment/run_solver_kitchen.sh [hydra overrides...]
#
# Required environment:
#   SOLVER_PYTHON  interpreter of the solver venv (Python 3.11 + Newton/Warp),
#                  created by examples/embodiment/setup_solver_kitchen_venv.sh
#   TRAIN_PYTHON   interpreter of the RLinf training venv (default below)
set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOLVER_ROOT=${SOLVER_ROOT:-$(cd "${RLINF_ROOT}/../solver" 2>/dev/null && pwd || echo "")}
SOLVER_PYTHON=${SOLVER_PYTHON:-${SOLVER_ROOT}/.venv/bin/python}
TRAIN_PYTHON=${TRAIN_PYTHON:-/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python}
STARVLA_ROOT=${STARVLA_ROOT:-/data/dengyixuan/wyz/robots/starVLA}
ROBOCASA_SITE_PACKAGES=${ROBOCASA_SITE_PACKAGES:-/data/dengyixuan/wyz/.venvs/robocasa-gr1-cve/lib/python3.10/site-packages}
WANDB_SITE_PACKAGES=${WANDB_SITE_PACKAGES:-/home/dengyixuan/anaconda3/envs/VAMPO/lib/python3.10/site-packages}
CONFIG_NAME=${CONFIG_NAME:-solver_kitchen_ppo_mlp}

if [[ ! -x "${SOLVER_PYTHON}" ]]; then
    echo "SOLVER_PYTHON=${SOLVER_PYTHON} is not executable; run setup_solver_kitchen_venv.sh first." >&2
    exit 1
fi

export SOLVER_ROOT SOLVER_PYTHON
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export REPO_PATH="${RLINF_ROOT}"
export PYTHONPATH="${RLINF_ROOT}:${STARVLA_ROOT}:${RLINF_ROOT}/examples/embodiment/runtime_site${PYTHONPATH:+:${PYTHONPATH}}"
export RLINF_EXTRA_SITE_PACKAGES="${ROBOCASA_SITE_PACKAGES}:${WANDB_SITE_PACKAGES}${RLINF_EXTRA_SITE_PACKAGES:+:${RLINF_EXTRA_SITE_PACKAGES}}"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# Warp compiles kernels into this cache; keep it off the shared home quota.
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-/tmp/solver-kitchen-warp-cache-$(id -u)}

echo "RLinf root:     ${RLINF_ROOT}"
echo "Solver python:  ${SOLVER_PYTHON}"
echo "Train python:   ${TRAIN_PYTHON}"
echo "Config:         ${CONFIG_NAME}"

# StarVLA checkpoints resolve their base VLM relative to the StarVLA root.
if [[ -d "${STARVLA_ROOT}" ]]; then
    cd "${STARVLA_ROOT}"
fi

exec "${TRAIN_PYTHON}" "${EMBODIED_PATH}/train_embodied_agent.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name "${CONFIG_NAME}" \
    "$@"
