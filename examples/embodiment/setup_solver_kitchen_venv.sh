#!/usr/bin/env bash
# Create the solver virtual environment used by the solver_kitchen EnvWorker.
#
#   SOLVER_ROOT=/path/to/solver bash examples/embodiment/setup_solver_kitchen_venv.sh
#
# The solver lock pins a MuJoCo nightly that is no longer downloadable and a
# private mujoco-warp fork (SSH access to GitHub required), and its default
# torch build targets CUDA 13 while this host runs driver 570 (CUDA 12.8).
# This script therefore installs the lock with three substitutions:
#   * mujoco==3.10.0 from PyPI instead of the missing nightly,
#   * mujoco-warp from the fork through SSH,
#   * torch 2.12.0 from the cu126 index.
set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOLVER_ROOT=${SOLVER_ROOT:-$(cd "${RLINF_ROOT}/../solver" && pwd)}
PYTHON_VERSION=${SOLVER_PYTHON_VERSION:-3.11}
TORCH_INDEX=${SOLVER_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
TORCH_VERSION=${SOLVER_TORCH_VERSION:-2.12.0}
MUJOCO_VERSION=${SOLVER_MUJOCO_VERSION:-3.10.0}
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.git@github.com:.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/

cd "${SOLVER_ROOT}"
uv venv .venv --python "${PYTHON_VERSION}" --allow-existing
uv export --frozen --no-hashes --no-emit-project --format requirements-txt -o "${WORK}/req.txt"
grep -v '^mujoco==\|^mujoco-warp @' "${WORK}/req.txt" > "${WORK}/req_base.txt"
grep '^mujoco-warp @' "${WORK}/req.txt" > "${WORK}/req_mjwarp.txt"
uv pip install --python .venv/bin/python -r "${WORK}/req_base.txt"
uv pip install --python .venv/bin/python "mujoco==${MUJOCO_VERSION}"
uv pip install --python .venv/bin/python --no-deps -r "${WORK}/req_mjwarp.txt"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python --index-url "${TORCH_INDEX}" \
    --reinstall-package torch "torch==${TORCH_VERSION}"
.venv/bin/python - <<'PY'
import torch, warp as wp, solver.rl, mujoco_warp  # noqa: F401
wp.config.quiet = True
wp.init()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp devices", [d.alias for d in wp.get_cuda_devices()])
PY
echo "solver venv ready: ${SOLVER_ROOT}/.venv/bin/python"
