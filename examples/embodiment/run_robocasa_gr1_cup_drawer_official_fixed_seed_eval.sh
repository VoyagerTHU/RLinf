#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STARVLA_ROOT=${STARVLA_ROOT:-/data/dengyixuan/wyz/robots/starVLA}
ROBOCASA_TASK_ROOT=${ROBOCASA_TASK_ROOT:-/data/dengyixuan/wyz/robots/robocasa-gr1-tabletop-tasks}
STARVLA_PYTHON=${STARVLA_PYTHON:-/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python}
ROBOCASA_PYTHON=${ROBOCASA_PYTHON:-/data/dengyixuan/wyz/.venvs/robocasa-gr1-cve/bin/python}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/data/dengyixuan/wyz/models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt}
CHECKPOINT=${1:?Usage: $0 FULL_WEIGHTS_CHECKPOINT [RUN_ROOT]}
RUN_ROOT=${2:-/data/dengyixuan/wyz/experiments/rlinf_starvla_robocasa_gr1_step10_official_fixed50_eval_$(date +%Y%m%d_%H%M%S)}
SEED_MANIFEST=${SEED_MANIFEST:-/data/dengyixuan/wyz/experiments/starvla_robocasa_gr1/seeds_24x50_base20260819.json}
# Any of the 24 gr1_unified tasks in the seed manifest; the base-checkpoint
# reference result is looked up per task unless REFERENCE_RESULT is given.
TASK_NAME=${TASK_NAME:-gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env}
TASK_SLUG=${TASK_NAME#*/}
REFERENCE_ROOT=${REFERENCE_ROOT:-/data/dengyixuan/wyz/experiments/starvla_robocasa_gr1/oft_steps90000_fixed_seed_base20260819/results_merged}
REFERENCE_RESULT=${REFERENCE_RESULT:-${REFERENCE_ROOT}/${TASK_SLUG}.json}
NUM_SHARDS=${NUM_SHARDS:-8}
GPU_IDS=${GPU_IDS:-}
ENSEMBLE_CHECKPOINT_B=${ENSEMBLE_CHECKPOINT_B:-}
ENSEMBLE_WEIGHT_B=${ENSEMBLE_WEIGHT_B:-0.5}
N_EPISODES=50

export TORCH_HOME=${TORCH_HOME:-/data/dengyixuan/wyz/.cache/torch}
export HF_HOME=${HF_HOME:-/data/dengyixuan/wyz/huggingface}
export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-/data/dengyixuan/wyz/.cache/vbench}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/data/dengyixuan/wyz/.cache/matplotlib}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

if ((NUM_SHARDS <= 0 || NUM_SHARDS > 8)); then
    echo "NUM_SHARDS must be between 1 and 8" >&2
    exit 1
fi

gpu_ids=()
if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
else
    for ((shard = 0; shard < NUM_SHARDS; shard++)); do
        gpu_ids+=("${shard}")
    done
fi
if ((${#gpu_ids[@]} != NUM_SHARDS)); then
    echo "GPU_IDS must contain exactly NUM_SHARDS=${NUM_SHARDS} comma-separated IDs" >&2
    exit 1
fi
declare -A seen_gpu_ids=()
for gpu in "${gpu_ids[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-7]$ ]]; then
        echo "GPU_IDS entries must be physical GPU IDs between 0 and 7; got ${gpu}" >&2
        exit 1
    fi
    if [[ -n "${seen_gpu_ids[${gpu}]:-}" ]]; then
        echo "GPU_IDS must not contain duplicates; got ${GPU_IDS}" >&2
        exit 1
    fi
    seen_gpu_ids[${gpu}]=1
done

ensemble_server_args=()
if [[ -n "${ENSEMBLE_CHECKPOINT_B}" ]]; then
    if [[ ! -f "${ENSEMBLE_CHECKPOINT_B}" ]]; then
        echo "ENSEMBLE_CHECKPOINT_B does not exist: ${ENSEMBLE_CHECKPOINT_B}" >&2
        exit 1
    fi
    ensemble_server_args+=(
        --checkpoint-b "${ENSEMBLE_CHECKPOINT_B}"
        --ensemble-weight-b "${ENSEMBLE_WEIGHT_B}"
    )
fi

mkdir -p "${RUN_ROOT}/bridge" "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/results_shards" "${RUN_ROOT}/videos" "${MPLCONFIGDIR}"
if [[ -f "${RUN_ROOT}/seed_manifest.json" ]]; then
    cmp --silent "${SEED_MANIFEST}" "${RUN_ROOT}/seed_manifest.json" || {
        echo "Seed manifest differs from the frozen run manifest" >&2
        exit 1
    }
else
    cp "${SEED_MANIFEST}" "${RUN_ROOT}/seed_manifest.json"
fi
SEED_MANIFEST=${RUN_ROOT}/seed_manifest.json

SERVER_PIDS=()
EVAL_PIDS=()
cleanup() {
    for pid in "${EVAL_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    gpu=${gpu_ids[shard]}
    bridge_dir=${RUN_ROOT}/bridge/shard_${shard}
    mkdir -p "${bridge_dir}"
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH="${RLINF_ROOT}:${STARVLA_ROOT}" \
        "${STARVLA_PYTHON}" \
        "${RLINF_ROOT}/examples/embodiment/eval_starvla_robocasa_gr1_server.py" \
        --checkpoint "${CHECKPOINT}" \
        --base-checkpoint "${BASE_CHECKPOINT}" \
        --bridge-dir "${bridge_dir}" \
        --starvla-repo-root "${STARVLA_ROOT}" \
        "${ensemble_server_args[@]}" \
        > "${RUN_ROOT}/logs/server_shard_${shard}_gpu_${gpu}.log" 2>&1 &
    SERVER_PIDS+=("$!")
done

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    metadata=${RUN_ROOT}/bridge/shard_${shard}/metadata.msgpack
    deadline=$((SECONDS + 900))
    while [[ ! -f "${metadata}" ]]; do
        if ! kill -0 "${SERVER_PIDS[shard]}" 2>/dev/null; then
            echo "Policy server shard ${shard} exited during startup" >&2
            gpu=${gpu_ids[shard]}
            tail -n 100 "${RUN_ROOT}/logs/server_shard_${shard}_gpu_${gpu}.log" >&2
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for policy server shard ${shard}" >&2
            exit 1
        fi
        sleep 5
    done
done

base_count=$((N_EPISODES / NUM_SHARDS))
remainder=$((N_EPISODES % NUM_SHARDS))
seed_offset=0
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    shard_count=${base_count}
    if ((shard < remainder)); then
        shard_count=$((shard_count + 1))
    fi
    video_dir=${RUN_ROOT}/videos/shard_${shard}/${TASK_SLUG}
    mkdir -p "${video_dir}"
    env -u CUDA_VISIBLE_DEVICES -u __EGL_VENDOR_LIBRARY_FILENAMES \
        PYTHONPATH="${STARVLA_ROOT}:${ROBOCASA_TASK_ROOT}" \
        STARVLA_FILE_BRIDGE_DIR="${RUN_ROOT}/bridge/shard_${shard}" \
        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=8 \
        "${ROBOCASA_PYTHON}" \
        "${STARVLA_ROOT}/examples/Robocasa_tabletop/eval_files/simulation_env.py" \
        --args.env_name "${TASK_NAME}" \
        --args.n_episodes "${shard_count}" \
        --args.n_envs 1 \
        --args.max_episode_steps 720 \
        --args.n_action_steps 12 \
        --args.pretrained_path "${BASE_CHECKPOINT}" \
        --args.seed_manifest_path "${SEED_MANIFEST}" \
        --args.seed_offset "${seed_offset}" \
        --args.result_out_path "${RUN_ROOT}/results_shards/shard_${shard}.json" \
        --args.video_out_path "${video_dir}" \
        > "${RUN_ROOT}/logs/eval_shard_${shard}.log" 2>&1 &
    EVAL_PIDS+=("$!")
    seed_offset=$((seed_offset + shard_count))
done

for pid in "${EVAL_PIDS[@]}"; do
    wait "${pid}"
done
EVAL_PIDS=()

"${ROBOCASA_PYTHON}" \
    "${RLINF_ROOT}/examples/embodiment/summarize_robocasa_gr1_single_task_eval.py" \
    --run-root "${RUN_ROOT}" \
    --seed-manifest "${SEED_MANIFEST}" \
    --task-name "${TASK_NAME}" \
    --reference-result "${REFERENCE_RESULT}"
"${ROBOCASA_PYTHON}" \
    "${STARVLA_ROOT}/examples/Robocasa_tabletop/eval_files/verify_rollout_videos.py" \
    --run-root "${RUN_ROOT}" --workers 8 \
    --output "${RUN_ROOT}/video_decode_audit.json"

echo "Official-compatible fixed-seed evaluation complete: ${RUN_ROOT}"
