#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STARVLA_ROOT=${STARVLA_ROOT:-/data/dengyixuan/wyz/robots/starVLA}
RLINF_PYTHON=${RLINF_PYTHON:-/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python}
ROBOCASA_PYTHON=${ROBOCASA_PYTHON:-/data/dengyixuan/wyz/.venvs/robocasa-gr1-cve/bin/python}
CHECKPOINT=${1:?Usage: $0 CHECKPOINT [RUN_ROOT]}
RUN_ROOT=${2:-/data/dengyixuan/wyz/experiments/rlinf_starvla_robocasa_gr1/$(date +%Y%m%d_%H%M%S)}
SEED_MANIFEST=${SEED_MANIFEST:-/data/dengyixuan/wyz/experiments/starvla_robocasa_gr1/seeds_24x50_base20260819.json}
NUM_WORKERS=${NUM_WORKERS:-24}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-8}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}

export TORCH_HOME=${TORCH_HOME:-/data/dengyixuan/wyz/.cache/torch}
export HF_HOME=${HF_HOME:-/data/dengyixuan/wyz/huggingface}
export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-/data/dengyixuan/wyz/.cache/vbench}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/data/dengyixuan/wyz/.cache/matplotlib}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
# On this host EGL devices 0-7 are GPU DRI nodes that are not accessible to
# the current user; device 8 is the permission-free Mesa llvmpipe backend.
# Do not set CUDA_VISIBLE_DEVICES for simulators: robosuite incorrectly treats
# its value as an EGL enumeration index on this version.
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-8}
export PYTHONPATH="${RLINF_ROOT}:${STARVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_ROOT}/bridge" "${RUN_ROOT}/logs_servers" "${MPLCONFIGDIR}"
if [[ -f "${RUN_ROOT}/seed_manifest.json" ]]; then
    cmp --silent "${SEED_MANIFEST}" "${RUN_ROOT}/seed_manifest.json" || {
        echo "Seed manifest differs from the frozen run manifest" >&2
        exit 1
    }
else
    cp "${SEED_MANIFEST}" "${RUN_ROOT}/seed_manifest.json"
fi
SEED_MANIFEST="${RUN_ROOT}/seed_manifest.json"

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

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    gpu=$((worker % PHYSICAL_GPUS))
    bridge_dir="${RUN_ROOT}/bridge/worker${worker}"
    mkdir -p "${bridge_dir}"
    CUDA_VISIBLE_DEVICES=${gpu} "${RLINF_PYTHON}" \
        "${RLINF_ROOT}/examples/embodiment/eval_starvla_robocasa_gr1_server.py" \
        --checkpoint "${CHECKPOINT}" \
        --bridge-dir "${bridge_dir}" \
        --starvla-repo-root "${STARVLA_ROOT}" \
        > "${RUN_ROOT}/logs_servers/server_worker${worker}_gpu${gpu}.log" 2>&1 &
    SERVER_PIDS+=("$!")
done

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    metadata="${RUN_ROOT}/bridge/worker${worker}/metadata.msgpack"
    while [[ ! -f "${metadata}" ]]; do
        if ! kill -0 "${SERVER_PIDS[worker]}" 2>/dev/null; then
            echo "RLinf policy worker ${worker} exited during startup" >&2
            tail -n 100 "${RUN_ROOT}/logs_servers/server_worker${worker}_gpu$((worker % PHYSICAL_GPUS)).log" >&2
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for RLinf policy worker ${worker}" >&2
            exit 1
        fi
        sleep 5
    done
done

run_shard() {
    local seed_offset=$1
    local results_subdir=$2
    local videos_subdir=$3
    local logs_subdir=$4
    env \
        STARVLA_PYTHON="${RLINF_PYTHON}" \
        ROBOCASA_PYTHON="${ROBOCASA_PYTHON}" \
        SEED_MANIFEST="${SEED_MANIFEST}" \
        N_EPISODES=25 \
        SEED_OFFSET="${seed_offset}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        PHYSICAL_GPUS="${PHYSICAL_GPUS}" \
        START_POLICY_SERVERS=0 \
        POLICY_TRANSPORT=file \
        SIM_SET_CUDA_VISIBLE_DEVICES=0 \
        RECORD_VIDEO=1 \
        RESULTS_SUBDIR="${results_subdir}" \
        VIDEOS_SUBDIR="${videos_subdir}" \
        LOGS_SUBDIR="${logs_subdir}" \
        SKIP_SUMMARY=1 \
        bash "${STARVLA_ROOT}/examples/Robocasa_tabletop/run_fixed_seed_eval.sh" \
        "${CHECKPOINT}" "${RUN_ROOT}"
}

run_shard 0 results videos logs &
EVAL_PIDS+=("$!")
run_shard 25 results_shard25 videos_shard25 logs_shard25 &
EVAL_PIDS+=("$!")
for pid in "${EVAL_PIDS[@]}"; do
    wait "${pid}"
done
EVAL_PIDS=()

cd "${STARVLA_ROOT}"
"${ROBOCASA_PYTHON}" examples/Robocasa_tabletop/eval_files/merge_fixed_seed_shards.py \
    --run-root "${RUN_ROOT}" --seed-manifest "${SEED_MANIFEST}" --split 25
"${ROBOCASA_PYTHON}" examples/Robocasa_tabletop/eval_files/summarize_fixed_seed_results.py \
    --results-dir "${RUN_ROOT}/results_merged" \
    --seed-manifest "${SEED_MANIFEST}" \
    --output "${RUN_ROOT}/summary.json"
"${ROBOCASA_PYTHON}" examples/Robocasa_tabletop/eval_files/verify_rollout_videos.py \
    --run-root "${RUN_ROOT}" --workers 24 \
    --output "${RUN_ROOT}/video_decode_audit.json"

echo "RLinf fixed-seed evaluation complete: ${RUN_ROOT}"
