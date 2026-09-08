#!/usr/bin/env bash

set -euo pipefail

RLINF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAIN_ROOT=${TRAIN_ROOT:-/data/dengyixuan/wyz/experiments/rlinf_starvla_robocasa_gr1_grpo_actionhead_fp32}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-cup_to_drawer_close_grpo_actionhead_fp32}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/data/dengyixuan/wyz/models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt}
BASE_EVAL_ROOT=${BASE_EVAL_ROOT:-/data/dengyixuan/wyz/experiments/rlinf_starvla_robocasa_gr1_base_official_repro_fixed50_eval_20260820}
EVAL_MODEL_BASE_CHECKPOINT=${EVAL_MODEL_BASE_CHECKPOINT:-/data/dengyixuan/wyz/models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt}
INITIAL_CKPT_PATH=${INITIAL_CKPT_PATH:-}
EVAL_INTERVAL=${EVAL_INTERVAL:-2}
TARGET_MAX_STEPS=${TARGET_MAX_STEPS:-4}
ACTOR_GLOBAL_BATCH_SIZE=${ACTOR_GLOBAL_BATCH_SIZE:-512}
ACTOR_MICRO_BATCH_SIZE=${ACTOR_MICRO_BATCH_SIZE:-1}
ROLLOUT_INFERENCE_MICRO_BATCH_SIZE=${ROLLOUT_INFERENCE_MICRO_BATCH_SIZE:-1}
ACTOR_LOGPROB_RECOMPUTE_MICRO_BATCH_SIZE=${ACTOR_LOGPROB_RECOMPUTE_MICRO_BATCH_SIZE:-1}
REFERENCE_FORWARD_MICRO_BATCH_SIZE=${REFERENCE_FORWARD_MICRO_BATCH_SIZE:-1}
POST_UPDATE_LOGPROB_AUDIT_MICRO_BATCH_SIZE=${POST_UPDATE_LOGPROB_AUDIT_MICRO_BATCH_SIZE:-1}
ACTOR_LR=${ACTOR_LR:-1.0e-9}
ACTOR_INITIAL_LOGSTD=${ACTOR_INITIAL_LOGSTD:--3.5}
GROUP_SIZE=${GROUP_SIZE:-8}
ROLLOUT_EPOCH=${ROLLOUT_EPOCH:-4}
ADV_TYPE=${ADV_TYPE:-grpo}
POSITIVE_ADVANTAGES_ONLY=${POSITIVE_ADVANTAGES_ONLY:-false}
PRIORITIZE_POSITIVE_ADVANTAGE_SAMPLES=${PRIORITIZE_POSITIVE_ADVANTAGE_SAMPLES:-false}
TARGET_KL=${TARGET_KL:-0.02}
REFERENCE_TARGET_KL=${REFERENCE_TARGET_KL:-0.02}
MAX_OPTIMIZER_STEPS_PER_UPDATE=${MAX_OPTIMIZER_STEPS_PER_UPDATE:-5}
RECOMPUTE_ROLLOUT_LOGPROBS_ON_ACTOR=${RECOMPUTE_ROLLOUT_LOGPROBS_ON_ACTOR:-true}
AUDIT_POST_UPDATE_LOGPROBS=${AUDIT_POST_UPDATE_LOGPROBS:-true}
TRAIN_SAMPLING_SEED=${TRAIN_SAMPLING_SEED:-20260821}
TRAINABLE_PARAMETER_PREFIXES=${TRAINABLE_PARAMETER_PREFIXES:-'[starvla_model.action_model]'}
FSDP_USE_ORIG_PARAMS=${FSDP_USE_ORIG_PARAMS:-true}
PATCH_INIT_SYNC_ENABLED=${PATCH_INIT_SYNC_ENABLED:-true}
PATCH_INIT_SYNC_PREFIXES=${PATCH_INIT_SYNC_PREFIXES:-'[starvla_model.action_model,actor_logstd]'}
SUBTASK_REWARD_SHAPING_ENABLED=${SUBTASK_REWARD_SHAPING_ENABLED:-false}
SUBTASK_GRASP_REWARD=${SUBTASK_GRASP_REWARD:-0.1}
SUBTASK_IN_DRAWER_REWARD=${SUBTASK_IN_DRAWER_REWARD:-0.5}
SUBTASK_SUCCESS_REWARD=${SUBTASK_SUCCESS_REWARD:-1.0}
WANDB_RUN_ID=${WANDB_RUN_ID:-robocasa-gr1-actionhead-fp32}
# Keep this long-running local experiment offline unless the user explicitly
# authorizes external W&B uploads.
LOGGER_BACKENDS_OVERRIDE=${LOGGER_BACKENDS_OVERRIDE:-'[]'}
REGISTRY_PYTHON=${REGISTRY_PYTHON:-/data/dengyixuan/wyz/.venvs/robocasa-gr1-cve/bin/python}
AUDIT_PYTHON=${AUDIT_PYTHON:-/data/dengyixuan/wyz/.venvs/starvla-cve/bin/python}
UPDATE_HEALTH_AUDIT=${UPDATE_HEALTH_AUDIT:-true}
MIN_CHANGED_FRACTION=${MIN_CHANGED_FRACTION:-0.001}
MIN_MIXED_GROUPS=${MIN_MIXED_GROUPS:-16}
MAX_ROLLOUT_ACTOR_LOGPROB_DELTA=${MAX_ROLLOUT_ACTOR_LOGPROB_DELTA:-1.0e-6}
MIN_POST_UPDATE_LOGPROB_MEAN_ABS_DELTA=${MIN_POST_UPDATE_LOGPROB_MEAN_ABS_DELTA:-1.0e-6}
MAX_POST_UPDATE_LOGPROB_MEAN_ABS_DELTA=${MAX_POST_UPDATE_LOGPROB_MEAN_ABS_DELTA:-0.02}
MAX_POST_UPDATE_RATIO_MEAN_ABS_DELTA=${MAX_POST_UPDATE_RATIO_MEAN_ABS_DELTA:-0.02}

CHECKPOINT_ROOT=${TRAIN_ROOT}/${EXPERIMENT_NAME}/checkpoints
EVALUATION_ROOT=${TRAIN_ROOT}/official_fixed50_evaluations
REGISTRY=${TRAIN_ROOT}/fixed50_eval_registry.json
HEALTH_ROOT=${TRAIN_ROOT}/update_health
METRICS_PATH=${TRAIN_ROOT}/metrics/training_step_metrics.jsonl
ROLLOUTS_PATH=${TRAIN_ROOT}/rollout_tables/train_seed_rollouts.jsonl
mkdir -p "${CHECKPOINT_ROOT}" "${EVALUATION_ROOT}" "${HEALTH_ROOT}"

if ((EVAL_INTERVAL <= 0 || TARGET_MAX_STEPS <= 0 || ROLLOUT_EPOCH <= 0)); then
    echo "EVAL_INTERVAL, TARGET_MAX_STEPS, and ROLLOUT_EPOCH must be positive" >&2
    exit 1
fi
if ((GROUP_SIZE <= 1 || 32 % GROUP_SIZE != 0)); then
    echo "GROUP_SIZE must be greater than 1 and divide the 32 training environments" >&2
    exit 1
fi
if [[ -n "${INITIAL_CKPT_PATH}" && ! -f "${INITIAL_CKPT_PATH}" ]]; then
    echo "INITIAL_CKPT_PATH does not exist: ${INITIAL_CKPT_PATH}" >&2
    exit 1
fi

initial_ckpt_args=()
if [[ -n "${INITIAL_CKPT_PATH}" ]]; then
    initial_ckpt_args+=("runner.ckpt_path=${INITIAL_CKPT_PATH}")
fi

if [[ ! -f "${BASE_EVAL_ROOT}/summary.json" ]]; then
    BASE_CHECKPOINT="${EVAL_MODEL_BASE_CHECKPOINT}" NUM_SHARDS=8 bash \
        "${RLINF_ROOT}/examples/embodiment/run_robocasa_gr1_cup_drawer_official_fixed_seed_eval.sh" \
        "${BASE_CHECKPOINT}" "${BASE_EVAL_ROOT}"
fi
"${REGISTRY_PYTHON}" \
    "${RLINF_ROOT}/examples/embodiment/update_robocasa_gr1_eval_registry.py" \
    --registry "${REGISTRY}" --step 0 --checkpoint "${BASE_CHECKPOINT}" \
    --eval-root "${BASE_EVAL_ROOT}"

current_step=0
for checkpoint_dir in "${CHECKPOINT_ROOT}"/global_step_*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    checkpoint_step=${checkpoint_dir##*_}
    if [[ "${checkpoint_step}" =~ ^[0-9]+$ ]] && ((checkpoint_step > current_step)); then
        current_step=${checkpoint_step}
    fi
done

evaluate_checkpoint() {
    local step=$1
    local checkpoint=${CHECKPOINT_ROOT}/global_step_${step}/actor/model_state_dict/full_weights.pt
    local eval_root=${EVALUATION_ROOT}/global_step_${step}
    if [[ ! -f "${eval_root}/summary.json" ]]; then
        BASE_CHECKPOINT="${EVAL_MODEL_BASE_CHECKPOINT}" NUM_SHARDS=8 bash \
            "${RLINF_ROOT}/examples/embodiment/run_robocasa_gr1_cup_drawer_official_fixed_seed_eval.sh" \
            "${checkpoint}" "${eval_root}"
    fi
    "${REGISTRY_PYTHON}" \
        "${RLINF_ROOT}/examples/embodiment/update_robocasa_gr1_eval_registry.py" \
        --registry "${REGISTRY}" --step "${step}" --checkpoint "${checkpoint}" \
        --eval-root "${eval_root}"
}

audit_checkpoint_update() {
    local step=$1
    local before_checkpoint=$2
    local checkpoint=${CHECKPOINT_ROOT}/global_step_${step}/actor/model_state_dict/full_weights.pt
    local output=${HEALTH_ROOT}/global_step_${step}.json
    if [[ "${UPDATE_HEALTH_AUDIT}" != "true" ]]; then
        return
    fi
    "${AUDIT_PYTHON}" \
        "${RLINF_ROOT}/examples/embodiment/audit_robocasa_gr1_grpo_update.py" \
        --before-checkpoint "${before_checkpoint}" \
        --after-checkpoint "${checkpoint}" \
        --metrics "${METRICS_PATH}" \
        --rollouts "${ROLLOUTS_PATH}" \
        --step "${step}" \
        --output "${output}" \
        --group-size "${GROUP_SIZE}" \
        --expected-trajectories "$((32 * ROLLOUT_EPOCH))" \
        --expected-optimizer-steps "${MAX_OPTIMIZER_STEPS_PER_UPDATE}" \
        --expected-logstd "${ACTOR_INITIAL_LOGSTD}" \
        --min-mixed-groups "${MIN_MIXED_GROUPS}" \
        --min-changed-fraction "${MIN_CHANGED_FRACTION}" \
        --max-rollout-actor-logprob-delta "${MAX_ROLLOUT_ACTOR_LOGPROB_DELTA}" \
        --min-post-update-logprob-mean-abs-delta \
        "${MIN_POST_UPDATE_LOGPROB_MEAN_ABS_DELTA}" \
        --max-post-update-logprob-mean-abs-delta \
        "${MAX_POST_UPDATE_LOGPROB_MEAN_ABS_DELTA}" \
        --max-post-update-ratio-mean-abs-delta \
        "${MAX_POST_UPDATE_RATIO_MEAN_ABS_DELTA}"
}

if ((current_step > 0)); then
    evaluate_checkpoint "${current_step}"
fi

while ((current_step < TARGET_MAX_STEPS)); do
    if ((current_step > 0)); then
        before_checkpoint=${CHECKPOINT_ROOT}/global_step_${current_step}/actor/model_state_dict/full_weights.pt
    elif [[ -n "${INITIAL_CKPT_PATH}" ]]; then
        before_checkpoint=${INITIAL_CKPT_PATH}
    else
        before_checkpoint=${BASE_CHECKPOINT}
    fi
    next_step=$((current_step + EVAL_INTERVAL))
    if ((next_step > TARGET_MAX_STEPS)); then
        next_step=${TARGET_MAX_STEPS}
    fi
    resume_args=()
    if ((current_step > 0)); then
        resume_args+=(
            "runner.resume_dir=${CHECKPOINT_ROOT}/global_step_${current_step}"
        )
    fi
    echo "Training reference-KL GRPO from step ${current_step} to ${next_step}"
    echo "Update controls: adv_type=${ADV_TYPE}, positive_advantages_only=${POSITIVE_ADVANTAGES_ONLY}, prioritize_positive_samples=${PRIORITIZE_POSITIVE_ADVANTAGE_SAMPLES}, group_size=${GROUP_SIZE}, rollout_epoch=${ROLLOUT_EPOCH}, global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}, actor_micro_batch_size=${ACTOR_MICRO_BATCH_SIZE}, rollout_micro_batch_size=${ROLLOUT_INFERENCE_MICRO_BATCH_SIZE}, recompute_micro_batch_size=${ACTOR_LOGPROB_RECOMPUTE_MICRO_BATCH_SIZE}, reference_micro_batch_size=${REFERENCE_FORWARD_MICRO_BATCH_SIZE}, post_audit_micro_batch_size=${POST_UPDATE_LOGPROB_AUDIT_MICRO_BATCH_SIZE}, lr=${ACTOR_LR}, target_kl=${TARGET_KL}, reference_target_kl=${REFERENCE_TARGET_KL}, max_optimizer_steps=${MAX_OPTIMIZER_STEPS_PER_UPDATE}, recompute_actor_logprobs=${RECOMPUTE_ROLLOUT_LOGPROBS_ON_ACTOR}, train_sampling_seed=${TRAIN_SAMPLING_SEED}"
    echo "Parameter controls: trainable_prefixes=${TRAINABLE_PARAMETER_PREFIXES}, use_orig_params=${FSDP_USE_ORIG_PARAMS}, initial_logstd=${ACTOR_INITIAL_LOGSTD}, init_sync=${PATCH_INIT_SYNC_ENABLED}:${PATCH_INIT_SYNC_PREFIXES}"
    echo "Initialization checkpoint: ${INITIAL_CKPT_PATH:-actor.model.model_path}"
    echo "Reward controls: shaped=${SUBTASK_REWARD_SHAPING_ENABLED}, grasp=${SUBTASK_GRASP_REWARD}, in_drawer=${SUBTASK_IN_DRAWER_REWARD}, success=${SUBTASK_SUCCESS_REWARD}"
    ROBOCASA_GR1_GRPO_LOG_ROOT="${TRAIN_ROOT}" \
    WANDB_RUN_ID="${WANDB_RUN_ID}" WANDB_RESUME=allow \
    WANDB_NAME="${EXPERIMENT_NAME}" \
        bash "${RLINF_ROOT}/examples/embodiment/run_robocasa_gr1_cup_drawer_grpo.sh" \
        "runner.logger.log_path=${TRAIN_ROOT}" \
        "runner.logger.experiment_name=${EXPERIMENT_NAME}" \
        "runner.logger.logger_backends=${LOGGER_BACKENDS_OVERRIDE}" \
        "runner.max_epochs=${TARGET_MAX_STEPS}" \
        "runner.max_steps=${next_step}" \
        "runner.save_interval=${EVAL_INTERVAL}" \
        "actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}" \
        "actor.micro_batch_size=${ACTOR_MICRO_BATCH_SIZE}" \
        "rollout.inference_micro_batch_size=${ROLLOUT_INFERENCE_MICRO_BATCH_SIZE}" \
        "actor.optim.lr=${ACTOR_LR}" \
        "actor.model.initial_logstd=${ACTOR_INITIAL_LOGSTD}" \
        "algorithm.group_size=${GROUP_SIZE}" \
        "algorithm.rollout_epoch=${ROLLOUT_EPOCH}" \
        "algorithm.adv_type=${ADV_TYPE}" \
        "algorithm.positive_advantages_only=${POSITIVE_ADVANTAGES_ONLY}" \
        "algorithm.prioritize_positive_advantage_samples=${PRIORITIZE_POSITIVE_ADVANTAGE_SAMPLES}" \
        "actor.trainable_parameter_prefixes=${TRAINABLE_PARAMETER_PREFIXES}" \
        "actor.fsdp_config.use_orig_params=${FSDP_USE_ORIG_PARAMS}" \
        "weight_syncer.patch.init_sync.enabled=${PATCH_INIT_SYNC_ENABLED}" \
        "weight_syncer.patch.init_sync.prefixes=${PATCH_INIT_SYNC_PREFIXES}" \
        "env.train.subtask_reward_shaping.enabled=${SUBTASK_REWARD_SHAPING_ENABLED}" \
        "env.train.subtask_reward_shaping.grasp_object=${SUBTASK_GRASP_REWARD}" \
        "env.train.subtask_reward_shaping.obj_in_drawer=${SUBTASK_IN_DRAWER_REWARD}" \
        "env.train.subtask_reward_shaping.success=${SUBTASK_SUCCESS_REWARD}" \
        "algorithm.target_kl=${TARGET_KL}" \
        "algorithm.reference_target_kl=${REFERENCE_TARGET_KL}" \
        "algorithm.max_optimizer_steps_per_update=${MAX_OPTIMIZER_STEPS_PER_UPDATE}" \
        "algorithm.recompute_rollout_logprobs_on_actor=${RECOMPUTE_ROLLOUT_LOGPROBS_ON_ACTOR}" \
        "algorithm.actor_logprob_recompute_micro_batch_size=${ACTOR_LOGPROB_RECOMPUTE_MICRO_BATCH_SIZE}" \
        "algorithm.reference_forward_micro_batch_size=${REFERENCE_FORWARD_MICRO_BATCH_SIZE}" \
        "algorithm.audit_post_update_logprobs=${AUDIT_POST_UPDATE_LOGPROBS}" \
        "algorithm.post_update_logprob_audit_micro_batch_size=${POST_UPDATE_LOGPROB_AUDIT_MICRO_BATCH_SIZE}" \
        "rollout.training_sampling_seed=${TRAIN_SAMPLING_SEED}" \
        "${initial_ckpt_args[@]}" \
        "${resume_args[@]}"

    checkpoint=${CHECKPOINT_ROOT}/global_step_${next_step}/actor/model_state_dict/full_weights.pt
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Training exited without expected checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    audit_checkpoint_update "${next_step}" "${before_checkpoint}"
    current_step=${next_step}
    evaluate_checkpoint "${current_step}"
done

echo "Train/eval loop reached step ${TARGET_MAX_STEPS}: ${TRAIN_ROOT}"
