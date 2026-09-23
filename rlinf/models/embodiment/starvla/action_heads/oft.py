# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared OFT handlers for rollout and default_forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import (
    resize_images as starvla_resize_images,
)
from torch.distributions.normal import Normal

from ..utils import data_pipeline as data_pipeline_utils
from ..utils.backbone_pipeline import compute_values_from_hidden, run_backbone_pipeline
from ..utils.profile import (
    RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    resolve_action_chunk_len,
    resolve_vlm_interface,
)

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def _build_oft_vlm_inputs(
    starvla_model,
    *,
    num_action_chunks: int,
    examples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Build OFT prompt format by appending action-token placeholders."""
    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]

    from ..utils.vlm_preprocess import get_train_image_size

    train_obs_image_size = get_train_image_size(starvla_model)
    if train_obs_image_size:
        batch_images = starvla_resize_images(
            batch_images, target_size=train_obs_image_size
        )

    chunk_len = resolve_action_chunk_len(
        starvla_model,
        num_action_chunks,
        action_head_name="oft",
    )
    action_token = str(getattr(starvla_model, "action_token", ""))
    action_tokens = action_token * chunk_len
    prompt_suffix = (
        f" Please predict the next {chunk_len} robot actions: "
        f"<action>{action_tokens}<action>."
    )
    instructions = [instruction + prompt_suffix for instruction in instructions]

    qwen_vl_interface = getattr(starvla_model, "qwen_vl_interface", None)
    build_inputs = getattr(qwen_vl_interface, "build_qwenvl_inputs", None)
    # TODO: Whether this fallback is necessary. need further test
    if not callable(build_inputs):
        vlm_interface = resolve_vlm_interface(starvla_model)
        build_inputs = getattr(vlm_interface, "build_qwenvl_inputs", None)
    if not callable(build_inputs):
        raise RuntimeError("VLM interface does not provide 'build_qwenvl_inputs(...)'.")
    return build_inputs(images=batch_images, instructions=instructions)


def _run_oft_backbone_and_head(
    policy: StarVLAForRLActionPrediction,
    *,
    model_inputs: dict[str, torch.Tensor],
    use_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, Normal, torch.Tensor]:
    """Run the shared OFT backbone/action-head path.

    Also returns the action-token queries the head consumed, so a frozen copy of
    the head can be evaluated on the same features without a second backbone
    pass.
    """
    backbone_output = run_backbone_pipeline(
        policy,
        action_head_name="oft",
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    model = policy.starvla_model
    last_hidden = backbone_output["last_hidden"]
    with torch.autocast("cuda", dtype=torch.float32):
        input_ids = model_inputs["input_ids"]
        action_queries = model._gather_action_token_embeddings(
            last_hidden,
            input_ids,
            action_token_id=getattr(model, "action_token_id", None),
        )
        action_model_dtype = next(
            (
                parameter.dtype
                for parameter in model.action_model.parameters()
                if parameter.is_floating_point()
            ),
            action_queries.dtype,
        )
        action_queries = action_queries.to(dtype=action_model_dtype)
        # Call the module rather than its helper method so a separately wrapped
        # FSDP action head executes its pre-forward all-gather hook. The native
        # OFT head's forward delegates to predict_action, so unwrapped numerics
        # remain identical.
        mean_actions = model.action_model(action_queries)

    # Some deployed policies predict a longer horizon than the environment
    # executes before replanning. Keep all query tokens in the backbone, but
    # compute sampling and PPO likelihoods only for actions that affect the
    # environment.
    mean_actions = mean_actions[:, : policy.num_executed_action_chunks]
    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))
    return mean_actions, last_hidden, dist, action_queries


def run_default_forward_oft(
    policy: StarVLAForRLActionPrediction,
    *,
    data: dict[str, torch.Tensor],
    compute_logprobs: bool,
    compute_entropy: bool,
    compute_values: bool,
    use_cache: bool,
) -> dict[str, torch.Tensor | None]:
    """Compute training-time PPO terms for the OFT action head."""
    data_pipeline_utils.forward_input_check(data)

    model_inputs = data_pipeline_utils.collect_default_forward_model_inputs(
        data,
        skip_keys={
            "action",
            "action_tokens",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
        },
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )

    mean_actions, last_hidden, dist, _ = _run_oft_backbone_and_head(
        policy,
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    action_for_logprob = (
        data_pipeline_utils.fetch_action_for_logprob_for_default_forward(
            policy,
            data=data,
            reference=mean_actions,
        )
    )

    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }
    if compute_logprobs:
        result["logprobs"] = dist.log_prob(action_for_logprob).to(dtype=torch.float32)
    if compute_entropy:
        result["entropy"] = dist.entropy().to(dtype=torch.float32)
    if compute_values:
        result["values"] = compute_values_from_hidden(
            value_head=policy.value_head,
            hidden=last_hidden,
            attention_mask=model_inputs.get("attention_mask"),
        )
    return result


def run_rollout_oft(
    policy: StarVLAForRLActionPrediction,
    *,
    examples: list[dict[str, Any]],
    mode: str,
    calculate_logprobs: bool,
    calculate_values: bool,
    sampling_kwargs: dict[str, Any],
    env_obs: dict[str, Any],
) -> dict[str, Any]:
    """Roll out the OFT action head and pack replay caches for training."""
    del env_obs

    model_inputs = _build_oft_vlm_inputs(
        starvla_model=policy.starvla_model,
        num_action_chunks=policy.num_action_chunks,
        examples=examples,
    )
    mean_actions, last_hidden, dist, _ = _run_oft_backbone_and_head(
        policy,
        model_inputs=model_inputs,
        use_cache=False,
    )
    sample_actions = bool(sampling_kwargs.get("do_sample")) and mode == "train"
    edit_policy = getattr(policy, "edit_policy", None)
    critic_gate_open = getattr(policy, "critic_gate_open", None)
    # Both training and eval rollouts wait for the critic gate. The Q head
    # trains every step regardless of the actor gate, so its hidden layers
    # rank resampled candidates by noise long before that ranking carries
    # real signal, and best-of-N over that noise systematically steers
    # rollout away from the pretrained policy's own well-calibrated
    # behaviour. A first run that used best-of-N unconditionally from step 0
    # averaged sampled success 0.08 over 120 steps, against 0.35-0.5 in the
    # same early phase of the anchor-based recipes that just sampled the base
    # distribution. Eval follows the same gate so the reported number is the
    # policy training actually deploys, not a Q-ranked pick by a critic that
    # training itself has judged untrustworthy.
    use_best_of_n = edit_policy is not None and bool(
        critic_gate_open is not None and critic_gate_open.item()
    )
    if use_best_of_n:
        # EXPO-FT-style recipe: the executed action is whichever of the base
        # head's own resampled candidates (or their edited versions) the
        # twin Q heads currently prefer, not a plain sample from the base
        # distribution. See StarVLAForRLActionPrediction._select_best_of_n.
        # predict_action_batch unnormalizes whatever is stored under
        # "normalized_actions" exactly once, so the value handed back here
        # must be in the policy's own normalized units, not environment
        # units (see the return-value docstring on _select_best_of_n).
        _, executed_actions, _, _ = policy._select_best_of_n(
            mean_actions,
            last_hidden,
            dist,
            model_inputs,
            num_candidates=policy.expo_num_candidates,
            mode=mode,
        )
    else:
        # Either no edit policy (plain SAC/PPO/GRPO), or one exists but the
        # critic is not informative yet: fall back to sampling the base
        # policy's own distribution, exactly as every recipe without
        # best-of-N always has.
        executed_actions = dist.sample() if sample_actions else mean_actions

    prev_logprobs = None
    prev_values = None
    if calculate_logprobs:
        # SAC does not consume prev_logprobs (it recomputes everything from
        # raw replay observations); the base distribution has no meaningful
        # density for a best-of-N pick that may not even be one of its raw
        # samples, so this is a harmless constant placeholder that satisfies
        # the storage contract without claiming a probability that isn't one.
        prev_logprobs = dist.log_prob(
            executed_actions if not use_best_of_n else mean_actions
        ).to(dtype=torch.float32)
    if calculate_values:
        prev_values = compute_values_from_hidden(
            value_head=policy.value_head,
            hidden=last_hidden,
            attention_mask=model_inputs.get("attention_mask"),
        )

    return {
        "output": {
            "normalized_actions": data_pipeline_utils.tensor_to_numpy_compatible(
                executed_actions
            )
        },
        "model_inputs": model_inputs,
        "prev_logprobs": prev_logprobs,
        "prev_values": prev_values,
        "extra_forward_inputs": {
            "action_for_logprob": executed_actions.to(dtype=torch.float32)
        },
        "state": None,
    }
