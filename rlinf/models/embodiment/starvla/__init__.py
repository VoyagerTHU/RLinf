# Copyright 2026 The RLinf Authors.
# Signed-off-by: The RLinf Authors.
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

"""starVLA embodied policy wrapper for RLinf.

This module exposes 'get_model', which loads a starVLA checkpoint and returns a
'StarVLAForRLActionPrediction' instance compatible with RLinf.
"""

from __future__ import annotations

import os

import torch
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

from .starvla_action_model import StarVLAForRLActionPrediction
from .utils.profile import resolve_vlm_interface

_ACTION_MODEL_PRECISION_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_DEFAULT_QWEN3_VL_LORA_TARGET_MODULES = "all-linear"


def cast_action_model_precision(
    starvla_model: torch.nn.Module,
    precision: str | torch.dtype | None,
) -> torch.dtype | None:
    """Cast only the StarVLA action model to an explicitly requested dtype.

    Keeping the action model in FP32 lets small optimizer updates accumulate
    without converting the much larger VLM backbone away from its native BF16
    checkpoint dtype.
    """
    if precision is None:
        return None
    if isinstance(precision, torch.dtype):
        dtype = precision
    else:
        normalized = str(precision).strip().lower()
        try:
            dtype = _ACTION_MODEL_PRECISION_TO_DTYPE[normalized]
        except KeyError as error:
            raise ValueError(
                "action_model_precision must be one of "
                f"{sorted(_ACTION_MODEL_PRECISION_TO_DTYPE)}, got {precision!r}"
            ) from error

    action_model = getattr(starvla_model, "action_model", None)
    if not isinstance(action_model, torch.nn.Module):
        raise ValueError(
            "action_model_precision was provided, but the loaded StarVLA policy "
            "does not expose an nn.Module at 'action_model'"
        )
    action_model.to(dtype=dtype)
    return dtype


def apply_qwen3_vl_lora(
    policy: StarVLAForRLActionPrediction,
    cfg: DictConfig,
) -> StarVLAForRLActionPrediction:
    """Add LoRA adapters to StarVLA's Qwen3-VL subtree.

    The complete policy is frozen before adapter injection. This is important
    for StarVLA because wrapping only the Qwen module does not otherwise change
    ``requires_grad`` on the sibling action head. Set
    ``lora_train_action_head=true`` to explicitly re-enable the complete action
    head alongside the Qwen adapters; the exploration log standard deviation
    and Qwen base weights remain frozen.

    Args:
        policy: Loaded RLinf StarVLA policy wrapper.
        cfg: Model config containing ``lora_rank`` and optional
            ``lora_target_modules`` / ``lora_train_action_head``.

    Returns:
        The same policy with its Qwen3-VL model replaced by a PEFT model.

    Raises:
        ValueError: If the expected Qwen3-VL subtree or requested action head
            is missing, no adapters are trainable, or a parameter outside the
            requested scope is trainable.
    """
    from peft import LoraConfig, get_peft_model

    qwen_vl_interface = getattr(policy.starvla_model, "qwen_vl_interface", None)
    qwen_model = getattr(qwen_vl_interface, "model", None)
    if not isinstance(qwen_model, torch.nn.Module):
        raise ValueError(
            "StarVLA Qwen3-VL LoRA requires an nn.Module at "
            "starvla_model.qwen_vl_interface.model"
        )

    rank = int(getattr(cfg, "lora_rank", 32))
    if rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {rank}")
    target_modules = getattr(
        cfg,
        "lora_target_modules",
        _DEFAULT_QWEN3_VL_LORA_TARGET_MODULES,
    )
    if not isinstance(target_modules, str):
        target_modules = [str(module_name) for module_name in target_modules]
        if not target_modules:
            raise ValueError("lora_target_modules must not be empty")

    # PEFT freezes the wrapped Qwen base, but it cannot freeze sibling modules
    # such as StarVLA's OFT action head. Freeze the whole policy explicitly.
    policy.requires_grad_(False)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    qwen_vl_interface.model = get_peft_model(qwen_model, lora_config)

    train_action_head = bool(getattr(cfg, "lora_train_action_head", False))
    action_model = getattr(policy.starvla_model, "action_model", None)
    if train_action_head:
        if not isinstance(action_model, torch.nn.Module):
            raise ValueError(
                "lora_train_action_head=true requires an nn.Module at "
                "starvla_model.action_model"
            )
        action_model.requires_grad_(True)

    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]
    qwen_prefix = "starvla_model.qwen_vl_interface.model."
    action_prefix = "starvla_model.action_model."
    unexpected_names = [
        name
        for name in trainable_names
        if not name.startswith(qwen_prefix)
        and not (train_action_head and name.startswith(action_prefix))
    ]
    if not trainable_names:
        raise ValueError("Qwen3-VL LoRA injection produced no trainable parameters")
    if unexpected_names:
        raise ValueError(
            "StarVLA LoRA left parameters trainable outside the requested "
            f"Qwen/action-head scope: {unexpected_names[:8]}"
        )

    qwen_trainable_names = [
        name for name in trainable_names if name.startswith(qwen_prefix)
    ]
    non_adapter_qwen_names = [
        name
        for name in qwen_trainable_names
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    if non_adapter_qwen_names:
        raise ValueError(
            "Qwen3-VL base parameters unexpectedly remained trainable: "
            f"{non_adapter_qwen_names[:8]}"
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    get_logger().info(
        "Enabled Qwen3-VL LoRA: rank=%d, targets=%s, "
        "trainable=%d/%d (%.4f%%), tensors=%d; action_head_trainable=%s",
        rank,
        target_modules,
        trainable_parameters,
        total_parameters,
        100.0 * trainable_parameters / total_parameters,
        len(trainable_names),
        train_action_head,
    )
    return policy


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> StarVLAForRLActionPrediction:
    """Load a starVLA checkpoint and wrap it into RLinf's embodied policy interface.

    Args:
        cfg: Model config. Must specify a starVLA checkpoint path via
            'actor.model.model_path'.
        torch_dtype: Optional torch dtype to cast the loaded model to.

    Returns:
        A 'StarVLAForRLActionPrediction' instance.

    Raises:
        ValueError: If no checkpoint path is provided in 'cfg'.
    """
    logger = get_logger()
    model_path = getattr(cfg, "model_path", None)
    if model_path is None:
        raise ValueError(
            "starVLA requires 'actor.model.model_path'. Set it to a .pt checkpoint inside "
            "a starVLA run directory."
        )
    if model_path.endswith(".pt"):
        assert os.path.exists(model_path), (
            f"Checkpoint path {model_path} does not exist"
        )
        ckpt_path = model_path
    else:
        # Try to find the latest checkpoint in the checkpoints directory
        model_path = os.path.join(os.fspath(model_path), "checkpoints")
        assert os.path.exists(model_path), (
            f"Checkpoint path {model_path} does not exist"
        )
        ckpt_files = os.listdir(model_path)
        ckpt_files = sorted([f for f in ckpt_files if f.endswith(".pt")])
        assert len(ckpt_files) > 0, f"No checkpoint files found in {model_path}"
        ckpt_path = os.path.join(model_path, ckpt_files[-1])
    logger.info(f"Loading checkpoint file: {ckpt_path}")

    try:
        from starVLA.model.framework.base_framework import baseframework
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "starVLA is required to load starVLA checkpoints. Please install starVLA and "
            "ensure it is importable as the Python module 'starVLA'."
        ) from e

    starvla_model = baseframework.from_pretrained(ckpt_path)

    # Check early whether the loaded model provides a compatible interface.
    resolve_vlm_interface(starvla_model)

    # 'framework_name' is optional but helps infer the expected wiring for some checkpoints.
    starvla_cfg = getattr(cfg, "starvla", None)
    framework_name = getattr(starvla_cfg, "framework_name", None)
    if framework_name is not None:
        framework_name = str(framework_name).strip()
    if framework_name:
        starvla_model.framework_name = framework_name

    enable_state_input = getattr(starvla_cfg, "enable_state_input", None)
    if enable_state_input is None:
        enable_state_input = getattr(cfg, "enable_state_input", True)

    # Cast the full checkpoint first, then optionally restore only the action
    # model to a higher precision. This ordering is important: doing the full
    # cast last would silently undo action_model_precision.
    if torch_dtype is not None:
        starvla_model = starvla_model.to(dtype=torch_dtype)
    action_model_dtype = cast_action_model_precision(
        starvla_model,
        getattr(cfg, "action_model_precision", None),
    )
    if action_model_dtype is not None:
        action_model_parameters = sum(
            parameter.numel() for parameter in starvla_model.action_model.parameters()
        )
        logger.info(
            "Keeping StarVLA action_model in %s (%d parameters)",
            action_model_dtype,
            action_model_parameters,
        )

    return StarVLAForRLActionPrediction(
        starvla_model=starvla_model,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=getattr(cfg, "add_value_head", True),
        unnorm_key=getattr(cfg, "unnorm_key", None),
        action_stats_source=getattr(cfg, "action_stats_source", "minmax"),
        enable_state_input=enable_state_input,
        policy_setup=getattr(cfg, "policy_setup", None),
        initial_logstd=getattr(cfg, "initial_logstd", -2.5),
        trainable_logstd=getattr(cfg, "trainable_logstd", True),
        num_executed_action_chunks=getattr(cfg, "num_executed_action_chunks", None),
        rollout_prompt_seq_len=getattr(cfg, "rollout_prompt_seq_len", None),
        add_q_head=getattr(cfg, "add_q_head", False),
        num_q_heads=getattr(cfg, "num_q_heads", 2),
        q_hidden_dims=tuple(getattr(cfg, "q_hidden_dims", (512, 256))),
        sac_task_description=getattr(cfg, "sac_task_description", None),
    )


__all__ = [
    "StarVLAForRLActionPrediction",
    "apply_qwen3_vl_lora",
    "cast_action_model_precision",
    "get_model",
]
