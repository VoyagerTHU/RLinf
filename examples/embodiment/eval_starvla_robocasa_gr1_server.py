"""Serve a StarVLA or RLinf-trained StarVLA checkpoint for official GR1 eval.

The RoboCasa-GR1 simulator runs in its official environment. Requests and
responses use atomic msgpack files so GPU inference and EGL simulation remain
in separate processes. RLinf ``full_weights.pt`` files are loaded on top of the
original StarVLA checkpoint, while inference and action unnormalization retain
the official StarVLA evaluation path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from deployment.model_server.tools import msgpack_numpy
from omegaconf import OmegaConf

from rlinf.models.embodiment.starvla import apply_qwen3_vl_lora, get_model

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_CHECKPOINT = Path(
    "/data/dengyixuan/wyz/models/StarVLA/Qwen3-VL-OFT-Robocasa/"
    "checkpoints/steps_90000_pytorch_model.pt"
)


def infer_action_model_precision(
    state_dict: dict[str, torch.Tensor],
) -> str | None:
    """Infer a uniform floating dtype for an RLinf action-model checkpoint."""
    prefixes = ("starvla_model.action_model.", "action_model.")
    dtypes = {
        value.dtype
        for key, value in state_dict.items()
        if key.startswith(prefixes)
        and isinstance(value, torch.Tensor)
        and value.is_floating_point()
    }
    if not dtypes:
        return None
    if len(dtypes) != 1:
        raise ValueError(
            f"Action-model checkpoint has mixed dtypes: {sorted(map(str, dtypes))}"
        )
    dtype = next(iter(dtypes))
    precision_by_dtype = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }
    if dtype not in precision_by_dtype:
        raise ValueError(f"Unsupported action-model checkpoint dtype: {dtype}")
    return precision_by_dtype[dtype]


def infer_qwen3_vl_lora_rank(
    state_dict: dict[str, torch.Tensor],
) -> int | None:
    """Infer and validate Qwen3-VL-only LoRA adapters in a full checkpoint."""
    lora_a_suffix = ".lora_A.default.weight"
    lora_keys = [key for key in state_dict if ".lora_A." in key or ".lora_B." in key]
    if not lora_keys:
        return None

    qwen_prefix = "starvla_model.qwen_vl_interface.model."
    unexpected_keys = [key for key in lora_keys if not key.startswith(qwen_prefix)]
    if unexpected_keys:
        raise ValueError(
            "RLinf checkpoint contains LoRA weights outside Qwen3-VL: "
            f"{unexpected_keys[:8]}"
        )

    ranks = {
        int(value.shape[0])
        for key, value in state_dict.items()
        if key.endswith(lora_a_suffix)
        and isinstance(value, torch.Tensor)
        and value.ndim >= 2
    }
    if not ranks:
        raise ValueError(
            "Checkpoint contains LoRA keys but no rank-bearing lora_A weights"
        )
    if len(ranks) != 1:
        raise ValueError(f"Checkpoint contains mixed LoRA ranks: {sorted(ranks)}")
    return next(iter(ranks))


def remove_training_only_value_head(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove PPO/SAC critic tensors before deterministic policy evaluation.

    The value head (PPO) and twin Q heads (SAC) are trained and checkpointed
    with the actor, but neither is used to choose actions.  The
    official-compatible evaluator therefore builds the inference policy without
    them.  Matching whole critic subtrees by prefix keeps strict loading enabled
    for every tensor that can affect actions while staying robust to the head's
    internal layout (``value_head.weight`` for a bare linear head,
    ``value_head.proj.weight`` once the head normalizes its input).
    """
    critic_prefixes = (
        "value_head.",
        "model.value_head.",
        "q_head.",
        "model.q_head.",
    )
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(critic_prefixes)
    }


def predict_normalized_actions(policy, examples: list[dict]) -> np.ndarray:
    """Run the same deterministic OFT path used by RLinf actor replay."""
    if policy.action_head_type != "oft":
        return np.asarray(
            policy.starvla_model.predict_action(examples=examples, do_sample=False)[
                "normalized_actions"
            ]
        )

    from rlinf.models.embodiment.starvla.action_heads.oft import run_rollout_oft

    payload = run_rollout_oft(
        policy,
        examples=examples,
        env_obs={},
        mode="eval",
        calculate_logprobs=False,
        calculate_values=False,
        sampling_kwargs={"do_sample": False},
    )
    return np.asarray(payload["output"]["normalized_actions"])


def blend_normalized_actions(
    actions_a: np.ndarray,
    actions_b: np.ndarray,
    weight_b: float,
) -> np.ndarray:
    """Interpolate two policies in their shared normalized GR1 action space."""
    if not 0.0 <= weight_b <= 1.0:
        raise ValueError(f"ensemble_weight_b must be in [0, 1], got {weight_b}")
    actions_a = np.asarray(actions_a)
    actions_b = np.asarray(actions_b)
    if actions_a.shape != actions_b.shape:
        raise ValueError(
            f"Ensemble action shapes differ: {actions_a.shape} != {actions_b.shape}"
        )
    actions_b = actions_b.astype(actions_a.dtype, copy=False)
    weight = np.asarray(weight_b, dtype=actions_a.dtype)
    return (actions_a + (actions_b - actions_a) * weight).astype(
        actions_a.dtype, copy=False
    )


def query_to_env_obs(query: dict) -> dict:
    """Convert the official StarVLA request schema to RLinf env observations."""
    examples = query.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Expected a non-empty query['examples'] list")
    main_images = []
    task_descriptions = []
    states = []
    has_state = True
    for example in examples:
        images = example.get("image")
        if not isinstance(images, list) or not images:
            raise ValueError("Each example must contain at least one image")
        main_images.append(np.asarray(images[0], dtype=np.uint8))
        task_descriptions.append(str(example.get("lang", "")))
        state = example.get("state")
        if state is None:
            has_state = False
        else:
            states.append(np.asarray(state, dtype=np.float32))
    env_obs = {
        "main_images": np.stack(main_images),
        "task_descriptions": task_descriptions,
    }
    if has_state:
        env_obs["states"] = np.stack(states)
    return env_obs


def write_atomic(path: Path, payload: bytes) -> None:
    """Write bytes atomically in the bridge directory."""
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build_policy(checkpoint: Path, base_checkpoint: Path):
    """Load a GR1 OFT base model and optionally overlay RLinf full weights."""
    state_dict = None
    action_model_precision = None
    qwen3_vl_lora_rank = None
    if checkpoint.resolve() != base_checkpoint.resolve():
        LOGGER.info("Inspecting RLinf full weights from %s", checkpoint)
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        action_model_precision = infer_action_model_precision(state_dict)
        qwen3_vl_lora_rank = infer_qwen3_vl_lora_rank(state_dict)
    cfg = OmegaConf.create(
        {
            "model_path": str(base_checkpoint),
            "action_model_precision": action_model_precision,
            "lora_rank": qwen3_vl_lora_rank or 32,
            "lora_target_modules": "all-linear",
            "action_dim": 29,
            "num_action_chunks": 16,
            "unnorm_key": "gr1",
            "action_stats_source": "minmax",
            "policy_setup": "gr1",
            "add_value_head": False,
            "starvla": {
                "framework_name": "QwenOFT",
                "enable_state_input": False,
            },
        }
    )
    policy = get_model(cfg, torch_dtype=torch.bfloat16)
    if qwen3_vl_lora_rank is not None:
        LOGGER.info(
            "Building Qwen3-VL-only LoRA adapters with rank %d",
            qwen3_vl_lora_rank,
        )
        policy = apply_qwen3_vl_lora(policy, cfg)
    if state_dict is not None:
        LOGGER.info("Loading RLinf full weights from %s", checkpoint)
        state_dict = remove_training_only_value_head(state_dict)
        policy.load_state_dict(state_dict, strict=True)
        del state_dict
    policy = policy.to("cuda").eval()
    # The official RoboCasa adapter loads JSON min/max statistics as float64.
    # Preserve that dtype here: rounding environment actions to float32 changes
    # MuJoCo state on the first step and compounds in closed-loop rollouts.
    action_stats = policy.starvla_model.norm_stats["gr1"]["action"]
    policy._action_norm_stats = {
        "q99": np.asarray(action_stats["max"], dtype=np.float64),
        "q01": np.asarray(action_stats["min"], dtype=np.float64),
        "mask": np.asarray(action_stats.get("mask", [True] * 29), dtype=bool),
    }
    return policy


def serve(
    checkpoint: Path,
    base_checkpoint: Path,
    bridge_dir: Path,
    starvla_repo_root: Path,
    checkpoint_b: Path | None = None,
    ensemble_weight_b: float = 0.5,
) -> None:
    """Run the file-backed inference loop."""
    bridge_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(starvla_repo_root)
    policy = build_policy(checkpoint, base_checkpoint)
    policy_b = (
        build_policy(checkpoint_b, base_checkpoint)
        if checkpoint_b is not None
        else None
    )
    if policy_b is not None and not 0.0 <= ensemble_weight_b <= 1.0:
        raise ValueError(
            f"ensemble_weight_b must be in [0, 1], got {ensemble_weight_b}"
        )
    packer = msgpack_numpy.Packer()
    metadata = {
        "framework": "RLinf",
        "policy_class": type(policy).__name__,
        "checkpoint": str(checkpoint),
        "checkpoint_b": str(checkpoint_b) if checkpoint_b is not None else None,
        "ensemble_weight_b": ensemble_weight_b if policy_b is not None else None,
        "base_checkpoint": str(base_checkpoint),
        "starvla_repo_root": str(starvla_repo_root),
        "transport": "file",
        "action_model_precision": str(
            next(policy.starvla_model.action_model.parameters()).dtype
        ),
        "action_space": "GR1 29-D normalized actions (official unnormalization)",
    }
    write_atomic(bridge_dir / "metadata.msgpack", packer.pack(metadata))
    (bridge_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("RLinf StarVLA server ready in %s", bridge_dir)
    while True:
        requests = sorted(bridge_dir.glob("request.*.msgpack"))
        if not requests:
            time.sleep(0.002)
            continue
        for request_path in requests:
            request_id = request_path.name.removeprefix("request.").removesuffix(
                ".msgpack"
            )
            try:
                query = msgpack_numpy.unpackb(request_path.read_bytes())
                with torch.inference_mode():
                    normalized_actions = predict_normalized_actions(
                        policy, query["examples"]
                    )
                    if policy_b is not None:
                        normalized_actions_b = predict_normalized_actions(
                            policy_b, query["examples"]
                        )
                        normalized_actions = blend_normalized_actions(
                            normalized_actions,
                            normalized_actions_b,
                            ensemble_weight_b,
                        )
                response = {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "request_id": request_id,
                    "data": {"normalized_actions": normalized_actions},
                }
            except Exception as error:  # noqa: BLE001 - return inference traceback
                LOGGER.exception("RLinf inference failed for request %s", request_id)
                response = {
                    "status": "error",
                    "ok": False,
                    "request_id": request_id,
                    "error": {
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            finally:
                request_path.unlink(missing_ok=True)
            write_atomic(
                bridge_dir / f"response.{request_id}.msgpack", packer.pack(response)
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--starvla-repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    parser.add_argument("--ensemble-weight-b", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(
        args.checkpoint,
        args.base_checkpoint,
        args.bridge_dir,
        args.starvla_repo_root,
        args.checkpoint_b,
        args.ensemble_weight_b,
    )


if __name__ == "__main__":
    main()
