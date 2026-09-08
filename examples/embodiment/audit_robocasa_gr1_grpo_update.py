"""Audit one RoboCasa-GR1 GRPO update before it is allowed to continue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write a JSON document atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def normalize_state_dict(state_dict: dict) -> dict[str, torch.Tensor]:
    """Normalize original StarVLA and RLinf wrapper checkpoint key names."""
    normalized = {}
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_name = (
            name.removeprefix("starvla_model.")
            if name.startswith("starvla_model.")
            else name
        )
        normalized[normalized_name] = value
    return normalized


def checkpoint_update_stats(
    before_path: Path,
    after_path: Path,
    expected_logstd: float,
) -> dict:
    """Compare trainable head, frozen backbone, and fixed exploration state."""
    before = normalize_state_dict(
        torch.load(before_path, map_location="cpu", weights_only=True, mmap=True)
    )
    after = normalize_state_dict(
        torch.load(after_path, map_location="cpu", weights_only=True, mmap=True)
    )
    comparable_before_keys = set(before) - {"actor_logstd"}
    comparable_after_keys = set(after) - {"actor_logstd"}
    if comparable_before_keys != comparable_after_keys:
        missing = sorted(comparable_before_keys - comparable_after_keys)
        extra = sorted(comparable_after_keys - comparable_before_keys)
        raise ValueError(
            f"Checkpoint key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    trainable_elements = 0
    changed_elements = 0
    changed_abs_sum = 0.0
    changed_max_abs = 0.0
    trainable_fp32_elements = 0
    frozen_changed_names = []
    for name, before_value in before.items():
        after_value = after[name]
        if name.startswith("action_model."):
            if after_value.dtype == torch.float32:
                trainable_fp32_elements += after_value.numel()
            delta = after_value.float() - before_value.float()
            changed = delta.ne(0)
            changed_count = int(changed.sum().item())
            trainable_elements += after_value.numel()
            changed_elements += changed_count
            if changed_count:
                changed_abs = delta[changed].abs().double()
                changed_abs_sum += changed_abs.sum().item()
                changed_max_abs = max(changed_max_abs, changed_abs.max().item())
        elif name == "actor_logstd":
            continue
        elif not torch.equal(before_value, after_value):
            frozen_changed_names.append(name)

    if "actor_logstd" not in after:
        raise ValueError("Checkpoint is missing actor_logstd")
    runtime_logstd = after["actor_logstd"].float()
    logstd_matches = bool(
        torch.allclose(
            runtime_logstd,
            torch.full_like(runtime_logstd, expected_logstd),
            rtol=0.0,
            atol=1e-6,
        )
    )
    return {
        "trainable_elements": trainable_elements,
        "changed_elements": changed_elements,
        "changed_fraction": (
            changed_elements / trainable_elements if trainable_elements else 0.0
        ),
        "changed_mean_abs_delta": (
            changed_abs_sum / changed_elements if changed_elements else 0.0
        ),
        "changed_max_abs_delta": changed_max_abs,
        "trainable_fp32_fraction": (
            trainable_fp32_elements / trainable_elements if trainable_elements else 0.0
        ),
        "frozen_changed_count": len(frozen_changed_names),
        "frozen_changed_names": frozen_changed_names[:50],
        "runtime_logstd_min": runtime_logstd.min().item(),
        "runtime_logstd_mean": runtime_logstd.mean().item(),
        "runtime_logstd_max": runtime_logstd.max().item(),
        "runtime_logstd_matches_expected": logstd_matches,
    }


def rollout_group_stats(
    rows: list[dict], completed_global_step: int, expected_group_size: int
) -> dict:
    """Count mixed groups in the rollout that produced a checkpoint."""
    training_step = completed_global_step - 1
    selected = [
        row for row in rows if int(row.get("training_step", -1)) == training_step
    ]
    if not selected:
        raise ValueError(f"No rollout rows found for training_step={training_step}")
    if len(selected) % expected_group_size:
        raise ValueError(
            f"{len(selected)} rollout rows are not divisible by group size "
            f"{expected_group_size}"
        )

    group_successes = []
    for offset in range(0, len(selected), expected_group_size):
        group = selected[offset : offset + expected_group_size]
        trajectory_ids = sorted(int(row["trajectory_index"]) for row in group)
        if trajectory_ids != list(range(expected_group_size)):
            raise ValueError(
                f"Invalid trajectory ids at rollout offset {offset}: {trajectory_ids}"
            )
        if len({int(row["seed"]) for row in group}) != 1:
            raise ValueError(f"A GRPO group spans multiple seeds at offset {offset}")
        group_successes.append(sum(float(row["trajectory_success"]) for row in group))

    mixed_groups = sum(
        0.0 < successes < expected_group_size for successes in group_successes
    )
    all_failure_groups = sum(successes == 0.0 for successes in group_successes)
    all_success_groups = sum(
        successes == expected_group_size for successes in group_successes
    )
    return {
        "training_step": training_step,
        "trajectories": len(selected),
        "groups": len(group_successes),
        "group_size": expected_group_size,
        "mixed_groups": mixed_groups,
        "mixed_group_fraction": mixed_groups / len(group_successes),
        "advantage_active_trajectories": mixed_groups * expected_group_size,
        "all_failure_groups": all_failure_groups,
        "all_success_groups": all_success_groups,
        "successes": int(sum(group_successes)),
        "success_rate": sum(group_successes) / len(selected),
    }


def select_step_metrics(rows: list[dict], completed_global_step: int) -> dict:
    """Select exactly one persisted metric record for a global step."""
    selected = [
        row
        for row in rows
        if int(row.get("completed_global_step", -1)) == completed_global_step
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one metrics row for step {completed_global_step}, got "
            f"{len(selected)}"
        )
    return selected[0]


def check_thresholds(args: argparse.Namespace, payload: dict) -> list[str]:
    """Return human-readable health check failures."""
    checkpoint = payload["checkpoint"]
    rollout = payload["rollout"]
    metrics = payload["metrics"]
    failures = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(
        checkpoint["frozen_changed_count"] == 0,
        f"{checkpoint['frozen_changed_count']} frozen tensors changed",
    )
    require(
        checkpoint["trainable_fp32_fraction"] == 1.0,
        "not all trainable action-head elements are FP32",
    )
    require(
        checkpoint["runtime_logstd_matches_expected"],
        "checkpoint actor_logstd does not match the configured value",
    )
    require(
        checkpoint["changed_fraction"] >= args.min_changed_fraction,
        f"changed parameter fraction {checkpoint['changed_fraction']:.9g} is below "
        f"{args.min_changed_fraction:.9g}",
    )
    require(
        rollout["trajectories"] == args.expected_trajectories,
        f"expected {args.expected_trajectories} trajectories, got "
        f"{rollout['trajectories']}",
    )
    require(
        rollout["mixed_groups"] >= args.min_mixed_groups,
        f"mixed groups {rollout['mixed_groups']} is below {args.min_mixed_groups}",
    )
    require(
        float(metrics["train/actor/recomputed_rollout_logprobs"]) == 1.0,
        "actor-side old-logprob replay was disabled",
    )
    require(
        float(metrics["train/actor/rollout_actor_logprob_max_abs_delta"])
        <= args.max_rollout_actor_logprob_delta,
        "rollout/actor old-logprob mismatch exceeded its limit",
    )
    require(
        float(metrics["train/actor/post_update_logprob_audit"]) == 1.0,
        "post-update logprob audit was disabled",
    )
    require(
        float(metrics["train/actor/post_update_logprob_finite_fraction"]) == 1.0,
        "non-finite post-update logprobs were observed",
    )
    require(
        float(metrics["train/actor/post_update_ratio_finite_fraction"]) == 1.0,
        "non-finite post-update ratios were observed",
    )
    post_mean_abs = float(metrics["train/actor/post_update_logprob_mean_abs_delta"])
    require(
        post_mean_abs >= args.min_post_update_logprob_mean_abs_delta,
        f"post-update mean absolute logprob delta {post_mean_abs:.9g} is below "
        f"{args.min_post_update_logprob_mean_abs_delta:.9g}",
    )
    require(
        post_mean_abs <= args.max_post_update_logprob_mean_abs_delta,
        f"post-update mean absolute logprob delta {post_mean_abs:.9g} exceeds "
        f"{args.max_post_update_logprob_mean_abs_delta:.9g}",
    )
    require(
        float(metrics["train/actor/post_update_ratio_mean_abs_delta"])
        <= args.max_post_update_ratio_mean_abs_delta,
        "post-update mean absolute ratio delta exceeded its limit",
    )
    require(
        int(round(float(metrics["train/actor/optimizer_steps_this_update"])))
        == args.expected_optimizer_steps,
        "unexpected number of optimizer steps",
    )
    require(
        abs(float(metrics["train/actor/runtime_logstd_mean"]) - args.expected_logstd)
        <= 1e-6,
        "runtime actor logstd metric does not match the configured value",
    )
    return failures


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-checkpoint", type=Path, required=True)
    parser.add_argument("--after-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--expected-trajectories", type=int, default=512)
    parser.add_argument("--expected-optimizer-steps", type=int, default=1)
    parser.add_argument("--expected-logstd", type=float, required=True)
    parser.add_argument("--min-mixed-groups", type=int, default=16)
    parser.add_argument("--min-changed-fraction", type=float, default=0.001)
    parser.add_argument("--max-rollout-actor-logprob-delta", type=float, default=1e-6)
    parser.add_argument(
        "--min-post-update-logprob-mean-abs-delta", type=float, default=1e-6
    )
    parser.add_argument(
        "--max-post-update-logprob-mean-abs-delta", type=float, default=0.02
    )
    parser.add_argument(
        "--max-post-update-ratio-mean-abs-delta", type=float, default=0.02
    )
    return parser.parse_args()


def main() -> None:
    """Run the checkpoint, rollout, and policy-movement audits."""
    args = parse_args()
    payload = {
        "schema_version": 1,
        "completed_global_step": args.step,
        "before_checkpoint": str(args.before_checkpoint.resolve()),
        "after_checkpoint": str(args.after_checkpoint.resolve()),
    }
    try:
        payload["checkpoint"] = checkpoint_update_stats(
            args.before_checkpoint, args.after_checkpoint, args.expected_logstd
        )
        payload["rollout"] = rollout_group_stats(
            read_jsonl(args.rollouts), args.step, args.group_size
        )
        selected_metrics = select_step_metrics(read_jsonl(args.metrics), args.step)
        payload["metrics"] = {
            key: value
            for key, value in selected_metrics.items()
            if key.startswith("train/actor/")
            or key in {"completed_global_step", "env/success_once"}
        }
        payload["failures"] = check_thresholds(args, payload)
    except Exception as error:
        payload["failures"] = [f"{type(error).__name__}: {error}"]

    payload["status"] = "pass" if not payload["failures"] else "fail"
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
