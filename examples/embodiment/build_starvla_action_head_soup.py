#!/usr/bin/env python3

"""Average StarVLA action-head weights while preserving the first checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


DEFAULT_PREFIX = "starvla_model.action_model."


def blend_prefixed_tensors(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    *,
    prefix: str,
    weight_b: float,
    source_prefix_b: str | None = None,
    cast_b_to_a_dtype: bool = False,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return state A with matching tensors linearly interpolated toward state B."""
    if not 0.0 <= weight_b <= 1.0:
        raise ValueError(f"weight_b must be in [0, 1], got {weight_b}")
    if source_prefix_b is None and state_a.keys() != state_b.keys():
        missing_from_a = sorted(state_b.keys() - state_a.keys())
        missing_from_b = sorted(state_a.keys() - state_b.keys())
        raise ValueError(
            "Checkpoint keys differ: "
            f"missing_from_a={missing_from_a}, missing_from_b={missing_from_b}"
        )

    selected_keys = sorted(key for key in state_a if key.startswith(prefix))
    if not selected_keys:
        raise ValueError(f"No checkpoint keys matched prefix {prefix!r}")

    result = dict(state_a)
    for key in selected_keys:
        tensor_a = state_a[key]
        source_key = (
            key
            if source_prefix_b is None
            else source_prefix_b + key.removeprefix(prefix)
        )
        if source_key not in state_b:
            raise ValueError(
                f"Checkpoint B is missing source tensor {source_key!r} for {key!r}"
            )
        tensor_b = state_b[source_key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Shape mismatch for {key} from {source_key}: "
                f"{tensor_a.shape} != {tensor_b.shape}"
            )
        if tensor_a.dtype != tensor_b.dtype:
            if not cast_b_to_a_dtype:
                raise ValueError(
                    f"Dtype mismatch for {key} from {source_key}: "
                    f"{tensor_a.dtype} != {tensor_b.dtype}"
                )
            tensor_b = tensor_b.to(dtype=tensor_a.dtype)
        if not tensor_a.is_floating_point():
            raise TypeError(f"Matched tensor {key} is not floating point")
        result[key] = torch.lerp(tensor_a, tensor_b, weight_b).contiguous()

    return result, selected_keys


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError(f"Expected a flat tensor state dict in {path}")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--source-prefix-b",
        default=None,
        help=(
            "Optional checkpoint-B prefix mapped onto --prefix. This supports "
            "blending an RLinf full checkpoint with an unwrapped StarVLA state dict."
        ),
    )
    parser.add_argument(
        "--cast-b-to-a-dtype",
        action="store_true",
        help="Explicitly cast selected checkpoint-B tensors to checkpoint A dtype.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    state_a = load_state_dict(args.checkpoint_a)
    state_b = load_state_dict(args.checkpoint_b)
    blended, selected_keys = blend_prefixed_tensors(
        state_a,
        state_b,
        prefix=args.prefix,
        weight_b=args.weight_b,
        source_prefix_b=args.source_prefix_b,
        cast_b_to_a_dtype=args.cast_b_to_a_dtype,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    if temporary_output.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_output}")
    try:
        torch.save(blended, temporary_output)
        os.replace(temporary_output, args.output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    print(
        json.dumps(
            {
                "checkpoint_a": str(args.checkpoint_a),
                "checkpoint_b": str(args.checkpoint_b),
                "output": str(args.output),
                "weight_b": args.weight_b,
                "prefix": args.prefix,
                "source_prefix_b": args.source_prefix_b,
                "cast_b_to_a_dtype": args.cast_b_to_a_dtype,
                "blended_tensors": len(selected_keys),
                "blended_parameters": sum(
                    state_a[key].numel() for key in selected_keys
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
