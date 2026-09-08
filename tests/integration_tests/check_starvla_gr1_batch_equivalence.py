"""Check StarVLA GR1 OFT batch-one versus batch-two replay equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.embodiment.eval_starvla_robocasa_gr1_server import build_policy
from rlinf.models.embodiment.starvla.action_heads.oft import (
    _run_oft_backbone_and_head,
    run_rollout_oft,
)
from rlinf.models.embodiment.starvla.utils import data_pipeline
from rlinf.models.embodiment.starvla.utils.profile import (
    RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    resolve_vlm_interface,
)


def make_image(offset: int) -> np.ndarray:
    rows = np.arange(224, dtype=np.uint16)[:, None]
    cols = np.arange(224, dtype=np.uint16)[None, :]
    return np.stack(
        [
            np.broadcast_to((rows + offset) % 256, (224, 224)),
            np.broadcast_to((cols + 2 * offset) % 256, (224, 224)),
            (rows + cols + 3 * offset) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def combine_forward_inputs(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise RuntimeError("Individual forward-input key sets differ")
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in keys}


def model_inputs(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return data_pipeline.collect_default_forward_model_inputs(
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


def tensor_delta(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    delta = (lhs.float() - rhs.float()).abs()
    return {
        "mean_abs": float(delta.mean().item()),
        "max_abs": float(delta.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    policy = build_policy(args.checkpoint, args.base_checkpoint)
    with torch.no_grad():
        policy.actor_logstd.fill_(-5.0)

    instruction = "pick the cup and place it in the drawer then close the drawer"
    images = [make_image(0), make_image(37)]

    def predict(selected_images: list[np.ndarray]):
        examples = [
            {"image": [image], "lang": instruction} for image in selected_images
        ]
        with torch.inference_mode():
            payload = run_rollout_oft(
                policy,
                examples=examples,
                env_obs={},
                mode="eval",
                calculate_logprobs=True,
                calculate_values=False,
                sampling_kwargs={"do_sample": False},
            )
        stored_inputs, _ = data_pipeline.normalize_model_inputs_for_storage(
            model_inputs=payload["model_inputs"],
            starvla_model=policy.starvla_model,
            rollout_prompt_seq_len=155,
        )
        stored_inputs = data_pipeline.pack_model_inputs_for_storage(
            model_inputs=stored_inputs,
            batch_size=len(selected_images),
        )
        action_for_logprob = payload["extra_forward_inputs"]["action_for_logprob"]
        forward_inputs = {
            "action": action_for_logprob.reshape(len(selected_images), -1),
            "action_for_logprob": action_for_logprob,
            **stored_inputs,
        }
        return payload, forward_inputs

    individual = [predict([image]) for image in images]
    native_batch = predict(images)
    duplicate_batch = predict([images[0], images[0]])
    reversed_batch = predict(list(reversed(images)))
    combined = combine_forward_inputs([result[1] for result in individual])

    individual_means = torch.cat(
        [result[1]["action_for_logprob"] for result in individual],
        dim=0,
    ).cuda()
    native_means = native_batch[1]["action_for_logprob"].cuda()
    duplicate_means = duplicate_batch[1]["action_for_logprob"].cuda()
    reversed_means = reversed_batch[1]["action_for_logprob"].flip(0).cuda()

    with torch.inference_mode():
        individual_replay = torch.cat(
            [
                policy.default_forward(
                    forward_inputs=result[1],
                    compute_logprobs=True,
                )["logprobs"]
                for result in individual
            ],
            dim=0,
        )
        combined_replay = policy.default_forward(
            forward_inputs=combined,
            compute_logprobs=True,
        )["logprobs"]
        native_batch_replay = policy.default_forward(
            forward_inputs=native_batch[1],
            compute_logprobs=True,
        )["logprobs"]
        individual_backbone = [
            _run_oft_backbone_and_head(
                policy,
                model_inputs=model_inputs(result[1]),
                use_cache=False,
            )
            for result in individual
        ]
        combined_predicted_means, combined_hidden, _ = _run_oft_backbone_and_head(
            policy,
            model_inputs=model_inputs(combined),
            use_cache=False,
        )
        individual_predicted_means = torch.cat(
            [row[0] for row in individual_backbone], dim=0
        )
        combined_model_inputs = model_inputs(combined)
        individual_action_queries = torch.cat(
            [
                policy.starvla_model._gather_action_token_embeddings(
                    row[1],
                    model_inputs(result[1])["input_ids"],
                    action_token_id=policy.starvla_model.action_token_id,
                )
                for row, result in zip(individual_backbone, individual)
            ],
            dim=0,
        )
        combined_action_queries = policy.starvla_model._gather_action_token_embeddings(
            combined_hidden,
            combined_model_inputs["input_ids"],
            action_token_id=policy.starvla_model.action_token_id,
        )

        vlm = resolve_vlm_interface(policy.starvla_model).model
        individual_vision = []
        individual_deepstack = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for result in individual:
                inputs = model_inputs(result[1])
                vision, deepstack = vlm.get_image_features(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
                individual_vision.extend(vision)
                individual_deepstack.append(deepstack)
            combined_vision, combined_deepstack = vlm.get_image_features(
                combined_model_inputs["pixel_values"],
                combined_model_inputs["image_grid_thw"],
            )
        individual_vision_tensor = torch.cat(individual_vision, dim=0)
        combined_vision_tensor = torch.cat(combined_vision, dim=0)
        individual_deepstack_tensors = [
            torch.cat([sample[layer] for sample in individual_deepstack], dim=0)
            for layer in range(len(combined_deepstack))
        ]

    input_deltas: dict[str, dict[str, float] | str] = {}
    native_inputs = native_batch[1]
    for key in sorted(set(combined) & set(native_inputs)):
        if combined[key].shape != native_inputs[key].shape:
            input_deltas[key] = (
                f"shape mismatch {tuple(combined[key].shape)} vs "
                f"{tuple(native_inputs[key].shape)}"
            )
        else:
            input_deltas[key] = tensor_delta(combined[key], native_inputs[key])

    report = {
        "actor_logstd": policy.actor_logstd.detach().float().cpu().tolist(),
        "native_batch_mean_vs_individual": tensor_delta(native_means, individual_means),
        "duplicate_batch_mean_vs_repeated_individual": tensor_delta(
            duplicate_means, individual_means[0:1].expand_as(duplicate_means)
        ),
        "reversed_batch_mean_vs_individual": tensor_delta(
            reversed_means, individual_means
        ),
        "combined_batch_predicted_mean_vs_individual": tensor_delta(
            combined_predicted_means, individual_predicted_means
        ),
        "combined_batch_action_query_vs_individual": tensor_delta(
            combined_action_queries, individual_action_queries
        ),
        "combined_batch_vision_vs_individual": tensor_delta(
            combined_vision_tensor, individual_vision_tensor
        ),
        "combined_batch_deepstack_vs_individual": [
            tensor_delta(combined, individual)
            for combined, individual in zip(
                combined_deepstack, individual_deepstack_tensors
            )
        ],
        "combined_batch_logprob_vs_individual": tensor_delta(
            combined_replay, individual_replay
        ),
        "native_batch_replay_logprob_vs_native_rollout": tensor_delta(
            native_batch_replay, native_batch[0]["prev_logprobs"]
        ),
        "combined_batch_logprob_delta_per_sample": [
            tensor_delta(combined_replay[i], individual_replay[i]) for i in range(2)
        ],
        "combined_vs_native_forward_inputs": input_deltas,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
