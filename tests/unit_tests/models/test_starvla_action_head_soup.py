import pytest
import torch

from examples.embodiment.build_starvla_action_head_soup import (
    blend_prefixed_tensors,
)


def test_blend_prefixed_tensors_only_changes_selected_prefix():
    state_a = {
        "starvla_model.backbone.weight": torch.tensor([7.0]),
        "starvla_model.action_model.weight": torch.tensor([0.0, 2.0]),
        "actor_logstd": torch.tensor([-3.5]),
    }
    state_b = {
        "starvla_model.backbone.weight": torch.tensor([11.0]),
        "starvla_model.action_model.weight": torch.tensor([2.0, 6.0]),
        "actor_logstd": torch.tensor([-4.0]),
    }

    blended, selected_keys = blend_prefixed_tensors(
        state_a,
        state_b,
        prefix="starvla_model.action_model.",
        weight_b=0.25,
    )

    assert selected_keys == ["starvla_model.action_model.weight"]
    torch.testing.assert_close(
        blended["starvla_model.action_model.weight"], torch.tensor([0.5, 3.0])
    )
    assert blended["starvla_model.backbone.weight"] is state_a[
        "starvla_model.backbone.weight"
    ]
    assert blended["actor_logstd"] is state_a["actor_logstd"]


def test_blend_prefixed_tensors_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="Checkpoint keys differ"):
        blend_prefixed_tensors(
            {"starvla_model.action_model.a": torch.ones(1)},
            {"starvla_model.action_model.b": torch.ones(1)},
            prefix="starvla_model.action_model.",
            weight_b=0.5,
        )


def test_blend_prefixed_tensors_maps_unwrapped_b_and_casts_explicitly():
    state_a = {
        "actor_logstd": torch.tensor([-3.5]),
        "starvla_model.backbone.weight": torch.tensor([7.0]),
        "starvla_model.action_model.weight": torch.tensor(
            [0.0, 2.0], dtype=torch.float32
        ),
    }
    state_b = {
        "backbone.weight": torch.tensor([11.0], dtype=torch.bfloat16),
        "action_model.weight": torch.tensor(
            [2.0, 6.0], dtype=torch.bfloat16
        ),
    }

    blended, selected_keys = blend_prefixed_tensors(
        state_a,
        state_b,
        prefix="starvla_model.action_model.",
        source_prefix_b="action_model.",
        weight_b=0.25,
        cast_b_to_a_dtype=True,
    )

    assert selected_keys == ["starvla_model.action_model.weight"]
    torch.testing.assert_close(
        blended["starvla_model.action_model.weight"], torch.tensor([0.5, 3.0])
    )
    assert blended["starvla_model.backbone.weight"] is state_a[
        "starvla_model.backbone.weight"
    ]
