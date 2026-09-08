"""Tests for the StarVLA RoboCasa-GR1 evaluation bridge."""

from unittest.mock import patch

import numpy as np

from examples.embodiment.eval_starvla_robocasa_gr1_server import (
    infer_action_model_precision,
    infer_qwen3_vl_lora_rank,
    query_to_env_obs,
    remove_training_only_value_head,
)
from rlinf.models.embodiment.starvla import (
    apply_qwen3_vl_lora,
    cast_action_model_precision,
)
from rlinf.models.embodiment.starvla.starvla_action_model import (
    StarVLAForRLActionPrediction,
)
from rlinf.models.embodiment.starvla.utils.action_space import (
    unnormalize_actions_for_env,
    unnormalize_actions_for_env_torch,
)
from rlinf.models.embodiment.starvla.utils.data_pipeline import (
    fetch_action_for_logprob_for_default_forward,
    resize_env_obs_images_cv2,
)


def test_query_to_env_obs_preserves_batch_and_values():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    state = np.arange(29, dtype=np.float32)
    result = query_to_env_obs(
        {"examples": [{"image": [image], "lang": "test task", "state": state}]}
    )

    assert result["main_images"].shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(result["main_images"][0], image)
    np.testing.assert_array_equal(result["states"][0], state)
    assert result["task_descriptions"] == ["test task"]


def test_action_model_precision_override_preserves_backbone_dtype():
    import torch

    class FakeStarVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 3).to(torch.bfloat16)
            self.action_model = torch.nn.Linear(3, 2).to(torch.bfloat16)

    model = FakeStarVLA()
    result = cast_action_model_precision(model, "fp32")

    assert result == torch.float32
    assert model.backbone.weight.dtype == torch.bfloat16
    assert model.action_model.weight.dtype == torch.float32


def test_infer_action_model_precision_from_rlinf_checkpoint():
    import torch

    state_dict = {
        "starvla_model.backbone.weight": torch.zeros(2, dtype=torch.bfloat16),
        "starvla_model.action_model.weight": torch.zeros(2, dtype=torch.float32),
        "actor_logstd": torch.zeros(2, dtype=torch.float32),
    }

    assert infer_action_model_precision(state_dict) == "fp32"


def test_remove_training_only_value_head_preserves_policy_tensors():
    import torch

    state_dict = {
        "starvla_model.action_model.weight": torch.ones(2),
        "value_head.weight": torch.ones(2),
        "value_head.bias": torch.ones(1),
    }

    result = remove_training_only_value_head(state_dict)

    assert set(result) == {"starvla_model.action_model.weight"}
    assert result["starvla_model.action_model.weight"] is state_dict[
        "starvla_model.action_model.weight"
    ]


def test_starvla_ppo_value_head_stays_fp32_with_bf16_policy():
    import torch

    class FakeStarVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_model = torch.nn.Linear(4, 2).to(torch.bfloat16)

    with (
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "action_space_utils.resolve_action_norm_stats",
            return_value={},
        ),
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "infer_policy_profile",
            return_value={
                "action_head_type": "oft",
                "state_adapter_type": None,
                "vlm_type": "fake",
            },
        ),
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "infer_hidden_size",
            return_value=4,
        ),
    ):
        policy = StarVLAForRLActionPrediction(
            starvla_model=FakeStarVLA(),
            action_dim=2,
            num_action_chunks=1,
            add_value_head=True,
            unnorm_key="gr1",
        )

    assert policy.value_head.weight.dtype == torch.float32
    assert policy.starvla_model.action_model.weight.dtype == torch.bfloat16


def test_qwen3_vl_lora_freezes_action_head_and_other_parameters():
    import torch
    from omegaconf import OmegaConf

    class FakeQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.vision_proj = torch.nn.Linear(4, 4)

    class FakeInterface(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeQwen()

    class FakeStarVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen_vl_interface = FakeInterface()
            self.action_model = torch.nn.Linear(4, 2)

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.starvla_model = FakeStarVLA()
            self.actor_logstd = torch.nn.Parameter(torch.zeros(2))

    policy = apply_qwen3_vl_lora(
        FakePolicy(),
        OmegaConf.create(
            {
                "lora_rank": 2,
                "lora_target_modules": "all-linear",
            }
        ),
    )
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]

    assert trainable_names
    assert all("qwen_vl_interface.model" in name for name in trainable_names)
    assert all("lora_" in name for name in trainable_names)
    assert not any(
        parameter.requires_grad
        for parameter in policy.starvla_model.action_model.parameters()
    )
    assert policy.actor_logstd.requires_grad is False


def test_infer_qwen3_vl_lora_rank_rejects_action_head_adapters():
    import pytest
    import torch

    state_dict = {
        "starvla_model.qwen_vl_interface.model.base_model.model.q_proj."
        "lora_A.default.weight": torch.zeros(32, 4),
        "starvla_model.qwen_vl_interface.model.base_model.model.q_proj."
        "lora_B.default.weight": torch.zeros(4, 32),
    }
    assert infer_qwen3_vl_lora_rank(state_dict) == 32

    state_dict["starvla_model.action_model.fc1.lora_A.default.weight"] = torch.zeros(
        32, 4
    )
    with pytest.raises(ValueError, match="outside Qwen3-VL"):
        infer_qwen3_vl_lora_rank(state_dict)


def test_qwen3_vl_lora_can_train_action_head_without_unfreezing_qwen_base():
    import torch
    from omegaconf import OmegaConf

    class FakeQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class FakeInterface(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeQwen()

    class FakeStarVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen_vl_interface = FakeInterface()
            self.action_model = torch.nn.Linear(4, 2)

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.starvla_model = FakeStarVLA()
            self.actor_logstd = torch.nn.Parameter(torch.zeros(2))

    policy = apply_qwen3_vl_lora(
        FakePolicy(),
        OmegaConf.create(
            {
                "lora_rank": 2,
                "lora_target_modules": "all-linear",
                "lora_train_action_head": True,
            }
        ),
    )
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]

    assert trainable_names
    assert any("qwen_vl_interface.model" in name for name in trainable_names)
    assert any("starvla_model.action_model" in name for name in trainable_names)
    assert all(
        ".lora_A." in name or ".lora_B." in name
        for name in trainable_names
        if "qwen_vl_interface.model" in name
    )
    assert all(
        parameter.requires_grad
        for parameter in policy.starvla_model.action_model.parameters()
    )
    assert policy.actor_logstd.requires_grad is False


def test_gr1_unnormalization_keeps_seventh_channel_continuous():
    normalized = np.zeros((1, 2, 29), dtype=np.float32)
    normalized[..., 6] = np.array([0.25, 0.75], dtype=np.float32)
    stats = {
        "q01": np.full(29, -2.0, dtype=np.float64),
        "q99": np.full(29, 2.0, dtype=np.float64),
        "mask": np.ones(29, dtype=bool),
    }

    result = unnormalize_actions_for_env(normalized, stats, policy_setup="gr1")

    assert result.dtype == np.float64
    np.testing.assert_allclose(result[..., 6], [[0.5, 1.5]])


def test_gr1_torch_unnormalization_matches_numpy_and_keeps_gradient():
    import torch

    normalized = torch.linspace(-0.75, 0.75, 58, dtype=torch.float32).reshape(
        1, 2, 29
    )
    normalized.requires_grad_(True)
    stats = {
        "q01": np.linspace(-2.0, -1.0, 29, dtype=np.float64),
        "q99": np.linspace(1.0, 3.0, 29, dtype=np.float64),
        "mask": np.ones(29, dtype=bool),
    }

    result = unnormalize_actions_for_env_torch(
        normalized, stats, policy_setup="gr1"
    )
    expected = unnormalize_actions_for_env(
        normalized.detach().numpy(), stats, policy_setup="gr1"
    )

    np.testing.assert_allclose(
        result.detach().numpy(), expected, rtol=1e-6, atol=3e-7
    )
    result.sum().backward()
    assert normalized.grad is not None
    assert torch.isfinite(normalized.grad).all()
    assert (normalized.grad > 0).all()


def test_starvla_sac_q_heads_start_at_zero():
    import torch

    class FakeStarVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_model = torch.nn.Linear(4, 2)

    profile = {
        "action_head_type": "oft",
        "state_adapter_type": None,
        "vlm_type": "qwen3_vl",
    }
    stats = {
        "q01": np.full(2, -1.0),
        "q99": np.full(2, 1.0),
        "mask": np.ones(2, dtype=bool),
    }
    with (
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "infer_policy_profile",
            return_value=profile,
        ),
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "action_space_utils.resolve_action_norm_stats",
            return_value=stats,
        ),
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "infer_hidden_size",
            return_value=4,
        ),
    ):
        policy = StarVLAForRLActionPrediction(
            starvla_model=FakeStarVLA(),
            action_dim=2,
            num_action_chunks=2,
            num_executed_action_chunks=2,
            add_value_head=False,
            add_q_head=True,
            q_hidden_dims=(8,),
            unnorm_key="gr1",
            policy_setup="gr1",
        )

    features = torch.randn(3, 4)
    actions = torch.randn(3, 4)
    q_values = policy.q_head(features, actions)
    torch.testing.assert_close(q_values, torch.zeros(3, 2))


def test_transition_storage_does_not_mutate_live_language_observation():
    import torch

    from rlinf.data.embodied_io_struct import EmbodiedRolloutResult

    current = {
        "main_images": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
        "task_descriptions": ["test task"],
    }
    following = {
        "main_images": torch.ones(1, 2, 2, 3, dtype=torch.uint8),
        "task_descriptions": ["test task"],
    }
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.append_transitions(current, following)

    assert current["task_descriptions"] == ["test task"]
    assert following["task_descriptions"] == ["test task"]
    assert "task_descriptions" not in rollout.curr_obs[0]
    assert "task_descriptions" not in rollout.next_obs[0]


def test_gr1_image_resize_matches_official_opencv_area():
    import cv2 as cv
    import torch

    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    result = resize_env_obs_images_cv2(
        {"main_images": torch.from_numpy(image[None])}, (7, 5)
    )
    expected = cv.resize(image, (7, 5), interpolation=cv.INTER_AREA)

    assert result["main_images"].shape == (1, 5, 7, 3)
    np.testing.assert_array_equal(result["main_images"][0].numpy(), expected)


def test_fixed_actor_logstd_is_persistent_buffer():
    import torch

    from rlinf.models.embodiment.starvla.starvla_action_model import (
        StarVLAForRLActionPrediction,
    )

    backbone = torch.nn.Linear(1, 1, bias=False)
    profile = {
        "action_head_type": "oft",
        "state_adapter_type": None,
        "vlm_type": "qwen3_vl",
    }
    stats = {
        "q01": np.full(29, -1.0),
        "q99": np.full(29, 1.0),
        "mask": np.ones(29, dtype=bool),
    }
    with (
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model.infer_policy_profile",
            return_value=profile,
        ),
        patch(
            "rlinf.models.embodiment.starvla.starvla_action_model."
            "action_space_utils.resolve_action_norm_stats",
            return_value=stats,
        ),
    ):
        model = StarVLAForRLActionPrediction(
            starvla_model=backbone,
            action_dim=29,
            num_action_chunks=16,
            add_value_head=False,
            unnorm_key="robocasa_gr1",
            trainable_logstd=False,
            initial_logstd=-3.0,
            num_executed_action_chunks=12,
            rollout_prompt_seq_len=152,
        )

    assert "actor_logstd" not in dict(model.named_parameters())
    assert "actor_logstd" in dict(model.named_buffers())
    assert "actor_logstd" in model.state_dict()
    assert model.num_action_chunks == 16
    assert model.num_executed_action_chunks == 12
    assert model._rollout_prompt_seq_len == 152
    torch.testing.assert_close(model.actor_logstd, torch.full((29,), -3.0))

    model.actor_logstd.fill_(-1.0)
    model.restore_configured_fixed_actor_logstd()
    torch.testing.assert_close(model.actor_logstd, torch.full((29,), -3.0))


def test_cached_actions_follow_executed_horizon_not_query_horizon():
    import torch

    class Policy:
        num_action_chunks = 16
        action_dim = 29

    cached = torch.arange(2 * 12 * 29, dtype=torch.float32).reshape(2, -1)
    reference = torch.zeros((2, 12, 29), dtype=torch.bfloat16)

    result = fetch_action_for_logprob_for_default_forward(
        Policy(), data={"action_for_logprob": cached}, reference=reference
    )

    assert result.shape == (2, 12, 29)
    assert result.dtype == torch.bfloat16
