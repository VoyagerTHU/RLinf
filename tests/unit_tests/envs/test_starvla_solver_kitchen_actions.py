"""StarVLA action (un)normalization for the solver_kitchen platform."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.starvla.utils import action_space as action_space_utils


class _NoStatsModel:
    norm_stats = {}


def test_identity_override_produces_unit_range_stats():
    stats = action_space_utils.resolve_action_norm_stats(
        _NoStatsModel(),
        unnorm_key="solver_kitchen",
        action_dim=16,
        action_stats_source="minmax",
        override_stats="identity",
    )
    assert stats["q99"].shape == (16,) and stats["q01"].shape == (16,)
    assert np.all(stats["q99"] == 1.0) and np.all(stats["q01"] == -1.0)
    assert stats["mask"].all()


def test_mapping_override_and_bad_override():
    stats = action_space_utils.resolve_action_norm_stats(
        _NoStatsModel(),
        unnorm_key="k",
        action_dim=2,
        action_stats_source="minmax",
        override_stats={"min": [-2.0, 0.0], "max": [2.0, 1.0], "mask": [True, False]},
    )
    assert stats["q99"].tolist() == [2.0, 1.0]
    assert stats["mask"].tolist() == [True, False]
    with pytest.raises(ValueError, match="identity"):
        action_space_utils.resolve_action_norm_stats(
            _NoStatsModel(), unnorm_key="k", action_dim=2, override_stats="bogus"
        )
    with pytest.raises(RuntimeError, match="dim mismatch"):
        action_space_utils.resolve_action_norm_stats(
            _NoStatsModel(), unnorm_key="k", action_dim=3, override_stats="identity"
        )


def test_solver_kitchen_unnormalization_is_pure_minmax_without_gripper_mapping():
    stats = {
        "q99": np.full(16, 1.0),
        "q01": np.full(16, -1.0),
        "mask": np.ones(16, dtype=bool),
    }
    normalized = np.linspace(-1.5, 1.5, 16, dtype=np.float32)[None, None, :]
    env_actions = action_space_utils.unnormalize_actions_for_env(
        normalized, stats, policy_setup="solver_kitchen"
    )
    assert env_actions.shape == normalized.shape
    # Clipped to [-1, 1] and otherwise unchanged; channel 6 is not remapped.
    expected = np.clip(normalized, -1.0, 1.0)
    assert np.allclose(env_actions, expected)
    libero_actions = action_space_utils.unnormalize_actions_for_env(
        normalized, stats, policy_setup="gr1"
    )
    assert np.allclose(libero_actions, expected)


def test_solver_kitchen_torch_round_trip():
    stats = {
        "q99": np.full(4, 0.5),
        "q01": np.full(4, -0.25),
        "mask": np.ones(4, dtype=bool),
    }
    normalized = torch.tensor([[[-1.0, -0.5, 0.0, 1.0]]], dtype=torch.float32)
    env_actions = action_space_utils.unnormalize_actions_for_env_torch(
        normalized, stats, policy_setup="solver_kitchen"
    )
    back = action_space_utils.normalize_actions_from_env_torch(
        env_actions, stats, policy_setup="solver_kitchen"
    )
    assert torch.allclose(back, normalized, atol=1e-6)
    with pytest.raises(ValueError):
        action_space_utils.normalize_actions_from_env_torch(
            env_actions, stats, policy_setup="libero"
        )
