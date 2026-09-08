"""Tests for policy-query and environment execution horizon alignment."""

import pytest
from omegaconf import OmegaConf

from rlinf.utils.rollout_horizon import (
    resolve_action_steps_per_chunk,
    resolve_num_chunk_steps,
)


def test_executed_horizon_controls_rollout_query_count():
    env_cfg = OmegaConf.create({"max_steps_per_rollout_epoch": 720})
    model_cfg = OmegaConf.create(
        {"num_action_chunks": 16, "num_executed_action_chunks": 12}
    )

    assert resolve_action_steps_per_chunk(env_cfg, model_cfg) == 12
    assert resolve_num_chunk_steps(env_cfg, model_cfg) == 60


def test_environment_execution_horizon_has_priority():
    env_cfg = OmegaConf.create(
        {"max_steps_per_rollout_epoch": 720, "action_steps_per_chunk": 8}
    )
    model_cfg = OmegaConf.create(
        {"num_action_chunks": 16, "num_executed_action_chunks": 12}
    )

    assert resolve_action_steps_per_chunk(env_cfg, model_cfg) == 8
    assert resolve_num_chunk_steps(env_cfg, model_cfg) == 90


def test_non_divisible_execution_horizon_is_rejected():
    env_cfg = OmegaConf.create({"max_steps_per_rollout_epoch": 10})
    model_cfg = OmegaConf.create(
        {"num_action_chunks": 16, "num_executed_action_chunks": 6}
    )

    with pytest.raises(ValueError, match="must be divisible"):
        resolve_num_chunk_steps(env_cfg, model_cfg)
