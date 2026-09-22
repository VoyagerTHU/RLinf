"""Registration and config validation for the solver_kitchen environment."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.config import _validate_solver_kitchen_env
from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.solver_kitchen import SolverKitchenEnv


def test_env_type_registered():
    assert SupportedEnvType("solver_kitchen") is SupportedEnvType.SOLVER_KITCHEN
    assert get_env_cls("solver_kitchen", OmegaConf.create({})) is SolverKitchenEnv


def test_prepare_actions_passthrough():
    chunk = np.random.rand(4, 3, 16).astype(np.float32)
    out = prepare_actions(
        raw_chunk_actions=chunk,
        env_type="solver_kitchen",
        model_type="mlp_policy",
        num_action_chunks=3,
        action_dim=16,
        policy=None,
        wm_env_type=None,
    )
    assert out is chunk


def _env_cfg(**overrides):
    base = {"env_type": "solver_kitchen", "auto_reset": False, "render": True}
    base.update(overrides)
    return OmegaConf.create(base)


def test_validation_accepts_defaults_and_ignores_other_envs():
    _validate_solver_kitchen_env(_env_cfg(), "train")
    _validate_solver_kitchen_env(OmegaConf.create({"env_type": "robocasa_gr1", "auto_reset": True}), "train")
    _validate_solver_kitchen_env(None, "train")


def test_validation_rejects_auto_reset():
    with pytest.raises(ValueError, match="auto_reset"):
        _validate_solver_kitchen_env(_env_cfg(auto_reset=True), "train")


def test_validation_rejects_bad_backend_reward_mode_and_empty_cameras():
    with pytest.raises(ValueError, match="backend"):
        _validate_solver_kitchen_env(_env_cfg(backend="thread"), "eval")
    with pytest.raises(ValueError, match="reward_mode"):
        _validate_solver_kitchen_env(_env_cfg(reward_mode="dense"), "eval")
    with pytest.raises(ValueError, match="cameras"):
        _validate_solver_kitchen_env(_env_cfg(cameras=[]), "eval")
    # No cameras needed when rendering is disabled.
    _validate_solver_kitchen_env(_env_cfg(render=False, cameras=[]), "eval")
