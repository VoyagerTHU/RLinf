"""Contract tests for ``SolverKitchenEnv`` driven by a fake solver backend."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from solver_kitchen_fakes import FakeBackend

from rlinf.envs.solver_kitchen.solver_kitchen_env import SolverKitchenEnv


def _make_cfg(**overrides):
    base = {
        "env_type": "solver_kitchen",
        "group_size": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        "max_episode_steps": 5,
        "seed": 100,
        "seed_sampler_seed": 7,
        "seed_pool_size": 16,
        "task_description": "reach the knife",
        "reward_mode": "success",
        "render": True,
        "render_every_step": False,
        "backend": "inprocess",
        "sim": {"fps": 60, "sim_substeps": 8},
        "task": {"target_offset": [0.0, 0.0, 0.05]},
        "torch_env": {"normalized_actions": True},
        "cameras": [
            {"name": "main", "width": 8, "height": 6, "position": [0, 0, 1], "look_at": [0, 1, 0]}
        ],
        "video_cfg": {"save_video": False},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_env(num_envs=4, seed_offset=0, total_num_processes=1, backend=None, **overrides):
    backend = FakeBackend() if backend is None else backend
    env = SolverKitchenEnv(
        _make_cfg(**overrides),
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=total_num_processes,
        worker_info=None,
        backend=backend,
    )
    return env, backend


def test_reset_observation_contract():
    env, backend = _make_env(num_envs=4)
    obs, info = env.reset()
    assert info == {}
    assert set(obs) >= {"main_images", "states", "task_descriptions"}
    assert obs["main_images"].shape == (4, 6, 8, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["states"].shape == (4, backend.core.obs_dim)
    assert obs["states"].dtype == torch.float32
    assert obs["task_descriptions"] == ["reach the knife"] * 4
    assert "extra_view_images" not in obs
    assert env.capture_image().shape == (4, 6, 8, 3)
    assert backend.core.render_calls == 1
    # The core config forwarded to the backend carries the task horizon and
    # forces the solver's own auto reset off.
    assert env.core_config["task"]["max_episode_steps"] == 5
    assert env.core_config["torch_env"]["auto_reset"] is False
    assert env.core_config["num_envs"] == 4
    env.close()
    assert backend.closed


def test_extra_cameras_are_exposed_separately():
    cameras = [
        {"name": "main", "width": 8, "height": 6},
        {"name": "side", "width": 8, "height": 6},
    ]
    env, _ = _make_env(num_envs=2, cameras=cameras, main_camera="side")
    obs, _ = env.reset()
    assert obs["main_images"].shape == (2, 6, 8, 3)
    assert obs["extra_view_images"].shape == (2, 1, 6, 8, 3)
    env.close()


def test_grouped_seeds_share_initial_state():
    env, backend = _make_env(num_envs=8, group_size=4)
    obs, _ = env.reset()
    seeds = env.env_seeds
    assert seeds.shape == (8,)
    assert len(set(seeds[:4].tolist())) == 1
    assert len(set(seeds[4:].tolist())) == 1
    assert seeds[0] != seeds[4]
    # Fake core writes the seed into states[:, 0].
    states = obs["states"].numpy()
    assert np.all(states[:4, 0] == seeds[0])
    assert np.all(states[4:, 0] == seeds[4])
    assert env.trajectory_ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert env.global_group_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # Seeds passed to the backend match the assignment.
    _, passed_seeds = backend.core.reset_calls[-1]
    assert passed_seeds.tolist() == seeds.tolist()
    env.close()


def test_seed_groups_are_disjoint_across_processes():
    env0, _ = _make_env(num_envs=4, group_size=2, seed_offset=0, total_num_processes=2)
    env1, _ = _make_env(num_envs=4, group_size=2, seed_offset=1, total_num_processes=2)
    assert set(env0.group_seeds.tolist()).isdisjoint(env1.group_seeds.tolist())
    assert env1.global_group_ids.tolist() == [2, 2, 3, 3]
    env0.update_reset_state_ids()
    env1.update_reset_state_ids()
    assert set(env0.group_seeds.tolist()).isdisjoint(env1.group_seeds.tolist())
    env0.close()
    env1.close()


def test_chunk_step_renders_only_last_substep_by_default():
    env, backend = _make_env(num_envs=3, max_episode_steps=50)
    env.reset()
    chunk = np.zeros((3, 4, 16), dtype=np.float32)
    obs_list, rewards, terms, truncs, infos = env.chunk_step(chunk)
    assert len(obs_list) == 4 and len(infos) == 4
    assert rewards.shape == (3, 4)
    assert terms.shape == (3, 4) and terms.dtype == torch.bool
    assert truncs.shape == (3, 4) and truncs.dtype == torch.bool
    assert backend.core.step_calls == 4
    # One render at reset, one at the end of the chunk.
    assert backend.core.render_calls == 2
    assert int(obs_list[-1]["main_images"][0, 0, 0, 0]) == 2
    # Intermediate observations carry the last rendered frame.
    assert int(obs_list[0]["main_images"][0, 0, 0, 0]) == 1
    env.close()


def test_chunk_step_renders_every_substep_when_requested():
    env, backend = _make_env(num_envs=2, max_episode_steps=50, render_every_step=True)
    env.reset()
    env.chunk_step(np.zeros((2, 3, 16), dtype=np.float32))
    assert backend.core.render_calls == 1 + 3
    env.close()


def test_video_recording_enables_per_step_rendering():
    env, _ = _make_env(
        num_envs=2, video_cfg={"save_video": True}, render_every_step=None
    )
    assert env.render_every_step is True
    env.close()
    env, _ = _make_env(
        num_envs=2, video_cfg={"save_video": True}, render_every_step=False
    )
    assert env.render_every_step is False
    env.close()


def test_action_steps_per_chunk_limits_executed_actions():
    env, backend = _make_env(num_envs=2, max_episode_steps=50, action_steps_per_chunk=2)
    env.reset()
    obs_list, rewards, *_ = env.chunk_step(np.zeros((2, 4, 16), dtype=np.float32))
    assert len(obs_list) == 2
    assert rewards.shape == (2, 2)
    assert backend.core.step_calls == 2
    with pytest.raises(ValueError, match="exceeds"):
        env.chunk_step(np.zeros((2, 1, 16), dtype=np.float32))
    env.close()


def test_success_reward_terminations_and_truncation():
    backend = FakeBackend(success_after=2, success_worlds=[0])
    env, _ = _make_env(num_envs=2, max_episode_steps=3, backend=backend)
    env.reset()
    zero = np.zeros((2, 16), dtype=np.float32)
    _, r1, t1, tr1, i1 = env.step(zero)
    assert r1.tolist() == [0.0, 0.0]
    assert t1.tolist() == [False, False]
    _, r2, t2, tr2, i2 = env.step(zero)
    assert r2.tolist() == [1.0, 0.0]
    assert t2.tolist() == [True, False]
    assert tr2.tolist() == [False, False]
    assert i2["episode"]["success_once"].tolist() == [1.0, 0.0]
    assert i2["episode"]["success_first_step"].tolist() == [2, -1]
    _, r3, t3, tr3, i3 = env.step(zero)
    # Success keeps being reported but the return no longer accumulates.
    assert r3.tolist() == [0.0, 0.0]
    assert t3.tolist() == [True, False]
    assert tr3.tolist() == [True, True]
    assert i3["episode"]["return"].tolist() == [1.0, 0.0]
    assert i3["episode"]["episode_len"].tolist() == [3, 3]
    assert i3["episode"]["sample_seed"].tolist() == env.env_seeds.tolist()
    assert i3["episode"]["metric_valid"].tolist() == [True, True]
    assert i3["episode"]["min_distance"].tolist() == [0.0, 0.0]
    env.close()


def test_solver_reward_mode_uses_shaped_reward_and_coef():
    backend = FakeBackend(success_after=3, success_worlds=[])
    env, _ = _make_env(num_envs=1, reward_mode="solver", reward_coef=2.0, backend=backend)
    env.reset()
    _, reward, *_ = env.step(np.zeros((1, 16), dtype=np.float32))
    # Fake shaped reward is -distance = -(3 - 1) = -2, times coef 2.
    assert reward.tolist() == [-4.0]
    env.close()


def test_ignore_terminations_reports_success_at_end():
    backend = FakeBackend(success_after=1)
    env, _ = _make_env(num_envs=2, ignore_terminations=True, backend=backend)
    env.reset()
    _, _, terms, _, infos = env.step(np.zeros((2, 16), dtype=np.float32))
    assert terms.tolist() == [False, False]
    assert infos["episode"]["success_at_end"].tolist() == [True, True]
    env.close()


def test_reset_metrics_per_env_subset():
    backend = FakeBackend(success_after=1)
    env, _ = _make_env(num_envs=2, backend=backend)
    env.reset()
    env.step(np.zeros((2, 16), dtype=np.float32))
    assert env.elapsed_steps.tolist() == [1, 1]
    env.reset(env_idx=[1])
    assert env.elapsed_steps.tolist() == [1, 0]
    assert env.success_once.tolist() == [True, False]
    env.close()


def test_eval_padding_marks_invalid_environments_and_wraps_rounds():
    env, _ = _make_env(
        num_envs=4,
        is_eval=True,
        eval_seed_count=3,
        seed_pool_size=8,
    )
    assert env.eval_rounds == 1
    assert env.metric_valid.tolist() == [True, True, True, False]
    assert env.env_seeds[:3].tolist() == env.seed_pool[:3]
    first_round = env.env_seeds.copy()
    env.update_reset_state_ids()
    assert env.env_seeds.tolist() == first_round.tolist()
    # Training rounds advance and set_seed_selection_round restores them.
    train_env, _ = _make_env(num_envs=2, seed_pool_size=8)
    round0 = train_env.env_seeds.copy()
    train_env.update_reset_state_ids()
    round1 = train_env.env_seeds.copy()
    assert round0.tolist() != round1.tolist()
    train_env.set_seed_selection_round(0)
    assert train_env.env_seeds.tolist() == round0.tolist()
    env.close()
    train_env.close()


def test_step_without_actions_only_before_first_reset():
    env, _ = _make_env(num_envs=2)
    obs, reward, terms, truncs, infos = env.step(None)
    assert obs["main_images"].shape == (2, 6, 8, 3)
    assert reward.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        env.step(None)
    env.close()


def test_invalid_group_size_rejected():
    with pytest.raises(ValueError, match="divisible"):
        _make_env(num_envs=3, group_size=2)


def test_render_disabled_yields_state_only_observations():
    env, backend = _make_env(num_envs=2, render=False)
    obs, _ = env.reset()
    assert "main_images" not in obs
    assert obs["states"].shape == (2, backend.core.obs_dim)
    assert env.capture_image() is None
    env.close()


def test_actions_are_clipped_to_normalized_range():
    env, backend = _make_env(num_envs=1, max_episode_steps=10)
    env.reset()
    env.step(np.full((1, 16), 3.0, dtype=np.float32))
    assert np.all(backend.core.actions[-1] == 1.0)
    env.close()
    env, backend = _make_env(num_envs=1, max_episode_steps=10, clip_actions=False)
    env.reset()
    env.step(np.full((1, 16), 3.0, dtype=np.float32))
    assert np.all(backend.core.actions[-1] == 3.0)
    env.close()
