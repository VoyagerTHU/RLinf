# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the replay-buffer transition validity mask used by SAC."""

import pytest
import torch

from rlinf.data.embodied_io_struct import (
    TRANSITION_VALID_KEY,
    ChunkStepResult,
    EmbodiedRolloutResult,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_sac_policy_worker import (
    masked_transition_mean,
    pop_transition_valid,
)


def _rollout_with_valid_mask(steps: int = 3, envs: int = 2) -> EmbodiedRolloutResult:
    result = EmbodiedRolloutResult(max_episode_length=steps)
    done_once = torch.zeros(envs, dtype=torch.bool)
    for step in range(steps):
        obs = {"main_images": torch.zeros(envs, 2, 2, 3, dtype=torch.uint8)}
        obs["task_descriptions"] = ["pick up the cup"] * envs
        result.append_step_result(
            ChunkStepResult(
                actions=torch.zeros(envs, 4),
                rewards=torch.zeros(envs, 2),
                dones=torch.zeros(envs, 2, dtype=torch.bool),
                terminations=torch.zeros(envs, 2, dtype=torch.bool),
                truncations=torch.zeros(envs, 2, dtype=torch.bool),
            )
        )
        result.append_transitions(obs, obs, valid=~done_once)
        # Env 0 terminates during step 0: its later transitions are invalid.
        if step == 0:
            done_once[0] = True
    return result


def test_append_transitions_stores_mask_and_strips_text() -> None:
    result = _rollout_with_valid_mask()
    trajectory = result.to_trajectory()
    assert "task_descriptions" not in trajectory.curr_obs
    mask = trajectory.curr_obs[TRANSITION_VALID_KEY]
    assert mask.shape == (3, 2, 1)
    assert mask.dtype == torch.bool
    assert mask[:, 0, 0].tolist() == [True, False, False]
    assert mask[:, 1, 0].tolist() == [True, True, True]

    flat = TrajectoryReplayBuffer._flatten_trajectory(None, trajectory)
    assert flat["curr_obs"][TRANSITION_VALID_KEY].shape == (6, 1)


def test_pop_transition_valid_removes_mask_from_observations() -> None:
    batch = {
        "curr_obs": {
            "main_images": torch.zeros(4, 2),
            TRANSITION_VALID_KEY: torch.tensor([[True], [False], [True], [True]]),
        },
        "next_obs": {"main_images": torch.zeros(4, 2)},
    }
    valid = pop_transition_valid(batch)
    assert TRANSITION_VALID_KEY not in batch["curr_obs"]
    assert valid.shape == (4, 1)
    assert valid.dtype == torch.float32
    assert valid.squeeze(-1).tolist() == [1.0, 0.0, 1.0, 1.0]
    assert pop_transition_valid({"curr_obs": {"main_images": torch.zeros(1)}}) is None


def test_masked_transition_mean_ignores_invalid_transitions() -> None:
    values = torch.tensor([[1.0, 3.0], [100.0, 100.0], [5.0, 7.0]])
    valid = torch.tensor([[1.0], [0.0], [1.0]])
    assert masked_transition_mean(values, valid).item() == pytest.approx(4.0)
    assert masked_transition_mean(values, None).item() == pytest.approx(
        values.mean().item()
    )
    all_invalid = masked_transition_mean(values, torch.zeros(3, 1))
    assert all_invalid.item() == 0.0
