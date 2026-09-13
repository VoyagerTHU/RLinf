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

"""Startup check that the per-rank rollout divides into optimizer batches."""

import pytest
from omegaconf import OmegaConf

from rlinf.config import _validate_actor_batch_divides_rollout


def _cfg(envs: int, rollout_epoch: int, global_batch: int, loss_type="actor_critic"):
    return OmegaConf.create(
        {
            "algorithm": {"loss_type": loss_type, "rollout_epoch": rollout_epoch},
            "env": {"train": {"total_num_envs": envs}},
            "actor": {"global_batch_size": global_batch},
        }
    )


def test_single_task_recipe_divides() -> None:
    # 32 envs x 4 epochs x 60 chunks / 4 ranks = 1920 = 15 x 128
    _validate_actor_batch_divides_rollout(
        _cfg(32, 4, 512), actor_world_size=4, chunk_steps_per_epoch=60
    )


def test_multitask_recipe_with_512_is_rejected_and_768_accepted() -> None:
    # 48 envs -> 2880 per rank: 2880 % 128 != 0 (the crash), 2880 % 192 == 0
    with pytest.raises(AssertionError, match="not a multiple"):
        _validate_actor_batch_divides_rollout(
            _cfg(48, 4, 512), actor_world_size=4, chunk_steps_per_epoch=60
        )
    _validate_actor_batch_divides_rollout(
        _cfg(48, 4, 768), actor_world_size=4, chunk_steps_per_epoch=60
    )


def test_replay_based_losses_are_not_checked() -> None:
    _validate_actor_batch_divides_rollout(
        _cfg(48, 4, 512, loss_type="embodied_sac"),
        actor_world_size=4,
        chunk_steps_per_epoch=60,
    )
