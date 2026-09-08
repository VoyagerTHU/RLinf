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

"""Tests for keeping GRPO seed groups intact across the env -> actor split."""

import pytest

from rlinf.utils.rollout_grouping import (
    GROUP_RELATIVE_ADV_TYPES,
    compute_trajectory_split,
    resolve_envs_per_actor_trajectory,
)


def test_group_relative_adv_types_cover_grpo_family() -> None:
    assert {"grpo", "temporal_grpo", "grpo_dynamic", "reinpp_baseline"} <= set(
        GROUP_RELATIVE_ADV_TYPES
    )
    assert "gae" not in GROUP_RELATIVE_ADV_TYPES


@pytest.mark.parametrize(
    ("send", "recv", "expected"),
    [(4, 4, 1), (4, 8, 2), (8, 4, 1), (3, 4, 4), (2, 6, 3)],
)
def test_compute_trajectory_split_matches_env_worker(send, recv, expected) -> None:
    """Mirrors EnvWorker.get_actor_split_num: lcm(recv, send) // send."""
    assert compute_trajectory_split(send, recv) == expected


def test_equal_env_and_actor_world_sizes_keep_groups_whole() -> None:
    assert (
        resolve_envs_per_actor_trajectory(
            num_envs_per_env_stage=8,
            env_world_size=4,
            pipeline_stage_num=1,
            actor_world_size=4,
            group_size=8,
        )
        == 8
    )


def test_more_actor_ranks_than_env_workers_splits_seed_groups() -> None:
    """The original recipe: env=0-3, actor=0-7, 8 envs (one group) per worker."""
    with pytest.raises(ValueError, match="seed groups would be split"):
        resolve_envs_per_actor_trajectory(
            num_envs_per_env_stage=8,
            env_world_size=4,
            pipeline_stage_num=1,
            actor_world_size=8,
            group_size=8,
        )


def test_split_is_allowed_when_every_piece_holds_whole_groups() -> None:
    assert (
        resolve_envs_per_actor_trajectory(
            num_envs_per_env_stage=16,
            env_world_size=4,
            pipeline_stage_num=1,
            actor_world_size=8,
            group_size=8,
        )
        == 8
    )


def test_pipeline_stages_count_as_separate_senders() -> None:
    # 4 env workers x 2 stages = 8 senders feeding 8 actors: no split at all.
    assert (
        resolve_envs_per_actor_trajectory(
            num_envs_per_env_stage=8,
            env_world_size=4,
            pipeline_stage_num=2,
            actor_world_size=8,
            group_size=8,
        )
        == 8
    )


def test_uneven_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="equal pieces"):
        resolve_envs_per_actor_trajectory(
            num_envs_per_env_stage=6,
            env_world_size=1,
            pipeline_stage_num=1,
            actor_world_size=4,
            group_size=1,
        )


def test_group_size_one_never_fails_on_grouping() -> None:
    assert (
        resolve_envs_per_actor_trajectory(
            num_envs_per_env_stage=8,
            env_world_size=4,
            pipeline_stage_num=1,
            actor_world_size=8,
            group_size=1,
        )
        == 4
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs_per_env_stage": 0, "group_size": 1},
        {"num_envs_per_env_stage": 8, "group_size": 0},
    ],
)
def test_invalid_sizes_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        resolve_envs_per_actor_trajectory(
            env_world_size=1,
            pipeline_stage_num=1,
            actor_world_size=1,
            **kwargs,
        )
