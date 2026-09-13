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

"""Multi-task seed-group selection for the RoboCasa GR1 environment."""

import importlib.util
import pathlib

import numpy as np
import pytest

# Import the sampler module directly: the package __init__ pulls in gymnasium
# and the simulator, which unit-test environments do not have.
_SPEC = importlib.util.spec_from_file_location(
    "robocasa_gr1_seed_pool",
    pathlib.Path(__file__).resolve().parents[2]
    / "rlinf/envs/robocasa_gr1/seed_pool.py",
)
seed_pool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(seed_pool)


def _pools(num_tasks: int, size: int = 500) -> list[list[int]]:
    return [
        list(range(t * 10_000 + 1, t * 10_000 + 1 + size)) for t in range(num_tasks)
    ]


def test_single_task_train_ring_matches_legacy_selection() -> None:
    pool = list(range(1000, 1500))
    shuffled = np.random.default_rng(20260820).permutation(np.asarray(pool))
    for selection_round in (0, 1, 7):
        seeds, valid, group_ids = seed_pool.select_process_seed_groups(
            pool,
            groups_per_process=8,
            process_index=1,
            total_processes=4,
            selection_round=selection_round,
            sampler_seed=20260820,
            shuffle=True,
        )
        expected = shuffled[(selection_round * 32 + np.arange(32)) % 500][8:16]
        assert np.array_equal(seeds, expected)
        assert valid.all()
        assert group_ids.tolist() == list(range(8, 16))


def test_single_task_eval_tiles_and_flags_invalid_slots() -> None:
    seeds, valid, _ = seed_pool.select_process_seed_groups(
        list(range(1, 51)),
        groups_per_process=14,
        process_index=3,
        total_processes=4,
        selection_round=0,
        sampler_seed=1,
        shuffle=False,
        valid_seed_count=50,
    )
    assert seeds.tolist() == [43, 44, 45, 46, 47, 48, 49, 50, 1, 2, 3, 4, 5, 6]
    assert valid.tolist() == [True] * 8 + [False] * 6


def test_tasks_are_round_robin_over_global_groups_and_fixed_across_rounds() -> None:
    pools = _pools(24)
    per_process = [
        seed_pool.select_multitask_process_seed_groups(
            pools,
            groups_per_process=12,
            process_index=process_index,
            total_processes=4,
            selection_round=0,
            sampler_seed=5,
            shuffle=True,
        )
        for process_index in range(4)
    ]
    all_task_ids = np.concatenate([task_ids for task_ids, _, _, _ in per_process])
    assert np.bincount(all_task_ids, minlength=24).tolist() == [2] * 24
    assert per_process[0][0].tolist() == list(range(12))
    assert per_process[1][0].tolist() == list(range(12, 24))

    later_task_ids, later_seeds, _, _ = seed_pool.select_multitask_process_seed_groups(
        pools,
        groups_per_process=12,
        process_index=0,
        total_processes=4,
        selection_round=3,
        sampler_seed=5,
        shuffle=True,
    )
    assert np.array_equal(later_task_ids, per_process[0][0])
    assert not np.array_equal(later_seeds, per_process[0][1])
    for task_id, seed in zip(later_task_ids, later_seeds):
        assert seed in pools[task_id]


def test_multitask_eval_rounds_cover_every_task_seed_exactly_once() -> None:
    pools = _pools(24, size=50)
    rounds = seed_pool.eval_rounds_required(24, 96, 10)
    assert rounds == 3
    covered = {task_id: [] for task_id in range(24)}
    for selection_round in range(rounds):
        for process_index in range(4):
            task_ids, seeds, valid, _ = seed_pool.select_multitask_process_seed_groups(
                pools,
                groups_per_process=24,
                process_index=process_index,
                total_processes=4,
                selection_round=selection_round,
                sampler_seed=5,
                shuffle=False,
                valid_seed_count=10,
            )
            for task_id, seed, is_valid in zip(task_ids, seeds, valid):
                if is_valid:
                    covered[int(task_id)].append(int(seed))
    for task_id in range(24):
        assert sorted(covered[task_id]) == pools[task_id][:10]


def test_more_tasks_than_groups_is_rejected() -> None:
    with pytest.raises(ValueError, match="every task needs at least one group"):
        seed_pool.assign_group_tasks(8, 24)


def test_train_pool_smaller_than_task_slots_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot sample"):
        seed_pool.select_multitask_process_seed_groups(
            _pools(2, size=3),
            groups_per_process=4,
            process_index=0,
            total_processes=2,
            selection_round=0,
            sampler_seed=1,
            shuffle=True,
        )
