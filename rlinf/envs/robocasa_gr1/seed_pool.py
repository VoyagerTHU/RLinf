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

"""Deterministic seed-pool sampling for grouped GRPO rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_task_seeds(
    manifest_path: str | Path,
    task_name: str,
    *,
    expected_size: int | None = None,
) -> list[int]:
    """Load and validate one task's seed list from a JSON manifest."""
    path = Path(manifest_path).expanduser().resolve()
    with path.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)

    if isinstance(payload, list):
        raw_seeds = payload
    elif isinstance(payload, dict) and "tasks" in payload:
        tasks = payload["tasks"]
        if task_name not in tasks:
            raise KeyError(f"Task {task_name!r} is not present in seed manifest {path}")
        raw_seeds = tasks[task_name]
    elif isinstance(payload, dict) and "seeds" in payload:
        raw_seeds = payload["seeds"]
    else:
        raise ValueError(
            f"Unsupported seed manifest schema in {path}; expected a list, "
            "a {'seeds': [...]} object, or a {'tasks': {task: [...]}} object."
        )

    seeds = [int(seed) for seed in raw_seeds]
    if not seeds:
        raise ValueError(f"Seed manifest {path} contains no seeds for {task_name}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"Seed manifest {path} contains duplicate seeds for {task_name}"
        )
    if any(seed <= 0 or seed >= 2**31 - 1 for seed in seeds):
        raise ValueError(f"Seed manifest {path} contains seeds outside (0, 2**31-1)")
    if expected_size is not None and len(seeds) != int(expected_size):
        raise ValueError(
            f"Expected {expected_size} seeds for {task_name}, got {len(seeds)} in {path}"
        )
    return seeds


def _select_task_seeds(
    pool: np.ndarray,
    count: int,
    *,
    selection_round: int,
    rng_seed,
    shuffle: bool,
    valid_seed_count: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select ``count`` seeds of one task for ``selection_round``.

    Training (``shuffle=True``) walks one deterministic shuffled ring so a
    contiguous slice visits the whole pool before repeating and stays exactly
    resumable from ``selection_round``. Evaluation (``shuffle=False``) walks
    the ordered pool prefix ``pool[:valid_seed_count]`` in slices of ``count``;
    slots past the prefix are tiled but flagged invalid so a fixed seed set can
    be covered by fewer environments over several rounds.
    """
    selection_round = int(selection_round)
    if selection_round < 0:
        raise ValueError("selection_round must be non-negative")
    count = int(count)
    if shuffle:
        if count > len(pool):
            raise ValueError(
                f"Cannot sample {count} unique groups from {len(pool)} seeds"
            )
        generator = np.random.default_rng(rng_seed)
        shuffled_pool = generator.permutation(pool)
        round_begin = selection_round * count
        selected_indices = (round_begin + np.arange(count, dtype=np.int64)) % len(
            shuffled_pool
        )
        return shuffled_pool[selected_indices], np.ones(count, dtype=bool)

    prefix = len(pool) if valid_seed_count is None else int(valid_seed_count)
    if prefix <= 0 or prefix > len(pool):
        raise ValueError(f"valid_seed_count must be in [1, {len(pool)}], got {prefix}")
    ordered = pool[:prefix]
    indices = selection_round * count + np.arange(count, dtype=np.int64)
    return ordered[indices % prefix], indices < prefix


def select_process_seed_groups(
    seed_pool: list[int],
    *,
    groups_per_process: int,
    process_index: int,
    total_processes: int,
    selection_round: int,
    sampler_seed: int,
    shuffle: bool,
    valid_seed_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select globally coordinated seed groups and return one process's slice.

    Every process recreates the same global selection, then takes a disjoint
    contiguous slice. This keeps sampling reproducible without cross-worker RPC.
    """
    _, seeds, valid, global_group_ids = select_multitask_process_seed_groups(
        [seed_pool],
        groups_per_process=groups_per_process,
        process_index=process_index,
        total_processes=total_processes,
        selection_round=selection_round,
        sampler_seed=sampler_seed,
        shuffle=shuffle,
        valid_seed_count=valid_seed_count,
    )
    return seeds, valid, global_group_ids


def assign_group_tasks(global_group_count: int, num_tasks: int) -> np.ndarray:
    """Map every global seed group to a task index, round-robin.

    Task assignment is a function of the group id only, so it never changes
    between rollout rounds and each simulator subprocess can keep the task it
    was built with. Round-robin over contiguous per-process slices spreads
    tasks evenly across processes as long as ``groups_per_process`` and
    ``num_tasks`` are not both small multiples of each other.
    """
    global_group_count = int(global_group_count)
    num_tasks = int(num_tasks)
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if global_group_count < num_tasks:
        raise ValueError(
            f"{global_group_count} seed groups cannot cover {num_tasks} tasks; "
            "every task needs at least one group"
        )
    return np.arange(global_group_count, dtype=np.int64) % num_tasks


def select_multitask_process_seed_groups(
    seed_pools: list[list[int]],
    *,
    groups_per_process: int,
    process_index: int,
    total_processes: int,
    selection_round: int,
    sampler_seed: int,
    shuffle: bool,
    valid_seed_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-task variant of :func:`select_process_seed_groups`.

    Global groups are assigned to tasks with :func:`assign_group_tasks`; each
    task then walks its own seed ring (or ordered eval prefix) independently.
    With one pool this reduces exactly to the single-task selection.

    Returns:
        ``(task_ids, seeds, valid, global_group_ids)`` for this process's slice.
    """
    groups_per_process = int(groups_per_process)
    process_index = int(process_index)
    total_processes = int(total_processes)
    if groups_per_process <= 0 or total_processes <= 0:
        raise ValueError("groups_per_process and total_processes must be positive")
    if process_index < 0 or process_index >= total_processes:
        raise ValueError(
            f"process_index must be in [0, {total_processes}), got {process_index}"
        )
    if not seed_pools:
        raise ValueError("seed_pools must contain at least one task pool")

    global_group_count = groups_per_process * total_processes
    num_tasks = len(seed_pools)
    task_ids = assign_group_tasks(global_group_count, num_tasks)
    seeds = np.zeros(global_group_count, dtype=np.int64)
    valid = np.zeros(global_group_count, dtype=bool)
    for task_index, task_pool in enumerate(seed_pools):
        group_slots = np.flatnonzero(task_ids == task_index)
        # One task keeps the historical generator seed so single-task runs
        # reproduce their seed order; multi-task pools get per-task streams.
        rng_seed = (
            int(sampler_seed) if num_tasks == 1 else [int(sampler_seed), task_index]
        )
        task_seeds, task_valid = _select_task_seeds(
            np.asarray(task_pool, dtype=np.int64),
            len(group_slots),
            selection_round=selection_round,
            rng_seed=rng_seed,
            shuffle=shuffle,
            valid_seed_count=valid_seed_count,
        )
        seeds[group_slots] = task_seeds
        valid[group_slots] = task_valid

    begin = process_index * groups_per_process
    end = begin + groups_per_process
    global_group_ids = np.arange(begin, end, dtype=np.int64)
    return task_ids[begin:end], seeds[begin:end], valid[begin:end], global_group_ids


def eval_rounds_required(
    num_tasks: int, global_group_count: int, valid_seed_count: int
) -> int:
    """Number of ordered evaluation rounds needed to visit every task seed once."""
    task_ids = assign_group_tasks(global_group_count, num_tasks)
    slots_per_task = np.bincount(task_ids, minlength=int(num_tasks))
    return int(np.ceil(int(valid_seed_count) / slots_per_task.min()))
