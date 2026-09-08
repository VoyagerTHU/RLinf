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
        raise ValueError(f"Seed manifest {path} contains duplicate seeds for {task_name}")
    if any(seed <= 0 or seed >= 2**31 - 1 for seed in seeds):
        raise ValueError(f"Seed manifest {path} contains seeds outside (0, 2**31-1)")
    if expected_size is not None and len(seeds) != int(expected_size):
        raise ValueError(
            f"Expected {expected_size} seeds for {task_name}, got {len(seeds)} in {path}"
        )
    return seeds


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
    groups_per_process = int(groups_per_process)
    process_index = int(process_index)
    total_processes = int(total_processes)
    if groups_per_process <= 0 or total_processes <= 0:
        raise ValueError("groups_per_process and total_processes must be positive")
    if process_index < 0 or process_index >= total_processes:
        raise ValueError(
            f"process_index must be in [0, {total_processes}), got {process_index}"
        )

    global_group_count = groups_per_process * total_processes
    pool = np.asarray(seed_pool, dtype=np.int64)
    if shuffle:
        selection_round = int(selection_round)
        if selection_round < 0:
            raise ValueError("selection_round must be non-negative")
        if global_group_count > len(pool):
            raise ValueError(
                f"Cannot sample {global_group_count} unique groups from {len(pool)} seeds"
            )
        # Walk one deterministic shuffled ring instead of drawing an
        # independent sample each round. Independent draws are unique within a
        # round but can collide across adjacent rollout epochs, wasting GRPO
        # seed groups. A contiguous ring slice visits the entire pool before
        # repeating and remains exactly resumable from ``selection_round``.
        generator = np.random.default_rng(int(sampler_seed))
        shuffled_pool = generator.permutation(pool)
        round_begin = selection_round * global_group_count
        selected_indices = (
            round_begin + np.arange(global_group_count, dtype=np.int64)
        ) % len(shuffled_pool)
        selected = shuffled_pool[selected_indices]
        valid = np.ones(global_group_count, dtype=bool)
    else:
        count = len(pool) if valid_seed_count is None else int(valid_seed_count)
        if count <= 0 or count > len(pool):
            raise ValueError(
                f"valid_seed_count must be in [1, {len(pool)}], got {count}"
            )
        ordered = pool[:count]
        repeats = (global_group_count + count - 1) // count
        selected = np.tile(ordered, repeats)[:global_group_count]
        valid = np.arange(global_group_count) < count

    begin = process_index * groups_per_process
    end = begin + groups_per_process
    global_group_ids = np.arange(begin, end, dtype=np.int64)
    return selected[begin:end], valid[begin:end], global_group_ids
