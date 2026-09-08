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

"""Helpers for keeping grouped rollouts intact across worker boundaries.

Group-relative advantage estimators (GRPO and friends) normalize trajectory
scores *within* the local batch of every actor rank.  Environment workers own
the trajectories and split them across actor ranks purely by batch position, so
an ``env`` world size that does not match the ``actor`` world size can send half
of a seed group to one actor rank and the other half to another.  The actor
then standardizes trajectories of *different* seeds against each other, which
silently biases every advantage.  These helpers make that configuration a
validation error instead of a silent training bug.
"""

from __future__ import annotations

import math

# Advantage estimators that normalize scores within seed groups on the actor.
GROUP_RELATIVE_ADV_TYPES = frozenset(
    {"grpo", "temporal_grpo", "grpo_dynamic", "reinpp_baseline"}
)


def compute_trajectory_split(send_world_size: int, recv_world_size: int) -> int:
    """Number of pieces each sender splits its rollout result into.

    Mirrors ``EnvWorker.get_actor_split_num`` / ``compute_split_num`` so the
    validation stays in lock-step with the runtime communication pattern.
    """
    if send_world_size <= 0 or recv_world_size <= 0:
        raise ValueError("World sizes must be positive.")
    return math.lcm(recv_world_size, send_world_size) // send_world_size


def resolve_envs_per_actor_trajectory(
    *,
    num_envs_per_env_stage: int,
    env_world_size: int,
    pipeline_stage_num: int,
    actor_world_size: int,
    group_size: int,
) -> int:
    """Return how many environments each actor-rank trajectory contains.

    Args:
        num_envs_per_env_stage: Environments hosted by one pipeline stage of one
            environment worker (``total_num_envs // env_world // stage_num``).
        env_world_size: Number of environment workers.
        pipeline_stage_num: Rollout pipeline stages per environment worker.
        actor_world_size: Number of actor (data-parallel) ranks.
        group_size: Trajectories that share one seed / prompt.

    Returns:
        The number of environments in each trajectory piece received by an
        actor rank.  It is always a multiple of ``group_size``.

    Raises:
        ValueError: If the environment batch cannot be split evenly, or if a
            split would place only part of a seed group on an actor rank.
    """
    if num_envs_per_env_stage <= 0:
        raise ValueError("num_envs_per_env_stage must be positive.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    send_world_size = env_world_size * pipeline_stage_num
    split = compute_trajectory_split(send_world_size, actor_world_size)
    if num_envs_per_env_stage % split != 0:
        raise ValueError(
            f"Each environment worker stage hosts {num_envs_per_env_stage} envs "
            f"but must split its rollout into {split} equal pieces for "
            f"{actor_world_size} actor ranks ({send_world_size} senders)."
        )
    envs_per_piece = num_envs_per_env_stage // split
    if envs_per_piece % group_size != 0:
        raise ValueError(
            "GRPO seed groups would be split across actor ranks: each "
            f"environment worker stage hosts {num_envs_per_env_stage} envs "
            f"(group_size={group_size}) and sends {split} piece(s) of "
            f"{envs_per_piece} envs to {actor_world_size} actor ranks. Group-"
            "relative advantages are normalized within each actor rank's local "
            "batch, so every piece must contain whole groups. Use the same world "
            "size for `env` and `actor`, or make total_num_envs / env_world_size "
            f"/ {split} a multiple of group_size."
        )
    return envs_per_piece
