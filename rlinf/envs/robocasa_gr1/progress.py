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

"""Progress milestones and potential functions for RoboCasa GR1 tasks.

Separate from the environment module so the potentials can be unit tested
without gymnasium or the simulator; ``compute_progress_stages`` imports
RoboCasa lazily because it only runs inside a simulator subprocess.
"""

from __future__ import annotations

import numpy as np

# Ordered milestones shared by every gr1_unified pick-and-place task. The three
# "Close" families (drawer / microwave / cabinet) place the object into a
# fixture and then shut its door; the novel families place it into a container
# on the table. Both shapes reduce to "grasp it, put it in, let go", and the
# door is covered by the task's own success check.
PROGRESS_STAGES = ("grasped", "placed", "released")


def compute_progress_stages(task_env) -> dict[str, bool]:
    """Evaluate the shared milestones from simulator state.

    Every predicate here is one the task itself uses inside ``_check_success``
    or ``get_subtask_term_signals``, so the result is exact simulator state
    rather than a learned progress estimate. Unavailable stages are omitted
    instead of guessed, and the caller treats a missing stage as not reached.
    """
    if task_env is None:
        return {}
    import robocasa.utils.object_utils as object_utils

    stages: dict[str, bool] = {}
    objects = getattr(task_env, "objects", None) or {}

    grasp_check = getattr(task_env, "_check_grasp", None)
    target_object = objects.get("obj")
    if callable(grasp_check) and target_object is not None:
        grippers = getattr(task_env.robots[0], "gripper", None)
        candidates = (
            list(grippers.values()) if isinstance(grippers, dict) else [grippers]
        )
        stages["grasped"] = any(
            bool(grasp_check(gripper=gripper, object_geoms=target_object))
            for gripper in candidates
            if gripper is not None
        )

    # Fixture families first: their target is a drawer / microwave / cabinet
    # rather than an object on the table.
    for fixture_name in ("drawer", "microwave", "cabinet"):
        fixture = getattr(task_env, fixture_name, None)
        if fixture is not None and target_object is not None:
            stages["placed"] = bool(
                object_utils.obj_inside_of(
                    env=task_env,
                    obj_name=target_object.name,
                    fixture_id=fixture,
                    partial_check=True,
                )
            )
            break
    else:
        if "container" in objects and "obj" in objects:
            stages["placed"] = bool(
                object_utils.check_obj_in_receptacle(task_env, "obj", "container")
            )

    if "obj" in objects:
        stages["released"] = bool(
            object_utils.any_gripper_obj_far(task_env, obj_name="obj")
        )
    return stages


def drawer_progress_potential(
    info_lists: list[dict],
    terminations: np.ndarray,
    *,
    grasp_reward: float,
    in_drawer_reward: float,
    success_reward: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-environment drawer-task milestone potentials."""
    grasped = np.asarray(
        [
            bool(info.get("subtask_signals", {}).get("grasp_object", 0))
            for info in info_lists
        ]
    )
    in_drawer = np.asarray(
        [
            bool(info.get("subtask_signals", {}).get("obj_in_drawer", 0))
            for info in info_lists
        ]
    )
    potential = np.zeros(len(info_lists), dtype=np.float32)
    potential[grasped] = float(grasp_reward)
    potential[in_drawer] = float(in_drawer_reward)
    potential[np.asarray(terminations, dtype=bool)] = float(success_reward)
    return potential, grasped, in_drawer


def task_general_progress_potential(
    info_lists: list[dict],
    terminations: np.ndarray,
    *,
    stage_rewards: dict[str, float],
    success_reward: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Potential over the milestones shared by all gr1_unified tasks.

    Stages are strictly ordered: ``released`` only counts once ``placed``
    holds, so an untouched object at reset (grippers trivially far from it)
    scores zero rather than the release reward. The potential equals
    ``success_reward`` exactly when the task's own success check fires, which
    keeps it consistent with the metric the policy is evaluated on.
    """
    stages = {
        name: np.asarray(
            [
                bool(info.get("progress_stages", {}).get(name, False))
                for info in info_lists
            ]
        )
        for name in PROGRESS_STAGES
    }
    succeeded = np.asarray(terminations, dtype=bool)
    potential = np.zeros(len(info_lists), dtype=np.float32)
    potential[stages["grasped"]] = float(stage_rewards["grasped"])
    potential[stages["placed"]] = float(stage_rewards["placed"])
    potential[stages["placed"] & stages["released"]] = float(stage_rewards["released"])
    potential[succeeded] = float(success_reward)
    return potential, stages


def potential_shaping_reward(
    potential: np.ndarray,
    prev_potential: np.ndarray,
    episode_over: np.ndarray,
    coef: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Additive potential-based shaping with a zero terminal potential.

    Returns ``(shaping_reward, next_prev_potential)``. Forcing the potential to
    zero on the step an episode ends makes the shaping terms telescope to
    ``-Phi(s_0)`` over the episode, which is zero because nothing is grasped at
    reset. The episodic (undiscounted) return therefore equals the unshaped one
    and the optimal policy is unchanged (Ng et al., 1999); only the
    distribution of credit across the episode changes.
    """
    next_potential = np.where(
        np.asarray(episode_over, dtype=bool), 0.0, potential
    ).astype(np.float32)
    return (float(coef) * (next_potential - prev_potential)).astype(
        np.float32
    ), next_potential
