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

"""Task-general progress potential for the RoboCasa GR1 tasks."""

import importlib.util
import pathlib

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "robocasa_gr1_progress",
    pathlib.Path(__file__).resolve().parents[2] / "rlinf/envs/robocasa_gr1/progress.py",
)
progress = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(progress)

WEIGHTS = {"grasped": 0.2, "placed": 0.5, "released": 0.7}


def _info(**stages):
    return {
        "progress_stages": {
            name: bool(stages.get(name, False)) for name in progress.PROGRESS_STAGES
        }
    }


def _potential(infos, terminations):
    values, _ = progress.task_general_progress_potential(
        infos,
        np.asarray(terminations),
        stage_rewards=WEIGHTS,
        success_reward=1.0,
    )
    return values


def test_stages_form_an_increasing_ladder() -> None:
    infos = [
        _info(),
        _info(grasped=True),
        _info(grasped=True, placed=True),
        _info(placed=True, released=True),
    ]
    assert np.allclose(_potential(infos, [False] * 4), [0.0, 0.2, 0.5, 0.7])


def test_untouched_object_at_reset_scores_zero() -> None:
    # Both grippers are trivially far from an object nobody has touched, so
    # "released" must not pay out before "placed" holds.
    assert _potential([_info(released=True)], [False]).tolist() == [0.0]


def test_success_pins_the_potential_to_one() -> None:
    # Even a state whose visible stages look partial scores 1.0 once the task's
    # own success check fires, so the potential never disagrees with the metric.
    assert _potential([_info(grasped=True)], [True]).tolist() == [1.0]


def test_missing_stages_are_treated_as_not_reached() -> None:
    # Families whose simulator exposes no container predicate still produce a
    # finite potential instead of raising.
    assert _potential([{"progress_stages": {}}, {}], [False, False]).tolist() == [
        0.0,
        0.0,
    ]


def test_dropping_the_object_lowers_the_potential() -> None:
    # The unclamped potential difference is what makes a drop cost reward.
    held = _potential([_info(grasped=True, placed=True)], [False])[0]
    dropped = _potential([_info()], [False])[0]
    assert dropped - held < 0


def test_stage_masks_are_returned_per_environment() -> None:
    _, stages = progress.task_general_progress_potential(
        [_info(grasped=True), _info(placed=True, released=True)],
        np.asarray([False, False]),
        stage_rewards=WEIGHTS,
        success_reward=1.0,
    )
    assert stages["grasped"].tolist() == [True, False]
    assert (stages["placed"] & stages["released"]).tolist() == [False, True]


def test_drawer_potential_is_unchanged() -> None:
    infos = [
        {"subtask_signals": {}},
        {"subtask_signals": {"grasp_object": 1}},
        {"subtask_signals": {"grasp_object": 1, "obj_in_drawer": 1}},
    ]
    values, grasped, in_drawer = progress.drawer_progress_potential(
        infos,
        np.asarray([False, False, True]),
        grasp_reward=0.1,
        in_drawer_reward=0.5,
        success_reward=1.0,
    )
    assert np.allclose(values, [0.0, 0.1, 1.0])
    assert grasped.tolist() == [False, True, True]
    assert in_drawer.tolist() == [False, False, True]


def _episode_shaping(potentials, coef=1.0):
    """Shaping rewards for one episode whose last step ends the episode."""
    prev = np.zeros(1, dtype=np.float32)
    rewards = []
    for step, value in enumerate(potentials):
        reward, prev = progress.potential_shaping_reward(
            np.asarray([value], dtype=np.float32),
            prev,
            episode_over=np.asarray([step == len(potentials) - 1]),
            coef=coef,
        )
        rewards.append(float(reward[0]))
    return rewards


def test_shaping_telescopes_to_zero_over_an_episode() -> None:
    # Whatever route the episode takes, the shaping terms cancel, so the
    # episodic return is exactly the unshaped one.
    for route in ([0.2, 0.5, 0.7, 1.0], [0.2, 0.0, 0.2, 0.5], [0.0] * 5, [0.2, 0.5]):
        assert abs(sum(_episode_shaping(route))) < 1e-6


def test_shaping_pays_progress_early_and_charges_for_undoing_it() -> None:
    rewards = _episode_shaping([0.2, 0.5, 0.2, 0.5, 1.0])
    assert rewards[0] > 0 and rewards[1] > 0  # grasp, then place
    assert rewards[2] < 0  # dropped it back out
    assert abs(sum(rewards)) < 1e-6


def test_coef_scales_the_shaping_without_breaking_telescoping() -> None:
    small = _episode_shaping([0.2, 0.5, 1.0], coef=0.25)
    full = _episode_shaping([0.2, 0.5, 1.0], coef=1.0)
    assert np.allclose(np.asarray(small)[:2], 0.25 * np.asarray(full)[:2])
    assert abs(sum(small)) < 1e-6


def test_running_episodes_keep_their_potential() -> None:
    # A mid-episode step must not zero the potential, or credit would be paid
    # twice on the following step.
    reward, prev = progress.potential_shaping_reward(
        np.asarray([0.5], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
        episode_over=np.asarray([False]),
    )
    assert np.allclose(reward, [0.3]) and np.allclose(prev, [0.5])


def test_bootstrapping_mode_does_not_charge_back_at_the_episode_end() -> None:
    # With zero_at_episode_end=False the last step is an ordinary transition,
    # so a completed task is not taught to a critic as a large negative reward.
    prev = np.zeros(1, dtype=np.float32)
    rewards = []
    for step, value in enumerate([0.2, 0.5, 1.0]):
        reward, prev = progress.potential_shaping_reward(
            np.asarray([value], dtype=np.float32),
            prev,
            episode_over=np.asarray([step == 2]),
            zero_at_episode_end=False,
        )
        rewards.append(float(reward[0]))
    assert np.allclose(rewards, [0.2, 0.3, 0.5])
    assert all(r >= 0 for r in rewards)
    assert np.allclose(prev, [1.0])


def test_zeroing_mode_still_telescopes() -> None:
    prev = np.zeros(1, dtype=np.float32)
    total = 0.0
    for step, value in enumerate([0.2, 0.5, 1.0]):
        reward, prev = progress.potential_shaping_reward(
            np.asarray([value], dtype=np.float32),
            prev,
            episode_over=np.asarray([step == 2]),
            zero_at_episode_end=True,
        )
        total += float(reward[0])
    assert abs(total) < 1e-6


def test_standing_still_in_a_high_potential_state_pays_nothing_with_gamma() -> None:
    # Without the discount, holding a grasp paid (1 - gamma) * Phi every step,
    # which a bootstrapping critic banks as a perpetuity; the policy learned to
    # grasp and stop. With gamma the reward for staying put is zero.
    held = np.asarray([0.2], dtype=np.float32)
    plain, _ = progress.potential_shaping_reward(
        held, held, episode_over=np.asarray([False]), zero_at_episode_end=False
    )
    discounted, _ = progress.potential_shaping_reward(
        held,
        held,
        episode_over=np.asarray([False]),
        zero_at_episode_end=False,
        gamma=0.999,
    )
    assert float(plain[0]) == 0.0
    assert float(discounted[0]) < 0.0  # standing still now costs, never pays


def test_progress_still_pays_under_the_discount() -> None:
    reward, _ = progress.potential_shaping_reward(
        np.asarray([0.5], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
        episode_over=np.asarray([False]),
        zero_at_episode_end=False,
        gamma=0.999,
    )
    assert 0.29 < float(reward[0]) < 0.3


def test_gamma_defaults_to_one_for_episodic_recipes() -> None:
    # The PPO recipe relies on zeroing rather than the discount, so the default
    # must keep its exact telescoping behaviour.
    reward, _ = progress.potential_shaping_reward(
        np.asarray([0.5], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
        episode_over=np.asarray([False]),
    )
    assert np.allclose(reward, [0.3])
