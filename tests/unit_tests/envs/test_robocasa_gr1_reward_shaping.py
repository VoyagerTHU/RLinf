"""Tests for RoboCasa-GR1 potential-based subtask reward shaping."""

import numpy as np

from rlinf.envs.robocasa_gr1.robocasa_gr1_env import (
    RoboCasaGR1Env,
    drawer_progress_potential,
)


def test_drawer_progress_potential_orders_task_milestones():
    infos = [
        {"subtask_signals": {}},
        {"subtask_signals": {"grasp_object": 1}},
        {"subtask_signals": {"grasp_object": 1, "obj_in_drawer": 1}},
        {"subtask_signals": {}},
    ]

    potential, grasped, in_drawer = drawer_progress_potential(
        infos,
        np.asarray([False, False, False, True]),
        grasp_reward=0.1,
        in_drawer_reward=0.5,
        success_reward=1.0,
    )

    np.testing.assert_allclose(potential, [0.0, 0.1, 0.5, 1.0])
    np.testing.assert_array_equal(grasped, [False, True, True, False])
    np.testing.assert_array_equal(in_drawer, [False, False, True, False])


def _stub_env(*, shaping: bool, mode: str = "drawer"):
    """A bare environment carrying only the state ``_calc_step_reward`` touches."""
    env = RoboCasaGR1Env.__new__(RoboCasaGR1Env)
    env.subtask_reward_shaping = shaping
    env.shaping_mode = mode
    env.shaping_coef = 1.0
    env.grasp_reward = 0.1
    env.in_drawer_reward = 0.5
    env.success_reward = 1.0
    env.stage_rewards = {"grasped": 0.2, "placed": 0.5, "released": 0.7}
    env.use_rel_reward = True
    env.prev_step_reward = np.zeros(1, dtype=np.float32)
    env.prev_potential = np.zeros(1, dtype=np.float32)
    env.grasped_once = np.zeros(1, dtype=bool)
    env.obj_in_drawer_once = np.zeros(1, dtype=bool)
    env.placed_once = np.zeros(1, dtype=bool)
    env.released_once = np.zeros(1, dtype=bool)
    env.grasp_first_step = np.full(1, -1, dtype=np.int32)
    env.obj_in_drawer_first_step = np.full(1, -1, dtype=np.int32)
    env.placed_first_step = np.full(1, -1, dtype=np.int32)
    env.released_first_step = np.full(1, -1, dtype=np.int32)
    env._elapsed_steps = np.asarray([12], dtype=np.int32)
    return env


def test_drawer_shaping_pays_each_milestone_when_it_is_reached():
    env = _stub_env(shaping=True)

    def reward(signals, success=False, over=False):
        return env._calc_step_reward(
            np.asarray([success]), np.asarray([over]), [{"subtask_signals": signals}]
        )

    np.testing.assert_allclose(reward({"grasp_object": 1}), [0.1])
    assert env.grasp_first_step.tolist() == [12]
    env._elapsed_steps[:] = 24
    # Losing the grasp charges the milestone back, which a monotone potential
    # would not do; that is what keeps the episode return unchanged.
    np.testing.assert_allclose(reward({}), [-0.1])
    np.testing.assert_allclose(reward({"grasp_object": 1, "obj_in_drawer": 1}), [0.5])
    assert env.obj_in_drawer_first_step.tolist() == [24]
    # Success pays the remaining potential plus the binary task reward.
    np.testing.assert_allclose(reward({"obj_in_drawer": 1}, success=True), [1.5])
    assert env.grasped_once.tolist() == [True]
    assert env.obj_in_drawer_once.tolist() == [True]


def test_shaping_leaves_the_episode_return_unchanged():
    env = _stub_env(shaping=True)
    route = [
        ({"grasp_object": 1}, False),
        ({"grasp_object": 1, "obj_in_drawer": 1}, False),
        ({}, False),
        ({"grasp_object": 1, "obj_in_drawer": 1}, False),
    ]
    total = 0.0
    for step, (signals, success) in enumerate(route):
        total += float(
            env._calc_step_reward(
                np.asarray([success]),
                np.asarray([step == len(route) - 1]),
                [{"subtask_signals": signals}],
            )[0]
        )
    # No success anywhere on this route, so the shaped return must be zero.
    assert abs(total) < 1e-6


def test_task_general_mode_scores_the_shared_milestones():
    env = _stub_env(shaping=True, mode="task_general")

    def reward(stages, success=False, over=False):
        return env._calc_step_reward(
            np.asarray([success]), np.asarray([over]), [{"progress_stages": stages}]
        )

    np.testing.assert_allclose(reward({"grasped": True}), [0.2])
    np.testing.assert_allclose(reward({"grasped": True, "placed": True}), [0.3])
    np.testing.assert_allclose(reward({"placed": True, "released": True}), [0.2])
    assert env.placed_once.tolist() == [True]
    assert env.released_once.tolist() == [True]


def test_binary_reward_still_tracks_subtask_diagnostics():
    env = _stub_env(shaping=False)

    reward = env._calc_step_reward(
        np.asarray([False]),
        np.asarray([False]),
        [{"subtask_signals": {"grasp_object": 1}}],
    )
    np.testing.assert_allclose(reward, [0.0])
    assert env.grasped_once.tolist() == [True]
    assert env.grasp_first_step.tolist() == [12]

    env._elapsed_steps[:] = 24
    reward = env._calc_step_reward(
        np.asarray([False]),
        np.asarray([False]),
        [{"subtask_signals": {"obj_in_drawer": 1}}],
    )
    np.testing.assert_allclose(reward, [0.0])
    assert env.obj_in_drawer_once.tolist() == [True]
    assert env.obj_in_drawer_first_step.tolist() == [24]

    reward = env._calc_step_reward(
        np.asarray([True]), np.asarray([False]), [{"subtask_signals": {}}]
    )
    np.testing.assert_allclose(reward, [1.0])
