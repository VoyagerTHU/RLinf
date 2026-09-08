"""Tests for monotonic RoboCasa-GR1 subtask reward shaping."""

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


def test_relative_shaping_rewards_each_milestone_only_once():
    env = RoboCasaGR1Env.__new__(RoboCasaGR1Env)
    env.subtask_reward_shaping = True
    env.grasp_reward = 0.1
    env.in_drawer_reward = 0.5
    env.success_reward = 1.0
    env.use_rel_reward = True
    env.prev_step_reward = np.zeros(1, dtype=np.float32)
    env.grasped_once = np.zeros(1, dtype=bool)
    env.obj_in_drawer_once = np.zeros(1, dtype=bool)
    env.grasp_first_step = np.full(1, -1, dtype=np.int32)
    env.obj_in_drawer_first_step = np.full(1, -1, dtype=np.int32)
    env._elapsed_steps = np.asarray([12], dtype=np.int32)

    def reward(signals, success=False):
        return env._calc_step_reward(
            np.asarray([success]), [{"subtask_signals": signals}]
        )

    np.testing.assert_allclose(reward({"grasp_object": 1}), [0.1])
    assert env.grasp_first_step.tolist() == [12]
    env._elapsed_steps[:] = 24
    np.testing.assert_allclose(reward({}), [0.0])
    np.testing.assert_allclose(reward({"obj_in_drawer": 1}), [0.4])
    assert env.obj_in_drawer_first_step.tolist() == [24]
    np.testing.assert_allclose(reward({"grasp_object": 1}), [0.0])
    np.testing.assert_allclose(reward({}, success=True), [0.5])
    assert env.grasped_once.tolist() == [True]
    assert env.obj_in_drawer_once.tolist() == [True]


def test_binary_reward_still_tracks_subtask_diagnostics():
    env = RoboCasaGR1Env.__new__(RoboCasaGR1Env)
    env.subtask_reward_shaping = False
    env.grasp_reward = 0.1
    env.in_drawer_reward = 0.5
    env.success_reward = 1.0
    env.use_rel_reward = True
    env.prev_step_reward = np.zeros(1, dtype=np.float32)
    env.grasped_once = np.zeros(1, dtype=bool)
    env.obj_in_drawer_once = np.zeros(1, dtype=bool)
    env.grasp_first_step = np.full(1, -1, dtype=np.int32)
    env.obj_in_drawer_first_step = np.full(1, -1, dtype=np.int32)
    env._elapsed_steps = np.asarray([12], dtype=np.int32)

    reward = env._calc_step_reward(
        np.asarray([False]),
        [{"subtask_signals": {"grasp_object": 1}}],
    )
    np.testing.assert_allclose(reward, [0.0])
    assert env.grasped_once.tolist() == [True]
    assert env.grasp_first_step.tolist() == [12]

    env._elapsed_steps[:] = 24
    reward = env._calc_step_reward(
        np.asarray([False]),
        [{"subtask_signals": {"obj_in_drawer": 1}}],
    )
    np.testing.assert_allclose(reward, [0.0])
    assert env.obj_in_drawer_once.tolist() == [True]
    assert env.obj_in_drawer_first_step.tolist() == [24]

    reward = env._calc_step_reward(np.asarray([True]), [{"subtask_signals": {}}])
    np.testing.assert_allclose(reward, [1.0])
