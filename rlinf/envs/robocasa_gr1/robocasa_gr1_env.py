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

"""RLinf environment adapter for the official RoboCasa GR1 tabletop tasks."""

from __future__ import annotations

import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.robocasa.venv import RobocasaSubprocEnv
from rlinf.envs.robocasa_gr1.progress import (
    compute_progress_stages,
    drawer_progress_potential,
    potential_shaping_reward,
    task_general_progress_potential,
)
from rlinf.envs.robocasa_gr1.seed_pool import (
    eval_rounds_required,
    load_task_seeds,
    select_multitask_process_seed_groups,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


def configure_simulator_egl(device_id: int) -> None:
    """Pin a simulator subprocess to NVIDIA EGL without software fallback."""
    vendor_manifest = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
    if not vendor_manifest or "nvidia" not in vendor_manifest.lower():
        raise RuntimeError(
            "RoboCasa GR1 requires an NVIDIA EGL vendor manifest via "
            "__EGL_VENDOR_LIBRARY_FILENAMES; refusing to fall back to Mesa."
        )
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(device_id))


def configure_simulator_mesa(
    device_id: int = 8, llvmpipe_threads: int | None = None
) -> None:
    """Select the Mesa EGL backend used by StarVLA's reference evaluator.

    This function runs inside the isolated simulator subprocess before
    RoboCasa, Robosuite, MuJoCo, or OpenGL is imported. CUDA remains available
    to the parent rollout worker, while the simulator itself is prevented from
    accidentally selecting an NVIDIA EGL device.
    """
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(device_id))
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
    if llvmpipe_threads is not None:
        threads = int(llvmpipe_threads)
        if threads <= 0:
            raise ValueError("llvmpipe_threads must be positive")
        os.environ["LP_NUM_THREADS"] = str(threads)


class RoboCasaSubtaskSignalWrapper(gym.Wrapper):
    """Expose task-provided progress signals without changing task success."""

    def _progress_info(self) -> dict[str, dict]:
        task_env = getattr(self.env.unwrapped, "env", None)
        signal_fn = getattr(task_env, "get_subtask_term_signals", None)
        signals = (
            {} if signal_fn is None else {k: int(v) for k, v in signal_fn().items()}
        )
        return {
            "subtask_signals": signals,
            "progress_stages": compute_progress_stages(task_env),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._progress_info())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(self._progress_info())
        return obs, reward, terminated, truncated, info


def resolve_task_names(cfg) -> list[str]:
    """Return the ordered task list from ``task_names`` or the single ``task_name``.

    Every task is a registered ``gr1_unified/...`` Gym id. The list is
    order-sensitive: task index ``t`` owns every global seed group ``g`` with
    ``g % num_tasks == t`` and is reported as ``sample_task == t``.
    """
    task_names = cfg.get("task_names", None)
    if task_names is None or len(task_names) == 0:
        return [str(cfg.task_name)]
    if isinstance(task_names, str):
        task_names = [task_names]
    names = [str(name) for name in task_names]
    if len(set(names)) != len(names):
        raise ValueError(f"task_names contains duplicates: {names}")
    return names


class RoboCasaGR1Env(gym.Env):
    """Vectorized GR1 tabletop tasks with true grouped-seed GRPO semantics.

    One or many tasks: each simulator subprocess is built for one task and
    keeps it for the whole run; seeds are drawn per task from the manifest.
    """

    metadata = {"render_fps": 20}
    _IMAGE_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
    _ACTION_SLICES = {
        "action.left_arm": slice(0, 7),
        "action.right_arm": slice(7, 14),
        "action.left_hand": slice(14, 20),
        "action.right_hand": slice(20, 26),
        "action.waist": slice(26, 29),
    }

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.group_size = int(cfg.group_size)
        if self.num_envs % self.group_size != 0:
            raise ValueError(
                f"num_envs={self.num_envs} must be divisible by group_size={self.group_size}"
            )
        self.num_group = self.num_envs // self.group_size
        self.task_names = resolve_task_names(cfg)
        self.num_tasks = len(self.task_names)
        # Kept for single-task callers and logging; multi-task code must use
        # ``task_names[env_task_ids[i]]``.
        self.task_name = self.task_names[0]
        self.seed = int(cfg.seed) + self.seed_offset
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.use_rel_reward = bool(cfg.use_rel_reward)
        shaping_cfg = cfg.get("subtask_reward_shaping", {})
        self.subtask_reward_shaping = bool(shaping_cfg.get("enabled", False))
        self.grasp_reward = float(shaping_cfg.get("grasp_object", 0.1))
        self.in_drawer_reward = float(shaping_cfg.get("obj_in_drawer", 0.5))
        self.success_reward = float(shaping_cfg.get("success", 1.0))
        if not (
            0.0 <= self.grasp_reward <= self.in_drawer_reward <= self.success_reward
        ):
            raise ValueError(
                "RoboCasa GR1 subtask rewards must satisfy "
                "0 <= grasp_object <= obj_in_drawer <= success"
            )
        self.shaping_mode = str(shaping_cfg.get("mode", "drawer")).strip().lower()
        if self.shaping_mode not in ("drawer", "task_general"):
            raise ValueError(
                "subtask_reward_shaping.mode must be 'drawer' or 'task_general', "
                f"got {self.shaping_mode!r}"
            )
        self.stage_rewards = {
            "grasped": float(shaping_cfg.get("grasped", 0.2)),
            "placed": float(shaping_cfg.get("placed", 0.5)),
            "released": float(shaping_cfg.get("released", 0.7)),
        }
        if not (
            0.0
            <= self.stage_rewards["grasped"]
            <= self.stage_rewards["placed"]
            <= self.stage_rewards["released"]
            <= self.success_reward
        ):
            raise ValueError(
                "RoboCasa GR1 task-general stage rewards must satisfy "
                "0 <= grasped <= placed <= released <= success"
            )
        self.shaping_coef = float(shaping_cfg.get("coef", 1.0))
        if self.shaping_coef < 0.0:
            raise ValueError("subtask_reward_shaping.coef must be non-negative")
        self.action_steps_per_chunk = cfg.get("action_steps_per_chunk", None)
        if self.action_steps_per_chunk is not None:
            self.action_steps_per_chunk = int(self.action_steps_per_chunk)
            if self.action_steps_per_chunk <= 0:
                raise ValueError("action_steps_per_chunk must be positive")
        self._is_start = True
        self._selection_round = 0

        self.seed_pools = [
            load_task_seeds(
                cfg.seed_manifest,
                task_name,
                expected_size=cfg.get("seed_pool_size", None),
            )
            for task_name in self.task_names
        ]
        self.env_task_ids: Optional[np.ndarray] = None
        self.eval_rounds = 1
        if bool(cfg.get("is_eval", False)):
            valid_seed_count = cfg.get("eval_seed_count", None)
            if valid_seed_count is None:
                valid_seed_count = min(len(pool) for pool in self.seed_pools)
            self.eval_rounds = eval_rounds_required(
                self.num_tasks,
                self.num_group * self.total_num_processes,
                int(valid_seed_count),
            )
        self._assign_seed_groups()
        self.env = RobocasaSubprocEnv(self._get_env_fns())

        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_potential = np.zeros(self.num_envs, dtype=np.float32)
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.grasped_once = np.zeros(self.num_envs, dtype=bool)
        self.obj_in_drawer_once = np.zeros(self.num_envs, dtype=bool)
        self.grasp_first_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.obj_in_drawer_first_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.placed_once = np.zeros(self.num_envs, dtype=bool)
        self.released_once = np.zeros(self.num_envs, dtype=bool)
        self.placed_first_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.released_first_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.success_first_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.current_raw_obs = None

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    def _assign_seed_groups(self) -> None:
        is_eval = bool(self.cfg.get("is_eval", False))
        group_task_ids, group_seeds, group_valid, global_group_ids = (
            select_multitask_process_seed_groups(
                self.seed_pools,
                groups_per_process=self.num_group,
                process_index=self.seed_offset,
                total_processes=self.total_num_processes,
                selection_round=self._selection_round,
                sampler_seed=int(self.cfg.get("seed_sampler_seed", self.cfg.seed)),
                shuffle=not is_eval,
                valid_seed_count=self.cfg.get("eval_seed_count", None),
            )
        )
        env_task_ids = np.repeat(group_task_ids, self.group_size)
        if self.env_task_ids is None:
            self.env_task_ids = env_task_ids
        elif not np.array_equal(self.env_task_ids, env_task_ids):
            # Subprocesses are built once per task; the assignment must not
            # drift between rollout rounds.
            raise RuntimeError("RoboCasa GR1 task assignment changed between rounds")
        self.group_seeds = group_seeds
        self.env_seeds = np.repeat(group_seeds, self.group_size)
        self.metric_valid = np.repeat(group_valid, self.group_size)
        self.global_group_ids = np.repeat(global_group_ids, self.group_size)
        self.trajectory_ids = np.tile(np.arange(self.group_size), self.num_group)

    def update_reset_state_ids(self):
        if bool(self.cfg.get("is_eval", False)):
            # Ordered evaluation walks each task's seed prefix over
            # ``eval_rounds`` rounds (one per eval_rollout_epoch) and wraps, so
            # every evaluation call visits the same seeds in the same order.
            self._selection_round = (self._selection_round + 1) % self.eval_rounds
        else:
            self._selection_round += 1
        self._assign_seed_groups()

    def set_seed_selection_round(self, selection_round: int) -> None:
        """Restore the deterministic training-seed cursor after a resume."""
        selection_round = int(selection_round)
        if selection_round < 0:
            raise ValueError("selection_round must be non-negative")
        if bool(self.cfg.get("is_eval", False)):
            return
        self._selection_round = selection_round
        self._assign_seed_groups()

    def _get_env_fns(self):
        env_fns = []
        renderer_backend = (
            str(self.cfg.get("renderer_backend", "nvidia")).strip().lower()
        )
        if renderer_backend == "nvidia":
            configured_device = self.cfg.get("egl_device", None)
            if configured_device is None:
                configured_device = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
            if configured_device is None:
                raise RuntimeError(
                    "No EGL device was assigned to this RoboCasa GR1 EnvWorker."
                )
            renderer_device = int(configured_device)
        elif renderer_backend == "mesa":
            renderer_device = int(self.cfg.get("mesa_egl_device", 8))
        else:
            raise ValueError(
                "renderer_backend must be either 'nvidia' or 'mesa', got "
                f"{renderer_backend!r}"
            )
        llvmpipe_threads = self.cfg.get("llvmpipe_threads", None)
        for env_id in range(self.num_envs):

            def env_fn(
                task_name=self.task_names[int(self.env_task_ids[env_id])],
                backend=renderer_backend,
                device_id=renderer_device,
                software_threads=llvmpipe_threads,
            ):
                if backend == "nvidia":
                    # EGL device indices refer to the physical NVIDIA EGL
                    # device list. CUDA visibility remains pinned in this mode.
                    configure_simulator_egl(device_id)
                else:
                    configure_simulator_mesa(device_id, software_threads)
                # RoboSuite repeats expected GR1 controller-component warnings
                # at every seeded scene reset. Keep errors visible without
                # flooding Ray's driver stream or duplicating root-log output.
                import logging

                import robocasa  # noqa: F401 - registers GR1 Gym environments
                import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

                robosuite_logger = logging.getLogger("robosuite_logs")
                robosuite_logger.setLevel(logging.ERROR)
                robosuite_logger.propagate = False
                return RoboCasaSubtaskSignalWrapper(
                    gym.make(task_name, enable_render=True, disable_env_checker=True)
                )

            env_fns.append(env_fn)
        return env_fns

    def _reset_subprocesses(self, env_idx: np.ndarray):
        for idx in env_idx:
            self.env.workers[int(idx)].send(None, seed=int(self.env_seeds[int(idx)]))
        raw_obs = []
        infos = []
        for idx in env_idx:
            result = self.env.workers[int(idx)].recv()
            if not (isinstance(result, tuple) and len(result) == 2):
                raise RuntimeError("RoboCasa GR1 reset must return (observation, info)")
            raw_obs.append(result[0])
            infos.append(result[1])
        return np.asarray(raw_obs, dtype=object), infos

    def _wrap_obs(self, raw_obs):
        observations = list(raw_obs)
        images = np.stack(
            [np.asarray(obs[self._IMAGE_KEY], dtype=np.uint8) for obs in observations]
        )
        task_descriptions = [
            str(obs.get("annotation.human.coarse_action", "")) for obs in observations
        ]
        self.current_raw_obs = images
        return {
            "main_images": torch.from_numpy(images),
            "task_descriptions": task_descriptions,
        }

    def capture_image(self):
        return self.current_raw_obs

    def _reset_metrics(self, env_idx: np.ndarray) -> None:
        self.prev_step_reward[env_idx] = 0.0
        self.prev_potential[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0
        self.success_once[env_idx] = False
        self.grasped_once[env_idx] = False
        self.obj_in_drawer_once[env_idx] = False
        self.placed_once[env_idx] = False
        self.released_once[env_idx] = False
        self.grasp_first_step[env_idx] = -1
        self.obj_in_drawer_first_step[env_idx] = -1
        self.placed_first_step[env_idx] = -1
        self.released_first_step[env_idx] = -1
        self.success_first_step[env_idx] = -1
        self.returns[env_idx] = 0.0

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        options: Optional[dict] = None,
    ):
        del options
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        elif isinstance(env_idx, int):
            env_idx = np.asarray([env_idx])
        else:
            env_idx = np.asarray(env_idx)
        self._is_start = False
        raw_obs, _ = self._reset_subprocesses(env_idx)
        self._reset_metrics(env_idx)
        return self._wrap_obs(raw_obs), {}

    @classmethod
    def _action_dict(cls, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action)
        if action.shape != (29,):
            raise ValueError(f"Expected one 29-D GR1 action, got {action.shape}")
        return {key: action[value].copy() for key, value in cls._ACTION_SLICES.items()}

    def _record_metrics(self, step_reward, terminations, infos):
        self.returns += step_reward * (~self.success_once)
        first_success = terminations & (~self.success_once)
        self.success_first_step[first_success] = self.elapsed_steps[first_success]
        self.success_once |= terminations
        episode_info = {
            # Metric consumers and the video overlay handle numeric scalars, but
            # NumPy bool scalars otherwise fall through to the unsupported-type
            # warning path. These copies are logging-only; the internal state
            # remains boolean for reward and termination bookkeeping.
            "success_once": self.success_once.astype(np.float32, copy=True),
            "grasped_once": self.grasped_once.astype(np.float32, copy=True),
            "obj_in_drawer_once": self.obj_in_drawer_once.astype(np.float32, copy=True),
            "grasp_first_step": self.grasp_first_step.copy(),
            "obj_in_drawer_first_step": self.obj_in_drawer_first_step.copy(),
            "placed_once": self.placed_once.astype(np.float32, copy=True),
            "released_once": self.released_once.astype(np.float32, copy=True),
            "placed_first_step": self.placed_first_step.copy(),
            "released_first_step": self.released_first_step.copy(),
            "success_first_step": self.success_first_step.copy(),
            "return": self.returns.copy(),
            "episode_len": self.elapsed_steps.copy(),
            "reward": self.returns / np.maximum(self.elapsed_steps, 1),
            "sample_seed": self.env_seeds.copy(),
            "sample_task": self.env_task_ids.copy(),
            "sample_group": self.global_group_ids.copy(),
            "sample_trajectory": self.trajectory_ids.copy(),
            "metric_valid": self.metric_valid.copy(),
        }
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _calc_step_reward(self, terminations, truncations, info_lists):
        drawer_potential, drawer_grasped, in_drawer = drawer_progress_potential(
            info_lists,
            terminations,
            grasp_reward=self.grasp_reward,
            in_drawer_reward=self.in_drawer_reward,
            success_reward=self.success_reward,
        )
        general_potential, stages = task_general_progress_potential(
            info_lists,
            terminations,
            stage_rewards=self.stage_rewards,
            success_reward=self.success_reward,
        )
        # The drawer signals exist only for the drawer family; the generic
        # stages exist for every task, so grasp bookkeeping takes either.
        grasped = drawer_grasped | stages["grasped"]
        released = stages["placed"] & stages["released"]
        for reached, reached_once, first_step in (
            (grasped, self.grasped_once, self.grasp_first_step),
            (in_drawer, self.obj_in_drawer_once, self.obj_in_drawer_first_step),
            (stages["placed"], self.placed_once, self.placed_first_step),
            (released, self.released_once, self.released_first_step),
        ):
            first = reached & (~reached_once)
            first_step[first] = self.elapsed_steps[first]
        self.grasped_once |= grasped
        self.obj_in_drawer_once |= in_drawer
        self.placed_once |= stages["placed"]
        self.released_once |= released

        # Task reward: unchanged binary success, in the relative form the
        # recipes use (+1 on the step success first fires, -1 if it is undone).
        task_level = np.asarray(terminations, dtype=np.float32)
        task_reward = task_level - self.prev_step_reward
        self.prev_step_reward = task_level
        if not self.use_rel_reward:
            task_reward = task_level
        if not self.subtask_reward_shaping:
            return task_reward

        # Potential-based shaping, added rather than substituted, with the
        # potential forced to zero at the episode boundary. The shaping terms
        # then telescope to -Phi(s_0) = 0 over an episode, so every episode's
        # undiscounted return is exactly the binary one and the optimal policy
        # is unchanged (Ng et al., 1999); only the temporal distribution of
        # credit changes. Substituting the potential instead would pay a
        # "placed but never closed" rollout more than some tasks' success rate.
        potential = (
            drawer_potential if self.shaping_mode == "drawer" else general_potential
        )
        shaping_reward, self.prev_potential = potential_shaping_reward(
            potential,
            self.prev_potential,
            episode_over=truncations,
            coef=self.shaping_coef,
        )
        return task_reward + shaping_reward

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            if not self._is_start:
                raise ValueError("Actions may be None only for the initial reset")
            obs, infos = self.reset()
            zeros = torch.zeros(self.num_envs, dtype=torch.bool)
            return obs, zeros.float(), zeros, zeros, infos

        del auto_reset
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        action_dicts = np.asarray(
            [self._action_dict(action) for action in np.asarray(actions)], dtype=object
        )
        raw_obs, _raw_rewards, _raw_terminated, _raw_truncated, info_lists = (
            self.env.step(action_dicts)
        )
        self._elapsed_steps += 1
        terminations = np.asarray(
            [bool(info.get("success", False)) for info in info_lists], dtype=bool
        )
        truncations = self._elapsed_steps >= int(self.cfg.max_episode_steps)
        step_reward = self._calc_step_reward(terminations, truncations, info_lists)
        obs = self._wrap_obs(raw_obs)
        infos = list_of_dict_to_dict_of_list(list(info_lists))
        infos = self._record_metrics(step_reward, terminations, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        if (
            self.action_steps_per_chunk is not None
            and self.action_steps_per_chunk > chunk_actions.shape[1]
        ):
            raise ValueError(
                "action_steps_per_chunk exceeds the policy action chunk: "
                f"{self.action_steps_per_chunk} > {chunk_actions.shape[1]}"
            )
        action_steps = (
            chunk_actions.shape[1]
            if self.action_steps_per_chunk is None
            else self.action_steps_per_chunk
        )
        obs_list = []
        rewards = []
        terminations = []
        truncations = []
        infos_list = []
        for action_step in range(action_steps):
            obs, reward, terminated, truncated, infos = self.step(
                chunk_actions[:, action_step], auto_reset=False
            )
            obs_list.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos_list.append(infos)
        return (
            obs_list,
            torch.stack(rewards, dim=1),
            torch.stack(terminations, dim=1),
            torch.stack(truncations, dim=1),
            infos_list,
        )

    def close(self):
        self.env.close()
