# Copyright 2025 The RLinf Authors.
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

"""RLinf environment backed by the ``solver`` kitchen simulation.

The environment follows the RoboCasa-GR1 contract used by the StarVLA GRPO
recipes: grouped seeds, no auto reset, chunked stepping, image observations
in ``main_images`` and episode metrics in ``infos["episode"]``. Physics and
rendering run in the solver core (see :mod:`solver_core`), either in-process
or in a subprocess with its own interpreter (see :mod:`backend`).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.robocasa_gr1.seed_pool import (
    eval_rounds_required,
    load_task_seeds,
    select_process_seed_groups,
)
from rlinf.envs.solver_kitchen.backend import SolverBackend, make_backend
from rlinf.envs.utils import to_tensor

DEFAULT_TASK_DESCRIPTION = "reach the knife handle with the right gripper"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _to_plain(value: Any) -> Any:
    """Convert OmegaConf containers into plain Python objects."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is an RLinf dependency
        pass
    return value


class SolverKitchenEnv(gym.Env):
    """Vectorized solver kitchen reach task with grouped-seed GRPO semantics."""

    metadata = {"render_fps": 20}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any = None,
        backend: Optional[SolverBackend] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.num_envs % self.group_size != 0:
            raise ValueError(
                f"num_envs={self.num_envs} must be divisible by group_size={self.group_size}"
            )
        self.num_group = self.num_envs // self.group_size
        self.seed = int(_cfg_get(cfg, "seed", 0)) + self.seed_offset
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", False))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.max_episode_steps = int(_cfg_get(cfg, "max_episode_steps", 300))
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.task_description = str(
            _cfg_get(cfg, "task_description", DEFAULT_TASK_DESCRIPTION)
        )
        self.reward_mode = str(_cfg_get(cfg, "reward_mode", "solver")).strip().lower()
        if self.reward_mode not in ("solver", "success"):
            raise ValueError("reward_mode must be 'solver' or 'success'")
        self.success_reward = float(_cfg_get(cfg, "success_reward", 1.0))
        self.reward_coef = float(_cfg_get(cfg, "reward_coef", 1.0))
        self.action_steps_per_chunk = _cfg_get(cfg, "action_steps_per_chunk", None)
        if self.action_steps_per_chunk is not None:
            self.action_steps_per_chunk = int(self.action_steps_per_chunk)
            if self.action_steps_per_chunk <= 0:
                raise ValueError("action_steps_per_chunk must be positive")
        self.render_enabled = bool(_cfg_get(cfg, "render", True))
        video_cfg = _cfg_get(cfg, "video_cfg", None)
        save_video = bool(_cfg_get(video_cfg, "save_video", False))
        render_every_step = _cfg_get(cfg, "render_every_step", None)
        # Video recording needs a frame per physics step; otherwise render
        # only when the policy consumes an observation (end of each chunk).
        self.render_every_step = (
            save_video if render_every_step is None else bool(render_every_step)
        )
        self.main_camera = _cfg_get(cfg, "main_camera", None)
        self.is_eval = bool(_cfg_get(cfg, "is_eval", False))
        self._is_start = True
        self._selection_round = 0
        self._closed = False

        # ---------------------------------------------------------- seeds
        self.seed_pool = self._build_seed_pool()
        self.eval_rounds = 1
        if self.is_eval:
            valid_seed_count = _cfg_get(cfg, "eval_seed_count", None)
            if valid_seed_count is None:
                valid_seed_count = len(self.seed_pool)
            self.eval_rounds = eval_rounds_required(
                1, self.num_group * self.total_num_processes, int(valid_seed_count)
            )
        self._assign_seed_groups()

        # -------------------------------------------------------- backend
        self.core_config = self._build_core_config()
        if backend is None:
            backend = make_backend(
                str(_cfg_get(cfg, "backend", "subprocess")),
                python_executable=self._resolve_solver_python(),
                extra_env=self._solver_process_env(),
            )
        self.backend = backend
        self.core_metadata = self.backend.init(self.core_config)
        self.action_dim = int(self.core_metadata["action_dim"])
        self.obs_dim = int(self.core_metadata["obs_dim"])
        self.camera_names = list(self.core_metadata.get("camera_names", []))
        if self.render_enabled and not self.camera_names:
            raise ValueError("rendering is enabled but the solver registered no cameras")
        if self.main_camera is None:
            self.main_camera_index = 0
        else:
            if self.main_camera not in self.camera_names:
                raise ValueError(
                    f"main_camera {self.main_camera!r} is not one of {self.camera_names}"
                )
            self.main_camera_index = self.camera_names.index(self.main_camera)
        if bool(self.core_metadata.get("normalized_actions", True)):
            low, high = -1.0, 1.0
        else:
            low = np.asarray(self.core_metadata["action_lower_limits"], dtype=np.float32)
            high = np.asarray(self.core_metadata["action_upper_limits"], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_dim, low, dtype=np.float32)
            if np.isscalar(low)
            else low,
            high=np.full(self.action_dim, high, dtype=np.float32)
            if np.isscalar(high)
            else high,
            dtype=np.float32,
        )
        # Gaussian policies (PPO-MLP) emit unbounded samples; keep them inside
        # the normalized range the solver expects.
        self.clip_actions = bool(_cfg_get(cfg, "clip_actions", True))
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
                )
            }
        )

        # -------------------------------------------------------- metrics
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.failure_once = np.zeros(self.num_envs, dtype=bool)
        self.success_first_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.last_distance = np.zeros(self.num_envs, dtype=np.float32)
        self.min_distance = np.full(self.num_envs, np.inf, dtype=np.float32)
        self.current_states: Optional[np.ndarray] = None
        self.current_images: Optional[np.ndarray] = None
        self.current_raw_obs: Optional[np.ndarray] = None
        self.render_count = 0

    # ------------------------------------------------------------ config
    def _build_seed_pool(self) -> list[int]:
        manifest = _cfg_get(self.cfg, "seed_manifest", None)
        pool_size = _cfg_get(self.cfg, "seed_pool_size", None)
        if manifest:
            task_name = str(_cfg_get(self.cfg, "task_name", "solver_kitchen"))
            return load_task_seeds(
                manifest,
                task_name,
                expected_size=None if pool_size is None else int(pool_size),
            )
        pool_size = 500 if pool_size is None else int(pool_size)
        if pool_size <= 0:
            raise ValueError("seed_pool_size must be positive")
        base = int(_cfg_get(self.cfg, "seed", 0))
        # Seeds must stay strictly positive int32 values.
        start = max(base, 0) + 1
        return list(range(start, start + pool_size))

    def _build_core_config(self) -> dict[str, Any]:
        sim = dict(_to_plain(_cfg_get(self.cfg, "sim", None)) or {})
        task = dict(_to_plain(_cfg_get(self.cfg, "task", None)) or {})
        torch_env = dict(_to_plain(_cfg_get(self.cfg, "torch_env", None)) or {})
        cameras = _to_plain(_cfg_get(self.cfg, "cameras", None))
        # Solver-side truncation must never fire before RLinf's own horizon,
        # otherwise KitchenTorchEnv flags `truncated` mid-episode.
        task.setdefault("max_episode_steps", self.max_episode_steps)
        task["max_episode_steps"] = max(int(task["max_episode_steps"]), self.max_episode_steps)
        torch_env["auto_reset"] = False
        core_config: dict[str, Any] = {
            "num_envs": self.num_envs,
            "device": str(sim.pop("device", "cuda:0")),
            "render": self.render_enabled,
            "task": task,
            "torch_env": torch_env,
        }
        for key in (
            "fps",
            "sim_substeps",
            "graph_capture",
            "deterministic_gpu",
            "deterministic_contacts",
            "deterministic_solver",
            "enable_shadows",
        ):
            if key in sim:
                core_config[key] = sim[key]
        unknown = sorted(set(sim) - set(core_config))
        if unknown:
            raise ValueError(f"unknown solver_kitchen sim keys: {unknown}")
        if cameras is not None:
            core_config["cameras"] = [dict(camera) for camera in cameras]
        return core_config

    def _resolve_solver_python(self) -> Optional[str]:
        python = _cfg_get(self.cfg, "solver_python", None)
        if python:
            return os.path.expanduser(str(python))
        return os.environ.get("SOLVER_PYTHON")

    def _solver_process_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        extra = _to_plain(_cfg_get(self.cfg, "solver_env_vars", None)) or {}
        for key, value in dict(extra).items():
            env[str(key)] = str(value)
        solver_root = _cfg_get(self.cfg, "solver_root", None) or os.environ.get(
            "SOLVER_ROOT"
        )
        if solver_root:
            env.setdefault("SOLVER_ROOT", str(solver_root))
        return env

    # ------------------------------------------------------------- seeds
    def _assign_seed_groups(self) -> None:
        group_seeds, group_valid, global_group_ids = select_process_seed_groups(
            self.seed_pool,
            groups_per_process=self.num_group,
            process_index=self.seed_offset,
            total_processes=self.total_num_processes,
            selection_round=self._selection_round,
            sampler_seed=int(_cfg_get(self.cfg, "seed_sampler_seed", self.seed)),
            shuffle=not self.is_eval,
            valid_seed_count=_cfg_get(self.cfg, "eval_seed_count", None),
        )
        self.group_seeds = np.asarray(group_seeds, dtype=np.int64)
        self.env_seeds = np.repeat(self.group_seeds, self.group_size)
        self.metric_valid = np.repeat(np.asarray(group_valid, dtype=bool), self.group_size)
        self.global_group_ids = np.repeat(
            np.asarray(global_group_ids, dtype=np.int64), self.group_size
        )
        self.trajectory_ids = np.tile(np.arange(self.group_size), self.num_group)

    def update_reset_state_ids(self) -> None:
        if self.is_eval:
            self._selection_round = (self._selection_round + 1) % self.eval_rounds
        else:
            self._selection_round += 1
        self._assign_seed_groups()

    def set_seed_selection_round(self, selection_round: int) -> None:
        selection_round = int(selection_round)
        if selection_round < 0:
            raise ValueError("selection_round must be non-negative")
        if self.is_eval:
            return
        self._selection_round = selection_round
        self._assign_seed_groups()

    # -------------------------------------------------------- properties
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    # ------------------------------------------------------- observations
    def _render_images(self) -> Optional[np.ndarray]:
        if not self.render_enabled:
            return None
        images = np.asarray(self.backend.render())
        if images.ndim != 5 or images.dtype != np.uint8:
            raise RuntimeError(
                f"solver render must return uint8 [N, C, H, W, 3], got "
                f"{images.dtype} {images.shape}"
            )
        self.render_count += 1
        return images

    def _wrap_obs(
        self, states: np.ndarray, images: Optional[np.ndarray]
    ) -> dict[str, Any]:
        states = np.asarray(states, dtype=np.float32)
        if states.shape != (self.num_envs, self.obs_dim):
            raise RuntimeError(
                f"solver states must have shape {(self.num_envs, self.obs_dim)}, "
                f"got {states.shape}"
            )
        self.current_states = states
        obs: dict[str, Any] = {
            "states": torch.from_numpy(np.ascontiguousarray(states)),
            "task_descriptions": [self.task_description] * self.num_envs,
        }
        if images is not None:
            self.current_images = images
        if self.current_images is not None:
            main = np.ascontiguousarray(self.current_images[:, self.main_camera_index])
            self.current_raw_obs = main
            obs["main_images"] = torch.from_numpy(main)
            if self.current_images.shape[1] > 1:
                extra_indices = [
                    index
                    for index in range(self.current_images.shape[1])
                    if index != self.main_camera_index
                ]
                obs["extra_view_images"] = torch.from_numpy(
                    np.ascontiguousarray(self.current_images[:, extra_indices])
                )
        return obs

    def capture_image(self) -> Optional[np.ndarray]:
        return self.current_raw_obs

    # ------------------------------------------------------------ metrics
    def _reset_metrics(self, env_idx: np.ndarray) -> None:
        self._elapsed_steps[env_idx] = 0
        self.success_once[env_idx] = False
        self.failure_once[env_idx] = False
        self.success_first_step[env_idx] = -1
        self.returns[env_idx] = 0.0
        self.last_distance[env_idx] = 0.0
        self.min_distance[env_idx] = np.inf

    def _episode_info(self) -> dict[str, Any]:
        min_distance = np.where(
            np.isfinite(self.min_distance), self.min_distance, 0.0
        ).astype(np.float32)
        return {
            "success_once": self.success_once.astype(np.float32, copy=True),
            "failure_once": self.failure_once.astype(np.float32, copy=True),
            "success_first_step": self.success_first_step.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
            "reward": self.returns / np.maximum(self._elapsed_steps, 1),
            "final_distance": self.last_distance.copy(),
            "min_distance": min_distance,
            "sample_seed": self.env_seeds.copy(),
            "sample_group": self.global_group_ids.copy(),
            "sample_trajectory": self.trajectory_ids.copy(),
            "metric_valid": self.metric_valid.copy(),
        }

    # --------------------------------------------------------------- gym
    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        options: Optional[dict] = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options, kwargs
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        elif isinstance(env_idx, int):
            env_idx = np.asarray([env_idx])
        else:
            env_idx = np.asarray(env_idx, dtype=np.int64).reshape(-1)
        self._is_start = False
        states = self.backend.reset(env_idx, self.env_seeds[env_idx])
        self._reset_metrics(env_idx)
        images = self._render_images()
        return self._wrap_obs(states, images), {}

    def step(
        self,
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        auto_reset: bool = True,
        render: bool = True,
    ):
        if actions is None:
            if not self._is_start:
                raise ValueError("Actions may be None only for the initial reset")
            obs, infos = self.reset()
            zeros = torch.zeros(self.num_envs, dtype=torch.bool)
            return obs, zeros.float(), zeros, zeros, infos
        del auto_reset
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = np.tile(actions[None, :], (self.num_envs, 1))
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"expected actions of shape {(self.num_envs, self.action_dim)}, "
                f"got {actions.shape}"
            )

        if self.clip_actions:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        result = self.backend.step(actions)
        self._elapsed_steps += 1
        success = np.asarray(result["success"], dtype=bool)
        failure = np.asarray(result["failure"], dtype=bool)
        distance = np.asarray(result["distance"], dtype=np.float32)
        first_success = success & (~self.success_once)
        if self.reward_mode == "success":
            step_reward = self.success_reward * first_success.astype(np.float32)
        else:
            step_reward = np.asarray(result["reward"], dtype=np.float32)
        step_reward = step_reward * self.reward_coef

        terminations = success.copy()
        if self.ignore_terminations:
            terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = self._elapsed_steps >= self.max_episode_steps

        # Bookkeeping mirrors RoboCasa-GR1: reward stops accumulating after the
        # first success so the return equals the binary outcome for GRPO.
        self.returns += step_reward * (~self.success_once)
        self.success_first_step[first_success] = self._elapsed_steps[first_success]
        self.success_once |= success
        self.failure_once |= failure
        self.last_distance = distance
        self.min_distance = np.minimum(self.min_distance, distance)

        images = self._render_images() if render else None
        obs = self._wrap_obs(result["states"], images)
        infos: dict[str, Any] = {
            "success": torch.from_numpy(success.copy()),
            "failure": torch.from_numpy(failure.copy()),
            "distance": torch.from_numpy(distance.copy()),
            "solver_reward": torch.from_numpy(
                np.asarray(result["reward"], dtype=np.float32).copy()
            ),
            "episode": to_tensor(self._episode_info()),
        }
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = torch.from_numpy(success.copy())
        self._is_start = False
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions: Union[np.ndarray, torch.Tensor]):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        if chunk_actions.ndim != 3:
            raise ValueError(
                "chunk_actions must have shape [num_envs, chunk_steps, action_dim], "
                f"got {chunk_actions.shape}"
            )
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
            render = self.render_every_step or action_step == action_steps - 1
            obs, reward, terminated, truncated, infos = self.step(
                chunk_actions[:, action_step], auto_reset=False, render=render
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.backend.close()
