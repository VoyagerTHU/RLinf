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

"""Solver-side runtime for the RLinf ``solver_kitchen`` environment.

This module runs inside the ``solver`` virtual environment (Python 3.11,
Newton, Warp, MuJoCo-Warp). It must stay importable **without** ``rlinf`` and
its dependencies, because :mod:`bridge_server` executes it in a separate
interpreter. Only ``numpy`` and the solver stack are imported here.

The core wraps the solver kitchen RL stack
(``KitchenVectorEnv -> KitchenReachTask -> KitchenTorchEnv``) plus a
``TiledCameraViewer`` for batched RGB rendering, and exposes a small
NumPy-only API used by both the in-process and the subprocess backends.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

DEFAULT_TASK_DESCRIPTION = "reach the knife handle with the right gripper"


@dataclass(frozen=True)
class CameraConfig:
    """Pinhole camera placed relative to the environment origin.

    ``position`` is in meters and ``orientation`` is a Newton-native ``xyzw``
    quaternion. When ``look_at`` is given the orientation is derived from it
    and ``orientation`` is ignored.
    """

    name: str = "main"
    width: int = 224
    height: int = 224
    position: tuple[float, float, float] = (0.0, 0.55, 1.75)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    look_at: Optional[tuple[float, float, float]] = (0.0, 1.65, 0.95)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_fov_degrees: float = 45.0
    near: float = 0.01
    far: float = 100.0


@dataclass(frozen=True)
class SolverKitchenCoreConfig:
    """Everything the solver process needs to build the batched kitchen."""

    num_envs: int = 8
    device: str = "cuda:0"
    fps: int = 60
    sim_substeps: int = 8
    graph_capture: bool = True
    deterministic_gpu: bool = True
    deterministic_contacts: bool = True
    deterministic_solver: bool = True
    enable_shadows: bool = True
    render: bool = True
    task: dict[str, Any] = field(default_factory=dict)
    torch_env: dict[str, Any] = field(default_factory=dict)
    cameras: tuple[CameraConfig, ...] = (CameraConfig(),)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SolverKitchenCoreConfig":
        payload = dict(payload)
        cameras = payload.pop("cameras", None)
        if cameras is None:
            camera_specs: tuple[CameraConfig, ...] = (CameraConfig(),)
        else:
            camera_specs = tuple(
                camera if isinstance(camera, CameraConfig) else _camera_from_dict(camera)
                for camera in cameras
            )
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown solver core config keys: {unknown}")
        return cls(cameras=camera_specs, **payload)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["cameras"] = [dataclasses.asdict(camera) for camera in self.cameras]
        return data


def _camera_from_dict(payload: dict[str, Any]) -> CameraConfig:
    payload = dict(payload)
    for key in ("position", "orientation", "look_at", "up"):
        if key in payload and payload[key] is not None:
            payload[key] = tuple(float(v) for v in payload[key])
    return CameraConfig(**payload)


def look_at_quaternion_xyzw(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Return an ``xyzw`` quaternion for a camera at ``eye`` looking at ``target``.

    Newton's camera convention looks down the local ``-Z`` axis with ``+Y`` up,
    matching OpenGL. The returned rotation maps that local frame to world.
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    forward = target_v - eye_v
    norm = np.linalg.norm(forward)
    if norm == 0.0:
        raise ValueError("camera eye and look_at must differ")
    forward /= norm
    up_v = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up_v)
    if np.linalg.norm(right) < 1e-8:
        # Forward is parallel to up; pick any perpendicular axis.
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # Columns are the world-space directions of the camera's local X, Y, Z.
    rotation = np.stack([right, true_up, -forward], axis=1)
    return _matrix_to_quaternion_xyzw(rotation)


def _matrix_to_quaternion_xyzw(m: np.ndarray) -> tuple[float, float, float, float]:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def seeded_target_offsets(
    seeds: np.ndarray,
    base_offset: tuple[float, float, float],
    randomization_range: tuple[float, float, float],
    randomization_probability: float,
) -> np.ndarray:
    """Deterministic per-seed reach targets.

    Every world that receives the same seed gets exactly the same target, which
    is what grouped GRPO needs. The sampling mirrors the solver's own
    ``_reset_randomized_targets`` semantics (uniform in ``[-range, range]`` with
    probability ``randomization_probability``, else the base offset).
    """
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    offsets = np.empty((seeds.shape[0], 3), dtype=np.float32)
    base = np.asarray(base_offset, dtype=np.float64)
    spread = np.asarray(randomization_range, dtype=np.float64)
    for row, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        randomize = rng.random() < randomization_probability
        if randomize:
            offsets[row] = base + spread * rng.uniform(-1.0, 1.0, size=3)
        else:
            offsets[row] = base
    return offsets


class SolverKitchenCore:
    """Batched kitchen simulation plus renderer with a NumPy-only surface."""

    def __init__(self, config: SolverKitchenCoreConfig) -> None:
        import warp as wp
        from solver.rl import (
            KitchenReachTask,
            KitchenReachTaskConfig,
            KitchenTorchEnv,
            KitchenTorchEnvConfig,
            KitchenVectorEnv,
            KitchenVectorEnvConfig,
        )

        self.config = config
        self.num_envs = int(config.num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        wp.set_device(config.device)
        self._wp = wp

        viewer = None
        if config.render:
            from solver.output.camera import CameraSpec
            from solver.output.tiled_camera import TiledCameraViewer

            device = wp.get_device(config.device)
            if not device.is_cuda:
                raise RuntimeError(
                    "solver_kitchen rendering requires a CUDA device; "
                    "set render=false for CPU-only smoke tests"
                )
            viewer = TiledCameraViewer(
                num_envs=self.num_envs,
                env_origins=np.zeros((self.num_envs, 3), dtype=np.float32),
                enable_shadows=config.enable_shadows,
            )
            self._camera_ids = []
            for camera in config.cameras:
                orientation = camera.orientation
                if camera.look_at is not None:
                    orientation = look_at_quaternion_xyzw(
                        camera.position, camera.look_at, camera.up
                    )
                self._camera_ids.append(
                    viewer.add_camera(
                        CameraSpec(
                            name=camera.name,
                            width=int(camera.width),
                            height=int(camera.height),
                            position=tuple(camera.position),
                            orientation=orientation,
                            vertical_fov_degrees=float(camera.vertical_fov_degrees),
                            near=float(camera.near),
                            far=float(camera.far),
                        )
                    )
                )
        self.viewer = viewer

        self.physics_env = KitchenVectorEnv(
            KitchenVectorEnvConfig(
                world_count=self.num_envs,
                fps=int(config.fps),
                sim_substeps=int(config.sim_substeps),
                device=config.device,
                graph_capture=bool(config.graph_capture),
                deterministic_contacts=bool(config.deterministic_contacts),
                deterministic_solver=bool(config.deterministic_solver),
                deterministic_gpu=bool(config.deterministic_gpu),
            ),
            viewer=viewer,
        )
        task_kwargs = dict(config.task)
        self.task_config = KitchenReachTaskConfig(**task_kwargs)
        self.task = KitchenReachTask(self.physics_env, self.task_config)
        torch_env_kwargs = dict(config.torch_env)
        # RLinf owns episode boundaries; the solver adapter must never reset on
        # its own, otherwise terminal observations and grouped seeds drift.
        torch_env_kwargs["auto_reset"] = False
        self.torch_env_config = KitchenTorchEnvConfig(**torch_env_kwargs)
        self.torch_env = KitchenTorchEnv(self.task, self.torch_env_config)

        self.action_dim = int(self.torch_env.num_actions)
        self.obs_dim = int(self.task.observation_shape[1])
        self.controlled_joint_names = tuple(
            self.physics_env.scene.controlled_joint_names
        )
        self._all_worlds = np.ones(self.num_envs, dtype=bool)
        self._camera_names = tuple(camera.name for camera in config.cameras)

    # ------------------------------------------------------------------ API
    def metadata(self) -> dict[str, Any]:
        scene = self.physics_env.scene
        return {
            "num_envs": self.num_envs,
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "controlled_joint_names": list(self.controlled_joint_names),
            "action_lower_limits": np.asarray(
                scene.action_lower_limits, dtype=np.float32
            ),
            "action_upper_limits": np.asarray(
                scene.action_upper_limits, dtype=np.float32
            ),
            "normalized_actions": bool(self.torch_env_config.normalized_actions),
            "camera_names": list(self._camera_names),
            "image_shapes": [
                (int(camera.height), int(camera.width), 3)
                for camera in self.config.cameras
            ],
            "render": bool(self.config.render),
            "object_label": self.task_config.object_label,
            "end_effector_label": self.task_config.end_effector_label,
        }

    def reset(
        self,
        env_ids: Optional[np.ndarray] = None,
        seeds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reset selected worlds (all when ``env_ids`` is None) and return states."""
        import torch

        if env_ids is None:
            mask_np = self._all_worlds.copy()
        else:
            mask_np = np.zeros(self.num_envs, dtype=bool)
            mask_np[np.asarray(env_ids, dtype=np.int64)] = True
        if not mask_np.any():
            return self.states()
        torch_device = self.torch_env.device
        if mask_np.all():
            self.torch_env.reset()
        else:
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=torch_device)
            self.torch_env.reset(mask)
        if seeds is not None:
            seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
            if seeds.shape[0] != int(mask_np.sum()):
                raise ValueError(
                    f"expected {int(mask_np.sum())} seeds for the reset worlds, "
                    f"got {seeds.shape[0]}"
                )
            self._apply_seeded_targets(mask_np, seeds)
        return self.states()

    def _apply_seeded_targets(self, mask_np: np.ndarray, seeds: np.ndarray) -> None:
        """Overwrite the reach targets of the masked worlds and refresh observations."""
        wp = self._wp
        offsets = self.task.target_offsets.numpy()
        offsets[mask_np] = seeded_target_offsets(
            seeds,
            self.task_config.target_offset,
            self.task_config.target_randomization_range,
            self.task_config.target_randomization_probability,
        )
        self.task.target_offsets.assign(offsets)
        from solver.rl import kitchen_task as kt

        mask = wp.array(mask_np, dtype=wp.bool, device=self.task.device)
        wp.launch(
            kt._gather_reach_features,
            dim=self.task.world_count,
            inputs=[
                self.physics_env.sim.body_transform_tensor(),
                self.task.end_effector_body_index,
                self.task.object_body_index,
                self.task.end_effector_offset,
                self.task.target_offsets,
                self.task.object_reference_position,
                self.task.world_fixed_target,
                self.task.task_features,
            ],
            device=self.task.device,
        )
        wp.launch(
            kt._reset_previous_distance,
            dim=self.task.world_count,
            inputs=[mask, self.task.task_features, self.task.previous_distance],
            device=self.task.device,
        )
        self.task._assemble_observation(self.physics_env.observation)
        wp.synchronize_device(self.task.device)

    def states(self) -> np.ndarray:
        return self._to_numpy(self.torch_env.observation, np.float32)

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"actions must have shape {(self.num_envs, self.action_dim)}, "
                f"got {actions.shape}"
            )
        action_tensor = torch.as_tensor(actions, device=self.torch_env.device)
        observation, reward, terminated, truncated, info = self.torch_env.step(
            action_tensor
        )
        result = {
            "states": self._to_numpy(observation, np.float32),
            "reward": self._to_numpy(reward, np.float32),
            "terminated": self._to_numpy(terminated, bool),
            "truncated": self._to_numpy(truncated, bool),
            "success": self._to_numpy(info["success"], bool),
            "failure": self._to_numpy(info["failure"], bool),
            "distance": self._to_numpy(info["distance"], np.float32),
        }
        return result

    def render(self) -> np.ndarray:
        """Return ``uint8 [num_envs, num_cameras, H, W, 3]`` RGB images."""
        if self.viewer is None:
            raise RuntimeError("rendering is disabled for this solver core")
        self.viewer.sync_camera_mounts(self.physics_env.sim.state_0)
        batch = self.viewer.capture(self._camera_ids, outputs=("rgb",))
        rgb = np.ascontiguousarray(batch.rgb, dtype=np.uint8)
        if rgb.ndim != 5:
            raise RuntimeError(f"unexpected camera batch shape {rgb.shape}")
        return rgb

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:  # pragma: no cover - best effort
                pass

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _to_numpy(tensor, dtype) -> np.ndarray:
        array = tensor.detach().to("cpu").numpy()
        return np.ascontiguousarray(array.astype(dtype, copy=False))
