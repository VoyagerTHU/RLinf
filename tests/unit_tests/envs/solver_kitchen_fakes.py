"""Fake solver backends/cores shared by the solver_kitchen unit tests."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class FakeCore:
    """Deterministic stand-in for ``SolverKitchenCore`` (no solver needed).

    * ``states[:, 0]`` equals the seed given at reset so grouped-seed tests can
      check that same-seed worlds share an initial state.
    * The distance shrinks by 1.0 per step from ``initial_distance``; success
      fires when the distance reaches zero, which happens after
      ``success_after`` steps for every world in ``success_worlds``.
    * Rendered pixels equal the render counter so tests can see when frames
      were captured.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        action_dim: int = 16,
        obs_dim: int = 40,
        success_after: int = 2,
        success_worlds: Optional[list[int]] = None,
    ) -> None:
        self.config = dict(config)
        self.num_envs = int(config["num_envs"])
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.success_after = success_after
        self.success_worlds = (
            list(range(self.num_envs)) if success_worlds is None else success_worlds
        )
        cameras = config.get("cameras") or [{"name": "main", "width": 8, "height": 6}]
        self.cameras = cameras
        self.render_calls = 0
        self.step_calls = 0
        self.reset_calls: list[tuple[np.ndarray, Optional[np.ndarray]]] = []
        self.actions: list[np.ndarray] = []
        self._states = np.zeros((self.num_envs, obs_dim), dtype=np.float32)
        self._steps = np.zeros(self.num_envs, dtype=np.int32)
        self.closed = False

    def metadata(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "controlled_joint_names": [f"j{i}" for i in range(self.action_dim)],
            "action_lower_limits": -np.ones(self.action_dim, dtype=np.float32),
            "action_upper_limits": np.ones(self.action_dim, dtype=np.float32),
            "normalized_actions": True,
            "camera_names": [camera["name"] for camera in self.cameras],
            "image_shapes": [
                (int(camera.get("height", 6)), int(camera.get("width", 8)), 3)
                for camera in self.cameras
            ],
            "render": bool(self.config.get("render", True)),
        }

    def reset(self, env_ids=None, seeds=None) -> np.ndarray:
        env_ids = (
            np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        )
        self.reset_calls.append((env_ids.copy(), None if seeds is None else np.asarray(seeds).copy()))
        self._states[env_ids] = 0.0
        if seeds is not None:
            self._states[env_ids, 0] = np.asarray(seeds, dtype=np.float32)
        self._steps[env_ids] = 0
        return self._states.copy()

    def states(self) -> np.ndarray:
        return self._states.copy()

    def step(self, actions) -> dict[str, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self.num_envs, self.action_dim), actions.shape
        self.actions.append(actions.copy())
        self.step_calls += 1
        self._steps += 1
        self._states[:, 1] = self._steps
        distance = np.maximum(self.success_after - self._steps, 0).astype(np.float32)
        success = np.zeros(self.num_envs, dtype=bool)
        for world in self.success_worlds:
            success[world] = self._steps[world] >= self.success_after
        return {
            "states": self._states.copy(),
            "reward": -distance,
            "terminated": success.copy(),
            "truncated": np.zeros(self.num_envs, dtype=bool),
            "success": success,
            "failure": np.zeros(self.num_envs, dtype=bool),
            "distance": distance,
        }

    def render(self) -> np.ndarray:
        self.render_calls += 1
        frames = []
        for camera in self.cameras:
            h, w = int(camera.get("height", 6)), int(camera.get("width", 8))
            frames.append(np.full((self.num_envs, h, w, 3), self.render_calls, dtype=np.uint8))
        return np.stack(frames, axis=1)

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Backend that wraps :class:`FakeCore` in-process."""

    def __init__(self, **core_kwargs: Any) -> None:
        self.core_kwargs = core_kwargs
        self.core: Optional[FakeCore] = None
        self.closed = False

    def init(self, core_config):
        self.core = FakeCore(core_config, **self.core_kwargs)
        return self.core.metadata()

    def reset(self, env_ids, seeds):
        return self.core.reset(env_ids, seeds)

    def step(self, actions):
        return self.core.step(actions)

    def render(self):
        return self.core.render()

    def states(self):
        return self.core.states()

    def close(self):
        self.closed = True
        self.core.close()


def make_fake_core(config: dict[str, Any]) -> FakeCore:
    """Factory used by ``bridge_server.py --fake-core``."""
    if config.get("explode_on_init"):
        raise RuntimeError("fake core refused to initialize")
    return FakeCore(config)
