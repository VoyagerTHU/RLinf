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

"""Backends that give :class:`SolverKitchenEnv` access to the solver core.

``InProcessBackend`` imports the solver stack into the current interpreter and
is only usable when that interpreter has ``solver`` installed (integration
tests inside the solver venv). ``SubprocessBackend`` launches
:mod:`bridge_server` with a dedicated interpreter, which is how RLinf training
runs: the actor/rollout workers keep their torch build while the simulator
lives in the solver venv.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

_BRIDGE_SERVER_PATH = Path(__file__).resolve().with_name("bridge_server.py")
_SOLVER_CORE_PATH = Path(__file__).resolve().with_name("solver_core.py")


class SolverBackend(Protocol):
    """Minimal contract shared by all backends."""

    def init(self, core_config: dict[str, Any]) -> dict[str, Any]: ...

    def reset(
        self, env_ids: Optional[np.ndarray], seeds: Optional[np.ndarray]
    ) -> np.ndarray: ...

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]: ...

    def render(self) -> np.ndarray: ...

    def states(self) -> np.ndarray: ...

    def close(self) -> None: ...


class InProcessBackend:
    """Run the solver core inside the current interpreter."""

    def __init__(self) -> None:
        self._core = None

    def init(self, core_config: dict[str, Any]) -> dict[str, Any]:
        from rlinf.envs.solver_kitchen.solver_core import (
            SolverKitchenCore,
            SolverKitchenCoreConfig,
        )

        self._core = SolverKitchenCore(SolverKitchenCoreConfig.from_dict(core_config))
        return self._core.metadata()

    def reset(self, env_ids, seeds):
        return self._core.reset(env_ids, seeds)

    def step(self, actions):
        return self._core.step(actions)

    def render(self):
        return self._core.render()

    def states(self):
        return self._core.states()

    def close(self) -> None:
        if self._core is not None:
            self._core.close()
            self._core = None


class SubprocessBackend:
    """Run the solver core in a child process with its own interpreter."""

    def __init__(
        self,
        python_executable: str,
        *,
        extra_env: Optional[dict[str, str]] = None,
        connect_timeout: float = 120.0,
        init_timeout: float = 1800.0,
        call_timeout: float = 600.0,
        fake_core: Optional[str] = None,
        socket_dir: Optional[str] = None,
    ) -> None:
        self.python_executable = str(python_executable)
        if not os.path.exists(self.python_executable):
            raise FileNotFoundError(
                f"solver interpreter not found: {self.python_executable}"
            )
        self.extra_env = dict(extra_env or {})
        self.connect_timeout = float(connect_timeout)
        self.init_timeout = float(init_timeout)
        self.call_timeout = float(call_timeout)
        self.fake_core = fake_core
        self.socket_dir = socket_dir
        self._process: Optional[subprocess.Popen] = None
        self._conn = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None

    # ----------------------------------------------------------- lifecycle
    def _spawn(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="solver_kitchen_", dir=self.socket_dir
        )
        address = os.path.join(self._tmpdir.name, "bridge.sock")
        authkey = secrets.token_bytes(16)
        cmd = [
            self.python_executable,
            str(_BRIDGE_SERVER_PATH),
            "--address",
            address,
            "--authkey",
            authkey.hex(),
            "--core-path",
            str(_SOLVER_CORE_PATH),
        ]
        if self.fake_core:
            cmd += ["--fake-core", self.fake_core]
        env = os.environ.copy()
        # The training interpreter's site-packages must not leak into the
        # solver interpreter: drop PYTHONPATH entries that point at RLinf
        # dependency trees while keeping the caller's explicit additions.
        env.pop("PYTHONHOME", None)
        env.update(self.extra_env)
        self._process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + self.connect_timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "solver bridge process exited during startup with code "
                    f"{self._process.returncode}"
                )
            try:
                self._conn = Client(address, family="AF_UNIX", authkey=authkey)
                break
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_error = exc
                time.sleep(0.1)
        if self._conn is None:
            self._kill()
            raise RuntimeError(
                f"could not connect to the solver bridge within {self.connect_timeout}s"
            ) from last_error

    def _kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass

    def _call(self, command: str, payload: Any = None, *, timeout=None):
        if self._conn is None:
            raise RuntimeError("solver bridge is not connected")
        timeout = self.call_timeout if timeout is None else timeout
        self._conn.send((command, payload))
        deadline = time.monotonic() + timeout
        while not self._conn.poll(0.5):
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"solver bridge process died during {command!r} with code "
                    f"{self._process.returncode}"
                )
            if time.monotonic() > deadline:
                self._kill()
                raise TimeoutError(
                    f"solver bridge {command!r} exceeded {timeout}s"
                )
        status, result = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"solver bridge {command!r} failed:\n{result}")
        return result

    # ----------------------------------------------------------------- API
    def init(self, core_config: dict[str, Any]) -> dict[str, Any]:
        self._spawn()
        self._call("ping", timeout=self.connect_timeout)
        return self._call("init", core_config, timeout=self.init_timeout)

    def reset(self, env_ids, seeds):
        return self._call(
            "reset",
            {
                "env_ids": None if env_ids is None else np.asarray(env_ids),
                "seeds": None if seeds is None else np.asarray(seeds),
            },
        )

    def step(self, actions):
        return self._call("step", {"actions": np.asarray(actions, dtype=np.float32)})

    def render(self):
        return self._call("render")

    def states(self):
        return self._call("states")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._call("close", timeout=60.0)
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._process is not None:
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._kill()
            self._process = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    @property
    def pid(self) -> Optional[int]:
        return None if self._process is None else self._process.pid

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def make_backend(
    backend: str,
    *,
    python_executable: Optional[str] = None,
    **kwargs: Any,
) -> SolverBackend:
    backend = str(backend).strip().lower()
    if backend in ("inprocess", "in_process", "in-process"):
        return InProcessBackend()
    if backend == "subprocess":
        if not python_executable:
            raise ValueError(
                "solver_kitchen subprocess backend requires `solver_python` "
                "(path to the solver venv interpreter); set env.train.solver_python "
                "or the SOLVER_PYTHON environment variable"
            )
        return SubprocessBackend(python_executable, **kwargs)
    if backend == "auto":
        if python_executable and os.path.abspath(python_executable) != os.path.abspath(
            sys.executable
        ):
            return SubprocessBackend(python_executable, **kwargs)
        return InProcessBackend()
    raise ValueError(f"unknown solver_kitchen backend {backend!r}")
