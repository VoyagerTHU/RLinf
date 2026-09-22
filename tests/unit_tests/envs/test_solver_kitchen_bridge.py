"""Protocol tests for the solver bridge subprocess backend (fake core)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from rlinf.envs.solver_kitchen.backend import SubprocessBackend, make_backend
from rlinf.envs.solver_kitchen.bridge_server import BridgeServer

_TEST_DIR = Path(__file__).resolve().parent


def _extra_env():
    existing = os.environ.get("PYTHONPATH", "")
    return {"PYTHONPATH": f"{_TEST_DIR}{os.pathsep}{existing}" if existing else str(_TEST_DIR)}


def _fake_backend(**kwargs):
    return SubprocessBackend(
        sys.executable,
        fake_core="solver_kitchen_fakes:make_fake_core",
        extra_env=_extra_env(),
        connect_timeout=60.0,
        **kwargs,
    )


def test_bridge_round_trip_with_fake_core():
    backend = _fake_backend()
    metadata = backend.init({"num_envs": 3, "cameras": [{"name": "main", "width": 4, "height": 2}]})
    assert metadata["action_dim"] == 16
    assert metadata["camera_names"] == ["main"]
    assert backend.pid is not None

    states = backend.reset(np.array([0, 2]), np.array([11, 13]))
    assert states.shape == (3, metadata["obs_dim"])
    assert states[0, 0] == 11 and states[2, 0] == 13 and states[1, 0] == 0

    result = backend.step(np.zeros((3, 16), dtype=np.float32))
    assert set(result) >= {"states", "reward", "terminated", "truncated", "success", "failure", "distance"}
    assert result["states"].dtype == np.float32
    assert result["success"].dtype == bool

    images = backend.render()
    assert images.shape == (3, 1, 2, 4, 3)
    assert images.dtype == np.uint8

    pid = backend.pid
    backend.close()
    assert backend.pid is None
    with pytest.raises(OSError):
        os.kill(pid, 0)  # process must be gone


def test_bridge_forwards_core_exceptions():
    backend = _fake_backend()
    backend.init({"num_envs": 2})
    with pytest.raises(RuntimeError, match="AssertionError"):
        backend.step(np.zeros((5, 16), dtype=np.float32))
    # The bridge keeps serving after a failed command.
    states = backend.states()
    assert states.shape[0] == 2
    backend.close()


def test_bridge_init_failure_is_reported():
    backend = _fake_backend()
    with pytest.raises(RuntimeError, match="refused to initialize"):
        backend.init({"num_envs": 2, "explode_on_init": True})
    backend.close()


def test_missing_interpreter_is_rejected():
    with pytest.raises(FileNotFoundError):
        SubprocessBackend("/definitely/not/a/python")


def test_make_backend_requires_python_for_subprocess():
    with pytest.raises(ValueError, match="solver_python"):
        make_backend("subprocess", python_executable=None)
    with pytest.raises(ValueError, match="unknown"):
        make_backend("mystery")


def test_bridge_server_dispatch_without_process():
    from solver_kitchen_fakes import make_fake_core

    server = BridgeServer(make_fake_core)
    assert server.handle("ping", None)["protocol"] == 1
    with pytest.raises(RuntimeError, match="requires init"):
        server.handle("step", {"actions": np.zeros((1, 16))})
    metadata = server.handle("init", {"num_envs": 1})
    assert metadata["num_envs"] == 1
    with pytest.raises(RuntimeError, match="already initialized"):
        server.handle("init", {"num_envs": 1})
    with pytest.raises(ValueError, match="unknown bridge command"):
        server.handle("dance", None)
    assert server.handle("close", None) is None
