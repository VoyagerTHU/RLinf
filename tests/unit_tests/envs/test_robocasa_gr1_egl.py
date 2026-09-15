"""Tests for pinning RoboCasa subprocesses to NVIDIA EGL."""

import os

from rlinf.envs.robocasa_gr1.robocasa_gr1_env import (
    configure_simulator_egl,
    configure_simulator_mesa,
)


def test_simulator_egl_preserves_cuda_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/tmp/10_nvidia.json")

    configure_simulator_egl(5)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_simulator_mesa_removes_cuda_and_nvidia_vendor(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/tmp/10_nvidia.json")

    configure_simulator_mesa(device_id=8, llvmpipe_threads=2)

    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "8"
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "true"
    assert os.environ["LP_NUM_THREADS"] == "2"
