"""GPU integration tests for the solver kitchen core.

Run inside the *solver* virtual environment (not the RLinf training venv)::

    CUDA_VISIBLE_DEVICES=3 SOLVER_SMOKE_OUT=/tmp/solver_smoke \
    /path/to/solver/.venv/bin/python -m pytest \
        tests/integration_tests/solver_kitchen/test_core_smoke.py -q -s

The tests skip when ``solver`` or a CUDA device is unavailable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_CORE_PATH = (
    Path(__file__).resolve().parents[3] / "rlinf" / "envs" / "solver_kitchen" / "solver_core.py"
)


def _load_core():
    spec = importlib.util.spec_from_file_location("solver_kitchen_core_test", _CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cuda_available() -> bool:
    try:
        import warp as wp

        wp.config.quiet = True
        wp.init()
        return len(wp.get_cuda_devices()) > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("solver") is None or not _cuda_available(),
    reason="requires the solver package and a CUDA device",
)

NUM_ENVS = int(os.environ.get("SOLVER_SMOKE_NUM_ENVS", "8"))
OUT_DIR = os.environ.get("SOLVER_SMOKE_OUT")


@pytest.fixture(scope="module")
def core_module():
    return _load_core()


@pytest.fixture(scope="module")
def core(core_module):
    config = core_module.SolverKitchenCoreConfig.from_dict(
        {
            "num_envs": NUM_ENVS,
            "device": "cuda:0",
            "render": True,
            "task": {
                "target_randomization_range": [0.05, 0.05, 0.02],
                "max_episode_steps": 300,
            },
            "torch_env": {"normalized_actions": True, "action_scale": 0.5},
            "cameras": [
                {"name": "main", "width": 224, "height": 224},
                {"name": "side", "width": 224, "height": 224,
                 "position": [0.9, 1.65, 1.35], "look_at": [0.0, 1.65, 0.95]},
            ],
        }
    )
    instance = core_module.SolverKitchenCore(config)
    yield instance
    instance.close()


def _save_png(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def test_metadata_and_shapes(core):
    metadata = core.metadata()
    assert metadata["num_envs"] == NUM_ENVS
    assert metadata["action_dim"] == 16
    assert metadata["camera_names"] == ["main", "side"]
    states = core.reset()
    assert states.shape == (NUM_ENVS, metadata["obs_dim"])
    assert np.isfinite(states).all()


def test_render_produces_non_trivial_images(core):
    core.reset()
    images = core.render()
    assert images.shape == (NUM_ENVS, 2, 224, 224, 3)
    assert images.dtype == np.uint8
    for cam in range(2):
        frame = images[0, cam]
        assert frame.std() > 5.0, f"camera {cam} rendered a flat image"
        # Every world renders the same initial scene.
        assert np.array_equal(images[0, cam], images[-1, cam])
    if OUT_DIR:
        _save_png(Path(OUT_DIR) / "reset_main.png", images[0, 0])
        _save_png(Path(OUT_DIR) / "reset_side.png", images[0, 1])


def test_seeded_targets_are_grouped_and_deterministic(core):
    seeds = np.array([11, 11, 11, 11, 23, 23, 23, 23][:NUM_ENVS])
    first = core.reset(None, seeds).copy()
    second = core.reset(None, seeds).copy()
    assert np.array_equal(first, second), "reset with identical seeds must be bitwise equal"
    offsets = core.task.target_offsets.numpy()
    half = NUM_ENVS // 2
    assert np.allclose(offsets[:half], offsets[0])
    assert np.allclose(offsets[half:], offsets[half])
    assert not np.allclose(offsets[0], offsets[half])
    # The task feature block (last 4 entries) differs across groups.
    assert not np.allclose(first[0, -4:], first[half, -4:])
    assert np.allclose(first[0, -4:], first[half - 1, -4:])


def test_step_moves_arm_and_changes_image(core):
    core.reset()
    before = core.render()
    actions = np.zeros((NUM_ENVS, 16), dtype=np.float32)
    actions[:, 0] = 1.0  # swing the first right-arm joint
    for _ in range(20):
        result = core.step(actions)
    assert result["states"].shape[0] == NUM_ENVS
    assert np.isfinite(result["states"]).all()
    assert result["reward"].shape == (NUM_ENVS,)
    assert result["distance"].dtype == np.float32
    after = core.render()
    diff = np.abs(after[0, 0].astype(np.int32) - before[0, 0].astype(np.int32)).mean()
    assert diff > 0.5, "image did not change after moving the arm"
    if OUT_DIR:
        _save_png(Path(OUT_DIR) / "after_20_steps_main.png", after[0, 0])
        _save_png(Path(OUT_DIR) / "after_20_steps_side.png", after[0, 1])


def test_partial_reset_only_touches_selected_worlds(core):
    core.reset()
    actions = np.zeros((NUM_ENVS, 16), dtype=np.float32)
    actions[:, 1] = -1.0
    for _ in range(10):
        core.step(actions)
    moved = core.states().copy()
    states = core.reset(np.array([0]), np.array([5]))
    assert not np.allclose(states[0, :16], moved[0, :16])
    assert np.allclose(states[1:, :16], moved[1:, :16])


def test_throughput_report(core):
    core.reset()
    actions = np.zeros((NUM_ENVS, 16), dtype=np.float32)
    steps = 100
    for _ in range(5):
        core.step(actions)
    start = time.perf_counter()
    for _ in range(steps):
        core.step(actions)
    physics_only = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(steps):
        core.step(actions)
        core.render()
    with_render = time.perf_counter() - start
    print(
        f"\n[solver_kitchen] {NUM_ENVS} envs: physics {steps * NUM_ENVS / physics_only:.0f} env-step/s, "
        f"physics+render(2x224x224) {steps * NUM_ENVS / with_render:.0f} env-step/s"
    )
    assert physics_only > 0 and with_render > 0
