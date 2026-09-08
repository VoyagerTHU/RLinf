from types import SimpleNamespace

import numpy as np
import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _sampling_worker(base_seed):
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = SimpleNamespace(
        rollout={"training_sampling_seed": base_seed}
    )
    worker.version = 2
    worker._world_size = 8
    worker._rank = 3
    worker.log_info = lambda _message: None
    return worker


def test_train_sampling_rng_is_reproducible_and_version_rank_scoped():
    worker = _sampling_worker(100)

    assert worker._reset_train_sampling_rng() == 119
    torch_values = torch.randn(4)
    numpy_values = np.random.randn(4)

    assert worker._reset_train_sampling_rng() == 119
    torch.testing.assert_close(torch.randn(4), torch_values)
    np.testing.assert_array_equal(np.random.randn(4), numpy_values)


def test_train_sampling_rng_can_remain_unconfigured():
    worker = _sampling_worker(None)

    assert worker._reset_train_sampling_rng() is None


def test_train_sampling_rng_rejects_negative_seed():
    worker = _sampling_worker(-1)

    try:
        worker._reset_train_sampling_rng()
    except ValueError as exc:
        assert "must be non-negative" in str(exc)
    else:
        raise AssertionError("Expected a negative training sampling seed to fail")
