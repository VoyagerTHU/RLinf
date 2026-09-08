"""Tests for RoboCasa-GR1 episode metric filtering."""

import torch

from rlinf.workers.env.env_worker import EnvWorker


def test_episode_metrics_filter_padded_evaluation_trajectories():
    episode = {
        "success_once": torch.tensor([True, False, True, False]),
        "sample_seed": torch.tensor([10, 11, 12, 10]),
        "metric_valid": torch.tensor([True, True, True, False]),
    }

    metrics = EnvWorker._extract_episode_metrics(
        episode, selection=torch.tensor([True, False, True, True])
    )

    assert metrics["success_once"].tolist() == [True, True]
    assert metrics["sample_seed"].tolist() == [10, 12]
    assert "metric_valid" not in metrics
