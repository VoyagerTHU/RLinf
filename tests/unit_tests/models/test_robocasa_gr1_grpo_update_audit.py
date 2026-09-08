"""Tests for the RoboCasa-GR1 GRPO update health audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = REPO_ROOT / "examples/embodiment/audit_robocasa_gr1_grpo_update.py"


def load_audit_module():
    """Load the standalone audit script as a module."""
    spec = importlib.util.spec_from_file_location("robocasa_grpo_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_update_stats_preserve_frozen_backbone(tmp_path):
    audit = load_audit_module()
    before_path = tmp_path / "before.pt"
    after_path = tmp_path / "after.pt"
    torch.save(
        {
            "qwen.weight": torch.ones(2, dtype=torch.bfloat16),
            "action_model.weight": torch.zeros(4, dtype=torch.float32),
            "actor_logstd": torch.full((2,), -3.0),
        },
        before_path,
    )
    torch.save(
        {
            "starvla_model.qwen.weight": torch.ones(2, dtype=torch.bfloat16),
            "starvla_model.action_model.weight": torch.tensor(
                [0.0, 1e-3, 0.0, -1e-3], dtype=torch.float32
            ),
            "actor_logstd": torch.full((2,), -6.0),
        },
        after_path,
    )

    stats = audit.checkpoint_update_stats(before_path, after_path, -6.0)

    assert stats["frozen_changed_count"] == 0
    assert stats["trainable_fp32_fraction"] == 1.0
    assert stats["changed_elements"] == 2
    assert stats["changed_fraction"] == 0.5
    assert stats["runtime_logstd_matches_expected"]


def test_rollout_group_stats_count_only_mixed_groups_as_active():
    audit = load_audit_module()
    rows = []
    for group_index, outcomes in enumerate(([0, 1], [0, 0], [1, 1])):
        for trajectory_index, outcome in enumerate(outcomes):
            rows.append(
                {
                    "training_step": 2,
                    "seed": 100 + group_index,
                    "trajectory_index": trajectory_index,
                    "trajectory_success": outcome,
                }
            )

    stats = audit.rollout_group_stats(
        rows, completed_global_step=3, expected_group_size=2
    )

    assert stats["groups"] == 3
    assert stats["mixed_groups"] == 1
    assert stats["all_failure_groups"] == 1
    assert stats["all_success_groups"] == 1
    assert stats["advantage_active_trajectories"] == 2
