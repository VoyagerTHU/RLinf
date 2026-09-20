# Copyright 2026 The RLinf Authors.
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

"""The SAC critic action probe and the within-chunk reward discount."""

import torch

from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _RecordingModel:
    """Stands in for the FSDP model: Q depends on the action only."""

    def __init__(self, action_weight: float):
        self.action_weight = action_weight
        self.calls = []

    def __call__(self, forward_type, obs, actions, shared_feature, detach_encoder):
        self.calls.append((actions.shape, shared_feature.shape))
        q = self.action_weight * actions.float().sum(dim=-1, keepdim=True)
        return torch.cat([q, q + 0.01], dim=-1)  # twin heads


def _probe(model, samples=4, sigma_scale=1.0, with_reference=True):
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.model = model
    worker.q_action_probe_samples = samples
    worker.q_action_probe_sigma_scale = sigma_scale
    worker.action_sigma = 0.0302
    worker._action_norm_stats = None  # identity map in the stub below
    worker._policy_setup = None

    import rlinf.models.embodiment.starvla.utils.action_space as asu

    original = asu.unnormalize_actions_for_env_torch
    asu.unnormalize_actions_for_env_torch = lambda a, stats, policy_setup=None: a
    try:
        torch.manual_seed(0)
        mean = torch.zeros(3, 12, 29)
        extras = {"mean_actions": mean}
        if with_reference:
            extras["reference_mean_actions"] = torch.full((3, 12, 29), 0.1)
        actions = torch.full((3, 12 * 29), 0.2)
        features = torch.randn(3, 8)
        return worker._probe_action_discrimination(features, extras, actions)
    finally:
        asu.unnormalize_actions_for_env_torch = original


def test_probe_batches_every_row_through_one_q_call() -> None:
    model = _RecordingModel(action_weight=1.0)
    metrics = _probe(model)
    # 4 perturbations + mean + reference, each on 3 states, plus 3 data rows.
    assert model.calls == [((6 * 3 + 3, 12 * 29), (6 * 3 + 3, 8))]
    assert set(metrics) == {
        "q_action_std",
        "q_action_range",
        "q_mean_action",
        "q_mean_minus_data",
        "q_ref_action",
        "q_mean_minus_ref",
    }


def test_a_state_only_critic_reports_no_action_spread() -> None:
    metrics = _probe(_RecordingModel(action_weight=0.0))
    assert metrics["q_action_std"] == 0.0
    assert metrics["q_mean_minus_ref"] == 0.0


def test_an_action_sensitive_critic_reports_spread_and_ranking() -> None:
    metrics = _probe(_RecordingModel(action_weight=1.0))
    assert metrics["q_action_std"] > 0.0
    # Q grows with the action sum, so the reference (0.1) and data (0.2)
    # actions must both score above the zero mean.
    assert metrics["q_mean_minus_ref"] < 0.0
    assert metrics["q_mean_minus_data"] < metrics["q_mean_minus_ref"]


def test_probe_without_a_reference_head_skips_the_reference_metrics() -> None:
    metrics = _probe(_RecordingModel(action_weight=1.0), with_reference=False)
    assert "q_ref_action" not in metrics and "q_action_std" in metrics


def test_within_chunk_discount_matches_the_chunk_bootstrap() -> None:
    gamma = 0.999
    rewards = torch.zeros(2, 12)
    rewards[0, 0] = 1.0  # reward on the first physical step of the chunk
    rewards[1, 11] = 1.0  # reward on the last
    weights = gamma ** torch.arange(12, dtype=torch.float32)
    discounted = (rewards * weights).sum(dim=-1)
    assert discounted[0].item() == 1.0
    assert abs(discounted[1].item() - gamma**11) < 1e-6
    # The plain sum the worker used before treated both the same.
    assert torch.equal(rewards.sum(dim=-1), torch.ones(2))
