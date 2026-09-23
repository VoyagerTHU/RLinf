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

"""StarVLAForRLActionPrediction._select_best_of_n, exercised for real.

Builds a bare policy instance (no VLM backbone) carrying only the pieces
the selection needs -- a real StarVLAEditPolicy, a real StarVLAMultiQHead,
action statistics -- and calls the method under test, so a regression in
its unit conventions or index alignment cannot hide behind a re-implemented
mirror of the logic.
"""

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.starvla.starvla_action_model import (
    StarVLAEditPolicy,
    StarVLAForRLActionPrediction,
    StarVLAMultiQHead,
)
from rlinf.models.embodiment.starvla.utils.action_space import (
    normalize_actions_from_env_torch,
)

HIDDEN, CHUNKS, DIM = 16, 3, 29
STATS = {
    "q99": np.linspace(0.5, 3.0, DIM),
    "q01": np.linspace(-1.5, -0.2, DIM),
    "mask": np.ones(DIM, dtype=bool),
}


def _bare_policy(seed: int = 0) -> StarVLAForRLActionPrediction:
    torch.manual_seed(seed)
    policy = StarVLAForRLActionPrediction.__new__(StarVLAForRLActionPrediction)
    nn.Module.__init__(policy)
    policy.action_dim = DIM
    policy.num_executed_action_chunks = CHUNKS
    policy.action_feature_dim = CHUNKS * DIM
    policy.policy_setup = "gr1"
    policy._action_norm_stats = STATS
    policy.expo_num_candidates = 4
    policy.q_head = StarVLAMultiQHead(
        hidden_size=HIDDEN,
        action_feature_dim=CHUNKS * DIM,
        hidden_dims=[32, 16],
        num_q_heads=2,
    )
    policy.edit_policy = StarVLAEditPolicy(
        hidden_size=HIDDEN, action_feature_dim=CHUNKS * DIM, hidden_dims=[32], beta=0.1
    )
    return policy


def _inputs(batch: int = 3):
    mean = torch.rand(batch, CHUNKS, DIM) * 0.6 - 0.3
    dist = torch.distributions.Normal(mean, torch.full_like(mean, 0.05))
    last_hidden = torch.randn(batch, 5, HIDDEN)
    return mean, last_hidden, dist, {"attention_mask": torch.ones(batch, 5)}


def test_returned_env_and_normalized_actions_are_the_same_candidate() -> None:
    """The regression this file exists for: both tensors must be one pick."""
    policy = _bare_policy()
    mean, last_hidden, dist, model_inputs = _inputs()
    env_sel, norm_sel, features, metrics = policy._select_best_of_n(
        mean, last_hidden, dist, model_inputs, num_candidates=4, mode="train"
    )
    assert env_sel.shape == norm_sel.shape == (3, CHUNKS, DIM)
    assert features.shape == (3, HIDDEN)
    # Normalizing the env-unit pick must give back the normalized pick exactly:
    # the two came from the same row, related only by the affine stats map.
    back = normalize_actions_from_env_torch(env_sel, STATS, policy_setup="gr1")
    assert torch.allclose(back, norm_sel, atol=1e-5)
    assert 0.0 <= metrics["frac_selected_is_edited"] <= 1.0


def test_eval_mode_offers_only_the_mean_and_its_edit() -> None:
    policy = _bare_policy()
    mean, last_hidden, dist, model_inputs = _inputs()
    _, norm_sel, _, _ = policy._select_best_of_n(
        mean, last_hidden, dist, model_inputs, mode="eval"
    )
    # With a zero-init edit head the edited mean equals the mean, so every
    # row's pick is the mean itself regardless of Q's tie-break.
    assert torch.allclose(norm_sel, mean, atol=1e-6)


def test_selection_follows_q_not_candidate_order() -> None:
    """Rig the critic to prefer large actions; the pick must be the max-sum row."""
    policy = _bare_policy()
    mean, last_hidden, dist, model_inputs = _inputs()

    class SumQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.qs = nn.ModuleList()

        def forward(self, state_features, action_features):
            q = action_features.float().sum(dim=-1, keepdim=True)
            return torch.cat([q, q], dim=-1)

        def parameters(self, recurse=True):
            yield torch.zeros(1)

    policy.q_head = SumQ()
    torch.manual_seed(1)
    _, norm_sel, _, _ = policy._select_best_of_n(
        mean, last_hidden, dist, model_inputs, num_candidates=4, mode="train"
    )
    # Reproduce the candidate set and check the pick maximizes the critic.
    torch.manual_seed(1)
    n, batch = 4, mean.shape[0]
    base = dist.rsample((n,)).reshape(n, batch, -1)
    feats = policy._pool_sac_features(last_hidden, model_inputs["attention_mask"])
    edits, _ = policy.edit_policy(feats.repeat(n, 1), base.reshape(n * batch, -1), mode="train")
    all_norm = torch.cat([base.reshape(n * batch, -1), base.reshape(n * batch, -1) + edits], 0)
    all_norm = all_norm.reshape(2 * n, batch, CHUNKS, DIM)
    # The critic sees re-normalized env actions, which round-trip to all_norm,
    # so the max of the normalized sum identifies the winning row.
    sums = all_norm.sum(dim=(-1, -2))  # [2n, batch]
    best = sums.argmax(dim=0)
    expected = all_norm[best, torch.arange(batch)]
    assert torch.allclose(norm_sel, expected, atol=1e-4)
