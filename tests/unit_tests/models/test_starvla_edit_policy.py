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

"""The EXPO-FT residual edit policy (StarVLAEditPolicy)."""

import torch

from rlinf.models.embodiment.starvla.starvla_action_model import StarVLAEditPolicy


def _policy(beta: float = 0.1) -> StarVLAEditPolicy:
    return StarVLAEditPolicy(
        hidden_size=16, action_feature_dim=12 * 29, hidden_dims=[32, 32], beta=beta
    )


def test_zero_init_starts_the_edit_at_exactly_zero() -> None:
    """Training must begin at the pretrained policy's own behaviour."""
    policy = _policy()
    features = torch.randn(4, 16)
    base = torch.randn(4, 12 * 29)
    eval_edit, eval_log_prob = policy.sample(features, base, mode="eval")
    assert torch.equal(eval_edit, torch.zeros_like(eval_edit))
    assert eval_log_prob is None
    torch.manual_seed(0)
    train_edit, log_prob = policy.sample(features, base, mode="train")
    # mean_head is zero-init, so the pre-tanh mean is 0 regardless of input;
    # only the (also zero-init-weight but non-zero-bias) log_std varies the
    # sample around that.
    assert train_edit.shape == base.shape
    assert log_prob.shape == (4, 1)
    assert torch.isfinite(log_prob).all()


def test_edit_is_bounded_by_beta() -> None:
    beta = 0.1
    policy = _policy(beta=beta)
    # Push the mean head far from zero to test the bound under an extreme
    # pre-tanh value, not just near the zero-init default.
    with torch.no_grad():
        policy.mean_head.bias.fill_(50.0)
    features = torch.randn(8, 16)
    base = torch.randn(8, 12 * 29)
    for mode in ("eval", "train"):
        edit, _ = policy.sample(features, base, mode=mode)
        assert edit.abs().max().item() <= beta + 1e-6


def test_train_mode_log_prob_matches_a_manual_tanh_gaussian_computation() -> None:
    """Pin the squash-correction formula against an independent computation."""
    policy = _policy(beta=0.2)
    features = torch.randn(2, 16)
    base = torch.randn(2, 12 * 29)
    torch.manual_seed(1)
    edit, log_prob = policy.sample(features, base, mode="train")

    # Recompute log_prob from the edit itself: pre_tanh = atanh(edit / beta),
    # then the standard tanh-Gaussian correction. This does not reuse any of
    # the module's own log_prob code path.
    with torch.no_grad():
        cat = torch.cat([policy.state_norm(features), base], dim=-1)
        hidden = policy.trunk(cat)
        mean = policy.mean_head(hidden)
        log_std = policy.log_std_head(hidden).clamp(-5.0, 2.0)
        std = log_std.exp()
        tanh_pre = (edit / policy.beta).clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(tanh_pre)
        gaussian = torch.distributions.Normal(mean, std).log_prob(pre_tanh).sum(
            dim=-1, keepdim=True
        )
        import math

        correction = (
            math.log(policy.beta) + torch.log1p(-tanh_pre.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        expected = gaussian - correction
    assert torch.allclose(log_prob, expected, atol=1e-3)


def test_a_zero_beta_edit_policy_reduces_to_a_pure_passthrough() -> None:
    """beta -> 0 means the edit can never move the base action at all."""
    policy = _policy(beta=1e-8)
    features = torch.randn(3, 16)
    base = torch.randn(3, 12 * 29)
    edit, _ = policy.sample(features, base, mode="train")
    assert edit.abs().max().item() < 1e-6
