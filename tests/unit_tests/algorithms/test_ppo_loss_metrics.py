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

"""Tests for PPO loss metrics under embodied token-level masks."""

import pytest
import torch

from rlinf.algorithms.losses import (
    compute_decoupled_ppo_actor_loss,
    compute_ppo_actor_loss,
    compute_ppo_critic_loss,
)
from rlinf.algorithms.utils import compute_update_level_explained_variance


def _token_level_inputs(batch: int = 4, chunks: int = 3, dims: int = 5):
    generator = torch.Generator().manual_seed(0)
    logprobs = torch.randn(batch, chunks, dims, generator=generator)
    old_logprobs = logprobs + 0.3 * torch.randn(
        batch, chunks, dims, generator=generator
    )
    advantages = torch.randn(batch, 1, 1, generator=generator)
    # Chunk-level mask broadcast over action dimensions; last sample invalid.
    loss_mask = torch.ones(batch, 1, 1, dtype=torch.bool)
    loss_mask[-1] = False
    return logprobs, old_logprobs, advantages, loss_mask


def test_ppo_actor_loss_metrics_count_every_masked_element() -> None:
    logprobs, old_logprobs, advantages, loss_mask = _token_level_inputs()
    loss, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=loss_mask,
    )
    full_mask = loss_mask.expand_as(logprobs)
    expected_kl = -(logprobs - old_logprobs)[full_mask].mean()
    assert metrics["actor/approx_kl"] == pytest.approx(expected_kl.item(), rel=1e-5)
    assert 0.0 <= metrics["actor/clip_fraction"].item() <= 1.0

    # The loss itself averages the same elements as the fully expanded mask.
    expanded_loss, _ = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages.expand_as(logprobs),
        loss_mask=full_mask,
    )
    assert loss.item() == pytest.approx(expanded_loss.item(), rel=1e-6)


def test_decoupled_ppo_actor_loss_metrics_count_every_masked_element() -> None:
    logprobs, old_logprobs, advantages, loss_mask = _token_level_inputs()
    _, metrics = compute_decoupled_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=loss_mask,
    )
    assert 0.0 <= metrics["actor/clip_fraction"].item() <= 1.0
    full_mask = loss_mask.expand_as(logprobs)
    expected_kl = -(logprobs - old_logprobs)[full_mask].mean()
    assert metrics["actor/proximal_approx_kl"] == pytest.approx(
        expected_kl.item(), rel=1e-5
    )


def test_critic_loss_reports_sufficient_statistics_for_explained_variance() -> None:
    generator = torch.Generator().manual_seed(1)
    returns = torch.rand(8, generator=generator)
    values = returns + 0.1 * torch.randn(8, generator=generator)
    loss_mask = torch.ones(8, dtype=torch.bool)
    loss_mask[:2] = False
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=values.clone(),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=loss_mask,
    )
    valid_returns = returns[loss_mask]
    valid_values = values[loss_mask]
    assert metrics["critic/ev_count"] == 6
    assert metrics["critic/ev_sq_error_sum"].item() == pytest.approx(
        ((valid_returns - valid_values) ** 2).sum().item(), rel=1e-5
    )
    assert metrics["critic/ev_returns_sum"].item() == pytest.approx(
        valid_returns.sum().item(), rel=1e-5
    )

    # Averaging the sums over two identical micro-batches (as the actor does)
    # leaves the recombined explained variance unchanged.
    averaged = {
        key: (value.item() if torch.is_tensor(value) else value)
        for key, value in metrics.items()
    }
    recombined = compute_update_level_explained_variance(averaged)
    expected = 1.0 - ((valid_returns - valid_values) ** 2).mean() / valid_returns.var(
        correction=0
    )
    assert recombined["critic/explained_variance"] == pytest.approx(
        expected.item(), rel=1e-4
    )
    assert "critic/explained_variance_micro_batch" in recombined


def test_update_level_explained_variance_handles_missing_and_constant() -> None:
    assert compute_update_level_explained_variance({}) == {}
    constant = compute_update_level_explained_variance(
        {
            "critic/ev_count": 4.0,
            "critic/ev_sq_error_sum": 1.0,
            "critic/ev_returns_sum": 4.0,
            "critic/ev_returns_sq_sum": 4.0,
        }
    )
    assert constant["critic/explained_variance"] == 0.0
