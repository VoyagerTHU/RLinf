"""Regression tests for PPO critic diagnostics."""

import torch

from rlinf.algorithms.losses import compute_ppo_critic_loss


def test_explained_variance_is_finite_for_single_valid_transition():
    """A singleton masked micro-batch must not poison update-level metrics."""
    values = torch.tensor([0.25, 0.5])
    returns = torch.tensor([1.0, 2.0])
    previous_values = torch.zeros_like(values)
    loss_mask = torch.tensor([True, False])

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=previous_values,
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=loss_mask,
    )

    assert torch.isfinite(loss)
    assert metrics["critic/explained_variance"].item() == 0.0


def test_explained_variance_preserves_nonconstant_signal():
    """The finite fallback must not alter the ordinary EV calculation."""
    values = torch.tensor([0.0, 1.0, 2.0])
    returns = torch.tensor([0.0, 2.0, 4.0])
    previous_values = values.clone()

    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=previous_values,
        value_clip=0.2,
        huber_delta=10.0,
    )

    expected = 1 - torch.var(returns - values, correction=0) / torch.var(
        returns, correction=0
    )
    torch.testing.assert_close(metrics["critic/explained_variance"], expected)
