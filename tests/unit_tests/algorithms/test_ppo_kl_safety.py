"""Tests for PPO KL early-stop safety checks."""

import math

import pytest

from rlinf.algorithms.utils import should_stop_ppo_update


def test_should_stop_ppo_update_respects_disabled_and_finite_thresholds() -> None:
    """A disabled boundary never stops; a configured boundary is strict."""
    assert not should_stop_ppo_update(100.0, None)
    assert not should_stop_ppo_update(5.0, 5.0)
    assert should_stop_ppo_update(5.01, 5.0)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_should_stop_ppo_update_rejects_nonfinite_kl(value: float) -> None:
    """Non-finite KL always stops an enabled PPO update."""
    assert should_stop_ppo_update(value, 5.0)


def test_should_stop_ppo_update_rejects_nonpositive_target() -> None:
    """A non-positive target is a configuration error."""
    with pytest.raises(ValueError, match="target_kl must be positive"):
        should_stop_ppo_update(0.0, 0.0)
