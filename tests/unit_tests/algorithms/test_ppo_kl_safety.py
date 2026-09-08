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

"""Tests for PPO KL early-stop safety checks and the adaptive reference leash."""

import math

import pytest

from rlinf.algorithms.utils import adapt_kl_beta, should_stop_ppo_update


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


def test_adapt_kl_beta_tightens_when_far_from_reference() -> None:
    assert adapt_kl_beta(0.001, measured_kl=0.2, target_kl=0.05) == pytest.approx(0.002)


def test_adapt_kl_beta_relaxes_when_close_to_reference() -> None:
    assert adapt_kl_beta(0.001, measured_kl=0.01, target_kl=0.05) == pytest.approx(
        0.0005
    )


@pytest.mark.parametrize("measured", [0.05, 0.05 * 1.5, 0.05 / 1.5, 0.04, 0.07])
def test_adapt_kl_beta_dead_band_keeps_coefficient(measured: float) -> None:
    assert adapt_kl_beta(0.001, measured_kl=measured, target_kl=0.05) == 0.001


def test_adapt_kl_beta_clamps_to_bounds() -> None:
    assert adapt_kl_beta(0.9, 1.0, 0.05, max_beta=1.0) == 1.0
    assert adapt_kl_beta(1.5e-4, 0.0, 0.05, min_beta=1e-4) == 1e-4
    # A coefficient outside the bounds is pulled back even inside the dead band.
    assert adapt_kl_beta(5.0, 0.05, 0.05, max_beta=1.0) == 1.0


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_adapt_kl_beta_treats_nonfinite_kl_as_too_far(value: float) -> None:
    assert adapt_kl_beta(0.001, value, 0.05) == pytest.approx(0.002)


def test_adapt_kl_beta_never_reaches_zero() -> None:
    """The penalty must stay measurable so the leash can re-engage later."""
    beta = 1.0
    for _ in range(100):
        beta = adapt_kl_beta(beta, measured_kl=0.0, target_kl=0.05)
    assert beta == 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_kl": 0.0},
        {"target_kl": 0.05, "factor": 1.0},
        {"target_kl": 0.05, "tolerance": 0.5},
        {"target_kl": 0.05, "min_beta": 0.0},
        {"target_kl": 0.05, "min_beta": 2.0, "max_beta": 1.0},
    ],
)
def test_adapt_kl_beta_rejects_invalid_controller_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        adapt_kl_beta(0.001, 0.05, **kwargs)
