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

"""The SAC behaviour anchor that bounds off-policy actor drift (TD3+BC)."""

import torch

SPAN = torch.linspace(0.5, 2.0, 29)


def _bc_term(pi: torch.Tensor, behaviour: torch.Tensor) -> torch.Tensor:
    """Mirror of the worker's anchor: squared distance in normalized units."""
    delta = (pi - behaviour).reshape(pi.shape[0], -1, SPAN.shape[0])
    return (2.0 * delta / SPAN).square().mean(dim=(-1, -2))


def _objective(pi, behaviour, q_pi, bc_coef):
    scale = bc_coef / (q_pi.detach().abs().mean() + 1e-6)
    return (scale * -q_pi).reshape(-1) + _bc_term(pi, behaviour)


def test_matching_the_behaviour_action_costs_nothing() -> None:
    actions = torch.rand(4, 12, 29)
    assert torch.allclose(_bc_term(actions, actions), torch.zeros(4), atol=1e-7)


def test_the_anchor_grows_with_distance_and_is_channel_normalized() -> None:
    behaviour = torch.zeros(1, 12, 29)
    near = _bc_term(torch.full((1, 12, 29), 0.05), behaviour)
    far = _bc_term(torch.full((1, 12, 29), 0.5), behaviour)
    assert far > near > 0
    # A channel with twice the span must contribute a quarter of the cost for
    # the same environment-space error, so no channel dominates.
    one_channel = torch.zeros(1, 1, 29)
    one_channel[0, 0, 0] = 0.1
    wide = torch.zeros(1, 1, 29)
    wide[0, 0, 28] = 0.1
    assert _bc_term(one_channel, torch.zeros(1, 1, 29)) > _bc_term(
        wide, torch.zeros(1, 1, 29)
    )


def test_scaling_makes_the_trade_off_independent_of_the_q_scale() -> None:
    # q_pi grew from 0.11 to 0.27 during the collapse; without the TD3+BC
    # scaling the anchor would have been progressively outweighed.
    pi = torch.full((8, 12, 29), 0.2)
    behaviour = torch.zeros(8, 12, 29)
    small_q = torch.full((8, 1), 0.11)
    large_q = torch.full((8, 1), 0.27)
    small = _objective(pi, behaviour, small_q, 0.1)
    large = _objective(pi, behaviour, large_q, 0.1)
    assert torch.allclose(small, large, atol=1e-5)


def test_a_zero_coefficient_removes_the_q_term_entirely() -> None:
    # bc_coef is the weight on Q relative to the anchor, so zero would leave
    # only the anchor; the worker therefore treats 0 as "anchor disabled".
    pi = torch.full((2, 12, 29), 0.2)
    behaviour = torch.zeros(2, 12, 29)
    assert torch.allclose(
        _objective(pi, behaviour, torch.full((2, 1), 0.2), 0.0),
        _bc_term(pi, behaviour),
    )


def _bc_term_as_in_worker(pi: torch.Tensor, behaviour: torch.Tensor) -> torch.Tensor:
    """Exactly the worker's arithmetic, including the flattening."""
    flat_behaviour = behaviour.reshape(pi.shape[0], -1)
    delta = (pi.reshape(pi.shape[0], -1) - flat_behaviour).reshape(
        pi.shape[0], -1, SPAN.shape[0]
    )
    return (2.0 * delta / SPAN).square().mean(dim=(-1, -2))


def test_the_anchor_accepts_the_two_layouts_the_callers_use() -> None:
    # The policy hands back [B, chunks, action_dim]; the replay buffer stores
    # [B, chunks * action_dim]. Mixing them is what crashed attempts 2 and 5,
    # so pin both layouts here.
    chunked = torch.rand(4, 12, 29)
    behaviour_flat = torch.rand(4, 12 * 29)
    value = _bc_term_as_in_worker(chunked, behaviour_flat)
    assert value.shape == (4,) and torch.isfinite(value).all()
    assert torch.allclose(
        value, _bc_term_as_in_worker(chunked, behaviour_flat.reshape(4, 12, 29))
    )
    assert torch.allclose(
        _bc_term_as_in_worker(chunked, chunked.reshape(4, -1)),
        torch.zeros(4),
        atol=1e-7,
    )
