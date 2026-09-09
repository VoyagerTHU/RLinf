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

"""Tests for per-parameter-group gradient clipping."""

import math

import pytest
import torch

from rlinf.hybrid_engines.fsdp.utils import clip_grad_norm_per_group_


def _groups():
    actor = torch.nn.Parameter(torch.zeros(4))
    actor.grad = torch.full((4,), 0.5)  # norm 1.0
    critic_a = torch.nn.Parameter(torch.zeros(3))
    critic_a.grad = torch.full((3,), 4.0)
    critic_b = torch.nn.Parameter(torch.zeros(1))
    critic_b.grad = torch.full((1,), 2.0)  # critic norm sqrt(48 + 4)
    return [
        {"name": "actor", "params": [actor]},
        {"name": "critic", "params": [critic_a, critic_b]},
    ]


def test_each_group_is_clipped_to_its_own_budget() -> None:
    groups = _groups()
    norms = clip_grad_norm_per_group_(
        groups,
        max_norms={"actor": 2.0, "critic": 1.0},
        default_max_norm=2.0,
        reduce_group=None,
        device=torch.device("cpu"),
    )
    assert norms["actor"] == pytest.approx(1.0)
    assert norms["critic"] == pytest.approx(math.sqrt(52.0))
    # Actor is within budget: untouched.
    assert torch.allclose(groups[0]["params"][0].grad, torch.full((4,), 0.5))
    # Critic is scaled to norm 1 while the actor keeps its full gradient.
    critic_grads = torch.cat([p.grad for p in groups[1]["params"]])
    assert critic_grads.norm().item() == pytest.approx(1.0, rel=1e-4)


def test_groups_without_gradients_report_zero_and_use_default_budget() -> None:
    frozen = torch.nn.Parameter(torch.zeros(2))
    active = torch.nn.Parameter(torch.zeros(2))
    active.grad = torch.full((2,), 3.0)  # norm sqrt(18)
    groups = [{"params": [frozen]}, {"name": "misc", "params": [active]}]
    norms = clip_grad_norm_per_group_(
        groups,
        max_norms={},
        default_max_norm=1.0,
        reduce_group=None,
        device=torch.device("cpu"),
    )
    assert norms["0"] == 0.0
    assert norms["misc"] == pytest.approx(math.sqrt(18.0))
    assert active.grad.norm().item() == pytest.approx(1.0, rel=1e-4)
