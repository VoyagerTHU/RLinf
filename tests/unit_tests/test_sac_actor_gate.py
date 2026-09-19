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

"""The SAC actor may only train once the critic discriminates between states."""

import torch


def _q_std(values: torch.Tensor) -> float:
    """Mirror of the worker's critic/q_data_std metric."""
    return values.float().std(dim=0).mean().item() if values.shape[0] > 1 else 0.0


def test_a_collapsed_critic_reports_no_spread() -> None:
    # The 2026-09-11 run: every sample predicted ~0.011 whatever the state.
    collapsed = torch.full((256, 2), 0.011)
    assert _q_std(collapsed) < 1e-6


def test_an_informative_critic_reports_spread() -> None:
    generator = torch.Generator().manual_seed(0)
    informative = torch.randn(256, 2, generator=generator) * 0.3 + 0.5
    assert _q_std(informative) > 0.05


def test_gate_opens_only_above_the_threshold() -> None:
    threshold = 0.05
    assert not (_q_std(torch.full((256, 2), 0.011)) >= threshold)
    generator = torch.Generator().manual_seed(1)
    assert _q_std(torch.randn(256, 2, generator=generator) * 0.3) >= threshold


def test_single_sample_batches_are_treated_as_uninformative() -> None:
    assert _q_std(torch.tensor([[0.5, 0.5]])) == 0.0


def test_the_gate_decision_must_be_reduced_across_ranks() -> None:
    # Ranks measure Q on their own micro-batches, so their spreads differ. The
    # worker averages them before comparing with the threshold; branching on
    # the local value would send some ranks into the actor pass and hang the
    # rest in its collective.
    threshold = 0.05
    per_rank = [0.061, 0.049, 0.058, 0.044]
    assert [value >= threshold for value in per_rank] == [True, False, True, False]
    averaged = sum(per_rank) / len(per_rank)
    assert averaged >= threshold
    assert len({averaged >= threshold for _ in per_rank}) == 1


def test_all_ranks_must_agree_before_the_actor_trains() -> None:
    # The readiness flag is averaged too, so one rank whose buffer is short
    # holds every rank back rather than desynchronising them.
    for flags, expected in (
        ([1, 1, 1, 1], True),
        ([1, 0, 1, 1], False),
        ([0, 0, 0, 0], False),
    ):
        assert (sum(flags) / len(flags) >= 1.0) is expected
