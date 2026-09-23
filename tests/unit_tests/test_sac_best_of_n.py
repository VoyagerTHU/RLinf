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

"""The EXPO-FT best-of-N candidate selection (StarVLAForRLActionPrediction._select_best_of_n).

Mirrors the exact reshape/argmax/gather pattern the model method uses, with a
mock critic in place of the real twin Q heads, so the selection algorithm is
pinned independently of the backbone, the edit policy, and action
normalization.
"""

import torch


def _select(all_env_actions: torch.Tensor, q_agg_flat: torch.Tensor, num_base: int, batch_size: int):
    """Reimplementation of the tail of _select_best_of_n's selection step."""
    num_groups = all_env_actions.shape[0] // batch_size
    q_agg = q_agg_flat.reshape(num_groups, batch_size)
    best_group = torch.argmax(q_agg, dim=0)
    grouped_actions = all_env_actions.reshape(num_groups, batch_size, *all_env_actions.shape[1:])
    batch_index = torch.arange(batch_size)
    selected = grouped_actions[best_group, batch_index]
    frac_edited = (best_group >= num_base).float().mean().item()
    return selected, best_group, frac_edited


def test_selection_picks_the_highest_q_row_independently_per_batch_element() -> None:
    batch_size, num_base, chunks, dim = 3, 2, 4, 5
    num_groups = 2 * num_base
    all_actions = torch.arange(num_groups * batch_size * chunks * dim, dtype=torch.float32).reshape(
        num_groups * batch_size, chunks, dim
    )
    # Row layout is [group, batch] flattened group-major, matching
    # `all_env_actions` in _select_best_of_n (concat of base then edited,
    # each [num_base, batch, ...]).
    q_agg_flat = torch.zeros(num_groups * batch_size)
    winners = {0: 3, 1: 0, 2: 2}  # batch element -> winning group index
    for group in range(num_groups):
        for b in range(batch_size):
            row = group * batch_size + b
            q_agg_flat[row] = 10.0 if group == winners[b] else float(group)

    selected, best_group, _ = _select(all_actions, q_agg_flat, num_base, batch_size)
    for b in range(batch_size):
        expected_row = winners[b] * batch_size + b
        assert torch.equal(selected[b], all_actions[expected_row])
    assert best_group.tolist() == [winners[0], winners[1], winners[2]]


def test_frac_selected_is_edited_reports_the_edited_share() -> None:
    batch_size, num_base = 4, 3
    num_groups = 2 * num_base
    all_actions = torch.zeros(num_groups * batch_size, 1)
    q_agg_flat = torch.zeros(num_groups * batch_size)
    # Force every batch element to pick the last edited candidate (group
    # index num_groups - 1, which is >= num_base).
    winning_group = num_groups - 1
    for b in range(batch_size):
        q_agg_flat[winning_group * batch_size + b] = 100.0
    _, best_group, frac_edited = _select(all_actions, q_agg_flat, num_base, batch_size)
    assert (best_group == winning_group).all()
    assert frac_edited == 1.0


def test_eval_mode_has_exactly_two_candidates() -> None:
    # num_base=1 in eval mode (only the mean and its deterministic edit),
    # so num_groups must be exactly 2.
    batch_size, num_base = 5, 1
    num_groups = 2 * num_base
    all_actions = torch.randn(num_groups * batch_size, 2)
    q_agg_flat = torch.randn(num_groups * batch_size)
    selected, best_group, _ = _select(all_actions, q_agg_flat, num_base, batch_size)
    assert selected.shape == (batch_size, 2)
    assert set(best_group.tolist()) <= {0, 1}


def test_the_two_returned_action_tensors_stay_aligned_to_the_same_selection() -> None:
    """_select_best_of_n returns env-unit and normalized-unit actions from the
    SAME best_group index. A regression here (returning an env-unit action
    into a slot a caller then unnormalizes again, as happened once) produces
    a syntactically valid but numerically wrong action; this pins that the
    two tensors are gathered with identical indices, which is the part that
    matters regardless of which affine map connects their values."""
    batch_size, num_base, chunks, dim = 3, 2, 2, 4
    num_groups = 2 * num_base
    env_actions = torch.arange(num_groups * batch_size * chunks * dim, dtype=torch.float32).reshape(
        num_groups * batch_size, chunks, dim
    )
    # A different, independently identifiable tensor standing in for the
    # normalized-unit candidates, so a mix-up between the two would be
    # detectable (not just a scale difference that could hide a bug).
    norm_actions = -env_actions - 1000.0

    q_agg_flat = torch.arange(num_groups * batch_size, dtype=torch.float32)
    num_groups_actual = env_actions.shape[0] // batch_size
    q_agg = q_agg_flat.reshape(num_groups_actual, batch_size)
    best_group = torch.argmax(q_agg, dim=0)
    batch_index = torch.arange(batch_size)

    grouped_env = env_actions.reshape(num_groups_actual, batch_size, chunks, dim)
    grouped_norm = norm_actions.reshape(num_groups_actual, batch_size, chunks, dim)
    selected_env = grouped_env[best_group, batch_index]
    selected_norm = grouped_norm[best_group, batch_index]

    for b in range(batch_size):
        row = best_group[b].item() * batch_size + b
        assert torch.equal(selected_env[b], env_actions[row])
        assert torch.equal(selected_norm[b], norm_actions[row])
        # The two must name the same (group, env) pick, not an independent one.
        assert torch.equal(selected_norm[b], -selected_env[b] - 1000.0)


def _use_best_of_n(has_edit_policy: bool, mode: str, gate_open) -> bool:
    """Mirror of run_rollout_oft's use_best_of_n condition."""
    return has_edit_policy and (
        mode == "eval" or bool(gate_open is not None and gate_open)
    )


def test_training_rollout_falls_back_to_plain_sampling_until_the_gate_opens() -> None:
    assert _use_best_of_n(True, "train", False) is False
    assert _use_best_of_n(True, "train", None) is False
    assert _use_best_of_n(True, "train", True) is True


def test_eval_always_uses_best_of_n_once_an_edit_policy_exists() -> None:
    # Eval is the deterministic 2-candidate path regardless of the training
    # gate, so official evaluation always reflects what the edit policy has
    # learned, not a frozen fallback.
    assert _use_best_of_n(True, "eval", False) is True
    assert _use_best_of_n(True, "eval", None) is True


def test_recipes_without_an_edit_policy_never_use_best_of_n() -> None:
    assert _use_best_of_n(False, "eval", True) is False
    assert _use_best_of_n(False, "train", True) is False
