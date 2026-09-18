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

"""Input conditioning of the StarVLA SAC critic."""

import numpy as np
import torch

from rlinf.models.embodiment.starvla.starvla_action_model import StarVLAMultiQHead
from rlinf.models.embodiment.starvla.utils.action_space import (
    normalize_actions_from_env_torch,
    unnormalize_actions_for_env_torch,
)

STATS = {
    "q99": np.linspace(0.5, 3.0, 29),
    "q01": np.linspace(-1.5, -0.2, 29),
    "mask": np.ones(29, dtype=bool),
}


def _massive_features(batch: int = 8, hidden: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(batch, hidden, generator=generator)
    features[:, 5] *= 2000.0  # a Qwen massive-activation channel
    return features


def test_state_norm_removes_the_massive_channel_scale() -> None:
    head = StarVLAMultiQHead(
        hidden_size=64, action_feature_dim=29, hidden_dims=[32, 16], num_q_heads=2
    )
    normalized = head.state_norm(_massive_features().float())
    assert normalized.pow(2).mean(dim=-1).max().item() < 2.0
    assert torch.allclose(
        head.state_norm(100.0 * _massive_features().float()), normalized, atol=1e-4
    )


def test_state_norm_adds_no_parameters() -> None:
    plain = StarVLAMultiQHead(
        hidden_size=64,
        action_feature_dim=29,
        hidden_dims=[32, 16],
        num_q_heads=2,
        input_norm=False,
    )
    normed = StarVLAMultiQHead(
        hidden_size=64, action_feature_dim=29, hidden_dims=[32, 16], num_q_heads=2
    )
    assert set(dict(plain.named_parameters())) == set(dict(normed.named_parameters()))


def test_twin_q_output_shape_is_preserved() -> None:
    head = StarVLAMultiQHead(
        hidden_size=64, action_feature_dim=29, hidden_dims=[32, 16], num_q_heads=2
    )
    values = head(_massive_features(), torch.zeros(8, 29))
    assert values.shape == (8, 2) and torch.isfinite(values).all()


def test_action_normalization_round_trips() -> None:
    generator = torch.Generator().manual_seed(1)
    normalized = torch.rand(6, 2, 29, generator=generator) * 2 - 1
    env = unnormalize_actions_for_env_torch(normalized, STATS, policy_setup="gr1")
    back = normalize_actions_from_env_torch(env, STATS, policy_setup="gr1")
    assert torch.allclose(back, normalized, atol=1e-5)


def test_action_normalization_keeps_the_gradient_to_the_actor() -> None:
    normalized = torch.zeros(1, 1, 29, requires_grad=True)
    env = unnormalize_actions_for_env_torch(normalized, STATS, policy_setup="gr1")
    normalize_actions_from_env_torch(env, STATS, policy_setup="gr1").sum().backward()
    assert normalized.grad is not None and torch.isfinite(normalized.grad).all()
    assert normalized.grad.abs().sum() > 0


def test_environment_scale_actions_are_brought_to_unit_range() -> None:
    # Raw env actions span offsets up to 3.0; the critic should not see that.
    generator = torch.Generator().manual_seed(2)
    env = torch.as_tensor(STATS["q01"], dtype=torch.float32) + torch.rand(
        16, 29, generator=generator
    ) * torch.as_tensor(STATS["q99"] - STATS["q01"], dtype=torch.float32)
    assert env.abs().max() > 1.5
    assert (
        normalize_actions_from_env_torch(env, STATS, policy_setup="gr1").abs().max()
        <= 1.0 + 1e-5
    )
