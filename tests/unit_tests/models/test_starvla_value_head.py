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

"""Tests for the StarVLA PPO value head (input normalization + zero init)."""

import pytest
import torch

from rlinf.models.embodiment.starvla.starvla_action_model import StarVLAValueHead


def _massive_activation_features(batch: int = 6, hidden: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(batch, hidden, generator=generator)
    features[:, 3] *= (
        2000.0  # one massive-activation dimension, as in Qwen hidden states
    )
    return features


def test_zero_init_head_predicts_zero_and_exposes_proj_parameters() -> None:
    head = StarVLAValueHead(32)
    names = dict(head.named_parameters())
    assert set(names) == {"proj.weight", "proj.bias"}
    assert torch.all(head(_massive_activation_features()) == 0)
    assert head(_massive_activation_features()).dtype == torch.float32


@pytest.mark.parametrize("norm", ["layer_norm", "rms"])
def test_input_norm_removes_feature_scale(norm: str) -> None:
    head = StarVLAValueHead(32, input_norm=norm)
    x = _massive_activation_features()
    normalized = head.norm(x)
    # Unit-scale inputs regardless of the massive dimension.
    assert normalized.pow(2).mean(dim=-1).max().item() < 2.0
    # Scaling the raw feature by 100x does not change what the head sees.
    assert torch.allclose(head.norm(100.0 * x), normalized, atol=1e-4)


def test_no_norm_keeps_raw_scale() -> None:
    head = StarVLAValueHead(32, input_norm="none", zero_init=False)
    x = _massive_activation_features()
    torch.nn.init.ones_(head.proj.weight)
    assert head(x).abs().max().item() > 1000.0


def test_bf16_features_are_upcast() -> None:
    head = StarVLAValueHead(32, zero_init=False)
    out = head(_massive_activation_features().to(torch.bfloat16))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_invalid_norm_rejected() -> None:
    with pytest.raises(ValueError, match="value_head_input_norm"):
        StarVLAValueHead(32, input_norm="batch_norm")
