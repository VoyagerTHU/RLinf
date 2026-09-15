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

"""Routing of trainable parameters to the actor / critic / lora optimizer groups."""

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import classify_optimizer_param_group


def test_value_head_is_critic_even_with_norm_submodule() -> None:
    assert classify_optimizer_param_group("value_head.proj.weight") == "critic"
    assert classify_optimizer_param_group("model.value_head.weight") == "critic"


def test_peft_adapters_form_the_lora_group() -> None:
    name = (
        "starvla_model.qwen_vl_interface.model.base_model.model.model."
        "language_model.layers.3.self_attn.q_proj.lora_A.default.weight"
    )
    assert classify_optimizer_param_group(name) == "lora"
    assert classify_optimizer_param_group(name.replace("lora_A", "lora_B")) == "lora"


def test_action_head_and_logstd_stay_in_the_actor_group() -> None:
    assert (
        classify_optimizer_param_group("starvla_model.action_model.model.0.weight")
        == "actor"
    )
    assert classify_optimizer_param_group("actor_logstd") == "actor"
