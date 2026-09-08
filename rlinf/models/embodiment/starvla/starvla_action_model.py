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

"""RLinf 'BasePolicy' adapter for starVLA checkpoints."""

from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiQHead

from .dispatch import get_default_forward_handler, get_rollout_handler
from .utils import action_space as action_space_utils
from .utils import data_pipeline as data_pipeline_utils
from .utils import state as state_utils
from .utils.profile import (
    infer_hidden_size,
    infer_policy_profile,
    iter_gradient_checkpointing_targets,
)

logger = logging.getLogger(__name__)


class StarVLAValueHead(nn.Linear):
    """Small PPO critic head that can be targeted by FSDP wrap policies."""


class StarVLAMultiQHead(MultiQHead):
    """Twin SAC critics exposed as a distinct FSDP wrapping target."""


class StarVLAForRLActionPrediction(nn.Module, BasePolicy):
    """RLinf policy wrapper for starVLA checkpoints.

    This wrapper adapts a starVLA model to RLinf's embodied 'BasePolicy'
    interface. Training-time 'default_forward' and rollout-time
    'predict_action_batch' are dispatched based on the action head type inferred
    from the loaded checkpoint.

    For continuous-action heads, the wrapper can optionally unnormalize actions
    into environment space using normalization statistics from the checkpoint
    (or runtime overrides).
    """

    def __init__(
        self,
        starvla_model: nn.Module,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool = True,
        unnorm_key: Optional[str] = None,
        action_stats_source: str = "minmax",
        enable_state_input: bool = True,
        policy_setup: Optional[str] = None,
        initial_logstd: float = -2.5,
        trainable_logstd: bool = True,
        num_executed_action_chunks: Optional[int] = None,
        rollout_prompt_seq_len: Optional[int] = None,
        add_q_head: bool = False,
        num_q_heads: int = 2,
        q_hidden_dims: tuple[int, ...] = (512, 256),
        sac_task_description: Optional[str] = None,
    ):
        super().__init__()

        # 1) Core wiring / shape contracts.
        self.starvla_model = starvla_model
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.num_executed_action_chunks = (
            self.num_action_chunks
            if num_executed_action_chunks is None
            else int(num_executed_action_chunks)
        )
        if not 1 <= self.num_executed_action_chunks <= self.num_action_chunks:
            raise ValueError(
                "num_executed_action_chunks must be between 1 and "
                f"num_action_chunks={self.num_action_chunks}, got "
                f"{self.num_executed_action_chunks}"
            )
        if unnorm_key is None:
            raise ValueError(
                "starVLA requires cfg.unnorm_key to unnormalize actions for env rollout. "
                "Set 'actor.model.unnorm_key' (e.g. 'franka' for LIBERO)."
            )
        self.unnorm_key = str(unnorm_key)
        self.action_stats_source = str(action_stats_source)
        self.enable_state_input = bool(enable_state_input)
        self.policy_setup = str(policy_setup).strip().lower() if policy_setup else None

        # 2) Action unnormalization stats (strict: required when unnorm_key is set).
        self._action_norm_stats = action_space_utils.resolve_action_norm_stats(
            starvla_model=self.starvla_model,
            unnorm_key=self.unnorm_key,
            action_dim=self.action_dim,
            action_stats_source=self.action_stats_source,
            preserve_float64=self.policy_setup == "gr1",
        )

        # 3) Dispatch profile (action head + state adapter).
        policy_profile = infer_policy_profile(starvla_model)
        self.action_head_type = policy_profile["action_head_type"]
        self.state_adapter_type = policy_profile["state_adapter_type"]
        self.vlm_type = policy_profile["vlm_type"]

        # 4) Resolve policy parameter dtype (used for added heads/params).
        policy_param_dtype = next(
            (p.dtype for p in starvla_model.parameters() if p.is_floating_point()),
            torch.float32,
        )

        # 5) RL heads/params (optional value head + Gaussian log-std).
        hidden_size = (
            infer_hidden_size(starvla_model)
            if add_value_head or add_q_head
            else None
        )
        self.value_head: Optional[nn.Module] = None
        if add_value_head:
            # The head is randomly initialized and optimized separately from
            # the policy.  Keeping its two small tensors in FP32 avoids losing
            # 1e-4-scale critic updates to the BF16 checkpoint grid.
            self.value_head = StarVLAValueHead(
                hidden_size, 1, dtype=torch.float32
            )

        self.q_head: Optional[nn.Module] = None
        if add_q_head:
            action_feature_dim = (
                self.num_executed_action_chunks * self.action_dim
            )
            self.q_head = StarVLAMultiQHead(
                hidden_size=hidden_size,
                action_feature_dim=action_feature_dim,
                hidden_dims=[int(dim) for dim in q_hidden_dims],
                num_q_heads=int(num_q_heads),
            ).to(dtype=torch.float32)
            # Sparse binary RoboCasa returns have an initial value of zero. A
            # zero output layer prevents arbitrary random Q bootstraps while
            # preserving trainable hidden layers and independent twin critics.
            for critic in self.q_head.qs:
                nn.init.zeros_(critic.net[-1].weight)
                if critic.net[-1].bias is not None:
                    nn.init.zeros_(critic.net[-1].bias)
        self.sac_task_description = (
            None
            if sac_task_description is None
            else str(sac_task_description).strip()
        )

        action_model = getattr(starvla_model, "action_model", None)
        action_param_dtype = next(
            (
                p.dtype
                for p in action_model.parameters()
                if p.is_floating_point()
            ),
            policy_param_dtype,
        ) if isinstance(action_model, nn.Module) else policy_param_dtype
        actor_logstd = torch.full(
            (self.action_dim,), float(initial_logstd), dtype=action_param_dtype
        )
        self._configured_fixed_actor_logstd = (
            None if trainable_logstd else float(initial_logstd)
        )
        if trainable_logstd:
            self.actor_logstd = nn.Parameter(actor_logstd)
        else:
            # FSDP with ``use_orig_params=False`` cannot flatten a module that
            # mixes trainable and frozen parameters. A persistent buffer keeps
            # the exploration scale fixed while retaining it in state_dicts and
            # actor-to-rollout weight synchronization.
            self.register_buffer("actor_logstd", actor_logstd, persistent=True)

        # 6) Rollout/training caches. When supplied, this must match or exceed
        # the tokenized prompt. Using the exact fixed length keeps rollout and
        # actor replay on the same numerical computation path.
        self._rollout_prompt_seq_len = (
            None if rollout_prompt_seq_len is None else int(rollout_prompt_seq_len)
        )
        if (
            self._rollout_prompt_seq_len is not None
            and self._rollout_prompt_seq_len <= 0
        ):
            raise ValueError("rollout_prompt_seq_len must be positive when provided")

    @torch.no_grad()
    def restore_configured_fixed_actor_logstd(self) -> None:
        """Restore the configured exploration scale after checkpoint loading.

        Older RL checkpoints persist ``actor_logstd`` even when it is a fixed
        buffer. Loading one of those checkpoints must not silently override a
        new run's explicitly configured exploration scale.
        """
        if self._configured_fixed_actor_logstd is None:
            return
        self.actor_logstd.fill_(self._configured_fixed_actor_logstd)

    @property
    def uses_state_input(self) -> bool:
        """Return whether the active policy path should consume proprio/state."""
        return self.enable_state_input and self.action_head_type != "oft"

    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Dispatch forward passes for RLinf.

        Args:
            forward_type: Which forward path to run.
            **kwargs: Forward inputs forwarded to 'default_forward'.

        Returns:
            A dict containing optional RL terms: 'logprobs', 'entropy', and 'values'.

        Raises:
            NotImplementedError: If 'forward_type' is not supported.
        """
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def _run_sac_oft_policy(
        self, obs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Normal, dict]:
        """Run the pretrained OFT actor from raw replay-buffer observations."""
        if self.action_head_type != "oft":
            raise NotImplementedError(
                "StarVLA SAC currently supports only the OFT action head, got "
                f"{self.action_head_type!r}"
            )
        if not isinstance(obs, dict) or "main_images" not in obs:
            raise ValueError("StarVLA SAC requires obs['main_images']")

        sac_obs = dict(obs)
        if not sac_obs.get("task_descriptions"):
            if not self.sac_task_description:
                raise ValueError(
                    "StarVLA SAC replay observations do not store strings; set "
                    "actor.model.sac_task_description"
                )
            batch_size = int(sac_obs["main_images"].shape[0])
            sac_obs["task_descriptions"] = [self.sac_task_description] * batch_size

        if self.policy_setup == "gr1":
            from .utils.vlm_preprocess import get_train_image_size

            target_size = get_train_image_size(self.starvla_model)
            if target_size:
                sac_obs = data_pipeline_utils.resize_env_obs_images_cv2(
                    sac_obs, target_size
                )

        examples = data_pipeline_utils.build_examples_from_env_obs(
            env_obs=sac_obs,
            state_adapter_name=self.state_adapter_type,
            prepare_state_tensor=partial(
                state_utils.prepare_state_tensor,
                starvla_model=self.starvla_model,
                default_state_adapter_name=self.state_adapter_type,
            ),
            include_state=self.uses_state_input,
        )
        from .action_heads.oft import _build_oft_vlm_inputs, _run_oft_backbone_and_head

        model_inputs = _build_oft_vlm_inputs(
            self.starvla_model,
            num_action_chunks=self.num_action_chunks,
            examples=examples,
        )
        mean_actions, last_hidden, dist = _run_oft_backbone_and_head(
            self,
            model_inputs=model_inputs,
            use_cache=False,
        )
        return mean_actions, last_hidden, dist, model_inputs

    @staticmethod
    def _pool_sac_features(
        last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden[:, -1]
        attention_mask = attention_mask.to(device=last_hidden.device)
        indices = attention_mask.long().sum(dim=1) - 1
        rows = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        return last_hidden[rows, indices]

    def sac_forward(
        self, obs: dict[str, Any], mode: str = "train", **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample differentiable environment-space action chunks for SAC."""
        del kwargs
        mean_actions, last_hidden, dist, model_inputs = self._run_sac_oft_policy(obs)
        normalized_actions = mean_actions if mode == "eval" else dist.rsample()
        env_actions = action_space_utils.unnormalize_actions_for_env_torch(
            normalized_actions,
            self._action_norm_stats,
            policy_setup=self.policy_setup,
        )
        logprobs = dist.log_prob(normalized_actions).sum(dim=(-1, -2))
        features = self._pool_sac_features(
            last_hidden, model_inputs.get("attention_mask")
        )
        return env_actions, logprobs.to(torch.float32), features

    def sac_q_forward(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        shared_feature: Optional[torch.Tensor] = None,
        detach_encoder: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate twin Q heads on a raw observation and action chunk."""
        del kwargs
        if self.q_head is None:
            raise RuntimeError("StarVLA SAC_Q requires actor.model.add_q_head=true")
        if shared_feature is None:
            _, last_hidden, _, model_inputs = self._run_sac_oft_policy(obs)
            shared_feature = self._pool_sac_features(
                last_hidden, model_inputs.get("attention_mask")
            )
        if detach_encoder:
            shared_feature = shared_feature.detach()
        actions = actions.reshape(actions.shape[0], -1)
        q_dtype = next(self.q_head.parameters()).dtype
        return self.q_head(
            shared_feature.to(dtype=q_dtype), actions.to(dtype=q_dtype)
        )

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Run training-time forward for PPO terms (logprob/entropy/value).

        This method delegates to an action-head-specific handler based on the
        inferred 'action_head_type' of the wrapped starVLA checkpoint.

        Args:
            forward_inputs: Cached rollout tensors produced by
                'predict_action_batch'.
            compute_logprobs: Whether to compute action log-probabilities.
            compute_entropy: Whether to compute policy entropy.
            compute_values: Whether to compute value baseline.
            use_cache: Whether to enable backbone kv-cache when supported.

        Returns:
            Dict with optional RL terms: 'logprobs', 'entropy', and 'values'.

        Raises:
            ValueError: If no usable batch tensors are provided.
            NotImplementedError: If the action head type is not supported.
        """
        if not isinstance(forward_inputs, dict) or not forward_inputs:
            raise ValueError(
                "starVLA.default_forward requires 'forward_inputs' (dict[str, Tensor]) in RLinf training."
            )
        data = {k: v for k, v in forward_inputs.items() if isinstance(v, torch.Tensor)}
        if not data:
            raise ValueError(
                "starVLA.default_forward requires tensor values inside 'forward_inputs'."
            )
        # Automatically dispatch to the correct default forward handler based on action head type.
        handler = get_default_forward_handler(self.action_head_type)
        if handler is None:
            raise NotImplementedError(
                "default_forward not implemented for starVLA action head "
                f"{self.action_head_type}."
            )
        return handler(
            self,
            data=data,
            compute_logprobs=compute_logprobs,
            compute_entropy=compute_entropy,
            compute_values=compute_values,
            use_cache=use_cache,
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict a batch of actions and return rollout caches for training replay.

        Args:
            env_obs: Environment observation dict. Must include fields required
                by the configured state adapter and VLM pre-processing.
            calculate_logprobs: Whether to compute rollout-time logprob baseline.
            calculate_values: Whether to compute rollout-time value baseline.
            return_obs: Kept for compatibility with RLinf policy interface.
            mode: Rollout mode, typically "train" or "eval".
            **kwargs: Sampling overrides, such as 'do_sample', 'temperature',
                'top_k', 'top_p', 'max_new_tokens', and 'max_length'.

        Returns:
            Tuple of '(actions, result)' where:
                - 'actions' is a numpy array shaped [B, T, D] in env action space.
                - 'result' contains 'prev_logprobs', 'prev_values', and
                  'forward_inputs' for training replay.
        """
        del return_obs
        if self.value_head is None:
            calculate_values = False

        if self.policy_setup == "gr1":
            from .utils.vlm_preprocess import get_train_image_size

            target_size = get_train_image_size(self.starvla_model)
            if target_size:
                env_obs = data_pipeline_utils.resize_env_obs_images_cv2(
                    env_obs, target_size
                )

        # Build examples based on env_obs and state adapter.
        examples = data_pipeline_utils.build_examples_from_env_obs(
            env_obs=env_obs,
            state_adapter_name=self.state_adapter_type,
            prepare_state_tensor=partial(
                state_utils.prepare_state_tensor,
                starvla_model=self.starvla_model,
                default_state_adapter_name=self.state_adapter_type,
            ),
            include_state=self.uses_state_input,
        )
        # Build sampling kwargs and initialize forward_inputs with batch-aligned sampling tensors.
        sampling_kwargs = {
            "do_sample": kwargs.pop("do_sample", False),
            "temperature": kwargs.pop("temperature", 1.0),
            "top_k": kwargs.pop("top_k", 0),
            "top_p": kwargs.pop("top_p", 1.0),
            "max_new_tokens": kwargs.pop("max_new_tokens", None),
            "max_length": kwargs.pop("max_length", None),
        }
        if mode in {"train", "training"}:
            sampling_kwargs["do_sample"] = True
        bsz = int(len(examples))
        if bsz <= 0:
            raise ValueError(f"Empty rollout batch: len(examples)={bsz}.")
        forward_inputs = {
            "do_sample": torch.full(
                (bsz,), int(sampling_kwargs["do_sample"]), dtype=torch.int64
            ),
            "temperature": torch.full(
                (bsz,), float(sampling_kwargs["temperature"]), dtype=torch.float32
            ),
            "top_k": torch.full(
                (bsz,), int(sampling_kwargs["top_k"]), dtype=torch.int64
            ),
            "top_p": torch.full(
                (bsz,), float(sampling_kwargs["top_p"]), dtype=torch.float32
            ),
        }
        # initialize rollout caches
        prev_logprobs: Optional[torch.Tensor] = None
        prev_values: Optional[torch.Tensor] = None
        output: dict[str, Any]
        model_inputs: dict[str, Any] = {}
        extra_forward_inputs: dict[str, Any] = {}
        state: Optional[torch.Tensor] = None
        # Automatically dispatch to the correct rollout handler based on action head type.
        rollout_handler = get_rollout_handler(self.action_head_type)
        if rollout_handler is not None:
            payload = rollout_handler(
                self,
                examples=examples,
                env_obs=env_obs,
                mode=mode,
                calculate_logprobs=calculate_logprobs,
                calculate_values=calculate_values,
                sampling_kwargs=sampling_kwargs,
            )
            output = payload["output"]
            model_inputs = payload.get("model_inputs", {})
            if not isinstance(model_inputs, dict):
                model_inputs = (
                    dict(model_inputs) if hasattr(model_inputs, "items") else {}
                )
            prev_logprobs = payload.get("prev_logprobs")
            prev_values = payload.get("prev_values")
            extra_forward_inputs = payload.get("extra_forward_inputs", {})
            state = payload.get("state")
        else:
            raise NotImplementedError(
                "Unsupported starVLA model for rollout fallback: "
                f"action_head={self.action_head_type}."
            )

        if calculate_logprobs and prev_logprobs is None:
            raise RuntimeError(
                "Rollout handler did not return 'prev_logprobs' but calculate_logprobs=True was requested."
            )

        if calculate_values and prev_values is None:
            raise RuntimeError(
                "Rollout handler did not return 'prev_values' but calculate_values=True was requested."
            )

        if model_inputs:
            model_inputs, target_len = (
                data_pipeline_utils.normalize_model_inputs_for_storage(
                    model_inputs=model_inputs,
                    starvla_model=self.starvla_model,
                    rollout_prompt_seq_len=self._rollout_prompt_seq_len,
                )
            )
            self._rollout_prompt_seq_len = target_len
            model_inputs = data_pipeline_utils.pack_model_inputs_for_storage(
                model_inputs=model_inputs,
                batch_size=bsz,
            )
        # prepare actions: ensure shape [B, T, D] and unnormalize if needed
        normalized_actions = np.asarray(output["normalized_actions"])
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[:, None, :]

        act_bsz, n_chunks, act_dim = normalized_actions.shape
        if act_bsz != bsz:
            raise RuntimeError(
                "Rollout output batch size mismatch: "
                f"len(examples)={bsz}, actions_bsz={act_bsz}."
            )
        if act_dim != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: model returns {act_dim}, expected {self.action_dim}"
            )
        if n_chunks != self.num_executed_action_chunks:
            raise ValueError(
                "Executed action-chunk mismatch: model returns "
                f"{n_chunks}, expected {self.num_executed_action_chunks}"
            )
        env_chunk_actions = action_space_utils.unnormalize_actions_for_env(
            normalized_actions=normalized_actions.astype(np.float32),
            action_norm_stats=self._action_norm_stats,
            policy_setup=self.policy_setup,
        )

        forward_inputs["action"] = torch.from_numpy(env_chunk_actions.reshape(bsz, -1))

        storage_inputs = dict(model_inputs)
        storage_inputs.update(extra_forward_inputs)
        forward_inputs.update(
            {k: v for k, v in storage_inputs.items() if isinstance(v, torch.Tensor)}
        )
        if state is not None:
            forward_inputs["state"] = state.detach().cpu()

        for key, tensor in list(forward_inputs.items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim == 0:
                forward_inputs[key] = tensor.view(1).repeat(bsz)
                continue
            if tensor.shape[0] == 1:
                forward_inputs[key] = tensor.expand(bsz, *tensor.shape[1:]).clone()
                continue
            if tensor.shape[0] != bsz:
                raise RuntimeError(
                    f"forward_inputs['{key}'] has leading dim {tensor.shape[0]}, "
                    f"but rollout batch size is {bsz}. "
                    "Expected scalar/[1,...]/[B,...] tensor for trajectory splitting."
                )

        result = {
            "prev_logprobs": prev_logprobs if calculate_logprobs else None,
            "prev_values": prev_values if calculate_values else None,
            "forward_inputs": forward_inputs,
        }
        return env_chunk_actions, result

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enable gradient checkpointing on supported starVLA submodules.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs forwarded to submodules
                that support 'gradient_checkpointing_enable'.
        """
        enabled = False
        warned_types: set[type] = set()
        for module in iter_gradient_checkpointing_targets(self.starvla_model):
            fn = getattr(module, "gradient_checkpointing_enable", None)
            if not callable(fn):
                continue
            try:
                if gradient_checkpointing_kwargs is None:
                    fn()
                else:
                    try:
                        fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                    except TypeError:
                        fn()
            except ValueError as exc:
                if type(module) not in warned_types:
                    warnings.warn(
                        f"gradient_checkpointing_enable skipped for {type(module).__name__}: {exc}",
                        stacklevel=2,
                    )
                    warned_types.add(type(module))
                continue
            enabled = True

        if not enabled:
            warnings.warn(
                "gradient_checkpointing_enable() was requested, but no wrapped starVLA "
                "submodule exposes this API.",
                stacklevel=2,
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on supported starVLA submodules."""
        disabled = False
        warned_types: set[type] = set()
        for module in iter_gradient_checkpointing_targets(self.starvla_model):
            fn = getattr(module, "gradient_checkpointing_disable", None)
            if not callable(fn):
                continue
            try:
                fn()
            except ValueError as exc:
                if type(module) not in warned_types:
                    warnings.warn(
                        f"gradient_checkpointing_disable skipped for {type(module).__name__}: {exc}",
                        stacklevel=2,
                    )
                    warned_types.add(type(module))
                continue
            disabled = True

        if not disabled:
            warnings.warn(
                "gradient_checkpointing_disable() was requested, but no wrapped starVLA "
                "submodule exposes this API.",
                stacklevel=2,
            )
