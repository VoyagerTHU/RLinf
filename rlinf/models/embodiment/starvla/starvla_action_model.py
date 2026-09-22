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

import copy
import logging
import math
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


class StarVLAValueHead(nn.Module):
    """Linear PPO critic on a normalized backbone feature (FSDP wrap target).

    The critic reads the last-layer hidden state of a Qwen3-VL token. Those
    hidden states carry a few massive-activation dimensions (magnitudes in the
    hundreds to thousands), so a bare linear head sees critic gradient norms of
    ~1e3-1e4, oscillates with Adam at value_lr 1e-4, and its bootstrapped GAE
    returns swing negative (explained variance stayed below zero for 20
    updates on RoboCasa GR1). Normalizing the feature first (LayerNorm without
    affine parameters, or RMS) puts the head on a unit-scale input.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        input_norm: Optional[str] = "layer_norm",
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        norm = str(input_norm or "none").strip().lower()
        if norm == "layer_norm":
            self.norm: nn.Module = nn.LayerNorm(
                hidden_size, eps=eps, elementwise_affine=False, dtype=torch.float32
            )
        elif norm == "rms":
            self.norm = _RMSNorm(eps=eps)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(
                "value_head_input_norm must be one of 'layer_norm', 'rms', 'none', "
                f"got {input_norm!r}"
            )
        self.proj = nn.Linear(hidden_size, 1, dtype=torch.float32)
        if zero_init:
            # Sparse binary returns start near zero. A randomly initialized head
            # predicts values of magnitude ~10 on these features and makes the
            # first dozens of GAE updates fit noise.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(features.to(dtype=torch.float32)))


class _RMSNorm(nn.Module):
    """Parameter-free RMS normalization over the last dimension."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class StarVLAMultiQHead(MultiQHead):
    """Twin SAC critics exposed as a distinct FSDP wrapping target.

    The state feature is the Qwen3-VL last-layer hidden state, which carries
    massive-activation dimensions in the hundreds to thousands. ``QHead``
    applies LayerNorm only after its hidden layers, so raw features let a few
    channels dominate the first linear layer. That is the same failure that
    held the PPO value head's explained variance below zero for twenty updates;
    here it showed up as a critic whose batch-wise spread of Q grew a hundred
    times too slowly to ever inform the actor. A parameter-free LayerNorm on
    the input fixes it and adds no parameters, so checkpoints are unaffected.
    """

    def __init__(self, *args, input_norm: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = kwargs.get("hidden_size", args[0] if args else None)
        self.state_norm: nn.Module = (
            nn.LayerNorm(int(hidden_size), eps=1e-6, elementwise_affine=False)
            if input_norm and hidden_size
            else nn.Identity()
        )

    def forward(self, state_features, action_features):
        normalized = self.state_norm(state_features.float()).to(state_features.dtype)
        return super().forward(normalized, action_features)


class StarVLAEditPolicy(nn.Module):
    """Lightweight residual actor for EXPO-FT-style SAC fine-tuning.

    Maps (pooled state features, a candidate action in normalized units) to a
    small additive correction bounded to ``[-beta, beta]`` per channel. This
    module and the twin Q heads are the only parameters an EXPO-FT recipe
    trains with RL; the pretrained OFT head that generates candidate actions
    is frozen throughout, so nothing here can drift the pretrained policy the
    way a directly-trained actor mean can (see attempts 9-12 in
    SAC_SUMMARY.md). Reference: Dong et al., "EXPO-FT" (arXiv:2605.25477),
    which does the same thing on top of a pi0.5 flow-matching base; ours sits
    on top of a deterministic point-regression base instead, so the diversity
    among candidates in ``sac_best_of_n`` below comes entirely from resampling
    the base head's own fixed-variance Gaussian, not from an inherently
    multimodal generator.
    """

    def __init__(
        self,
        hidden_size: int,
        action_feature_dim: int,
        hidden_dims: tuple[int, ...],
        beta: float,
        input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.action_feature_dim = int(action_feature_dim)
        self.state_norm: nn.Module = (
            nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            if input_norm
            else nn.Identity()
        )
        layers: list[nn.Module] = []
        prev_dim = hidden_size + self.action_feature_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.ReLU(),
            ]
            prev_dim = int(hidden_dim)
        self.trunk: nn.Module = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(prev_dim, self.action_feature_dim)
        self.log_std_head = nn.Linear(prev_dim, self.action_feature_dim)
        # Zero-init the mean head so the edit starts at exactly 0
        # (tanh(0) = 0): training begins with the pretrained policy's
        # behaviour unchanged, and only moves it once Q says to.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(
        self,
        state_features: torch.Tensor,
        base_action_flat: torch.Tensor,
        mode: str = "train",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample (or take the mode of) the edit distribution.

        Returns ``(edit, log_prob)`` in normalized action units, both flat
        over the chunk/dim axes like ``base_action_flat``. ``log_prob`` is
        ``None`` in eval mode (the edit is then the deterministic mode).

        Named ``forward`` rather than a differently-named method (this
        module used to expose ``sample``) so calling it as ``self.edit_policy(...)``
        goes through ``nn.Module.__call__``: FSDP's parameter-gathering hooks
        fire on ``forward``/``__call__`` only, and this module is wrapped as
        its own FSDP unit (see wrap_policy.module_classes_to_wrap), so a
        direct ``self.edit_policy.sample(...)`` call ran every Linear against
        this rank's raw, un-gathered 1-D parameter shard instead of the full
        weight matrix.
        """
        features = torch.cat(
            [
                self.state_norm(state_features.float()),
                base_action_flat.float(),
            ],
            dim=-1,
        )
        hidden = self.trunk(features)
        mean = self.mean_head(hidden)
        if mode == "eval":
            return self.beta * torch.tanh(mean), None
        log_std = self.log_std_head(hidden).clamp(-5.0, 2.0)
        std = log_std.exp()
        pre_tanh = mean + std * torch.randn_like(mean)
        tanh_pre = torch.tanh(pre_tanh)
        edit = self.beta * tanh_pre
        # Standard SAC tanh-squash log-density, adjusted for the extra beta
        # scale factor: log p(edit) = log N(pre_tanh) - log|d edit/d pre_tanh|
        # = log N(pre_tanh) - sum[log(beta) + log(1 - tanh(pre_tanh)^2)].
        gaussian_log_prob = (
            -0.5
            * (((pre_tanh - mean) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        ).sum(dim=-1, keepdim=True)
        squash_correction = (
            math.log(self.beta) + torch.log1p(-tanh_pre.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return edit, gaussian_log_prob - squash_correction


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
        add_edit_policy: bool = False,
        edit_hidden_dims: tuple[int, ...] = (256, 256, 256),
        edit_beta: float = 0.1,
        expo_num_candidates: int = 8,
        sac_task_description: Optional[str] = None,
        value_head_zero_init: bool = True,
        value_head_input_norm: Optional[str] = "layer_norm",
        action_norm_stats: Optional[Any] = None,
        resize_images_to_train_size: Optional[bool] = None,
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
        # Resize env images to the checkpoint's training size with cv2
        # INTER_AREA (as StarVLA's own eval interfaces do). Legacy default:
        # only the GR1 setup resized; other setups opt in per config.
        self.resize_env_images = (
            self.policy_setup == "gr1"
            if resize_images_to_train_size is None
            else bool(resize_images_to_train_size)
        )

        # 2) Action unnormalization stats (strict: required when unnorm_key is set).
        self._action_norm_stats = action_space_utils.resolve_action_norm_stats(
            starvla_model=self.starvla_model,
            unnorm_key=self.unnorm_key,
            action_dim=self.action_dim,
            action_stats_source=self.action_stats_source,
            preserve_float64=self.policy_setup == "gr1",
            override_stats=action_norm_stats,
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
            if add_value_head or add_q_head or add_edit_policy
            else None
        )
        self.value_head: Optional[nn.Module] = None
        if add_value_head:
            # The head is optimized separately from the policy.  Keeping its
            # two small tensors in FP32 avoids losing 1e-4-scale critic updates
            # to the BF16 checkpoint grid; the input normalization tames the
            # massive-activation dimensions of the VLM hidden state.
            self.value_head = StarVLAValueHead(
                hidden_size,
                input_norm=value_head_input_norm,
                zero_init=value_head_zero_init,
            )

        self.action_feature_dim = self.num_executed_action_chunks * self.action_dim
        self.q_head: Optional[nn.Module] = None
        if add_q_head:
            action_feature_dim = self.action_feature_dim
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
            # Frozen copy of the pretrained action head. The SAC actor is
            # anchored to this, not to the replay buffer: the buffer is a
            # sliding window refilled by the changing policy, so an anchor to
            # it drifts along with the policy. Constructed here, before any
            # training, so it holds the checkpoint's weights; it lives outside
            # the trainable prefixes and the weight syncer never touches it.
            self.reference_action_model = copy.deepcopy(starvla_model.action_model)
            self.reference_action_model.requires_grad_(False)
            self.reference_action_model.eval()
        # EXPO-FT-style residual actor (Dong et al., arXiv:2605.25477). Needs
        # the twin Q heads to pick among candidates in sac_best_of_n, so it
        # requires add_q_head too.
        self.expo_num_candidates = int(expo_num_candidates)
        self.edit_policy: Optional[nn.Module] = None
        if add_edit_policy:
            if not add_q_head:
                raise ValueError(
                    "actor.model.add_edit_policy=true requires add_q_head=true"
                )
            self.edit_policy = StarVLAEditPolicy(
                hidden_size=hidden_size,
                action_feature_dim=self.action_feature_dim,
                hidden_dims=[int(dim) for dim in edit_hidden_dims],
                beta=edit_beta,
            ).to(dtype=torch.float32)
        self.sac_task_description = (
            None if sac_task_description is None else str(sac_task_description).strip()
        )

        action_model = getattr(starvla_model, "action_model", None)
        action_param_dtype = (
            next(
                (p.dtype for p in action_model.parameters() if p.is_floating_point()),
                policy_param_dtype,
            )
            if isinstance(action_model, nn.Module)
            else policy_param_dtype
        )
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

    def _check_sac_task_description(self, env_obs: dict[str, Any]) -> None:
        """Fail fast when replay-time prompts would differ from rollout prompts.

        Replay storage keeps tensors only, so SAC actor/critic forwards rebuild
        the prompt from ``sac_task_description``. If that string does not match
        the instruction the environment actually emitted, every Q value and
        actor gradient is computed under a prompt that is never deployed.
        """
        if self.sac_task_description is None or not isinstance(env_obs, dict):
            return
        observed = env_obs.get("task_descriptions")
        if observed is None:
            return
        mismatched = sorted(
            {str(text) for text in observed if str(text) != self.sac_task_description}
        )
        if mismatched:
            raise ValueError(
                "actor.model.sac_task_description does not match the environment "
                f"instruction. configured={self.sac_task_description!r}, "
                f"observed={mismatched[:3]!r}. Replayed SAC forwards would use a "
                "prompt the rollout policy never sees."
            )

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
        if forward_type == ForwardType.SAC_EDIT:
            return self.sac_edit_forward(**kwargs)
        if forward_type == ForwardType.SAC_BEST_OF_N:
            return self.sac_best_of_n(**kwargs)
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

        if self.resize_env_images:
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
        mean_actions, last_hidden, dist, action_queries = _run_oft_backbone_and_head(
            self,
            model_inputs=model_inputs,
            use_cache=False,
        )
        return mean_actions, last_hidden, dist, model_inputs, action_queries

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Sample differentiable environment-space action chunks for SAC.

        The fourth element carries the policy's normalized mean action and the
        frozen pretrained head's mean on the same features, so the actor can be
        anchored to, and its drift measured against, the checkpoint itself.
        """
        del kwargs
        mean_actions, last_hidden, dist, model_inputs, action_queries = (
            self._run_sac_oft_policy(obs)
        )
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
        extras = {"mean_actions": mean_actions}
        reference = getattr(self, "reference_action_model", None)
        if reference is not None:
            reference.eval()  # root .train() calls recurse into it
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                ref_dtype = next(
                    (p.dtype for p in reference.parameters() if p.is_floating_point()),
                    action_queries.dtype,
                )
                ref_mean = reference(action_queries.detach().to(dtype=ref_dtype))
            extras["reference_mean_actions"] = ref_mean[
                :, : self.num_executed_action_chunks
            ].to(mean_actions.dtype)
        return env_actions, logprobs.to(torch.float32), features, extras

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
            _, last_hidden, _, model_inputs, _ = self._run_sac_oft_policy(obs)
            shared_feature = self._pool_sac_features(
                last_hidden, model_inputs.get("attention_mask")
            )
        if detach_encoder:
            shared_feature = shared_feature.detach()
        # The critic sees actions in the policy's own [-1, 1] space rather than
        # environment units, so no channel's offset or scale dominates the
        # fusion layer. The map is affine, so gradients to the actor survive.
        # Callers disagree on layout: the actor passes [B, chunks, action_dim]
        # straight from the policy while the replay buffer stores the chunk
        # already flattened, so restore the per-channel axis before applying
        # per-channel statistics.
        if self._action_norm_stats is not None:
            actions = action_space_utils.normalize_actions_from_env_torch(
                actions.reshape(actions.shape[0], -1, self.action_dim),
                self._action_norm_stats,
                policy_setup=self.policy_setup,
            )
        actions = actions.reshape(actions.shape[0], -1)
        q_dtype = next(self.q_head.parameters()).dtype
        return self.q_head(shared_feature.to(dtype=q_dtype), actions.to(dtype=q_dtype))

    def sac_edit_forward(
        self,
        obs: dict[str, Any],
        base_actions: torch.Tensor,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """EXPO-FT actor step: edit an already-executed action, base frozen.

        ``base_actions`` is the replay-stored executed action, in environment
        units, in either layout ([B, chunks, dim] or the buffer's flattened
        [B, chunks*dim]). Returns
        ``(edited_env_actions, log_prob, pooled_features, extras)``; ``log_prob``
        is zero in eval mode and ``extras["edit"]`` is the raw normalized-space
        correction, for diagnostics.
        """
        del kwargs
        if self.edit_policy is None:
            raise RuntimeError(
                "sac_edit_forward requires actor.model.add_edit_policy=true"
            )
        _, last_hidden, _, model_inputs, _ = self._run_sac_oft_policy(obs)
        pooled_features = self._pool_sac_features(
            last_hidden, model_inputs.get("attention_mask")
        )
        batch_size = base_actions.shape[0]
        base_norm_flat = action_space_utils.normalize_actions_from_env_torch(
            base_actions.reshape(batch_size, -1, self.action_dim),
            self._action_norm_stats,
            policy_setup=self.policy_setup,
        ).reshape(batch_size, self.action_feature_dim)
        edit, log_prob = self.edit_policy(pooled_features, base_norm_flat, mode=mode)
        edited_norm = (base_norm_flat + edit).reshape(
            batch_size, self.num_executed_action_chunks, self.action_dim
        )
        edited_env_actions = action_space_utils.unnormalize_actions_for_env_torch(
            edited_norm, self._action_norm_stats, policy_setup=self.policy_setup
        )
        if log_prob is None:
            log_prob = torch.zeros(
                (batch_size, 1), dtype=torch.float32, device=base_actions.device
            )
        extras = {"edit": edit.detach()}
        return edited_env_actions, log_prob.to(torch.float32), pooled_features, extras

    def sac_best_of_n(
        self,
        obs: dict[str, Any],
        num_candidates: Optional[int] = None,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample N base actions, edit each, execute the highest-Q candidate.

        The base head that produces the raw candidates is frozen throughout
        an EXPO-FT recipe, so resampling it N times never drifts the
        pretrained policy; training can only change which candidate gets
        picked (by shaping Q) and how the edit head reshapes each one. Used
        both to act during rollout and to pick the bootstrap action at the
        next state in the critic's TD target. In eval mode this collapses to
        exactly two deterministic candidates -- the pretrained mean and its
        deterministically edited version -- so official evaluation stays
        reproducible per seed.
        """
        del kwargs
        if self.edit_policy is None or self.q_head is None:
            raise RuntimeError(
                "sac_best_of_n requires actor.model.add_edit_policy=true and "
                "add_q_head=true"
            )
        mean_actions, last_hidden, dist, model_inputs, _ = self._run_sac_oft_policy(
            obs
        )
        return self._select_best_of_n(
            mean_actions,
            last_hidden,
            dist,
            model_inputs,
            num_candidates=num_candidates,
            mode=mode,
        )

    def _select_best_of_n(
        self,
        mean_actions: torch.Tensor,
        last_hidden: torch.Tensor,
        dist: torch.distributions.Normal,
        model_inputs: dict[str, Any],
        num_candidates: Optional[int] = None,
        mode: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Shared best-of-N core, given an already-computed backbone pass.

        Split out of ``sac_best_of_n`` so callers that already ran the
        backbone this step (the OFT rollout handler) can select without
        paying for a second forward pass; ``sac_best_of_n`` itself runs the
        backbone fresh for callers (the SAC critic's next-state term) that
        do not already have one.

        Returns ``(selected_env_actions, selected_norm_actions, pooled_features,
        metrics)``. Both action tensors name the SAME selected candidate, in
        environment units and in the policy's own normalized units
        respectively: the critic's next-state term needs environment units
        (sac_q_forward renormalizes internally), while the rollout handler
        needs normalized units (predict_action_batch unnormalizes the value
        it is given exactly once). An earlier version returned only the
        environment-unit action and the rollout handler stored it straight
        into the "normalized_actions" field, so predict_action_batch
        unnormalized an already-unnormalized action and every episode
        executed a doubly-transformed action -- 0/96 grasped in the smoke
        test that caught it.
        """
        pooled_features = self._pool_sac_features(
            last_hidden, model_inputs.get("attention_mask")
        )
        batch_size = mean_actions.shape[0]
        flat_dim = self.action_feature_dim

        if mode == "eval":
            base_candidates = mean_actions.reshape(1, batch_size, flat_dim)
        else:
            n = int(num_candidates or self.expo_num_candidates)
            # The candidates are never trained: this base head has no RL
            # gradient path (it is excluded from actor.trainable_parameter_prefixes),
            # so nothing upstream of these samples needs a graph. Detaching
            # is a defensive no-op that keeps it that way even if that
            # changes.
            base_candidates = (
                dist.rsample((n,)).detach().reshape(n, batch_size, flat_dim)
            )
        num_base = base_candidates.shape[0]

        features_rep = pooled_features.repeat(num_base, 1)
        base_flat = base_candidates.reshape(num_base * batch_size, flat_dim)
        edits, _ = self.edit_policy(features_rep, base_flat, mode=mode)
        edited_flat = base_flat + edits

        all_norm_flat = torch.cat([base_flat, edited_flat], dim=0)
        all_norm_chunked = all_norm_flat.reshape(
            -1, self.num_executed_action_chunks, self.action_dim
        )
        all_env_actions = action_space_utils.unnormalize_actions_for_env_torch(
            all_norm_chunked, self._action_norm_stats, policy_setup=self.policy_setup
        )
        num_groups = all_env_actions.shape[0] // batch_size
        features_all = pooled_features.repeat(num_groups, 1)
        all_q_values = self.sac_q_forward(
            obs=None,
            actions=all_env_actions,
            shared_feature=features_all,
            detach_encoder=True,
        )
        q_agg = all_q_values.float().min(dim=-1).values.reshape(num_groups, batch_size)
        best_group = torch.argmax(q_agg, dim=0)
        grouped_env_actions = all_env_actions.reshape(
            num_groups, batch_size, self.num_executed_action_chunks, self.action_dim
        )
        grouped_norm_actions = all_norm_chunked.reshape(
            num_groups, batch_size, self.num_executed_action_chunks, self.action_dim
        )
        batch_index = torch.arange(batch_size, device=best_group.device)
        selected_env_actions = grouped_env_actions[best_group, batch_index]
        selected_norm_actions = grouped_norm_actions[best_group, batch_index]

        metrics = {
            "frac_selected_is_edited": (best_group >= num_base).float().mean().item(),
        }
        return selected_env_actions, selected_norm_actions, pooled_features, metrics

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
        self._check_sac_task_description(env_obs)

        if self.resize_env_images:
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
