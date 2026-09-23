# Copyright 2025 The RLinf Authors.
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


import os
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import SupportedModel
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import TRANSITION_VALID_KEY, Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def unwrap_module(model):
    """Return the innermost module behind FSDP/DDP wrappers."""
    seen = 0
    while hasattr(model, "module") and seen < 8:
        model = model.module
        seen += 1
    return model


def pop_transition_valid(batch: dict) -> Optional[torch.Tensor]:
    """Remove the replay validity mask from a sampled batch and return it.

    The mask travels inside ``curr_obs`` so that the replay buffer can store it
    like any other observation tensor. It must not reach the policy forward
    (observation dicts are converted to model inputs key by key).

    Returns:
        A ``[B, 1]`` float tensor, or ``None`` when the batch carries no mask.
    """
    valid = None
    for obs_key in ("curr_obs", "next_obs"):
        obs = batch.get(obs_key)
        if isinstance(obs, dict) and TRANSITION_VALID_KEY in obs:
            popped = obs.pop(TRANSITION_VALID_KEY)
            if valid is None:
                valid = popped
    if valid is None:
        return None
    return valid.reshape(valid.shape[0], -1)[:, :1].to(torch.float32)


def masked_transition_mean(
    values: torch.Tensor, valid: Optional[torch.Tensor]
) -> torch.Tensor:
    """Average ``values`` over valid transitions only.

    ``values`` is ``[B, ...]``; ``valid`` is ``[B, 1]`` (or ``None`` for all
    valid). Invalid transitions (steps after an episode already terminated)
    receive zero weight so they neither shape the critic target fit nor the
    actor objective. An all-invalid micro-batch yields a zero loss that still
    carries a graph, so gradient accumulation stays well defined.
    """
    if valid is None:
        return values.mean()
    valid = valid.to(device=values.device, dtype=values.dtype)
    weight = valid.reshape(valid.shape[0], *([1] * (values.dim() - 1))).expand_as(
        values
    )
    denominator = weight.sum()
    if denominator <= 0:
        return (values * weight).sum() * 0.0
    return (values * weight).sum() / denominator


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if use_dsrl:
            # DSRL: separate actor/critic encoders into different optimizer groups
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
            }
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        # SAC alpha
        # Initialize temperature parameter for automatic entropy tuning
        alpha_type = self.cfg.algorithm.entropy_tuning.get(
            "alpha_type", "softplus"
        )  # supported type: ["softplus","exp","fixed_alpha"]
        self.entropy_temp = EntropyTemperature(
            initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if alpha_type != "fixed_alpha":
            # log_pi is summed over every executed action dimension, so the
            # default target must cover the whole executed chunk rather than a
            # single action vector.
            executed_action_dims = int(self.cfg.actor.model.action_dim)
            if self.cfg.actor.model.model_type == "starvla":
                executed_action_dims *= int(
                    self.cfg.actor.model.get(
                        "num_executed_action_chunks",
                        self.cfg.actor.model.get("num_action_chunks", 1),
                    )
                )
            self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                "target_entropy",
                -executed_action_dims,
            )

            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=self.cfg.algorithm.entropy_tuning.optim.lr,
            )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        # Minimum batch-wise standard deviation of Q before the actor is
        # allowed to train. 0 disables the gate and reproduces the old
        # behaviour.
        self.min_q_std_for_actor = float(
            self.cfg.algorithm.get("min_q_std_for_actor", 0.0)
        )
        # Behaviour anchor (TD3+BC). Off-policy actors drift onto actions the
        # critic has never seen and overestimates; the 2026-09-19 run showed
        # exactly that, with q_pi climbing 0.11 -> 0.27 while real success fell
        # to 0.06. Pulling the policy toward the actions actually taken bounds
        # that drift. 0 disables the term.
        self.bc_coef = float(self.cfg.algorithm.get("bc_coef", 0.0))
        self.action_sigma = float(
            np.exp(self.cfg.actor.model.get("initial_logstd", -2.5))
        )
        # Named channel groups for the drift breakdown, {name: [start, end)}
        # over the per-step action vector. Optional; the overall drift is
        # always reported.
        groups = self.cfg.algorithm.get("policy_drift_groups", None) or {}
        self.policy_drift_groups = {
            str(name): (int(bounds[0]), int(bounds[1])) for name, bounds in groups.items()
        }
        # Per-channel anchor weight, built from policy_drift_groups plus
        # algorithm.policy_drift_group_weights ({name: multiplier}, default 1
        # for any channel outside a listed group or any unlisted group). Runs
        # 9-11 all showed damage concentrated in specific channels (arms and
        # waist drifted 2-4x more than hands, and that is exactly where the
        # task fails: grasp, driven by the hands, survives while placement,
        # driven by the arms and waist, does not) while a single scalar
        # bc_coef anchors every channel equally. Left at 1.0 everywhere this
        # is exactly the old uniform anchor.
        self.action_channel_weight = None
        group_weights = self.cfg.algorithm.get("policy_drift_group_weights", None)
        if group_weights:
            action_dim = int(self.cfg.actor.model.action_dim)
            weight = torch.ones(action_dim, dtype=torch.float32, device=self.device)
            for name, multiplier in group_weights.items():
                if name not in self.policy_drift_groups:
                    raise ValueError(
                        f"policy_drift_group_weights names {name!r}, which is "
                        "not one of policy_drift_groups "
                        f"{sorted(self.policy_drift_groups)}"
                    )
                lo, hi = self.policy_drift_groups[name]
                weight[lo:hi] = float(multiplier)
            self.action_channel_weight = weight
        # Critic action-discrimination probe. For each state the critic is
        # evaluated on `q_action_probe_samples` actions drawn around the
        # policy mean at `q_action_probe_sigma_scale` exploration sigmas; the
        # spread over those actions (q_action_std) is what the actor gradient
        # can actually use, unlike the spread over states (q_data_std), which
        # a critic that only learned V(s) also has. 0 samples disables it.
        self.q_action_probe_samples = int(
            self.cfg.algorithm.get("q_action_probe_samples", 0)
        )
        self.q_action_probe_sigma_scale = float(
            self.cfg.algorithm.get("q_action_probe_sigma_scale", 1.0)
        )
        # Optional second gate: q_action_std / q_data_std must reach this
        # before the actor trains. 0 leaves it as a diagnostic.
        self.min_q_action_ratio_for_actor = float(
            self.cfg.algorithm.get("min_q_action_ratio_for_actor", 0.0)
        )
        # Discount rewards inside the executed chunk (sum_i gamma^i r_i) so the
        # chunk return matches the gamma^chunk bootstrap instead of a plain sum.
        self.discount_within_chunk = bool(
            self.cfg.algorithm.get("discount_within_chunk", False)
        )
        self._action_norm_stats = getattr(
            unwrap_module(self.model), "_action_norm_stats", None
        )
        self._policy_setup = getattr(unwrap_module(self.model), "policy_setup", None)
        if self.q_action_probe_samples > 0 and self._action_norm_stats is None:
            raise ValueError(
                "algorithm.q_action_probe_samples needs the policy's action "
                "statistics, which this model does not expose"
            )
        # EXPO-FT-style recipe (arXiv:2605.25477): the pretrained OFT head is
        # frozen (excluded from actor.trainable_parameter_prefixes) and only
        # a small residual edit policy plus the twin Q heads are trained.
        # Detected from the model rather than a separate config flag so the
        # actor and the model's own construction can never disagree about
        # which recipe is active.
        self.expo_ft_enabled = getattr(unwrap_module(self.model), "edit_policy", None) is not None
        self.expo_num_candidates = int(
            self.cfg.algorithm.get(
                "expo_num_candidates",
                getattr(unwrap_module(self.model), "expo_num_candidates", 8),
            )
        )

        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        for name, param in self.target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if not online_param.requires_grad:
                            # Frozen (e.g. VLM backbone) parameters never
                            # diverge from the target copy; skip the copy.
                            continue
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32[name1]
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32[name1]
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    def _policy_forward(self, obs, **kwargs):
        """Run the SAC policy; models may return a fourth ``extras`` dict.

        StarVLA returns the policy's normalized mean action and the frozen
        pretrained head's mean on the same features there; other policies
        return the classic three-tuple.
        """
        outputs = self.model(forward_type=ForwardType.SAC, obs=obs, **kwargs)
        if len(outputs) == 4:
            return outputs
        actions, log_pi, shared_feature = outputs
        return actions, log_pi, shared_feature, {}

    def _probe_action_discrimination(self, shared_feature, extras, actions):
        """Measure how much Q moves with the action at a fixed state.

        Evaluates the critic on ``q_action_probe_samples`` perturbations of the
        policy mean, plus the mean itself, the frozen pretrained mean and the
        replay action, all on the same pooled features. Returns per-batch
        scalar metrics; the caller runs it under ``no_grad``.
        """
        from rlinf.models.embodiment.starvla.utils import action_space as asu

        mean = extras["mean_actions"].float()  # [B, chunks, dim], normalized
        ref = extras.get("reference_mean_actions")
        batch_size = mean.shape[0]
        k = self.q_action_probe_samples
        noise = torch.randn(
            (k, *mean.shape), device=mean.device, dtype=mean.dtype
        ) * (self.q_action_probe_sigma_scale * self.action_sigma)
        probes = [mean.unsqueeze(0) + noise, mean.unsqueeze(0)]
        if ref is not None:
            probes.append(ref.float().unsqueeze(0))
        stacked = torch.cat(probes, dim=0)  # [k + 1 (+1), B, chunks, dim]
        num_rows = stacked.shape[0]
        env_actions = asu.unnormalize_actions_for_env_torch(
            stacked.reshape(num_rows * batch_size, *mean.shape[1:]),
            self._action_norm_stats,
            policy_setup=self._policy_setup,
        )
        data_actions = actions.float().reshape(batch_size, -1)
        env_actions = torch.cat(
            [env_actions.reshape(num_rows * batch_size, -1), data_actions], dim=0
        )
        features = torch.cat(
            [shared_feature.repeat(num_rows, 1), shared_feature], dim=0
        )
        q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=None,
            actions=env_actions,
            shared_feature=features,
            detach_encoder=True,
        )
        q = q.float().min(dim=-1).values  # same aggregation the actor sees
        q_probe = q[: k * batch_size].reshape(k, batch_size)
        q_mean = q[k * batch_size : (k + 1) * batch_size]
        q_data = q[-batch_size:]
        metrics = {
            "q_action_std": q_probe.std(dim=0).mean().item(),
            "q_action_range": (q_probe.max(dim=0).values - q_probe.min(dim=0).values)
            .mean()
            .item(),
            "q_mean_action": q_mean.mean().item(),
            "q_mean_minus_data": (q_mean - q_data).mean().item(),
        }
        if ref is not None:
            q_ref = q[(k + 1) * batch_size : (k + 2) * batch_size]
            metrics["q_ref_action"] = q_ref.mean().item()
            metrics["q_mean_minus_ref"] = (q_mean - q_ref).mean().item()
        return metrics

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        else:
            if self.cfg.actor.model.model_type == "starvla":
                executed_chunks = self.cfg.actor.model.get(
                    "num_executed_action_chunks",
                    self.cfg.actor.model.get("num_action_chunks", 1),
                )
                discount = self.cfg.algorithm.gamma ** int(executed_chunks)
            else:
                discount = self.cfg.algorithm.gamma
            step_rewards = batch["rewards"]
            if self.discount_within_chunk and step_rewards.ndim == 2:
                # sum_i gamma^i r_i, consistent with the gamma^chunk bootstrap.
                weights = self.cfg.algorithm.gamma ** torch.arange(
                    step_rewards.shape[-1],
                    device=step_rewards.device,
                    dtype=torch.float32,
                )
                step_rewards = step_rewards.float() * weights
            rewards_for_bootstrap = step_rewards.sum(dim=-1, keepdim=True).to(
                self.torch_dtype
            )
        terminations = batch["terminations"].to(self.torch_dtype)
        is_starvla = self.cfg.actor.model.model_type == "starvla"

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            if self.expo_ft_enabled and bool(
                unwrap_module(self.model).critic_gate_open.item()
            ):
                # EXPO-FT: the bootstrap action at the next state is whichever
                # of the (frozen) base head's resampled candidates, or their
                # edited versions, Q currently prefers -- never a plain
                # sample from the base's own Gaussian, and never something a
                # gradient step could have dragged off distribution. Gated on
                # the same critic_gate_open buffer rollout uses, so the
                # bootstrap target does not start trusting the critic's own
                # ranking before rollout does -- an uninformative critic
                # would otherwise bootstrap off its own noise.
                # The TD target's next-state Q call renormalizes internally
                # (sac_q_forward), so it needs environment units here, not
                # the normalized-unit selection the rollout handler uses.
                next_state_actions, _, shared_feature, _ = self.model(
                    forward_type=ForwardType.SAC_BEST_OF_N,
                    obs=next_obs,
                    num_candidates=self.expo_num_candidates,
                    mode="train",
                )
                next_state_log_pi = torch.zeros(
                    (next_state_actions.shape[0], 1),
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            else:
                next_state_actions, next_state_log_pi, shared_feature, _ = (
                    self._policy_forward(next_obs, **kwargs)
                )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=(
                        shared_feature
                        if self.cfg.actor.model.model_type == "starvla"
                        else None
                    ),
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        probe_metrics = {}
        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            curr_feature = None
            curr_extras = {}
            if is_starvla and self.q_action_probe_samples > 0:
                # One frozen-backbone pass on the current observation serves
                # both the action probe and the data-Q input. The probe runs
                # before the gradient-carrying Q forward so every no-grad
                # forward precedes it, the same order the rest of this
                # function already relies on under FSDP.
                with torch.no_grad():
                    _, _, curr_feature, curr_extras = self._policy_forward(
                        curr_obs, **kwargs
                    )
                    if "mean_actions" in curr_extras:
                        probe_metrics = self._probe_action_discrimination(
                            curr_feature, curr_extras, actions
                        )
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                shared_feature=curr_feature,
                # StarVLA reuses the frozen VLM representation as critic
                # state.  The critic optimizer owns only q_head parameters;
                # detaching here prevents critic gradients from leaking into
                # the action head and contaminating global gradient clipping.
                detach_encoder=is_starvla,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        transition_valid = batch.get(TRANSITION_VALID_KEY)
        squared_error = (
            all_data_q_values - target_q_values.expand_as(all_data_q_values)
        ).square()
        critic_loss = masked_transition_mean(squared_error, transition_valid)
        # Spread of Q across the batch. dQ/da can only steer the actor when the
        # critic actually discriminates between states and actions; a critic
        # that has collapsed to a constant (q_data ~ 0.011 for every sample in
        # the sparse-reward run) hands the actor pure noise.
        metrics = {
            "q_data": all_data_q_values.mean().item(),
            "q_data_std": all_data_q_values.float().std(dim=0).mean().item()
            if all_data_q_values.shape[0] > 1
            else 0.0,
            **probe_metrics,
        }
        if transition_valid is not None:
            metrics["transition_valid_fraction"] = transition_valid.mean().item()
        return critic_loss, metrics

    def _forward_actor_expo_ft(self, batch):
        """EXPO-FT actor step: train only the residual edit policy.

        The pretrained OFT head never receives a gradient in this recipe (it
        is excluded from actor.trainable_parameter_prefixes), so unlike the
        anchor-based path below there is no drift to bound or measure -- the
        base action literally cannot move.
        """
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        edited_actions, edit_log_pi, shared_feature, extras = self.model(
            forward_type=ForwardType.SAC_EDIT,
            obs=curr_obs,
            base_actions=batch["actions"],
            mode="train",
        )
        all_qf_pi = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=edited_actions,
            shared_feature=shared_feature,
            detach_encoder=True,
        )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        transition_valid = batch.get(TRANSITION_VALID_KEY)
        objective = (self.entropy_temp.alpha * edit_log_pi) - qf_pi
        actor_loss = masked_transition_mean(objective, transition_valid)
        entropy = -edit_log_pi.mean()
        metrics["edit_magnitude"] = extras["edit"].abs().mean().item()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        if self.expo_ft_enabled:
            return self._forward_actor_expo_ft(batch)
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        pi, log_pi, shared_feature, extras = self._policy_forward(
            curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=(
                    shared_feature
                    if self.cfg.actor.model.model_type == "starvla"
                    else None
                ),
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        transition_valid = batch.get(TRANSITION_VALID_KEY)
        objective = (self.entropy_temp.alpha * log_pi) - qf_pi
        if self.bc_coef > 0:
            # TD3+BC anchor to the frozen pretrained head, mean to mean, in
            # the policy's own normalized units. Anchoring to the replay
            # action instead (attempts 5-8) anchors to a sliding window that
            # the policy itself refills, so the leash moved with the policy and
            # its length, measured between two noisy samples, was unreadable.
            # This distance has no sampling noise in it: zero means the policy
            # is exactly the checkpoint.
            if "reference_mean_actions" not in extras:
                raise ValueError(
                    "algorithm.bc_coef > 0 needs the policy's reference mean "
                    "action, which this model does not expose"
                )
            mean = extras["mean_actions"].float()
            reference = extras["reference_mean_actions"].float()
            delta = mean - reference  # [B, chunks, dim]
            if self.action_channel_weight is not None:
                bc_term = (
                    self.action_channel_weight.to(delta.dtype) * delta.square()
                ).mean(dim=(-1, -2))
            else:
                bc_term = delta.square().mean(dim=(-1, -2))
            # TD3+BC scaling keeps the trade-off independent of the Q scale,
            # which grows over training.
            scale = self.bc_coef / (qf_pi.detach().abs().mean() + 1e-6)
            objective = scale * objective + bc_term.reshape(-1, 1)
            metrics["bc_distance"] = bc_term.mean().item()
            metrics["bc_scale"] = scale.item()
            with torch.no_grad():
                sq = delta.square()
                sigma = self.action_sigma
                # RMS drift in exploration sigmas: overall, per channel group,
                # and for the first and last executed chunk.
                metrics["policy_drift_sigma"] = sq.mean().sqrt().item() / sigma
                for name, (lo, hi) in self.policy_drift_groups.items():
                    metrics[f"policy_drift_sigma/{name}"] = (
                        sq[..., lo:hi].mean().sqrt().item() / sigma
                    )
                metrics["policy_drift_sigma/chunk_first"] = (
                    sq[:, 0].mean().sqrt().item() / sigma
                )
                metrics["policy_drift_sigma/chunk_last"] = (
                    sq[:, -1].mean().sqrt().item() / sigma
                )
                metrics["policy_drift_sigma/max_channel"] = (
                    sq.mean(dim=(0, 1)).max().sqrt().item() / sigma
                )
        actor_loss = masked_transition_mean(objective, transition_valid)

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            if self.expo_ft_enabled:
                _, log_pi, _, _ = self.model(
                    forward_type=ForwardType.SAC_EDIT,
                    obs=curr_obs,
                    base_actions=batch["actions"],
                    mode="train",
                )
            else:
                kwargs = {}
                if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                    kwargs["temperature"] = (
                        self.cfg.algorithm.sampling_params.temperature_train
                    )
                if self.use_dsrl:
                    kwargs["train"] = True
                _, log_pi, _, _ = self._policy_forward(curr_obs, **kwargs)
                if log_pi.ndim == 1:
                    log_pi = log_pi.unsqueeze(-1)
                log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            transition_valid = pop_transition_valid(batch)
            if transition_valid is not None:
                batch[TRANSITION_VALID_KEY] = transition_valid
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        # Optimizer.step() deliberately leaves gradients populated. Clear both
        # parameter sets at the start of every critic pass so the FSDP-wide
        # norm below measures only this critic update.
        self.optimizer.zero_grad()
        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        # The actor pass runs FSDP collectives, so whether it runs has to be
        # decided identically on every rank. Both inputs to that decision are
        # rank-local (each rank measures Q on its own micro-batches and owns
        # its own replay buffer), so reduce them before branching: a rank that
        # took the branch alone would hang the others in an all-reduce.
        q_data_std = float(all_critic_metrics.get("critic/q_data_std", 0.0))
        q_action_std = float(all_critic_metrics.get("critic/q_action_std", 0.0))
        decision = torch.tensor(
            [q_data_std, float(train_actor), q_action_std],
            dtype=torch.float32,
            device=self.device,
        )
        if torch.distributed.is_initialized() and self._world_size > 1:
            torch.distributed.all_reduce(decision, op=torch.distributed.ReduceOp.AVG)
        q_data_std = float(decision[0])
        q_action_std = float(decision[2])
        # AVG of the per-rank booleans is 1.0 only when every rank agrees.
        train_actor = bool(decision[1] >= 1.0)
        # Spread over actions at a fixed state, relative to the spread over
        # states. A critic that only learned V(s) scores ~0 here while passing
        # the q_data_std gate.
        q_action_ratio = q_action_std / (q_data_std + 1e-8)
        critic_informative = q_data_std >= self.min_q_std_for_actor and (
            q_action_ratio >= self.min_q_action_ratio_for_actor
        )
        metrics_data["sac/q_data_std"] = q_data_std
        metrics_data["sac/q_action_std"] = q_action_std
        metrics_data["sac/q_action_ratio"] = q_action_ratio
        metrics_data["sac/critic_informative"] = float(critic_informative)
        if self.expo_ft_enabled:
            # Rollout (oft.py's run_rollout_oft) reads this same buffer to
            # decide whether best-of-N is safe yet; it is weight-synced to
            # rollout exactly like actor_logstd already is. Every rank sets
            # the identical, already all-reduced value.
            unwrap_module(self.model).critic_gate_open.fill_(critic_informative)
        train_actor = train_actor and critic_informative
        metrics_data["sac/actor_trained"] = float(
            train_actor and self.update_step % self.critic_actor_ratio == 0
        )

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            self.qf_optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, entropy, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
            # The actor needs dQ/da, but the Q parameters themselves are not
            # part of the actor optimizer. Backward still materializes their
            # gradients, so clear those before the FSDP-wide actor norm/clip.
            self.qf_optimizer.zero_grad()
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch in train_micro_batch_list:
                    alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        # Delay actor training until buffer has enough samples
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # FSDP local-state checkpoints are pathological for StarVLA's partial
        # fine-tuning layout: the frozen, unwrapped VLM is replicated into each
        # rank file. The target model is another full copy, so a single SAC
        # checkpoint can exceed 150 GB. For StarVLA, save one rank-0 full state
        # (needed by deterministic evaluation) and reconstruct target Q / replay
        # state if a watchdog has to resume after a crash. The uninterrupted run
        # keeps optimizer, target, and replay state in memory as usual.
        lightweight_starvla = (
            self.cfg.actor.model.model_type == "starvla"
            and self.cfg.algorithm.get("lightweight_starvla_checkpoint", True)
        )
        if lightweight_starvla:
            model_save_path = os.path.join(save_base_path, "model_state_dict")
            os.makedirs(model_save_path, exist_ok=True)
            model_state_dict = self._strategy.get_model_state_dict(
                self.model, cpu_offload=True, full_state_dict=True
            )
            if self._rank == 0:
                torch.save(
                    model_state_dict,
                    os.path.join(model_save_path, "full_weights.pt"),
                )
                torch.save(
                    {"step": int(step), "update_step": int(self.update_step)},
                    os.path.join(save_base_path, "sac_resume_state.pt"),
                )
            del model_state_dict
            torch.distributed.barrier()
            return

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        lightweight_path = os.path.join(
            load_base_path, "model_state_dict", "full_weights.pt"
        )
        if (
            self.cfg.actor.model.model_type == "starvla"
            and os.path.isfile(lightweight_path)
            and not os.path.isdir(
                os.path.join(load_base_path, "local_shard_checkpoint")
            )
        ):
            model_state_dict = torch.load(
                lightweight_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            self._strategy.load_model_with_state_dict(
                self.model,
                model_state_dict,
                cpu_offload=True,
                full_state_dict=True,
            )
            del model_state_dict
            # A checkpoint from another run may carry a different fixed
            # exploration scale; the configured value must win.
            restore_fixed_logstd = getattr(
                self.model, "restore_configured_fixed_actor_logstd", None
            )
            if restore_fixed_logstd is not None:
                restore_fixed_logstd()
            resume_state_path = os.path.join(load_base_path, "sac_resume_state.pt")
            if os.path.isfile(resume_state_path):
                resume_state = torch.load(
                    resume_state_path, map_location="cpu", weights_only=True
                )
                self.update_step = int(resume_state.get("update_step", 0))
            # The lightweight emergency-resume format intentionally resets
            # Adam moments and replay contents. Start the target critic from the
            # restored online Q instead of loading eight redundant model copies.
            self.soft_update_target_model(tau=1.0)
            torch.distributed.barrier()
            return

        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
