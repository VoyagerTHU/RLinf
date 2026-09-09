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

import asyncio
import os
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from torch.utils._pytree import tree_map

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    adapt_kl_beta,
    compute_embodied_reference_kl,
    compute_update_level_explained_variance,
    kl_penalty,
    positive_advantage_sample_mask,
    prioritize_positive_advantage_samples,
    retain_positive_advantages,
    should_stop_ppo_update,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].to(Worker.torch_device_type)
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self.kl_beta = float(self.cfg.algorithm.get("kl_beta", 0.0))
        self.kl_penalty_type = str(
            self.cfg.algorithm.get(
                "kl_penalty_type",
                self.cfg.algorithm.get("kl_penalty", "low_var_kl"),
            )
        )
        self.combine_reference_model = bool(
            self.cfg.actor.get("combine_reference_model", True)
        )
        if self.kl_beta < 0:
            raise ValueError(f"algorithm.kl_beta must be non-negative, got {self.kl_beta}.")
        if self.kl_beta > 0 and not self.combine_reference_model:
            raise NotImplementedError(
                "EmbodiedFSDPActor currently requires actor.combine_reference_model=true "
                "when algorithm.kl_beta is positive."
            )
        self.ref_policy_state_dict = None
        self.offload_model_buffer = None
        # Reference-KL leash: ``reference_target_kl`` adapts ``kl_beta`` after
        # every update (PPO-penalty style) instead of rejecting optimizer steps.
        # A hard stop against an immutable reference can never recover once it
        # is exceeded, because only optimizer steps can move the policy back.
        self.reference_target_kl = self.cfg.algorithm.get("reference_target_kl", None)
        self.kl_beta_min = float(self.cfg.algorithm.get("kl_beta_min", 1e-4))
        self.kl_beta_max = float(self.cfg.algorithm.get("kl_beta_max", 1.0))
        self.kl_beta_adapt_factor = float(
            self.cfg.algorithm.get("kl_beta_adapt_factor", 2.0)
        )
        if self.reference_target_kl is not None:
            self.reference_target_kl = float(self.reference_target_kl)
            if self.reference_target_kl <= 0:
                raise ValueError(
                    "algorithm.reference_target_kl must be positive, got "
                    f"{self.reference_target_kl}."
                )
            if self.kl_beta <= 0:
                raise ValueError(
                    "algorithm.reference_target_kl adapts algorithm.kl_beta and "
                    "therefore requires a positive initial kl_beta."
                )
            # Validate the controller bounds once up front.
            adapt_kl_beta(
                self.kl_beta,
                self.reference_target_kl,
                self.reference_target_kl,
                factor=self.kl_beta_adapt_factor,
                min_beta=self.kl_beta_min,
                max_beta=self.kl_beta_max,
            )

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.kl_beta > 0:
            # Keep the model loaded from actor.model.model_path as an immutable
            # reference.  The runner restores resume checkpoints only after
            # init_worker(), so resumed training remains anchored to the same
            # original pretrained policy rather than to the latest checkpoint.
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

        self._setup_rollout_weight_dst_ranks()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        trainable_prefixes = self.cfg.actor.get("trainable_parameter_prefixes", None)
        if trainable_prefixes:
            if not self.cfg.actor.fsdp_config.get("use_orig_params", False):
                raise ValueError(
                    "actor.trainable_parameter_prefixes requires "
                    "actor.fsdp_config.use_orig_params=true so FSDP can mix "
                    "frozen and trainable parameters safely"
                )
            if isinstance(trainable_prefixes, str):
                trainable_prefixes = [trainable_prefixes]
            prefixes = tuple(
                str(prefix).strip().rstrip(".")
                for prefix in trainable_prefixes
                if str(prefix).strip()
            )
            if not prefixes:
                raise ValueError(
                    "actor.trainable_parameter_prefixes must contain a non-empty prefix"
                )

            total_parameters = sum(param.numel() for param in model.parameters())
            trainable_parameters = 0
            matched_names = []
            for name, param in model.named_parameters():
                is_trainable = any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in prefixes
                )
                param.requires_grad_(is_trainable)
                if is_trainable:
                    trainable_parameters += param.numel()
                    matched_names.append(name)
            if not matched_names:
                raise ValueError(
                    "No model parameters matched actor.trainable_parameter_prefixes="
                    f"{list(prefixes)}"
                )
            self._logger.info(
                "[FSDP] Parameter whitelist enabled: prefixes=%s, "
                "trainable=%d/%d (%.4f%%), tensors=%d",
                list(prefixes),
                trainable_parameters,
                total_parameters,
                100.0 * trainable_parameters / total_parameters,
                len(matched_names),
            )

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)
            restore_fixed_logstd = getattr(
                model, "restore_configured_fixed_actor_logstd", None
            )
            if restore_fixed_logstd is not None:
                restore_fixed_logstd()

        return model

    def load_checkpoint(self, load_path: str) -> None:
        """Load training state while honoring a configured fixed log standard deviation."""
        super().load_checkpoint(load_path)
        restore_fixed_logstd = getattr(
            self.model, "restore_configured_fixed_actor_logstd", None
        )
        if restore_fixed_logstd is not None:
            restore_fixed_logstd()

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            handle = []
            for rank in self._weight_dst_rank_in_rollout:
                handle.append(
                    self.send(
                        data,
                        dst_group_name=self._rollout_group_name,
                        dst_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            await asyncio.gather(*handle)

        async def recv_func():
            handle = []
            for rank in self._weight_dst_rank_in_rollout:
                handle.append(
                    self.recv(
                        src_group_name=self._rollout_group_name,
                        src_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            metadata_list = await asyncio.gather(*handle)
            metadata = metadata_list[0]
            for other_metadata in metadata_list[1:]:
                if other_metadata != metadata:
                    raise ValueError("Patch metadata differs across rollout ranks")
            return metadata

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
            )

        await self.weight_syncer.sync(state_dict, send_func, version=self.version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

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

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        if self.cfg.algorithm.get("positive_advantages_only", False):
            self.rollout_batch["advantages"] = retain_positive_advantages(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # NOTE: This must be set before importing openpi.training.data_loader
            if self.cfg.actor.get("sft_data_path", None):
                os.environ["HF_LEROBOT_HOME"] = self.cfg.actor.sft_data_path

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ):
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: x.to(self.device) if x is not None else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        sft_losses = self.model(
            data={"observation": observation, "actions": actions},
            forward_type=ForwardType.SFT,
        )
        # Ensure losses is a tensor and handle different return types
        if isinstance(sft_losses, list | tuple):
            sft_losses = torch.stack(sft_losses)
        elif not isinstance(sft_losses, torch.Tensor):
            sft_losses = torch.tensor(
                sft_losses, device=self.device, dtype=torch.float32
            )

        sft_loss = sft_losses.mean()
        metrics_data["sft_loss"] = sft_loss.clone().detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )

    @torch.no_grad()
    def _recompute_rollout_logprobs_on_actor(self) -> dict[str, float]:
        """Recompute old-policy log-probabilities with the training actor.

        Rollout and FSDP actor workers can produce slightly different action
        means even when they hold the same logical weights. Small exploration
        standard deviations amplify that numerical difference in PPO ratios.
        Replaying sampled actions once on the actor makes the stored old policy
        and subsequent update forwards use the same numerical path.
        """
        enabled = bool(
            self.cfg.algorithm.get("recompute_rollout_logprobs_on_actor", False)
        )
        metrics = {
            "actor/recomputed_rollout_logprobs": float(enabled),
            "actor/recompute_rollout_logprobs_seconds": 0.0,
            "actor/rollout_actor_logprob_mean_abs_delta": 0.0,
            "actor/rollout_actor_logprob_max_abs_delta": 0.0,
        }
        if not enabled:
            return metrics

        rollout_logprobs = self.rollout_batch["prev_logprobs"]
        rollout_size = rollout_logprobs.shape[0]
        micro_batch_size = int(
            self.cfg.algorithm.get(
                "actor_logprob_recompute_micro_batch_size",
                self.cfg.algorithm.get(
                    "reference_forward_micro_batch_size",
                    self.cfg.actor.micro_batch_size,
                ),
            )
        )
        if micro_batch_size <= 0:
            raise ValueError(
                "algorithm.actor_logprob_recompute_micro_batch_size must be "
                f"positive, got {micro_batch_size}."
            )
        if rollout_size % micro_batch_size != 0:
            raise ValueError(
                "Flattened rollout size must be divisible by the actor log-prob "
                f"recompute micro-batch size: {rollout_size} vs {micro_batch_size}."
            )

        start_time = time.perf_counter()
        replay_batches = split_dict_to_chunk(
            self.rollout_batch, rollout_size // micro_batch_size
        )
        actor_logprobs = []
        self.model.eval()
        for replay_batch in replay_batches:
            replay_batch = put_tensor_device(
                replay_batch,
                f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
            )
            with self.amp_context:
                output_dict = self.model(
                    forward_inputs=replay_batch.get("forward_inputs", None),
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                )
            actor_logprobs.append(
                output_dict["logprobs"].detach().to(device="cpu").contiguous()
            )
            del replay_batch, output_dict

        actor_logprobs = torch.cat(actor_logprobs, dim=0)
        rollout_logprobs = rollout_logprobs.detach().to(device="cpu")
        if actor_logprobs.shape != rollout_logprobs.shape:
            raise RuntimeError(
                "Actor-recomputed and rollout log-probability shapes differ: "
                f"{actor_logprobs.shape} vs {rollout_logprobs.shape}."
            )
        delta = actor_logprobs.float() - rollout_logprobs.float()
        if not torch.isfinite(delta).all():
            raise RuntimeError(
                "Non-finite actor-vs-rollout log-probability delta detected."
            )
        self.rollout_batch["prev_logprobs"] = actor_logprobs
        metrics.update(
            {
                "actor/recompute_rollout_logprobs_seconds": (
                    time.perf_counter() - start_time
                ),
                "actor/rollout_actor_logprob_mean_abs_delta": (
                    delta.abs().mean().item()
                ),
                "actor/rollout_actor_logprob_max_abs_delta": (delta.abs().max().item()),
            }
        )
        self.model.train()
        clear_memory(sync=False)
        return metrics

    @torch.no_grad()
    def _compute_post_update_logprob_metrics(self) -> dict[str, float]:
        """Measure the realized policy movement after optimizer updates.

        Pre-update PPO KL is necessarily zero for a single-minibatch update.
        Replaying the rollout after the optimizer step makes update magnitude
        observable and catches both ineffective sub-ULP updates and excessive
        policy movement.
        """
        enabled = bool(self.cfg.algorithm.get("audit_post_update_logprobs", False))
        metrics = {
            "actor/post_update_logprob_audit": float(enabled),
            "actor/post_update_logprob_audit_seconds": 0.0,
            "actor/post_update_logprob_finite_fraction": 1.0,
            "actor/post_update_ratio_finite_fraction": 1.0,
            "actor/post_update_logprob_mean_delta": 0.0,
            "actor/post_update_logprob_mean_abs_delta": 0.0,
            "actor/post_update_logprob_max_abs_delta": 0.0,
            "actor/post_update_ratio_mean_abs_delta": 0.0,
            "actor/post_update_ratio_max_abs_delta": 0.0,
            "actor/post_update_reference_logprob_mean_abs_delta": 0.0,
            "actor/post_update_reference_logprob_max_abs_delta": 0.0,
        }
        if not enabled:
            return metrics

        rollout_size = self.rollout_batch["prev_logprobs"].shape[0]
        micro_batch_size = int(
            self.cfg.algorithm.get(
                "post_update_logprob_audit_micro_batch_size",
                self.cfg.algorithm.get(
                    "actor_logprob_recompute_micro_batch_size",
                    self.cfg.actor.micro_batch_size,
                ),
            )
        )
        if micro_batch_size <= 0 or rollout_size % micro_batch_size != 0:
            raise ValueError(
                "The post-update log-probability audit micro-batch size must be "
                f"positive and divide {rollout_size}, got {micro_batch_size}."
            )

        start_time = time.perf_counter()
        replay_batches = split_dict_to_chunk(
            self.rollout_batch, rollout_size // micro_batch_size
        )
        device = Worker.torch_platform.current_device()
        sums = torch.zeros(8, dtype=torch.float64, device=device)
        maxima = torch.zeros(3, dtype=torch.float64, device=device)

        self.model.eval()
        for replay_batch in replay_batches:
            replay_batch = put_tensor_device(
                replay_batch,
                f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
            )
            with self.amp_context:
                output_dict = self.model(
                    forward_inputs=replay_batch.get("forward_inputs", None),
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                )

            current = output_dict["logprobs"].float()
            previous = replay_batch["prev_logprobs"].to(current).float()
            delta = (current - previous).reshape(current.shape[0], -1)
            loss_mask = replay_batch.get("loss_mask", None)
            if loss_mask is not None:
                flat_mask = loss_mask.bool().reshape(loss_mask.shape[0], -1)
                if flat_mask.shape[1] == 1:
                    flat_mask = flat_mask.expand_as(delta)
                elif flat_mask.shape != delta.shape:
                    raise RuntimeError(
                        "Post-update audit cannot broadcast loss_mask "
                        f"{flat_mask.shape} to logprobs {delta.shape}."
                    )
                delta = delta[flat_mask]
            else:
                delta = delta.reshape(-1)

            sums[0] += delta.numel()
            finite_delta = torch.isfinite(delta)
            sums[1] += finite_delta.sum()
            finite_values = delta[finite_delta].double()
            if finite_values.numel() > 0:
                abs_values = finite_values.abs()
                sums[2] += finite_values.sum()
                sums[3] += abs_values.sum()
                maxima[0] = torch.maximum(maxima[0], abs_values.max())
                ratio_delta = torch.expm1(finite_values)
                finite_ratio = torch.isfinite(ratio_delta)
                sums[5] += finite_ratio.sum()
                if finite_ratio.any():
                    abs_ratio_delta = ratio_delta[finite_ratio].abs()
                    sums[4] += abs_ratio_delta.sum()
                    maxima[1] = torch.maximum(maxima[1], abs_ratio_delta.max())

            reference = replay_batch.get("ref_logprobs", None)
            if reference is not None:
                reference_delta = (current - reference.to(current).float()).reshape(
                    current.shape[0], -1
                )
                if loss_mask is not None:
                    reference_delta = reference_delta[flat_mask]
                else:
                    reference_delta = reference_delta.reshape(-1)
                finite_reference = reference_delta[
                    torch.isfinite(reference_delta)
                ].double()
                sums[7] += finite_reference.numel()
                if finite_reference.numel() > 0:
                    abs_reference = finite_reference.abs()
                    sums[6] += abs_reference.sum()
                    maxima[2] = torch.maximum(maxima[2], abs_reference.max())

            del replay_batch, output_dict

        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
        total = max(sums[0].item(), 1.0)
        finite_count = max(sums[1].item(), 1.0)
        ratio_finite_count = max(sums[5].item(), 1.0)
        reference_count = max(sums[7].item(), 1.0)
        metrics.update(
            {
                "actor/post_update_logprob_audit_seconds": (
                    time.perf_counter() - start_time
                ),
                "actor/post_update_logprob_finite_fraction": (sums[1].item() / total),
                "actor/post_update_ratio_finite_fraction": (
                    sums[5].item() / finite_count
                ),
                "actor/post_update_logprob_mean_delta": (sums[2].item() / finite_count),
                "actor/post_update_logprob_mean_abs_delta": (
                    sums[3].item() / finite_count
                ),
                "actor/post_update_logprob_max_abs_delta": maxima[0].item(),
                "actor/post_update_ratio_mean_abs_delta": (
                    sums[4].item() / ratio_finite_count
                ),
                "actor/post_update_ratio_max_abs_delta": maxima[1].item(),
                "actor/post_update_reference_logprob_mean_abs_delta": (
                    sums[6].item() / reference_count
                ),
                "actor/post_update_reference_logprob_max_abs_delta": (maxima[2].item()),
            }
        )
        self.model.train()
        clear_memory(sync=False)
        return metrics

    def _actor_logstd_metrics(self) -> dict[str, float]:
        """Return the runtime exploration scale for checkpoint/config auditing."""
        actor_logstd = None
        for name, value in list(self.model.named_parameters()) + list(
            self.model.named_buffers()
        ):
            if name == "actor_logstd" or name.endswith(".actor_logstd"):
                actor_logstd = value
                break
        if actor_logstd is None:
            return {}
        values = actor_logstd.detach().float()
        return {
            "actor/runtime_logstd_min": values.min().item(),
            "actor/runtime_logstd_mean": values.mean().item(),
            "actor/runtime_logstd_max": values.max().item(),
        }

    @torch.no_grad()
    def _compute_reference_logprobs(self) -> float:
        """Replay the received rollout once under the immutable reference policy."""
        if self.kl_beta <= 0:
            return 0.0
        if self.ref_policy_state_dict is None:
            raise RuntimeError(
                "Reference policy weights are missing while algorithm.kl_beta is positive."
            )

        rollout_size = self.rollout_batch["prev_logprobs"].shape[0]
        reference_micro_batch_size = int(
            self.cfg.algorithm.get(
                "reference_forward_micro_batch_size", self.cfg.actor.micro_batch_size
            )
        )
        if reference_micro_batch_size <= 0:
            raise ValueError(
                "algorithm.reference_forward_micro_batch_size must be positive, "
                f"got {reference_micro_batch_size}."
            )
        if rollout_size % reference_micro_batch_size != 0:
            raise ValueError(
                "Flattened rollout size must be divisible by the reference forward "
                f"micro-batch size: {rollout_size} vs {reference_micro_batch_size}."
            )

        start_time = time.perf_counter()
        reference_batches = split_dict_to_chunk(
            self.rollout_batch, rollout_size // reference_micro_batch_size
        )
        reference_logprobs = []
        self.model.eval()
        with cpu_weight_swap(
            self.model,
            self.ref_policy_state_dict,
            self.offload_model_buffer,
        ):
            for reference_batch in reference_batches:
                reference_batch = put_tensor_device(
                    reference_batch,
                    f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                )
                with self.amp_context:
                    output_dict = self.model(
                        forward_inputs=reference_batch.get("forward_inputs", None),
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                    )
                reference_logprobs.append(
                    output_dict["logprobs"].detach().to(device="cpu").contiguous()
                )
                del reference_batch, output_dict

        self.rollout_batch["ref_logprobs"] = torch.cat(reference_logprobs, dim=0)
        self.model.train()
        clear_memory(sync=False)
        return time.perf_counter() - start_time

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)
        prioritize_positive_samples = bool(
            self.cfg.algorithm.get("prioritize_positive_advantage_samples", False)
        )
        if prioritize_positive_samples:
            if not self.cfg.algorithm.get("positive_advantages_only", False):
                raise ValueError(
                    "algorithm.prioritize_positive_advantage_samples requires "
                    "algorithm.positive_advantages_only=true."
                )
            shuffle_id = prioritize_positive_advantage_samples(
                self.rollout_batch["advantages"],
                shuffle_id,
                self.rollout_batch.get("loss_mask", None),
            )

        positive_chunk_samples_available = (
            positive_advantage_sample_mask(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )
            .sum()
            .item()
        )

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        recompute_logprob_metrics = self._recompute_rollout_logprobs_on_actor()
        reference_forward_seconds = self._compute_reference_logprobs()

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        target_kl = self.cfg.algorithm.get("target_kl", None)
        if target_kl is not None:
            target_kl = float(target_kl)
            # Validate before any optimizer work is performed.
            should_stop_ppo_update(0.0, target_kl)
        reference_target_kl = self.reference_target_kl
        max_optimizer_steps_per_update = self.cfg.algorithm.get(
            "max_optimizer_steps_per_update", None
        )
        if max_optimizer_steps_per_update is not None:
            max_optimizer_steps_per_update = int(max_optimizer_steps_per_update)
            if max_optimizer_steps_per_update <= 0:
                raise ValueError(
                    "algorithm.max_optimizer_steps_per_update must be positive "
                    "when configured."
                )
        optimizer_steps_this_update = 0
        early_stopped_on_kl = False
        stopped_on_optimizer_step_limit = False
        first_global_batch_kl = None
        last_global_batch_kl = 0.0
        max_global_batch_abs_kl = 0.0
        first_global_batch_reference_kl = None
        last_global_batch_reference_kl = 0.0
        max_global_batch_reference_kl = 0.0
        global_batch_reference_kl_values: list[float] = []
        # Proximal (old-policy vs. current) KL uses the non-negative low-variance
        # estimator aggregated exactly like the loss; the signed k1 mean stored
        # in ``actor/approx_kl`` can stay near zero while the policy moves.
        first_global_batch_proximal_kl = None
        last_global_batch_proximal_kl = 0.0
        max_global_batch_proximal_kl = 0.0
        positive_chunk_samples_this_update = 0
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )
                train_global_positive_chunk_samples = (
                    train_global_batch["advantages"]
                    .reshape(train_global_batch_size, -1)
                    .gt(0)
                    .any(dim=-1)
                    .sum()
                    .item()
                )

                self.optimizer.zero_grad()
                global_batch_kl_sum = torch.zeros(
                    (), device=Worker.torch_platform.current_device()
                )
                global_batch_abs_kl_sum = torch.zeros_like(global_batch_kl_sum)
                global_batch_reference_kl_sum = torch.zeros_like(global_batch_kl_sum)
                global_batch_proximal_kl_sum = torch.zeros_like(global_batch_kl_sum)
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                            **kwargs,
                        )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    reference_kl_loss = torch.tensor(
                        0.0, device=Worker.torch_platform.current_device()
                    )
                    # During a critic warmup the policy must not move at all,
                    # so the reference penalty is skipped together with the
                    # policy loss.
                    if self.kl_beta > 0 and not kwargs["critic_warmup"]:
                        ref_logprobs = batch.get("ref_logprobs", None)
                        if ref_logprobs is None:
                            raise RuntimeError(
                                "Reference log-probabilities were not attached to the "
                                "embodied rollout batch."
                            )
                        reference_kl_loss = compute_embodied_reference_kl(
                            output_dict["logprobs"],
                            ref_logprobs,
                            kl_penalty_type=self.kl_penalty_type,
                            logprob_type=self.cfg.algorithm.logprob_type,
                            single_action_dim=self.cfg.actor.model.get("action_dim", 7),
                            loss_mask=loss_mask,
                        )
                        loss = loss + self.kl_beta * reference_kl_loss
                    metrics_data["actor/reference_kl_loss"] = (
                        reference_kl_loss.detach().item()
                    )
                    with torch.no_grad():
                        proximal_kl = compute_embodied_reference_kl(
                            output_dict["logprobs"].detach(),
                            prev_logprobs.reshape_as(output_dict["logprobs"]).to(
                                output_dict["logprobs"]
                            ),
                            kl_penalty_type="low_var_kl",
                            logprob_type=self.cfg.algorithm.logprob_type,
                            single_action_dim=self.cfg.actor.model.get("action_dim", 7),
                            loss_mask=loss_mask,
                        )
                    metrics_data["actor/proximal_low_var_kl"] = proximal_kl.item()
                    global_batch_proximal_kl_sum += proximal_kl
                    micro_batch_kl = torch.as_tensor(
                        metrics_data["actor/approx_kl"],
                        device=global_batch_kl_sum.device,
                        dtype=torch.float32,
                    ).detach()
                    global_batch_kl_sum += micro_batch_kl
                    global_batch_abs_kl_sum += micro_batch_kl.abs()
                    global_batch_reference_kl_sum += reference_kl_loss.detach()

                    entropy_loss = torch.tensor(
                        0.0, device=Worker.torch_platform.current_device()
                    )
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

                    if self.enable_sft_co_train:
                        self._train_sft_epoch(metrics_data, loss)

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch, output_dict, forward_inputs, loss, metrics_data

                self.torch_platform.empty_cache()

                num_micro_batches = len(train_micro_batch)
                global_batch_kl = global_batch_kl_sum / num_micro_batches
                global_batch_abs_kl = global_batch_abs_kl_sum / num_micro_batches
                global_batch_reference_kl = (
                    global_batch_reference_kl_sum / num_micro_batches
                )
                global_batch_proximal_kl = (
                    global_batch_proximal_kl_sum / num_micro_batches
                )
                torch.distributed.all_reduce(
                    global_batch_kl, op=torch.distributed.ReduceOp.AVG
                )
                torch.distributed.all_reduce(
                    global_batch_abs_kl, op=torch.distributed.ReduceOp.AVG
                )
                torch.distributed.all_reduce(
                    global_batch_reference_kl, op=torch.distributed.ReduceOp.AVG
                )
                torch.distributed.all_reduce(
                    global_batch_proximal_kl, op=torch.distributed.ReduceOp.AVG
                )
                global_batch_kl_value = global_batch_kl.item()
                global_batch_abs_kl_value = global_batch_abs_kl.item()
                global_batch_reference_kl_value = global_batch_reference_kl.item()
                global_batch_proximal_kl_value = global_batch_proximal_kl.item()
                if first_global_batch_kl is None:
                    first_global_batch_kl = global_batch_kl_value
                if first_global_batch_reference_kl is None:
                    first_global_batch_reference_kl = global_batch_reference_kl_value
                if first_global_batch_proximal_kl is None:
                    first_global_batch_proximal_kl = global_batch_proximal_kl_value
                last_global_batch_kl = global_batch_kl_value
                last_global_batch_reference_kl = global_batch_reference_kl_value
                last_global_batch_proximal_kl = global_batch_proximal_kl_value
                max_global_batch_abs_kl = max(
                    max_global_batch_abs_kl, global_batch_abs_kl_value
                )
                max_global_batch_reference_kl = max(
                    max_global_batch_reference_kl,
                    global_batch_reference_kl_value,
                )
                max_global_batch_proximal_kl = max(
                    max_global_batch_proximal_kl, global_batch_proximal_kl_value
                )
                global_batch_reference_kl_values.append(global_batch_reference_kl_value)

                # Only the proximal trust region stops an update. The reference
                # KL is enforced through the adaptive penalty coefficient below;
                # a hard reference stop could never be recovered from.
                if should_stop_ppo_update(global_batch_proximal_kl_value, target_kl):
                    early_stopped_on_kl = True
                    self.optimizer.zero_grad()
                    self._logger.warning(
                        "Skipping the pending optimizer step and stopping remaining PPO "
                        "minibatches because the proximal KL boundary was exceeded: "
                        "low-var KL=%.6f (target=%s), signed k1=%.6f, reference "
                        "KL=%.6f (target=%s, kl_beta=%.3g).",
                        global_batch_proximal_kl_value,
                        target_kl,
                        global_batch_kl_value,
                        global_batch_reference_kl_value,
                        reference_target_kl,
                        self.kl_beta,
                    )
                    break

                grad_norm, lr_list = self.optimizer_step()
                optimizer_steps_this_update += 1
                positive_chunk_samples_this_update += (
                    train_global_positive_chunk_samples * self._world_size
                )
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                for group_name, group_norm in self.last_grad_norms.items():
                    data[f"{group_name}/grad_norm_pre_clip"] = float(group_norm)
                append_to_dict(metrics, data)
                if (
                    max_optimizer_steps_per_update is not None
                    and optimizer_steps_this_update >= max_optimizer_steps_per_update
                ):
                    stopped_on_optimizer_step_limit = True
                    self._logger.info(
                        "Stopping remaining PPO minibatches after the configured "
                        "optimizer-step limit (%d).",
                        max_optimizer_steps_per_update,
                    )
                    break
            if early_stopped_on_kl or stopped_on_optimizer_step_limit:
                break

        kl_beta_before_update = self.kl_beta
        mean_global_batch_reference_kl = (
            float(np.mean(global_batch_reference_kl_values))
            if global_batch_reference_kl_values
            else 0.0
        )
        if reference_target_kl is not None and global_batch_reference_kl_values:
            # The reference KL values were all-reduced, so every rank derives
            # the same coefficient without additional communication.
            self.kl_beta = adapt_kl_beta(
                self.kl_beta,
                mean_global_batch_reference_kl,
                reference_target_kl,
                factor=self.kl_beta_adapt_factor,
                min_beta=self.kl_beta_min,
                max_beta=self.kl_beta_max,
            )
            if self.kl_beta != kl_beta_before_update:
                self._logger.info(
                    "Adapted reference-KL coefficient: kl_beta %.3g -> %.3g "
                    "(mean reference KL %.6f, target %.6f).",
                    kl_beta_before_update,
                    self.kl_beta,
                    mean_global_batch_reference_kl,
                    reference_target_kl,
                )
        append_to_dict(
            metrics,
            {
                "actor/kl_beta": kl_beta_before_update,
                "actor/kl_beta_next": self.kl_beta,
                "actor/mean_global_batch_reference_kl": (
                    mean_global_batch_reference_kl
                ),
                "actor/first_global_batch_proximal_kl": (
                    first_global_batch_proximal_kl
                    if first_global_batch_proximal_kl is not None
                    else 0.0
                ),
                "actor/last_global_batch_proximal_kl": last_global_batch_proximal_kl,
                "actor/max_global_batch_proximal_kl": max_global_batch_proximal_kl,
                "actor/first_global_batch_kl": (
                    first_global_batch_kl if first_global_batch_kl is not None else 0.0
                ),
                "actor/last_global_batch_kl": last_global_batch_kl,
                "actor/max_global_batch_abs_kl": max_global_batch_abs_kl,
                "actor/first_global_batch_reference_kl": (
                    first_global_batch_reference_kl
                    if first_global_batch_reference_kl is not None
                    else 0.0
                ),
                "actor/last_global_batch_reference_kl": (
                    last_global_batch_reference_kl
                ),
                "actor/max_global_batch_reference_kl": (max_global_batch_reference_kl),
                "actor/optimizer_steps_this_update": optimizer_steps_this_update,
                "actor/optimizer_chunk_samples_this_update": (
                    optimizer_steps_this_update * self.cfg.actor.global_batch_size
                ),
                "actor/positive_chunk_samples_available": (
                    positive_chunk_samples_available * self._world_size
                ),
                "actor/positive_chunk_samples_this_update": (
                    positive_chunk_samples_this_update
                ),
                "actor/prioritized_positive_advantage_samples": float(
                    prioritize_positive_samples
                ),
                "actor/optimizer_mean_passes_over_rollout": (
                    optimizer_steps_this_update * batch_size_per_rank / rollout_size
                ),
                "actor/early_stopped_on_kl": float(early_stopped_on_kl),
                "actor/stopped_on_optimizer_step_limit": float(
                    stopped_on_optimizer_step_limit
                ),
                "actor/reference_forward_seconds": reference_forward_seconds,
            },
        )
        append_to_dict(metrics, recompute_logprob_metrics)
        post_update_logprob_metrics = self._compute_post_update_logprob_metrics()
        append_to_dict(metrics, post_update_logprob_metrics)
        append_to_dict(metrics, self._actor_logstd_metrics())
        # Keep the LR schedule aligned with actual parameter updates. A KL
        # rejection can skip every pending optimizer step in this rollout.
        if optimizer_steps_this_update > 0:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        mean_metric_dict.update(
            compute_update_level_explained_variance(mean_metric_dict)
        )
        self.log_on_first_rank(
            "Embodied actor KL/update summary: "
            f"proximal_first={mean_metric_dict['actor/first_global_batch_kl']:.9g}, "
            f"proximal_last={mean_metric_dict['actor/last_global_batch_kl']:.9g}, "
            f"proximal_max_abs={mean_metric_dict['actor/max_global_batch_abs_kl']:.9g}, "
            "proximal_lowvar_first="
            f"{mean_metric_dict['actor/first_global_batch_proximal_kl']:.9g}, "
            "proximal_lowvar_last="
            f"{mean_metric_dict['actor/last_global_batch_proximal_kl']:.9g}, "
            "proximal_lowvar_max="
            f"{mean_metric_dict['actor/max_global_batch_proximal_kl']:.9g}, "
            f"kl_beta={mean_metric_dict['actor/kl_beta']:.9g}->"
            f"{mean_metric_dict['actor/kl_beta_next']:.9g}, "
            "reference_mean="
            f"{mean_metric_dict['actor/mean_global_batch_reference_kl']:.9g}, "
            "reference_first="
            f"{mean_metric_dict['actor/first_global_batch_reference_kl']:.9g}, "
            "reference_last="
            f"{mean_metric_dict['actor/last_global_batch_reference_kl']:.9g}, "
            "reference_max="
            f"{mean_metric_dict['actor/max_global_batch_reference_kl']:.9g}, "
            f"optimizer_steps={mean_metric_dict['actor/optimizer_steps_this_update']:.9g}, "
            "optimizer_chunk_samples="
            f"{mean_metric_dict['actor/optimizer_chunk_samples_this_update']:.9g}, "
            "positive_chunk_samples="
            f"{mean_metric_dict['actor/positive_chunk_samples_this_update']:.9g}/"
            f"{mean_metric_dict['actor/positive_chunk_samples_available']:.9g}, "
            "prioritized_positive_samples="
            f"{mean_metric_dict['actor/prioritized_positive_advantage_samples']:.9g}, "
            "optimizer_mean_passes_over_rollout="
            f"{mean_metric_dict['actor/optimizer_mean_passes_over_rollout']:.9g}, "
            f"early_stopped={mean_metric_dict['actor/early_stopped_on_kl']:.9g}, "
            "stopped_on_optimizer_step_limit="
            f"{mean_metric_dict['actor/stopped_on_optimizer_step_limit']:.9g}, "
            "recomputed_rollout_logprobs="
            f"{mean_metric_dict['actor/recomputed_rollout_logprobs']:.9g}, "
            "rollout_actor_logprob_mean_abs_delta="
            f"{mean_metric_dict['actor/rollout_actor_logprob_mean_abs_delta']:.9g}, "
            "post_update_logprob_mean_abs_delta="
            f"{mean_metric_dict['actor/post_update_logprob_mean_abs_delta']:.9g}, "
            "post_update_ratio_mean_abs_delta="
            f"{mean_metric_dict['actor/post_update_ratio_mean_abs_delta']:.9g}, "
            f"reference_forward_seconds="
            f"{mean_metric_dict['actor/reference_forward_seconds']:.9g}"
        )

        return mean_metric_dict

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
