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

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    _ROLLOUT_IDENTITY_FIELDS = {
        "sample_seed",
        "sample_group",
        "sample_trajectory",
        "sample_task",
    }

    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self, step: int | None = None):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        self._log_seed_rollout_table(
            eval_metrics_list,
            step=self.global_step if step is None else step,
            mode="eval",
        )
        per_task_metrics = self._per_task_metrics(eval_metrics_list, mode="eval")
        eval_metrics_list = self._without_rollout_identity(eval_metrics_list)
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        eval_metrics.update(per_task_metrics)
        return eval_metrics

    def _without_rollout_identity(self, metrics_list: list[dict]) -> list[dict]:
        return [
            {
                key: value
                for key, value in metrics.items()
                if key not in self._ROLLOUT_IDENTITY_FIELDS
            }
            for metrics in metrics_list
        ]

    def _env_task_names(self, mode: str) -> list[str]:
        """Ordered task list of the train or eval environment config."""
        env_cfg = self.cfg.env.eval if mode == "eval" else self.cfg.env.train
        # getattr rather than .get: the env config is a DictConfig in a real
        # run but a plain namespace in the runner unit tests.
        task_names = getattr(env_cfg, "task_names", None)
        if task_names:
            return [str(name) for name in task_names]
        task_name = getattr(env_cfg, "task_name", None)
        return [str(task_name)] if task_name is not None else []

    def _per_task_metrics(self, metrics_list: list[dict], *, mode: str) -> dict:
        """Success rate and episode count per task, keyed ``task/<name>/...``.

        Only produced when the environment reports ``sample_task`` and more
        than one task is configured; single-task runs keep their flat metrics.
        """
        task_ids = self._metric_values(metrics_list, "sample_task")
        task_names = self._env_task_names(mode)
        if not task_ids or len(task_names) <= 1:
            return {}
        successes = self._metric_values(metrics_list, "success_once")
        if len(successes) != len(task_ids):
            raise RuntimeError(
                f"Mismatched {mode} per-task metric lengths: "
                f"{len(successes)} successes vs {len(task_ids)} task ids"
            )
        outcomes: dict[int, list[float]] = defaultdict(list)
        for task_id, success in zip(task_ids, successes):
            outcomes[int(task_id)].append(float(success))
        metrics = {}
        for task_id in sorted(outcomes):
            name = task_names[task_id].split("/")[-1]
            values = outcomes[task_id]
            metrics[f"task/{name}/success_once"] = sum(values) / len(values)
            metrics[f"task/{name}/episodes"] = float(len(values))
        return metrics

    @staticmethod
    def _metric_values(metrics_list: list[dict], key: str) -> list:
        values = []
        for metrics in metrics_list:
            value = metrics.get(key, None)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().cpu().reshape(-1).tolist()
            elif hasattr(value, "reshape"):
                value = value.reshape(-1).tolist()
            elif isinstance(value, (tuple, list)):
                value = list(value)
            else:
                value = [value]
            values.extend(value)
        return values

    def _log_seed_rollout_table(
        self, metrics_list: list[dict], *, step: int, mode: str
    ) -> None:
        """Persist per-seed/per-trajectory outcomes and mirror them to W&B."""
        seeds = self._metric_values(metrics_list, "sample_seed")
        if not seeds:
            return

        groups = self._metric_values(metrics_list, "sample_group")
        trajectories = self._metric_values(metrics_list, "sample_trajectory")
        successes = self._metric_values(metrics_list, "success_once")
        grasped = self._metric_values(metrics_list, "grasped_once")
        in_drawer = self._metric_values(metrics_list, "obj_in_drawer_once")
        grasp_steps = self._metric_values(metrics_list, "grasp_first_step")
        drawer_steps = self._metric_values(metrics_list, "obj_in_drawer_first_step")
        success_steps = self._metric_values(metrics_list, "success_first_step")
        episode_lengths = self._metric_values(metrics_list, "episode_len")
        returns = self._metric_values(metrics_list, "return")
        field_lengths = {
            len(seeds),
            len(groups),
            len(trajectories),
            len(successes),
            len(episode_lengths),
            len(returns),
        }
        if len(field_lengths) != 1:
            raise RuntimeError(
                f"Mismatched {mode} rollout table field lengths: {field_lengths}"
            )
        trajectory_count = len(seeds)
        for name, values in (
            ("grasped_once", grasped),
            ("obj_in_drawer_once", in_drawer),
            ("grasp_first_step", grasp_steps),
            ("obj_in_drawer_first_step", drawer_steps),
            ("success_first_step", success_steps),
        ):
            if values and len(values) != trajectory_count:
                raise RuntimeError(
                    f"Mismatched {mode} rollout table {name} field length: "
                    f"{len(values)} != {trajectory_count}"
                )

        success_by_seed: dict[int, list[float]] = defaultdict(list)
        for seed, success in zip(seeds, successes):
            success_by_seed[int(seed)].append(float(success))
        rate_by_seed = {
            seed: sum(outcomes) / len(outcomes)
            for seed, outcomes in success_by_seed.items()
        }

        def rates_by_seed(values: list) -> dict[int, float]:
            outcomes_by_seed: dict[int, list[float]] = defaultdict(list)
            for seed, outcome in zip(seeds, values):
                outcomes_by_seed[int(seed)].append(float(outcome))
            return {
                seed: sum(outcomes) / len(outcomes)
                for seed, outcomes in outcomes_by_seed.items()
            }

        grasp_rate_by_seed = rates_by_seed(grasped) if grasped else {}
        drawer_rate_by_seed = rates_by_seed(in_drawer) if in_drawer else {}

        task_names = self._env_task_names(mode)
        task_ids = self._metric_values(metrics_list, "sample_task")
        if task_ids and len(task_ids) != trajectory_count:
            raise RuntimeError(
                f"Mismatched {mode} rollout table sample_task field length: "
                f"{len(task_ids)} != {trajectory_count}"
            )
        rows = []
        for row_index, (
            seed,
            group,
            trajectory,
            success,
            episode_len,
            episode_return,
        ) in enumerate(
            zip(
                seeds,
                groups,
                trajectories,
                successes,
                episode_lengths,
                returns,
            )
        ):
            rows.append(
                {
                    "training_step": int(step),
                    "mode": mode,
                    "task": (
                        task_names[int(task_ids[row_index])]
                        if task_ids
                        else (task_names[0] if task_names else "")
                    ),
                    "seed": int(seed),
                    "group_index": int(group),
                    "trajectory_index": int(trajectory),
                    "trajectory_success": float(success),
                    "seed_success_rate": float(rate_by_seed[int(seed)]),
                    "trajectory_grasped": (
                        float(grasped[row_index]) if grasped else None
                    ),
                    "seed_grasp_rate": grasp_rate_by_seed.get(int(seed)),
                    "trajectory_obj_in_drawer": (
                        float(in_drawer[row_index]) if in_drawer else None
                    ),
                    "seed_obj_in_drawer_rate": drawer_rate_by_seed.get(int(seed)),
                    "grasp_first_step": (
                        int(grasp_steps[row_index]) if grasp_steps else None
                    ),
                    "obj_in_drawer_first_step": (
                        int(drawer_steps[row_index]) if drawer_steps else None
                    ),
                    "success_first_step": (
                        int(success_steps[row_index]) if success_steps else None
                    ),
                    "episode_length": int(episode_len),
                    "episode_return": float(episode_return),
                }
            )

        table_dir = os.path.join(self.cfg.runner.logger.log_path, "rollout_tables")
        os.makedirs(table_dir, exist_ok=True)
        jsonl_path = os.path.join(table_dir, f"{mode}_seed_rollouts.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        if "wandb" in self.metric_logger.logger_backends:
            import pandas as pd

            self.metric_logger.log_table(
                pd.DataFrame(rows),
                name=f"{mode}/seed_rollouts",
                step=int(step),
            )

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _log_step_metrics_jsonl(self, metrics: dict, step: int) -> None:
        """Persist complete step metrics even when no remote logger is enabled."""
        row = {
            "training_step": int(step),
            "completed_global_step": int(self.global_step),
            "timestamp_unix": time.time(),
        }
        for key, value in metrics.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (bool, int, float, str)) or value is None:
                row[key] = value
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = str(value)

        metrics_dir = os.path.join(self.cfg.runner.logger.log_path, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        jsonl_path = os.path.join(metrics_dir, "training_step_metrics.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step).wait()

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                actor_training_handle: Handle = self.actor.run_training()
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_training_metrics = actor_training_handle.wait()
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate(step=_step)
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
                return_per_rank=True
            )
            rollout_time_metrics, rollout_time_metrics_per_rank = (
                rollout_handle.consume_durations(return_per_rank=True)
            )
            actor_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
            )
            if self.reward is not None:
                reward_time_metrics, reward_time_metrics_per_rank = (
                    reward_handle.consume_durations(return_per_rank=True)
                )
                time_metrics.update(
                    {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
                )

            env_results = env_handle.wait()
            env_results_list = [
                results for results in env_results if results is not None
            ]
            self._log_seed_rollout_table(env_results_list, step=_step, mode="train")
            env_metrics = compute_evaluate_metrics(
                self._without_rollout_identity(env_results_list)
            )
            env_metrics.update(self._per_task_metrics(env_results_list, mode="train"))
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            ranked_env_results = [
                {"rank": rank, "env": rank_metrics}
                for rank, rank_metrics in enumerate(env_results)
                if rank_metrics is not None
            ]
            _, env_metrics_per_rank = self._process_ranked_eval_results(
                ranked_env_results, metric_field="env"
            )

            rollout_metrics = {
                f"rollout/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_rollout_metrics
                ).items()
            }
            training_metrics = {
                f"train/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_training_metrics
                ).items()
            }

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            self._log_ranked_metrics(
                metrics_list=actor_rollout_metrics,
                step=_step,
                prefix="rollout",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_training_metrics,
                step=_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
            )
            if self.reward is not None:
                self._log_ranked_metrics(
                    metrics_list=reward_time_metrics_per_rank,
                    step=_step,
                    prefix="time/reward",
                    worker_group_name=self.reward.worker_group_name,
                )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self._log_step_metrics_jsonl(logging_metrics, _step)

            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
