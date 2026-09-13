RL on StarVLA Models
====================

This page explains how to run reinforcement learning fine-tuning for **StarVLA**
in **RLinf**. StarVLA is an open-source Vision-Language-Action (VLA) toolkit
that composes a VLM backbone and an action head in a modular way. This example
uses **LIBERO Spatial** with the **QwenOFT** setup.

Training Goal Overview
----------------------

This training setup focuses on:

1. Vision understanding from RGB observations.
2. Language understanding from natural-language task instructions.
3. Action generation by reading VLM hidden states and regressing an action
   chunk in parallel with an MLP action head.
4. RL optimization with **GRPO** based on environment feedback.

Environment and Interface Conventions
-------------------------------------

LIBERO Environment
^^^^^^^^^^^^^^^^^^

* **Environment**: LIBERO benchmark based on robosuite / MuJoCo.
* **Task**: control a 7-DoF robot arm for household manipulation skills.
* **Observation**: multi-view RGB images with optional proprio/state.
* **Action space**: continuous actions, commonly 7-D
  (6-D end-effector delta pose + 1-D gripper).
* **Robot platform**: RLinf selects platform-dependent action dimensions and
  (un)normalization behavior via ``ROBOT_PLATFORM``. This page assumes
  ``ROBOT_PLATFORM=libero``.

Task Description Format
^^^^^^^^^^^^^^^^^^^^^^^

StarVLA directly consumes environment-provided natural-language task
descriptions as language-model input.

Environment Observation Structure (``env_obs``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the RLinf StarVLA wrapper, ``env_obs`` is expected to be a dict in
batch-first format (dimension 0 is batch size ``B``).

Required fields:

* ``main_images``: main-view RGB, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``states``: proprio/state tensor, ``torch.float32``, shape ``[B, D_state]``.
* ``task_descriptions``: natural-language descriptions, ``list[str]`` with
  length ``B``.

Optional fields:

* ``wrist_images``: wrist-view RGB, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``extra_view_images``: additional RGB views, recommended shape
  ``[B, V, H, W, 3]`` where ``V`` is number of extra views. A single extra
  view may also be provided as ``[B, H, W, 3]`` and is treated as ``V=1``.

In default LIBERO usage, ``states`` is commonly:

* end-effector position ``(x, y, z)`` (3-D)
* end-effector axis-angle ``(rx, ry, rz)`` (3-D)
* gripper state (originally 2-D)

So ``D_state`` is often ``3 + 3 + 2 = 8``. If a checkpoint expects 7-D state,
the wrapper compresses the 2-D gripper state into:

``[x, y, z, rx, ry, rz, g_mean]`` where ``g_mean = 0.5 * (g0 + g1)``.

Action Chunk Interface
^^^^^^^^^^^^^^^^^^^^^^

StarVLA inference outputs chunked actions:

* ``actions``: ``torch.float32``, shape ``[B, T, D_action]``
* ``T = actor.model.num_action_chunks``: chunk length / planning horizon
* ``D_action = actor.model.action_dim``: action dimension (commonly 7 on LIBERO)

Rollout typically follows a receding-horizon strategy: each policy forward pass
predicts ``T`` actions; the environment executes the first ``N`` steps
(``1 <= N <= T``), then replans.

Algorithm Notes
---------------

* **StarVLA (QwenOFT)**: the VLM backbone provides multimodal understanding,
  while the MLP action head regresses continuous chunked actions in parallel
  (non-diffusion decoding).
* **GRPO**: policy optimization with environment feedback, integrated with
  RLinf embodied training and LIBERO.

Installation
------------

1. Clone RLinf
^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

.. note::

   Docker image tags are pinned for reproducibility and may lag behind the latest
   RLinf dependencies. If you need newer versions, rebuild the image from ``docker/``
   or use the custom environment installation below.

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env starvla

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model starvla --env maniskill_libero
   source .venv/bin/activate

Model Download
----------------

Before training, download the required StarVLA checkpoint and base VLM:

* ``StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1``
* ``Qwen/Qwen2.5-VL-3B-Instruct``

.. code-block:: bash

   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   uv pip install huggingface-hub
   hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
   hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

.. note::

   After download, update ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` so
   ``framework.qwenvl.base_vlm`` points to your local
   ``Qwen2.5-VL-3B-Instruct`` path.

Quickstart
----------

Config file
^^^^^^^^^^^

StarVLA + GRPO + LIBERO (spatial tasks) example config:

* ``examples/embodiment/config/libero_spatial_grpo_starvla.yaml``

Key config snippet
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   defaults:
      - env/libero_spatial@env.train
      - env/libero_spatial@env.eval

   rollout:
     model:
       model_path: "/path/to/model"

   actor:
     model:
       model_path: "/path/to/model"
       action_dim: 7
       num_action_chunks: 8
       action_stats_source: "minmax"
       starvla:
         framework_name: "QwenOFT"
         expected_action_dim: ${actor.model.action_dim}
         expected_num_action_chunks: ${actor.model.num_action_chunks}
         enable_state_input: False

Run training
^^^^^^^^^^^^

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_starvla

RoboCasa GR1 CupToDrawerClose
-----------------------------

The repository also includes a grouped-GRPO setup for the StarVLA Qwen3 OFT
checkpoint on the RoboCasa GR1 ``CupToDrawerClose`` task:

* a fixed, disjoint 500-seed training pool;
* 16 seeds sampled without replacement per rollout collection;
* 8 stochastic trajectories for every sampled seed (128 rollouts total);
* checkpoint evaluation on the same fixed 50-seed suite at a configured step
  interval;
* train and validation videos; and
* local JSONL plus W&B tables containing every seed, trajectory outcome, and
  per-seed success rate.

Update the model and dependency paths in the config or launcher if they differ
from the local installation, then run:

.. code-block:: bash

   bash examples/embodiment/run_robocasa_gr1_cup_drawer_grpo.sh

The main config is
``examples/embodiment/config/robocasa_gr1_cup_drawer_grpo_starvla.yaml``.
Detailed rollout records are written under ``rollout_tables/`` in the log
directory and uploaded as ``train/seed_rollouts`` and
``eval/seed_rollouts`` W&B tables. Videos are stored under ``video/train`` and
``video/eval``. The evaluation uses 56 simulator slots for communication
divisibility, but six slots are marked as padding and excluded from metrics, so
``eval/success_once`` is computed over exactly 50 fixed seeds.
On the configured host, simulator subprocesses remove ``CUDA_VISIBLE_DEVICES``
before importing robosuite and select the physical NVIDIA EGL device assigned
to each environment worker. Actor and rollout workers keep their CUDA
assignments. Set ``egl_device`` explicitly only on hosts where automatic
physical-device selection is unavailable.
This host recipe pins Ray to ``2.49.2``: Ray ``2.57.0`` was observed to abort
RLinf ``ChannelWorker`` process-group initialization with a pybind11
deallocation error. The launcher checks the version before starting. It also
keeps ``gradient_checkpointing: false``, matching RLinf's official StarVLA
recipe because QwenOFT does not provide a checkpoint-safe forward under FSDP.

The checkpoint stores the action head in BF16. Small Adam updates can therefore
round back to the same stored value even when the optimizer step succeeds. The
RoboCasa config keeps the VLM backbone in BF16 while setting
``actor.model.action_model_precision: fp32``. It also wraps the OFT
``L1RegressionActionHead`` as a separate FSDP unit, preserving uniform dtypes
inside every flat parameter. Actor, Hugging Face rollout, sparse weight sync,
and standalone checkpoint evaluation all retain the per-module dtype.

The StarVLA model config also supports parameter-efficient Qwen3-VL updates.
Set ``actor.model.is_lora: true`` and
``actor.model.lora_scope: qwen3_vl`` to inject LoRA only into the Qwen3-VL
vision/language subtree. By default the action head remains frozen. Set
``actor.model.lora_train_action_head: true`` to jointly optimize the Qwen3-VL
adapters and the complete action head; Qwen base weights and a fixed
``actor_logstd`` remain frozen. For sparse patch synchronization, select the
action-head prefix together with the ``.lora_A.`` and ``.lora_B.`` name
fragments so resumed rollout workers receive both trained components.

The GR1 policy executes ``12 x 29 = 348`` Gaussian action dimensions per
chunk with ``sigma = exp(-3.5)``, so a chunk-level likelihood ratio sums 348
log-ratios and PPO clipping saturates after tiny mean shifts. The supplied
config therefore optimizes ``token_level`` (per-dimension) ratios with a
``0.2`` clip, a ``1e-6`` learning rate on the FP32 OFT head, 512 chunk samples
per global batch, and a per-dimension ``target_kl`` trust region measured with
the low-variance estimator. The reference-policy KL is a leash rather than a
boundary: ``reference_target_kl`` adapts ``kl_beta`` after every update
(doubling above ``1.5x`` target, halving below ``target / 1.5``, clamped to
``[kl_beta_min, kl_beta_max]``). A hard stop against the immutable reference
could never recover once exceeded. Metrics report the first, last, and maximum
global-batch proximal and reference KL, the coefficient used and its next
value, and whether the trust region or optimizer-step limit stopped the
remaining minibatches.

``robocasa_gr1_cup_drawer_ppo_starvla.yaml`` is the GAE / value-head variant.
Its critic is a zero-initialized FP32 linear head (``value_head_zero_init``),
actor and critic parameter groups are clipped separately
(``optim.clip_grad_per_group`` with ``clip_grad`` / ``value_clip_grad``), and a
soft critic warmup (``optim.critic_warmup_steps`` with
``critic_warmup_mode: soft``) drops the actor gradients for the first
optimizer steps without toggling ``requires_grad`` on the FSDP-wrapped model.
``critic/explained_variance`` is recombined from per-micro-batch sufficient
statistics at the update level; the raw per-micro-batch value is kept as
``critic/explained_variance_micro_batch``.

``robocasa_gr1_cup_drawer_sac_starvla.yaml`` (twin-Q SAC) stores a
``transition_valid`` flag with every replay transition so steps after an
episode already terminated (no auto-reset) get zero weight in the critic and
actor losses, and it fails fast if ``actor.model.sac_task_description`` differs
from the instruction the environment emits, because replayed forwards rebuild
the prompt from that string.

``robocasa_gr1_24task_ppo_starvla.yaml`` trains one policy on all 24
``gr1_unified`` tabletop tasks with the same PPO recipe. Multi-task support
lives entirely in the environment, as in RLinf's LIBERO-130 recipe: the env
config lists ``task_names`` (``env/robocasa_gr1_24task.yaml``), every global
seed group ``g`` is owned by task ``g % num_tasks`` so each simulator
subprocess keeps one task for the whole run, and each task walks its own seed
ring from a ``{"tasks": {...}}`` manifest
(``seeds/robocasa_gr1_24task_train_500.json``; the evaluation manifest is the
official 24 x 50 fixed-seed list). The policy is conditioned only through the
per-environment instruction in ``task_descriptions``; because prompts are
padded to ``actor.model.rollout_prompt_seq_len`` for tensor stacking, that
value must cover the longest instruction (192 for this suite). Evaluation
covers ``eval_seed_count`` seeds per task over several ordered rounds when
there are fewer environments than seeds; ``algorithm.eval_rollout_epoch`` must
equal the required round count, which ``validate_embodied_cfg`` checks. The
environment reports ``sample_task`` per trajectory, the rollout tables carry
the task name per row, and the runner logs ``env/task/<task>/success_once``
and ``eval/task/<task>/success_once`` next to the aggregate metrics.

Grouped GRPO requires ``env`` and ``actor`` to have the same world size (or at
least that every trajectory piece an actor rank receives holds whole seed
groups): environment workers split their rollouts across actor ranks by batch
position, and group-relative advantages are normalized inside each actor rank.
``validate_embodied_cfg`` rejects placements that would cut a seed group.

Use the controller script to alternate training and official-compatible fixed
50-seed evaluation. It registers the original checkpoint as step 0, keeps
periodic checkpoints, and records the checkpoint with the best fixed-seed
result:

.. code-block:: bash

   EVAL_INTERVAL=2 TARGET_MAX_STEPS=10 \
     bash examples/embodiment/run_robocasa_gr1_cup_drawer_grpo_eval_loop.sh

Both actor and rollout load the same StarVLA checkpoint at startup, so this
recipe disables the patch syncer's redundant full-state ``init_sync``. On the
FSDP-DTensor stack used for this task, that initial full overwrite makes the
pre-update KL invalid; later sparse weight deltas remain enabled and synchronize
normal training updates.

Evaluation
----------

For evaluation, we recommend RLinf's unified VLA evaluation workflow
(see the Embodied VLA Evaluation tutorial in RLinf docs).

Reference results

Note: the following results use the model from
https://huggingface.co/JasonYang66/LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT.

LIBERO Goal:

.. image:: https://github.com/RLinf/misc/raw/main/pic/libero_goal_starvla_baseline.png
   :alt: libero_goal_starvla_baseline result curve
   :width: 95%
   :align: center

LIBERO Object:

.. image:: https://github.com/RLinf/misc/raw/main/pic/libero_object_starvla_baseline.png
   :alt: libero_object_starvla_baseline result curve
   :width: 95%
   :align: center

LIBERO Spatial:

.. image:: https://github.com/RLinf/misc/raw/main/pic/libero_spatial_starvla_baseline.png
   :alt: libero_spatial_starvla_baseline result curve
   :width: 95%
   :align: center
