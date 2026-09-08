#!/usr/bin/env python3

"""Validate RoboCasa-GR1 milestone rewards in a real EGL environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.envs.robocasa_gr1.robocasa_gr1_env import RoboCasaGR1Env


DEFAULT_SEED_MANIFEST = Path(
    "/data/dengyixuan/wyz/robots/RLinf/examples/embodiment/config/seeds/"
    "robocasa_gr1_cup_drawer_train_500.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--egl-device", type=int, default=0)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--seed-pool-size", type=int, default=500)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.seed_pool_size <= 0:
        raise ValueError("--seed-pool-size must be positive")

    cfg = OmegaConf.create(
        {
            "task_name": (
                "gr1_unified/"
                "PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
            ),
            "group_size": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_episode_steps": 720,
            "use_rel_reward": True,
            "seed": 20260820,
            "seed_sampler_seed": 20260820,
            "seed_manifest": str(args.seed_manifest),
            "seed_pool_size": args.seed_pool_size,
            "eval_seed_count": None,
            "is_eval": False,
            "egl_device": args.egl_device,
            "renderer_backend": "nvidia",
            "action_steps_per_chunk": 12,
            "subtask_reward_shaping": {
                "enabled": True,
                "grasp_object": 0.05,
                "obj_in_drawer": 0.2,
                "success": 1.0,
            },
        }
    )
    env = RoboCasaGR1Env(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    records = []
    try:
        observations, _ = env.reset()
        assert observations["main_images"].shape[0] == 1
        for _ in range(args.steps):
            _, reward, terminated, truncated, infos = env.step(
                torch.zeros((1, 29), dtype=torch.float32), auto_reset=False
            )
            episode = infos["episode"]
            for key in (
                "grasp_first_step",
                "obj_in_drawer_first_step",
                "success_first_step",
            ):
                assert key in episode
            assert reward.shape == (1,)
            subtask_signals = infos["subtask_signals"][0]
            records.append(
                {
                    "elapsed_step": int(episode["episode_len"][0]),
                    "reward": float(reward[0]),
                    "terminated": bool(terminated[0]),
                    "truncated": bool(truncated[0]),
                    "subtask_signals": {
                        key: int(value) for key, value in subtask_signals.items()
                    },
                    "grasp_first_step": int(episode["grasp_first_step"][0]),
                    "obj_in_drawer_first_step": int(
                        episode["obj_in_drawer_first_step"][0]
                    ),
                    "success_first_step": int(episode["success_first_step"][0]),
                }
            )
    finally:
        env.close()

    print(
        json.dumps(
            {
                "egl_device": args.egl_device,
                "steps": args.steps,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
