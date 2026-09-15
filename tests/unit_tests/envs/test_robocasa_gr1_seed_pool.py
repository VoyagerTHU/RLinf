"""Tests for grouped RoboCasa-GR1 training and evaluation seeds."""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.robocasa_gr1.robocasa_gr1_env import RoboCasaGR1Env
from rlinf.envs.robocasa_gr1.seed_pool import (
    load_task_seeds,
    select_process_seed_groups,
)

TASK = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_DIR = REPO_ROOT / "examples/embodiment/config/seeds"


def test_training_and_evaluation_manifests_are_disjoint():
    train = load_task_seeds(
        SEED_DIR / "robocasa_gr1_cup_drawer_train_500.json",
        TASK,
        expected_size=500,
    )
    evaluation = load_task_seeds(
        SEED_DIR / "robocasa_gr1_cup_drawer_eval_50.json",
        TASK,
        expected_size=50,
    )

    assert len(set(train)) == 500
    assert len(set(evaluation)) == 50
    assert set(train).isdisjoint(evaluation)


def test_one_training_round_selects_16_unique_groups_reproducibly():
    pool = list(range(1, 501))
    selections = [
        select_process_seed_groups(
            pool,
            groups_per_process=2,
            process_index=process_index,
            total_processes=8,
            selection_round=3,
            sampler_seed=20260820,
            shuffle=True,
        )[0]
        for process_index in range(8)
    ]
    selected = np.concatenate(selections)

    assert selected.shape == (16,)
    assert len(set(selected.tolist())) == 16
    np.testing.assert_array_equal(
        selections[0],
        select_process_seed_groups(
            pool,
            groups_per_process=2,
            process_index=0,
            total_processes=8,
            selection_round=3,
            sampler_seed=20260820,
            shuffle=True,
        )[0],
    )


def test_consecutive_training_rounds_cover_pool_before_repeating():
    pool = list(range(1, 501))
    selected = []
    for selection_round in range(125):
        process_selections = [
            select_process_seed_groups(
                pool,
                groups_per_process=1,
                process_index=process_index,
                total_processes=4,
                selection_round=selection_round,
                sampler_seed=20260820,
                shuffle=True,
            )[0]
            for process_index in range(4)
        ]
        selected.extend(np.concatenate(process_selections).tolist())

    assert len(selected) == 500
    assert len(set(selected)) == 500
    assert set(selected) == set(pool)


def test_evaluation_padding_marks_exactly_50_valid_seeds():
    valid_masks = []
    selected = []
    for process_index in range(8):
        seeds, valid, _ = select_process_seed_groups(
            list(range(100, 150)),
            groups_per_process=7,
            process_index=process_index,
            total_processes=8,
            selection_round=0,
            sampler_seed=0,
            shuffle=False,
            valid_seed_count=50,
        )
        selected.extend(seeds.tolist())
        valid_masks.extend(valid.tolist())

    assert len(selected) == 56
    assert sum(valid_masks) == 50
    assert selected[:50] == list(range(100, 150))
    assert selected[50:] == list(range(100, 106))


def test_manifest_metadata_matches_seed_array():
    path = SEED_DIR / "robocasa_gr1_cup_drawer_train_500.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == TASK
    assert payload["seed_count"] == len(payload["seeds"]) == 500


def test_restoring_selection_round_skips_completed_rollout_epochs():
    env = RoboCasaGR1Env.__new__(RoboCasaGR1Env)
    env.cfg = OmegaConf.create(
        {
            "is_eval": False,
            "seed_sampler_seed": 20260820,
            "seed": 20260820,
        }
    )
    env.seed_pools = [list(range(1, 501))]
    env.env_task_ids = None
    env.num_group = 2
    env.seed_offset = 0
    env.total_num_processes = 8
    env.group_size = 8
    env._selection_round = 0

    env._assign_seed_groups()
    initial_seeds = env.group_seeds.copy()
    env.set_seed_selection_round(8)

    assert env._selection_round == 8
    assert not np.array_equal(env.group_seeds, initial_seeds)
    expected, _, _ = select_process_seed_groups(
        env.seed_pools[0],
        groups_per_process=2,
        process_index=0,
        total_processes=8,
        selection_round=8,
        sampler_seed=20260820,
        shuffle=True,
    )
    np.testing.assert_array_equal(env.group_seeds, expected)
