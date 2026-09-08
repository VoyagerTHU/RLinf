"""Merge and verify sharded fixed-seed evaluation for one RoboCasa GR1 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--reference-result", type=Path)
    args = parser.parse_args()

    manifest = read_json(args.seed_manifest)
    expected_seeds = manifest["tasks"][args.task_name]
    shard_paths = sorted((args.run_root / "results_shards").glob("shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError("No result shards were produced")
    records = [
        record
        for shard_path in shard_paths
        for record in read_json(shard_path)["episodes"]
    ]
    records.sort(key=lambda record: int(record["episode_index"]))

    actual_indices = [int(record["episode_index"]) for record in records]
    actual_seeds = [int(record["seed"]) for record in records]
    if actual_indices != list(range(len(expected_seeds))):
        raise ValueError(f"Episode indices are incomplete or duplicated: {actual_indices}")
    if actual_seeds != expected_seeds:
        raise ValueError("Evaluated seeds differ from the frozen manifest")

    hashes = [record.get("initial_observation_sha256") for record in records]
    if not all(isinstance(value, str) and len(value) == 64 for value in hashes):
        raise ValueError("One or more episodes lack an initial observation hash")

    reference_hashes_match = None
    reference_successes = None
    gained_success_seeds: list[int] = []
    lost_success_seeds: list[int] = []
    unchanged_success_count = None
    unchanged_failure_count = None
    if args.reference_result is not None:
        reference = read_json(args.reference_result)
        reference_by_seed = {int(record["seed"]): record for record in reference["episodes"]}
        mismatches = [
            seed
            for seed, value in zip(actual_seeds, hashes, strict=True)
            if reference_by_seed.get(seed, {}).get("initial_observation_sha256") != value
        ]
        if mismatches:
            raise ValueError(
                f"Initial observations differ from the reference for seeds: {mismatches}"
            )
        reference_hashes_match = True
        reference_successes = sum(
            bool(reference_by_seed[seed]["success"]) for seed in actual_seeds
        )
        gained_success_seeds = [
            int(record["seed"])
            for record in records
            if bool(record["success"])
            and not bool(reference_by_seed[int(record["seed"])]["success"])
        ]
        lost_success_seeds = [
            int(record["seed"])
            for record in records
            if not bool(record["success"])
            and bool(reference_by_seed[int(record["seed"])]["success"])
        ]
        unchanged_success_count = sum(
            bool(record["success"])
            and bool(reference_by_seed[int(record["seed"])]["success"])
            for record in records
        )
        unchanged_failure_count = sum(
            not bool(record["success"])
            and not bool(reference_by_seed[int(record["seed"])]["success"])
            for record in records
        )

    videos = []
    for record in records:
        prefix = f"episode_{int(record['episode_index']):03d}_seed_{int(record['seed'])}"
        candidates = [
            path
            for path in (args.run_root / "videos").glob(f"**/{prefix}*.mp4")
            if path.is_file() and path.stat().st_size > 0
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one completed video for {prefix}, found {candidates}"
            )
        videos.append(str(candidates[0]))

    successes = sum(bool(record["success"]) for record in records)
    task_slug = args.task_name.split("/", 1)[-1]
    merged_result = {
        "env_name": args.task_name,
        "n_episodes": len(records),
        "completed_episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "episodes": records,
    }
    merged_path = args.run_root / "results_merged" / f"{task_slug}.json"
    write_json_atomic(merged_path, merged_result)

    summary = {
        "task_name": args.task_name,
        "seed_manifest_sha256": hashlib.sha256(args.seed_manifest.read_bytes()).hexdigest(),
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "unique_seeds": len(set(actual_seeds)),
        "unique_initial_observations": len(set(hashes)),
        "initial_observation_hashes_match_reference": reference_hashes_match,
        "reference_successes": reference_successes,
        "reference_success_rate": (
            reference_successes / len(records)
            if reference_successes is not None
            else None
        ),
        "success_rate_delta_vs_reference": (
            (successes - reference_successes) / len(records)
            if reference_successes is not None
            else None
        ),
        "gained_success_seeds": gained_success_seeds,
        "lost_success_seeds": lost_success_seeds,
        "unchanged_success_count": unchanged_success_count,
        "unchanged_failure_count": unchanged_failure_count,
        "video_count": len(videos),
        "merged_result": str(merged_path),
    }
    write_json_atomic(args.run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
