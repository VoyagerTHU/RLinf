"""Record fixed-50 evaluations and maintain latest/best checkpoint links."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
    temporary = link.with_name(link.name + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.eval_root / "summary.json"
    audit_path = args.eval_root / "video_decode_audit.json"
    summary = read_json(summary_path)
    audit = read_json(audit_path)
    if summary.get("episodes") != 50 or summary.get("unique_seeds") != 50:
        raise ValueError(f"Incomplete fixed-50 summary: {summary_path}")
    if not summary.get("initial_observation_hashes_match_reference"):
        raise ValueError(f"Observation hashes failed reference audit: {summary_path}")
    if audit.get("decoded_video_count") != 50 or audit.get("failure_count") != 0:
        raise ValueError(f"Video audit failed: {audit_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    registry = (
        read_json(args.registry)
        if args.registry.exists()
        else {"schema_version": 1, "evaluations": []}
    )
    record = {
        "step": args.step,
        "checkpoint": str(args.checkpoint.resolve()),
        "eval_root": str(args.eval_root.resolve()),
        "successes": int(summary["successes"]),
        "episodes": int(summary["episodes"]),
        "success_rate": float(summary["success_rate"]),
        "seed_manifest_sha256": summary["seed_manifest_sha256"],
        "initial_observation_hashes_match_reference": True,
        "decoded_video_count": 50,
    }
    evaluations = [
        previous
        for previous in registry.get("evaluations", [])
        if int(previous["step"]) != args.step
    ]
    evaluations.append(record)
    evaluations.sort(key=lambda value: int(value["step"]))
    best = max(
        evaluations,
        key=lambda value: (int(value["successes"]), -int(value["step"])),
    )
    latest = evaluations[-1]
    registry.update(
        {
            "evaluations": evaluations,
            "latest": latest,
            "best": best,
            "baseline": next(
                (value for value in evaluations if int(value["step"]) == 0),
                None,
            ),
        }
    )
    write_json_atomic(args.registry, registry)

    root = args.registry.parent
    replace_symlink(
        root / "latest_50seed_checkpoint", Path(latest["checkpoint"])
    )
    replace_symlink(root / "latest_50seed_eval", Path(latest["eval_root"]))
    replace_symlink(root / "best_50seed_checkpoint", Path(best["checkpoint"]))
    replace_symlink(root / "best_50seed_eval", Path(best["eval_root"]))
    print(json.dumps({"latest": latest, "best": best}, indent=2))


if __name__ == "__main__":
    main()
