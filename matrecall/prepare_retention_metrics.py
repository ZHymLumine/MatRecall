"""Join frozen-transfer predictions to gold only after prediction completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matrecall.episodic_memory import write_jsonl
from matrecall.run_frozen_memory import METHODS
from matrecall.run_online_adaptation import file_sha256, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--exclude-task", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    completion = json.loads((args.run / "completion.json").read_text())
    manifest = json.loads((args.release / "manifest.json").read_text())
    if not str(completion.get("status", "")).startswith("completed"):
        raise ValueError("Transfer predictions are incomplete")
    if completion.get("memory_writes") != 0 or completion.get("transfer_feedback_reveals") != 0:
        raise ValueError("Transfer run violated frozen-memory policy")
    if completion.get("methods") != args.methods:
        raise ValueError("Method list mismatch")
    excluded = set(args.exclude_task)
    available = {row["task"] for row in manifest["tasks"]}
    unknown = excluded - available
    if unknown:
        raise ValueError(f"Unknown task exclusions: {sorted(unknown)}")
    task_records = [row for row in manifest["tasks"] if row["task"] not in excluded]
    if completion.get("task_count") != len(task_records):
        raise ValueError("Prediction task coverage does not match metric task selection")
    hashes = {}
    for method in args.methods:
        for family in sorted({row["family"] for row in manifest["tasks"]}):
            predictions = read_jsonl(
                args.run / "combined" / method / family / "predictions.jsonl"
            )
            gold = []
            for task in [row for row in task_records if row["family"] == family]:
                gold.extend(read_jsonl(
                    args.release / task["relative_path"] / "gold.jsonl"
                ))
            if {row["case_id"] for row in predictions} != {row["case_id"] for row in gold}:
                raise ValueError(f"Prediction/gold cohort mismatch: {method}/{family}")
            base = args.output / method / family
            write_jsonl(base / "predictions.jsonl", predictions)
            write_jsonl(base / "gold.jsonl", gold)
            hashes[f"{method}/{family}/predictions"] = file_sha256(
                base / "predictions.jsonl"
            )
            hashes[f"{method}/{family}/gold"] = file_sha256(base / "gold.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "task_count": len(task_records),
        "source_task_count": manifest["task_count"],
        "excluded_tasks": sorted(excluded),
        "case_count_per_method": sum(row["transfer_cases"] for row in task_records),
        "methods": args.methods,
        "gold_joined_after_prediction_completion": True,
        "prediction_completion_sha256": file_sha256(args.run / "completion.json"),
        "files_sha256": hashes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
