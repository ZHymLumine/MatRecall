"""Join cross-task predictions to gold and bind frozen memory-retention controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matrecall.episodic_memory import write_jsonl
from matrecall.run_online_adaptation import file_sha256, read_jsonl


CROSS_METHODS = (
    "case_episodic_only", "full_episodic_only",
    "dynamic_case_rag", "matrecall",
)
CONTROL_METHODS = (
    "llm_only", "static_multiview_rag", "react_static_memory",
    "same_task_matrecall", "specialized_model",
)
ALL_METHODS = (*CONTROL_METHODS, *CROSS_METHODS)


def _filter(rows: list[dict[str, Any]], ids: list[str], label: str) -> list[dict[str, Any]]:
    by_id = {row["case_id"]: row for row in rows}
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Missing {label} cases: {sorted(missing)[:5]}")
    return [by_id[case_id] for case_id in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--specialized-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.release / "manifest.json").read_text())
    completion = json.loads((args.run / "completion.json").read_text())
    if manifest.get("status") != "frozen" or manifest.get("release_id") != (
        "matrecall-cross-task-transfer"
    ):
        raise ValueError("Expected frozen cross-task release")
    if tuple(completion.get("methods", ())) != CROSS_METHODS:
        raise ValueError("Cross-task method coverage mismatch")
    if not str(completion.get("status", "")).startswith("completed"):
        raise ValueError("Cross-task predictions are incomplete")
    if completion.get("memory_writes") != 0 or completion.get("transfer_feedback_reveals") != 0:
        raise ValueError("Cross-task run violated frozen-memory policy")

    specialized_by_family = {
        family: read_jsonl(args.specialized_run / "matched" / family / "predictions.jsonl")
        for family in ("unified", "process")
    }
    hashes: dict[str, str] = {}
    for unit in manifest["tasks"]:
        pair_id = unit["pair_id"]
        source = args.release / unit["relative_path"]
        gold = read_jsonl(source / "gold.jsonl")
        ids = [row["case_id"] for row in gold]
        family, task = unit["family"], unit["target_task"]
        predictions: dict[str, list[dict[str, Any]]] = {}
        for method in CROSS_METHODS:
            predictions[method] = read_jsonl(
                args.run / "test" / method / family / pair_id / "predictions.jsonl"
            )
        for method in ("llm_only", "static_multiview_rag", "react_static_memory"):
            predictions[method] = _filter(read_jsonl(
                args.base_run / "evaluation" / "test" / method / family / task /
                "predictions.jsonl"
            ), ids, f"{method}/{task}")
        predictions["same_task_matrecall"] = _filter(read_jsonl(
            args.base_run / "evaluation" / "test" / "matrecall" / family /
            task / "predictions.jsonl"
        ), ids, f"same-task full/{task}")
        predictions["specialized_model"] = _filter(
            specialized_by_family[family], ids, f"specialized/{task}"
        )
        for method in ALL_METHODS:
            values = predictions[method]
            if [row["case_id"] for row in values] != ids:
                raise ValueError(f"Prediction/gold order mismatch: {method}/{pair_id}")
            base = args.output / pair_id / method
            write_jsonl(base / "predictions.jsonl", values)
            write_jsonl(base / "gold.jsonl", gold)
            hashes[f"{pair_id}/{method}/predictions"] = file_sha256(
                base / "predictions.jsonl"
            )
            hashes[f"{pair_id}/{method}/gold"] = file_sha256(base / "gold.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "completed",
        "pair_count": manifest["pair_count"],
        "target_task_count": manifest["target_task_count"],
        "cross_methods": list(CROSS_METHODS),
        "control_methods": list(CONTROL_METHODS),
        "all_methods": list(ALL_METHODS),
        "gold_joined_after_prediction_completion": True,
        "prediction_completion_sha256": file_sha256(args.run / "completion.json"),
        "files_sha256": hashes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
