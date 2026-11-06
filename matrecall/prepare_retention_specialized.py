"""Rebind frozen specialized predictions to the memory-retention cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matrecall.episodic_memory import digest, write_jsonl
from matrecall.run_online_adaptation import file_sha256, read_jsonl


CORE_FIELDS = (
    "protocol", "case_id", "task", "task_kind", "task_instruction", "query_visible",
)


def prepare(
    *, release: Path, original_unified: Path, original_process: Path,
    prediction_paths: list[Path], output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest.get("status") != "frozen" or manifest.get("task_count", 0) < 1:
        raise ValueError("Expected a non-empty frozen transfer release")
    old_cases, old_gold = {}, {}
    for base in (original_unified, original_process):
        for row in read_jsonl(base / "cases.jsonl"):
            old_cases[(row["task"], row["case_id"])] = row
        for row in read_jsonl(base / "gold.jsonl"):
            old_gold[(row["task"], row["case_id"])] = row
    predictions = {}
    for path in prediction_paths:
        for row in read_jsonl(path):
            key = (row["task"], row["case_id"])
            if key in predictions:
                raise ValueError(f"Duplicate specialized prediction: {key}")
            predictions[key] = row

    by_family = {
        "unified": {"predictions": [], "gold": []},
        "process": {"predictions": [], "gold": []},
    }
    verified = 0
    task_names = set()
    for task_meta in manifest["tasks"]:
        family = task_meta["family"]
        base = release / task_meta["relative_path"]
        cases = {row["case_id"]: row for row in read_jsonl(base / "cases.jsonl")}
        for gold in read_jsonl(base / "gold.jsonl"):
            key = (gold["task"], gold["case_id"])
            if key not in old_cases or key not in old_gold or key not in predictions:
                raise ValueError(f"Missing specialized source record: {key}")
            case, old_case, old_target = cases[gold["case_id"]], old_cases[key], old_gold[key]
            mismatched = [field for field in CORE_FIELDS if case.get(field) != old_case.get(field)]
            if mismatched:
                raise ValueError(f"Core query mismatch {key}: {mismatched}")
            if gold["target"] != old_target["target"]:
                raise ValueError(f"Target mismatch: {key}")
            source_prediction = predictions[key]
            source_sha = digest(old_case)
            if source_prediction.get("case_sha256") != source_sha:
                raise ValueError(f"Prediction/source hash mismatch: {key}")
            if old_target.get("case_sha256") != source_sha:
                raise ValueError(f"Gold/source hash mismatch: {key}")
            if gold.get("case_sha256") != digest(case):
                raise ValueError(f"Transfer gold/case hash mismatch: {key}")
            rebound = dict(source_prediction)
            rebound["source_case_sha256"] = source_prediction["case_sha256"]
            rebound["case_sha256"] = gold["case_sha256"]
            rebound["matched_transfer_rebinding"] = {
                "verified_core_fields": list(CORE_FIELDS),
                "verified_target": True,
                "source_release": (
                    "unified_evaluation"
                    if family == "unified" else "process_evaluation"
                ),
                "target_release": manifest["release_id"],
                "online_updates": False,
            }
            by_family[family]["predictions"].append(rebound)
            by_family[family]["gold"].append(gold)
            verified += 1
            task_names.add(gold["task"])
    for family, values in by_family.items():
        if values["predictions"]:
            write_jsonl(output / family / "predictions.jsonl", values["predictions"])
            write_jsonl(output / family / "gold.jsonl", values["gold"])
    payload = {
        "status": "verified",
        "release_id": manifest["release_id"],
        "task_count": len(task_names),
        "case_count": verified,
        "online_updates": False,
        "counts": {
            family: len(values["predictions"]) for family, values in by_family.items()
        },
        "source_prediction_sha256": {
            str(path): file_sha256(path) for path in prediction_paths
        },
        "core_query_field_mismatches": 0,
        "target_mismatches": 0,
        "source_binding_hash_errors": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--original-unified", type=Path, required=True)
    parser.add_argument("--original-process", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(
        release=args.release, original_unified=args.original_unified,
        original_process=args.original_process,
        prediction_paths=args.predictions, output=args.output,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
