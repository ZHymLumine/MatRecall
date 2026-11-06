"""Materialize prediction/gold subsets for memory-ablation native-metric rescoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotate_prior_same_group_counts(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Count completed queries from the same sequence group before each row."""
    seen: dict[tuple[str, str], int] = {}
    annotated = []
    previous_step: dict[str, int] = {}
    for row in predictions:
        task = str(row["task"])
        step = int(row["sequence_step"])
        if step <= previous_step.get(task, 0):
            raise ValueError(
                f"Predictions are not in strictly increasing sequence order: {task}"
            )
        previous_step[task] = step
        group = str(row["sequence_group_id"])
        key = (task, group)
        prior = seen.get(key, 0)
        expected_first = prior == 0
        if row.get("group_first_occurrence") is not expected_first:
            raise ValueError(
                f"Group-first flag disagrees with sequence order: {row['case_id']}"
            )
        annotated.append({**row, "prior_same_group_count": prior})
        seen[key] = prior + 1
    return annotated


def build_subset(run: Path, output: Path, methods: list[str], subset: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if subset not in ("group_first", "group_later"):
        raise ValueError(f"Unsupported subset: {subset}")
    records = []
    for method in methods:
        for family in ("unified", "process"):
            source = run / "combined" / method / family
            predictions = read_jsonl(source / "predictions.jsonl")
            gold_rows = read_jsonl(source / "gold.jsonl")
            gold_by_id = {row["case_id"]: row for row in gold_rows}
            annotated = annotate_prior_same_group_counts(predictions)
            selected = [
                row
                for row in annotated
                if (row["prior_same_group_count"] == 0) == (subset == "group_first")
            ]
            if not selected:
                raise ValueError(f"No {subset} rows for {method}/{family}")
            if len({row["case_id"] for row in selected}) != len(selected):
                raise ValueError(f"Duplicate selected IDs for {method}/{family}")
            missing = [row["case_id"] for row in selected if row["case_id"] not in gold_by_id]
            if missing:
                raise ValueError(f"Missing subset gold: {missing[:3]}")
            target = output / method / family
            write_jsonl(target / "predictions.jsonl", selected)
            write_jsonl(
                target / "gold.jsonl", [gold_by_id[row["case_id"]] for row in selected]
            )
            records.append({
                "method": method,
                "family": family,
                "rows": len(selected),
                "predictions_sha256": file_sha256(target / "predictions.jsonl"),
                "gold_sha256": file_sha256(target / "gold.jsonl"),
            })
    payload = {"status": "completed", "subset": subset, "records": records}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--subset", choices=("group_first", "group_later"), default="group_first"
    )
    args = parser.parse_args()
    payload = build_subset(args.run, args.output, args.methods, args.subset)
    print(canonical_json({
        "event": "complete", "subset": args.subset,
        "records": len(payload["records"]),
    }))


if __name__ == "__main__":
    main()
