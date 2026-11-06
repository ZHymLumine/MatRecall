"""Build a sampled, leakage-ordered prequential benchmark from the static-memory release.

Each task becomes one deterministic sequence.  A case is predicted before its
gold row may be exposed to a method.  The builder keeps the original case and
gold checksums unchanged so that the existing native evaluators remain usable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SEQUENTIAL_PROTOCOL = "matrecall-online-adaptation"
SEQUENCE_REVISION = "prequential-sequence"
# Fixed hash salt of the released sampling; changing it changes which cases are selected.
SAMPLING_SALT = "2026-10-27-prequential-sequence-v1"
FAMILIES = ("unified", "process")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, case_id: str) -> str:
    return hashlib.sha256(f"{SAMPLING_SALT}:{seed}:{namespace}:{case_id}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line: {path}:{number}")
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _label_key(row: dict[str, Any]) -> str:
    target = row.get("target", {})
    if isinstance(target, dict) and "value" in target:
        return canonical_json(target["value"])
    return canonical_json(target)


def select_gold_rows(
    rows: list[dict[str, Any]], *, task: str, maximum: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic sample and return it in deterministic stream order.

    Classification tasks receive at least one row per observed class when the
    budget permits.  Other tasks use a hash sample, which does not inspect the
    target and therefore preserves the source distribution in expectation.
    """
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate case IDs in {task}")
    if len(rows) <= maximum:
        selected = list(rows)
    elif "classification" in str(rows[0].get("task_kind", "")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_label_key(row)].append(row)
        selected = []
        for label, values in sorted(groups.items()):
            values.sort(key=lambda row: stable_key(seed, f"sample:{task}:{label}", row["case_id"]))
            if len(selected) < maximum:
                selected.append(values[0])
        chosen = {row["case_id"] for row in selected}
        remainder = sorted(
            (row for row in rows if row["case_id"] not in chosen),
            key=lambda row: stable_key(seed, f"sample:{task}", row["case_id"]),
        )
        selected.extend(remainder[: maximum - len(selected)])
    else:
        selected = sorted(
            rows, key=lambda row: stable_key(seed, f"sample:{task}", row["case_id"])
        )[:maximum]
    return sorted(
        selected, key=lambda row: stable_key(seed, f"order:{task}", row["case_id"])
    )


def _load_selected_cases(path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line: {path}:{number}")
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id in selected_ids:
                if case_id in cases:
                    raise ValueError(f"Duplicate selected case: {case_id}")
                cases[case_id] = row
    missing = sorted(selected_ids - cases.keys())
    if missing:
        raise ValueError(f"Selected cases missing from source: {missing[:3]}")
    return cases


def build_release(source: Path, output: Path, maximum: int, seed: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen" or source_manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task static-memory source release")

    task_records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {"manifest.json": file_sha256(source / "manifest.json")}
    for family in FAMILIES:
        case_path = source / family / "test" / "cases.jsonl"
        gold_path = source / family / "test" / "gold.jsonl"
        source_hashes[f"{family}/test/cases.jsonl"] = file_sha256(case_path)
        source_hashes[f"{family}/test/gold.jsonl"] = file_sha256(gold_path)
        gold_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(gold_path):
            gold_by_task[row["task"]].append(row)
        selected_by_task = {
            task: select_gold_rows(rows, task=task, maximum=maximum, seed=seed)
            for task, rows in sorted(gold_by_task.items())
        }
        selected_ids = {
            row["case_id"] for rows in selected_by_task.values() for row in rows
        }
        selected_cases = _load_selected_cases(case_path, selected_ids)
        for task, gold_rows in selected_by_task.items():
            cases = [selected_cases[row["case_id"]] for row in gold_rows]
            if any(case["task"] != task for case in cases):
                raise ValueError(f"Task mismatch in selected source cases: {task}")
            relative = Path("sequences") / family / task
            write_jsonl(output / relative / "cases.jsonl", cases)
            write_jsonl(output / relative / "gold.jsonl", gold_rows)
            order_rows = [
                {
                    "protocol": SEQUENTIAL_PROTOCOL,
                    "sequence_revision": SEQUENCE_REVISION,
                    "sequence_id": f"{family}:{task}:seed-{seed}",
                    "sequence_step": step,
                    "case_id": row["case_id"],
                    "task": task,
                    "family": family,
                }
                for step, row in enumerate(gold_rows, start=1)
            ]
            write_jsonl(output / relative / "order.jsonl", order_rows)
            task_records.append({
                "family": family,
                "task": task,
                "task_kind": gold_rows[0]["task_kind"],
                "source_cases": len(gold_by_task[task]),
                "sequence_cases": len(gold_rows),
                "sequence_id": order_rows[0]["sequence_id"],
                "relative_path": str(relative),
                "cases_sha256": file_sha256(output / relative / "cases.jsonl"),
                "gold_sha256": file_sha256(output / relative / "gold.jsonl"),
                "order_sha256": file_sha256(output / relative / "order.jsonl"),
            })

    manifest = {
        "release_id": SEQUENTIAL_PROTOCOL,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_revision": SEQUENCE_REVISION,
        "source_release_id": source_manifest["release_id"],
        "source_split": "official test",
        "task_count": len(task_records),
        "sequence_cases": sum(row["sequence_cases"] for row in task_records),
        "max_cases_per_task": maximum,
        "sampling_seed": seed,
        "tasks": task_records,
        "protocol": {
            "evaluation": "prequential test-then-learn",
            "prediction_precedes_current_gold_reveal": True,
            "feedback_available_only_to_later_steps": True,
            "sequences_are_task_local": True,
            "source_cases_unchanged": True,
        },
        "source_sha256": source_hashes,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion = {
        "status": "completed",
        "task_count": manifest["task_count"],
        "sequence_cases": manifest["sequence_cases"],
    }
    (output / "completion.json").write_text(canonical_json(completion) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases-per-task", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build_release(args.source, args.output, args.max_cases_per_task, args.seed)
    print(canonical_json({
        "event": "complete",
        "task_count": manifest["task_count"],
        "sequence_cases": manifest["sequence_cases"],
    }), flush=True)


if __name__ == "__main__":
    main()
