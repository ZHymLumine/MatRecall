"""Build a group-disjoint frozen-memory transfer benchmark from online adaptation.

The exposure memory is copied from the completed online-adaptation MatRecall run.
Transfer cases are selected from the remaining official-test pool only when
their leakage group is absent from exposure.  Selection uses query/source
metadata and stable hashes; target values never define or order the cohort.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from matrecall.prepare_memory_ablation import group_id_for_case


PROTOCOL = "matrecall-memory-retention"
SELECTION_REVISION = "group-disjoint-retention"
# Fixed hash salt of the released sampling; changing it changes which cases are selected.
SAMPLING_SALT = "2026-10-28-v04-group-disjoint-frozen-transfer-v1"
FAMILIES = ("unified", "process")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{SAMPLING_SALT}:{seed}:{namespace}:{value}".encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def select_group_diverse_rows(
    eligible: list[dict[str, Any]], *, task: str, maximum: int, seed: int
) -> list[dict[str, Any]]:
    """Select deterministically, taking one row per group before repeats."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row["transfer_group_id"]].append(row)
    first_pass = []
    remainder = []
    for group_id, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda row: stable_key(
                seed, f"within-group:{task}:{group_id}", row["case_id"]
            ),
        )
        first_pass.append(ranked[0])
        remainder.extend(ranked[1:])
    first_pass.sort(
        key=lambda row: stable_key(
            seed, f"group-sample:{task}", row["transfer_group_id"]
        )
    )
    chosen = first_pass[:maximum]
    if len(chosen) < maximum:
        remainder.sort(
            key=lambda row: stable_key(seed, f"remainder:{task}", row["case_id"])
        )
        chosen.extend(remainder[: maximum - len(chosen)])
    return sorted(chosen, key=lambda row: stable_key(seed, f"order:{task}", row["case_id"]))


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate case IDs in {label}")
    return indexed


def build_release(
    *, source: Path, exposure_release: Path, exposure_run: Path, output: Path,
    maximum: int, seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    exposure_manifest = json.loads(
        (exposure_release / "manifest.json").read_text(encoding="utf-8")
    )
    exposure_completion = json.loads(
        (exposure_run / "completion.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "frozen" or source_manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task static-memory source release")
    if exposure_manifest.get("status") != "frozen" or exposure_manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task online-adaptation exposure release")
    if not str(exposure_completion.get("status", "")).startswith("completed"):
        raise ValueError("The online-adaptation exposure run is incomplete")
    if "matrecall" not in exposure_completion.get("methods", []):
        raise ValueError("The online-adaptation exposure run lacks MatRecall")

    exposure_tasks = {
        (row["family"], row["task"]): row for row in exposure_manifest["tasks"]
    }
    task_records = []
    excluded_tasks = []
    source_hashes = {"manifest.json": file_sha256(source / "manifest.json")}
    memory_hashes = {}

    for family in FAMILIES:
        case_path = source / family / "test" / "cases.jsonl"
        gold_path = source / family / "test" / "gold.jsonl"
        source_hashes[f"{family}/test/cases.jsonl"] = file_sha256(case_path)
        source_hashes[f"{family}/test/gold.jsonl"] = file_sha256(gold_path)
        cases = _index(read_jsonl(case_path), f"{family} source cases")
        gold = _index(read_jsonl(gold_path), f"{family} source gold")
        if set(cases) != set(gold):
            raise ValueError(f"Source case/gold mismatch: {family}")
        task_names = sorted({row["task"] for row in gold.values()})
        for task in task_names:
            exposure_meta = exposure_tasks[(family, task)]
            exposure_base = exposure_release / exposure_meta["relative_path"]
            exposure_cases = read_jsonl(exposure_base / "cases.jsonl")
            exposure_ids = {row["case_id"] for row in exposure_cases}
            if not exposure_ids <= cases.keys():
                raise ValueError(f"Exposure cases missing from source: {task}")

            memory_source = (
                exposure_run / "test" / "matrecall" / family / task /
                "verified_feedback_memory.jsonl"
            )
            memory_rows = read_jsonl(memory_source)
            if len(memory_rows) != len(exposure_ids):
                raise ValueError(f"Exposure memory count mismatch: {task}")
            if {row["case_id"] for row in memory_rows} != exposure_ids:
                raise ValueError(f"Exposure memory case mismatch: {task}")
            if any(
                row.get("verified_after_prediction") is not True
                or row.get("epistemic_status") != "verified_feedback"
                for row in memory_rows
            ):
                raise ValueError(f"Unverified exposure memory: {task}")

            task_ids = [case_id for case_id, row in gold.items() if row["task"] == task]
            group_info = {
                case_id: group_id_for_case(cases[case_id], gold[case_id])
                for case_id in task_ids
            }
            exposure_groups = {group_info[case_id][0] for case_id in exposure_ids}
            eligible = []
            for case_id in task_ids:
                group_id, group_source, proxy = group_info[case_id]
                if case_id in exposure_ids or group_id in exposure_groups:
                    continue
                eligible.append({
                    "case_id": case_id,
                    "transfer_group_id": group_id,
                    "group_source": group_source,
                    "grouping_is_proxy": proxy,
                })
            if not eligible:
                excluded_tasks.append({
                    "family": family,
                    "task": task,
                    "reason": "no_group_disjoint_cases_after_exposure",
                    "source_cases": len(task_ids),
                    "exposure_cases": len(exposure_ids),
                    "exposure_groups": len(exposure_groups),
                })
                continue

            selected = select_group_diverse_rows(
                eligible, task=task, maximum=maximum, seed=seed
            )
            selected_ids = [row["case_id"] for row in selected]
            selected_groups = {row["transfer_group_id"] for row in selected}
            if set(selected_ids) & exposure_ids:
                raise AssertionError(f"Case overlap survived selection: {task}")
            if selected_groups & exposure_groups:
                raise AssertionError(f"Group overlap survived selection: {task}")

            relative = Path("tasks") / family / task
            selected_cases = [cases[case_id] for case_id in selected_ids]
            selected_gold = [gold[case_id] for case_id in selected_ids]
            transfer_index = [
                {
                    "protocol": PROTOCOL,
                    "transfer_index": index,
                    "case_id": row["case_id"],
                    "task": task,
                    "family": family,
                    "transfer_group_id": row["transfer_group_id"],
                    "group_source": row["group_source"],
                    "grouping_is_proxy": row["grouping_is_proxy"],
                    "exposure_case_overlap": False,
                    "exposure_group_overlap": False,
                }
                for index, row in enumerate(selected, start=1)
            ]
            write_jsonl(output / relative / "cases.jsonl", selected_cases)
            write_jsonl(output / relative / "gold.jsonl", selected_gold)
            write_jsonl(output / relative / "transfer_index.jsonl", transfer_index)
            memory_target = output / relative / "frozen_exposure_memory.jsonl"
            memory_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(memory_source, memory_target)
            memory_hash = file_sha256(memory_target)
            if memory_hash != file_sha256(memory_source):
                raise ValueError(f"Memory snapshot copy hash mismatch: {task}")
            memory_hashes[f"{family}/{task}"] = memory_hash
            grouping_is_proxy = any(row["grouping_is_proxy"] for row in selected)
            task_records.append({
                "family": family,
                "task": task,
                "task_kind": selected_gold[0]["task_kind"],
                "relative_path": str(relative),
                "source_cases": len(task_ids),
                "exposure_cases": len(exposure_ids),
                "exposure_groups": len(exposure_groups),
                "eligible_group_disjoint_cases": len(eligible),
                "eligible_group_disjoint_groups": len({
                    row["transfer_group_id"] for row in eligible
                }),
                "transfer_cases": len(selected),
                "transfer_groups": len(selected_groups),
                "group_sources": sorted({row["group_source"] for row in selected}),
                "grouping_is_proxy": grouping_is_proxy,
                "primary_strict_group_disjoint": not grouping_is_proxy,
                "low_sample_transfer": len(selected) < 10,
                "case_overlap_count": 0,
                "group_overlap_count": 0,
                "cases_sha256": file_sha256(output / relative / "cases.jsonl"),
                "gold_sha256": file_sha256(output / relative / "gold.jsonl"),
                "transfer_index_sha256": file_sha256(
                    output / relative / "transfer_index.jsonl"
                ),
                "frozen_memory_sha256": memory_hash,
            })

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": PROTOCOL,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_revision": SELECTION_REVISION,
        "source_release_id": source_manifest["release_id"],
        "source_split": "official test",
        "exposure_release_id": exposure_manifest["release_id"],
        "exposure_method": "matrecall",
        "task_count": len(task_records),
        "source_task_count": 22,
        "excluded_task_count": len(excluded_tasks),
        "transfer_cases": sum(row["transfer_cases"] for row in task_records),
        "strict_primary_task_count": sum(
            row["primary_strict_group_disjoint"] for row in task_records
        ),
        "max_cases_per_task": maximum,
        "selection_seed": seed,
        "tasks": task_records,
        "excluded_tasks": excluded_tasks,
        "protocol": {
            "exposure_memory_is_complete_snapshot": True,
            "transfer_case_ids_disjoint_from_exposure": True,
            "transfer_groups_disjoint_from_exposure": True,
            "all_transfer_queries_share_one_fixed_task_memory_snapshot": True,
            "no_transfer_feedback_or_memory_writes": True,
            "selection_uses_no_target_values": True,
            "proxy_group_tasks_are_supplemental": True,
        },
        "source_sha256": source_hashes,
        "exposure_release_manifest_sha256": file_sha256(
            exposure_release / "manifest.json"
        ),
        "exposure_run_completion_sha256": file_sha256(
            exposure_run / "completion.json"
        ),
        "memory_sha256": memory_hashes,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "completion.json").write_text(
        canonical_json({
            "status": "completed",
            "task_count": manifest["task_count"],
            "transfer_cases": manifest["transfer_cases"],
            "excluded_task_count": manifest["excluded_task_count"],
        }) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exposure-release", type=Path, required=True)
    parser.add_argument("--exposure-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases-per-task", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build_release(
        source=args.source,
        exposure_release=args.exposure_release,
        exposure_run=args.exposure_run,
        output=args.output,
        maximum=args.max_cases_per_task,
        seed=args.seed,
    )
    print(canonical_json({
        "event": "complete",
        "task_count": manifest["task_count"],
        "transfer_cases": manifest["transfer_cases"],
        "excluded_task_count": manifest["excluded_task_count"],
    }), flush=True)


if __name__ == "__main__":
    main()
