"""Build a frozen related/unrelated cross-task transfer benchmark.

Targets are inherited from the memory-retention cohort.  For each target,
related and unrelated source memories are evaluated on the same target cases.
Cases overlapping either source exposure by an auditable shared group key or
an exact visible-query hash are removed before the release is frozen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from matrecall.static_retrieval import digest
from matrecall.prepare_memory_retention import (
    canonical_json,
    file_sha256,
    read_jsonl,
    write_jsonl,
)
from matrecall.prepare_memory_ablation import group_id_for_case


PROTOCOL = "matrecall-cross-task-transfer"
PAIR_PROTOCOL = "matrecall-cross-task-pairs"
SELECTION_REVISION = "related-unrelated-pairs"


def _by_case(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate case IDs in {label}")
    return result


def _namespace(group_id: str) -> str:
    prefix = group_id.split(":", 1)[0]
    return {
        "chemical_system": "chemical_system",
        "elements": "element_set",
        "doi": "doi",
        "document": "document",
        "topology": "topology",
        "anchor": "anchor",
    }.get(prefix, prefix)


def _source_snapshot(
    *, task: str, exposure_tasks: dict[str, dict[str, Any]],
    exposure_release: Path, exposure_run: Path,
) -> dict[str, Any]:
    meta = exposure_tasks[task]
    base = exposure_release / meta["relative_path"]
    cases = _by_case(read_jsonl(base / "cases.jsonl"), f"{task} exposure cases")
    gold = _by_case(read_jsonl(base / "gold.jsonl"), f"{task} exposure gold")
    if set(cases) != set(gold):
        raise ValueError(f"Exposure case/gold mismatch: {task}")
    memory_path = (
        exposure_run / "test" / "matrecall" / meta["family"] / task /
        "verified_feedback_memory.jsonl"
    )
    memory = read_jsonl(memory_path)
    if {row["case_id"] for row in memory} != set(cases):
        raise ValueError(f"Exposure memory alignment mismatch: {task}")
    groups = set()
    group_sources = set()
    namespaces = set()
    for case_id, case in cases.items():
        group_id, group_source, _ = group_id_for_case(case, gold[case_id])
        groups.add(group_id)
        group_sources.add(group_source)
        namespaces.add(_namespace(group_id))
    return {
        "task": task,
        "family": meta["family"],
        "cases": cases,
        "memory_path": memory_path,
        "memory_count": len(memory),
        "memory_sha256": file_sha256(memory_path),
        "groups": groups,
        "group_sources": sorted(group_sources),
        "group_namespaces": sorted(namespaces),
        "query_hashes": {digest(row["query_visible"]) for row in cases.values()},
    }


def jointly_disjoint_offsets(
    cases: list[dict[str, Any]], index: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]], namespace: str,
) -> tuple[list[int], int, int]:
    """Return a common target cohort disjoint from every source snapshot."""
    if len(cases) != len(index):
        raise ValueError("Target case/index length mismatch")
    eligible = []
    excluded_group = 0
    excluded_query = 0
    for offset, (case, target_index) in enumerate(zip(cases, index)):
        query_hash = digest(case["query_visible"])
        exact_query_overlap = any(
            query_hash in source["query_hashes"] for source in sources.values()
        )
        group_overlap = any(
            namespace in source["group_namespaces"]
            and target_index["transfer_group_id"] in source["groups"]
            for source in sources.values()
        )
        if exact_query_overlap:
            excluded_query += 1
        elif group_overlap:
            excluded_group += 1
        else:
            eligible.append(offset)
    return eligible, excluded_group, excluded_query


def build_release(
    *, target_release: Path, exposure_release: Path, exposure_run: Path,
    pair_config: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    target_manifest = json.loads((target_release / "manifest.json").read_text())
    exposure_manifest = json.loads((exposure_release / "manifest.json").read_text())
    exposure_completion = json.loads((exposure_run / "completion.json").read_text())
    config = json.loads(pair_config.read_text())
    if target_manifest.get("release_id") != (
        "matrecall-memory-retention"
    ) or target_manifest.get("status") != "frozen":
        raise ValueError("Expected the frozen memory-retention target release")
    if exposure_manifest.get("task_count") != 22 or exposure_manifest.get("status") != "frozen":
        raise ValueError("Expected the frozen 22-task online-adaptation exposure release")
    if not str(exposure_completion.get("status", "")).startswith("completed"):
        raise ValueError("The online-adaptation exposure run is incomplete")
    if config.get("protocol") != PAIR_PROTOCOL:
        raise ValueError("Unexpected cross-task pair configuration")

    target_tasks = {row["task"]: row for row in target_manifest["tasks"]}
    exposure_tasks = {row["task"]: row for row in exposure_manifest["tasks"]}
    registered = config.get("pairs", [])
    if not registered or len({row["target_task"] for row in registered}) != len(registered):
        raise ValueError("Cross-task targets must be non-empty and unique")
    named_tasks = {
        value
        for row in registered
        for value in (
            row["target_task"], row["related_source_task"], row["unrelated_source_task"]
        )
    }
    missing_targets = {row["target_task"] for row in registered} - set(target_tasks)
    missing_sources = named_tasks - set(exposure_tasks) - set(target_tasks)
    if missing_targets or missing_sources:
        raise ValueError(
            f"Missing tasks; targets={sorted(missing_targets)}, sources={sorted(missing_sources)}"
        )

    snapshots = {
        task: _source_snapshot(
            task=task, exposure_tasks=exposure_tasks,
            exposure_release=exposure_release, exposure_run=exposure_run,
        )
        for task in sorted({
            value
            for row in registered
            for value in (row["related_source_task"], row["unrelated_source_task"])
        })
    }
    task_records = []
    target_records = []
    for pair in registered:
        target_task = pair["target_task"]
        target_meta = target_tasks[target_task]
        target_base = target_release / target_meta["relative_path"]
        cases = read_jsonl(target_base / "cases.jsonl")
        gold = read_jsonl(target_base / "gold.jsonl")
        index = read_jsonl(target_base / "transfer_index.jsonl")
        if not (len(cases) == len(gold) == len(index)):
            raise ValueError(f"Target alignment mismatch: {target_task}")
        sources = {
            "related": snapshots[pair["related_source_task"]],
            "unrelated": snapshots[pair["unrelated_source_task"]],
        }
        namespace = pair["group_namespace"]
        if namespace not in sources["related"]["group_namespaces"]:
            raise ValueError(
                f"Related source grouping is not comparable for {target_task}: {namespace}"
            )
        target_namespaces = {_namespace(row["transfer_group_id"]) for row in index}
        if target_namespaces != {namespace}:
            raise ValueError(
                f"Target grouping namespace mismatch for {target_task}: "
                f"{sorted(target_namespaces)} != {[namespace]}"
            )
        eligible_offsets, excluded_group, excluded_query = jointly_disjoint_offsets(
            cases, index, sources, namespace
        )
        if not eligible_offsets:
            raise ValueError(f"No jointly source-disjoint target cases: {target_task}")
        selected_case_ids = [cases[offset]["case_id"] for offset in eligible_offsets]
        target_case_set_sha = digest(selected_case_ids)
        target_records.append({
            **pair,
            "family": target_meta["family"],
            "source_target_cases": len(cases),
            "jointly_disjoint_target_cases": len(eligible_offsets),
            "excluded_exact_query_overlap": excluded_query,
            "excluded_source_group_overlap": excluded_group,
            "group_comparable_relations": sorted(
                relation for relation, source in sources.items()
                if namespace in source["group_namespaces"]
            ),
            "target_case_set_sha256": target_case_set_sha,
        })
        for relation, source in sources.items():
            source_task = pair[f"{relation}_source_task"]
            group_overlap_checked = namespace in source["group_namespaces"]
            pair_id = f"{source_task}__to__{target_task}__{relation}"
            relative = Path("pairs") / target_task / relation
            selected_cases = [cases[offset] for offset in eligible_offsets]
            selected_gold = [gold[offset] for offset in eligible_offsets]
            selected_index = []
            for transfer_index, offset in enumerate(eligible_offsets, start=1):
                row = index[offset]
                selected_index.append({
                    **row,
                    "protocol": PROTOCOL,
                    "transfer_index": transfer_index,
                    "pair_id": pair_id,
                    "target_task": target_task,
                    "source_task": source_task,
                    "memory_relation": relation,
                    "cross_task_exact_query_overlap": False,
                    "cross_task_group_overlap": (
                        False if group_overlap_checked else None
                    ),
                    "exposure_case_overlap": False,
                    "exposure_group_overlap": False,
                })
            write_jsonl(output / relative / "cases.jsonl", selected_cases)
            write_jsonl(output / relative / "gold.jsonl", selected_gold)
            write_jsonl(output / relative / "transfer_index.jsonl", selected_index)
            memory_target = output / relative / "frozen_exposure_memory.jsonl"
            memory_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source["memory_path"], memory_target)
            if file_sha256(memory_target) != source["memory_sha256"]:
                raise ValueError(f"Memory copy hash mismatch: {pair_id}")
            task_records.append({
                "pair_id": pair_id,
                "family": target_meta["family"],
                "task": target_task,
                "target_task": target_task,
                "source_task": source_task,
                "memory_relation": relation,
                "stratum": pair["stratum"],
                "rationale": pair["rationale"],
                "group_namespace": namespace,
                "task_kind": target_meta["task_kind"],
                "relative_path": str(relative),
                "transfer_cases": len(selected_cases),
                "transfer_groups": len({
                    selected_index_row["transfer_group_id"]
                    for selected_index_row in selected_index
                }),
                "memory_cases": source["memory_count"],
                "source_memory_family": source["family"],
                "source_memory_group_sources": source["group_sources"],
                "source_memory_group_namespaces": source["group_namespaces"],
                "cross_task_group_overlap_checked": group_overlap_checked,
                "target_case_set_sha256": target_case_set_sha,
                "frozen_memory_sha256": source["memory_sha256"],
                "cases_sha256": file_sha256(output / relative / "cases.jsonl"),
                "gold_sha256": file_sha256(output / relative / "gold.jsonl"),
                "transfer_index_sha256": file_sha256(
                    output / relative / "transfer_index.jsonl"
                ),
            })

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": PROTOCOL,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_revision": SELECTION_REVISION,
        "target_release_id": target_manifest["release_id"],
        "exposure_release_id": exposure_manifest["release_id"],
        "exposure_method": "matrecall",
        "target_task_count": len(target_records),
        "pair_count": len(task_records),
        "task_count": len(task_records),
        "unique_target_cases": sum(
            row["jointly_disjoint_target_cases"] for row in target_records
        ),
        "transfer_cases": sum(row["transfer_cases"] for row in task_records),
        "tasks": task_records,
        "target_pairs": target_records,
        "pair_config": config,
        "protocol": {
            "pairs_registered_before_prediction": True,
            "related_and_unrelated_use_identical_target_cases": True,
            "target_queries_have_no_exact_visible_query_match_in_either_source": True,
            "group_disjointness_is_enforced_when_source_and_target_share_the_registered_namespace": True,
            "noncomparable_domain_mismatched_controls_are_explicitly_marked": True,
            "memory_is_read_only": True,
            "transfer_feedback_reveals": 0,
            "selection_uses_no_target_values": True,
            "exploratory_cross_schema_pairs_are_separately_labelled": True,
        },
        "provenance": {
            "target_manifest_sha256": file_sha256(target_release / "manifest.json"),
            "exposure_manifest_sha256": file_sha256(exposure_release / "manifest.json"),
            "exposure_completion_sha256": file_sha256(exposure_run / "completion.json"),
            "pair_config_sha256": file_sha256(pair_config),
            "builder_sha256": file_sha256(Path(__file__)),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (output / "completion.json").write_text(canonical_json({
        "status": "completed",
        "target_task_count": manifest["target_task_count"],
        "pair_count": manifest["pair_count"],
        "unique_target_cases": manifest["unique_target_cases"],
        "transfer_cases": manifest["transfer_cases"],
    }) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-release", type=Path, required=True)
    parser.add_argument("--exposure-release", type=Path, required=True)
    parser.add_argument("--exposure-run", type=Path, required=True)
    parser.add_argument(
        "--pair-config", type=Path,
        default=Path(__file__).with_name("cross_task_pairs.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_release(
        target_release=args.target_release,
        exposure_release=args.exposure_release,
        exposure_run=args.exposure_run,
        pair_config=args.pair_config,
        output=args.output,
    )
    print(canonical_json({
        "event": "complete",
        "target_tasks": manifest["target_task_count"],
        "pairs": manifest["pair_count"],
        "unique_target_cases": manifest["unique_target_cases"],
        "transfer_cases": manifest["transfer_cases"],
    }))


if __name__ == "__main__":
    main()
