"""Build the frozen multi-order prequential memory-ablation suite.

The sampled cohort is fixed once and then reused across every order seed.  Two
stream regimes are emitted: IID hash order and leakage-group-blocked order.
Grouping uses query/source metadata only; target values never define groups.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SUITE_PROTOCOL = "matrecall-memory-ablation"
SEQUENCE_REVISION = "multiseed-group-aware"
# Fixed hash salt of the released sampling; changing it changes which cases are selected.
SAMPLING_SALT = "2026-10-28-multiseed-group-aware-v1"
FAMILIES = ("unified", "process")
REGIMES = ("iid", "group_blocked")


def stream_task_relative_path(family: str, task: str) -> Path:
    """Return the task path relative to one stream root."""
    return Path("sequences") / family / task


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


def _label_key(row: dict[str, Any]) -> str:
    target = row.get("target", {})
    if isinstance(target, dict) and "value" in target:
        return canonical_json(target["value"])
    return canonical_json(target)


def select_fixed_cohort(
    rows: list[dict[str, Any]], *, task: str, maximum: int, sample_seed: int
) -> list[dict[str, Any]]:
    """Select one cohort independently of all later stream-order seeds."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate case IDs in {task}")
    if len(rows) <= maximum:
        return sorted(rows, key=lambda row: row["case_id"])
    selected: list[dict[str, Any]] = []
    if "classification" in str(rows[0].get("task_kind", "")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_label_key(row)].append(row)
        for label, values in sorted(groups.items()):
            values.sort(
                key=lambda row: stable_key(
                    sample_seed, f"sample:{task}:{label}", row["case_id"]
                )
            )
            if len(selected) < maximum:
                selected.append(values[0])
    chosen = {row["case_id"] for row in selected}
    remainder = sorted(
        (row for row in rows if row["case_id"] not in chosen),
        key=lambda row: stable_key(sample_seed, f"sample:{task}", row["case_id"]),
    )
    selected.extend(remainder[: maximum - len(selected)])
    return sorted(selected, key=lambda row: row["case_id"])


def _element_set(value: Any) -> str:
    text = canonical_json(value)
    symbols = sorted(set(re.findall(r"[A-Z][a-z]?", text)))
    if not symbols:
        raise ValueError(f"No element symbols in formula payload: {text[:120]}")
    return "-".join(symbols)


def group_id_for_case(
    case: dict[str, Any], gold: dict[str, Any]
) -> tuple[str, str, bool]:
    """Return group ID, source field, and whether the grouping is a proxy."""
    task = case["task"]
    query = case.get("query_visible", {})
    metadata = case.get("query_metadata", {}) or gold.get("query_metadata", {}) or {}
    if task.startswith("inp_") or task.startswith("meltpool_") or task in {
        "mgt_reported_product", "raccuglia_crystallization_success",
    }:
        value = gold.get("group_id")
        if value in (None, ""):
            raise ValueError(f"Missing process leakage group: {case['case_id']}")
        return str(value), "gold.group_id", False
    if task == "synthesis_text_to_action_graph":
        value = metadata.get("document_id")
        if value in (None, ""):
            raise ValueError(f"Missing document ID: {case['case_id']}")
        return f"document:{value}", "query_metadata.document_id", False
    if task in {
        "target_and_precursors_to_heating_conditions",
        "target_and_precursors_to_operations_and_conditions",
    }:
        value = metadata.get("source_doi")
        if value in (None, ""):
            raise ValueError(f"Missing source DOI: {case['case_id']}")
        return f"doi:{str(value).lower()}", "query_metadata.source_doi", False
    if task == "target_composition_to_precursor_set":
        return (
            f"elements:{_element_set(query.get('target_formula'))}",
            "query_visible.target_formula.element_set",
            False,
        )
    if task == "target_property_to_formulation_composition":
        anchors = case.get("initial_evidence_ids", [])
        if not anchors:
            raise ValueError(f"Missing inverse-design anchor: {case['case_id']}")
        return f"anchor:{anchors[0]}", "initial_evidence_ids[0]", True
    if task == "zeolite_and_osda_to_synthesis_route_distribution":
        value = query.get("zeolite_code")
        if value in (None, ""):
            raise ValueError(f"Missing zeolite code: {case['case_id']}")
        return f"topology:{value}", "query_visible.zeolite_code", False
    chemical_system = metadata.get("chemical_system") or query.get("chemical_system")
    if chemical_system not in (None, ""):
        return f"chemical_system:{chemical_system}", "chemical_system", False
    for field in ("reduced_formula", "reported_formula", "target_formula", "target_formulas"):
        if query.get(field) not in (None, "", []):
            return (
                f"elements:{_element_set(query[field])}",
                f"query_visible.{field}.element_set",
                False,
            )
    raise ValueError(f"No auditable group key for {task}/{case['case_id']}")


def order_rows(
    rows: list[dict[str, Any]], *, task: str, regime: str, order_seed: int
) -> list[dict[str, Any]]:
    if regime == "iid":
        return sorted(
            rows,
            key=lambda row: stable_key(order_seed, f"iid-order:{task}", row["case_id"]),
        )
    if regime != "group_blocked":
        raise ValueError(f"Unknown regime: {regime}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_group_id"]].append(row)
    group_ids = sorted(
        grouped,
        key=lambda value: stable_key(order_seed, f"group-order:{task}", value),
    )
    ordered = []
    for group_id in group_ids:
        ordered.extend(sorted(
            grouped[group_id],
            key=lambda row: stable_key(
                order_seed, f"within-group:{task}:{group_id}", row["case_id"]
            ),
        ))
    return ordered


def build_suite(
    source: Path, output: Path, maximum: int, sample_seed: int,
    order_seeds: list[int],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen" or source_manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task static-memory source release")
    if len(set(order_seeds)) != len(order_seeds) or not order_seeds:
        raise ValueError("Order seeds must be non-empty and unique")

    source_hashes = {"manifest.json": file_sha256(source / "manifest.json")}
    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    case_maps: dict[str, dict[str, dict[str, Any]]] = {}
    group_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    source_counts: dict[tuple[str, str], int] = {}
    for family in FAMILIES:
        case_path = source / family / "test" / "cases.jsonl"
        gold_path = source / family / "test" / "gold.jsonl"
        source_hashes[f"{family}/test/cases.jsonl"] = file_sha256(case_path)
        source_hashes[f"{family}/test/gold.jsonl"] = file_sha256(gold_path)
        cases = read_jsonl(case_path)
        gold_rows = read_jsonl(gold_path)
        case_maps[family] = {row["case_id"]: row for row in cases}
        gold_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in gold_rows:
            gold_by_task[row["task"]].append(row)
        for task, rows in sorted(gold_by_task.items()):
            selected = select_fixed_cohort(
                rows, task=task, maximum=maximum, sample_seed=sample_seed
            )
            enriched = []
            sources: set[str] = set()
            proxy = False
            for gold in selected:
                case = case_maps[family][gold["case_id"]]
                group_id, group_source, is_proxy = group_id_for_case(case, gold)
                sources.add(group_source)
                proxy = proxy or is_proxy
                enriched.append({**gold, "sequence_group_id": group_id})
            groups = [row["sequence_group_id"] for row in enriched]
            cohorts[(family, task)] = enriched
            source_counts[(family, task)] = len(rows)
            group_metadata[(family, task)] = {
                "group_count": len(set(groups)),
                "singleton_group_count": sum(groups.count(value) == 1 for value in set(groups)),
                "group_sources": sorted(sources),
                "grouping_is_proxy": proxy,
            }

    stream_records = []
    for regime in REGIMES:
        for order_seed in order_seeds:
            task_records = []
            for family, task in sorted(cohorts):
                enriched = cohorts[(family, task)]
                ordered_gold = order_rows(
                    enriched, task=task, regime=regime, order_seed=order_seed
                )
                cases = [case_maps[family][row["case_id"]] for row in ordered_gold]
                clean_gold = [
                    {key: value for key, value in row.items() if key != "sequence_group_id"}
                    for row in ordered_gold
                ]
                stream_relative = stream_task_relative_path(family, task)
                suite_relative = (
                    Path("streams") / regime / f"seed-{order_seed}" / stream_relative
                )
                write_jsonl(output / suite_relative / "cases.jsonl", cases)
                write_jsonl(output / suite_relative / "gold.jsonl", clean_gold)
                seen_groups: set[str] = set()
                order_payload = []
                for step, row in enumerate(ordered_gold, start=1):
                    group_id = row["sequence_group_id"]
                    first = group_id not in seen_groups
                    order_payload.append({
                        "protocol": SUITE_PROTOCOL,
                        "sequence_revision": SEQUENCE_REVISION,
                        "sequence_id": f"{regime}:{family}:{task}:seed-{order_seed}",
                        "sequence_regime": regime,
                        "order_seed": order_seed,
                        "sequence_step": step,
                        "case_id": row["case_id"],
                        "task": task,
                        "family": family,
                        "sequence_group_id": group_id,
                        "group_first_occurrence": first,
                        "same_group_seen_before": not first,
                    })
                    seen_groups.add(group_id)
                write_jsonl(output / suite_relative / "order.jsonl", order_payload)
                group_info = group_metadata[(family, task)]
                task_records.append({
                    "family": family,
                    "task": task,
                    "task_kind": ordered_gold[0]["task_kind"],
                    "source_cases": source_counts[(family, task)],
                    "sequence_cases": len(ordered_gold),
                    "sequence_id": order_payload[0]["sequence_id"],
                    "relative_path": str(stream_relative),
                    "cases_sha256": file_sha256(output / suite_relative / "cases.jsonl"),
                    "gold_sha256": file_sha256(output / suite_relative / "gold.jsonl"),
                    "order_sha256": file_sha256(output / suite_relative / "order.jsonl"),
                    **group_info,
                })
            stream_root = output / "streams" / regime / f"seed-{order_seed}"
            stream_manifest = {
                "release_id": f"{SUITE_PROTOCOL}:{regime}:seed-{order_seed}",
                "suite_release_id": SUITE_PROTOCOL,
                "status": "frozen",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sequence_revision": SEQUENCE_REVISION,
                "sequence_regime": regime,
                "order_seed": order_seed,
                "sample_seed": sample_seed,
                "source_release_id": source_manifest["release_id"],
                "source_split": "official test",
                "task_count": len(task_records),
                "sequence_cases": sum(row["sequence_cases"] for row in task_records),
                "max_cases_per_task": maximum,
                "tasks": task_records,
                "protocol": {
                    "evaluation": "prequential test-then-learn",
                    "prediction_precedes_current_gold_reveal": True,
                    "feedback_available_only_to_later_steps": True,
                    "cohort_fixed_across_order_seeds": True,
                    "group_first_occurrence_has_no_prior_same_group_feedback": True,
                    "grouping_uses_no_target_values": True,
                },
            }
            (stream_root / "manifest.json").write_text(
                json.dumps(stream_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (stream_root / "completion.json").write_text(
                canonical_json({
                    "status": "completed",
                    "task_count": len(task_records),
                    "sequence_cases": stream_manifest["sequence_cases"],
                }) + "\n",
                encoding="utf-8",
            )
            stream_records.append({
                "regime": regime,
                "order_seed": order_seed,
                "relative_path": str(stream_root.relative_to(output)),
                "sequence_cases": stream_manifest["sequence_cases"],
                "manifest_sha256": file_sha256(stream_root / "manifest.json"),
            })

    cohort_signatures = {}
    for family, task in sorted(cohorts):
        ids = sorted(row["case_id"] for row in cohorts[(family, task)])
        cohort_signatures[f"{family}/{task}"] = hashlib.sha256(
            canonical_json(ids).encode()
        ).hexdigest()
    suite_manifest = {
        "release_id": SUITE_PROTOCOL,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_revision": SEQUENCE_REVISION,
        "source_release_id": source_manifest["release_id"],
        "source_split": "official test",
        "task_count": 22,
        "stream_count": len(stream_records),
        "regimes": list(REGIMES),
        "order_seeds": order_seeds,
        "sample_seed": sample_seed,
        "max_cases_per_task": maximum,
        "sequence_cases_per_stream": stream_records[0]["sequence_cases"],
        "streams": stream_records,
        "cohort_signatures": cohort_signatures,
        "source_sha256": source_hashes,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(suite_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "completion.json").write_text(
        canonical_json({
            "status": "completed",
            "task_count": 22,
            "stream_count": len(stream_records),
            "sequence_cases_per_stream": stream_records[0]["sequence_cases"],
        }) + "\n",
        encoding="utf-8",
    )
    return suite_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases-per-task", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--order-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    manifest = build_suite(
        args.source, args.output, args.max_cases_per_task,
        args.sample_seed, args.order_seeds,
    )
    print(canonical_json({
        "event": "complete",
        "tasks": manifest["task_count"],
        "streams": manifest["stream_count"],
        "cases_per_stream": manifest["sequence_cases_per_stream"],
    }), flush=True)


if __name__ == "__main__":
    main()
