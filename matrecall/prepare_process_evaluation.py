"""Develop train-only episode retrieval on validation and freeze LLM cases."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrecall.process_evaluation import (
    PROTOCOL,
    PROMPT_REVISION,
    TASK_INSTRUCTIONS,
    class_labels,
    digest,
    load_task,
    read_jsonl,
    retrieval_predictions,
    retrieve,
    score_values,
    selection_metric,
    sha256_file,
    target_values,
    task_dirs,
    write_jsonl,
)


VIEWS = ("text_word", "text_char")
K_VALUES = (3, 5, 10)


def _json(value: str) -> Any:
    return json.loads(value)


def _prediction(value: Any, evidence_ids: list[str]) -> dict[str, Any]:
    return {
        "analysis_process": [
            {
                "comparison": "Similarity-weighted transfer from train-only episodes.",
                "evidence_ids": evidence_ids,
                "implication": "This is a deterministic retrieval estimate, not an observed query label.",
            }
        ],
        "reasoning_results": {
            "prediction": {"value": value},
            "evidence_ids": evidence_ids,
        },
    }


def develop_task(task_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    memory, queries, labels, manifest = load_task(task_dir, "validation")
    kind = manifest["task_kind"]
    expected = target_values(labels, kind)
    trials = []
    for view in VIEWS:
        for k in K_VALUES:
            neighbors = retrieve(memory, queries, view=view, k=k)
            predicted = retrieval_predictions(memory, neighbors, kind)
            metrics = score_values(kind, expected, predicted)
            trials.append({"view": view, "k": k, "metrics": metrics})
    metric, maximize = selection_metric(kind)
    usable = [trial for trial in trials if trial["metrics"].get(metric) is not None]
    if not usable:
        raise ValueError(f"No finite validation selection metric for {task_dir.name}")
    selected = sorted(
        usable,
        key=lambda trial: (
            -trial["metrics"][metric] if maximize else trial["metrics"][metric],
            VIEWS.index(trial["view"]),
            trial["k"],
        ),
    )[0]
    result = {
        "task": task_dir.name,
        "task_kind": kind,
        "selection_metric": metric,
        "larger_is_better": maximize,
        "selected": selected,
        "candidates": trials,
        "policy": "Select view and k on validation only; do not retune from test outcomes.",
    }
    return result, trials


def build_split(
    task_dir: Path,
    split: str,
    retrieval_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    memory, queries, labels, manifest = load_task(task_dir, split)
    kind = manifest["task_kind"]
    selected = retrieval_config["selected"]
    neighbors = retrieve(memory, queries, view=selected["view"], k=selected["k"])
    deterministic = retrieval_predictions(memory, neighbors, kind)
    memory_targets = target_values(memory, kind)
    allowed = class_labels(memory, kind)
    cases, gold, retrieval_rows = [], [], []
    for row_number, ((_, query), (_, label), matches, estimate) in enumerate(
        zip(queries.iterrows(), labels.iterrows(), neighbors, deterministic)
    ):
        references = []
        for rank, (index, similarity) in enumerate(matches, start=1):
            episode = memory.iloc[index]
            references.append(
                {
                    "evidence_id": str(episode["episode_id"]),
                    "rank": rank,
                    "similarity": similarity,
                    "input_text": str(episode["input_text"]),
                    "input": _json(str(episode["input_json"])),
                    "observed_outcome": {"value": memory_targets[index]},
                    "epistemic_status": str(episode["epistemic_status"]),
                    "source_dataset": str(episode["source_dataset"]),
                }
            )
        case = {
            "protocol": PROTOCOL,
            "prompt_revision": PROMPT_REVISION,
            "case_id": str(query["query_id"]),
            "task": task_dir.name,
            "task_kind": kind,
            "task_instruction": TASK_INSTRUCTIONS[task_dir.name],
            "target_name": manifest["target_name"],
            "target_unit": manifest["target_unit"],
            "allowed_labels": allowed,
            "query_visible": {
                "input_text": str(query["input_text"]),
                "input": _json(str(query["input_json"])),
                "protocol": str(query["protocol"]),
            },
            "retrieval": {
                "view": selected["view"],
                "k": selected["k"],
                "selection_split": "validation",
            },
            "references": references,
        }
        case_sha = digest(case)
        target = target_values(labels.iloc[[row_number]], kind)[0]
        target_row = {
            "protocol": PROTOCOL,
            "case_id": case["case_id"],
            "case_sha256": case_sha,
            "task": task_dir.name,
            "task_kind": kind,
            "group_id": str(query["leakage_group"]),
            "source_record_id": str(label["source_record_id"]),
            "target": {"value": target},
        }
        evidence_ids = [reference["evidence_id"] for reference in references]
        retrieval_rows.append(
            {
                "protocol": PROTOCOL,
                "case_id": case["case_id"],
                "case_sha256": case_sha,
                "task": task_dir.name,
                "mode": "retrieval_only",
                "model": None,
                "status": "success",
                "response": _prediction(estimate, evidence_ids),
            }
        )
        cases.append(case)
        gold.append(target_row)
    return cases, gold, retrieval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_manifest = json.loads((args.release / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("release_id") != "matrecall-process-tasks":
        raise ValueError(f"Unexpected source release: {source_manifest.get('release_id')}")

    args.output.mkdir(parents=True)
    development = []
    configs: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs(args.release):
        selected, _ = develop_task(task_dir)
        development.append(selected)
        configs[task_dir.name] = selected
    (args.output / "retrieval_development.json").write_text(
        json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for split in ("validation", "test"):
        split_cases, split_gold, split_retrieval = [], [], []
        for task_dir in task_dirs(args.release):
            cases, gold, retrieval_rows = build_split(task_dir, split, configs[task_dir.name])
            split_cases.extend(cases)
            split_gold.extend(gold)
            split_retrieval.extend(retrieval_rows)
        split_dir = args.output / split
        write_jsonl(split_dir / "cases.jsonl", split_cases)
        write_jsonl(split_dir / "gold.jsonl", split_gold)
        write_jsonl(split_dir / "retrieval_only.jsonl", split_retrieval)

    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(read_jsonl(path)) if path.suffix == ".jsonl" else None,
            }
    manifest = {
        "protocol": PROTOCOL,
        "prompt_revision": PROMPT_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_release": str(args.release),
        "source_manifest_sha256": sha256_file(args.release / "manifest.json"),
        "retrieval_policy": {
            "candidate_views": list(VIEWS),
            "candidate_k": list(K_VALUES),
            "selection": "per task on private validation labels only",
            "memory": "training episodes only",
            "test_retuning": False,
        },
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "tasks": len(configs), "files": len(files)}))


if __name__ == "__main__":
    main()
