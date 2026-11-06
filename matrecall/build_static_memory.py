"""Build the multiview static-memory (episodic-graph) release.

Retrieval policies are selected per task on validation labels.  The selected
policy is then frozen and applied to the unchanged unified and process test cohorts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matrecall.static_retrieval import (
    AGENT_PROMPT_REVISION,
    AGENT_PROTOCOL,
    FUSION_DEPTH,
    K_VALUES,
    JarvisStructureIndexes,
    base_views,
    build_graph_frames,
    canonical_json,
    digest,
    graph_overlap_paths,
    ranking_score_maps,
    retrieve_multiview,
)
from matrecall.process_evaluation import (
    PROTOCOL as PROCESS_PROTOCOL,
    TASK_INSTRUCTIONS as PROCESS_INSTRUCTIONS,
    TASK_ORDER as PROCESS_TASKS,
    class_labels,
    load_task as load_process_task,
    retrieval_predictions,
    score_values,
    selection_metric,
    target_values,
    task_dirs,
)
from matrecall.unified_evaluation import (
    PROTOCOL as UNIFIED_PROTOCOL,
    TASK_INSTRUCTIONS as UNIFIED_INSTRUCTIONS,
    TASK_KINDS as UNIFIED_KINDS,
    TASK_ORDER as UNIFIED_TASKS,
    envelope,
    file_sha256,
    gold_targets,
    load_task as load_unified_task,
    memory_input,
    memory_target,
    prediction_schema,
    primary_selection_score,
    query_input,
    transfer_prediction,
)


EXPANSION_K = 5


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _query_metadata(row: pd.Series) -> dict[str, Any]:
    fields = (
        "challenge_any", "challenge_new_reduced_formula", "challenge_new_chemical_system",
        "challenge_new_structure_family_proxy", "chemical_system", "document_id",
        "source_doi", "source_record_id", "leakage_group",
    )
    output: dict[str, Any] = {}
    for field in fields:
        if field not in row or pd.isna(row[field]):
            continue
        output[field] = bool(row[field]) if field.startswith("challenge_") else str(row[field])
    return output


def _episode_id(task: str, row: pd.Series, index: int) -> str:
    return str(row.get("episode_id", row.get("query_id", f"{task}:memory:{index}")))


def _reference(
    family: str,
    task: str,
    episode: pd.Series,
    target: Any,
    episode_id: str,
    view_trace: dict[str, dict[str, float | int]],
    paths: list[str],
) -> dict[str, Any]:
    primary = max((float(value["score"]) for value in view_trace.values()), default=0.0)
    if family == "unified":
        return {
            "evidence_id": episode_id,
            "similarity": primary,
            "input": memory_input(task, episode),
            "observed_output": target,
            "source_dataset": str(episode.get("source_dataset", "")),
            "retrieval_trace": view_trace,
            "graph_paths": paths,
        }
    return {
        "evidence_id": episode_id,
        "similarity": primary,
        "input_text": str(episode.get("input_text", "")),
        "input": json.loads(str(episode["input_json"])),
        "observed_outcome": {"value": target},
        "epistemic_status": str(episode.get("epistemic_status", "experimental")),
        "source_dataset": str(episode.get("source_dataset", "")),
        "retrieval_trace": view_trace,
        "graph_paths": paths,
    }


def _rank_trial(trial: dict[str, Any], views: tuple[str, ...]) -> tuple[Any, ...]:
    value = trial["selection_value"]
    oriented = -value if trial["larger_is_better"] else value
    order = (*views, "multiview_rrf")
    return oriented, order.index(trial["view"]), K_VALUES.index(trial["k"])


def _select_unified(
    task: str,
    memory: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    rankings: dict[str, list[list[tuple[int, float]]]],
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    targets = [memory_target(task, row) for _, row in memory.iterrows()]
    expected_map = gold_targets(task, labels)
    expected = [expected_map[str(value)] for value in queries["query_id"]]
    trials = []
    for view, rows in rankings.items():
        for k in K_VALUES:
            predictions = [
                transfer_prediction(
                    task,
                    [targets[index] for index, _ in matches[:k]],
                    [score for _, score in matches[:k]],
                )
                for matches in rows
            ]
            value, maximize = primary_selection_score(task, expected, predictions)
            trials.append({
                "view": view, "k": k, "selection_value": value,
                "larger_is_better": maximize,
            })
    usable = [trial for trial in trials if np.isfinite(trial["selection_value"])]
    if not usable:
        raise ValueError(f"No finite multiview validation score: {task}")
    views = base_views(task, "unified")
    selected = sorted(usable, key=lambda trial: _rank_trial(trial, views))[0]
    return selected, targets, {"expected_map": expected_map, "trials": trials}


def _select_process(
    task: str,
    memory: pd.DataFrame,
    labels: pd.DataFrame,
    kind: str,
    rankings: dict[str, list[list[tuple[int, float]]]],
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    targets = target_values(memory, kind)
    expected = target_values(labels, kind)
    metric, maximize = selection_metric(kind)
    trials = []
    for view, rows in rankings.items():
        for k in K_VALUES:
            predicted = retrieval_predictions(memory, [matches[:k] for matches in rows], kind)
            metrics = score_values(kind, expected, predicted)
            value = metrics.get(metric)
            trials.append({
                "view": view, "k": k, "selection_value": value,
                "larger_is_better": maximize, "metrics": metrics,
            })
    usable = [
        trial for trial in trials
        if trial["selection_value"] is not None and np.isfinite(trial["selection_value"])
    ]
    if not usable:
        raise ValueError(f"No finite multiview validation score: {task}")
    views = base_views(task, "process")
    selected = sorted(usable, key=lambda trial: _rank_trial(trial, views))[0]
    return selected, targets, {"trials": trials}


def _build_rows(
    *,
    family: str,
    task: str,
    memory: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    manifest: dict[str, Any],
    rankings: dict[str, list[list[tuple[int, float]]]],
    memory_entities: list[set[str]],
    query_entities: list[set[str]],
    selected: dict[str, Any],
    targets: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    selected_rows = rankings[selected["view"]]
    views = tuple(view for view in rankings if view != "multiview_rrf")
    ids = [_episode_id(task, row, index) for index, (_, row) in enumerate(memory.iterrows())]
    if family == "unified":
        expected_map = gold_targets(task, labels)
        schema = prediction_schema(task, memory)
    else:
        kind = manifest["task_kind"]
        expected_values = target_values(labels, kind)
        schema = None
        allowed = class_labels(memory, kind)

    for query_number, ((_, query), matches) in enumerate(zip(queries.iterrows(), selected_rows)):
        initial = matches[: int(selected["k"])]
        score_maps = ranking_score_maps(rankings, query_number)
        ordered_indices: list[int] = []
        for index, _ in initial:
            if index not in ordered_indices:
                ordered_indices.append(index)
        reference_views: dict[str, list[str]] = {}
        for view in (*views, "multiview_rrf"):
            view_indices = [index for index, _ in rankings[view][query_number][:EXPANSION_K]]
            reference_views[view] = [ids[index] for index in view_indices]
            for index in view_indices:
                if index not in ordered_indices:
                    ordered_indices.append(index)
        references = []
        for index in ordered_indices:
            episode = memory.iloc[index]
            references.append(_reference(
                family,
                task,
                episode,
                targets[index],
                ids[index],
                score_maps.get(index, {}),
                graph_overlap_paths(query_entities[query_number], memory_entities[index]),
            ))
        initial_ids = [ids[index] for index, _ in initial]
        if family == "unified":
            deterministic = transfer_prediction(
                task,
                [targets[index] for index, _ in initial],
                [score for _, score in initial],
            )
            query_id = str(query["query_id"])
            case = {
                "protocol": UNIFIED_PROTOCOL,
                "agent_protocol": AGENT_PROTOCOL,
                "prompt_revision": AGENT_PROMPT_REVISION,
                "case_id": query_id,
                "task": task,
                "task_kind": UNIFIED_KINDS[task],
                "task_instruction": UNIFIED_INSTRUCTIONS[task],
                "query_visible": query_input(task, query),
                "prediction_schema": schema,
                "retrieval": {**selected, "selection_split": "validation"},
                "initial_evidence_ids": initial_ids,
                "reference_views": reference_views,
                "references": references,
                "deterministic_tool_estimate": deterministic,
                "query_metadata": _query_metadata(query),
            }
            target = expected_map[query_id]
            gold_row = {
                "protocol": UNIFIED_PROTOCOL,
                "case_id": query_id,
                "task": task,
                "task_kind": UNIFIED_KINDS[task],
                "target": target,
                "query_metadata": case["query_metadata"],
            }
            retrieval_response = envelope(
                deterministic,
                initial_ids,
                "Deterministic transfer from the validation-selected domain-aware episode view.",
            )
        else:
            kind = manifest["task_kind"]
            deterministic = retrieval_predictions(memory, [initial], kind)[0]
            query_id = str(query["query_id"])
            case = {
                "protocol": PROCESS_PROTOCOL,
                "agent_protocol": AGENT_PROTOCOL,
                "prompt_revision": AGENT_PROMPT_REVISION,
                "case_id": query_id,
                "task": task,
                "task_kind": kind,
                "task_instruction": PROCESS_INSTRUCTIONS[task],
                "target_name": manifest["target_name"],
                "target_unit": manifest["target_unit"],
                "allowed_labels": allowed,
                "query_visible": {
                    "input_text": str(query["input_text"]),
                    "input": json.loads(str(query["input_json"])),
                    "protocol": str(query["protocol"]),
                },
                "retrieval": {**selected, "selection_split": "validation"},
                "initial_evidence_ids": initial_ids,
                "reference_views": reference_views,
                "references": references,
                "deterministic_tool_estimate": {"value": deterministic},
            }
            target = expected_values[query_number]
            gold_row = {
                "protocol": PROCESS_PROTOCOL,
                "case_id": query_id,
                "task": task,
                "task_kind": kind,
                "group_id": str(query["leakage_group"]),
                "source_record_id": str(labels.iloc[query_number]["source_record_id"]),
                "target": {"value": target},
            }
            retrieval_response = {
                "analysis_process": [{
                    "comparison": "Validation-selected domain-aware episode transfer.",
                    "evidence_ids": initial_ids,
                    "implication": "Deterministic estimate from train-only episodes.",
                }],
                "reasoning_results": {
                    "prediction": {"value": deterministic},
                    "evidence_ids": initial_ids,
                },
            }
        case_sha = digest(case)
        gold_row["case_sha256"] = case_sha
        retrieval_rows.append({
            "protocol": case["protocol"],
            "agent_protocol": AGENT_PROTOCOL,
            "case_id": query_id,
            "case_sha256": case_sha,
            "task": task,
            "mode": "retrieval_only_multiview",
            "model": None,
            "status": "success",
            "response": retrieval_response,
        })
        cases.append(case)
        gold.append(gold_row)
    return cases, gold, retrieval_rows


def _load_family_task(
    family: str, release: Path, task: str, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if family == "unified":
        return load_unified_task(release, task, split)
    directory = next(path for path in task_dirs(release) if path.name == task)
    return load_process_task(directory, split)


def _build_family(
    *,
    family: str,
    release: Path,
    tasks: tuple[str, ...],
    output: Path,
    jarvis_indexes: JarvisStructureIndexes,
) -> list[dict[str, Any]]:
    development: list[dict[str, Any]] = []
    split_rows = {
        split: {"cases": [], "gold": [], "retrieval_only": []}
        for split in ("validation", "test")
    }
    graph_root = output / "memory_graph" / family
    for task in tasks:
        print(canonical_json({"event": "develop", "family": family, "task": task}), flush=True)
        memory, validation_queries, validation_labels, manifest = _load_family_task(
            family, release, task, "validation"
        )
        rankings, memory_entities, query_entities = retrieve_multiview(
            task,
            family,
            memory,
            validation_queries,
            depth=FUSION_DEPTH,
            jarvis_indexes=jarvis_indexes if task in UNIFIED_TASKS[:5] else None,
        )
        if family == "unified":
            selected, targets, details = _select_unified(
                task, memory, validation_queries, validation_labels, rankings
            )
        else:
            selected, targets, details = _select_process(
                task, memory, validation_labels, manifest["task_kind"], rankings
            )
        development.append({
            "family": family,
            "task": task,
            "selected": selected,
            "candidate_views": list(rankings),
            "candidate_k": list(K_VALUES),
            "selection_split": "validation",
            "trials": details["trials"],
        })
        graph_nodes, graph_edges = build_graph_frames(
            task, family, memory, targets, memory_entities
        )
        graph_dir = graph_root / task
        graph_dir.mkdir(parents=True, exist_ok=False)
        graph_nodes.to_parquet(graph_dir / "nodes.parquet", index=False)
        graph_edges.to_parquet(graph_dir / "edges.parquet", index=False)
        (graph_dir / "manifest.json").write_text(canonical_json({
            "task": task,
            "family": family,
            "episodes": len(memory),
            "nodes": len(graph_nodes),
            "edges": len(graph_edges),
            "graph_role": "typed relations and provenance; similarity remains in external indexes",
        }) + "\n", encoding="utf-8")
        validation_built = _build_rows(
            family=family,
            task=task,
            memory=memory,
            queries=validation_queries,
            labels=validation_labels,
            manifest=manifest,
            rankings=rankings,
            memory_entities=memory_entities,
            query_entities=query_entities,
            selected=selected,
            targets=targets,
        )
        for name, rows in zip(("cases", "gold", "retrieval_only"), validation_built):
            split_rows["validation"][name].extend(rows)

        print(canonical_json({"event": "freeze-test", "family": family, "task": task}), flush=True)
        test_memory, test_queries, test_labels, test_manifest = _load_family_task(
            family, release, task, "test"
        )
        if len(test_memory) != len(memory):
            raise ValueError(f"Memory changed between splits: {family}/{task}")
        validation_memory_ids = [
            _episode_id(task, row, index) for index, (_, row) in enumerate(memory.iterrows())
        ]
        test_memory_ids = [
            _episode_id(task, row, index) for index, (_, row) in enumerate(test_memory.iterrows())
        ]
        if test_memory_ids != validation_memory_ids:
            raise ValueError(f"Memory order or identity changed between splits: {family}/{task}")
        test_rankings, test_entities, test_query_entities = retrieve_multiview(
            task,
            family,
            test_memory,
            test_queries,
            depth=FUSION_DEPTH,
            jarvis_indexes=jarvis_indexes if task in UNIFIED_TASKS[:5] else None,
        )
        test_built = _build_rows(
            family=family,
            task=task,
            memory=test_memory,
            queries=test_queries,
            labels=test_labels,
            manifest=test_manifest,
            rankings=test_rankings,
            memory_entities=test_entities,
            query_entities=test_query_entities,
            selected=selected,
            targets=targets,
        )
        for name, rows in zip(("cases", "gold", "retrieval_only"), test_built):
            split_rows["test"][name].extend(rows)

    for split, values in split_rows.items():
        for name, rows in values.items():
            _write_jsonl(output / family / split / f"{name}.jsonl", rows)
    return development


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-release", type=Path, required=True)
    parser.add_argument("--process-release", type=Path, required=True)
    parser.add_argument("--jarvis-indexes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    indexes = JarvisStructureIndexes(args.jarvis_indexes)
    development = []
    development.extend(_build_family(
        family="unified",
        release=args.unified_release,
        tasks=tuple(UNIFIED_TASKS),
        output=args.output,
        jarvis_indexes=indexes,
    ))
    development.extend(_build_family(
        family="process",
        release=args.process_release,
        tasks=tuple(PROCESS_TASKS),
        output=args.output,
        jarvis_indexes=indexes,
    ))
    (args.output / "retrieval_development.json").write_text(
        json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        str(path.relative_to(args.output)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows": sum(1 for _ in path.open(encoding="utf-8")) if path.suffix == ".jsonl" else None,
        }
        for path in sorted(args.output.rglob("*")) if path.is_file()
    }
    manifest = {
        "release_id": AGENT_PROTOCOL,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent_protocol": AGENT_PROTOCOL,
        "prompt_revision": AGENT_PROMPT_REVISION,
        "source_protocols": [UNIFIED_PROTOCOL, PROCESS_PROTOCOL],
        "task_count": len(UNIFIED_TASKS) + len(PROCESS_TASKS),
        "tasks": list(UNIFIED_TASKS) + list(PROCESS_TASKS),
        "retrieval": {
            "candidate_k": list(K_VALUES),
            "fusion_depth": FUSION_DEPTH,
            "selection": "per-task native primary metric on validation only",
            "test_retuning": False,
            "views": "task-routed lexical, composition, structure, numeric-process, graph-entity, and RRF",
        },
        "memory": {
            "source": "training episodes only",
            "representation": "typed episodic graph plus separate similarity indexes",
            "test_queries_written_to_memory": False,
        },
        "source_sha256": {
            "unified_manifest": file_sha256(args.unified_release / "manifest.json"),
            "process_manifest": file_sha256(args.process_release / "manifest.json"),
            "jarvis_index_manifest": file_sha256(args.jarvis_indexes / "manifest.json"),
        },
        "files": files,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "completion.json").write_text(canonical_json({
        "status": "completed", "tasks": manifest["task_count"],
        "test_cases": sum(files[f"{family}/test/cases.jsonl"]["rows"] for family in ("unified", "process")),
    }) + "\n", encoding="utf-8")
    print(canonical_json({"event": "complete", "tasks": manifest["task_count"]}), flush=True)


if __name__ == "__main__":
    main()

