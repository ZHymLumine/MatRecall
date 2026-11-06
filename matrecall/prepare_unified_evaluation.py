"""Select train-only retrieval on validation and freeze unified-task evaluation cases."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from matrecall.unified_evaluation import (
    PROTOCOL,
    PROMPT_REVISION,
    TASK_INSTRUCTIONS,
    TASK_KINDS,
    TASK_ORDER,
    canonical_json,
    digest,
    envelope,
    file_sha256,
    gold_targets,
    load_task,
    memory_input,
    memory_target,
    prediction_schema,
    primary_selection_score,
    query_input,
    retrieval_document,
    transfer_prediction,
    write_jsonl,
)


VIEWS = ("text_word", "text_char")
K_VALUES = (3, 5, 10)


def retrieve(
    memory_documents: list[str], query_documents: list[str], *, view: str, k: int
) -> list[list[tuple[int, float]]]:
    kwargs = (
        {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": 1,
            "sublinear_tf": True,
            "max_features": 120000,
            "dtype": np.float32,
        }
        if view == "text_char"
        else {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": 1,
            "sublinear_tf": True,
            "max_features": 120000,
            "dtype": np.float32,
        }
    )
    vectorizer = TfidfVectorizer(**kwargs)
    memory_vectors = vectorizer.fit_transform(memory_documents)
    query_vectors = vectorizer.transform(query_documents)
    count = min(k, len(memory_documents))
    model = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=4)
    model.fit(memory_vectors)
    output: list[list[tuple[int, float]]] = []
    for start in range(0, len(query_documents), 256):
        distances, indices = model.kneighbors(query_vectors[start : start + 256])
        for row_indices, row_distances in zip(indices, distances):
            output.append(
                [
                    (int(index), float(max(0.0, 1.0 - distance)))
                    for index, distance in zip(row_indices, row_distances)
                ]
            )
    if len(output) != len(query_documents):
        raise ValueError("Retrieval output count mismatch")
    return output


def develop_task(release: Path, task: str) -> dict[str, Any]:
    memory, queries, labels, _ = load_task(release, task, "validation")
    expected_map = gold_targets(task, labels)
    expected = [expected_map[str(value)] for value in queries["query_id"]]
    memory_documents = [
        retrieval_document(task, row, query=False) for _, row in memory.iterrows()
    ]
    query_documents = [
        retrieval_document(task, row, query=True) for _, row in queries.iterrows()
    ]
    targets = [memory_target(task, row) for _, row in memory.iterrows()]
    trials = []
    for view in VIEWS:
        full_neighbors = retrieve(
            memory_documents, query_documents, view=view, k=max(K_VALUES)
        )
        for k in K_VALUES:
            predictions = []
            for matches in full_neighbors:
                selected = matches[: min(k, len(matches))]
                predictions.append(
                    transfer_prediction(
                        task,
                        [targets[index] for index, _ in selected],
                        [similarity for _, similarity in selected],
                    )
                )
            value, maximize = primary_selection_score(task, expected, predictions)
            trials.append(
                {
                    "view": view,
                    "k": k,
                    "selection_value": value,
                    "larger_is_better": maximize,
                }
            )
    usable = [trial for trial in trials if np.isfinite(trial["selection_value"])]
    if not usable:
        raise ValueError(f"No finite validation retrieval result: {task}")
    selected = sorted(
        usable,
        key=lambda trial: (
            -trial["selection_value"]
            if trial["larger_is_better"]
            else trial["selection_value"],
            VIEWS.index(trial["view"]),
            K_VALUES.index(trial["k"]),
        ),
    )[0]
    return {
        "task": task,
        "task_kind": TASK_KINDS[task],
        "selected": selected,
        "candidates": trials,
        "selection_split": "validation",
        "policy": "Select TF-IDF view and k on validation labels only; freeze before test.",
    }


def _query_metadata(row: Any) -> dict[str, Any]:
    fields = (
        "challenge_any",
        "challenge_new_reduced_formula",
        "challenge_new_chemical_system",
        "challenge_new_structure_family_proxy",
        "chemical_system",
        "document_id",
        "source_doi",
        "source_record_id",
    )
    output = {}
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        if value is None:
            continue
        try:
            if bool(np.asarray(value).size == 1 and np.asarray(value).item() != np.asarray(value).item()):
                continue
        except (TypeError, ValueError):
            pass
        output[field] = bool(value) if field.startswith("challenge_") else str(value)
    return output


def build_split(
    release: Path, task: str, split: str, selected: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    memory, queries, labels, _ = load_task(release, task, split)
    expected_map = gold_targets(task, labels)
    memory_documents = [
        retrieval_document(task, row, query=False) for _, row in memory.iterrows()
    ]
    query_documents = [
        retrieval_document(task, row, query=True) for _, row in queries.iterrows()
    ]
    neighbors = retrieve(
        memory_documents,
        query_documents,
        view=selected["view"],
        k=selected["k"],
    )
    targets = [memory_target(task, row) for _, row in memory.iterrows()]
    schema = prediction_schema(task, memory)
    cases, gold, retrieval_rows = [], [], []
    for row_number, ((_, query), matches) in enumerate(zip(queries.iterrows(), neighbors)):
        query_id = str(query["query_id"])
        references = []
        for rank, (index, similarity) in enumerate(matches, start=1):
            episode = memory.iloc[index]
            references.append(
                {
                    "evidence_id": str(
                        episode.get("episode_id", episode.get("query_id", f"{task}:memory:{index}"))
                    ),
                    "rank": rank,
                    "similarity": similarity,
                    "input": memory_input(task, episode),
                    "observed_output": targets[index],
                    "source_dataset": str(episode.get("source_dataset", "")),
                }
            )
        case = {
            "protocol": PROTOCOL,
            "prompt_revision": PROMPT_REVISION,
            "case_id": query_id,
            "task": task,
            "task_kind": TASK_KINDS[task],
            "task_instruction": TASK_INSTRUCTIONS[task],
            "query_visible": query_input(task, query),
            "prediction_schema": schema,
            "retrieval": {
                "view": selected["view"],
                "k": selected["k"],
                "selection_split": "validation",
            },
            "references": references,
            "query_metadata": _query_metadata(query),
        }
        case_sha = digest(case)
        target = expected_map[query_id]
        gold.append(
            {
                "protocol": PROTOCOL,
                "case_id": query_id,
                "case_sha256": case_sha,
                "task": task,
                "task_kind": TASK_KINDS[task],
                "target": target,
                "query_metadata": case["query_metadata"],
            }
        )
        deterministic = transfer_prediction(
            task,
            [targets[index] for index, _ in matches],
            [similarity for _, similarity in matches],
        )
        evidence_ids = [row["evidence_id"] for row in references]
        retrieval_rows.append(
            {
                "protocol": PROTOCOL,
                "case_id": query_id,
                "case_sha256": case_sha,
                "task": task,
                "mode": "retrieval_only",
                "model": None,
                "status": "success",
                "response": envelope(
                    deterministic,
                    evidence_ids,
                    "Deterministic transfer from the validation-selected train-only neighbors.",
                ),
            }
        )
        cases.append(case)
    return cases, gold, retrieval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_manifest_path = args.release / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("release_id") != "matrecall-unified-tasks":
        raise ValueError("Unexpected release ID")
    args.output.mkdir(parents=True)

    development = []
    selected_by_task = {}
    for task in TASK_ORDER:
        print(canonical_json({"event": "develop-retrieval", "task": task}), flush=True)
        record = develop_task(args.release, task)
        development.append(record)
        selected_by_task[task] = record["selected"]
    (args.output / "retrieval_development.json").write_text(
        json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        all_cases, all_gold, all_retrieval = [], [], []
        for task in TASK_ORDER:
            print(canonical_json({"event": "freeze", "split": split, "task": task}), flush=True)
            cases, gold, retrieval_rows = build_split(
                args.release, task, split, selected_by_task[task]
            )
            all_cases.extend(cases)
            all_gold.extend(gold)
            all_retrieval.extend(retrieval_rows)
        split_dir = args.output / split
        for name, rows in (
            ("cases.jsonl", all_cases),
            ("gold.jsonl", all_gold),
            ("retrieval_only.jsonl", all_retrieval),
        ):
            path = split_dir / name
            write_jsonl(path, rows)
            files[str(path.relative_to(args.output))] = {
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }

    development_path = args.output / "retrieval_development.json"
    files[development_path.name] = {
        "bytes": development_path.stat().st_size,
        "sha256": file_sha256(development_path),
    }
    manifest = {
        "release_id": "matrecall-unified-evaluation",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_release_id": source_manifest["release_id"],
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "protocol": PROTOCOL,
        "prompt_revision": PROMPT_REVISION,
        "tasks": list(TASK_ORDER),
        "task_count": len(TASK_ORDER),
        "counts": {
            split: {
                "queries": files[f"{split}/cases.jsonl"]["rows"],
                "tasks": len(TASK_ORDER),
            }
            for split in ("validation", "test")
        },
        "retrieval_policy": {
            "views": list(VIEWS),
            "k_values": list(K_VALUES),
            "selection": "task-specific primary metric on validation only",
        },
        "target_normalization": {
            "target_composition_to_precursor_set": (
                "Null source placeholders are omitted from precursor formula sets; every retained set is non-empty."
            )
        },
        "files": files,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json(manifest["counts"]), flush=True)


if __name__ == "__main__":
    main()
