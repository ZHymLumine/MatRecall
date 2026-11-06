"""Prepare an aligned MatText representation-by-memory evaluation release.

The source MatText test parquet files intentionally omit labels.  MatBench folds
partition the same finite dataset, so a test material's label is available in
the training portion of the other folds.  This builder reconstructs a canonical
label registry, but writes those recovered targets only to separate gold files.
No evaluation case contains its current target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


RELEASE_ID = "matrecall-mattext"
SAMPLING_REVISION = "mattext-sampling"
# Fixed hash salt of the released sampling; changing it changes which cases are selected.
SAMPLING_SALT = "2026-10-30-mattext-representation-memory-v1"
FOLDS = tuple(range(5))
REPRESENTATIONS = (
    "composition",
    "atom_sequences_plusplus",
    "slices",
    "local_env",
    "crystal_text_llm",
)
REPRESENTATION_GROUP = {
    "composition": "category",
    "atom_sequences_plusplus": "category_and_cell",
    "slices": "bonding_without_explicit_coordinates",
    "local_env": "local_geometry",
    "crystal_text_llm": "explicit_geometry",
}


@dataclass(frozen=True)
class TaskSpec:
    source_prefix: str
    task_id: str
    instruction: str
    unit: str


TASKS = (
    TaskSpec(
        source_prefix="kvrh",
        task_id="mattext_bulk_modulus_log10_gpa",
        instruction=(
            "Predict the base-10 logarithm of the DFT Voigt-Reuss-Hill average "
            "bulk modulus in GPa."
        ),
        unit="log10(GPa)",
    ),
    TaskSpec(
        source_prefix="gvrh",
        task_id="mattext_shear_modulus_log10_gpa",
        instruction=(
            "Predict the base-10 logarithm of the DFT Voigt-Reuss-Hill average "
            "shear modulus in GPa."
        ),
        unit="log10(GPa)",
    ),
    TaskSpec(
        source_prefix="perovskites",
        task_id="mattext_perovskite_formation_energy",
        instruction=(
            "Predict the RPBE GGA-DFT heat of formation of the entire "
            "perovskite cell in eV."
        ),
        unit="eV/cell",
    ),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, identifier: str) -> str:
    payload = f"{SAMPLING_SALT}:{seed}:{namespace}:{identifier}"
    return hashlib.sha256(payload.encode()).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _parquet_path(raw_root: Path, spec: TaskSpec, split: str, fold: int) -> Path:
    config = f"{spec.source_prefix}-{split}-matbench"
    return raw_root / "benchmark_parquet" / config / f"fold_{fold}" / "0000.parquet"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _representation_payload(row: dict[str, Any]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for representation in REPRESENTATIONS:
        value = row.get(representation)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {representation} for {row.get('mbid')}")
        payload[representation] = value
    return payload


def _source_registry(
    raw_root: Path, spec: TaskSpec
) -> tuple[dict[str, float], set[str], dict[str, int]]:
    labels: dict[str, float] = {}
    representations: dict[str, dict[str, str]] = {}
    incomplete_by_representation = {name: set() for name in REPRESENTATIONS}
    for fold in FOLDS:
        for row in _read_rows(_parquet_path(raw_root, spec, "train", fold)):
            identifier = row["mbid"]
            label = float(row["labels"])
            if identifier in labels and labels[identifier] != label:
                raise ValueError(f"Inconsistent label for {identifier}")
            labels[identifier] = label
            missing = [
                name for name in REPRESENTATIONS
                if not isinstance(row.get(name), str) or not row[name].strip()
            ]
            if missing:
                for name in missing:
                    incomplete_by_representation[name].add(identifier)
                continue
            payload = _representation_payload(row)
            if identifier in representations and representations[identifier] != payload:
                raise ValueError(f"Inconsistent representations for {identifier}")
            representations[identifier] = payload
    incomplete_ids = set().union(*incomplete_by_representation.values())
    complete_ids = set(labels) - incomplete_ids
    missing_counts = {
        name: len(identifiers)
        for name, identifiers in incomplete_by_representation.items()
        if identifiers
    }
    return labels, complete_ids, missing_counts


def _ordered_sample(
    rows: list[dict[str, Any]], *, seed: int, namespace: str, maximum: int
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len({row["mbid"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate mbid in {namespace}")
    return sorted(
        rows,
        key=lambda row: stable_key(seed, namespace, row["mbid"]),
    )[:maximum]


def _case_row(
    row: dict[str, Any], spec: TaskSpec, *, fold: int, stage: str, step: int
) -> dict[str, Any]:
    return {
        "case_id": f"mattext:{spec.source_prefix}:fold-{fold}:{stage}:{row['mbid']}",
        "material_id": row["mbid"],
        "task": spec.task_id,
        "task_kind": "scalar_regression",
        "task_instruction": spec.instruction,
        "target_unit": spec.unit,
        "source_fold": fold,
        "stage": stage,
        "sequence_step": step,
        "representations": _representation_payload(row),
        "representation_groups": REPRESENTATION_GROUP,
    }


def _gold_row(
    case: dict[str, Any], label: float, *, recovered_from_other_training_fold: bool
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "material_id": case["material_id"],
        "task": case["task"],
        "task_kind": case["task_kind"],
        "target": {"value": label, "unit": case["target_unit"]},
        "label_provenance": {
            "test_parquet_contains_label": False,
            "recovered_from_other_training_fold": recovered_from_other_training_fold,
            "visible_to_method_before_prediction": False,
        },
    }


def _static_row(
    row: dict[str, Any], spec: TaskSpec, *, fold: int
) -> dict[str, Any]:
    return {
        "episode_id": f"mattext:static:{spec.source_prefix}:fold-{fold}:{row['mbid']}",
        "material_id": row["mbid"],
        "task": spec.task_id,
        "task_kind": "scalar_regression",
        "source_fold": fold,
        "source_split": "official MatBench training fold",
        "input_views": _representation_payload(row),
        "observed_output": {"value": float(row["labels"]), "unit": spec.unit},
        "epistemic_status": "dataset_record",
    }


def build_release(
    raw_root: Path,
    output: Path,
    *,
    static_cases: int = 256,
    exposure_cases: int = 32,
    frozen_cases: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if min(static_cases, exposure_cases, frozen_cases) < 1:
        raise ValueError("All sample budgets must be positive")

    source_hashes: dict[str, str] = {}
    task_records: list[dict[str, Any]] = []
    cohort_records: list[dict[str, Any]] = []
    output.mkdir(parents=True)

    for spec in TASKS:
        labels, complete_ids, missing_counts = _source_registry(raw_root, spec)
        test_occurrences: dict[str, int] = {}
        for fold in FOLDS:
            train_path = _parquet_path(raw_root, spec, "train", fold)
            test_path = _parquet_path(raw_root, spec, "test", fold)
            source_hashes[str(train_path.relative_to(raw_root))] = file_sha256(train_path)
            source_hashes[str(test_path.relative_to(raw_root))] = file_sha256(test_path)
            raw_train_rows = _read_rows(train_path)
            raw_test_rows = _read_rows(test_path)
            if any("labels" in row for row in raw_test_rows):
                raise ValueError(f"Expected hidden labels in {test_path}")
            train_ids = {row["mbid"] for row in raw_train_rows}
            test_ids = {row["mbid"] for row in raw_test_rows}
            if train_ids & test_ids:
                raise ValueError(f"Train/test overlap for {spec.task_id}, fold {fold}")
            missing_labels = sorted((test_ids & complete_ids) - labels.keys())
            if missing_labels:
                raise ValueError(f"Missing reconstructed labels: {missing_labels[:3]}")
            train_rows = [row for row in raw_train_rows if row["mbid"] in complete_ids]
            test_rows = [row for row in raw_test_rows if row["mbid"] in complete_ids]
            for identifier in (test_ids & complete_ids):
                test_occurrences[identifier] = test_occurrences.get(identifier, 0) + 1

            static_rows = _ordered_sample(
                train_rows,
                seed=seed,
                namespace=f"static:{spec.task_id}:fold-{fold}",
                maximum=static_cases,
            )
            ordered_test = _ordered_sample(
                test_rows,
                seed=seed,
                namespace=f"stream:{spec.task_id}:fold-{fold}",
                maximum=exposure_cases + frozen_cases,
            )
            if len(ordered_test) < exposure_cases + frozen_cases:
                raise ValueError(f"Insufficient test rows for {spec.task_id}, fold {fold}")
            exposure_rows = ordered_test[:exposure_cases]
            frozen_rows = ordered_test[exposure_cases:]
            selected_ids = {row["mbid"] for row in ordered_test}
            if selected_ids & {row["mbid"] for row in static_rows}:
                raise ValueError("Selected static/test overlap")

            relative = Path("tasks") / spec.task_id / f"fold_{fold}"
            write_jsonl(
                output / relative / "static_library.jsonl",
                (_static_row(row, spec, fold=fold) for row in static_rows),
            )

            stage_records: dict[str, int] = {}
            for stage, rows in (("exposure", exposure_rows), ("frozen_test", frozen_rows)):
                cases = [
                    _case_row(row, spec, fold=fold, stage=stage, step=step)
                    for step, row in enumerate(rows, start=1)
                ]
                gold = [
                    _gold_row(
                        case,
                        labels[case["material_id"]],
                        recovered_from_other_training_fold=True,
                    )
                    for case in cases
                ]
                order = [
                    {
                        "release_id": RELEASE_ID,
                        "task": spec.task_id,
                        "source_fold": fold,
                        "stage": stage,
                        "sequence_step": step,
                        "case_id": case["case_id"],
                        "material_id": case["material_id"],
                    }
                    for step, case in enumerate(cases, start=1)
                ]
                write_jsonl(output / relative / stage / "cases.jsonl", cases)
                write_jsonl(output / relative / stage / "gold.jsonl", gold)
                write_jsonl(output / relative / stage / "order.jsonl", order)
                stage_records[stage] = len(cases)

            task_records.append({
                "task": spec.task_id,
                "source_prefix": spec.source_prefix,
                "fold": fold,
                "source_train_rows": len(raw_train_rows),
                "source_test_rows": len(raw_test_rows),
                "matched_train_rows": len(train_rows),
                "matched_test_rows": len(test_rows),
                "static_cases": len(static_rows),
                **{f"{key}_cases": value for key, value in stage_records.items()},
                "relative_path": str(relative),
            })

        non_partitioned = {
            identifier: count for identifier, count in test_occurrences.items() if count != 1
        }
        if non_partitioned or set(test_occurrences) != complete_ids:
            raise ValueError(f"MatBench folds do not partition {spec.task_id}")
        cohort_records.append({
            "task": spec.task_id,
            "source_prefix": spec.source_prefix,
            "source_materials": len(labels),
            "matched_materials": len(complete_ids),
            "excluded_materials": len(labels) - len(complete_ids),
            "missing_representation_counts": missing_counts,
        })

    manifest = {
        "release_id": RELEASE_ID,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_revision": SAMPLING_REVISION,
        "source_dataset": "n0w0f/MatText",
        "source_dataset_revision": "1b31583fd9198f73d2fbd5287d2048205f4a3a03",
        "source_license": "MIT",
        "tasks": [spec.task_id for spec in TASKS],
        "task_fold_records": task_records,
        "matched_cohorts": cohort_records,
        "representations": [
            {"name": name, "group": REPRESENTATION_GROUP[name]}
            for name in REPRESENTATIONS
        ],
        "sampling": {
            "folds": list(FOLDS),
            "static_cases_per_task_fold": static_cases,
            "exposure_cases_per_task_fold": exposure_cases,
            "frozen_cases_per_task_fold": frozen_cases,
            "seed": seed,
        },
        "protocol": {
            "official_matbench_folds": True,
            "aligned_material_ids_across_representations": True,
            "current_answer_hidden_before_prediction": True,
            "exposure_feedback_revealed_after_prediction": True,
            "frozen_test_feedback_revealed_during_evaluation": False,
            "frozen_test_memory_writes": False,
            "static_library_uses_only_same_fold_training_records": True,
            "test_targets_stored_only_in_gold_files": True,
            "test_labels_recovered_from_other_folds_for_scoring_only": True,
            "matched_cohort_requires_all_selected_representations": True,
        },
        "experiment_cells": {
            "within_representation_prequential": list(REPRESENTATIONS),
            "cross_representation_memory_sources": [
                "composition", "crystal_text_llm"
            ],
            "cross_representation_query_targets": list(REPRESENTATIONS),
            "negative_controls": [
                "task_matched_shuffled_verified_episodes",
                "unrelated_task_verified_episodes",
            ],
        },
        "source_sha256": source_hashes,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-cases", type=int, default=256)
    parser.add_argument("--exposure-cases", type=int, default=32)
    parser.add_argument("--frozen-cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build_release(
        args.raw_root,
        args.output,
        static_cases=args.static_cases,
        exposure_cases=args.exposure_cases,
        frozen_cases=args.frozen_cases,
        seed=args.seed,
    )
    print(canonical_json({
        "event": "complete",
        "release_id": manifest["release_id"],
        "task_fold_records": len(manifest["task_fold_records"]),
    }))


if __name__ == "__main__":
    main()
