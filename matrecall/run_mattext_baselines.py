"""Run matched composition and geometry ExtraTrees baselines for MatText."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor

from matrecall.mattext import (
    composition_descriptor,
    file_sha256,
    read_jsonl,
    regression_metrics,
    structure_descriptor,
    write_jsonl,
)


TASK_PREFIX = {
    "mattext_bulk_modulus_log10_gpa": "kvrh",
    "mattext_shear_modulus_log10_gpa": "gvrh",
    "mattext_perovskite_formation_energy": "perovskites",
}
MODELS = ("composition_extra_trees", "geometry_extra_trees")
REPRESENTATIONS = (
    "composition", "atom_sequences_plusplus", "slices", "local_env",
    "crystal_text_llm",
)


def _parquet(raw_root: Path, prefix: str, split: str, fold: int) -> Path:
    return (
        raw_root / "benchmark_parquet" / f"{prefix}-{split}-matbench" /
        f"fold_{fold}" / "0000.parquet"
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _complete(row: dict[str, Any]) -> bool:
    return all(isinstance(row.get(key), str) and row[key].strip() for key in REPRESENTATIONS)


def run_baselines(
    *, raw_root: Path, release: Path, output: Path, task: str,
    n_estimators: int = 500, seed: int = 0, jobs: int = 8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if task not in TASK_PREFIX:
        raise ValueError(f"Unsupported real MatText task: {task}")
    manifest = json.loads((release / "manifest.json").read_text())
    if task not in manifest["tasks"]:
        raise ValueError(f"Task absent from release: {task}")
    prefix = TASK_PREFIX[task]
    registry: dict[str, dict[str, Any]] = {}
    labels: dict[str, float] = {}
    source_hashes: dict[str, str] = {}
    for fold in range(5):
        path = _parquet(raw_root, prefix, "train", fold)
        source_hashes[str(path.relative_to(raw_root))] = file_sha256(path)
        for row in _rows(path):
            identifier = str(row["mbid"])
            label = float(row["labels"])
            if identifier in labels and labels[identifier] != label:
                raise ValueError(f"Inconsistent label for {identifier}")
            labels[identifier] = label
            if _complete(row):
                registry[identifier] = row
    if not registry:
        raise ValueError("No complete training records")
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def features(identifier: str) -> tuple[np.ndarray, np.ndarray]:
        if identifier not in feature_cache:
            row = registry[identifier]
            composition = composition_descriptor(row["composition"])
            geometry = np.concatenate([
                composition, structure_descriptor(row["crystal_text_llm"])
            ])
            feature_cache[identifier] = composition, geometry
        return feature_cache[identifier]

    summaries: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    for fold in range(5):
        train_rows = [row for row in _rows(_parquet(raw_root, prefix, "train", fold)) if _complete(row)]
        train_ids = [str(row["mbid"]) for row in train_rows]
        if len(set(train_ids)) != len(train_ids):
            raise ValueError(f"Duplicate training material in fold {fold}")
        y_train = np.asarray([float(row["labels"]) for row in train_rows])
        composition_x = np.stack([features(identifier)[0] for identifier in train_ids])
        geometry_x = np.stack([features(identifier)[1] for identifier in train_ids])
        fitted = {
            "composition_extra_trees": ExtraTreesRegressor(
                n_estimators=n_estimators, min_samples_leaf=2,
                random_state=seed, n_jobs=jobs,
            ).fit(composition_x, y_train),
            "geometry_extra_trees": ExtraTreesRegressor(
                n_estimators=n_estimators, min_samples_leaf=2,
                random_state=seed, n_jobs=jobs,
            ).fit(geometry_x, y_train),
        }
        record = next(
            row for row in manifest["task_fold_records"]
            if row["task"] == task and int(row["fold"]) == fold
        )
        source = release / record["relative_path"]
        for stage in ("exposure", "frozen_test"):
            cases = read_jsonl(source / stage / "cases.jsonl")
            gold = read_jsonl(source / stage / "gold.jsonl")
            ids = [row["material_id"] for row in cases]
            if any(identifier not in registry for identifier in ids):
                raise ValueError(f"Selected test material missing from feature registry: {task}/{fold}")
            expected = [float(row["target"]["value"]) for row in gold]
            for model_name, model in fitted.items():
                matrix = np.stack([
                    features(identifier)[0 if model_name.startswith("composition") else 1]
                    for identifier in ids
                ])
                predicted = [float(value) for value in model.predict(matrix)]
                rows = [{
                    "case_id": case["case_id"],
                    "material_id": case["material_id"],
                    "task": task,
                    "fold": fold,
                    "stage": stage,
                    "method": model_name,
                    "prediction": {"value": value, "unit": case["target_unit"]},
                    "target_visible_to_model": False,
                } for case, value in zip(cases, predicted)]
                target = output / task / f"fold_{fold}" / stage / model_name
                write_jsonl(target / "predictions.jsonl", rows)
                metrics = regression_metrics(expected, predicted)
                (target / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n"
                )
                summaries.append({
                    "task": task, "fold": fold, "stage": stage,
                    "method": model_name, **metrics,
                })
    completion = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "models": list(MODELS),
        "fold_count": 5,
        "records": summaries,
        "training_policy": {
            "official_matbench_training_fold_only": True,
            "fixed_hyperparameters": True,
            "n_estimators": n_estimators,
            "min_samples_leaf": 2,
            "seed": seed,
            "composition_model_input": "118 element fractions plus elemental summaries",
            "geometry_model_input": "composition plus lattice, distance and coordination descriptors",
        },
        "source_sha256": source_hashes,
        "release_manifest_sha256": file_sha256(release / "manifest.json"),
    }
    (output / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_PREFIX), required=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    completion = run_baselines(
        raw_root=args.raw_root,
        release=args.release,
        output=args.output,
        task=args.task,
        n_estimators=args.n_estimators,
        seed=args.seed,
        jobs=args.jobs,
    )
    print(json.dumps({
        "event": "complete", "task": completion["task"],
        "models": completion["models"],
    }))


if __name__ == "__main__":
    main()

