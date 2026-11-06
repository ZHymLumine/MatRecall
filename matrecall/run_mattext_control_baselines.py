"""Run matched composition and geometry baselines for the MatText coordinate--category control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from matrecall.mattext import (
    CONTROL_RELEASE_ID,
    composition_descriptor,
    file_sha256,
    read_jsonl,
    regression_metrics,
    structure_descriptor,
    write_jsonl,
)


MODELS = ("composition_extra_trees", "geometry_extra_trees")


def _features(record: dict[str, Any], model: str) -> np.ndarray:
    views = record.get("input_views", record.get("representations"))
    composition = composition_descriptor(views["composition"])
    if model == "composition_extra_trees":
        return composition
    if model == "geometry_extra_trees":
        return np.concatenate([
            composition, structure_descriptor(views["crystal_text_llm"])
        ])
    raise ValueError(model)


def run_baselines(
    *, release: Path, output: Path, task: str, n_estimators: int = 500,
    seed: int = 0, jobs: int = 8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest.get("release_id") != CONTROL_RELEASE_ID:
        raise ValueError("Expected the MatText coordinate--category control release")
    if task not in manifest["tasks"]:
        raise ValueError(f"Unknown control task: {task}")
    output.mkdir(parents=True)
    summaries: list[dict[str, Any]] = []
    records = sorted(
        (
            row for row in manifest["task_fold_records"]
            if row["task"] == task
        ),
        key=lambda row: int(row["fold"]),
    )
    if len(records) != 5:
        raise ValueError(f"Expected five folds for {task}")
    for record in records:
        fold = int(record["fold"])
        source = release / record["relative_path"]
        train = read_jsonl(source / "static_library.jsonl")
        y_train = np.asarray([
            float(row["observed_output"]["value"]) for row in train
        ])
        fitted = {}
        for model_name in MODELS:
            matrix = np.stack([_features(row, model_name) for row in train])
            fitted[model_name] = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=jobs,
            ).fit(matrix, y_train)
        for stage in ("exposure", "frozen_test"):
            cases = read_jsonl(source / stage / "cases.jsonl")
            gold = read_jsonl(source / stage / "gold.jsonl")
            expected = [float(row["target"]["value"]) for row in gold]
            for model_name, model in fitted.items():
                matrix = np.stack([_features(row, model_name) for row in cases])
                predicted = [float(value) for value in model.predict(matrix)]
                rows = [{
                    "case_id": case["case_id"],
                    "material_id": case["material_id"],
                    "task": task,
                    "fold": fold,
                    "stage": stage,
                    "method": model_name,
                    "prediction": {
                        "value": value, "unit": case["target_unit"],
                    },
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
        "fold_count": len(records),
        "records": summaries,
        "training_policy": {
            "same_static_training_sample_as_rag": True,
            "fixed_hyperparameters": True,
            "n_estimators": n_estimators,
            "min_samples_leaf": 2,
            "seed": seed,
        },
        "release_manifest_sha256": file_sha256(release / "manifest.json"),
    }
    (output / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    completion = run_baselines(
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
