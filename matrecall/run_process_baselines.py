"""Train simple matched-data baselines and predict the frozen process test cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from matrecall.process_evaluation import (
    PROTOCOL,
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


def clean_feature(value: Any) -> Any:
    if value is None:
        return "__MISSING__"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else "__MISSING__"
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return number if math.isfinite(number) else "__MISSING__"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def features(frame) -> list[dict[str, Any]]:
    return [
        {str(key): clean_feature(value) for key, value in json.loads(payload).items()}
        for payload in frame["feature_json"]
    ]


def candidates(kind: str, seed: int, jobs: int) -> list[tuple[str, Any]]:
    if kind == "regression":
        return [
            ("dummy_median", DummyRegressor(strategy="median")),
            ("random_forest", RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=jobs, max_features=1.0)),
            ("random_forest_leaf2", RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=jobs, max_features=1.0, min_samples_leaf=2)),
            ("extra_trees", ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=jobs, max_features=1.0)),
            ("extra_trees_leaf2", ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=jobs, max_features=1.0, min_samples_leaf=2)),
        ]
    if kind in ("binary_classification", "multiclass_classification"):
        return [
            ("dummy_prior", DummyClassifier(strategy="prior", random_state=seed)),
            ("random_forest_balanced", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced")),
            ("random_forest_balanced_leaf2", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced", min_samples_leaf=2)),
            ("extra_trees_balanced", ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced")),
            ("extra_trees_balanced_leaf2", ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced", min_samples_leaf=2)),
        ]
    raise ValueError(kind)


def choose(trials: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    metric, maximize = selection_metric(kind)
    usable = [row for row in trials if row["metrics"].get(metric) is not None]
    return sorted(
        usable,
        key=lambda row: (-row["metrics"][metric] if maximize else row["metrics"][metric], row["name"]),
    )[0]


def tree_task(task_dir: Path, seed: int, jobs: int):
    memory, validation, validation_labels, manifest = load_task(task_dir, "validation")
    _, test, _, _ = load_task(task_dir, "test")
    kind = manifest["task_kind"]
    vectorizer = DictVectorizer(sparse=True)
    train_x = vectorizer.fit_transform(features(memory))
    validation_x = vectorizer.transform(features(validation))
    test_x = vectorizer.transform(features(test))
    train_y = target_values(memory, kind)
    validation_y = target_values(validation_labels, kind)
    trials, fitted = [], {}
    for name, model in candidates(kind, seed, jobs):
        model.fit(train_x, train_y)
        prediction = model.predict(validation_x).tolist()
        probability = None
        if kind == "binary_classification" and hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            probability = model.predict_proba(validation_x)[:, classes.index(1)].tolist()
        metrics = score_values(kind, validation_y, prediction, probabilities=probability)
        trials.append({"name": name, "metrics": metrics})
        fitted[name] = model
    selected = choose(trials, kind)
    model = fitted[selected["name"]]
    prediction = model.predict(test_x).tolist()
    probability = [None] * len(prediction)
    if kind == "binary_classification" and hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probability = model.predict_proba(test_x)[:, classes.index(1)].tolist()
    return prediction, probability, {
        "task": task_dir.name,
        "task_kind": kind,
        "selection_split": "validation",
        "selection_metric": selection_metric(kind)[0],
        "selected": selected,
        "candidates": trials,
        "feature_space_size": len(vectorizer.feature_names_),
        "train_rows": len(memory),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


def formula_task(task_dir: Path):
    memory, validation, validation_labels, manifest = load_task(task_dir, "validation")
    _, test, _, _ = load_task(task_dir, "test")
    kind = manifest["task_kind"]
    validation_y = target_values(validation_labels, kind)
    trials = []
    cache = {}
    for view in ("text_word", "text_char"):
        for k in (1, 3, 5, 10):
            neighbors = retrieve(memory, validation, view=view, k=k)
            prediction = retrieval_predictions(memory, neighbors, kind)
            metrics = score_values(kind, validation_y, prediction)
            name = f"tfidf_knn_{view}_k{k}"
            trials.append({"name": name, "view": view, "k": k, "metrics": metrics})
            cache[name] = (view, k)
    selected = choose(trials, kind)
    view, k = cache[selected["name"]]
    prediction = retrieval_predictions(memory, retrieve(memory, test, view=view, k=k), kind)
    return prediction, [None] * len(prediction), {
        "task": task_dir.name,
        "task_kind": kind,
        "selection_split": "validation",
        "selection_metric": selection_metric(kind)[0],
        "selected": selected,
        "candidates": trials,
        "note": "Matched simple baseline; not the published Materials Graph Transformer.",
        "train_rows": len(memory),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


def response(value: Any, probability: float | None) -> dict[str, Any]:
    prediction = {"value": value}
    if probability is not None:
        prediction["probability_success"] = float(probability)
    return {
        "analysis_process": [{
            "comparison": "Prediction from a validation-selected conventional matched-data baseline.",
            "evidence_ids": [],
            "implication": "The model was trained only on the benchmark training split.",
        }],
        "reasoning_results": {"prediction": prediction, "evidence_ids": []},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--evaluation-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cases = {row["task"]: [] for row in read_jsonl(args.evaluation_release / "test" / "cases.jsonl")}
    for row in read_jsonl(args.evaluation_release / "test" / "cases.jsonl"):
        cases[row["task"]].append(row)
    predictions, development = [], []
    for task_dir in task_dirs(args.release):
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["task_kind"] == "structured_prediction":
            values, probabilities, report = formula_task(task_dir)
        else:
            values, probabilities, report = tree_task(task_dir, args.seed, args.jobs)
        task_cases = cases[task_dir.name]
        if len(values) != len(task_cases):
            raise ValueError(f"Prediction/case count mismatch for {task_dir.name}")
        model_name = report["selected"]["name"]
        for case, value, probability in zip(task_cases, values, probabilities):
            if isinstance(value, np.generic):
                value = value.item()
            predictions.append({
                "protocol": PROTOCOL,
                "case_id": case["case_id"],
                "case_sha256": digest(case),
                "task": task_dir.name,
                "mode": "matched_specialized_baseline",
                "model": model_name,
                "status": "success",
                "response": response(value, probability),
            })
        development.append(report)
    prediction_path = args.output / "test_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    (args.output / "validation_model_selection.json").write_text(
        json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "training_data": "training split only",
        "selection_data": "validation split only",
        "test_labels_used_during_fit_or_selection": False,
        "prediction_rows": len(predictions),
        "source_release_manifest_sha256": sha256_file(args.release / "manifest.json"),
        "evaluation_release_manifest_sha256": sha256_file(args.evaluation_release / "manifest.json"),
        "predictions_sha256": sha256_file(prediction_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "scope": (
            "RF/ExtraTrees/Dummy candidates for numeric and class tasks; TF-IDF kNN for formula prediction. "
            "These matched baselines are distinct from paper-reported models on original random splits."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "prediction_rows": len(predictions), "tasks": len(development)}))


if __name__ == "__main__":
    main()
