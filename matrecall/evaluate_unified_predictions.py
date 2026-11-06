"""Score the thirteen non-process endpoints in the unified task release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import matrecall.unified_evaluation as scoring_library
from matrecall.unified_evaluation import (
    CRYSTAL_TASKS,
    FECONI_ELEMENTS,
    FECONI_PROPERTIES,
    PROTOCOL,
    composition_dict,
    file_sha256,
    prediction_from_row,
    read_jsonl,
    load_task,
    summarize_task,
)


def _finite_prediction(task_kind: str, prediction: Any) -> bool:
    """Reject values that JSON parsers accept but metrics cannot score."""
    if not isinstance(prediction, dict):
        return False
    if task_kind == "scalar_regression":
        try:
            return math.isfinite(float(prediction["value"]))
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _summarize_subset(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_task(
        task,
        [row["target"] for row in rows],
        [row["prediction"] if row["scoreable"] else None for row in rows],
    )


def _payload(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


def _fit_forward_surrogate(release: Path):
    task = "target_property_to_formulation_composition"
    memory, _, _, _ = load_task(release, task, "validation")
    x, y = [], []
    for _, row in memory.iterrows():
        composition = _payload(row["target_json"])["element_fractions"]
        properties = _payload(row["input_json"])["target_properties"]
        x.append([float(composition.get(element, 0.0)) for element in FECONI_ELEMENTS])
        y.append([float(properties[field]) for field in FECONI_PROPERTIES])
    matrix = np.asarray(y, dtype=float)
    scales = dict(zip(FECONI_PROPERTIES, np.std(matrix, axis=0, ddof=0).tolist()))
    model = ExtraTreesRegressor(n_estimators=500, random_state=0, n_jobs=1)
    model.fit(np.asarray(x, dtype=float), matrix)
    targets = {}
    for split in ("validation", "test"):
        _, queries, _, _ = load_task(release, task, split)
        for _, row in queries.iterrows():
            targets[str(row["query_id"])] = _payload(row["input_json"])["target_properties"]
    return model, scales, targets


def _enrich_inverse_design(rows: list[dict[str, Any]], release: Path) -> None:
    model, scales, targets = _fit_forward_surrogate(release)
    for row in rows:
        if row["task"] != "target_property_to_formulation_composition":
            continue
        row["target"]["target_properties"] = targets[row["case_id"]]
        row["target"]["forward_property_scales"] = scales
        if not row["scoreable"]:
            continue
        composition, valid = composition_dict(row["prediction"])
        if not valid:
            continue
        vector = np.asarray([[composition.get(element, 0.0) for element in FECONI_ELEMENTS]])
        values = model.predict(vector)[0]
        row["prediction"]["forward_surrogate_properties"] = {
            field: float(value) for field, value in zip(FECONI_PROPERTIES, values)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    gold_rows = read_jsonl(args.gold)
    prediction_rows = read_jsonl(args.predictions)
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    predictions_by_id = {row["case_id"]: row for row in prediction_rows}
    if len(gold_by_id) != len(gold_rows) or len(predictions_by_id) != len(prediction_rows):
        raise ValueError("Duplicate case IDs")
    unexpected = sorted(predictions_by_id.keys() - gold_by_id.keys())
    if unexpected:
        raise ValueError(f"Unexpected prediction IDs: {unexpected[:3]}")

    details: list[dict[str, Any]] = []
    for gold in gold_rows:
        result = predictions_by_id.get(gold["case_id"])
        if result and result.get("case_sha256") != gold["case_sha256"]:
            raise ValueError(f"Case checksum mismatch: {gold['case_id']}")
        if result and result.get("protocol") != PROTOCOL:
            raise ValueError(f"Protocol mismatch: {gold['case_id']}")
        prediction = prediction_from_row(result) if result else None
        scoreable = prediction is not None and _finite_prediction(gold["task_kind"], prediction)
        detail = {
            "case_id": gold["case_id"],
            "task": gold["task"],
            "task_kind": gold["task_kind"],
            "response_status": result.get("status") if result else "missing",
            "scoreable": scoreable,
            "target": gold["target"],
            "prediction": prediction,
            "query_metadata": gold.get("query_metadata", {}),
        }
        if not scoreable and result:
            detail["error"] = result.get("error", "Prediction is not scoreable")
        details.append(detail)

    if args.release is not None:
        _enrich_inverse_design(details, args.release)

    summaries = {}
    challenge_summaries = {}
    for task in sorted({row["task"] for row in details}):
        rows = [row for row in details if row["task"] == task]
        summaries[task] = _summarize_subset(task, rows)
        if task in CRYSTAL_TASKS:
            task_subsets = {}
            for field in (
                "challenge_any",
                "challenge_new_reduced_formula",
                "challenge_new_chemical_system",
                "challenge_new_structure_family_proxy",
            ):
                subset = [row for row in rows if row["query_metadata"].get(field) is True]
                if subset:
                    task_subsets[field] = _summarize_subset(task, subset)
            challenge_summaries[task] = task_subsets

    payload = {
        "protocol": PROTOCOL,
        "summary": summaries,
        "crystal_challenge_subsets": challenge_summaries,
        "details": details,
        "gold_sha256": file_sha256(args.gold),
        "predictions_sha256": file_sha256(args.predictions),
        "evaluator_sha256": file_sha256(Path(__file__)),
        "scoring_library_sha256": file_sha256(Path(scoring_library.__file__)),
        "source_release_manifest_sha256": file_sha256(args.release / "manifest.json") if args.release else None,
        "inverse_design_forward_surrogate": (
            "A deterministic ExtraTrees forward surrogate is fit on the Fe-Co-Ni training memory only. "
            "It supplies per-property and training-standardized target error for every method."
            if args.release else "Not computed because --release was not supplied."
        ),
        "semantics": (
            "All cases remain in response coverage. Metrics are conditional on scoreable "
            "responses and must be read together with n_success and response_success_rate. "
            "Crystal challenge subsets are pre-defined metadata filters of the official test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
