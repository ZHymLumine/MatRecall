"""Score process-task predictions against the separate private gold file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matrecall.process_evaluation as scoring_library
from matrecall.process_evaluation import (
    bootstrap_metrics,
    canonical_formula,
    read_jsonl,
    score_values,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gold_rows = read_jsonl(args.gold)
    prediction_rows = read_jsonl(args.predictions)
    gold = {row["case_id"]: row for row in gold_rows}
    predictions = {row["case_id"]: row for row in prediction_rows}
    if len(gold) != len(gold_rows) or len(predictions) != len(prediction_rows):
        raise ValueError("Duplicate case IDs")
    unexpected = sorted(predictions.keys() - gold.keys())
    if unexpected:
        raise ValueError(f"Unexpected prediction IDs: {unexpected[:3]}")

    details: list[dict[str, Any]] = []
    for target in gold_rows:
        result = predictions.get(target["case_id"])
        success = bool(result and result.get("status") == "success")
        detail = {
            "case_id": target["case_id"],
            "task": target["task"],
            "task_kind": target["task_kind"],
            "group_id": target["group_id"],
            "success": success,
            "expected": target["target"]["value"],
        }
        if result and result.get("case_sha256") != target["case_sha256"]:
            raise ValueError(f"Case checksum mismatch: {target['case_id']}")
        if success:
            try:
                prediction = result["response"]["reasoning_results"]["prediction"]
                value = prediction["value"]
                if target["task_kind"] == "binary_classification":
                    value = int(value)
                    detail["probability_success"] = prediction.get("probability_success")
                elif target["task_kind"] == "regression":
                    value = float(value)
                else:
                    value = str(value)
                detail["predicted"] = value
                if target["task_kind"] == "structured_prediction":
                    detail["expected_canonical"] = canonical_formula(detail["expected"])
                    detail["predicted_canonical"] = canonical_formula(value)
            except (KeyError, TypeError, ValueError) as exc:
                detail.update(success=False, error=f"Scoring input error: {exc}")
        if not detail["success"] and result:
            detail["error"] = detail.get("error", result.get("error"))
        details.append(detail)

    summary = {}
    for task in sorted({row["task"] for row in details}):
        rows = [row for row in details if row["task"] == task]
        scored = [row for row in rows if row["success"]]
        kind = rows[0]["task_kind"]
        metrics = None
        intervals = None
        if scored:
            expected = [row["expected"] for row in scored]
            predicted = [row["predicted"] for row in scored]
            probabilities = [row.get("probability_success") for row in scored]
            metrics = score_values(kind, expected, predicted, probabilities=probabilities)
            intervals = bootstrap_metrics(
                kind,
                expected,
                predicted,
                [row["group_id"] for row in scored],
                samples=args.bootstrap_samples,
            )
        summary[task] = {
            "task_kind": kind,
            "n_cases": len(rows),
            "n_scored": len(scored),
            "n_groups": len({row["group_id"] for row in rows}),
            "response_success_rate": len(scored) / len(rows),
            "metrics_conditional_on_valid_response": metrics,
            "group_bootstrap_95ci": intervals,
        }
    payload = {
        "summary": summary,
        "details": details,
        "gold_sha256": sha256_file(args.gold),
        "predictions_sha256": sha256_file(args.predictions),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "scoring_library_sha256": sha256_file(Path(scoring_library.__file__)),
        "semantics": (
            "All cases remain in response coverage. Numeric error metrics are conditional on valid responses; "
            "n_scored and success rate must accompany them. Classification and formula metrics are also reported "
            "on valid responses, with failures visible in details. Confidence intervals resample leakage groups."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
