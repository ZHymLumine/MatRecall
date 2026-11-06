"""Report native task metrics and prequential adaptation for online adaptation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from matrecall.unified_evaluation import file_sha256
from matrecall.report_common import metric_rows
from matrecall.run_online_adaptation import BASE_METHODS


METHODS = BASE_METHODS


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line: {path}:{number}")
            rows.append(json.loads(line))
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def sequential_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row["sequence_step"])
    scores = [row.get("feedback", {}).get("score") for row in rows]
    finite = [float(value) for value in scores if value is not None and math.isfinite(float(value))]
    window = max(1, math.ceil(len(rows) / 4))
    early_values = [
        float(value) for value in scores[:window]
        if value is not None and math.isfinite(float(value))
    ]
    late_values = [
        float(value) for value in scores[-window:]
        if value is not None and math.isfinite(float(value))
    ]
    early = fmean(early_values) if early_values else None
    late = fmean(late_values) if late_values else None
    metric_names = sorted({
        row.get("feedback", {}).get("metric_name") for row in rows
        if row.get("feedback", {}).get("metric_name") not in (None, "unscoreable")
    })
    return {
        "n_cases": len(rows),
        "n_success": sum(row.get("status") == "success" for row in rows),
        "response_success_rate": sum(row.get("status") == "success" for row in rows) / len(rows),
        "generic_feedback_metric": metric_names[0] if len(metric_names) == 1 else metric_names,
        "mean_generic_score": fmean(finite) if finite else None,
        "early_quartile_mean_score": early,
        "late_quartile_mean_score": late,
        "late_minus_early": None if early is None or late is None else late - early,
        "window_cases": window,
    }


def oriented_delta(left: float | None, right: float | None, higher: bool) -> float | None:
    if left is None or right is None:
        return None
    raw = left - right
    return raw if higher else -raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = load(args.release / "manifest.json")
    evaluation_run = args.run / "evaluation"
    completion = load(evaluation_run / "completion.json")
    if manifest.get("status") != "frozen" or manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task sequential release")
    if completion.get("task_count") != 22 or tuple(completion.get("methods", ())) != METHODS:
        raise ValueError("Sequential run does not cover all six methods and 22 tasks")
    if not str(completion.get("status", "")).startswith("completed"):
        raise ValueError("Sequential run is incomplete")

    task_records = {row["task"]: row for row in manifest["tasks"]}
    expected_tasks = set(task_records)
    native: dict[str, dict[str, dict[str, Any]]] = {}
    metric_paths: dict[str, Path] = {}
    for method in METHODS:
        native[method] = {}
        for family in ("unified", "process"):
            path = args.run / "metrics" / method / f"{family}.json"
            metric_paths[f"{method}/{family}"] = path
            native[method].update(load(path)["summary"])
        if set(native[method]) != expected_tasks:
            raise ValueError(f"Native metric task coverage mismatch: {method}")

    temporal: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[str, Path] = {}
    for task, task_record in task_records.items():
        temporal[task] = {}
        for method in METHODS:
            path = (
                evaluation_run / "test" / method / task_record["family"] / task /
                "predictions.jsonl"
            )
            prediction_paths[f"{method}/{task}"] = path
            rows = read_jsonl(path)
            expected = int(task_record["sequence_cases"])
            if len(rows) != expected or [row["sequence_step"] for row in rows] != list(range(1, expected + 1)):
                raise ValueError(f"Sequential output mismatch: {method}/{task}")
            temporal[task][method] = sequential_summary(rows)

    native_rows = []
    task_comparisons = []
    baseline_methods = [method for method in METHODS if method != "matrecall"]
    win_counts = {method: {"wins": 0, "ties": 0, "losses": 0, "unscored": 0} for method in baseline_methods}
    for task, task_record in task_records.items():
        process = task_record["family"] == "process"
        per_method = {
            method: {name: (value, higher) for name, value, higher in metric_rows(
                task, native[method][task], process=process
            )}
            for method in METHODS
        }
        primary_name = metric_rows(task, native["matrecall"][task], process=process)[0][0]
        primary_higher = per_method["matrecall"][primary_name][1]
        comparison = {
            "family": task_record["family"],
            "task": task,
            "primary_metric": primary_name,
            "higher_is_better": primary_higher,
            "matrecall_oriented_delta": {},
        }
        for method in METHODS:
            for name, (value, higher) in per_method[method].items():
                native_rows.append({
                    "family": task_record["family"],
                    "task": task,
                    "metric": name,
                    "higher_is_better": higher,
                    "method": method,
                    "value": value,
                })
        full_value = per_method["matrecall"].get(primary_name, (None, primary_higher))[0]
        for method in baseline_methods:
            baseline_value = per_method[method].get(primary_name, (None, primary_higher))[0]
            delta = oriented_delta(full_value, baseline_value, primary_higher)
            comparison["matrecall_oriented_delta"][method] = delta
            if delta is None:
                win_counts[method]["unscored"] += 1
            elif abs(delta) <= 1e-12:
                win_counts[method]["ties"] += 1
            elif delta > 0:
                win_counts[method]["wins"] += 1
            else:
                win_counts[method]["losses"] += 1
        task_comparisons.append(comparison)

    payload = {
        "release_id": manifest["release_id"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": 22,
        "method_count": len(METHODS),
        "methods": list(METHODS),
        "native_metric_table": native_rows,
        "temporal_learning_curves": temporal,
        "primary_task_comparisons": task_comparisons,
        "matrecall_task_win_counts": win_counts,
        "reporting_rule": (
            "Native metrics are reported per task and are not averaged across incompatible units. "
            "Generic prequential scores are used only within a task to compare sequence position and methods."
        ),
        "provenance": {
            "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
            "run_completion_sha256": file_sha256(evaluation_run / "completion.json"),
            "metric_files": {name: file_sha256(path) for name, path in metric_paths.items()},
            "prediction_files": {name: file_sha256(path) for name, path in prediction_paths.items()},
            "reporter_sha256": file_sha256(Path(__file__)),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    native_lookup = {
        (row["task"], row["metric"], row["method"]): row["value"] for row in native_rows
    }
    lines = [
        "# MatRecall online-adaptation results", "",
        f"Status: completed. Tasks: 22. Methods: {len(METHODS)}. Generated: {payload['created_at_utc']}.", "",
        "Each current answer was hidden until its prediction was recorded. Positive deltas mean MatRecall improved after orienting each task's primary native metric so larger is better.", "",
        "| Family | Task | Primary metric | LLM-only | Static RAG | ReAct static | Rolling ICL | Dynamic case RAG | MatRecall | Full−LLM | Full−Static RAG |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_comparisons:
        task, metric = row["task"], row["primary_metric"]
        lines.append(
            f"| {row['family']} | `{task}` | {metric} | "
            f"{fmt(native_lookup.get((task, metric, 'llm_only')))} | "
            f"{fmt(native_lookup.get((task, metric, 'static_multiview_rag')))} | "
            f"{fmt(native_lookup.get((task, metric, 'react_static_memory')))} | "
            f"{fmt(native_lookup.get((task, metric, 'rolling_context_icl')))} | "
            f"{fmt(native_lookup.get((task, metric, 'dynamic_case_rag')))} | "
            f"{fmt(native_lookup.get((task, metric, 'matrecall')))} | "
            f"{fmt(row['matrecall_oriented_delta']['llm_only'])} | "
            f"{fmt(row['matrecall_oriented_delta']['static_multiview_rag'])} |"
        )
    lines.extend(("", "## MatRecall task-level primary-metric wins", ""))
    for method, counts in win_counts.items():
        lines.append(
            f"- Versus `{method}`: {counts['wins']} wins, {counts['ties']} ties, "
            f"{counts['losses']} losses, {counts['unscored']} unscored."
        )
    lines.extend((
        "", "The JSON report also records early- and late-quartile generic feedback scores for every task and method. These curves are interpreted within task only.",
    ))
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "tasks": 22, "methods": len(METHODS)}, sort_keys=True))


if __name__ == "__main__":
    main()
