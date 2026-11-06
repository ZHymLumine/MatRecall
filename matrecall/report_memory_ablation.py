"""Aggregate memory ablations, order robustness, experience, and retrieval size."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import fmean, median
from typing import Any, Iterable

from matrecall.unified_evaluation import file_sha256
from matrecall.report_common import metric_rows
from matrecall.run_memory_ablation import (
    METHODS,
    ORDER_ROBUSTNESS_METHODS,
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_runs(root: Path, prefix: str) -> list[Path]:
    runs = sorted(path for path in root.glob(f"run_{prefix}_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs matching run_{prefix}_* under {root}")
    return runs


def run_record(path: Path) -> dict[str, Any]:
    run_manifest = load(path / "evaluation" / "run_manifest.json")
    completion = load(path / "evaluation" / "completion.json")
    release = Path(run_manifest["release_path"])
    release_manifest = load(release / "manifest.json")
    if completion.get("task_count") != 22:
        raise ValueError(f"Task coverage mismatch: {path}")
    if not str(completion.get("status", "")).startswith("completed"):
        raise ValueError(f"Incomplete run: {path}")
    expected = int(release_manifest["sequence_cases"])
    for method in completion["methods"]:
        values = [row for row in completion["counts"] if row["method"] == method]
        if len(values) != 22 or sum(row["rows"] for row in values) != expected:
            raise ValueError(f"Prediction coverage mismatch: {path}/{method}")
    return {
        "path": path,
        "manifest": run_manifest,
        "completion": completion,
        "release": release,
        "release_manifest": release_manifest,
        "regime": release_manifest["sequence_regime"],
        "order_seed": int(release_manifest["order_seed"]),
        "methods": tuple(completion["methods"]),
        "experience_k": int(run_manifest["memory"]["experience_k"]),
    }


def native_primary_rows(
    record: dict[str, Any],
    *,
    subset: str = "all",
    methods: Iterable[str] | None = None,
):
    roots = {
        "all": "metrics",
        "group_first": "metrics_group_first",
        "group_later": "metrics_group_later",
    }
    if subset not in roots:
        raise ValueError(f"Unsupported metric subset: {subset}")
    root = record["path"] / roots[subset]
    task_records = {row["task"]: row for row in record["release_manifest"]["tasks"]}
    output = []
    selected_methods = tuple(record["methods"] if methods is None else methods)
    missing_methods = set(selected_methods) - set(record["methods"])
    if missing_methods:
        raise ValueError(f"Methods not present in run: {sorted(missing_methods)}")
    for method in selected_methods:
        summaries = {}
        for family in ("unified", "process"):
            summaries.update(load(root / method / f"{family}.json")["summary"])
        if subset != "group_later" and set(summaries) != set(task_records):
            raise ValueError(f"Metric task coverage mismatch: {record['path']}/{method}")
        if not summaries or not set(summaries).issubset(task_records):
            raise ValueError(f"Invalid subset task coverage: {record['path']}/{method}")
        for task in sorted(summaries):
            task_record = task_records[task]
            metrics = metric_rows(
                task, summaries[task], process=task_record["family"] == "process"
            )
            name, value, higher = metrics[0]
            output.append({
                "run": str(record["path"]),
                "regime": record["regime"],
                "order_seed": record["order_seed"],
                "experience_k": record["experience_k"],
                "subset": subset,
                "family": task_record["family"],
                "task": task,
                "method": method,
                "metric": name,
                "value": value,
                "higher_is_better": higher,
            })
    return output


def oriented_delta(left: float | None, right: float | None, higher: bool):
    if left is None or right is None:
        return None
    value = float(left) - float(right)
    return value if higher else -value


def paired_rows(rows: list[dict[str, Any]], left: str, right: str):
    index = {
        (row["regime"], row["order_seed"], row["subset"], row["task"], row["method"]): row
        for row in rows
    }
    output = []
    for key, left_row in sorted(index.items()):
        regime, seed, subset, task, method = key
        if method != left:
            continue
        right_row = index.get((regime, seed, subset, task, right))
        if right_row is None or left_row["metric"] != right_row["metric"]:
            continue
        delta = oriented_delta(
            left_row["value"], right_row["value"], left_row["higher_is_better"]
        )
        denominator = max(abs(float(right_row["value"])), 1e-12)
        output.append({
            "regime": regime,
            "order_seed": seed,
            "subset": subset,
            "task": task,
            "metric": left_row["metric"],
            "left": left,
            "right": right,
            "left_value": left_row["value"],
            "right_value": right_row["value"],
            "oriented_delta": delta,
            "normalized_oriented_delta": None if delta is None else delta / denominator,
        })
    return output


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_summary(values: list[dict[str, Any]], *, samples: int = 10000):
    scored = [row for row in values if row["oriented_delta"] is not None]
    if not scored:
        return {"n": 0}
    observed_win = fmean(float(row["oriented_delta"] > 0) for row in scored)
    observed_norm = median(row["normalized_oriented_delta"] for row in scored)
    rng = random.Random(0)
    win_boot, norm_boot = [], []
    for _ in range(samples):
        resampled = [scored[rng.randrange(len(scored))] for _ in scored]
        win_boot.append(fmean(float(row["oriented_delta"] > 0) for row in resampled))
        norm_boot.append(median(row["normalized_oriented_delta"] for row in resampled))
    return {
        "n": len(scored),
        "wins": sum(row["oriented_delta"] > 0 for row in scored),
        "ties": sum(abs(row["oriented_delta"]) <= 1e-12 for row in scored),
        "losses": sum(row["oriented_delta"] < 0 for row in scored),
        "win_rate": observed_win,
        "win_rate_bootstrap_95ci": [
            percentile(win_boot, 0.025), percentile(win_boot, 0.975)
        ],
        "median_normalized_oriented_delta": observed_norm,
        "median_normalized_delta_bootstrap_95ci": [
            percentile(norm_boot, 0.025), percentile(norm_boot, 0.975)
        ],
    }


def temporal_pairwise(record: dict[str, Any], left: str, right: str):
    if left not in record["methods"] or right not in record["methods"]:
        return []
    output = []
    for task_record in record["release_manifest"]["tasks"]:
        task = task_record["task"]
        family = task_record["family"]
        left_rows = read_jsonl(
            record["path"] / "evaluation" / "test" / left / family / task / "predictions.jsonl"
        )
        right_rows = read_jsonl(
            record["path"] / "evaluation" / "test" / right / family / task / "predictions.jsonl"
        )
        if [row["case_id"] for row in left_rows] != [row["case_id"] for row in right_rows]:
            raise ValueError(f"Temporal alignment mismatch: {record['path']}/{task}")
        gains = []
        for lrow, rrow in zip(left_rows, right_rows):
            lscore = lrow.get("feedback", {}).get("score")
            rscore = rrow.get("feedback", {}).get("score")
            gains.append(
                None if lscore is None or rscore is None else float(lscore) - float(rscore)
            )
        window = max(1, math.ceil(len(gains) / 4))
        early_values = [value for value in gains[:window] if value is not None]
        late_values = [value for value in gains[-window:] if value is not None]
        early = fmean(early_values) if early_values else None
        late = fmean(late_values) if late_values else None
        output.append({
            "regime": record["regime"],
            "order_seed": record["order_seed"],
            "task": task,
            "left": left,
            "right": right,
            "mean_paired_generic_gain": fmean(
                value for value in gains if value is not None
            ),
            "early_quartile_paired_gain": early,
            "late_quartile_paired_gain": late,
            "late_minus_early_paired_gain": (
                None if early is None or late is None else late - early
            ),
            "window_cases": window,
        })
    return output


def response_coverage(records: Iterable[dict[str, Any]]):
    output = []
    for record in records:
        for method in record["methods"]:
            rows = [
                row for row in record["completion"]["counts"] if row["method"] == method
            ]
            output.append({
                "run": str(record["path"]),
                "regime": record["regime"],
                "order_seed": record["order_seed"],
                "experience_k": record["experience_k"],
                "method": method,
                "rows": sum(row["rows"] for row in rows),
                "success": sum(row["success"] for row in rows),
                "failed": sum(row["failed"] for row in rows),
                "retried_success": sum(row.get("retried_success", 0) for row in rows),
            })
    return output


def specialist_summary(path: Path | None):
    if path is None:
        return None
    payload = load(path)
    if payload.get("status") != "completed" or payload.get("task_count") != 22:
        raise ValueError("Specialized matched-cohort report is incomplete")
    return payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--core-run", type=Path, required=True)
    parser.add_argument("--order-root", type=Path, required=True)
    parser.add_argument("--order-array-job-id", required=True)
    parser.add_argument("--budget-root", type=Path, required=True)
    parser.add_argument("--budget-array-job-ids", nargs="+", required=True)
    parser.add_argument("--specialized-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = load(args.suite / "manifest.json")
    if suite.get("status") != "frozen" or suite.get("stream_count") != 10:
        raise ValueError("Expected the frozen ten-stream memory-ablation suite")

    core = run_record(args.core_run)
    if core["regime"] != "iid" or core["order_seed"] != 0:
        raise ValueError("Core run must use iid seed 0")
    if core["methods"] != METHODS:
        raise ValueError("Core method coverage mismatch")
    order_runs = [run_record(path) for path in discover_runs(
        args.order_root, args.order_array_job_id
    )]
    if len(order_runs) != 9:
        raise ValueError(f"Expected nine order runs, found {len(order_runs)}")
    if any(record["methods"] != ORDER_ROBUSTNESS_METHODS for record in order_runs):
        raise ValueError("Order-robustness method coverage mismatch")
    expected_streams = {
        *(('iid', seed) for seed in (1, 2, 3, 4)),
        *(('group_blocked', seed) for seed in (0, 1, 2, 3, 4)),
    }
    if {(record["regime"], record["order_seed"]) for record in order_runs} != expected_streams:
        raise ValueError("Order stream coverage mismatch")
    budget_paths = []
    for prefix in args.budget_array_job_ids:
        budget_paths.extend(discover_runs(args.budget_root, prefix))
    if len(set(budget_paths)) != len(budget_paths):
        raise ValueError("Duplicate memory-budget run paths")
    budget_runs = [run_record(path) for path in sorted(budget_paths)]
    expected_budget_k = {2, 4, 6, 8, 10, 12}
    if (
        len(budget_runs) != len(expected_budget_k)
        or {record["experience_k"] for record in budget_runs} != expected_budget_k
    ):
        raise ValueError(
            "Expected one memory-budget run at each of k=2,4,6,8,10,12"
        )
    if any(record["methods"] != ("matrecall",) for record in budget_runs):
        raise ValueError("Budget method coverage mismatch")

    robustness_runs = [core, *order_runs]
    core_rows = native_primary_rows(core)
    all_rows = []
    group_rows = []
    later_rows = []
    for record in robustness_runs:
        rows = native_primary_rows(record)
        all_rows.extend(
            row for row in rows if row["method"] in ORDER_ROBUSTNESS_METHODS
        )
        subset_rows = native_primary_rows(record, subset="group_first")
        group_rows.extend(
            row for row in subset_rows if row["method"] in ORDER_ROBUSTNESS_METHODS
        )
        later_rows.extend(native_primary_rows(
            record,
            subset="group_later",
            methods=("react_static_memory", "matrecall"),
        ))

    later_task_sets = {}
    for row in later_rows:
        key = (row["regime"], row["order_seed"], row["method"])
        later_task_sets.setdefault(key, set()).add(row["task"])
    expected_later_keys = {
        (record["regime"], record["order_seed"], method)
        for record in robustness_runs
        for method in ("react_static_memory", "matrecall")
    }
    if set(later_task_sets) != expected_later_keys:
        raise ValueError("Experience-subset stream or method coverage mismatch")
    distinct_later_task_sets = {frozenset(tasks) for tasks in later_task_sets.values()}
    if len(distinct_later_task_sets) != 1:
        raise ValueError("Experience-subset task coverage differs across streams")
    repeated_group_tasks = set(next(iter(distinct_later_task_sets)))
    if not repeated_group_tasks:
        raise ValueError("No tasks contain repeated same-group queries")

    experience_rows = []
    for row in group_rows:
        if (
            row["task"] in repeated_group_tasks
            and row["method"] in ("react_static_memory", "matrecall")
        ):
            experience_rows.append({**row, "subset": "no_prior_same_group"})
    for row in later_rows:
        if row["task"] in repeated_group_tasks:
            experience_rows.append({**row, "subset": "prior_same_group"})
    budget_rows = [
        row for record in budget_runs for row in native_primary_rows(record)
    ]

    core_comparisons = {
        method: bootstrap_summary(paired_rows(core_rows, "matrecall", method))
        for method in METHODS if method != "matrecall"
    }
    order_comparisons = {
        comparator: bootstrap_summary(
            paired_rows(all_rows, "matrecall", comparator)
        )
        for comparator in ("react_static_memory", "react_case_only")
    }
    group_comparisons = {
        comparator: bootstrap_summary(
            paired_rows(group_rows, "matrecall", comparator)
        )
        for comparator in ("react_static_memory", "react_case_only")
    }
    temporal = []
    for record in robustness_runs:
        for comparator in ("react_static_memory", "react_case_only"):
            temporal.extend(temporal_pairwise(record, "matrecall", comparator))

    budget_lookup = {}
    for row in budget_rows:
        budget_lookup[(row["experience_k"], row["task"])] = row
    budget_comparisons = {}
    budget_k_values = sorted({row["experience_k"] for row in budget_rows})
    comparison_pairs = [(left, 2) for left in budget_k_values if left != 2]
    comparison_pairs.extend(zip(budget_k_values[1:], budget_k_values[:-1]))
    for left, right in dict.fromkeys(comparison_pairs):
        pairs = []
        for task in sorted({row["task"] for row in budget_rows}):
            lrow, rrow = budget_lookup[(left, task)], budget_lookup[(right, task)]
            delta = oriented_delta(
                lrow["value"], rrow["value"], lrow["higher_is_better"]
            )
            pairs.append({
                "task": task,
                "oriented_delta": delta,
                "normalized_oriented_delta": delta / max(abs(float(rrow["value"])), 1e-12),
            })
        budget_comparisons[f"k{left}_vs_k{right}"] = bootstrap_summary(pairs)

    coverage = response_coverage([core, *order_runs, *budget_runs])
    payload = {
        "release_id": suite["release_id"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": 22,
        "stream_count": 10,
        "core_methods": list(METHODS),
        "order_robustness_methods": list(ORDER_ROBUSTNESS_METHODS),
        "core_ablation_matrecall_comparisons": core_comparisons,
        "order_robustness_matrecall_comparisons": order_comparisons,
        "group_first_matrecall_comparisons": group_comparisons,
        "experience_accumulation": {
            "task_count": len(repeated_group_tasks),
            "tasks": sorted(repeated_group_tasks),
            "subsets": ["no_prior_same_group", "prior_same_group"],
            "comparison": "matrecall_vs_react_static_memory",
        },
        "memory_budget_k_values": budget_k_values,
        "memory_budget_comparisons": budget_comparisons,
        "response_coverage": coverage,
        "specialized_matched_cohort": specialist_summary(args.specialized_summary),
        "reporting_rules": {
            "native_metrics": "Primary task-native metrics are never averaged in raw units.",
            "bootstrap": "CIs resample task-stream comparisons with deterministic seed 0.",
            "group_first": "A group-first query is predicted before feedback from the same material or process group exists.",
            "experience_accumulation": "The no-prior and prior subsets are disjoint and use the same tasks that contain repeated group members.",
            "failure": "Response coverage and retry recovery are reported separately; failures are not silently replaced.",
        },
        "provenance": {
            "suite_manifest_sha256": file_sha256(args.suite / "manifest.json"),
            "core_completion_sha256": file_sha256(core["path"] / "evaluation" / "completion.json"),
            "order_completion_sha256": {
                str(record["path"]): file_sha256(record["path"] / "evaluation" / "completion.json")
                for record in order_runs
            },
            "budget_completion_sha256": {
                str(record["path"]): file_sha256(record["path"] / "evaluation" / "completion.json")
                for record in budget_runs
            },
            "reporter_sha256": file_sha256(Path(__file__)),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output / "core_primary_metrics.jsonl", core_rows)
    write_jsonl(args.output / "order_primary_metrics.jsonl", all_rows)
    write_jsonl(args.output / "group_first_primary_metrics.jsonl", group_rows)
    write_jsonl(args.output / "experience_primary_metrics.jsonl", experience_rows)
    write_jsonl(args.output / "temporal_paired_gains.jsonl", temporal)
    write_jsonl(args.output / "memory_budget_primary_metrics.jsonl", budget_rows)

    lines = [
        "# MatRecall memory-ablation results", "",
        "The report holds the ReAct controller, static evidence, model, budgets, and retry policy fixed while changing writable-memory content or feedback timing.", "",
        "## Core memory-content and feedback ablations", "",
        "| Comparator | n task comparisons | Full wins | ties | losses | win rate (95% bootstrap CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, value in core_comparisons.items():
        ci = value["win_rate_bootstrap_95ci"]
        lines.append(
            f"| `{method}` | {value['n']} | {value['wins']} | {value['ties']} | {value['losses']} | "
            f"{value['win_rate']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) |"
        )
    lines.extend(("", "## Multi-order robustness", ""))
    for method, value in order_comparisons.items():
        ci = value["win_rate_bootstrap_95ci"]
        lines.append(
            f"- Full versus `{method}` across task–stream pairs: {value['wins']} wins, "
            f"{value['losses']} losses; win rate {value['win_rate']:.3f} "
            f"(95% bootstrap CI {ci[0]:.3f}–{ci[1]:.3f})."
        )
    lines.extend(("", "## Group-first challenge", ""))
    for method, value in group_comparisons.items():
        ci = value["win_rate_bootstrap_95ci"]
        lines.append(
            f"- Full versus `{method}` before same-group feedback: {value['wins']} wins, "
            f"{value['losses']} losses; win rate {value['win_rate']:.3f} "
            f"(95% bootstrap CI {ci[0]:.3f}–{ci[1]:.3f})."
        )
    lines.extend((
        "", "Detailed native metrics, paired temporal gains, memory-budget results, response coverage, retry recovery, provenance hashes, and the matched frozen-specialist comparison are stored beside this summary.",
    ))
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed", "tasks": 22, "streams": 10,
        "core_methods": len(METHODS),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
