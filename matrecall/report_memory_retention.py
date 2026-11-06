"""Report native metrics for the memory-retention (frozen-memory transfer) experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from matrecall.episodic_memory import write_jsonl
from matrecall.report_common import metric_rows
from matrecall.run_frozen_memory import BASE_METHODS
from matrecall.run_online_adaptation import file_sha256


METHODS = BASE_METHODS
ALL_METHODS = (*BASE_METHODS, "specialized_model")


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def oriented_delta(left: float | None, right: float | None, higher: bool) -> float | None:
    if left is None or right is None:
        return None
    value = left - right
    return value if higher else -value


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def select_report_tasks(
    manifest_tasks: list[dict[str, Any]], excluded_tasks: set[str]
) -> dict[str, dict[str, Any]]:
    """Select tasks for a derived report without altering the frozen cohort."""

    available = {row["task"] for row in manifest_tasks}
    unknown = excluded_tasks - available
    if unknown:
        raise ValueError(f"Unknown report exclusions: {sorted(unknown)}")
    selected = {
        row["task"]: row
        for row in manifest_tasks
        if row["task"] not in excluded_tasks
    }
    if not selected:
        raise ValueError("Report exclusions removed every transfer task")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--specialized-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help=(
            "Exclude a task from this derived analysis report while preserving the "
            "frozen release and raw predictions. May be repeated."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = load(args.release / "manifest.json")
    completion = load(args.run / "evaluation" / "completion.json")
    metric_join = load(args.run / "metric_inputs" / "manifest.json")
    specialized_verification = load(args.specialized_run / "matched" / "manifest.json")
    if manifest.get("status") != "frozen" or manifest.get("task_count", 0) < 1:
        raise ValueError("Transfer release is not frozen")
    if not str(completion.get("status", "")).startswith("completed"):
        raise ValueError("Transfer run is incomplete")
    if completion.get("methods") != list(METHODS):
        raise ValueError("Transfer method coverage mismatch")
    if completion.get("memory_writes") != 0 or completion.get("transfer_feedback_reveals") != 0:
        raise ValueError("Transfer run violated frozen-memory policy")
    if metric_join.get("gold_joined_after_prediction_completion") is not True:
        raise ValueError("Gold was not joined through the post-prediction gate")
    if specialized_verification.get("online_updates") is not False:
        raise ValueError("Specialized reference is not frozen")
    if specialized_verification.get("case_count") != manifest["transfer_cases"]:
        raise ValueError("Specialized transfer cohort count mismatch")

    all_tasks = {row["task"]: row for row in manifest["tasks"]}
    expected_tasks = set(all_tasks)
    report_excluded_tasks = set(args.exclude_task)
    tasks = select_report_tasks(manifest["tasks"], report_excluded_tasks)
    native: dict[str, dict[str, dict[str, Any]]] = {}
    metric_paths = {}
    for method in METHODS:
        native[method] = {}
        for family in ("unified", "process"):
            path = args.run / "metrics" / method / f"{family}.json"
            metric_paths[f"{method}/{family}"] = path
            if path.exists():
                native[method].update(load(path)["summary"])
        if set(native[method]) != expected_tasks:
            raise ValueError(f"Native task coverage mismatch: {method}")
    native["specialized_model"] = {}
    for family in ("unified", "process"):
        path = args.specialized_run / "metrics" / f"{family}.json"
        metric_paths[f"specialized_model/{family}"] = path
        if path.exists():
            native["specialized_model"].update(load(path)["summary"])
    if set(native["specialized_model"]) != expected_tasks:
        raise ValueError("Specialized native task coverage mismatch")

    native_rows = []
    comparisons = []
    baselines = [method for method in ALL_METHODS if method != "matrecall"]
    win_counts = {
        method: {"wins": 0, "ties": 0, "losses": 0, "unscored": 0}
        for method in baselines
    }
    subset_win_counts = {
        subset: {
            method: {"wins": 0, "ties": 0, "losses": 0, "unscored": 0}
            for method in baselines
        }
        for subset in ("strict_primary", "strict_minimum_10")
    }
    gap_rows = []
    for task, meta in tasks.items():
        process = meta["family"] == "process"
        per_method = {
            method: {
                name: (value, higher)
                for name, value, higher in metric_rows(
                    task, native[method][task], process=process
                )
            }
            for method in ALL_METHODS
        }
        primary_name = metric_rows(
            task, native["matrecall"][task], process=process
        )[0][0]
        higher = per_method["matrecall"][primary_name][1]
        for method in ALL_METHODS:
            for name, (value, metric_higher) in per_method[method].items():
                native_rows.append({
                    "family": meta["family"], "task": task,
                    "metric": name, "higher_is_better": metric_higher,
                    "method": method, "value": value,
                    "transfer_cases": meta["transfer_cases"],
                    "primary_strict_group_disjoint": meta["primary_strict_group_disjoint"],
                    "low_sample_transfer": meta["low_sample_transfer"],
                })
        full_value = per_method["matrecall"].get(primary_name, (None, higher))[0]
        comparison = {
            "family": meta["family"], "task": task,
            "primary_metric": primary_name, "higher_is_better": higher,
            "transfer_cases": meta["transfer_cases"],
            "primary_strict_group_disjoint": meta["primary_strict_group_disjoint"],
            "low_sample_transfer": meta["low_sample_transfer"],
            "matrecall_oriented_delta": {},
        }
        for method in baselines:
            baseline_value = per_method[method].get(primary_name, (None, higher))[0]
            delta = oriented_delta(full_value, baseline_value, higher)
            comparison["matrecall_oriented_delta"][method] = delta
            scopes = [win_counts[method]]
            if meta["primary_strict_group_disjoint"]:
                scopes.append(subset_win_counts["strict_primary"][method])
                if not meta["low_sample_transfer"]:
                    scopes.append(subset_win_counts["strict_minimum_10"][method])
            for counts in scopes:
                if delta is None:
                    counts["unscored"] += 1
                elif abs(delta) <= 1e-12:
                    counts["ties"] += 1
                elif delta > 0:
                    counts["wins"] += 1
                else:
                    counts["losses"] += 1
        llm_value = per_method["llm_only"].get(primary_name, (None, higher))[0]
        specialized_value = per_method["specialized_model"].get(
            primary_name, (None, higher)
        )[0]
        llm_to_specialized = oriented_delta(specialized_value, llm_value, higher)
        llm_to_full = oriented_delta(full_value, llm_value, higher)
        fraction = None
        if (
            llm_to_specialized is not None and llm_to_specialized > 0
            and llm_to_full is not None
        ):
            fraction = llm_to_full / llm_to_specialized
        gap_rows.append({
            "family": meta["family"], "task": task,
            "primary_metric": primary_name,
            "llm_to_specialized_oriented_gap": llm_to_specialized,
            "llm_to_full_oriented_gain": llm_to_full,
            "specialized_gap_closed_fraction": fraction,
            "full_reaches_or_exceeds_specialized": (
                None if full_value is None or specialized_value is None
                else oriented_delta(full_value, specialized_value, higher) >= 0
            ),
        })
        comparisons.append(comparison)

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "native_metric_table.jsonl", native_rows)
    write_jsonl(args.output / "task_comparisons.jsonl", comparisons)
    write_jsonl(args.output / "specialized_gap_analysis.jsonl", gap_rows)
    payload = {
        "release_id": manifest["release_id"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
        "source_task_count": manifest["task_count"],
        "excluded_task_count": manifest["excluded_task_count"],
        "analysis_excluded_task_count": len(report_excluded_tasks),
        "analysis_excluded_tasks": sorted(report_excluded_tasks),
        "transfer_cases_per_method": sum(
            row["transfer_cases"] for row in tasks.values()
        ),
        "source_transfer_cases_per_method": manifest["transfer_cases"],
        "method_count": len(ALL_METHODS),
        "methods": list(ALL_METHODS),
        "strict_primary_task_count": sum(
            row["primary_strict_group_disjoint"] for row in tasks.values()
        ),
        "strict_minimum_10_task_count": sum(
            row["primary_strict_group_disjoint"] and not row["low_sample_transfer"]
            for row in manifest["tasks"]
        ),
        "matrecall_task_win_counts": win_counts,
        "subset_win_counts": subset_win_counts,
        "primary_task_comparisons": comparisons,
        "specialized_gap_analysis": gap_rows,
        "frozen_memory_audit": {
            "memory_source": "completed online-adaptation MatRecall verified feedback memory",
            "memory_writes_during_transfer": completion["memory_writes"],
            "transfer_feedback_reveals": completion["transfer_feedback_reveals"],
            "gold_joined_after_prediction_completion": metric_join[
                "gold_joined_after_prediction_completion"
            ],
            "specialized_online_updates": specialized_verification["online_updates"],
        },
        "reporting_rule": (
            "Native metrics are reported per task and are not averaged across incompatible units. "
            "Task-level win counts are shown for all eligible tasks, the strict non-proxy subset, "
            "and the strict subset with at least ten transfer cases."
        ),
        "provenance": {
            "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
            "run_completion_sha256": file_sha256(
                args.run / "evaluation" / "completion.json"
            ),
            "metric_join_manifest_sha256": file_sha256(
                args.run / "metric_inputs" / "manifest.json"
            ),
            "specialized_verification_sha256": file_sha256(
                args.specialized_run / "matched" / "manifest.json"
            ),
            "metric_files": {
                name: file_sha256(path) for name, path in metric_paths.items()
            },
            "reporter_sha256": file_sha256(Path(__file__)),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    lookup = {
        (row["task"], row["metric"], row["method"]): row["value"]
        for row in native_rows
    }
    lines = [
        "# MatRecall memory-retention results", "",
        (
            f"Status: completed. Reported tasks: {len(tasks)}; "
            f"transfer cases per method: {sum(row['transfer_cases'] for row in tasks.values())}; "
            f"methods: {len(ALL_METHODS)}."
        ), "",
        (
            "Every transfer query used the same task-specific online-adaptation memory snapshot. "
            "No transfer answer was revealed and no episode was written during prediction."
        ), "",
        "| Family | Task | n | Primary metric | LLM-only | Static RAG | ReAct static | Rolling ICL | Dynamic case RAG | MatRecall | Specialized | Full−Dynamic | Full−Specialized |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        task, metric = row["task"], row["primary_metric"]
        delta = row["matrecall_oriented_delta"]
        lines.append(
            f"| {row['family']} | `{task}` | {row['transfer_cases']} | {metric} | "
            f"{fmt(lookup.get((task, metric, 'llm_only')))} | "
            f"{fmt(lookup.get((task, metric, 'static_multiview_rag')))} | "
            f"{fmt(lookup.get((task, metric, 'react_static_memory')))} | "
            f"{fmt(lookup.get((task, metric, 'rolling_context_icl')))} | "
            f"{fmt(lookup.get((task, metric, 'dynamic_case_rag')))} | "
            f"{fmt(lookup.get((task, metric, 'matrecall')))} | "
            f"{fmt(lookup.get((task, metric, 'specialized_model')))} | "
            f"{fmt(delta['dynamic_case_rag'])} | {fmt(delta['specialized_model'])} |"
        )
    lines.extend(("", "## MatRecall task-level primary-metric wins", ""))
    for scope, values in (
        ("All eligible tasks", win_counts),
        ("Strict non-proxy tasks", subset_win_counts["strict_primary"]),
        ("Strict tasks with at least 10 cases", subset_win_counts["strict_minimum_10"]),
    ):
        lines.append(f"### {scope}")
        lines.append("")
        for method, counts in values.items():
            lines.append(
                f"- Versus `{method}`: {counts['wins']} wins, {counts['ties']} ties, "
                f"{counts['losses']} losses, {counts['unscored']} unscored."
            )
        lines.append("")
    lines.extend((
        "## Scope notes", "",
        (
            f"{manifest['excluded_task_count']} source tasks had no group-disjoint cases after "
            "online-adaptation exposure and were excluded rather than weakened to ID-only separation."
        ),
        "The inverse-formulation grouping key is a proxy and is reported outside the strict-primary count.",
        (
            "Tasks excluded only from this derived analysis report: "
            + (", ".join(f"`{task}`" for task in sorted(report_excluded_tasks)) or "none")
            + ". Raw predictions and the frozen release remain unchanged."
        ),
    ))
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed", "tasks": len(tasks),
        "cases": sum(row["transfer_cases"] for row in tasks.values()),
        "methods": len(ALL_METHODS),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
