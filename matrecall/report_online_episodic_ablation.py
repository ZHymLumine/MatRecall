"""Report selected-task temporal episodic-only supplementation to online adaptation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from matrecall.episodic_memory import write_jsonl
from matrecall.report_memory_retention import fmt, oriented_delta
from matrecall.report_online_adaptation import sequential_summary
from matrecall.report_common import metric_rows
from matrecall.run_frozen_memory import EPISODIC_ONLY_METHODS
from matrecall.run_online_adaptation import BASE_METHODS, file_sha256


ALL_METHODS = (*BASE_METHODS, *EPISODIC_ONLY_METHODS)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--episodic-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = load(args.release / "manifest.json")
    completion = load(args.episodic_run / "evaluation" / "completion.json")
    if tuple(completion.get("methods", ())) != EPISODIC_ONLY_METHODS:
        raise ValueError("Temporal episodic-only coverage mismatch")
    if not str(completion.get("status", "")).startswith("completed"):
        raise ValueError("Temporal episodic-only run is incomplete")
    selected = completion["counts"]
    tasks = sorted({row["task"] for row in selected})
    records = {row["task"]: row for row in manifest["tasks"] if row["task"] in tasks}
    if set(records) != set(tasks):
        raise ValueError("Selected task metadata mismatch")

    native: dict[str, dict[str, Any]] = {method: {} for method in ALL_METHODS}
    metric_paths = {}
    for method in BASE_METHODS:
        for family in ("unified", "process"):
            path = args.base_run / "metrics" / method / f"{family}.json"
            metric_paths[f"{method}/{family}"] = path
            native[method].update(load(path)["summary"])
    for method in EPISODIC_ONLY_METHODS:
        for family in sorted({records[task]["family"] for task in tasks}):
            path = args.episodic_run / "metrics" / method / f"{family}.json"
            metric_paths[f"{method}/{family}"] = path
            if path.exists():
                native[method].update(load(path)["summary"])

    rows = []
    temporal = {}
    for task in tasks:
        meta = records[task]
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
        primary = metric_rows(
            task, native["matrecall"][task], process=process
        )[0][0]
        higher = per_method["matrecall"][primary][1]
        values = {
            method: per_method[method].get(primary, (None, higher))[0]
            for method in ALL_METHODS
        }
        temporal[task] = {}
        for method in ALL_METHODS:
            root = args.base_run if method in BASE_METHODS else args.episodic_run
            path = (
                root / "evaluation" / "test" / method / meta["family"] / task /
                "predictions.jsonl"
            )
            temporal[task][method] = sequential_summary(read_jsonl(path))
        case_alone = oriented_delta(
            values["case_episodic_only"], values["llm_only"], higher
        )
        case_marginal = oriented_delta(
            values["dynamic_case_rag"], values["static_multiview_rag"], higher
        )
        full_alone = oriented_delta(
            values["full_episodic_only"], values["llm_only"], higher
        )
        full_marginal = oriented_delta(
            values["matrecall"], values["react_static_memory"], higher
        )
        rows.append({
            "family": meta["family"], "task": task,
            "sequence_cases": meta["sequence_cases"],
            "primary_metric": primary, "higher_is_better": higher,
            "values": values,
            "case_episodic_only_gain_vs_llm": case_alone,
            "case_memory_marginal_gain_over_static": case_marginal,
            "case_static_episode_synergy": case_marginal - case_alone,
            "full_episodic_only_gain_vs_llm": full_alone,
            "full_memory_marginal_gain_over_react_static": full_marginal,
            "full_static_episode_synergy": full_marginal - full_alone,
        })

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "task_metric_table.jsonl", rows)
    payload: dict[str, Any] = {
        "release_id": manifest["release_id"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks), "tasks": tasks,
        "methods": list(ALL_METHODS),
        "task_metrics": rows,
        "temporal_learning_curves": temporal,
        "selection_rule": (
            "Tasks were fixed before episodic-only inference to cover crystal regression "
            "with positive and negative prior Full-MatRecall effects, structured phase, "
            "synthesis numeric and generation, and experimental binary outcome tasks."
        ),
        "provenance": {
            "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
            "episodic_completion_sha256": file_sha256(
                args.episodic_run / "evaluation" / "completion.json"
            ),
            "metric_files": {name: file_sha256(path) for name, path in metric_paths.items()},
            "reporter_sha256": file_sha256(Path(__file__)),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MatRecall online-adaptation episodic-only supplement", "",
        f"Selected tasks: {len(tasks)}.", "",
        "| Task | n | Metric | LLM | Static | Case only | Static+case | ReAct static | Full only | Static+full |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = row["values"]
        lines.append(
            f"| `{row['task']}` | {row['sequence_cases']} | {row['primary_metric']} | "
            f"{fmt(value['llm_only'])} | {fmt(value['static_multiview_rag'])} | "
            f"{fmt(value['case_episodic_only'])} | {fmt(value['dynamic_case_rag'])} | "
            f"{fmt(value['react_static_memory'])} | {fmt(value['full_episodic_only'])} | "
            f"{fmt(value['matrecall'])} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "tasks": len(tasks), "methods": len(ALL_METHODS)}))


if __name__ == "__main__":
    main()
