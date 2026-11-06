"""Report the memory-retention episodic-only additions beside the frozen base run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from matrecall.episodic_memory import write_jsonl
from matrecall.report_memory_retention import (
    fmt,
    load,
    oriented_delta,
    select_report_tasks,
)
from matrecall.report_common import metric_rows
from matrecall.run_frozen_memory import (
    BASE_METHODS,
    EPISODIC_ONLY_METHODS,
)
from matrecall.run_online_adaptation import file_sha256


ALL_METHODS = (*BASE_METHODS, *EPISODIC_ONLY_METHODS, "specialized_model")


def _load_native(run: Path, methods: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Path]]:
    native: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for method in methods:
        native[method] = {}
        for family in ("unified", "process"):
            path = run / "metrics" / method / f"{family}.json"
            paths[f"{method}/{family}"] = path
            if path.exists():
                native[method].update(load(path)["summary"])
    return native, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--episodic-run", type=Path, required=True)
    parser.add_argument("--specialized-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-task", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = load(args.release / "manifest.json")
    base_completion = load(args.base_run / "evaluation" / "completion.json")
    episodic_completion = load(args.episodic_run / "evaluation" / "completion.json")
    if tuple(base_completion.get("methods", ())) != BASE_METHODS:
        raise ValueError("Base memory-retention method coverage mismatch")
    if tuple(episodic_completion.get("methods", ())) != EPISODIC_ONLY_METHODS:
        raise ValueError("Episodic-only method coverage mismatch")
    for completion in (base_completion, episodic_completion):
        if not str(completion.get("status", "")).startswith("completed"):
            raise ValueError("Transfer run is incomplete")
        if completion.get("memory_writes") != 0 or completion.get("transfer_feedback_reveals") != 0:
            raise ValueError("Frozen-memory policy was violated")

    tasks = select_report_tasks(manifest["tasks"], set(args.exclude_task))
    expected_all = set(tasks) | set(args.exclude_task)
    expected_reported = set(tasks)
    native, paths = _load_native(args.base_run, BASE_METHODS)
    episodic_native, episodic_paths = _load_native(
        args.episodic_run, EPISODIC_ONLY_METHODS
    )
    native.update(episodic_native)
    paths.update(episodic_paths)
    native["specialized_model"] = {}
    for family in ("unified", "process"):
        path = args.specialized_run / "metrics" / f"{family}.json"
        paths[f"specialized_model/{family}"] = path
        if path.exists():
            native["specialized_model"].update(load(path)["summary"])
    for method in ALL_METHODS:
        expected = (
            expected_reported if method in EPISODIC_ONLY_METHODS else expected_all
        )
        if set(native[method]) != expected:
            raise ValueError(f"Native task coverage mismatch: {method}")

    rows = []
    effects = []
    for task, meta in tasks.items():
        process = meta["family"] == "process"
        metrics = {
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
        higher = metrics["matrecall"][primary][1]
        values = {
            method: metrics[method].get(primary, (None, higher))[0]
            for method in ALL_METHODS
        }
        for method in ALL_METHODS:
            for name, (value, metric_higher) in metrics[method].items():
                rows.append({
                    "family": meta["family"], "task": task, "metric": name,
                    "higher_is_better": metric_higher, "method": method,
                    "value": value, "transfer_cases": meta["transfer_cases"],
                })
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
        effects.append({
            "family": meta["family"], "task": task,
            "primary_metric": primary, "higher_is_better": higher,
            "transfer_cases": meta["transfer_cases"],
            "case_episodic_only_gain_vs_llm": case_alone,
            "case_memory_marginal_gain_over_static": case_marginal,
            "case_static_episode_synergy": (
                None if case_alone is None or case_marginal is None
                else case_marginal - case_alone
            ),
            "full_episodic_only_gain_vs_llm": full_alone,
            "full_memory_marginal_gain_over_react_static": full_marginal,
            "full_static_episode_synergy": (
                None if full_alone is None or full_marginal is None
                else full_marginal - full_alone
            ),
            "values": values,
        })

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "native_metric_table.jsonl", rows)
    write_jsonl(args.output / "episodic_factor_effects.jsonl", effects)
    payload = {
        "release_id": manifest["release_id"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
        "analysis_excluded_tasks": sorted(set(args.exclude_task)),
        "transfer_cases_per_method": sum(row["transfer_cases"] for row in tasks.values()),
        "method_count": len(ALL_METHODS),
        "methods": list(ALL_METHODS),
        "episodic_factor_effects": effects,
        "factor_definitions": {
            "case_episodic_only": "case_episodic_only minus llm_only",
            "case_marginal_over_static": "dynamic_case_rag minus static_multiview_rag",
            "case_synergy": "case_marginal_over_static minus case_episodic_only",
            "full_episodic_only": "full_episodic_only minus llm_only",
            "full_marginal_over_static_planner": "matrecall minus react_static_memory",
            "full_synergy": "full_marginal_over_static_planner minus full_episodic_only",
        },
        "provenance": {
            "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
            "base_completion_sha256": file_sha256(
                args.base_run / "evaluation" / "completion.json"
            ),
            "episodic_completion_sha256": file_sha256(
                args.episodic_run / "evaluation" / "completion.json"
            ),
            "metric_files": {name: file_sha256(path) for name, path in paths.items()},
            "reporter_sha256": file_sha256(Path(__file__)),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    lookup = {
        (row["task"], row["metric"], row["method"]): row["value"] for row in rows
    }
    lines = [
        "# MatRecall memory-retention episodic-only ablation", "",
        f"Tasks: {len(tasks)}; cases per method: {payload['transfer_cases_per_method']}.", "",
        "| Task | n | Metric | LLM | Static | Case only | Static+case | ReAct static | Full only | Static+full | Specialized |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for effect in effects:
        task, metric = effect["task"], effect["primary_metric"]
        lines.append(
            f"| `{task}` | {effect['transfer_cases']} | {metric} | "
            f"{fmt(lookup.get((task, metric, 'llm_only')))} | "
            f"{fmt(lookup.get((task, metric, 'static_multiview_rag')))} | "
            f"{fmt(lookup.get((task, metric, 'case_episodic_only')))} | "
            f"{fmt(lookup.get((task, metric, 'dynamic_case_rag')))} | "
            f"{fmt(lookup.get((task, metric, 'react_static_memory')))} | "
            f"{fmt(lookup.get((task, metric, 'full_episodic_only')))} | "
            f"{fmt(lookup.get((task, metric, 'matrecall')))} | "
            f"{fmt(lookup.get((task, metric, 'specialized_model')))} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "tasks": len(tasks), "methods": len(ALL_METHODS)}))


if __name__ == "__main__":
    main()
