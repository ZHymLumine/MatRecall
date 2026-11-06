"""Report related versus unrelated frozen cross-task memory transfer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from matrecall.episodic_memory import write_jsonl
from matrecall.prepare_cross_task_metrics import ALL_METHODS
from matrecall.report_memory_retention import fmt, oriented_delta
from matrecall.report_common import metric_rows
from matrecall.run_online_adaptation import file_sha256


CROSS_METHODS = (
    "case_episodic_only", "full_episodic_only",
    "dynamic_case_rag", "matrecall",
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--metric-inputs", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = load(args.release / "manifest.json")
    join = load(args.metric_inputs / "manifest.json")
    if join.get("gold_joined_after_prediction_completion") is not True:
        raise ValueError("Cross-task gold join gate is missing")

    unit_metrics: dict[str, dict[str, Any]] = {}
    metric_paths = {}
    for unit in manifest["tasks"]:
        pair_id = unit["pair_id"]
        unit_metrics[pair_id] = {}
        for method in ALL_METHODS:
            path = args.metrics / pair_id / f"{method}.json"
            metric_paths[f"{pair_id}/{method}"] = path
            summary = load(path)["summary"]
            if set(summary) != {unit["target_task"]}:
                raise ValueError(f"Metric task mismatch: {pair_id}/{method}")
            unit_metrics[pair_id][method] = summary[unit["target_task"]]

    unit_rows = []
    by_target: dict[str, dict[str, dict[str, Any]]] = {}
    for unit in manifest["tasks"]:
        task = unit["target_task"]
        process = unit["family"] == "process"
        per_method = {
            method: {
                name: (value, higher)
                for name, value, higher in metric_rows(
                    task, unit_metrics[unit["pair_id"]][method], process=process
                )
            }
            for method in ALL_METHODS
        }
        primary = metric_rows(
            task, unit_metrics[unit["pair_id"]]["same_task_matrecall"],
            process=process,
        )[0][0]
        higher = per_method["same_task_matrecall"][primary][1]
        values = {
            method: per_method[method].get(primary, (None, higher))[0]
            for method in ALL_METHODS
        }
        row = {
            "pair_id": unit["pair_id"], "family": unit["family"],
            "target_task": task, "source_task": unit["source_task"],
            "memory_relation": unit["memory_relation"],
            "stratum": unit["stratum"], "rationale": unit["rationale"],
            "transfer_cases": unit["transfer_cases"],
            "primary_metric": primary, "higher_is_better": higher,
            "values": values,
            "oriented_gain_vs_no_episode_control": {
                "case_episodic_only": oriented_delta(
                    values["case_episodic_only"], values["llm_only"], higher
                ),
                "full_episodic_only": oriented_delta(
                    values["full_episodic_only"], values["llm_only"], higher
                ),
                "dynamic_case_rag": oriented_delta(
                    values["dynamic_case_rag"], values["static_multiview_rag"], higher
                ),
                "matrecall": oriented_delta(
                    values["matrecall"], values["react_static_memory"], higher
                ),
            },
        }
        unit_rows.append(row)
        by_target.setdefault(task, {})[unit["memory_relation"]] = row

    comparisons = []
    for target_record in manifest["target_pairs"]:
        task = target_record["target_task"]
        relation_rows = by_target.get(task, {})
        if set(relation_rows) != {"related", "unrelated"}:
            raise ValueError(f"Missing related/unrelated result: {task}")
        related, unrelated = relation_rows["related"], relation_rows["unrelated"]
        higher = related["higher_is_better"]
        comparisons.append({
            "target_task": task,
            "stratum": target_record["stratum"],
            "primary_metric": related["primary_metric"],
            "higher_is_better": higher,
            "transfer_cases": related["transfer_cases"],
            "related_source_task": related["source_task"],
            "unrelated_source_task": unrelated["source_task"],
            "related_minus_unrelated": {
                method: oriented_delta(
                    related["values"][method], unrelated["values"][method], higher
                )
                for method in CROSS_METHODS
            },
            "related_gain_vs_no_episode_control": related[
                "oriented_gain_vs_no_episode_control"
            ],
            "unrelated_gain_vs_no_episode_control": unrelated[
                "oriented_gain_vs_no_episode_control"
            ],
            "controls": {
                method: related["values"][method]
                for method in (
                    "llm_only", "static_multiview_rag", "react_static_memory",
                    "same_task_matrecall", "specialized_model",
                )
            },
        })

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "pair_metric_table.jsonl", unit_rows)
    write_jsonl(args.output / "related_unrelated_comparisons.jsonl", comparisons)
    payload = {
        "release_id": manifest["release_id"],
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_task_count": manifest["target_task_count"],
        "pair_count": manifest["pair_count"],
        "unique_target_cases": manifest["unique_target_cases"],
        "methods": list(ALL_METHODS),
        "related_unrelated_comparisons": comparisons,
        "interpretation_rule": (
            "The preregistered directional contrast is the oriented related-memory score "
            "minus the oriented unrelated-memory score on identical target queries."
        ),
        "provenance": {
            "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
            "metric_join_manifest_sha256": file_sha256(
                args.metric_inputs / "manifest.json"
            ),
            "metric_files": {name: file_sha256(path) for name, path in metric_paths.items()},
            "reporter_sha256": file_sha256(Path(__file__)),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MatRecall cross-task frozen transfer", "",
        f"Targets: {manifest['target_task_count']}; related/unrelated pairs: {manifest['pair_count']}.", "",
        "| Target | n | Metric | Related source | Unrelated source | Case-only related | Case-only unrelated | Full-only related | Full-only unrelated | Static+full related | Static+full unrelated | Same-task full | Specialized |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        task = comparison["target_task"]
        related, unrelated = by_target[task]["related"], by_target[task]["unrelated"]
        lines.append(
            f"| `{task}` | {comparison['transfer_cases']} | {comparison['primary_metric']} | "
            f"`{comparison['related_source_task']}` | `{comparison['unrelated_source_task']}` | "
            f"{fmt(related['values']['case_episodic_only'])} | "
            f"{fmt(unrelated['values']['case_episodic_only'])} | "
            f"{fmt(related['values']['full_episodic_only'])} | "
            f"{fmt(unrelated['values']['full_episodic_only'])} | "
            f"{fmt(related['values']['matrecall'])} | "
            f"{fmt(unrelated['values']['matrecall'])} | "
            f"{fmt(comparison['controls']['same_task_matrecall'])} | "
            f"{fmt(comparison['controls']['specialized_model'])} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": "completed", "targets": len(comparisons),
        "pairs": len(unit_rows), "methods": len(ALL_METHODS),
    }))


if __name__ == "__main__":
    main()
