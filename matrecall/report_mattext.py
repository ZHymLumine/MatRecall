"""Aggregate MatRecall x MatText task-native and transfer results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from matrecall.mattext import (
    file_sha256,
    read_jsonl,
    regression_metrics,
    scalar_prediction,
)


def _gold_maps(release: Path, manifest: dict[str, Any]):
    maps: dict[tuple[str, int, str], dict[str, float]] = {}
    units: dict[str, str] = {}
    for record in manifest["task_fold_records"]:
        task, fold = record["task"], int(record["fold"])
        source = release / record["relative_path"]
        for stage in ("exposure", "frozen_test"):
            rows = read_jsonl(source / stage / "gold.jsonl")
            maps[(task, fold, stage)] = {
                row["case_id"]: float(row["target"]["value"]) for row in rows
            }
            if rows:
                units[task] = str(rows[0]["target"]["unit"])
    return maps, units


def _prediction_files(root: Path, stage: str) -> list[Path]:
    return sorted(root.glob(f"*/{stage}/**/predictions.jsonl"))


def _collect(root: Path, stage: str, gold_maps):
    gold_stage = "frozen_test" if stage == "frozen" else stage
    groups: dict[tuple[Any, ...], dict[str, list[Any]]] = defaultdict(
        lambda: {"expected": [], "predicted": [], "steps": [], "case_ids": []}
    )
    failures: list[dict[str, Any]] = []
    for path in _prediction_files(root, stage):
        for row in read_jsonl(path):
            task, fold = row["task"], int(row["fold"])
            prediction = scalar_prediction(row)
            key = (
                task,
                row["method"],
                row.get("representation"),
                row.get("source_representation"),
                row.get("target_representation"),
            )
            if prediction is None:
                failures.append({
                    "path": str(path), "case_id": row["case_id"],
                    "task": task, "fold": fold, "method": row["method"],
                    "status": row.get("status"), "error": row.get("error"),
                })
                continue
            expected = gold_maps[(task, fold, gold_stage)][row["case_id"]]
            group = groups[key]
            group["expected"].append(expected)
            group["predicted"].append(prediction)
            group["steps"].append(row.get("sequence_step"))
            group["case_ids"].append(row["case_id"])
    records = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        task, method, representation, source_representation, target_representation = key
        record = {
            "stage": stage,
            "task": task,
            "method": method,
            "representation": representation,
            "source_representation": source_representation,
            "target_representation": target_representation,
            **regression_metrics(values["expected"], values["predicted"]),
        }
        records.append(record)
    return records, failures


def _temporal_records(root: Path, gold_maps):
    groups: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"expected": [], "predicted": []}
    )
    for path in _prediction_files(root, "exposure"):
        for row in read_jsonl(path):
            prediction = scalar_prediction(row)
            if prediction is None:
                continue
            task, fold, step = row["task"], int(row["fold"]), int(row["sequence_step"])
            gold = gold_maps[(task, fold, "exposure")][row["case_id"]]
            sequence_length = len(gold_maps[(task, fold, "exposure")])
            fraction = step / sequence_length
            bin_name = "early" if fraction <= 1 / 3 else ("middle" if fraction <= 2 / 3 else "late")
            key = (task, row["method"], row["representation"], bin_name)
            groups[key]["expected"].append(gold)
            groups[key]["predicted"].append(prediction)
    return [{
        "task": key[0], "method": key[1], "representation": key[2],
        "temporal_bin": key[3],
        **regression_metrics(value["expected"], value["predicted"]),
    } for key, value in sorted(groups.items())]


def _specialized_records(root: Path | None) -> list[dict[str, Any]]:
    if root is None:
        return []
    records = []
    for path in sorted(root.glob("*/completion.json")):
        value = json.loads(path.read_text())
        if value.get("status") != "completed":
            raise ValueError(f"Incomplete specialized output: {path}")
        records.extend(value["records"])
    pooled: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "mae_sum": 0.0, "mse_sum": 0.0}
    )
    for row in records:
        key = (row["task"], row["stage"], row["method"])
        pooled[key]["n"] += row["n"]
        pooled[key]["mae_sum"] += row["mae"] * row["n"]
        pooled[key]["mse_sum"] += row["rmse"] ** 2 * row["n"]
    return [{
        "task": key[0], "stage": key[1], "method": key[2],
        "n": value["n"],
        "mae": value["mae_sum"] / value["n"],
        "rmse": (value["mse_sum"] / value["n"]) ** 0.5,
    } for key, value in sorted(pooled.items())]


def _comparison(records: list[dict[str, Any]], left: str, right: str):
    index = {
        (
            row["task"], row.get("representation"),
            row.get("source_representation"), row.get("target_representation"),
            row["method"],
        ): row
        for row in records
    }
    output = []
    contexts = sorted({key[:-1] for key in index})
    for context in contexts:
        left_row, right_row = index.get((*context, left)), index.get((*context, right))
        if left_row is None or right_row is None:
            continue
        output.append({
            "task": context[0],
            "representation": context[1],
            "source_representation": context[2],
            "target_representation": context[3],
            "left_method": left,
            "right_method": right,
            "left_mae": left_row["mae"],
            "right_mae": right_row["mae"],
            "mae_difference_left_minus_right": left_row["mae"] - right_row["mae"],
            "relative_mae_change": (
                (left_row["mae"] - right_row["mae"]) / right_row["mae"]
                if right_row["mae"] else None
            ),
        })
    return output


def _specialized_gap_closure(
    frozen: list[dict[str, Any]], specialized: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    static = {
        (row["task"], row["target_representation"]): row
        for row in frozen
        if row["method"] == "static_multiview_rag"
    }
    geometry = {
        row["task"]: row
        for row in specialized
        if row["stage"] == "frozen_test"
        and row["method"] == "geometry_extra_trees"
    }
    output = []
    for row in frozen:
        if row["method"] != "matrecall":
            continue
        comparator = static.get((row["task"], row["target_representation"]))
        specialist = geometry.get(row["task"])
        if comparator is None or specialist is None:
            continue
        denominator = comparator["mae"] - specialist["mae"]
        output.append({
            "task": row["task"],
            "source_representation": row["source_representation"],
            "target_representation": row["target_representation"],
            "static_rag_mae": comparator["mae"],
            "matrecall_mae": row["mae"],
            "geometry_specialized_mae": specialist["mae"],
            "static_to_specialized_gap_closure": (
                (comparator["mae"] - row["mae"]) / denominator
                if denominator > 0 else None
            ),
            "specialized_is_better_than_static": denominator > 0,
        })
    return output


def report(
    *, release: Path, llm_root: Path, output: Path,
    specialized_root: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((release / "manifest.json").read_text())
    gold_maps, units = _gold_maps(release, manifest)
    exposure, exposure_failures = _collect(llm_root, "exposure", gold_maps)
    frozen, frozen_failures = _collect(llm_root, "frozen", gold_maps)
    temporal = _temporal_records(llm_root, gold_maps)
    specialized = _specialized_records(specialized_root)
    summary = {
        "status": "completed" if not (exposure_failures or frozen_failures) else "completed_with_recorded_response_failures",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": manifest["release_id"],
        "tasks": manifest["tasks"],
        "units": units,
        "exposure_metrics": exposure,
        "frozen_metrics": frozen,
        "temporal_metrics": temporal,
        "specialized_metrics": specialized,
        "specialized_gap_closure": _specialized_gap_closure(
            frozen, specialized
        ),
        "primary_exposure_comparison": _comparison(
            exposure, "matrecall", "dynamic_case_rag"
        ),
        "primary_frozen_comparison": _comparison(
            frozen, "matrecall", "dynamic_case_rag"
        ),
        "shuffled_control_comparison": _comparison(
            frozen, "matrecall", "matrecall_shuffled"
        ),
        "unrelated_control_comparison": _comparison(
            frozen, "matrecall", "matrecall_unrelated"
        ),
        "response_failures": [*exposure_failures, *frozen_failures],
        "release_manifest_sha256": file_sha256(release / "manifest.json"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MatRecall x MatText",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Frozen cross-representation metrics",
        "",
        "| Task | Source | Target | Method | n | MAE | RMSE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in frozen:
        lines.append(
            f"| {row['task']} | {row['source_representation']} | "
            f"{row['target_representation']} | {row['method']} | {row['n']} | "
            f"{row['mae']:.6g} | {row['rmse']:.6g} |"
        )
    lines.extend(["", "## Specialized baselines", "", "| Task | Stage | Method | n | MAE | RMSE |", "|---|---|---|---:|---:|---:|"])
    for row in specialized:
        lines.append(
            f"| {row['task']} | {row['stage']} | {row['method']} | {row['n']} | "
            f"{row['mae']:.6g} | {row['rmse']:.6g} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--llm-root", type=Path, required=True)
    parser.add_argument("--specialized-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = report(
        release=args.release,
        llm_root=args.llm_root,
        specialized_root=args.specialized_root,
        output=args.output,
    )
    print(json.dumps({
        "event": "complete", "release_id": summary["release_id"],
        "status": summary["status"],
    }))


if __name__ == "__main__":
    main()

