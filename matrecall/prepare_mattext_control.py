"""Build a MatText-style coordinate--category control for MatRecall.

The control reuses the official MatBench fold membership and selected material
IDs from the real bulk-modulus cohort.  Its synthetic target is a tunable
mixture of a composition-only potential and a coordinate-only radial potential.
Each component is standardized using only the same-fold static training sample.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from matrecall.mattext import (
    CONTROL_RELEASE_ID,
    controlled_components,
    file_sha256,
    read_jsonl,
    write_jsonl,
)


SOURCE_TASK = "mattext_bulk_modulus_log10_gpa"
ALPHAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
REPRESENTATIONS = ("composition", "crystal_text_llm")


def alpha_name(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def _components(row: dict[str, Any]) -> tuple[float, float]:
    return controlled_components({"representations": row["input_views"]})


def _case_components(row: dict[str, Any]) -> tuple[float, float]:
    return controlled_components(row)


def _standardize(value: float, mean: float, std: float) -> float:
    if std <= 0:
        raise ValueError("Control component has zero training variance")
    return (value - mean) / std


def build_control(
    real_release: Path,
    output: Path,
    *,
    exposure_cases: int = 16,
    frozen_cases: int = 8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    real_manifest = json.loads((real_release / "manifest.json").read_text())
    if real_manifest.get("release_id") != (
        "matrecall-mattext"
    ):
        raise ValueError("Unexpected real MatText release")
    source_records = [
        row for row in real_manifest["task_fold_records"]
        if row["task"] == SOURCE_TASK
    ]
    if len(source_records) != 5:
        raise ValueError("Expected five source folds")
    output.mkdir(parents=True)
    task_fold_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []

    for source_record in source_records:
        fold = int(source_record["fold"])
        source = real_release / source_record["relative_path"]
        static_rows = read_jsonl(source / "static_library.jsonl")
        exposure = read_jsonl(source / "exposure" / "cases.jsonl")[:exposure_cases]
        frozen = read_jsonl(source / "frozen_test" / "cases.jsonl")[:frozen_cases]
        static_components = np.asarray([_components(row) for row in static_rows])
        means = static_components.mean(axis=0)
        stds = static_components.std(axis=0)
        if np.any(stds <= 0):
            raise ValueError(f"Zero control variance in fold {fold}")
        component_records.append({
            "fold": fold,
            "category_mean": float(means[0]),
            "category_std": float(stds[0]),
            "geometry_mean": float(means[1]),
            "geometry_std": float(stds[1]),
        })

        for alpha in ALPHAS:
            task = f"mattext_coordinate_category_alpha_{alpha_name(alpha)}"
            relative = Path("tasks") / task / f"fold_{fold}"
            control_static = []
            for row, (category, geometry) in zip(static_rows, static_components):
                category_z = _standardize(float(category), means[0], stds[0])
                geometry_z = _standardize(float(geometry), means[1], stds[1])
                target = alpha * category_z + (1.0 - alpha) * geometry_z
                control_static.append({
                    **row,
                    "episode_id": row["episode_id"].replace(
                        SOURCE_TASK, task
                    ) + f":alpha-{alpha_name(alpha)}",
                    "task": task,
                    "input_views": {
                        key: row["input_views"][key] for key in REPRESENTATIONS
                    },
                    "observed_output": {
                        "value": float(target), "unit": "standardized synthetic potential",
                    },
                    "control_components": {
                        "category_z": category_z,
                        "geometry_z": geometry_z,
                        "composition_weight_alpha": alpha,
                    },
                })
            write_jsonl(output / relative / "static_library.jsonl", control_static)

            for stage, rows in (("exposure", exposure), ("frozen_test", frozen)):
                cases: list[dict[str, Any]] = []
                gold: list[dict[str, Any]] = []
                order: list[dict[str, Any]] = []
                for step, row in enumerate(rows, start=1):
                    category, geometry = _case_components(row)
                    category_z = _standardize(category, means[0], stds[0])
                    geometry_z = _standardize(geometry, means[1], stds[1])
                    target = alpha * category_z + (1.0 - alpha) * geometry_z
                    case_id = (
                        f"mattext-control:alpha-{alpha_name(alpha)}:fold-{fold}:"
                        f"{stage}:{row['material_id']}"
                    )
                    case = {
                        "case_id": case_id,
                        "material_id": row["material_id"],
                        "task": task,
                        "task_kind": "scalar_regression",
                        "task_instruction": (
                            "Predict the standardized synthetic potential. Its composition "
                            f"contribution is alpha={alpha:.1f}; the remaining contribution "
                            "depends on atomic coordinates."
                        ),
                        "target_unit": "standardized synthetic potential",
                        "source_fold": fold,
                        "stage": stage,
                        "sequence_step": step,
                        "representations": {
                            key: row["representations"][key] for key in REPRESENTATIONS
                        },
                        "representation_groups": {
                            "composition": "category",
                            "crystal_text_llm": "explicit_geometry",
                        },
                        "control_metadata": {
                            "composition_weight_alpha": alpha,
                            "coordinate_weight": 1.0 - alpha,
                        },
                    }
                    cases.append(case)
                    gold.append({
                        "case_id": case_id,
                        "material_id": row["material_id"],
                        "task": task,
                        "task_kind": "scalar_regression",
                        "target": {
                            "value": float(target),
                            "unit": "standardized synthetic potential",
                        },
                        "control_components": {
                            "category_z": category_z,
                            "geometry_z": geometry_z,
                            "composition_weight_alpha": alpha,
                        },
                        "visible_to_method_before_prediction": False,
                    })
                    order.append({
                        "release_id": CONTROL_RELEASE_ID,
                        "task": task,
                        "source_fold": fold,
                        "stage": stage,
                        "sequence_step": step,
                        "case_id": case_id,
                        "material_id": row["material_id"],
                    })
                write_jsonl(output / relative / stage / "cases.jsonl", cases)
                write_jsonl(output / relative / stage / "gold.jsonl", gold)
                write_jsonl(output / relative / stage / "order.jsonl", order)
            task_fold_records.append({
                "task": task,
                "source_task": SOURCE_TASK,
                "fold": fold,
                "alpha": alpha,
                "coordinate_weight": 1.0 - alpha,
                "static_cases": len(control_static),
                "exposure_cases": len(exposure),
                "frozen_test_cases": len(frozen),
                "relative_path": str(relative),
            })

    manifest = {
        "release_id": CONTROL_RELEASE_ID,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_release_id": real_manifest["release_id"],
        "source_release_manifest_sha256": file_sha256(real_release / "manifest.json"),
        "source_task": SOURCE_TASK,
        "tasks": sorted({row["task"] for row in task_fold_records}),
        "task_fold_records": task_fold_records,
        "representations": [
            {"name": "composition", "group": "category"},
            {"name": "crystal_text_llm", "group": "explicit_geometry"},
        ],
        "alphas": list(ALPHAS),
        "component_standardization": component_records,
        "sampling": {
            "folds": list(range(5)),
            "exposure_cases_per_task_fold": exposure_cases,
            "frozen_cases_per_task_fold": frozen_cases,
        },
        "protocol": {
            "current_answer_hidden_before_prediction": True,
            "exposure_feedback_revealed_after_prediction": True,
            "frozen_test_memory_writes": False,
            "components_standardized_on_static_training_sample_only": True,
            "same_material_ids_across_alpha_and_representation": True,
            "alpha_is_composition_weight": True,
        },
        "builder_sha256": file_sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exposure-cases", type=int, default=16)
    parser.add_argument("--frozen-cases", type=int, default=8)
    args = parser.parse_args()
    manifest = build_control(
        args.real_release,
        args.output,
        exposure_cases=args.exposure_cases,
        frozen_cases=args.frozen_cases,
    )
    print(json.dumps({
        "event": "complete",
        "release_id": manifest["release_id"],
        "task_fold_records": len(manifest["task_fold_records"]),
    }))


if __name__ == "__main__":
    main()

