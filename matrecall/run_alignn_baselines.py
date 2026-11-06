"""Run official pretrained ALIGNN2 checkpoints on five crystal-property tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

from matrecall.unified_evaluation import (
    CRYSTAL_TASKS,
    PROTOCOL,
    canonical_json,
    digest,
    envelope,
    file_sha256,
    load_task,
    read_jsonl,
)


def _hash_optional(path: str | None) -> str | None:
    return file_sha256(Path(path)) if path and Path(path).is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--evaluation-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)

    from alignn.ff.unified_calculator import AlignnUnifiedCalculator, AlignnUnifiedConfig
    from alignn.pretrained import ALIGNN2_MODELS, get_alignn2_model
    from jarvis.core.atoms import Atoms

    all_cases = read_jsonl(args.evaluation_release / "test" / "cases.jsonl")
    cases_by_task = {
        task: {case["case_id"]: case for case in all_cases if case["task"] == task}
        for task in CRYSTAL_TASKS
    }
    prediction_path = args.output / "test_predictions.jsonl"
    rows: list[dict[str, Any]] = read_jsonl(prediction_path) if prediction_path.exists() else []
    seen = {row["case_id"] for row in rows}
    if len(seen) != len(rows):
        raise ValueError("Duplicate case IDs in resume file")
    model_records = {}
    started = time.perf_counter()
    with prediction_path.open("a", encoding="utf-8") as output_handle:
        for task in CRYSTAL_TASKS:
            print(json.dumps({"event": "alignn-task-start", "task": task}), flush=True)
            config = AlignnUnifiedConfig(
                energy=False,
                forces=False,
                stress=False,
                properties=[task],
                device=args.device,
                prop_graph="radius",
            )
            calculator = AlignnUnifiedCalculator(config)
            info = calculator._prop_models[task]
            model_paths = get_alignn2_model(info["name"], download=False)
            model_records[task] = {
                "model_name": info["name"],
                "registry": ALIGNN2_MODELS[info["name"]],
                "checkpoint_path": model_paths.get("best_model.pt"),
                "checkpoint_sha256": _hash_optional(model_paths.get("best_model.pt")),
                "config_sha256": _hash_optional(model_paths.get("config.json")),
                "split_ids_sha256": _hash_optional(model_paths.get("ids_train_val_test.json")),
            }
            _, queries, _, _ = load_task(args.release, task, "test")
            if args.limit_per_task is not None:
                queries = queries.head(args.limit_per_task)
            for number, (_, query) in enumerate(queries.iterrows(), start=1):
                case_id = str(query["query_id"])
                if case_id in seen:
                    continue
                case = cases_by_task[task].get(case_id)
                if case is None:
                    raise ValueError(f"Evaluation case missing for {case_id}")
                row = {
                "protocol": PROTOCOL,
                "case_id": case_id,
                "case_sha256": digest(case),
                "task": task,
                "mode": "official_pretrained_specialized_model",
                "model": info["name"],
                "model_revision": args.model_revision,
                "checkpoint_sha256": model_records[task]["checkpoint_sha256"],
            }
                item_started = time.perf_counter()
                try:
                    atoms = Atoms.from_dict(json.loads(query["atoms_json"])).ase_converter()
                    calculator.calculate(atoms=atoms)
                    value = float(calculator.predictions()[task])
                    row.update(
                        status="success",
                        response=envelope(
                            {"value": value},
                            [],
                            "Prediction from the official pretrained ALIGNN2 task checkpoint.",
                        ),
                    )
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["elapsed_seconds"] = time.perf_counter() - item_started
                rows.append(row)
                seen.add(case_id)
                output_handle.write(canonical_json(row) + "\n")
                output_handle.flush()
                if number % 100 == 0 or number == len(queries):
                    print(json.dumps({"event": "alignn-progress", "task": task, "completed": number, "total": len(queries)}), flush=True)
    manifest = {
        "protocol": PROTOCOL,
        "status": "pilot" if args.limit_per_task is not None else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_status": "official_pretrained_matched_to_official_checkpoint_test_ids",
        "tasks": list(CRYSTAL_TASKS),
        "model_family": "ALIGNN 2.0",
        "model_revision": args.model_revision,
        "models": model_records,
        "counts": {
            "rows": len(rows),
            "success": sum(row["status"] == "success" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {"python": sys.version, "platform": platform.platform(), "device": args.device},
        "source_release_manifest_sha256": file_sha256(args.release / "manifest.json"),
        "evaluation_release_manifest_sha256": file_sha256(args.evaluation_release / "manifest.json"),
        "predictions_sha256": file_sha256(prediction_path),
        "runner_sha256": file_sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
