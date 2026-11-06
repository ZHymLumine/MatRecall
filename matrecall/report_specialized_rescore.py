"""Summarize matched frozen-specialized metrics on a prequential cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matrecall.report_common import metric_rows


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--unified-metrics", type=Path, required=True)
    parser.add_argument("--process-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comparison-scope",
        default="frozen specialized predictions on the exact memory-ablation iid seed-0 cohort",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verification = load(args.verification)
    if verification.get("status") != "verified" or verification.get("task_count") != 22:
        raise ValueError("Specialized rebinding verification failed")
    unified = load(args.unified_metrics)["summary"]
    process = load(args.process_metrics)["summary"]
    if len(unified) != 13 or len(process) != 9:
        raise ValueError("Specialized metric task coverage mismatch")
    rows = []
    for task, summary in sorted(unified.items()):
        name, value, higher = metric_rows(task, summary, process=False)[0]
        rows.append({
            "family": "unified", "task": task, "metric": name,
            "value": value, "higher_is_better": higher,
        })
    for task, summary in sorted(process.items()):
        name, value, higher = metric_rows(task, summary, process=True)[0]
        rows.append({
            "family": "process", "task": task, "metric": name,
            "value": value, "higher_is_better": higher,
        })
    payload = {
        "status": "completed",
        "task_count": 22,
        "case_count": verification["case_count"],
        "comparison_scope": args.comparison_scope,
        "online_updates": False,
        "verification": verification,
        "primary_metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "tasks": 22}, sort_keys=True))


if __name__ == "__main__":
    main()
