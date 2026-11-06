import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matrecall.prepare_mattext import (
    REPRESENTATIONS,
    TASKS,
    build_release,
)


def _write_source_release(root: Path) -> None:
    for spec in TASKS:
        rows = []
        for index in range(10):
            identifier = f"mb-{spec.source_prefix}-{index:03d}"
            row = {
                "mbid": identifier,
                "labels": float(index) / 10,
                **{
                    representation: f"{representation} material-{index}"
                    for representation in REPRESENTATIONS
                },
            }
            rows.append(row)
        for fold in range(5):
            test_indices = {2 * fold, 2 * fold + 1}
            train_rows = [row for index, row in enumerate(rows) if index not in test_indices]
            test_rows = [
                {key: value for key, value in row.items() if key != "labels"}
                for index, row in enumerate(rows)
                if index in test_indices
            ]
            for split, values in (("train", train_rows), ("test", test_rows)):
                path = (
                    root
                    / "benchmark_parquet"
                    / f"{spec.source_prefix}-{split}-matbench"
                    / f"fold_{fold}"
                    / "0000.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(values), path)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_builder_keeps_recovered_test_labels_out_of_cases(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "prepared"
    _write_source_release(raw)

    manifest = build_release(
        raw,
        output,
        static_cases=2,
        exposure_cases=1,
        frozen_cases=1,
        seed=7,
    )

    assert manifest["status"] == "frozen"
    assert len(manifest["task_fold_records"]) == 15
    assert len(manifest["source_sha256"]) == 30
    for record in manifest["task_fold_records"]:
        base = output / record["relative_path"]
        static_rows = _read_jsonl(base / "static_library.jsonl")
        exposure_cases = _read_jsonl(base / "exposure" / "cases.jsonl")
        exposure_gold = _read_jsonl(base / "exposure" / "gold.jsonl")
        frozen_cases = _read_jsonl(base / "frozen_test" / "cases.jsonl")
        frozen_gold = _read_jsonl(base / "frozen_test" / "gold.jsonl")

        assert len(static_rows) == 2
        assert len(exposure_cases) == len(exposure_gold) == 1
        assert len(frozen_cases) == len(frozen_gold) == 1
        assert "target" not in exposure_cases[0]
        assert "labels" not in exposure_cases[0]
        assert "target" not in frozen_cases[0]
        assert set(exposure_cases[0]["representations"]) == set(REPRESENTATIONS)
        assert exposure_gold[0]["label_provenance"] == {
            "test_parquet_contains_label": False,
            "recovered_from_other_training_fold": True,
            "visible_to_method_before_prediction": False,
        }
        static_ids = {row["material_id"] for row in static_rows}
        test_ids = {
            exposure_cases[0]["material_id"], frozen_cases[0]["material_id"]
        }
        assert static_ids.isdisjoint(test_ids)


def test_sampling_is_deterministic_for_same_seed(tmp_path):
    raw = tmp_path / "raw"
    _write_source_release(raw)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_release(raw, first, static_cases=2, exposure_cases=1, frozen_cases=1, seed=3)
    build_release(raw, second, static_cases=2, exposure_cases=1, frozen_cases=1, seed=3)

    for spec in TASKS:
        for fold in range(5):
            relative = Path("tasks") / spec.task_id / f"fold_{fold}"
            for name in (
                "static_library.jsonl",
                "exposure/cases.jsonl",
                "exposure/gold.jsonl",
                "frozen_test/cases.jsonl",
                "frozen_test/gold.jsonl",
            ):
                assert (first / relative / name).read_bytes() == (second / relative / name).read_bytes()
