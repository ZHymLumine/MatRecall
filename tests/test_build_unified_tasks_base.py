from __future__ import annotations

import json

import pandas as pd

from matrecall.build_unified_tasks_base import (
    assign_diffsyn_system_splits,
    assign_synthesis_similarity_splits,
    load_spende_rows,
    load_synthesis_similarity_rows,
    normalize_zeolite_code,
)


def _synthesis_record(identifier: str, doi: str, year: int, target: str) -> dict:
    return {
        "_id": identifier,
        "doi": doi,
        "year": year,
        "synthesis_type": "solid_state_ceramic_synthesis",
        "targets_string": [target],
        "precursors": [
            {"material_formula": "A2O3"},
            {"material_formula": "BO2"},
        ],
    }


def test_synthesis_similarity_reader_and_grouped_chronological_split(tmp_path):
    path = tmp_path / "source.jsonl"
    records = [
        _synthesis_record(f"id-{year}", f"10.test/{year}", year, f"A{year}BO3")
        for year in range(2000, 2020)
    ]
    # A repeated target must remain in the earliest component/split.
    records.append(_synthesis_record("repeat", "10.test/repeat", 2019, "A2000BO3"))
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    frame = load_synthesis_similarity_rows(path)
    split = assign_synthesis_similarity_splits(frame)
    frame["split"] = split

    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame.loc[frame.target_formula == "A2000BO3", "split"].nunique() == 1
    for column in ("doi", "target_formula"):
        groups = frame.groupby(column)["split"].nunique()
        assert int(groups.max()) == 1


def test_spende_reader_preserves_structured_operations(tmp_path):
    path = tmp_path / "tmsr.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doi": "10.test/spende",
                    "target": {"composition": [{"formula": "AB2", "elements": {}}]},
                    "precursors": [
                        {"composition": [{"formula": "A", "elements": {}}]},
                        {"composition": [{"formula": "B", "elements": {}}]},
                    ],
                    "operations": [
                        {
                            "string": "heating",
                            "type": "HeatingOperation",
                            "conditions": "900 4",
                        }
                    ],
                    "material property": {"temperature (K)": 300},
                }
            ]
        ),
        encoding="utf-8",
    )

    frame = load_spende_rows(path)

    assert len(frame) == 1
    assert json.loads(frame.loc[0, "target_formulas_json"]) == ["AB2"]
    assert json.loads(frame.loc[0, "precursor_formulas_json"]) == ["A", "B"]
    assert json.loads(frame.loc[0, "operation_sequence_json"]) == ["HeatingOperation"]


def test_diffsyn_normalization_and_system_split_are_deterministic():
    assert normalize_zeolite_code(None) == "Dense/Amorphous"
    assert normalize_zeolite_code("*BEA") == "BEA"
    assert normalize_zeolite_code("*-ITN") == "ITN"
    assert normalize_zeolite_code("-CLO") == "CLO"

    systems = [[f"Z{i:03d}", f"OSDA-{i}"] for i in range(100)]
    first = assign_diffsyn_system_splits(systems)
    second = assign_diffsyn_system_splits(systems)
    assert first == second
    counts = pd.Series(first.values()).value_counts().to_dict()
    assert counts == {"train": 70, "test": 20, "validation": 10}
