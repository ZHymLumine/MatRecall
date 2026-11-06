import pandas as pd

from matrecall.build_process_tasks import (
    TASK_SPECS,
    _clean,
    _integrity,
    _task_tables,
    assign_group_splits,
    connect_source_and_input_groups,
)


def test_clean_normalizes_csv_missing_value_spellings():
    assert _clean("None") is None
    assert _clean(" null ") is None
    assert _clean("chemical name") == "chemical name"


def test_connected_groups_join_same_doi_and_same_exact_input():
    frame = pd.DataFrame(
        {
            "source_group": ["doi-a", "doi-a", "doi-b", "doi-c"],
            "input_signature": ["recipe-1", "recipe-2", "recipe-2", "recipe-3"],
        }
    )
    groups = connect_source_and_input_groups(frame)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.iloc[3] != groups.iloc[0]


def test_group_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame(
        {
            "leakage_group": [f"group-{index // 3}" for index in range(90)],
            "positive": [index % 2 for index in range(90)],
        }
    )
    first, first_meta = assign_group_splits(frame, ["positive"], candidates=32)
    second, second_meta = assign_group_splits(frame, ["positive"], candidates=32)
    assert list(first) == list(second)
    assert first_meta == second_meta
    membership = pd.DataFrame({"group": frame["leakage_group"], "split": first})
    assert membership.groupby("group")["split"].nunique().max() == 1
    assert set(first) == {"train", "validation", "test"}


def test_public_queries_hide_targets_and_memory_is_train_only():
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        rows.append(
            {
                "source_record_id": f"row-{index}",
                "source_group": f"doi-{index}",
                "leakage_group": f"group-{index}",
                "input_json": "{}",
                "input_text": "input",
                "feature_json": "{}",
                "context_json": "{}",
                "protocol": "experiment",
                "source_dataset": "fixture",
                "source_revision": "fixture.csv",
                "source_url": "https://example.test",
                "source_file_sha256": "abc",
                "epistemic_status": "experimental",
                "quality_json": "{}",
                "target_value_numeric": float(index),
                "target_value_text": None,
                "split": split,
            }
        )
    frame = pd.DataFrame(rows)
    spec = TASK_SPECS["inp_absorption_peak_nm"]
    memory, queries, labels = _task_tables(frame, spec)
    assert set(memory["split"]) == {"train"}
    assert not any(
        column.startswith("target_value")
        for query in queries.values()
        for column in query.columns
    )
    assert labels["test"].iloc[0]["target_value_numeric"] == 2.0
    assert _integrity(memory, queries, labels)["passed"]
