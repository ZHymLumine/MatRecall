from matrecall.prepare_online_adaptation import select_gold_rows


def _rows(count, *, task_kind="scalar_regression"):
    return [
        {
            "case_id": f"case-{index}",
            "task_kind": task_kind,
            "target": {"value": float(index)},
        }
        for index in range(count)
    ]


def test_sequence_sampling_is_deterministic_and_bounded():
    rows = _rows(20)
    first = select_gold_rows(rows, task="regression-task", maximum=7, seed=0)
    second = select_gold_rows(list(reversed(rows)), task="regression-task", maximum=7, seed=0)
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert len(first) == 7


def test_classification_sampling_keeps_each_class_when_budget_allows():
    rows = _rows(12, task_kind="multiclass_classification")
    for index, row in enumerate(rows):
        row["target"] = {"value": ["alpha", "beta", "gamma"][index % 3]}
    selected = select_gold_rows(rows, task="classification-task", maximum=5, seed=0)
    assert {row["target"]["value"] for row in selected} == {"alpha", "beta", "gamma"}
    assert len(selected) == 5


def test_small_task_keeps_all_cases_but_uses_frozen_stream_order():
    rows = _rows(4)
    selected = select_gold_rows(rows, task="small-task", maximum=10, seed=0)
    assert {row["case_id"] for row in selected} == {row["case_id"] for row in rows}
    assert selected == select_gold_rows(rows, task="small-task", maximum=10, seed=0)
