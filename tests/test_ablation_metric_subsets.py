import pytest

from matrecall.prepare_ablation_metric_subsets import (
    annotate_prior_same_group_counts,
)


def _row(case_id, step, group, first, task="task-a"):
    return {
        "case_id": case_id,
        "task": task,
        "sequence_step": step,
        "sequence_group_id": group,
        "group_first_occurrence": first,
    }


def test_prior_same_group_count_follows_query_order():
    rows = [
        _row("a1", 1, "a", True),
        _row("b1", 2, "b", True),
        _row("a2", 3, "a", False),
        _row("a3", 4, "a", False),
    ]
    annotated = annotate_prior_same_group_counts(rows)
    assert [row["prior_same_group_count"] for row in annotated] == [0, 0, 1, 2]


def test_prior_same_group_count_rejects_bad_first_flag():
    with pytest.raises(ValueError, match="Group-first flag"):
        annotate_prior_same_group_counts([_row("a1", 1, "a", False)])


def test_prior_same_group_count_rejects_nonsequential_rows():
    with pytest.raises(ValueError, match="strictly increasing"):
        annotate_prior_same_group_counts([
            _row("a1", 2, "a", True),
            _row("b1", 1, "b", True),
        ])


def test_prior_same_group_count_is_scoped_by_task():
    annotated = annotate_prior_same_group_counts([
        _row("a1", 1, "shared", True, task="task-a"),
        _row("b1", 1, "shared", True, task="task-b"),
    ])
    assert [row["prior_same_group_count"] for row in annotated] == [0, 0]
