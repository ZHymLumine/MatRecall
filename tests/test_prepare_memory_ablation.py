from matrecall.prepare_memory_ablation import (
    group_id_for_case,
    order_rows,
    select_fixed_cohort,
    stream_task_relative_path,
)


def test_stream_task_paths_are_stream_root_relative():
    relative = stream_task_relative_path("unified", "example")
    assert str(relative) == "sequences/unified/example"
    assert not str(relative).startswith("streams/")


def _gold(index, *, kind="scalar_regression"):
    return {
        "case_id": f"case-{index}",
        "task_kind": kind,
        "target": {"value": index},
    }


def test_fixed_cohort_does_not_depend_on_order_seed():
    rows = [_gold(index) for index in range(20)]
    selected = select_fixed_cohort(rows, task="property", maximum=8, sample_seed=3)
    reversed_selected = select_fixed_cohort(
        list(reversed(rows)), task="property", maximum=8, sample_seed=3
    )
    assert [row["case_id"] for row in selected] == [
        row["case_id"] for row in reversed_selected
    ]


def test_group_blocked_order_is_contiguous_and_marks_first_occurrence():
    rows = [
        {"case_id": "a1", "sequence_group_id": "a"},
        {"case_id": "b1", "sequence_group_id": "b"},
        {"case_id": "a2", "sequence_group_id": "a"},
        {"case_id": "b2", "sequence_group_id": "b"},
    ]
    ordered = order_rows(rows, task="task", regime="group_blocked", order_seed=0)
    assert [row["sequence_group_id"] for row in ordered] in (
        ["a", "a", "b", "b"], ["b", "b", "a", "a"]
    )


def test_group_key_uses_query_metadata_not_target_value():
    case = {
        "case_id": "q1",
        "task": "formation_energy_peratom",
        "query_visible": {"chemical_system": "Fe-O"},
        "query_metadata": {"chemical_system": "Fe-O"},
    }
    first = group_id_for_case(case, {"target": {"value": -1.0}})
    second = group_id_for_case(case, {"target": {"value": 100.0}})
    assert first == second == ("chemical_system:Fe-O", "chemical_system", False)


def test_inverse_design_group_is_explicitly_marked_as_proxy():
    case = {
        "case_id": "q2",
        "task": "target_property_to_formulation_composition",
        "initial_evidence_ids": ["train:anchor-1"],
        "query_visible": {"target_properties": {"HV": 250.0}},
    }
    group_id, source, proxy = group_id_for_case(case, {"target": {}})
    assert group_id == "anchor:train:anchor-1"
    assert source == "initial_evidence_ids[0]"
    assert proxy is True
