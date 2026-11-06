import json

from matrecall.prepare_memory_retention import select_group_diverse_rows
from matrecall.report_memory_retention import select_report_tasks
from matrecall.run_frozen_memory import (
    EPISODIC_ONLY_METHODS,
    final_messages,
    planner_messages,
)


def test_group_diverse_selection_prefers_distinct_groups():
    rows = [
        {"case_id": "a1", "transfer_group_id": "a"},
        {"case_id": "a2", "transfer_group_id": "a"},
        {"case_id": "b1", "transfer_group_id": "b"},
        {"case_id": "c1", "transfer_group_id": "c"},
    ]
    selected = select_group_diverse_rows(rows, task="task", maximum=3, seed=0)
    assert len(selected) == 3
    assert {row["transfer_group_id"] for row in selected} == {"a", "b", "c"}


def test_group_diverse_selection_is_deterministic():
    rows = [
        {"case_id": f"case-{index}", "transfer_group_id": f"g-{index % 3}"}
        for index in range(12)
    ]
    first = select_group_diverse_rows(rows, task="task", maximum=7, seed=4)
    second = select_group_diverse_rows(list(reversed(rows)), task="task", maximum=7, seed=4)
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]


def test_transfer_prompt_contains_no_current_target_and_marks_memory_frozen():
    case = {
        "task_instruction": "Predict a value",
        "query_visible": {"formula": "Al2O3"},
        "deterministic_tool_estimate": {"value": 1.0},
        "task_kind": "scalar_regression",
        "prediction_schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }
    messages, _ = final_messages(
        case=case,
        method="dynamic_case_rag",
        references=[],
        prior_cases=[{
            "case_memory_id": "experience:old",
            "prior_input": "old query",
            "verified_ground_truth": {"value": 2.0},
        }],
        experiences=[],
        planner=None,
        observation={},
        prompt_set="revised",
    )
    prompt = messages[1]["content"]
    assert "immutable exposure memory" in prompt
    assert "Al2O3" in prompt
    assert "current_target" not in prompt
    assert "experience:old" in prompt


def test_transfer_planner_bounds_explanations_for_fixed_output_budget():
    case = {
        "task_instruction": "Predict a value",
        "query_visible": {"formula": "Al2O3"},
        "deterministic_tool_estimate": {"value": 1.0},
        "initial_evidence_ids": ["evidence:1"],
        "reference_views": {"composition": ["evidence:1"]},
        "references": [{"evidence_id": "evidence:1", "input": {}}],
    }
    _, schema = planner_messages(
        case,
        [{"experience_id": "experience:1", "prior_input": "old"}],
        evidence_budget=8,
        experience_budget=6,
        prompt_set="original",
    )
    for field in ("comparison", "conflict_assessment", "memory_use"):
        assert schema["properties"][field]["maxLength"] == 240


def test_episodic_only_planner_cannot_select_static_evidence():
    case = {
        "task_instruction": "Predict a value",
        "query_visible": {"formula": "Al2O3"},
        "deterministic_tool_estimate": {"value": 1.0},
        "initial_evidence_ids": ["evidence:1"],
        "reference_views": {"composition": ["evidence:1"]},
        "references": [{"evidence_id": "evidence:1", "input": {}}],
    }
    messages, schema = planner_messages(
        case,
        [{"experience_id": "experience:1", "prior_input": "old"}],
        evidence_budget=8,
        experience_budget=6,
        prompt_set="revised",
        include_static=False,
    )
    payload = json.loads(messages[-1]["content"])
    assert set(EPISODIC_ONLY_METHODS) == {
        "case_episodic_only", "full_episodic_only"
    }
    assert "static_candidate_catalog" not in payload
    assert schema["properties"]["selected_evidence_ids"]["maxItems"] == 0
    assert schema["properties"]["requested_view"]["enum"] == ["none"]


def test_report_task_exclusion_preserves_other_manifest_tasks():
    tasks = [
        {"task": "meltpool_shape", "transfer_cases": 2},
        {"task": "formation_energy_peratom", "transfer_cases": 32},
    ]
    selected = select_report_tasks(tasks, {"meltpool_shape"})
    assert list(selected) == ["formation_energy_peratom"]
    assert len(tasks) == 2


def test_report_task_exclusion_rejects_unknown_task():
    tasks = [{"task": "formation_energy_peratom", "transfer_cases": 32}]
    try:
        select_report_tasks(tasks, {"not_a_task"})
    except ValueError as exc:
        assert "Unknown report exclusions" in str(exc)
    else:
        raise AssertionError("Expected unknown report exclusion to fail")


def _transfer_case():
    return {
        "task_instruction": "Predict a value",
        "query_visible": {"formula": "Al2O3"},
        "deterministic_tool_estimate": {"value": 1.0},
        "task_kind": "scalar_regression",
        "prediction_schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "initial_evidence_ids": ["evidence:1"],
        "reference_views": {"composition": ["evidence:1"]},
        "references": [{"evidence_id": "evidence:1", "input": {}}],
    }


def test_only_revised_transfer_prompt_mentions_cross_task_memory():
    instructions = {}
    for prompt_set in ("original", "revised"):
        messages, _ = final_messages(
            case=_transfer_case(), method="llm_only", references=[], prior_cases=[],
            experiences=[], planner=None, observation={}, prompt_set=prompt_set,
        )
        instructions[prompt_set] = json.loads(messages[-1]["content"])["instruction"]
    assert instructions["original"] == (
        "Predict independently using the immutable exposure memory supplied here. "
        "The current answer is unavailable; do not write or infer new feedback."
    )
    assert "different source task" in instructions["revised"]


def test_transfer_planner_prompt_sets_keep_their_published_instructions():
    instructions = {}
    for prompt_set in ("original", "revised"):
        messages, _ = planner_messages(
            _transfer_case(), [], evidence_budget=8, experience_budget=6,
            prompt_set=prompt_set,
        )
        instructions[prompt_set] = json.loads(messages[-1]["content"])["instruction"]
    assert instructions["original"].startswith(
        "Choose static evidence and frozen verified exposure experiences. "
    )
    assert instructions["revised"].startswith(
        "Choose frozen verified exposure experiences and static evidence. "
    )
