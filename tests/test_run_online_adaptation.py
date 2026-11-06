import json

from matrecall.run_online_adaptation import (
    EPISODIC_ONLY_METHODS,
    final_messages,
    planner_messages,
    response_schema,
)


def _case():
    return {
        "task": "property_task",
        "task_kind": "scalar_regression",
        "task_instruction": "Predict the scalar property.",
        "query_visible": {"formula": "NaCl", "secret_guard": "CURRENT_QUERY_ONLY"},
        "prediction_schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "deterministic_tool_estimate": {"value": 1.0},
    }


def test_final_prompt_contains_prior_verified_feedback_but_no_current_gold():
    case = _case()
    references = [{"evidence_id": "train:1", "observed_output": {"value": 1.0}}]
    prior = [{
        "case_memory_id": "experience:old",
        "prior_input": "old query",
        "verified_ground_truth": {"value": "PRIOR_GOLD"},
    }]
    messages, _ = final_messages(
        case=case, method="dynamic_case_rag", references=references,
        prior_cases=prior, experiences=[], planner=None,
        observation={"action": "predict", "requested_view": "none"},
    )
    payload = json.loads(messages[-1]["content"])
    assert payload["current_query"] == case["query_visible"]
    assert payload["verified_prior_case_memory"][0]["verified_ground_truth"] == {"value": "PRIOR_GOLD"}
    assert "current_gold" not in payload
    assert "target" not in payload


def test_response_schema_cannot_cite_unsupplied_memory():
    schema = response_schema(_case(), ["train:1"], ["experience:old"])
    properties = schema["properties"]["reasoning_results"]["properties"]
    assert properties["evidence_ids"]["items"]["enum"] == ["train:1"]
    assert properties["experience_ids"]["items"]["enum"] == ["experience:old"]


def test_llm_only_schema_requires_empty_evidence_and_memory_arrays():
    schema = response_schema(_case(), [], [])
    properties = schema["properties"]["reasoning_results"]["properties"]
    assert properties["evidence_ids"]["maxItems"] == 0
    assert properties["experience_ids"]["maxItems"] == 0


def test_temporal_full_episodic_only_planner_has_no_static_channel():
    case = {
        **_case(),
        "initial_evidence_ids": ["train:1"],
        "reference_views": {"composition": ["train:1"]},
        "references": [{"evidence_id": "train:1", "input": {}}],
    }
    messages, schema = planner_messages(
        case,
        [{"experience_id": "experience:old"}],
        evidence_budget=8,
        experience_budget=6,
        prompt_set="revised",
        include_static=False,
    )
    payload = json.loads(messages[-1]["content"])
    assert "full_episodic_only" in EPISODIC_ONLY_METHODS
    assert "static_candidate_catalog" not in payload
    assert schema["properties"]["selected_evidence_ids"]["maxItems"] == 0


def _planner_case():
    return {
        **_case(),
        "initial_evidence_ids": ["train:1"],
        "reference_views": {"composition": ["train:1"]},
        "references": [{"evidence_id": "train:1", "input": {}}],
    }


def test_planner_prompt_sets_keep_their_published_instructions():
    instructions = {}
    for prompt_set in ("original", "revised"):
        messages, _ = planner_messages(
            _planner_case(), [{"experience_id": "experience:old"}],
            evidence_budget=8, experience_budget=6, prompt_set=prompt_set,
        )
        instructions[prompt_set] = json.loads(messages[-1]["content"])["instruction"]
    assert instructions["original"] == (
        "Choose static evidence and verified prior experiences for the current prediction. "
        "The current query ground truth is unavailable."
    )
    assert instructions["revised"] == (
        "Choose verified prior experiences and static evidence for the current prediction. "
        "The current query ground truth is unavailable."
    )


def test_original_prompt_set_has_no_episodic_only_planner():
    try:
        planner_messages(
            _planner_case(), [], evidence_budget=8, experience_budget=6,
            prompt_set="original", include_static=False,
        )
    except ValueError as exc:
        assert "revised prompt set" in str(exc)
    else:
        raise AssertionError("Expected the original prompt set to reject episodic-only planning")
