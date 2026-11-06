from matrecall.episodic_memory import compress_experience
from matrecall.run_memory_ablation import (
    compact_response_schema,
    eligible_memory,
    project_episode,
)


def _case():
    return {
        "case_id": "q1",
        "task": "property",
        "task_kind": "scalar_regression",
        "query_visible": {"formula": "LiFePO4"},
        "deterministic_tool_estimate": {"value": 1.0},
        "prediction_schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }


def _episode(step=1):
    result = {
        "status": "success",
        "response": {"reasoning_results": {"prediction": {"value": 2.0}}},
        "selected_evidence_ids": ["train:1"],
        "selected_experience_ids": [],
        "agent_trace": [{"step": 1, "kind": "planner"}],
    }
    return compress_experience(
        case=_case(), gold={"target": {"value": 3.0}}, result=result,
        sequence_id="property:seed-0", sequence_step=step,
    )


def test_case_only_projection_hides_prediction_feedback_and_trajectory():
    projected = project_episode(_episode(), "case_only")
    assert projected["verified_ground_truth"] == {"value": 3.0}
    assert "prior_prediction" not in projected
    assert "feedback" not in projected
    assert "trajectory_summary" not in projected
    assert "lesson" not in projected


def test_score_only_projection_hides_ground_truth():
    projected = project_episode(_episode(), "score_only")
    assert "verified_ground_truth" not in projected
    assert projected["prior_prediction"] == {"value": 2.0}
    assert projected["feedback"]["score"] == -1.0
    assert "residual" not in projected["feedback"]
    assert "trajectory_summary" in projected
    assert "lesson" in projected
    assert "correction" not in projected["lesson"]


def test_delayed_feedback_is_not_available_early():
    memory = [_episode(step) for step in (1, 2, 3, 4)]
    assert eligible_memory(
        memory, policy="full_delayed", current_step=4, delay_steps=4
    ) == []
    available = eligible_memory(
        memory, policy="full_delayed", current_step=5, delay_steps=4
    )
    assert [row["sequence_step"] for row in available] == [1]


def test_compact_schema_bounds_reasoning_steps_and_text():
    schema = compact_response_schema(_case(), ["train:1"], ["experience:1"])
    process = schema["properties"]["analysis_process"]
    assert process["maxItems"] == 2
    assert process["items"]["properties"]["comparison"]["maxLength"] == 400
