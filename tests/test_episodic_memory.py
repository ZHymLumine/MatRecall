from matrecall.episodic_memory import (
    compress_experience,
    graph_records,
    retrieve_case_memory,
    retrieve_experience_memory,
)


def _case(case_id="q1", value=1.0):
    return {
        "case_id": case_id,
        "task": "test_property",
        "task_kind": "scalar_regression",
        "query_visible": {"formula": "LiFePO4", "pressure": value},
        "deterministic_tool_estimate": {"value": 1.5},
    }


def _result(prediction=2.0, *, selected_experiences=None):
    return {
        "status": "success",
        "response": {"reasoning_results": {"prediction": {"value": prediction}}},
        "selected_evidence_ids": ["train:1"],
        "selected_experience_ids": selected_experiences or [],
        "agent_trace": [{"step": 1, "kind": "planner"}],
        "elapsed_seconds": 0.25,
    }


def test_compression_records_prediction_then_verified_numeric_feedback():
    episode = compress_experience(
        case=_case(), gold={"target": {"value": 3.0}}, result=_result(2.0),
        sequence_id="task:seed-0", sequence_step=1,
    )
    assert episode["prediction"] == {"value": 2.0}
    assert episode["ground_truth"] == {"value": 3.0}
    assert episode["feedback"]["residual"] == 1.0
    assert episode["feedback"]["loss"] == 1.0
    assert episode["lesson"]["correction"] == "past_prediction_was_too_low"
    assert episode["verified_after_prediction"] is True
    assert len(episode["raw_trace_sha256"]) == 64


def test_case_memory_omits_prior_prediction_and_trajectory():
    episode = compress_experience(
        case=_case(), gold={"target": {"value": 3.0}}, result=_result(2.0),
        sequence_id="task:seed-0", sequence_step=1,
    )
    retrieved = retrieve_case_memory([episode], {"formula": "LiFePO4"}, 1)[0]
    assert retrieved["verified_ground_truth"] == {"value": 3.0}
    assert "prior_prediction" not in retrieved
    assert "trajectory_summary" not in retrieved


def test_large_structured_values_are_bounded_and_hash_bound():
    case = _case()
    case["task_kind"] = "structured_output"
    large = {"nodes": [{"label": "x" * 300} for _ in range(20)]}
    result = _result()
    result["response"]["reasoning_results"]["prediction"] = large
    episode = compress_experience(
        case=case, gold={"target": large}, result=result,
        sequence_id="task:seed-0", sequence_step=1,
    )
    assert episode["prediction"]["full_value_characters"] > 1600
    assert len(episode["prediction"]["full_value_sha256"]) == 64
    assert episode["ground_truth"]["full_value_sha256"] == episode["prediction"]["full_value_sha256"]


def test_experience_memory_exposes_feedback_and_graph_reuse_edge():
    first = compress_experience(
        case=_case("q1"), gold={"target": {"value": 2.0}}, result=_result(2.0),
        sequence_id="task:seed-0", sequence_step=1,
    )
    second = compress_experience(
        case=_case("q2", 2.0), gold={"target": {"value": 4.0}},
        result=_result(0.0, selected_experiences=[first["experience_id"]]),
        sequence_id="task:seed-0", sequence_step=2,
    )
    retrieved = retrieve_experience_memory([first, second], {"formula": "LiFePO4"}, 2)
    assert {row["contrast_role"] for row in retrieved} == {"helpful", "corrective"}
    assert all("feedback" in row and "lesson" in row and "trajectory_summary" in row for row in retrieved)
    nodes, edges = graph_records([first, second])
    assert {node["node_type"] for node in nodes} >= {
        "execution", "query", "prediction", "verified_outcome", "feedback", "consolidated_lesson"
    }
    assert any(edge["relation"] == "REUSED_EXPERIENCE" for edge in edges)
