from copy import deepcopy
import json

from matrecall.mattext import (
    controlled_components,
    retrieve_full_experience,
    runtime_case,
    shuffle_episode_feedback,
    structure_descriptor,
)
from matrecall.run_mattext import (
    V08_NUMERIC_PATTERN,
    compact_final_contract,
    compact_planner_contract,
    normalize_mattext_prediction,
)


CRYSTAL = """4.0 4.0 4.0
90 90 90
Na
0.00 0.00 0.00
Cl
0.50 0.50 0.50
"""

SINGLE_SITE_CRYSTAL = """3.0 4.0 5.0
90 90 90
Si
0.00 0.00 0.00
"""


def _static(identifier: str, formula: str, value: float):
    return {
        "episode_id": identifier,
        "input_views": {
            "composition": formula,
            "crystal_text_llm": CRYSTAL,
        },
        "observed_output": {"value": value, "unit": "u"},
    }


def _case():
    return {
        "case_id": "query-1",
        "material_id": "m-1",
        "task": "task-a",
        "task_kind": "scalar_regression",
        "task_instruction": "Predict a scalar.",
        "target_unit": "u",
        "representations": {
            "composition": "NaCl",
            "crystal_text_llm": CRYSTAL,
        },
    }


def test_runtime_case_never_contains_current_target():
    case = _case()
    runtime = runtime_case(
        case,
        [_static("e-1", "NaCl", 1.0), _static("e-2", "KCl", 2.0)],
        "crystal_text_llm",
    )

    assert "target" not in runtime
    assert "target" not in runtime["query_visible"]
    assert runtime["query_visible"] == {
        "representation": "crystal_text_llm",
        "material_text": CRYSTAL,
    }
    assert all(
        row["input"]["representation"] == "crystal_text_llm"
        for row in runtime["references"]
    )


def test_full_experience_retrieval_uses_normalized_composition():
    memory = [
        {
            "experience_id": "far",
            "task": "task-a",
            "sequence_step": 2,
            "memory_metadata": {"normalized_composition": "SiO2"},
            "query_summary": "far",
            "prediction": {"value": 0.0},
            "ground_truth": {"value": 0.0},
            "feedback": {},
            "trajectory_summary": {},
            "lesson": {},
        },
        {
            "experience_id": "near",
            "task": "task-a",
            "sequence_step": 1,
            "memory_metadata": {"normalized_composition": "NaCl"},
            "query_summary": "near",
            "prediction": {"value": 1.0},
            "ground_truth": {"value": 1.0},
            "feedback": {},
            "trajectory_summary": {},
            "lesson": {},
        },
    ]

    selected = retrieve_full_experience(memory, _case(), 1)
    assert [row["experience_id"] for row in selected] == ["near"]


def test_shuffled_control_preserves_queries_but_rotates_verified_feedback():
    memory = []
    for index in range(3):
        memory.append({
            "experience_id": f"e-{index}",
            "query_summary": f"query-{index}",
            "prediction": {"value": index},
            "ground_truth": {"value": 10 + index},
            "feedback": {"residual": index},
            "lesson": {"correction": f"c-{index}"},
            "trajectory_summary": {"selected_evidence_ids": [f"s-{index}"]},
        })
    original = deepcopy(memory)

    shuffled = shuffle_episode_feedback(memory, namespace="fixed")

    assert [row["query_summary"] for row in shuffled] == [
        row["query_summary"] for row in original
    ]
    assert [row["ground_truth"] for row in shuffled] != [
        row["ground_truth"] for row in original
    ]
    assert memory == original
    assert all(row["control"]["type"].startswith("task_matched") for row in shuffled)


def test_coordinate_category_components_are_deterministic_and_finite():
    category_1, geometry_1 = controlled_components(_case())
    category_2, geometry_2 = controlled_components(_case())

    assert (category_1, geometry_1) == (category_2, geometry_2)
    assert category_1 > 0
    assert geometry_1 > 0


def test_single_site_crystal_uses_finite_periodic_image_proxy():
    case = _case()
    case["representations"]["composition"] = "Si"
    case["representations"]["crystal_text_llm"] = SINGLE_SITE_CRYSTAL

    category, geometry = controlled_components(case)
    descriptor = structure_descriptor(SINGLE_SITE_CRYSTAL)

    assert category > 0
    assert geometry > 0
    assert all(value == value for value in descriptor)


def test_output_contract_bounds_trace_and_numeric_lexeme():
    messages = [{"role": "user", "content": '{"task":"predict"}'}]
    planner_schema = {
        "properties": {
            key: {"type": "string", "maxLength": 800}
            for key in ("comparison", "conflict_assessment", "memory_use")
        }
    }
    planner_messages, compact_planner = compact_planner_contract(
        messages, planner_schema
    )
    assert all(
        compact_planner["properties"][key]["maxLength"] == 240
        for key in ("comparison", "conflict_assessment", "memory_use")
    )
    assert "response_contract" in json.loads(planner_messages[-1]["content"])

    final_schema = {
        "properties": {
            "analysis_process": {
                "maxItems": 4,
                "items": {
                    "properties": {
                        "comparison": {"type": "string", "maxLength": 800},
                        "implication": {"type": "string", "maxLength": 800},
                    }
                },
            },
            "reasoning_results": {
                "properties": {
                    "prediction": {
                        "properties": {"value": {"type": "number"}}
                    }
                }
            },
        }
    }
    final_messages, compact_final = compact_final_contract(
        messages, final_schema
    )
    analysis = compact_final["properties"]["analysis_process"]
    value = compact_final["properties"]["reasoning_results"]["properties"][
        "prediction"
    ]["properties"]["value"]
    assert analysis["maxItems"] == 2
    assert analysis["items"]["properties"]["comparison"]["maxLength"] == 320
    assert value == {"type": "string", "pattern": V08_NUMERIC_PATTERN}
    assert "response_contract" in json.loads(final_messages[-1]["content"])


def test_numeric_prediction_is_normalized_to_float():
    response = {"reasoning_results": {"prediction": {"value": "-1.250000"}}}
    normalize_mattext_prediction(response)
    assert response["reasoning_results"]["prediction"]["value"] == -1.25
