import pandas as pd

from matrecall.static_retrieval import (
    AGENT_PROTOCOL,
    build_graph_frames,
    composition_features,
    entity_tokens,
    reciprocal_rank_fusion,
)


def test_composition_and_graph_entities_recover_formula_elements():
    payload = {
        "target_formula": "LiFePO4",
        "precursor_formulas": ["Li2CO3", "Fe2O3"],
        "temperature_c": 800,
    }
    vector = composition_features(payload)
    tokens = entity_tokens(payload)
    assert abs(float(vector.sum()) - 1.0) < 1e-6
    assert {"element:Li", "element:Fe", "element:P", "element:O"} <= tokens
    assert "formula:LiFePO4" in tokens


def test_rrf_rewards_episode_retrieved_by_multiple_views():
    rankings = {
        "composition": [[(0, 0.9), (1, 0.8)]],
        "process": [[(1, 0.95), (2, 0.7)]],
    }
    fused = reciprocal_rank_fusion(rankings, depth=3)[0]
    assert fused[0][0] == 1
    assert fused[0][1] == 1.0


def test_typed_graph_contains_task_source_outcome_and_entity_edges():
    memory = pd.DataFrame({
        "episode_id": ["train:1"],
        "source_dataset": ["source-a"],
    })
    nodes, edges = build_graph_frames(
        "task-a", "unified", memory, [{"value": 2.0}], [{"element:Fe"}]
    )
    assert {"episode", "task", "source", "outcome", "element"} <= set(nodes["node_type"])
    assert {"BELONGS_TO_TASK", "SUPPORTED_BY", "YIELDS", "HAS_ENTITY"} <= set(edges["relation"])


def _case():
    return {
        "agent_protocol": AGENT_PROTOCOL,
        "prompt_revision": "bounded-react-multiview",
        "task_kind": "scalar_regression",
        "prediction_schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "initial_evidence_ids": ["e1"],
        "reference_views": {"composition": ["e1"], "structure": ["e2"]},
        "references": [
            {"evidence_id": "e1", "observed_output": {"value": 1.0}},
            {"evidence_id": "e2", "observed_output": {"value": 2.0}},
        ],
    }


