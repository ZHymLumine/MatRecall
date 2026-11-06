import math

import pandas as pd

from matrecall.unified_evaluation import (
    graph_scores,
    operation_condition_scores,
    prediction_schema,
    route_distribution_distance,
    summarize_task,
    transfer_prediction,
    valid_formula_strings,
)
from matrecall.run_unified_baselines import _predict_graphs, token_features


def test_weighted_scalar_transfer():
    prediction = transfer_prediction(
        "formation_energy_peratom",
        [{"value": 1.0}, {"value": 3.0}],
        [0.75, 0.25],
    )
    assert math.isclose(prediction["value"], 1.5)


def test_space_group_schema_and_ambiguous_reference_scoring():
    memory = pd.DataFrame({"space_group_number": [195, 225, 225]})
    schema = prediction_schema("composition_crystal_system_context_to_space_group", memory)
    assert schema["properties"]["ranked_space_group_numbers"]["items"]["enum"] == [195, 225]
    summary = summarize_task(
        "composition_crystal_system_context_to_space_group",
        [{"allowed_space_group_numbers": [195, 225], "reference_space_group_number": 225}],
        [{"ranked_space_group_numbers": [195, 225]}],
    )
    assert summary["top1_accuracy"] == 1.0
    assert summary["top3_accuracy"] == 1.0
    assert summary["macro_f1"] == 0.0


def test_graph_relations_are_matched_by_typed_spans_not_node_ids():
    gold = {
        "nodes": [
            {"id": "g1", "type": "Material-target", "token_start": 0, "token_end": 1},
            {"id": "g2", "type": "Operation", "token_start": 2, "token_end": 3},
        ],
        "edges": [{"source": "g1", "target": "g2", "type": "Participant-material"}],
    }
    prediction = {
        "nodes": [
            {"id": "p9", "type": "Material-target", "token_start": 0, "token_end": 1},
            {"id": "p8", "type": "Operation", "token_start": 2, "token_end": 3},
        ],
        "edges": [{"source": "p9", "target": "p8", "type": "Participant-material"}],
    }
    scores = graph_scores(prediction, gold)
    assert scores["entity_f1"] == 1.0
    assert scores["relation_f1"] == 1.0


def test_operation_condition_scoring():
    predicted = [{"type": "HeatingOperation", "conditions": "800 C 4 h"}]
    gold = [{"type": "HeatingOperation", "conditions": "800 C 5 h"}]
    scores = operation_condition_scores(predicted, gold)
    assert scores["step_type_accuracy"] == 1.0
    assert 0 < scores["condition_token_f1"] < 1
    assert scores["condition_numeric_mae"] == 0.5


def test_route_distribution_distance_is_per_field():
    predicted = [{"Si/Al": 10.0, "cryst_temp": 100.0}]
    gold = [{"Si/Al": 12.0, "cryst_temp": 110.0}]
    result = route_distribution_distance(predicted, gold)
    assert result["per_field_wasserstein"]["Si/Al"] == 2.0
    assert result["per_field_wasserstein"]["cryst_temp"] == 10.0
    assert result["mean_wasserstein"] == 6.0


def test_missing_precursor_placeholders_are_not_labels():
    assert valid_formula_strings([None, "", " TiO2 "]) == ["TiO2"]


def test_missing_pcmsp_token_keeps_its_index_and_is_explicit():
    values = token_features(["at", None, "K"], 1)
    assert values["token=<missing_token>"] == 1
    assert values["prev=at"] == 1
    assert values["next=k"] == 1


def test_empty_pcmsp_token_sequence_maps_to_empty_graph_without_model_call():
    result = _predict_graphs(
        [{"text": "unannotated source fragment", "tokens": []}],
        token_vectorizer=None,
        token_model=None,
        relation_vectorizer=None,
        relation_model=None,
    )
    assert result == [{"nodes": [], "edges": []}]
