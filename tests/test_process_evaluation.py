import pandas as pd

from matrecall.process_evaluation import (
    bootstrap_metrics,
    canonical_formula,
    element_set_f1,
    retrieval_predictions,
    score_values,
)


def case(kind="regression"):
    return {
        "protocol": "matrecall-process-evaluation",
        "prompt_revision": "process-outcome-json",
        "case_id": "query:test",
        "task": "inp_absorption_peak_nm",
        "task_kind": kind,
        "task_instruction": "Predict the value.",
        "target_name": "absorption_peak",
        "target_unit": "nm",
        "allowed_labels": ["LOF", "keyhole"] if kind == "multiclass_classification" else None,
        "query_visible": {"input_text": "query secret-free input", "input": {"temperature": 250}},
        "references": [{
            "evidence_id": "episode:train:1",
            "input_text": "training input",
            "input": {"temperature": 240},
            "observed_outcome": {"value": 500.0},
        }],
    }


def test_variable_formula_element_metric_keeps_explicit_elements_without_inventing_stoichiometry():
    assert element_set_f1("BaZr0.8Y0.2O3-δ", "BaZr0.7Y0.3O3-δ") == 1.0
    assert element_set_f1("LiFePO4", "LiMnPO4") == 0.75
    assert canonical_formula("Fe2O3") == "Fe2O3"


def test_retrieval_regression_and_classification_predictions_are_similarity_weighted():
    regression = pd.DataFrame({"target_value_numeric": [1.0, 9.0], "target_value_text": [None, None]})
    assert retrieval_predictions(regression, [[(0, 0.9), (1, 0.1)]], "regression")[0] < 3
    classification = pd.DataFrame({"target_value_numeric": [0.0, 1.0, 1.0], "target_value_text": [None] * 3})
    prediction = retrieval_predictions(classification, [[(0, 0.8), (1, 0.5), (2, 0.5)]], "binary_classification")
    assert prediction == [1]


def test_structured_scoring_reports_full_lexical_element_coverage():
    metrics = score_values(
        "structured_prediction",
        ["BaZr0.8Y0.2O3-δ", "LiFePO4"],
        ["BaZr0.7Y0.3O3-δ", "LiMnPO4"],
    )
    assert metrics["element_set_coverage"] == 1.0
    assert metrics["element_set_n_scored"] == 2
    assert metrics["exact_formula_accuracy"] == 0.0


def test_structured_group_bootstrap_reports_cached_formula_metrics():
    intervals = bootstrap_metrics(
        "structured_prediction",
        ["LiFePO4", "LiMnPO4", "Fe2O3"],
        ["LiFePO4", "LiFePO4", "FeO"],
        ["paper-a", "paper-a", "paper-b"],
        samples=50,
        seed=7,
    )
    assert set(intervals) == {"exact_formula_accuracy", "element_set_f1"}
    assert all(len(bounds) == 2 for bounds in intervals.values())


def test_single_class_binary_pilot_omits_undefined_auc():
    metrics = score_values(
        "binary_classification", [1, 1], [1, 0], probabilities=[0.8, 0.2]
    )
    assert "roc_auc" not in metrics


def test_single_response_regression_pilot_omits_undefined_r2():
    metrics = score_values("regression", [10.0], [12.0])
    assert metrics == {"mae": 2.0, "rmse": 2.0}
