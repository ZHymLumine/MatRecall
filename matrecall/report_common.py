"""Primary-metric selection shared by the report modules."""

from __future__ import annotations

from typing import Any

from matrecall.unified_evaluation import CRYSTAL_TASKS


def metric_rows(task: str, summary: dict[str, Any], *, process: bool) -> list[tuple[str, float | None, bool]]:
    if process:
        values = summary.get("metrics_conditional_on_valid_response") or {}
        kind = summary["task_kind"]
        if kind == "regression":
            return [("MAE", values.get("mae"), False)]
        if kind in ("binary_classification", "multiclass_classification"):
            return [("macro-F1", values.get("macro_f1"), True)]
        return [("exact formula accuracy", values.get("exact_formula_accuracy"), True)]
    if task in CRYSTAL_TASKS:
        return [("MAE", summary.get("mae"), False)]
    if task == "composition_crystal_system_context_to_space_group":
        return [("top-1 accuracy", summary.get("top1_accuracy"), True)]
    if task == "composition_to_phase_assemblage":
        return [("set Jaccard", summary.get("set_jaccard"), True)]
    if task == "target_property_to_formulation_composition":
        return [
            ("forward-surrogate standardized MAE", summary.get("forward_surrogate_standardized_mae"), False),
            ("nearest-gold composition L1", summary.get("nearest_gold_l1"), False),
            ("composition validity", summary.get("composition_validity"), True),
        ]
    if task == "target_composition_to_precursor_set":
        return [("precursor set F1", summary.get("precursor_set_f1"), True)]
    if task == "target_and_precursors_to_heating_conditions":
        rows = []
        for field, values in (summary.get("per_field") or {}).items():
            rows.append((field + " MAE", values.get("mae"), False))
            rows.append((field + " presence F1", values.get("presence_f1"), True))
        return rows
    if task == "target_and_precursors_to_operations_and_conditions":
        return [
            ("operation sequence Jaccard", summary.get("operation_sequence_jaccard"), True),
            ("condition token F1", summary.get("condition_token_f1"), True),
        ]
    if task == "zeolite_and_osda_to_synthesis_route_distribution":
        return [("mean Wasserstein (unscaled; inspect per-field values)", summary.get("mean_wasserstein_unscaled"), False)]
    if task == "synthesis_text_to_action_graph":
        return [("entity F1", summary.get("entity_f1"), True), ("relation F1", summary.get("relation_f1"), True)]
    raise KeyError(task)
