"""Shared task adapters and scoring for the unified-task evaluation.

The nine process-outcome endpoints retain their already frozen evaluator.  This
module covers the other thirteen endpoints and keeps their heterogeneous target
semantics explicit instead of flattening every task to a string exact match.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROTOCOL = "matrecall-unified-evaluation"
PROMPT_REVISION = "unified-task-json"
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

CRYSTAL_TASKS = (
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "bulk_modulus_kv",
    "magmom_oszicar",
)
TASK_ORDER = (
    *CRYSTAL_TASKS,
    "composition_crystal_system_context_to_space_group",
    "composition_to_phase_assemblage",
    "target_property_to_formulation_composition",
    "target_composition_to_precursor_set",
    "target_and_precursors_to_heating_conditions",
    "target_and_precursors_to_operations_and_conditions",
    "zeolite_and_osda_to_synthesis_route_distribution",
    "synthesis_text_to_action_graph",
)

TASK_PATHS = {
    **{
        task: Path("crystal_to_property/alignn_jarvis/tasks") / task
        for task in CRYSTAL_TASKS
    },
    "composition_crystal_system_context_to_space_group": Path(
        "composition_to_crystallographic_class/sg_predict_cubic_space_group"
    ),
    "composition_to_phase_assemblage": Path(
        "composition_to_phase_assemblage/hea_phase"
    ),
    "target_property_to_formulation_composition": Path(
        "target_property_to_formulation_composition/feconi_multi_property"
    ),
    "target_composition_to_precursor_set": Path(
        "target_process_to_precursor_steps_conditions/synthesis_similarity_precursors"
    ),
    "target_and_precursors_to_heating_conditions": Path(
        "target_process_to_precursor_steps_conditions/syntmte_conditions"
    ),
    "target_and_precursors_to_operations_and_conditions": Path(
        "target_process_to_precursor_steps_conditions/spende_operations_conditions"
    ),
    "zeolite_and_osda_to_synthesis_route_distribution": Path(
        "target_process_to_precursor_steps_conditions/diffsyn_zeolite_routes"
    ),
    "synthesis_text_to_action_graph": Path("synthesis_text_to_action_graph/pcmsp"),
}

TASK_KINDS = {
    **{task: "scalar_regression" for task in CRYSTAL_TASKS},
    "composition_crystal_system_context_to_space_group": "ranked_classification",
    "composition_to_phase_assemblage": "set_classification",
    "target_property_to_formulation_composition": "composition_generation",
    "target_composition_to_precursor_set": "formula_set_generation",
    "target_and_precursors_to_heating_conditions": "nullable_numeric_object",
    "target_and_precursors_to_operations_and_conditions": "operation_sequence",
    "zeolite_and_osda_to_synthesis_route_distribution": "route_distribution",
    "synthesis_text_to_action_graph": "action_graph",
}

TASK_INSTRUCTIONS = {
    "formation_energy_peratom": "Predict formation energy per atom in eV/atom.",
    "optb88vdw_bandgap": "Predict the OptB88vdW band gap in eV.",
    "mbj_bandgap": "Predict the TB-mBJ band gap in eV.",
    "bulk_modulus_kv": "Predict the Voigt bulk modulus in GPa.",
    "magmom_oszicar": "Predict total magnetic moment in mu_B per simulation cell.",
    "composition_crystal_system_context_to_space_group": (
        "Given the composition and the supplied cubic crystal-system context, return up to five "
        "ranked cubic space-group numbers, most likely first."
    ),
    "composition_to_phase_assemblage": (
        "Predict the unordered set of reported phase categories for the alloy composition."
    ),
    "target_property_to_formulation_composition": (
        "Propose one normalized alloy composition matching the target property vector using only "
        "the allowed elements."
    ),
    "target_composition_to_precursor_set": (
        "Predict an unordered set of precursor formulas for the target composition."
    ),
    "target_and_precursors_to_heating_conditions": (
        "Predict available calcination and sintering temperatures in degrees Celsius and times "
        "in hours. Use null when a stage is not predicted."
    ),
    "target_and_precursors_to_operations_and_conditions": (
        "Predict the ordered synthesis-operation types and concise per-step condition strings."
    ),
    "zeolite_and_osda_to_synthesis_route_distribution": (
        "Return one to five plausible synthesis routes representing the conditional route "
        "distribution for the zeolite topology and OSDA."
    ),
    "synthesis_text_to_action_graph": (
        "Extract typed entity nodes using token_start inclusive and token_end exclusive, then "
        "directed typed relations between node IDs."
    ),
}

CONDITION_FIELDS = (
    "sintering_temperature_c",
    "sintering_time_h",
    "calcination_temperature_c",
    "calcination_time_h",
)
ROUTE_NUMERIC_FIELDS = (
    "Si/Al",
    "Al/P",
    "Si/Ge",
    "Si/B",
    "Na/T",
    "K/T",
    "OH/T",
    "F/T",
    "H2O/T",
    "sda1/T",
    "cryst_temp",
    "cryst_time",
)
FECONI_ELEMENTS = (
    "Fe", "Co", "Ni", "V", "Mo", "Cr", "Cu", "Mn", "C", "W", "Ta",
    "Nb", "Al", "Ti", "Si",
)
FECONI_PROPERTIES = ("Ms", "Hc", "Tc", "ER", "Elong", "HV", "Cost")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path}:{number}")
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def task_dir(release: Path, task: str) -> Path:
    if task not in TASK_PATHS:
        raise KeyError(task)
    path = release / TASK_PATHS[task]
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(path / "manifest.json")
    return path


def load_task(
    release: Path, task: str, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if split not in ("validation", "test"):
        raise ValueError(split)
    root = task_dir(release, task)
    memory = pd.read_parquet(root / "public" / "memory_episodes.parquet")
    queries = pd.read_parquet(root / "public" / f"{split}_queries.parquet")
    labels = pd.read_parquet(root / "private" / f"{split}_labels.parquet")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if queries["query_id"].duplicated().any():
        raise ValueError(f"Duplicate queries: {task}/{split}")
    if task != "zeolite_and_osda_to_synthesis_route_distribution":
        if labels["query_id"].duplicated().any() or list(queries["query_id"]) != list(labels["query_id"]):
            raise ValueError(f"Query/label mismatch: {task}/{split}")
    elif set(queries["query_id"]) != set(labels["query_id"]):
        raise ValueError(f"Distribution query/label mismatch: {task}/{split}")
    return memory, queries, labels, manifest


def _as_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def memory_input(task: str, row: pd.Series) -> dict[str, Any]:
    if task in CRYSTAL_TASKS:
        context = _as_json(row.get("context_json", "{}"))
        return {
            "formula": str(row["formula"]),
            "structure_reference": str(row["structure_ref"]),
            "structure_context": context,
        }
    return _as_json(row["input_json"])


def query_input(task: str, row: pd.Series) -> dict[str, Any]:
    if task in CRYSTAL_TASKS:
        return {
            "formula": str(row["formula"]),
            "reduced_formula": str(row["reduced_formula"]),
            "chemical_system": str(row["chemical_system"]),
            "anonymous_formula": str(row["anonymous_formula"]),
            "structure_family_proxy": str(row["structure_family_proxy"]),
            "crystal_structure": _compact_atoms(_as_json(row["atoms_json"])),
        }
    return _as_json(row["input_json"])


def _compact_atoms(atoms: dict[str, Any]) -> dict[str, Any]:
    lattice = atoms.get("lattice_mat") or atoms.get("lattice") or []
    coords = atoms.get("coords") or []
    elements = atoms.get("elements") or []
    output: dict[str, Any] = {
        "site_count": len(elements) or len(coords),
        "elements": sorted(Counter(map(str, elements)).items()),
        "coordinate_basis": "cartesian" if atoms.get("cartesian") else "fractional",
        "sites": [
            {
                "element": str(element),
                "coordinates": [round(float(value), 5) for value in coordinate],
            }
            for element, coordinate in list(zip(elements, coords))[:128]
        ],
        "sites_truncated": max(0, (len(elements) or len(coords)) - 128),
    }
    try:
        matrix = np.asarray(lattice, dtype=float).reshape(3, 3)
        lengths = np.linalg.norm(matrix, axis=1)
        angles = []
        for left, right in ((1, 2), (0, 2), (0, 1)):
            cosine = np.dot(matrix[left], matrix[right]) / (lengths[left] * lengths[right])
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        output.update(
            lattice_lengths_angstrom=[round(float(value), 5) for value in lengths],
            lattice_angles_degree=[round(value, 4) for value in angles],
            volume_angstrom3=round(float(abs(np.linalg.det(matrix))), 5),
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return output


def retrieval_document(task: str, row: pd.Series, *, query: bool) -> str:
    if task in CRYSTAL_TASKS:
        if query:
            values = [
                str(row.get("formula", "")),
                str(row.get("reduced_formula", "")),
                str(row.get("chemical_system", "")),
                str(row.get("anonymous_formula", "")),
                str(row.get("structure_family_proxy", "")),
            ]
        else:
            values = [str(row.get("formula", "")), str(row.get("context_json", ""))]
        return " ".join(values)
    text = str(row.get("input_text", "") or "")
    return text + "\n" + str(row.get("input_json", "") or "")


def memory_target(task: str, row: pd.Series) -> Any:
    if task in CRYSTAL_TASKS:
        return {"value": float(row["target_value"])}
    if task == "composition_crystal_system_context_to_space_group":
        return {"ranked_space_group_numbers": [int(row["space_group_number"])]}
    if task == "composition_to_phase_assemblage":
        payload = _as_json(row["target_json"])
        return {"phase_set": payload["phase_set"]}
    if task == "target_property_to_formulation_composition":
        payload = _as_json(row["target_json"])
        return {
            "composition": [
                {"element": element, "fraction": float(value)}
                for element, value in payload["element_fractions"].items()
            ]
        }
    if task == "target_composition_to_precursor_set":
        payload = _as_json(row["target_json"])
        formulas = valid_formula_strings(payload["precursor_formulas"])
        if not formulas:
            raise ValueError("Precursor target is empty after filtering missing placeholders")
        return {"precursor_formulas": formulas}
    if task == "target_and_precursors_to_heating_conditions":
        payload = _as_json(row["target_json"])
        return {name: payload.get(name) for name in CONDITION_FIELDS}
    if task == "target_and_precursors_to_operations_and_conditions":
        payload = _as_json(row["target_json"])
        return {
            "operation_sequence": payload["operation_sequence"],
            "operation_steps": payload["operation_steps"],
        }
    if task == "zeolite_and_osda_to_synthesis_route_distribution":
        return {"routes": [_route_from_row(row)]}
    if task == "synthesis_text_to_action_graph":
        return _as_json(row["target_json"])
    raise KeyError(task)


def _route_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    get = row.get
    route: dict[str, Any] = {}
    for name in ROUTE_NUMERIC_FIELDS:
        value = get(name)
        route[name] = float(value) if value is not None and pd.notna(value) else None
    route["precursors_text"] = str(get("precursors_text", "") or "")
    return route


def gold_targets(task: str, labels: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if task == "zeolite_and_osda_to_synthesis_route_distribution":
        for query_id, group in labels.groupby("query_id", sort=False):
            output[str(query_id)] = {"routes": [_route_from_row(row) for _, row in group.iterrows()]}
        return output
    for _, row in labels.iterrows():
        query_id = str(row["query_id"])
        if task in CRYSTAL_TASKS:
            target = {"value": float(row["target_value"])}
        elif task == "composition_crystal_system_context_to_space_group":
            allowed = _as_json(row["allowed_space_group_numbers_json"])
            counts = {
                int(key): int(value)
                for key, value in _as_json(row["label_counts_json"]).items()
            }
            reference = sorted(allowed, key=lambda value: (-counts.get(int(value), 0), int(value)))[0]
            target = {
                "allowed_space_group_numbers": allowed,
                "reference_space_group_number": int(reference),
            }
        elif task == "composition_to_phase_assemblage":
            target = {"allowed_phase_sets": _as_json(row["allowed_phase_sets_json"])}
        elif task == "target_property_to_formulation_composition":
            candidates = _as_json(row["candidate_compositions_json"])
            target = {
                "candidate_compositions": [
                    candidate["element_fractions"] for candidate in candidates
                ]
            }
        elif task == "target_composition_to_precursor_set":
            formulas = valid_formula_strings(_as_json(row["precursor_formulas_json"]))
            if not formulas:
                raise ValueError("Precursor gold is empty after filtering missing placeholders")
            target = {"precursor_formulas": formulas}
        elif task == "target_and_precursors_to_heating_conditions":
            target = {name: _finite_or_none(row[name]) for name in CONDITION_FIELDS}
        elif task == "target_and_precursors_to_operations_and_conditions":
            target = {
                "operation_sequence": _as_json(row["operation_sequence_json"]),
                "operation_steps": _as_json(row["operation_steps_json"]),
            }
        elif task == "synthesis_text_to_action_graph":
            target = _as_json(row["action_graph_json"])
        else:
            raise KeyError(task)
        output[query_id] = target
    return output


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def valid_formula_strings(values: Iterable[Any]) -> list[str]:
    """Remove source missing-value placeholders from formula-set targets."""
    return [str(value).strip() for value in values if isinstance(value, str) and value.strip()]


def prediction_schema(task: str, memory: pd.DataFrame) -> dict[str, Any]:
    kind = TASK_KINDS[task]
    if kind == "scalar_regression":
        return _object({"value": {"type": "number"}}, ["value"])
    if kind == "ranked_classification":
        labels = sorted(int(value) for value in memory["space_group_number"].unique())
        return _object(
            {
                "ranked_space_group_numbers": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "integer", "enum": labels},
                }
            },
            ["ranked_space_group_numbers"],
        )
    if kind == "set_classification":
        phases = sorted(
            {
                phase
                for value in memory["target_json"]
                for phase in _as_json(value)["phase_set"]
            }
        )
        return _object(
            {"phase_set": {"type": "array", "maxItems": len(phases), "items": {"type": "string", "enum": phases}}},
            ["phase_set"],
        )
    if kind == "composition_generation":
        component = _object(
            {
                "element": {"type": "string", "enum": list(FECONI_ELEMENTS)},
                "fraction": {"type": "number", "minimum": 0, "maximum": 1},
            },
            ["element", "fraction"],
        )
        return _object(
            {"composition": {"type": "array", "minItems": 1, "maxItems": len(FECONI_ELEMENTS), "items": component}},
            ["composition"],
        )
    if kind == "formula_set_generation":
        return _object(
            {"precursor_formulas": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 80}}},
            ["precursor_formulas"],
        )
    if kind == "nullable_numeric_object":
        nullable = {"anyOf": [{"type": "number"}, {"type": "null"}]}
        return _object({name: nullable for name in CONDITION_FIELDS}, list(CONDITION_FIELDS))
    if kind == "operation_sequence":
        operation_types = sorted(
            {
                operation
                for value in memory["operation_sequence_json"]
                for operation in _as_json(value)
            }
        )
        step = _object(
            {
                "type": {"type": "string", "enum": operation_types},
                "name": {"type": "string", "minLength": 1, "maxLength": 120},
                "conditions": {
                    "anyOf": [
                        {"type": "string", "maxLength": 500},
                        {"type": "null"},
                    ]
                },
            },
            ["type", "name", "conditions"],
        )
        return _object(
            {
                "operation_sequence": {"type": "array", "minItems": 1, "maxItems": 30, "items": {"type": "string", "enum": operation_types}},
                "operation_steps": {"type": "array", "minItems": 1, "maxItems": 30, "items": step},
            },
            ["operation_sequence", "operation_steps"],
        )
    if kind == "route_distribution":
        properties = {
            name: {"anyOf": [{"type": "number"}, {"type": "null"}]}
            for name in ROUTE_NUMERIC_FIELDS
        }
        properties["precursors_text"] = {"type": "string", "maxLength": 600}
        route = _object(properties, [*ROUTE_NUMERIC_FIELDS, "precursors_text"])
        return _object(
            {"routes": {"type": "array", "minItems": 1, "maxItems": 5, "items": route}},
            ["routes"],
        )
    if kind == "action_graph":
        entity_types = sorted(
            {
                node["type"]
                for value in memory["target_json"]
                for node in _as_json(value)["nodes"]
            }
        )
        relation_types = sorted(
            {
                edge["type"]
                for value in memory["target_json"]
                for edge in _as_json(value)["edges"]
            }
        )
        node = _object(
            {
                "id": {"type": "string", "minLength": 1, "maxLength": 40},
                "type": {"type": "string", "enum": entity_types},
                "token_start": {"type": "integer", "minimum": 0},
                "token_end": {"type": "integer", "minimum": 1},
            },
            ["id", "type", "token_start", "token_end"],
        )
        edge = _object(
            {
                "source": {"type": "string", "minLength": 1, "maxLength": 40},
                "target": {"type": "string", "minLength": 1, "maxLength": 40},
                "type": {"type": "string", "enum": relation_types},
            },
            ["source", "target", "type"],
        )
        return _object(
            {
                "nodes": {"type": "array", "maxItems": 80, "items": node},
                "edges": {"type": "array", "maxItems": 160, "items": edge},
            },
            ["nodes", "edges"],
        )
    raise KeyError(task)


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def envelope(prediction: Any, evidence_ids: list[str], comparison: str) -> dict[str, Any]:
    return {
        "analysis_process": [
            {
                "comparison": comparison,
                "evidence_ids": evidence_ids,
                "implication": "Prediction transferred from train-only task evidence.",
            }
        ],
        "reasoning_results": {"prediction": prediction, "evidence_ids": evidence_ids},
    }


def transfer_prediction(task: str, targets: list[Any], similarities: list[float]) -> Any:
    weights = np.asarray([max(float(value), 0.0) + 1e-9 for value in similarities])
    kind = TASK_KINDS[task]
    if kind == "scalar_regression":
        return {"value": float(np.average([value["value"] for value in targets], weights=weights))}
    if kind == "ranked_classification":
        votes: dict[int, float] = {}
        for target, weight in zip(targets, weights):
            label = int(target["ranked_space_group_numbers"][0])
            votes[label] = votes.get(label, 0.0) + float(weight)
        ranked = sorted(votes, key=lambda label: (-votes[label], label))[:5]
        return {"ranked_space_group_numbers": ranked}
    if kind == "set_classification":
        votes: dict[str, float] = {}
        values: dict[str, list[str]] = {}
        for target, weight in zip(targets, weights):
            key = canonical_json(sorted(set(target["phase_set"])))
            votes[key] = votes.get(key, 0.0) + float(weight)
            values[key] = json.loads(key)
        selected = max(votes, key=lambda key: (votes[key], key))
        return {"phase_set": values[selected]}
    if kind == "composition_generation":
        total: dict[str, float] = {}
        for target, weight in zip(targets, weights):
            for item in target["composition"]:
                total[item["element"]] = total.get(item["element"], 0.0) + float(weight) * float(item["fraction"])
        denominator = sum(total.values())
        return {"composition": [{"element": key, "fraction": value / denominator} for key, value in sorted(total.items()) if value > 0]}
    if kind == "nullable_numeric_object":
        result = {}
        for field in CONDITION_FIELDS:
            pairs = [(target.get(field), weight) for target, weight in zip(targets, weights) if target.get(field) is not None]
            result[field] = float(np.average([value for value, _ in pairs], weights=[weight for _, weight in pairs])) if pairs else None
        return result
    if kind == "route_distribution":
        routes = []
        for target in targets:
            routes.extend(target["routes"])
            if len(routes) >= 5:
                break
        return {"routes": routes[:5]}
    # Exact transfer is the only well-defined deterministic operation for
    # formula sets, operation sequences and token-indexed action graphs.
    return targets[0]


def prediction_from_row(row: dict[str, Any]) -> Any | None:
    if row.get("status") != "success":
        return None
    try:
        return row["response"]["reasoning_results"]["prediction"]
    except (KeyError, TypeError):
        return None


def primary_selection_score(task: str, expected: list[Any], predicted: list[Any]) -> tuple[float, bool]:
    """Return validation selection value and whether larger is better."""
    kind = TASK_KINDS[task]
    if kind == "scalar_regression":
        value = mean_absolute_error(
            [row["value"] for row in expected], [row["value"] for row in predicted]
        )
        return float(value), False
    if kind == "ranked_classification":
        values = [int(pred["ranked_space_group_numbers"][0] in gold["allowed_space_group_numbers"]) for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), True
    if kind == "set_classification":
        values = [best_set_jaccard(pred["phase_set"], gold["allowed_phase_sets"]) for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), True
    if kind == "composition_generation":
        values = [composition_l1(pred, gold) for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), False
    if kind == "formula_set_generation":
        values = [set_f1(pred["precursor_formulas"], gold["precursor_formulas"]) for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), True
    if kind == "nullable_numeric_object":
        errors = []
        for gold, pred in zip(expected, predicted):
            for field in CONDITION_FIELDS:
                if gold.get(field) is not None and pred.get(field) is not None:
                    errors.append(abs(float(gold[field]) - float(pred[field])))
        return float(np.mean(errors)) if errors else float("inf"), False
    if kind == "operation_sequence":
        values = [set_jaccard(pred["operation_sequence"], gold["operation_sequence"]) for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), True
    if kind == "route_distribution":
        values = [route_distribution_distance(pred["routes"], gold["routes"])["mean_wasserstein"] for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), False
    if kind == "action_graph":
        values = [graph_scores(pred, gold)["entity_f1"] for gold, pred in zip(expected, predicted)]
        return float(np.mean(values)), True
    raise KeyError(task)


def set_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(map(str, left)), set(map(str, right))
    return len(a & b) / len(a | b) if a or b else 1.0


def set_f1(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(map(str, left)), set(map(str, right))
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def best_set_jaccard(predicted: Iterable[Any], allowed: Iterable[Iterable[Any]]) -> float:
    return max((set_jaccard(predicted, value) for value in allowed), default=0.0)


def composition_dict(prediction: dict[str, Any]) -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    valid = True
    for item in prediction.get("composition", []):
        element = str(item.get("element", ""))
        try:
            fraction = float(item.get("fraction"))
        except (TypeError, ValueError):
            valid = False
            continue
        if element not in FECONI_ELEMENTS or not math.isfinite(fraction) or fraction < 0:
            valid = False
            continue
        values[element] = values.get(element, 0.0) + fraction
    total = sum(values.values())
    valid = valid and bool(values) and abs(total - 1.0) <= 0.02
    if total > 0:
        values = {key: value / total for key, value in values.items()}
    return values, valid


def composition_l1(prediction: dict[str, Any], gold: dict[str, Any]) -> float:
    values, _ = composition_dict(prediction)
    candidates = gold["candidate_compositions"]
    return min(
        sum(abs(values.get(element, 0.0) - candidate.get(element, 0.0)) for element in FECONI_ELEMENTS)
        for candidate in candidates
    )


def route_distribution_distance(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    per_field = {}
    distances = []
    for field in ROUTE_NUMERIC_FIELDS:
        left = [float(row[field]) for row in predicted if row.get(field) is not None and math.isfinite(float(row[field]))]
        right = [float(row[field]) for row in gold if row.get(field) is not None and math.isfinite(float(row[field]))]
        value = float(wasserstein_distance(left, right)) if left and right else None
        per_field[field] = value
        if value is not None:
            distances.append(value)
    return {"per_field_wasserstein": per_field, "mean_wasserstein": float(np.mean(distances)) if distances else float("inf")}


def _entity_key(node: dict[str, Any]) -> tuple[int, int, str] | None:
    try:
        start, end = int(node["token_start"]), int(node["token_end"])
        kind = str(node["type"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end, kind) if 0 <= start < end else None


def graph_scores(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    predicted_nodes = {_entity_key(node) for node in prediction.get("nodes", [])}
    predicted_nodes.discard(None)
    gold_nodes = {_entity_key(node) for node in gold.get("nodes", [])}
    gold_nodes.discard(None)
    entity_p, entity_r, entity_f = prf(predicted_nodes, gold_nodes)

    def edges(payload: dict[str, Any]) -> set[tuple[Any, Any, str]]:
        by_id = {str(node.get("id")): _entity_key(node) for node in payload.get("nodes", [])}
        result = set()
        for edge in payload.get("edges", []):
            source, target = by_id.get(str(edge.get("source"))), by_id.get(str(edge.get("target")))
            if source is not None and target is not None:
                result.add((source, target, str(edge.get("type"))))
        return result

    edge_p, edge_r, edge_f = prf(edges(prediction), edges(gold))
    return {
        "entity_precision": entity_p,
        "entity_recall": entity_r,
        "entity_f1": entity_f,
        "relation_precision": edge_p,
        "relation_recall": edge_r,
        "relation_f1": edge_f,
    }


def prf(predicted: set[Any], gold: set[Any]) -> tuple[float, float, float]:
    true = len(predicted & gold)
    precision = true / len(predicted) if predicted else float(not gold)
    recall = true / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_task(task: str, gold: list[Any], predicted: list[Any | None]) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError(f"Unaligned or empty predictions for {task}")
    successful = [(left, right) for left, right in zip(gold, predicted) if right is not None]
    summary: dict[str, Any] = {
        "task": task,
        "task_kind": TASK_KINDS[task],
        "n_cases": len(gold),
        "n_success": len(successful),
        "response_success_rate": len(successful) / len(gold),
    }
    if not successful:
        return summary
    expected, values = zip(*successful)
    kind = TASK_KINDS[task]
    if kind == "scalar_regression":
        y = np.asarray([row["value"] for row in expected], dtype=float)
        p = np.asarray([row["value"] for row in values], dtype=float)
        summary.update(
            mae=float(mean_absolute_error(y, p)),
            rmse=float(math.sqrt(mean_squared_error(y, p))),
            r2=float(r2_score(y, p)) if len(y) > 1 else None,
        )
    elif kind == "ranked_classification":
        top = [int(row["ranked_space_group_numbers"][0]) for row in values]
        reference = [int(row["reference_space_group_number"]) for row in expected]
        summary.update(
            top1_accuracy=float(np.mean([pred in row["allowed_space_group_numbers"] for pred, row in zip(top, expected)])),
            top3_accuracy=float(np.mean([bool(set(pred["ranked_space_group_numbers"][:3]) & set(row["allowed_space_group_numbers"])) for pred, row in zip(values, expected)])),
            top5_accuracy=float(np.mean([bool(set(pred["ranked_space_group_numbers"][:5]) & set(row["allowed_space_group_numbers"])) for pred, row in zip(values, expected)])),
            macro_f1=float(f1_score(reference, top, average="macro", zero_division=0)),
        )
    elif kind == "set_classification":
        exact = [sorted(set(pred["phase_set"])) in [sorted(set(value)) for value in row["allowed_phase_sets"]] for pred, row in zip(values, expected)]
        jaccard = [best_set_jaccard(pred["phase_set"], row["allowed_phase_sets"]) for pred, row in zip(values, expected)]
        summary.update(exact_set_accuracy=float(np.mean(exact)), set_jaccard=float(np.mean(jaccard)))
    elif kind == "composition_generation":
        validity = [composition_dict(row)[1] for row in values]
        summary.update(
            composition_validity=float(np.mean(validity)),
            nearest_gold_l1=float(np.mean([composition_l1(pred, row) for pred, row in zip(values, expected)])),
        )
        forward_pairs = [
            (target, prediction)
            for target, prediction in zip(expected, values)
            if target.get("target_properties") is not None
            and prediction.get("forward_surrogate_properties") is not None
        ]
        if forward_pairs:
            per_property = {}
            standardized = []
            for field in FECONI_PROPERTIES:
                errors = [
                    abs(float(prediction["forward_surrogate_properties"][field]) - float(target["target_properties"][field]))
                    for target, prediction in forward_pairs
                ]
                per_property[field] = {"mae": float(np.mean(errors)), "n_scored": len(errors)}
                scales = [max(float(target["forward_property_scales"][field]), 1e-12) for target, _ in forward_pairs]
                standardized.extend(error / scale for error, scale in zip(errors, scales))
            summary["forward_surrogate_per_property"] = per_property
            summary["forward_surrogate_standardized_mae"] = float(np.mean(standardized))
    elif kind == "formula_set_generation":
        scores = [set_f1(pred["precursor_formulas"], row["precursor_formulas"]) for pred, row in zip(values, expected)]
        exact = [set(pred["precursor_formulas"]) == set(row["precursor_formulas"]) for pred, row in zip(values, expected)]
        summary.update(precursor_set_f1=float(np.mean(scores)), precursor_exact_match=float(np.mean(exact)))
    elif kind == "nullable_numeric_object":
        per_field = {}
        for field in CONDITION_FIELDS:
            pairs = [(row[field], pred[field]) for row, pred in zip(expected, values) if row.get(field) is not None and pred.get(field) is not None]
            missing = sum(row.get(field) is not None and pred.get(field) is None for row, pred in zip(expected, values))
            presence_gold = [row.get(field) is not None for row in expected]
            presence_predicted = [pred.get(field) is not None for pred in values]
            per_field[field] = {
                "n_gold": sum(row.get(field) is not None for row in expected),
                "n_scored": len(pairs),
                "missing_prediction_rate": missing / max(sum(row.get(field) is not None for row in expected), 1),
                "presence_accuracy": float(accuracy_score(presence_gold, presence_predicted)),
                "presence_f1": float(f1_score(presence_gold, presence_predicted, zero_division=0)),
                "mae": float(mean_absolute_error([a for a, _ in pairs], [b for _, b in pairs])) if pairs else None,
                "rmse": float(math.sqrt(mean_squared_error([a for a, _ in pairs], [b for _, b in pairs]))) if pairs else None,
            }
        summary["per_field"] = per_field
    elif kind == "operation_sequence":
        condition_scores = [
            operation_condition_scores(pred.get("operation_steps", []), row.get("operation_steps", []))
            for pred, row in zip(values, expected)
        ]
        summary.update(
            operation_sequence_jaccard=float(np.mean([set_jaccard(pred["operation_sequence"], row["operation_sequence"]) for pred, row in zip(values, expected)])),
            exact_sequence_accuracy=float(np.mean([pred["operation_sequence"] == row["operation_sequence"] for pred, row in zip(values, expected)])),
            operation_step_type_accuracy=float(np.mean([row["step_type_accuracy"] for row in condition_scores])),
            condition_token_f1=float(np.mean([row["condition_token_f1"] for row in condition_scores])),
            condition_numeric_mae=float(np.mean([row["condition_numeric_mae"] for row in condition_scores if row["condition_numeric_mae"] is not None]))
            if any(row["condition_numeric_mae"] is not None for row in condition_scores) else None,
        )
    elif kind == "route_distribution":
        records = [route_distribution_distance(pred["routes"], row["routes"]) for pred, row in zip(values, expected)]
        finite_distances = [row["mean_wasserstein"] for row in records if math.isfinite(row["mean_wasserstein"])]
        summary["mean_wasserstein_unscaled"] = float(np.mean(finite_distances)) if finite_distances else None
        summary["per_field_wasserstein"] = {
            field: float(np.mean([row["per_field_wasserstein"][field] for row in records if row["per_field_wasserstein"][field] is not None]))
            if any(row["per_field_wasserstein"][field] is not None for row in records) else None
            for field in ROUTE_NUMERIC_FIELDS
        }
        summary["per_field_rbf_mmd"] = {
            field: _mean_optional([
                rbf_mmd_1d(
                    [float(route[field]) for route in pred["routes"] if route.get(field) is not None],
                    [float(route[field]) for route in row["routes"] if route.get(field) is not None],
                )
                for pred, row in zip(values, expected)
            ])
            for field in ROUTE_NUMERIC_FIELDS
        }
    elif kind == "action_graph":
        entity_predicted, entity_gold, relation_predicted, relation_gold = set(), set(), set(), set()
        for case_index, (prediction, target) in enumerate(zip(values, expected)):
            predicted_nodes = {_entity_key(node) for node in prediction.get("nodes", [])}
            predicted_nodes.discard(None)
            gold_nodes = {_entity_key(node) for node in target.get("nodes", [])}
            gold_nodes.discard(None)
            entity_predicted.update((case_index, value) for value in predicted_nodes)
            entity_gold.update((case_index, value) for value in gold_nodes)
            relation_predicted.update((case_index, value) for value in _graph_edges(prediction))
            relation_gold.update((case_index, value) for value in _graph_edges(target))
        ep, er, ef = prf(entity_predicted, entity_gold)
        rp, rr, rf = prf(relation_predicted, relation_gold)
        summary.update(
            entity_precision=ep,
            entity_recall=er,
            entity_f1=ef,
            relation_precision=rp,
            relation_recall=rr,
            relation_f1=rf,
        )
    return summary


def _graph_edges(payload: dict[str, Any]) -> set[tuple[Any, Any, str]]:
    by_id = {str(node.get("id")): _entity_key(node) for node in payload.get("nodes", [])}
    return {
        (by_id[str(edge.get("source"))], by_id[str(edge.get("target"))], str(edge.get("type")))
        for edge in payload.get("edges", [])
        if by_id.get(str(edge.get("source"))) is not None
        and by_id.get(str(edge.get("target"))) is not None
    }


def operation_condition_scores(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(predicted), len(gold), 1)
    matching_types = sum(
        str(predicted[index].get("type")) == str(gold[index].get("type"))
        for index in range(min(len(predicted), len(gold)))
    )
    token_f1s, numeric_errors = [], []
    for pred, target in zip(predicted, gold):
        pred_text = str(pred.get("conditions") or "")
        gold_text = str(target.get("conditions") or "")
        pred_tokens = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+", pred_text.lower())
        gold_tokens = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+", gold_text.lower())
        token_f1s.append(set_f1(pred_tokens, gold_tokens))
        pred_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", pred_text)]
        gold_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", gold_text)]
        numeric_errors.extend(abs(left - right) for left, right in zip(pred_numbers, gold_numbers))
    return {
        "step_type_accuracy": matching_types / count,
        "condition_token_f1": float(np.mean(token_f1s)) if token_f1s else float(not gold),
        "condition_numeric_mae": float(np.mean(numeric_errors)) if numeric_errors else None,
    }


def rbf_mmd_1d(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    combined = np.asarray(left + right, dtype=float)
    pairwise = np.abs(combined[:, None] - combined[None, :])
    positive = pairwise[pairwise > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    bandwidth = max(bandwidth, 1e-12)
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    kernel = lambda x, y: np.exp(-((x[:, None] - y[None, :]) ** 2) / (2 * bandwidth**2))
    value = kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean()
    return float(max(value, 0.0))


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None
