"""Shared retrieval, memory, and scoring helpers for MatRecall x MatText.

The language model sees exactly one selected text representation for the
current material.  Retrieval may use the aligned composition view as a tool
index, but the current target is never accepted by any function in this module.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Composition, Element, Lattice
from sklearn.feature_extraction.text import TfidfVectorizer

from matrecall.episodic_memory import canonical_json


REAL_RELEASE_ID = "matrecall-mattext"
CONTROL_RELEASE_ID = "matrecall-mattext-control"
RUN_PROTOCOL = "matrecall-mattext-run"
PROMPT_REVISION = "mattext-compact-numeric"

EXPOSURE_METHODS = (
    "llm_only",
    "static_multiview_rag",
    "dynamic_case_rag",
    "matrecall",
)
FROZEN_METHODS = (
    "llm_only",
    "static_multiview_rag",
    "dynamic_case_rag",
    "matrecall",
    "matrecall_shuffled",
    "matrecall_unrelated",
)

UNRELATED_TASK = {
    "mattext_bulk_modulus_log10_gpa": "mattext_perovskite_formation_energy",
    "mattext_shear_modulus_log10_gpa": "mattext_perovskite_formation_energy",
    "mattext_perovskite_formation_energy": "mattext_bulk_modulus_log10_gpa",
}


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{number}")
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")


def composition_vector(formula: str) -> np.ndarray:
    vector = np.zeros(118, dtype=np.float64)
    try:
        composition = Composition(str(formula)).fractional_composition
        for element, fraction in composition.items():
            vector[Element(element).Z - 1] = float(fraction)
    except (TypeError, ValueError):
        symbols = sorted(set(re.findall(r"[A-Z][a-z]?", str(formula))))
        symbols = [value for value in symbols if Element.is_valid_symbol(value)]
        for symbol in symbols:
            vector[Element(symbol).Z - 1] = 1.0 / max(len(symbols), 1)
    return vector


def cosine_rows(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominator = matrix_norm * query_norm
    scores = np.zeros(len(matrix), dtype=np.float64)
    valid = denominator > 0
    scores[valid] = matrix[valid] @ query / denominator[valid]
    return scores


def _text_scores(documents: list[str], query: str) -> np.ndarray:
    if not documents:
        return np.zeros(0, dtype=np.float64)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([*documents, query])
    values = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    return np.asarray(values, dtype=np.float64)


def retrieve_static(
    case: dict[str, Any],
    static_library: list[dict[str, Any]],
    representation: str,
    *,
    candidate_limit: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    """Retrieve static examples by composition and representation text.

    The selected examples expose the same representation as the query.  The
    aligned composition string is used only to rank records and is not appended
    to a non-composition query.
    """
    if representation not in case["representations"]:
        raise KeyError(representation)
    formula = case["representations"]["composition"]
    composition_matrix = np.stack([
        composition_vector(row["input_views"]["composition"])
        for row in static_library
    ])
    composition_scores = cosine_rows(composition_matrix, composition_vector(formula))
    documents = [str(row["input_views"][representation]) for row in static_library]
    representation_scores = _text_scores(
        documents, str(case["representations"][representation])
    )
    if representation == "composition":
        combined = composition_scores
    else:
        combined = 0.55 * composition_scores + 0.45 * representation_scores
    order_combined = np.argsort(-combined, kind="stable")
    order_composition = np.argsort(-composition_scores, kind="stable")
    order_representation = np.argsort(-representation_scores, kind="stable")
    selected_indices: list[int] = []
    for ordering in (order_combined, order_composition, order_representation):
        for index in ordering[:candidate_limit]:
            value = int(index)
            if value not in selected_indices:
                selected_indices.append(value)
            if len(selected_indices) >= candidate_limit:
                break
        if len(selected_indices) >= candidate_limit:
            break
    references: list[dict[str, Any]] = []
    for rank, index in enumerate(selected_indices, start=1):
        source = static_library[index]
        evidence_id = source["episode_id"] + f":{representation}"
        references.append({
            "evidence_id": evidence_id,
            "input": {
                "representation": representation,
                "material_text": source["input_views"][representation],
            },
            "observed_output": source["observed_output"],
            "retrieval_trace": {
                "rank": rank,
                "hybrid_score": float(combined[index]),
                "composition_cosine": float(composition_scores[index]),
                "representation_cosine": float(representation_scores[index]),
            },
            "graph_paths": [],
        })
    by_index = {
        index: static_library[index]["episode_id"] + f":{representation}"
        for index in selected_indices
    }
    views = {
        "hybrid": [by_index[int(value)] for value in order_combined if int(value) in by_index],
        "composition": [by_index[int(value)] for value in order_composition if int(value) in by_index],
        "representation": [by_index[int(value)] for value in order_representation if int(value) in by_index],
    }
    top = references[: min(6, len(references))]
    weights = np.asarray([
        1.0 / math.log2(index + 2) for index in range(len(top))
    ])
    values = np.asarray([float(row["observed_output"]["value"]) for row in top])
    anchor = {
        "value": float(np.average(values, weights=weights)),
        "unit": case["target_unit"],
        "rule": "rank-weighted mean of train-fold static examples",
    }
    return references, views, anchor


def runtime_case(
    case: dict[str, Any],
    static_library: list[dict[str, Any]],
    representation: str,
) -> dict[str, Any]:
    references, views, anchor = retrieve_static(case, static_library, representation)
    return {
        "protocol": RUN_PROTOCOL,
        "case_id": case["case_id"],
        "task": case["task"],
        "task_kind": "scalar_regression",
        "task_instruction": case["task_instruction"],
        "target_unit": case["target_unit"],
        "query_visible": {
            "representation": representation,
            "material_text": case["representations"][representation],
        },
        "prediction_schema": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "deterministic_tool_estimate": anchor,
        "references": references,
        "initial_evidence_ids": [row["evidence_id"] for row in references[:4]],
        "reference_views": views,
        "retrieval": {"view": "hybrid"},
        "memory_metadata": {
            "material_id": case["material_id"],
            "source_representation": representation,
            "normalized_composition": case["representations"]["composition"],
        },
    }


def _episode_composition(episode: dict[str, Any]) -> np.ndarray:
    metadata = episode.get("memory_metadata", {})
    return composition_vector(str(metadata.get("normalized_composition", "")))


def retrieve_full_experience(
    memory: list[dict[str, Any]], case: dict[str, Any], limit: int,
) -> list[dict[str, Any]]:
    if limit < 1 or not memory:
        return []
    query = composition_vector(case["representations"]["composition"])
    matrix = np.stack([_episode_composition(row) for row in memory])
    similarity = cosine_rows(matrix, query)
    ranked = sorted(
        range(len(memory)),
        key=lambda index: (-float(similarity[index]), -int(memory[index]["sequence_step"])),
    )
    chosen = ranked[:limit]
    output = []
    for index in chosen:
        episode = memory[index]
        output.append({
            "experience_id": episode["experience_id"],
            "source_task": episode["task"],
            "source_representation": episode.get("memory_metadata", {}).get(
                "source_representation"
            ),
            "normalized_composition": episode.get("memory_metadata", {}).get(
                "normalized_composition"
            ),
            "prior_input": episode["query_summary"],
            "prior_prediction": episode["prediction"],
            "verified_ground_truth": episode["ground_truth"],
            "feedback": episode["feedback"],
            "trajectory_summary": episode["trajectory_summary"],
            "lesson": episode["lesson"],
            "retrieval_trace": {
                "composition_cosine": float(similarity[index]),
                "rank": len(output) + 1,
            },
        })
    return output


def shuffle_episode_feedback(
    memory: list[dict[str, Any]], *, namespace: str,
) -> list[dict[str, Any]]:
    """Deterministically rotate outcomes while preserving memory size and queries."""
    if len(memory) < 2:
        return deepcopy(memory)
    shift = 1 + int(hashlib.sha256(namespace.encode()).hexdigest(), 16) % (len(memory) - 1)
    shuffled = deepcopy(memory)
    fields = (
        "prediction", "ground_truth", "feedback", "static_anchor_feedback",
        "lesson", "trajectory_summary", "memory_text",
    )
    for index, row in enumerate(shuffled):
        donor = memory[(index + shift) % len(memory)]
        for field in fields:
            if field in donor:
                row[field] = deepcopy(donor[field])
        row["experience_id"] = row["experience_id"] + ":shuffled"
        row["control"] = {
            "type": "task_matched_shuffled_verified_episode",
            "donor_experience_id": donor["experience_id"],
        }
    return shuffled


def scalar_prediction(row: dict[str, Any]) -> float | None:
    if row.get("status") != "success":
        return None
    try:
        value = float(row["response"]["reasoning_results"]["prediction"]["value"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def regression_metrics(expected: list[float], predicted: list[float]) -> dict[str, Any]:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Expected aligned nonempty regression arrays")
    errors = np.asarray(predicted, dtype=float) - np.asarray(expected, dtype=float)
    return {
        "n": len(expected),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_error": float(np.mean(errors)),
    }


def parse_crystal_text(text: str) -> tuple[Lattice, list[str], np.ndarray]:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError("Malformed crystal_text_llm representation")
    lengths = [float(value) for value in lines[0].split()]
    angles = [float(value) for value in lines[1].split()]
    if len(lengths) != 3 or len(angles) != 3:
        raise ValueError("Malformed lattice in crystal_text_llm representation")
    species: list[str] = []
    coordinates: list[list[float]] = []
    for index in range(2, len(lines), 2):
        if index + 1 >= len(lines):
            raise ValueError("Unpaired species/coordinate row")
        species.append(lines[index])
        coordinate = [float(value) for value in lines[index + 1].split()]
        if len(coordinate) != 3:
            raise ValueError("Malformed fractional coordinate")
        coordinates.append(coordinate)
    return Lattice.from_parameters(*lengths, *angles), species, np.asarray(coordinates)


def structure_descriptor(text: str) -> np.ndarray:
    lattice, species, fractional = parse_crystal_text(text)
    atomic_numbers = np.asarray([Element(value).Z for value in species], dtype=float)
    distance = lattice.get_all_distances(fractional, fractional)
    nonzero = distance[distance > 1e-8]
    nearest = np.where(distance > 1e-8, distance, np.inf).min(axis=1)
    # A primitive cell may contain one crystallographic site.  In that case
    # the finite nearest neighbour is a periodic image, which is not present
    # in the square within-cell distance matrix.  Use the shortest lattice
    # translation as a deterministic finite proxy.
    nearest = np.where(np.isfinite(nearest), nearest, min(lattice.abc))
    histogram, _ = np.histogram(nonzero, bins=np.linspace(0.0, 8.0, 17))
    histogram = histogram.astype(float)
    if histogram.sum():
        histogram /= histogram.sum()
    descriptor = np.asarray([
        *lattice.abc,
        *lattice.angles,
        lattice.volume,
        len(species),
        atomic_numbers.mean(),
        atomic_numbers.std(),
        atomic_numbers.min(),
        atomic_numbers.max(),
        nearest.mean(),
        nearest.std(),
        nearest.min(),
        nearest.max(),
        *[float(np.mean((distance > 1e-8) & (distance < cutoff))) for cutoff in (2, 3, 4, 5)],
        *histogram,
    ], dtype=float)
    if not np.isfinite(descriptor).all():
        raise ValueError("Non-finite structure descriptor")
    return descriptor


def composition_descriptor(formula: str) -> np.ndarray:
    fractions = composition_vector(formula)
    occupied = np.flatnonzero(fractions)
    numbers = occupied.astype(float) + 1.0
    weights = fractions[occupied]
    summary = np.asarray([
        len(occupied),
        float(np.sum(numbers * weights)) if len(occupied) else 0.0,
        float(np.sqrt(np.sum(weights * (numbers - np.sum(numbers * weights)) ** 2)))
        if len(occupied) else 0.0,
        float(numbers.min()) if len(occupied) else 0.0,
        float(numbers.max()) if len(occupied) else 0.0,
    ])
    return np.concatenate([fractions, summary])


def controlled_components(case: dict[str, Any]) -> tuple[float, float]:
    """Self-contained category and coordinate potentials for the coordinate--category control.

    The category component is a nonlinear atomic-number mixture.  The geometry
    component is a bounded nearest-neighbour radial potential.  Both are
    deterministic and are standardized from the training fold by the builder.
    """
    lattice, species, fractional = parse_crystal_text(
        case["representations"]["crystal_text_llm"]
    )
    numbers = np.asarray([Element(value).Z for value in species], dtype=float)
    category = float(np.mean(np.sqrt(numbers)) + 0.1 * np.std(numbers))
    distances = lattice.get_all_distances(fractional, fractional)
    nearest = np.where(distances > 1e-8, distances, np.inf).min(axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, min(lattice.abc))
    geometry = float(np.mean(np.exp(-nearest) + 0.25 / (1.0 + nearest**2)))
    if not (math.isfinite(category) and math.isfinite(geometry)):
        raise ValueError("Non-finite coordinate-category component")
    return category, geometry

