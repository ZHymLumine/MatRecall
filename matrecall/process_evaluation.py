"""Shared loading, retrieval, and scoring for the process-outcome tasks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


PROTOCOL = "matrecall-process-evaluation"
PROMPT_REVISION = "process-outcome-json"
TASK_ORDER = (
    "raccuglia_crystallization_success",
    "inp_absorption_peak_nm",
    "inp_emission_peak_nm",
    "inp_particle_diameter_nm",
    "meltpool_depth_um",
    "meltpool_width_um",
    "meltpool_length_um",
    "meltpool_shape",
    "mgt_reported_product",
)
TASK_INSTRUCTIONS = {
    "raccuglia_crystallization_success": (
        "Predict whether the reported hydrothermal reaction crystallizes successfully. "
        "Return integer 1 for success or 0 for failure."
    ),
    "inp_absorption_peak_nm": "Predict the InP nanocrystal absorption peak in nm.",
    "inp_emission_peak_nm": "Predict the InP nanocrystal emission peak in nm.",
    "inp_particle_diameter_nm": "Predict the InP nanocrystal particle diameter in nm.",
    "meltpool_depth_um": "Predict melt-pool depth in micrometres.",
    "meltpool_width_um": "Predict melt-pool width in micrometres.",
    "meltpool_length_um": "Predict melt-pool length in micrometres.",
    "meltpool_shape": (
        "Predict the reported melt-pool class. Use exactly one allowed class label."
    ),
    "mgt_reported_product": (
        "Predict the reported major solid-state synthesis product formula. Return one "
        "formula string, not a reaction equation or explanation."
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{number}")
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def task_dirs(release: Path) -> list[Path]:
    available = {path.name: path for path in (release / "tasks").iterdir() if path.is_dir()}
    missing = [task for task in TASK_ORDER if task not in available]
    if missing:
        raise ValueError(f"Release is missing tasks: {missing}")
    return [available[task] for task in TASK_ORDER]


def load_task(task_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if split not in ("validation", "test"):
        raise ValueError(f"Unsupported split: {split}")
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    memory = pd.read_parquet(task_dir / "public" / "memory_episodes.parquet")
    queries = pd.read_parquet(task_dir / "public" / f"{split}_queries.parquet")
    labels = pd.read_parquet(task_dir / "private" / f"{split}_labels.parquet")
    if list(queries["query_id"]) != list(labels["query_id"]):
        raise ValueError(f"Query/label order mismatch for {task_dir.name}/{split}")
    if any(column.startswith("target_value") for column in queries.columns):
        raise ValueError(f"Public queries contain target values for {task_dir.name}/{split}")
    return memory, queries, labels, manifest


def target_values(frame: pd.DataFrame, kind: str) -> list[Any]:
    if kind == "regression":
        return [float(value) for value in frame["target_value_numeric"]]
    if kind == "binary_classification":
        return [int(float(value)) for value in frame["target_value_numeric"]]
    return [str(value) for value in frame["target_value_text"]]


def retrieval_document(row: pd.Series | dict[str, Any], view: str) -> str:
    get = row.get if isinstance(row, dict) else row.get
    text = str(get("input_text", "") or "")
    input_json = str(get("input_json", "") or "")
    if view == "text_word":
        return text + "\n" + input_json
    if view == "text_char":
        return text + "\n" + input_json
    raise ValueError(f"Unknown retrieval view: {view}")


def retrieve(
    memory: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    view: str,
    k: int,
) -> list[list[tuple[int, float]]]:
    if k < 1 or k > len(memory):
        raise ValueError(f"Invalid k={k} for memory size {len(memory)}")
    documents = [retrieval_document(row, view) for _, row in memory.iterrows()]
    query_documents = [retrieval_document(row, view) for _, row in queries.iterrows()]
    kwargs = (
        {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 1, "sublinear_tf": True}
        if view == "text_char"
        else {"analyzer": "word", "ngram_range": (1, 2), "min_df": 1, "sublinear_tf": True}
    )
    vectorizer = TfidfVectorizer(**kwargs)
    memory_vectors = vectorizer.fit_transform(documents)
    query_vectors = vectorizer.transform(query_documents)
    result: list[list[tuple[int, float]]] = []
    batch = 256
    for start in range(0, len(queries), batch):
        similarities = cosine_similarity(query_vectors[start : start + batch], memory_vectors)
        for values in similarities:
            selected = np.argpartition(values, -k)[-k:]
            ordered = sorted(selected, key=lambda index: (-float(values[index]), int(index)))
            result.append([(int(index), float(values[index])) for index in ordered])
    return result


def retrieval_predictions(
    memory: pd.DataFrame,
    neighbors: list[list[tuple[int, float]]],
    kind: str,
) -> list[Any]:
    labels = target_values(memory, kind)
    predictions: list[Any] = []
    for matches in neighbors:
        weights = [max(similarity, 0.0) + 1e-9 for _, similarity in matches]
        values = [labels[index] for index, _ in matches]
        if kind == "regression":
            predictions.append(float(np.average(values, weights=weights)))
            continue
        votes: defaultdict[Any, float] = defaultdict(float)
        first_rank: dict[Any, int] = {}
        for rank, (value, weight) in enumerate(zip(values, weights)):
            votes[value] += weight
            first_rank.setdefault(value, rank)
        predictions.append(max(votes, key=lambda value: (votes[value], -first_rank[value], str(value))))
    return predictions


def _formula_composition(value: str) -> Composition:
    text = str(value).strip().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    if not text:
        raise ValueError("Empty formula")
    total = Composition()
    for number, part in enumerate(re.split("[·⋅•]", text)):
        factor = 1.0
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z(].*)", part)
        if number and match:
            factor, part = float(match[1]), match[2]
        # The source corpus contains valid conventional formulas whose element
        # ordering does not satisfy pymatgen's optional electronegativity-order
        # check.  Relax that ordering check, while still rejecting unknown
        # symbols, empty compositions, and non-positive amounts below.
        composition = Composition(part)
        if not composition or any(amount <= 0 for amount in composition.values()):
            raise ValueError(f"Invalid formula: {value}")
        total += factor * composition
    return total


def canonical_formula(value: str) -> str:
    try:
        return _formula_composition(value).reduced_formula
    except (ValueError, TypeError):
        return re.sub(r"\s+", "", str(value))


def element_set_f1(expected: str, predicted: str) -> float | None:
    if canonical_formula(expected) == canonical_formula(predicted):
        return 1.0
    try:
        left = {str(element) for element in _formula_composition(expected).elements}
        right = {str(element) for element in _formula_composition(predicted).elements}
    except (ValueError, TypeError):
        # Text-mined targets can contain variables, solid-solution notation,
        # slash fractions, or mixtures (for example O3-delta or 1/3).  Exact
        # stoichiometry is unresolved, but explicit element symbols remain a
        # usable set-valued endpoint.  Unknown capitalized tokens are ignored;
        # no abbreviation (such as YAG) is expanded into invented chemistry.
        def lexical_elements(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[A-Z][a-z]?", str(value))
                if Element.is_valid_symbol(token)
            }

        left, right = lexical_elements(expected), lexical_elements(predicted)
        if not left or not right:
            return None
    return 2 * len(left & right) / (len(left) + len(right)) if left or right else 1.0


def score_values(
    kind: str,
    expected: list[Any],
    predicted: list[Any],
    *,
    probabilities: list[float | None] | None = None,
) -> dict[str, Any]:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Expected and predicted values must be non-empty and aligned")
    if kind == "regression":
        left, right = np.asarray(expected, dtype=float), np.asarray(predicted, dtype=float)
        result = {
            "mae": float(mean_absolute_error(left, right)),
            "rmse": float(math.sqrt(mean_squared_error(left, right))),
        }
        r2 = float(r2_score(left, right))
        if math.isfinite(r2):
            result["r2"] = r2
        return result
    if kind in ("binary_classification", "multiclass_classification"):
        result = {
            "accuracy": float(accuracy_score(expected, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        }
        if kind == "binary_classification":
            result["f1"] = float(f1_score(expected, predicted, zero_division=0))
            if (
                probabilities
                and all(value is not None for value in probabilities)
                and len(set(expected)) == 2
            ):
                auc = float(roc_auc_score(expected, probabilities))
                # A tiny pilot or group-bootstrap replicate can contain one
                # class only.  Depending on the sklearn version, that case
                # raises ValueError or returns NaN, so skip the undefined
                # endpoint before calling the library.
                if math.isfinite(auc):
                    result["roc_auc"] = auc
        return result
    if kind == "structured_prediction":
        exact = [canonical_formula(left) == canonical_formula(right) for left, right in zip(expected, predicted)]
        element = [element_set_f1(left, right) for left, right in zip(expected, predicted)]
        available = [value for value in element if value is not None]
        return {
            "exact_formula_accuracy": float(np.mean(exact)),
            "element_set_f1": float(np.mean(available)) if available else None,
            "element_set_n_scored": len(available),
            "element_set_coverage": len(available) / len(element),
        }
    raise ValueError(f"Unknown task kind: {kind}")


def selection_metric(kind: str) -> tuple[str, bool]:
    """Return validation metric and whether larger is better."""
    if kind == "regression":
        return "mae", False
    if kind in ("binary_classification", "multiclass_classification"):
        return "balanced_accuracy", True
    if kind == "structured_prediction":
        return "element_set_f1", True
    raise ValueError(kind)


def bootstrap_metrics(
    kind: str,
    expected: list[Any],
    predicted: list[Any],
    groups: list[str],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, list[float] | None]:
    """Group bootstrap every scalar metric; undefined replicates are omitted."""
    if not (len(expected) == len(predicted) == len(groups)):
        raise ValueError("Bootstrap inputs are not aligned")
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    if len(by_group) < 2:
        return {}
    keys = list(by_group)
    rng = np.random.default_rng(seed)
    values: defaultdict[str, list[float]] = defaultdict(list)

    # Formula normalization is substantially more expensive than sampling
    # group indices.  Cache the two per-row structured outcomes once instead
    # of reparsing the same formulas in every bootstrap replicate.
    if kind == "structured_prediction":
        exact = np.asarray(
            [
                canonical_formula(left) == canonical_formula(right)
                for left, right in zip(expected, predicted)
            ],
            dtype=float,
        )
        element = np.asarray(
            [
                np.nan if value is None else float(value)
                for value in (
                    element_set_f1(left, right)
                    for left, right in zip(expected, predicted)
                )
            ],
            dtype=float,
        )
        for _ in range(samples):
            chosen = rng.integers(0, len(keys), size=len(keys))
            indices = [index for item in chosen for index in by_group[keys[int(item)]]]
            values["exact_formula_accuracy"].append(float(np.mean(exact[indices])))
            current = element[indices]
            available = current[np.isfinite(current)]
            if len(available):
                values["element_set_f1"].append(float(np.mean(available)))
        return {
            key: np.quantile(current, [0.025, 0.975]).tolist() if current else None
            for key, current in values.items()
        }

    for _ in range(samples):
        chosen = rng.integers(0, len(keys), size=len(keys))
        indices = [index for item in chosen for index in by_group[keys[int(item)]]]
        try:
            metrics = score_values(
                kind,
                [expected[index] for index in indices],
                [predicted[index] for index in indices],
            )
        except ValueError:
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None and math.isfinite(value):
                values[key].append(float(value))
    return {
        key: np.quantile(current, [0.025, 0.975]).tolist() if current else None
        for key, current in values.items()
        if key not in ("element_set_n_scored", "element_set_coverage")
    }


def class_labels(memory: pd.DataFrame, kind: str) -> list[Any] | None:
    if kind not in ("binary_classification", "multiclass_classification"):
        return None
    return sorted(set(target_values(memory, kind)), key=str)


def target_counter(frame: pd.DataFrame, kind: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(target_values(frame, kind)).items()}
