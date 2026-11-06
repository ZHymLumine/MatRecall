"""Domain-aware retrieval and typed episodic-memory helpers for the static memory.

The module deliberately keeps similarity indexes separate from the typed graph.
The graph stores task, source, material/process entities, and outcomes.  Dense
and sparse indexes retrieve episodes; reciprocal-rank fusion combines views.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


AGENT_PROTOCOL = "matrecall-static-memory"
AGENT_PROMPT_REVISION = "bounded-react-multiview"
K_VALUES = (3, 5, 10)
FUSION_DEPTH = 50
RRF_CONSTANT = 60
N_ELEMENTS = 118

CRYSTAL_TASKS = {
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "bulk_modulus_kv",
    "magmom_oszicar",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        if not value.strip():
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def row_payload(task: str, row: pd.Series | dict[str, Any], *, query: bool) -> dict[str, Any]:
    get = row.get
    if task in CRYSTAL_TASKS:
        payload: dict[str, Any] = {
            "formula": get("formula"),
            "chemical_system": get("chemical_system"),
            "anonymous_formula": get("anonymous_formula"),
            "structure_family_proxy": get("structure_family_proxy"),
            "material_id": get("material_id"),
        }
        context = json_value(get("context_json"), {})
        if isinstance(context, dict):
            payload["structure_context"] = context
        if query:
            atoms = json_value(get("atoms_json"), {})
            if isinstance(atoms, dict):
                payload["crystal_structure"] = atoms
        return {key: value for key, value in payload.items() if value not in (None, "")}
    payload = json_value(get("input_json"), {})
    if not isinstance(payload, dict):
        payload = {"input": payload}
    text = str(get("input_text", "") or "")
    if text:
        payload = {**payload, "input_text": text}
    return payload


def lexical_document(task: str, row: pd.Series | dict[str, Any], *, query: bool) -> str:
    payload = row_payload(task, row, query=query)
    return canonical_json(payload)


def _formula_vector(formula: str) -> tuple[np.ndarray, set[str], str | None]:
    vector = np.zeros(N_ELEMENTS, dtype=np.float32)
    try:
        raw_composition = Composition(str(formula))
        composition = raw_composition.fractional_composition
        for element, amount in composition.items():
            vector[Element(element).Z - 1] += float(amount)
        return vector, {str(element) for element in composition.elements}, raw_composition.reduced_formula
    except (ValueError, TypeError, KeyError):
        elements = {
            token for token in re.findall(r"[A-Z][a-z]?", str(formula))
            if Element.is_valid_symbol(token)
        }
        if elements:
            value = 1.0 / len(elements)
            for symbol in elements:
                vector[Element(symbol).Z - 1] = value
        return vector, elements, None


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{prefix}[]")
    else:
        yield prefix, value


def composition_features(payload: dict[str, Any]) -> np.ndarray:
    vector = np.zeros(N_ELEMENTS, dtype=np.float32)
    direct: dict[str, float] = {}
    formulas: list[str] = []
    for path, value in _walk(payload):
        key = path.lower()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            match = re.search(r"(?:^|\.)([A-Z][a-z]?)\s*\((?:wt|at)\s*%", path)
            if match and Element.is_valid_symbol(match.group(1)) and float(value) > 0:
                direct[match.group(1)] = direct.get(match.group(1), 0.0) + float(value)
            continue
        if isinstance(value, str) and ("formula" in key or key.endswith(".material")):
            formulas.append(value)
        if isinstance(value, str) and key.endswith("chemical_system"):
            formulas.extend(value.split("-"))
    fractions = payload.get("element_fractions")
    if isinstance(fractions, dict):
        for symbol, amount in fractions.items():
            if Element.is_valid_symbol(str(symbol)):
                direct[str(symbol)] = direct.get(str(symbol), 0.0) + max(0.0, float(amount))
    if direct:
        for symbol, amount in direct.items():
            vector[Element(symbol).Z - 1] += amount
    for formula in formulas:
        formula_vector, _, _ = _formula_vector(formula)
        vector += formula_vector
    total = float(vector.sum())
    return vector / total if total else vector


def entity_tokens(payload: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    composition = composition_features(payload)
    for index in np.flatnonzero(composition):
        tokens.add(f"element:{Element.from_Z(int(index) + 1).symbol}")
    for path, value in _walk(payload):
        key = path.lower()
        if value is None:
            continue
        if isinstance(value, bool):
            tokens.add(f"{key}:{str(value).lower()}")
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            if "formula" in key or key.endswith(".material"):
                _, _, reduced = _formula_vector(text)
                if reduced:
                    tokens.add(f"formula:{reduced}")
            if any(marker in key for marker in (
                "operation", "process", "protocol", "material", "source", "sol",
                "ligand", "acid", "precursor", "osda", "zeolite", "crystal_system",
                "structure_family", "space_group", "anonymous_formula",
            )):
                normalized = re.sub(r"\s+", " ", text.lower())[:160]
                tokens.add(f"{key}:{normalized}")
            if key.endswith("text") or key.endswith("tokens[]"):
                for lexical in re.findall(r"[A-Za-z][A-Za-z0-9+_.-]{1,31}", text.lower()):
                    tokens.add(f"lexeme:{lexical}")
    if not tokens:
        tokens.add("entity:<none>")
    return tokens


def numeric_features(payload: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    ignored = ("source_row", "year", "token_start", "token_end", "char_start", "char_end", ".i")
    for path, value in _walk(payload):
        if any(path.lower().endswith(marker) for marker in ignored):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            output[path.lower()] = number
    return output


def base_views(task: str, family: str) -> tuple[str, ...]:
    if task in CRYSTAL_TASKS:
        return (
            "text_char", "composition", "local_environment", "xrd", "soap_mu2",
            "prototype", "graph_entity",
        )
    if task == "synthesis_text_to_action_graph":
        return ("text_word", "text_char", "graph_entity")
    if task in {
        "composition_crystal_system_context_to_space_group",
        "composition_to_phase_assemblage",
    }:
        return ("text_char", "composition", "graph_entity")
    if task == "zeolite_and_osda_to_synthesis_route_distribution":
        return ("text_char", "graph_entity")
    if family == "process" or task in {
        "target_property_to_formulation_composition",
    }:
        return ("text_word", "text_char", "composition", "numeric", "graph_entity")
    if task in {
        "target_composition_to_precursor_set",
        "target_and_precursors_to_heating_conditions",
        "target_and_precursors_to_operations_and_conditions",
    }:
        return ("text_word", "text_char", "composition", "graph_entity")
    return ("text_word", "text_char", "graph_entity")


def _nearest_dense(memory: np.ndarray, queries: np.ndarray, depth: int) -> list[list[tuple[int, float]]]:
    memory_count = int(memory.shape[0])
    query_count = int(queries.shape[0])
    count = min(depth, memory_count)
    if not count:
        return [[] for _ in range(query_count)]
    model = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=4)
    model.fit(memory)
    result: list[list[tuple[int, float]]] = []
    for start in range(0, query_count, 256):
        distances, indices = model.kneighbors(queries[start : start + 256])
        result.extend([
            [(int(index), float(max(0.0, 1.0 - distance))) for index, distance in zip(row_i, row_d)]
            for row_i, row_d in zip(indices, distances)
        ])
    return result


def _nearest_sparse(memory: sparse.spmatrix, queries: sparse.spmatrix, depth: int) -> list[list[tuple[int, float]]]:
    return _nearest_dense(memory, queries, depth)


def _text_rankings(memory_docs: list[str], query_docs: list[str], view: str, depth: int):
    kwargs = (
        {"analyzer": "char_wb", "ngram_range": (3, 5)}
        if view == "text_char" else
        {"analyzer": "word", "ngram_range": (1, 2)}
    )
    vectorizer = TfidfVectorizer(
        **kwargs, min_df=1, sublinear_tf=True, max_features=120000, dtype=np.float32
    )
    memory_vectors = vectorizer.fit_transform(memory_docs)
    query_vectors = vectorizer.transform(query_docs)
    return _nearest_sparse(memory_vectors, query_vectors, depth)


def _entity_rankings(memory_tokens: list[set[str]], query_tokens: list[set[str]], depth: int):
    documents = [" ".join(sorted(tokens)) for tokens in memory_tokens]
    queries = [" ".join(sorted(tokens)) for tokens in query_tokens]
    vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, use_idf=False, norm="l2")
    memory_vectors = vectorizer.fit_transform(documents)
    query_vectors = vectorizer.transform(queries)
    return _nearest_sparse(memory_vectors, query_vectors, depth)


def _numeric_rankings(memory_values: list[dict[str, float]], query_values: list[dict[str, float]], depth: int):
    counts: defaultdict[str, int] = defaultdict(int)
    for row in memory_values:
        for key in row:
            counts[key] += 1
    columns = sorted(key for key, count in counts.items() if count >= 2)
    if not columns:
        return [[] for _ in query_values]
    memory = np.full((len(memory_values), len(columns)), np.nan, dtype=np.float32)
    queries = np.full((len(query_values), len(columns)), np.nan, dtype=np.float32)
    positions = {key: index for index, key in enumerate(columns)}
    for row_index, row in enumerate(memory_values):
        for key, value in row.items():
            if key in positions:
                memory[row_index, positions[key]] = value
    for row_index, row in enumerate(query_values):
        for key, value in row.items():
            if key in positions:
                queries[row_index, positions[key]] = value
    medians = np.nanmedian(memory, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    scales = np.nanmedian(np.abs(memory - medians), axis=0) * 1.4826
    fallback = np.nanstd(memory, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, fallback)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    memory = np.nan_to_num((memory - medians) / scales, nan=0.0)
    queries = np.nan_to_num((queries - medians) / scales, nan=0.0)
    count = min(depth, len(memory))
    model = NearestNeighbors(n_neighbors=count, metric="euclidean", algorithm="auto", n_jobs=4)
    model.fit(memory)
    output = []
    normalizer = max(1.0, math.sqrt(len(columns)))
    for start in range(0, len(queries), 256):
        distances, indices = model.kneighbors(queries[start : start + 256])
        output.extend([
            [(int(index), float(math.exp(-float(distance) / normalizer))) for index, distance in zip(row_i, row_d)]
            for row_i, row_d in zip(indices, distances)
        ])
    return output


class JarvisStructureIndexes:
    """Read-only adapter over the already built JARVIS structure-descriptor index."""

    def __init__(self, root: Path):
        self.root = root
        self.ids: np.ndarray | None = None
        self.positions: dict[str, int] = {}
        self.vectors: dict[str, np.ndarray] = {}
        for view in ("composition", "local_environment", "xrd", "soap_mu2"):
            path = root / f"{view}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                ids = payload["material_ids"].astype(str)
                vectors = payload["vectors"].astype(np.float32)
            if self.ids is None:
                self.ids = ids
                self.positions = {value: index for index, value in enumerate(ids)}
            elif not np.array_equal(ids, self.ids):
                raise ValueError(f"JARVIS index IDs differ for {view}")
            self.vectors[view] = vectors
        self.metadata = pd.read_parquet(root / "metadata.parquet")
        self.metadata.index = self.metadata["material_id"].astype(str)

    def ranking(self, view: str, memory_ids: list[str], query_ids: list[str], depth: int):
        missing = sorted((set(memory_ids) | set(query_ids)) - self.positions.keys())
        if missing:
            raise ValueError(f"JARVIS descriptor index misses IDs: {missing[:5]}")
        vectors = self.vectors[view]
        memory = vectors[[self.positions[value] for value in memory_ids]]
        queries = vectors[[self.positions[value] for value in query_ids]]
        return _nearest_dense(memory, queries, depth)

    def prototype_ranking(self, memory_ids: list[str], query_ids: list[str], depth: int):
        fields = ["spacegroup_number", "wyckoff_anonymous", "aflow_prototype", "pearson_symbol"]
        memory_docs = []
        query_docs = []
        for ids, target in ((memory_ids, memory_docs), (query_ids, query_docs)):
            for material_id in ids:
                row = self.metadata.loc[material_id]
                target.append(" ".join(
                    f"{field}:{row[field]}" for field in fields
                    if field in row and pd.notna(row[field]) and str(row[field])
                ))
        vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, use_idf=False)
        memory = vectorizer.fit_transform(memory_docs)
        queries = vectorizer.transform(query_docs)
        return _nearest_sparse(memory, queries, depth)


def reciprocal_rank_fusion(
    rankings: dict[str, list[list[tuple[int, float]]]],
    *,
    depth: int,
) -> list[list[tuple[int, float]]]:
    if not rankings:
        raise ValueError("No rankings to fuse")
    count = len(next(iter(rankings.values())))
    if any(len(value) != count for value in rankings.values()):
        raise ValueError("Ranking query counts differ")
    output = []
    for query_index in range(count):
        scores: defaultdict[int, float] = defaultdict(float)
        for rows in rankings.values():
            for rank, (memory_index, _) in enumerate(rows[query_index], start=1):
                scores[memory_index] += 1.0 / (RRF_CONSTANT + rank)
        ordered = sorted(scores, key=lambda index: (-scores[index], index))[:depth]
        maximum = max((scores[index] for index in ordered), default=1.0)
        output.append([(index, float(scores[index] / maximum)) for index in ordered])
    return output


def retrieve_multiview(
    task: str,
    family: str,
    memory: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    depth: int = FUSION_DEPTH,
    jarvis_indexes: JarvisStructureIndexes | None = None,
) -> tuple[dict[str, list[list[tuple[int, float]]]], list[set[str]], list[set[str]]]:
    views = base_views(task, family)
    memory_payloads = [row_payload(task, row, query=False) for _, row in memory.iterrows()]
    query_payloads = [row_payload(task, row, query=True) for _, row in queries.iterrows()]
    memory_entities = [entity_tokens(value) for value in memory_payloads]
    query_entities = [entity_tokens(value) for value in query_payloads]
    rankings: dict[str, list[list[tuple[int, float]]]] = {}
    memory_docs = [lexical_document(task, row, query=False) for _, row in memory.iterrows()]
    query_docs = [lexical_document(task, row, query=True) for _, row in queries.iterrows()]
    for view in views:
        if view in ("text_word", "text_char"):
            rankings[view] = _text_rankings(memory_docs, query_docs, view, depth)
        elif view == "composition" and task in CRYSTAL_TASKS:
            if jarvis_indexes is None:
                raise ValueError("Crystal tasks require JARVIS indexes")
            rankings[view] = jarvis_indexes.ranking(
                view,
                memory["material_id"].astype(str).tolist(),
                queries["material_id"].astype(str).tolist(),
                depth,
            )
        elif view == "composition":
            rankings[view] = _nearest_dense(
                np.stack([composition_features(value) for value in memory_payloads]),
                np.stack([composition_features(value) for value in query_payloads]),
                depth,
            )
        elif view in ("local_environment", "xrd", "soap_mu2"):
            if jarvis_indexes is None:
                raise ValueError("Crystal tasks require JARVIS indexes")
            rankings[view] = jarvis_indexes.ranking(
                view,
                memory["material_id"].astype(str).tolist(),
                queries["material_id"].astype(str).tolist(),
                depth,
            )
        elif view == "prototype":
            if jarvis_indexes is None:
                raise ValueError("Crystal tasks require JARVIS indexes")
            rankings[view] = jarvis_indexes.prototype_ranking(
                memory["material_id"].astype(str).tolist(),
                queries["material_id"].astype(str).tolist(),
                depth,
            )
        elif view == "numeric":
            rankings[view] = _numeric_rankings(
                [numeric_features(value) for value in memory_payloads],
                [numeric_features(value) for value in query_payloads],
                depth,
            )
        elif view == "graph_entity":
            rankings[view] = _entity_rankings(memory_entities, query_entities, depth)
        else:
            raise KeyError(view)
    rankings["multiview_rrf"] = reciprocal_rank_fusion(rankings, depth=depth)
    return rankings, memory_entities, query_entities


def ranking_score_maps(
    rankings: dict[str, list[list[tuple[int, float]]]], query_index: int
) -> dict[int, dict[str, dict[str, float | int]]]:
    output: defaultdict[int, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for view, all_rows in rankings.items():
        for rank, (memory_index, score) in enumerate(all_rows[query_index], start=1):
            output[memory_index][view] = {"rank": rank, "score": float(score)}
    return dict(output)


def build_graph_frames(
    task: str,
    family: str,
    memory: pd.DataFrame,
    targets: list[Any],
    entities: list[set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (len(memory) == len(targets) == len(entities)):
        raise ValueError("Graph inputs are not aligned")
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(node_id: str, node_type: str, payload: Any) -> None:
        record = {"node_id": node_id, "node_type": node_type, "payload_json": canonical_json(payload)}
        previous = nodes.setdefault(node_id, record)
        if previous != record:
            raise ValueError(f"Conflicting graph node: {node_id}")

    task_id = f"task:{task}"
    node(task_id, "task", {"task": task, "family": family})
    for index, (_, row) in enumerate(memory.iterrows()):
        episode_id = str(row.get("episode_id", row.get("query_id", f"{task}:memory:{index}")))
        episode_node = f"episode:{episode_id}"
        source = str(row.get("source_dataset", "") or "unknown")
        source_id = f"source:{digest(source)[:20]}"
        outcome_id = f"outcome:{digest([task, targets[index]])[:24]}"
        node(episode_node, "episode", {"episode_id": episode_id, "task": task})
        node(source_id, "source", {"source_dataset": source})
        node(outcome_id, "outcome", {"task": task, "value": targets[index]})
        edges.extend([
            {"source_id": episode_node, "relation": "BELONGS_TO_TASK", "target_id": task_id},
            {"source_id": episode_node, "relation": "SUPPORTED_BY", "target_id": source_id},
            {"source_id": episode_node, "relation": "YIELDS", "target_id": outcome_id},
        ])
        for token in sorted(entities[index]):
            entity_id = f"entity:{digest(token)[:24]}"
            entity_type = token.split(":", 1)[0]
            node(entity_id, entity_type, {"token": token})
            edges.append({"source_id": episode_node, "relation": "HAS_ENTITY", "target_id": entity_id})
    return pd.DataFrame(nodes.values()), pd.DataFrame(edges)


def graph_overlap_paths(query_entities: set[str], episode_entities: set[str], limit: int = 12) -> list[str]:
    return [f"query--HAS_ENTITY-->{token}<--HAS_ENTITY--episode" for token in sorted(query_entities & episode_entities)[:limit]]

