"""Build the unified task release: the base release plus four additional task families.

The builder copies a completed base unified release and adds datasets for:

* composition plus a known cubic crystal-system context to space group;
* synthesis sentence to typed action graph;
* a target Fe-Co-Ni property vector to candidate alloy composition; and
* composition to an allowed set of phase assemblages.

The historical PXRD-peak-list task is deliberately not built.  Its released
feature-table links no longer return CSV data, while reconstructing the task
from SIMPOD would require a new peak-finding protocol and roughly 26 GB of
source tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from matrecall.build_crystal_reference import file_sha256
from matrecall.build_process_tasks import (
    assign_group_splits,
    canonical_json,
    connect_source_and_input_groups,
)
from matrecall.build_unified_tasks_base import (
    _finalize_task,
    _new_task_dir,
    _standard_tables,
    _stable_hash,
    _write_frame,
    _write_json,
)


RELEASE_ID = "matrecall-unified-tasks"
SG_TASK = "composition_crystal_system_context_to_space_group"
PCMSP_TASK = "synthesis_text_to_action_graph"
FECONI_TASK = "target_property_to_formulation_composition"
HEA_TASK = "composition_to_phase_assemblage"

SG_URL = "https://github.com/Yuxinya/SG_predict"
PCMSP_URL = "https://github.com/Xianjun-Yang/PcMSP"
FECONI_URL = (
    "https://github.com/Shakti-95/"
    "Multi-propertyML_Multi-objectiveBO_FeCoNi_Alloy_design"
)
HEA_URL = "https://github.com/Iman-Peivaste/ML_HEAs_Phase_Dataset"

FECONI_ELEMENTS = (
    "Fe", "Co", "Ni", "V", "Mo", "Cr", "Cu", "Mn", "C", "W", "Ta",
    "Nb", "Al", "Ti", "Si",
)
FECONI_PROPERTIES = ("Ms", "Hc", "Tc", "ER", "Elong", "HV", "Cost")
HEA_ELEMENTS = (
    "Li", "Be", "B", "C", "N", "Na", "Mg", "Al", "Si", "P", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Hf", "Ta", "W", "Re", "Ir",
)


def _directory_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = str(path.relative_to(root)).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _drop_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [
        column
        for column in frame.columns
        if not str(column).strip() or str(column).startswith("Unnamed:")
    ]
    return frame.drop(columns=drop)


def _fraction_dict(values: dict[str, float], *, tolerance: float = 1e-12) -> dict[str, float]:
    positive = {key: float(value) for key, value in values.items() if float(value) > tolerance}
    total = sum(positive.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Composition has no positive finite components")
    return {key: round(value / total, 12) for key, value in sorted(positive.items())}


def _parse_spaced_composition(value: Any) -> dict[str, float]:
    text = str(value).strip()
    pairs = re.findall(r"([A-Z][a-z]?)\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not pairs:
        raise ValueError(f"Could not parse composition: {value!r}")
    amounts: dict[str, float] = {}
    for element, amount in pairs:
        amounts[element] = amounts.get(element, 0.0) + float(amount)
    return _fraction_dict(amounts)


def _input_signature(input_value: dict[str, Any]) -> str:
    return _stable_hash(input_value)


def _base_episode(
    *,
    episode_id: str,
    task_family: str,
    source_record_id: str,
    source_group: str,
    input_value: dict[str, Any],
    input_text: str,
    target_value: dict[str, Any],
    context_value: dict[str, Any],
    protocol: str,
    source_dataset: str,
    source_revision: str,
    source_url: str,
    source_sha256: str,
    epistemic_status: str,
    quality_value: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "task_family": task_family,
        "source_record_id": source_record_id,
        "source_group": source_group,
        "input_json": canonical_json(input_value),
        "input_text": input_text,
        "target_json": canonical_json(target_value),
        "context_json": canonical_json(context_value),
        "protocol": protocol,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_url": source_url,
        "source_file_sha256": source_sha256,
        "epistemic_status": epistemic_status,
        "quality_json": canonical_json(quality_value),
        "input_signature": _input_signature(input_value),
    }


def _query_from_row(row: pd.Series, query_id: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "task_family": row["task_family"],
        "input_json": row["input_json"],
        "input_text": row["input_text"],
        "context_json": row["context_json"],
        "protocol": row["protocol"],
        "input_signature": row["input_signature"],
        "source_dataset": row["source_dataset"],
    }


def _extra_frame_files(
    task_dir: Path, base: str, name: str, frame: pd.DataFrame
) -> list[tuple[Path, int | None]]:
    return [(path, len(frame)) for path in _write_frame(task_dir / base, name, frame)]


def _split_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()}


def _check_disjoint(frame: pd.DataFrame, column: str) -> None:
    overlap = frame.groupby(column)["split"].nunique()
    if int(overlap.max()) != 1:
        raise ValueError(f"{column} crosses splits")


def load_sg_predict_rows(path: Path, source_revision: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"mp-id": str, "space_group": int})
    required = {"mp-id", "space_group", "formula", "full_formula", "composition"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"SG_predict source is missing columns: {sorted(missing)}")
    source_sha = file_sha256(path)
    rows: list[dict[str, Any]] = []
    for source_row, row in raw.iterrows():
        fractions = _parse_spaced_composition(row["composition"])
        chemical_system = "-".join(sorted(fractions))
        input_value = {
            "reduced_formula": str(row["formula"]).strip(),
            "element_fractions": fractions,
            "crystal_system_context": "cubic",
        }
        target_value = {
            "space_group_number": int(row["space_group"]),
            "crystal_system": "cubic",
        }
        source_id = str(row["mp-id"]).strip()
        record = _base_episode(
            episode_id=f"sg_predict:{source_id}:space_group",
            task_family=SG_TASK,
            source_record_id=source_id,
            source_group="composition:" + _stable_hash(fractions)[:20],
            input_value=input_value,
            input_text=(
                f"Composition: {input_value['reduced_formula']}. Known crystal-system "
                "context: cubic. Predict the space-group number."
            ),
            target_value=target_value,
            context_value={"crystal_system_is_given": True},
            protocol="Materials Project record used by SG_predict",
            source_dataset="SG_predict cubic Materials Project subset",
            source_revision=source_revision,
            source_url=SG_URL,
            source_sha256=source_sha,
            epistemic_status="computed_structure_label",
            quality_value={"source_row": int(source_row), "full_formula": str(row["full_formula"])},
        )
        record.update(
            {
                "formula": input_value["reduced_formula"],
                "chemical_system": chemical_system,
                "space_group_number": int(row["space_group"]),
                "leakage_group": "composition:" + _stable_hash(fractions)[:20],
            }
        )
        rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("SG_predict source produced no rows")
    return frame


def _build_grouped_classification_tables(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label_output_key: str,
    query_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    memory = frame.loc[frame["split"] == "train"].copy()
    train_systems = set(memory.get("chemical_system", pd.Series(dtype=str)).astype(str))
    query_frames: dict[str, pd.DataFrame] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        queries: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        selected = frame.loc[frame["split"] == split]
        for ordinal, (_, group) in enumerate(selected.groupby("input_signature", sort=True)):
            first = group.iloc[0]
            query_id = f"{query_prefix}:{split}:{ordinal:06d}"
            query = _query_from_row(first, query_id)
            if "chemical_system" in group:
                system = str(first["chemical_system"])
                query["chemical_system"] = system
                query["challenge_new_chemical_system"] = system not in train_systems
            counts = Counter(int(value) if isinstance(value, (int, np.integer)) else str(value) for value in group[label_column])
            ordered_labels = sorted(counts, key=lambda value: (str(type(value)), value))
            queries.append(query)
            labels.append(
                {
                    "query_id": query_id,
                    f"allowed_{label_output_key}_json": canonical_json(ordered_labels),
                    "label_counts_json": canonical_json({str(key): counts[key] for key in ordered_labels}),
                    "source_record_ids_json": canonical_json(sorted(group["source_record_id"].astype(str))),
                }
            )
        query_frames[split] = pd.DataFrame(queries)
        label_frames[split] = pd.DataFrame(labels)
    return (
        memory,
        query_frames["validation"],
        query_frames["test"],
        label_frames["validation"],
        label_frames["test"],
    )


def build_sg_predict(path: Path, task_dir: Path, source_revision: str) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame = load_sg_predict_rows(path, source_revision)
    for label in sorted(frame["space_group_number"].unique()):
        frame[f"_label_{label}"] = (frame["space_group_number"] == label).astype(int)
    balance = [column for column in frame.columns if column.startswith("_label_")]
    frame["split"], split_metadata = assign_group_splits(
        frame, balance, fractions=(0.70, 0.15, 0.15), seed=20261023, candidates=256
    )
    frame = frame.drop(columns=balance)
    _check_disjoint(frame, "input_signature")
    tables = _build_grouped_classification_tables(
        frame,
        label_column="space_group_number",
        label_output_key="space_group_numbers",
        query_prefix="sgcubic",
    )
    membership = frame[
        [
            "source_record_id", "formula", "chemical_system", "space_group_number",
            "input_signature", "leakage_group", "split",
        ]
    ].copy()
    conflict_counts = frame.groupby("input_signature")["space_group_number"].nunique()
    membership["composition_has_multiple_labels"] = membership["input_signature"].map(conflict_counts).gt(1)
    written = _standard_tables(task_dir, *tables, membership)
    return _finalize_task(
        task_dir,
        {
            "dataset": "SG_predict cubic Materials Project subset",
            "task": SG_TASK,
            "scope_limit": "the crystal system is supplied as cubic; this is not unconditional seven-system prediction",
            "source": {
                "url": SG_URL,
                "revision": source_revision,
                "path": str(path),
                "sha256": file_sha256(path),
            },
            "specialized_model": "random-forest classifier with composition and enhanced descriptors",
            "split_protocol": split_metadata,
            "counts": {
                "source_rows": len(frame),
                "unique_compositions": int(frame["input_signature"].nunique()),
                "space_group_classes": int(frame["space_group_number"].nunique()),
                "conflicting_compositions": int((conflict_counts > 1).sum()),
                "rows_by_split": _split_counts(frame),
                "memory_episodes": len(tables[0]),
                "validation_queries": len(tables[1]),
                "test_queries": len(tables[2]),
            },
            "metrics": ["top-1 accuracy", "top-3 accuracy", "top-5 accuracy", "macro-F1"],
            "target_semantics": "a prediction is correct if it matches any released space-group label for the exact composition",
            "published_result_comparability": "requires retraining on this new composition-grouped split",
        },
        written,
    )


def _entity_label(value: str, token_position: str) -> tuple[str, str] | None:
    if value == "_":
        return None
    match = re.fullmatch(r"(.+?)(?:\[([^]]+)\])?", value)
    if not match:
        raise ValueError(f"Invalid WebAnno entity label: {value!r}")
    entity_type, annotation_id = match.groups()
    return entity_type, annotation_id or f"token:{token_position}"


def parse_webanno_tsv(path: Path) -> list[dict[str, Any]]:
    """Parse the PcMSP WebAnno TSV into sentence-level canonical graphs."""
    # PcMSP contains PDF-extraction control bytes inside #Text fields.  Python's
    # splitlines() recognizes several control bytes as line boundaries, whereas
    # the upstream parser and the TSV format split only on newline bytes.
    lines = [line.rstrip("\r") for line in path.read_text(encoding="utf-8-sig").split("\n")]
    sentences: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("#Text="):
            index += 1
            continue
        text = lines[index][6:]
        index += 1
        token_lines: list[str] = []
        while index < len(lines) and lines[index] and not lines[index].startswith("#Text="):
            if not lines[index].startswith("#"):
                token_lines.append(lines[index])
            index += 1
        sentences.append((text, token_lines))

    records: list[dict[str, Any]] = []
    for sentence_index, (text, token_lines) in enumerate(sentences):
        tokens: list[dict[str, Any]] = []
        for raw in token_lines:
            columns = raw.split("\t")
            if len(columns) < 6:
                raise ValueError(f"Malformed WebAnno row in {path}: {raw!r}")
            token_position, char_position, token_text, entity, relation_type, relation_of = columns[:6]
            token_number = int(token_position.rsplit("-", 1)[1]) - 1
            char_start, char_end = [int(value) for value in char_position.split("-", 1)]
            tokens.append(
                {
                    "position": token_position,
                    "index": token_number,
                    "text": token_text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "entity": entity,
                    "relation_type": relation_type,
                    "relation_of": relation_of,
                }
            )

        entity_members: dict[tuple[str, str], list[dict[str, Any]]] = {}
        token_to_key: dict[str, tuple[str, str]] = {}
        for token in tokens:
            parsed = _entity_label(token["entity"], token["position"])
            if parsed is None:
                continue
            entity_members.setdefault(parsed, []).append(token)
            token_to_key[token["position"]] = parsed
        ordered = sorted(entity_members.items(), key=lambda item: min(token["index"] for token in item[1]))
        key_to_id = {key: f"n{ordinal}" for ordinal, (key, _) in enumerate(ordered)}
        token_to_id: dict[str, str] = {}
        nodes: list[dict[str, Any]] = []
        for key, members in ordered:
            members = sorted(members, key=lambda token: token["index"])
            node_id = key_to_id[key]
            for member in members:
                token_to_id[member["position"]] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "type": key[0],
                    "token_start": members[0]["index"],
                    "token_end": members[-1]["index"] + 1,
                    "char_start": members[0]["char_start"],
                    "char_end": members[-1]["char_end"],
                    "text": " ".join(member["text"] for member in members),
                }
            )

        edges: list[dict[str, str]] = []
        for token in tokens:
            if token["relation_type"] == "_":
                continue
            target = token_to_id.get(token["position"])
            if target is None:
                raise ValueError(f"Relation target is not an entity in {path}: {token['position']}")
            relation_types = token["relation_type"].split("|")
            references = token["relation_of"].split("|")
            if len(relation_types) != len(references):
                raise ValueError(f"Mismatched WebAnno relations in {path}: {token['position']}")
            for relation_type, reference in zip(relation_types, references):
                reference_position = reference.split("[", 1)[0]
                source = token_to_id.get(reference_position)
                if source is None:
                    raise ValueError(f"Relation source is not an entity in {path}: {reference}")
                edges.append({"source": source, "target": target, "type": relation_type})
        edges = sorted(
            {canonical_json(edge): edge for edge in edges}.values(),
            key=lambda edge: (edge["source"], edge["target"], edge["type"]),
        )
        records.append(
            {
                "sentence_index": sentence_index,
                "text": text,
                "tokens": [token["text"] for token in tokens],
                "nodes": nodes,
                "edges": edges,
            }
        )
    return records


def load_pcmsp_rows(root: Path, source_revision: str) -> tuple[pd.DataFrame, str, int]:
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    tsv_files = [path for split in split_map for path in (root / split).glob("*.tsv") if not path.name.startswith(".")]
    if not tsv_files:
        raise ValueError(f"No PcMSP TSV files found under {root}")
    source_sha = _directory_sha256(tsv_files, root)
    rows: list[dict[str, Any]] = []
    for upstream_split, split in split_map.items():
        for path in sorted((root / upstream_split).glob("*.tsv")):
            if path.name.startswith("."):
                continue
            document_id = path.stem
            for record in parse_webanno_tsv(path):
                source_id = f"{document_id}:sentence:{record['sentence_index']:04d}"
                input_value = {
                    "text": record["text"],
                    "tokens": record["tokens"],
                }
                target_value = {"nodes": record["nodes"], "edges": record["edges"]}
                episode = _base_episode(
                    episode_id=f"pcmsp:{source_id}",
                    task_family=PCMSP_TASK,
                    source_record_id=source_id,
                    source_group=f"document:{document_id}",
                    input_value=input_value,
                    input_text=record["text"],
                    target_value=target_value,
                    context_value={"document_id": document_id, "sentence_index": record["sentence_index"]},
                    protocol="PcMSP WebAnno TSV 3.3 annotation",
                    source_dataset="PcMSP",
                    source_revision=source_revision,
                    source_url=PCMSP_URL,
                    source_sha256=source_sha,
                    epistemic_status="literature_annotated",
                    quality_value={
                        "node_count": len(record["nodes"]),
                        "edge_count": len(record["edges"]),
                        "empty_graph": not record["nodes"] and not record["edges"],
                    },
                )
                episode.update(
                    {
                        "document_id": document_id,
                        "sentence_index": record["sentence_index"],
                        "text_signature": _stable_hash(" ".join(record["text"].lower().split())),
                        "node_count": len(record["nodes"]),
                        "edge_count": len(record["edges"]),
                        "split": split,
                        "leakage_group": f"document:{document_id}",
                    }
                )
                rows.append(episode)
    frame = pd.DataFrame(rows)
    return frame, source_sha, len(tsv_files)


def build_pcmsp(root: Path, task_dir: Path, source_revision: str) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame, source_sha, document_count = load_pcmsp_rows(root, source_revision)
    signature_splits = frame.groupby("text_signature")["split"].agg(lambda values: sorted(set(values)))
    leaking_signatures = {
        signature for signature, splits in signature_splits.items() if "train" in splits and len(splits) > 1
    }
    evaluation_documents = set(frame.loc[frame["split"] != "train", "document_id"])
    document_overlap = (frame["split"] == "train") & frame["document_id"].isin(evaluation_documents)
    sentence_overlap = (frame["split"] == "train") & frame["text_signature"].isin(leaking_signatures)
    exclusion_mask = document_overlap | sentence_overlap
    memory_exclusions = frame.loc[
        exclusion_mask,
        ["source_record_id", "document_id", "sentence_index", "text_signature"],
    ].copy()
    if not memory_exclusions.empty:
        memory_exclusions["reason"] = [
            ";".join(
                reason
                for condition, reason in (
                    (bool(document_overlap.loc[index]), "document_also_present_in_evaluation"),
                    (bool(sentence_overlap.loc[index]), "exact_normalized_sentence_occurs_in_evaluation"),
                )
                if condition
            )
            for index in memory_exclusions.index
        ]
    memory = frame.loc[(frame["split"] == "train") & ~exclusion_mask].copy()
    if set(memory["document_id"]) & evaluation_documents:
        raise ValueError("PcMSP memory still overlaps evaluation documents after exclusions")
    query_frames: dict[str, pd.DataFrame] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        selected = frame.loc[frame["split"] == split]
        queries: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        for row in selected.itertuples(index=False):
            query_id = f"pcmsp:{split}:{row.document_id}:{row.sentence_index:04d}"
            queries.append(
                {
                    "query_id": query_id,
                    "task_family": PCMSP_TASK,
                    "document_id": row.document_id,
                    "sentence_index": int(row.sentence_index),
                    "input_json": row.input_json,
                    "input_text": row.input_text,
                    "context_json": row.context_json,
                    "protocol": row.protocol,
                    "input_signature": row.input_signature,
                    "source_dataset": row.source_dataset,
                }
            )
            labels.append({"query_id": query_id, "action_graph_json": row.target_json})
        query_frames[split] = pd.DataFrame(queries)
        label_frames[split] = pd.DataFrame(labels)
    membership = frame[
        [
            "source_record_id", "document_id", "sentence_index", "text_signature",
            "node_count", "edge_count", "leakage_group", "split",
        ]
    ].copy()
    written = _standard_tables(
        task_dir,
        memory,
        query_frames["validation"],
        query_frames["test"],
        label_frames["validation"],
        label_frames["test"],
        membership,
    )
    written.extend(_extra_frame_files(task_dir, "audit", "memory_exclusions", memory_exclusions))
    return _finalize_task(
        task_dir,
        {
            "dataset": "PcMSP",
            "task": PCMSP_TASK,
            "source": {
                "url": PCMSP_URL,
                "revision": source_revision,
                "directory": str(root),
                "directory_sha256": source_sha,
            },
            "specialized_models": ["MatBERT", "PURE joint entity-relation extraction"],
            "split_protocol": "released document directories define validation/test; train sentences from documents also present in evaluation, plus exact normalized evaluation sentences, are removed from memory",
            "counts": {
                "source_files": document_count,
                "unique_documents": int(frame["document_id"].nunique()),
                "documents_crossing_released_directories": int(
                    (frame.groupby("document_id")["split"].nunique() > 1).sum()
                ),
                "sentences_by_split": _split_counts(frame),
                "entities": int(frame["node_count"].sum()),
                "relations": int(frame["edge_count"].sum()),
                "memory_episodes": len(memory),
                "memory_exclusions": len(memory_exclusions),
                "memory_document_overlap_exclusions": int(document_overlap.sum()),
                "memory_exact_sentence_overlap_exclusions": int(sentence_overlap.sum()),
                "validation_queries": len(query_frames["validation"]),
                "test_queries": len(query_frames["test"]),
            },
            "metrics": [
                "exact-span entity micro-F1",
                "relation micro-F1",
                "typed-relation micro-F1",
                "graph edge micro-F1",
            ],
            "target_semantics": "canonical sentence-level typed nodes and directed edges; an empty graph is a valid target",
            "scope_limit": "this is an intra-sentence graph, not a reconstructed full-document procedure graph",
        },
        written,
    )


def load_feconi_rows(
    imputed_path: Path, raw_path: Path, source_revision: str
) -> pd.DataFrame:
    imputed = _drop_index_columns(pd.read_csv(imputed_path))
    raw = _drop_index_columns(pd.read_csv(raw_path, encoding="utf-8-sig"))
    required = set(FECONI_ELEMENTS) | set(FECONI_PROPERTIES)
    for name, frame in (("imputed", imputed), ("raw", raw)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"FeCoNi {name} source is missing columns: {sorted(missing)}")
    if len(imputed) != len(raw):
        raise ValueError("FeCoNi raw and imputed tables have different row counts")
    if imputed[list(FECONI_PROPERTIES)].isna().any().any():
        raise ValueError("FeCoNi selected imputed properties still contain missing values")
    imputed_sha = file_sha256(imputed_path)
    raw_sha = file_sha256(raw_path)
    rows: list[dict[str, Any]] = []
    for source_row, row in imputed.iterrows():
        fractions = _fraction_dict({element: float(row[element]) for element in FECONI_ELEMENTS})
        properties = {name: float(row[name]) for name in FECONI_PROPERTIES}
        input_value = {
            "target_properties": properties,
            "objective": "match all target properties",
        }
        target_value = {
            "composition_basis": "weight_fraction",
            "element_fractions": fractions,
        }
        observed = {
            name: bool(pd.notna(raw.loc[source_row, name])) for name in FECONI_PROPERTIES
        }
        source_id = f"feconi-row-{source_row:04d}"
        record = _base_episode(
            episode_id=f"feconi:{source_id}:inverse_design",
            task_family=FECONI_TASK,
            source_record_id=source_id,
            source_group="composition:" + _stable_hash(fractions)[:20],
            input_value=input_value,
            input_text=(
                "Design an alloy composition matching this Fe-Co-Ni property target vector: "
                + ", ".join(f"{key}={value:g}" for key, value in properties.items())
            ),
            target_value=target_value,
            context_value={"allowed_elements": list(FECONI_ELEMENTS), "composition_basis": "weight_fraction"},
            protocol="published imputation plus multi-property forward surrogate and Bayesian optimization",
            source_dataset="Fe-Co-Ni multi-property inverse-design release",
            source_revision=source_revision,
            source_url=FECONI_URL,
            source_sha256=_stable_hash({"imputed": imputed_sha, "raw": raw_sha}),
            epistemic_status="mixed_experimental_imputed_input",
            quality_value={
                "source_row": int(source_row),
                "property_observed_not_imputed": observed,
                "imputed_property_count": int(sum(not value for value in observed.values())),
            },
        )
        record.update(
            {
                "composition_signature": _stable_hash(fractions),
                "property_signature": _stable_hash(properties),
                "observed_property_count": int(sum(observed.values())),
            }
        )
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame["leakage_group"] = connect_source_and_input_groups(frame)
    return frame


def build_feconi(
    imputed_path: Path,
    raw_path: Path,
    task_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame = load_feconi_rows(imputed_path, raw_path, source_revision)
    frame["split"], split_metadata = assign_group_splits(
        frame, (), fractions=(0.70, 0.15, 0.15), seed=20261023, candidates=256
    )
    _check_disjoint(frame, "composition_signature")
    _check_disjoint(frame, "property_signature")
    memory = frame.loc[frame["split"] == "train"].copy()
    query_frames: dict[str, pd.DataFrame] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        queries: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        selected = frame.loc[frame["split"] == split]
        for ordinal, (_, group) in enumerate(selected.groupby("property_signature", sort=True)):
            first = group.iloc[0]
            query_id = f"feconi:{split}:{ordinal:05d}"
            queries.append(_query_from_row(first, query_id))
            compositions = sorted(set(group["target_json"]))
            labels.append(
                {
                    "query_id": query_id,
                    "candidate_compositions_json": canonical_json([json.loads(value) for value in compositions]),
                    "source_record_ids_json": canonical_json(sorted(group["source_record_id"].astype(str))),
                }
            )
        query_frames[split] = pd.DataFrame(queries)
        label_frames[split] = pd.DataFrame(labels)
    membership = frame[
        [
            "source_record_id", "composition_signature", "property_signature",
            "observed_property_count", "leakage_group", "split",
        ]
    ].copy()
    written = _standard_tables(
        task_dir,
        memory,
        query_frames["validation"],
        query_frames["test"],
        label_frames["validation"],
        label_frames["test"],
        membership,
    )
    return _finalize_task(
        task_dir,
        {
            "dataset": "Fe-Co-Ni multi-property inverse-design release",
            "task": FECONI_TASK,
            "source": {
                "url": FECONI_URL,
                "revision": source_revision,
                "imputed_path": str(imputed_path),
                "imputed_sha256": file_sha256(imputed_path),
                "raw_path": str(raw_path),
                "raw_sha256": file_sha256(raw_path),
            },
            "specialized_model": "ExtraTrees/neural multi-property forward surrogate plus Bayesian optimization",
            "split_protocol": split_metadata,
            "counts": {
                "source_rows": len(frame),
                "unique_compositions": int(frame["composition_signature"].nunique()),
                "unique_property_vectors": int(frame["property_signature"].nunique()),
                "rows_by_split": _split_counts(frame),
                "memory_episodes": len(memory),
                "validation_queries": len(query_frames["validation"]),
                "test_queries": len(query_frames["test"]),
            },
            "metrics": [
                "forward-surrogate target error",
                "top-k target-tolerance success",
                "composition normalization validity",
                "allowed-element validity",
            ],
            "target_semantics": "one-to-many constrained inverse design; exact match to one recorded composition is secondary only",
            "quality_limit": "many input properties are author-imputed; per-property observed/imputed flags are retained in every memory episode",
            "published_result_comparability": "requires retraining the surrogate and optimizer on this group-disjoint split",
        },
        written,
    )


def load_hea_rows(
    path: Path, source_revision: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _drop_index_columns(pd.read_csv(path))
    required = {"index", "Phase"} | set(HEA_ELEMENTS)
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"HEA source is missing columns: {sorted(missing)}")
    source_sha = file_sha256(path)
    rows: list[dict[str, Any]] = []
    composition_discrepancies: list[dict[str, Any]] = []
    for source_row, row in raw.iterrows():
        formula = str(row["index"]).strip()
        fractions = _fraction_dict(
            {
                str(element): float(amount)
                for element, amount in Composition(formula).get_el_amt_dict().items()
            }
        )
        source_values = {
            element: float(row[element])
            for element in HEA_ELEMENTS
            if math.isfinite(float(row[element])) and float(row[element]) > 1e-12
        }
        source_fractions = _fraction_dict(source_values) if source_values else {}
        all_elements = set(fractions) | set(source_fractions)
        max_fraction_delta = max(
            (
                abs(fractions.get(element, 0.0) - source_fractions.get(element, 0.0))
                for element in all_elements
            ),
            default=0.0,
        )
        if max_fraction_delta > 1e-6:
            composition_discrepancies.append(
                {
                    "source_row": int(source_row),
                    "reported_formula": formula,
                    "parsed_formula_fractions_json": canonical_json(fractions),
                    "source_element_column_fractions_json": canonical_json(source_fractions),
                    "elements_absent_from_source_columns_json": canonical_json(
                        sorted(set(fractions) - set(HEA_ELEMENTS))
                    ),
                    "max_abs_fraction_delta": max_fraction_delta,
                    "resolution": "use_reported_formula_parsed_by_pymatgen",
                }
            )
        chemical_system = "-".join(sorted(fractions))
        phases = sorted({value.strip() for value in str(row["Phase"]).split("+") if value.strip()})
        if not phases:
            raise ValueError(f"HEA row {source_row} has no phase label")
        input_value = {
            "reported_formula": formula,
            "element_fractions": fractions,
        }
        target_value = {"phase_set": phases, "reported_phase_label": str(row["Phase"])}
        source_id = f"hea-row-{source_row:05d}"
        record = _base_episode(
            episode_id=f"hea:{source_id}:phase_assemblage",
            task_family=HEA_TASK,
            source_record_id=source_id,
            source_group="composition:" + _stable_hash(fractions)[:20],
            input_value=input_value,
            input_text=f"Predict the phase assemblage for composition {formula}.",
            target_value=target_value,
            context_value={"processing_context_available": False},
            protocol="literature-compiled experimental HEA phase label",
            source_dataset="ML_HEAs_Phase_Dataset first-stage-cleaned table",
            source_revision=source_revision,
            source_url=HEA_URL,
            source_sha256=source_sha,
            epistemic_status="experimental_literature_compilation",
            quality_value={"source_row": int(source_row), "process_variables_missing": True},
        )
        record.update(
            {
                "formula": formula,
                "chemical_system": chemical_system,
                "phase_set_json": canonical_json(phases),
                "composition_signature": _stable_hash(fractions),
                "leakage_group": "composition:" + _stable_hash(fractions)[:20],
            }
        )
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(composition_discrepancies)


def build_hea(path: Path, task_dir: Path, source_revision: str) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame, composition_discrepancies = load_hea_rows(path, source_revision)
    phase_labels = sorted(frame["phase_set_json"].unique())
    for ordinal, label in enumerate(phase_labels):
        frame[f"_phase_{ordinal}"] = (frame["phase_set_json"] == label).astype(int)
    frame["split"], split_metadata = assign_group_splits(
        frame,
        [column for column in frame if column.startswith("_phase_")],
        fractions=(0.70, 0.15, 0.15),
        seed=20261023,
        candidates=256,
    )
    frame = frame.drop(columns=[column for column in frame if column.startswith("_phase_")])
    _check_disjoint(frame, "composition_signature")
    memory = frame.loc[frame["split"] == "train"].copy()
    train_systems = set(memory["chemical_system"])
    query_frames: dict[str, pd.DataFrame] = {}
    label_frames: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        queries: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        selected = frame.loc[frame["split"] == split]
        for ordinal, (_, group) in enumerate(selected.groupby("composition_signature", sort=True)):
            first = group.iloc[0]
            query_id = f"hea:{split}:{ordinal:05d}"
            query = _query_from_row(first, query_id)
            query["chemical_system"] = first["chemical_system"]
            query["challenge_new_chemical_system"] = first["chemical_system"] not in train_systems
            queries.append(query)
            counts = Counter(group["phase_set_json"])
            allowed = [json.loads(value) for value in sorted(counts)]
            labels.append(
                {
                    "query_id": query_id,
                    "allowed_phase_sets_json": canonical_json(allowed),
                    "phase_set_counts_json": canonical_json(
                        {canonical_json(json.loads(key)): int(value) for key, value in sorted(counts.items())}
                    ),
                    "source_record_ids_json": canonical_json(sorted(group["source_record_id"].astype(str))),
                }
            )
        query_frames[split] = pd.DataFrame(queries)
        label_frames[split] = pd.DataFrame(labels)
    conflict_counts = frame.groupby("composition_signature")["phase_set_json"].nunique()
    membership = frame[
        [
            "source_record_id", "formula", "chemical_system", "phase_set_json",
            "composition_signature", "leakage_group", "split",
        ]
    ].copy()
    membership["composition_has_multiple_phase_sets"] = membership["composition_signature"].map(conflict_counts).gt(1)
    written = _standard_tables(
        task_dir,
        memory,
        query_frames["validation"],
        query_frames["test"],
        label_frames["validation"],
        label_frames["test"],
        membership,
    )
    written.extend(
        _extra_frame_files(
            task_dir,
            "audit",
            "source_composition_discrepancies",
            composition_discrepancies,
        )
    )
    return _finalize_task(
        task_dir,
        {
            "dataset": "ML_HEAs_Phase_Dataset dataset10387_70.csv",
            "task": HEA_TASK,
            "source": {
                "url": HEA_URL,
                "revision": source_revision,
                "path": str(path),
                "sha256": file_sha256(path),
            },
            "specialized_models": ["XGBoost classifier", "random-forest classifier"],
            "split_protocol": split_metadata,
            "counts": {
                "source_rows": len(frame),
                "source_composition_discrepancies": len(composition_discrepancies),
                "source_columns_omitted_elements": sorted(
                    {
                        element
                        for value in composition_discrepancies.get(
                            "elements_absent_from_source_columns_json",
                            pd.Series(dtype=str),
                        )
                        for element in json.loads(value)
                    }
                ),
                "unique_compositions": int(frame["composition_signature"].nunique()),
                "phase_set_classes": len(phase_labels),
                "conflicting_compositions": int((conflict_counts > 1).sum()),
                "rows_by_split": _split_counts(frame),
                "memory_episodes": len(memory),
                "validation_queries": len(query_frames["validation"]),
                "test_queries": len(query_frames["test"]),
            },
            "metrics": ["macro-F1", "balanced accuracy", "exact-set accuracy", "set Jaccard"],
            "target_semantics": "all reported phase sets for one exact composition are accepted; phase strings are canonicalized as sets",
            "composition_semantics": "reported formula parsed by pymatgen; the released element-fraction columns omit Au, Bi, Pb, Pt, and U in some rows, recorded under audit/source_composition_discrepancies",
            "quality_limit": "processing variables are absent, so conflicting phase labels are not forced into one deterministic target",
            "published_result_comparability": "the paper's final table is absent; specialized baselines must be retrained on this transparent split",
        },
        written,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not (args.base_release / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing base release manifest: {args.base_release / 'manifest.json'}")
    staging = output.with_name(f"{output.name}.building-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {staging}")
    shutil.copytree(args.base_release, staging)

    task_manifests: dict[str, dict[str, Any]] = {}
    task_manifests["sg_predict_cubic_space_group"] = build_sg_predict(
        args.sg_predict_csv,
        staging / "composition_to_crystallographic_class" / "sg_predict_cubic_space_group",
        args.sg_predict_revision,
    )
    task_manifests["pcmsp"] = build_pcmsp(
        args.pcmsp_original,
        staging / "synthesis_text_to_action_graph" / "pcmsp",
        args.pcmsp_revision,
    )
    task_manifests["feconi_multi_property"] = build_feconi(
        args.feconi_imputed_csv,
        args.feconi_raw_csv,
        staging / "target_property_to_formulation_composition" / "feconi_multi_property",
        args.feconi_revision,
    )
    task_manifests["hea_phase"] = build_hea(
        args.hea_csv,
        staging / "composition_to_phase_assemblage" / "hea_phase",
        args.hea_revision,
    )

    excluded = {
        "pxrd_peaklist_to_symmetry": {
            "decision": "excluded",
            "reason": "the original Suzuki train/test Dropbox URLs now return HTML rather than CSV; SIMPOD reconstruction requires a new peak-finding protocol",
            "original_feature_urls": [
                "https://www.dropbox.com/s/8xjyxub9wavni4v/train.csv?dl=1",
                "https://www.dropbox.com/s/rl0755n11jum3r3/test.csv?dl=1",
            ],
            "reconstruction_source": "https://doi.org/10.57760/sciencedb.09755",
            "reconstruction_source_sizes_bytes": {
                "Structures_Modulo_Merged_2Folds.csv": 20492211232,
                "Structures_Test.csv": 5939791412,
            },
        },
        "cosy_full_composition_to_symmetry": {
            "decision": "not_used",
            "reason": "the 2.24 GB software archive was too slow to obtain and had not been shown to contain row-level training pairs; the release uses the inspectable SG_predict cubic subset instead",
            "zenodo_record": "https://doi.org/10.5281/zenodo.10152991",
        },
        "molten_salt_inverse_density": {
            "decision": "not_used",
            "reason": "the row-level source is distributed as a pickle; Fe-Co-Ni supplies a transparent CSV inverse-design task without deserializing executable objects",
        },
    }
    _write_json(staging / "excluded_tasks.json", excluded)

    base_manifest = _read_manifest(args.base_release / "manifest.json")
    base_registry_path = args.base_release / "source_registry.json"
    base_registry = _read_manifest(base_registry_path) if base_registry_path.is_file() else {}
    registry = {
        "release_id": RELEASE_ID,
        "base_release_id": base_manifest.get("release_id"),
        "inherited_task_families": base_registry.get("task_families", {}),
        "task_families_added": {
            "composition_to_crystallographic_class": ["sg_predict_cubic_space_group"],
            "synthesis_text_to_action_graph": ["pcmsp"],
            "target_property_to_formulation_composition": ["feconi_multi_property"],
            "composition_to_phase_assemblage": ["hea_phase"],
        },
        "excluded_or_deferred": excluded,
    }
    _write_json(staging / "source_registry.json", registry)
    _write_json(
        staging / "audit_summary.json",
        {
            "release_id": RELEASE_ID,
            "status": "candidate",
            "added_task_counts": {
                name: task.get("counts", {}) for name, task in task_manifests.items()
            },
            "excluded_tasks": excluded,
            "checks": {
                "training_only_memory": True,
                "public_evaluation_labels_absent": True,
                "group_disjoint_added_tasks": True,
                "source_hashes_recorded": True,
            },
        },
    )
    readme = """# MatRecall unified specialized-task release

This candidate release copies the base release without modifying it and adds four task
families backed by public specialized-model datasets.  Every leaf contains
training-only `public/memory_episodes`, label-free public validation/test
queries, evaluator-only labels under `private`, split membership under `audit`,
and a task manifest.

## Added families

- `composition_to_crystallographic_class/sg_predict_cubic_space_group`
- `synthesis_text_to_action_graph/pcmsp`
- `target_property_to_formulation_composition/feconi_multi_property`
- `composition_to_phase_assemblage/hea_phase`

The SG_predict task supplies the cubic crystal-system context; it is not a
seven-system composition-only benchmark.  Fe-Co-Ni and HEA have one-to-many
targets and must not be evaluated by forcing one arbitrary exact answer.
`pxrd_peaklist_to_symmetry` is excluded because its original feature downloads
are unavailable and SIMPOD would require a new reconstruction protocol.  Read
`excluded_tasks.json` and each task manifest before evaluation.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "release_id": RELEASE_ID,
        "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "base release plus four specialized-model input/output datasets",
        "base_release": {
            "path": str(args.base_release),
            "release_id": base_manifest.get("release_id"),
            "manifest_sha256": file_sha256(args.base_release / "manifest.json"),
        },
        "added_tasks": {
            name: {
                key: value
                for key, value in task.items()
                if key in {"dataset", "task", "counts", "split_protocol", "scope_limit", "quality_limit"}
            }
            for name, task in task_manifests.items()
        },
        "excluded_tasks": excluded,
        "visibility_policy": {
            "memory": "training rows only",
            "development": "public validation queries with private labels",
            "final": "public test queries with private labels after system settings are frozen",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "safety": "no pickle, joblib, torch checkpoint, or object-array NPZ was deserialized",
    }
    _write_json(staging / "manifest.json", manifest)
    os.replace(staging, output)
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--sg-predict-csv", type=Path, required=True)
    parser.add_argument("--sg-predict-revision", required=True)
    parser.add_argument("--pcmsp-original", type=Path, required=True)
    parser.add_argument("--pcmsp-revision", required=True)
    parser.add_argument("--feconi-imputed-csv", type=Path, required=True)
    parser.add_argument("--feconi-raw-csv", type=Path, required=True)
    parser.add_argument("--feconi-revision", required=True)
    parser.add_argument("--hea-csv", type=Path, required=True)
    parser.add_argument("--hea-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    manifest = build_release(make_parser().parse_args())
    print(json.dumps(manifest["added_tasks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
