"""Build the base unified specialized-model task release.

The release places the existing crystal-to-property and
process-to-outcome/property releases beside four inverse-synthesis datasets:
SynthesisSimilarity, SyntMTE, SPENDE, and DiffSyn.  Public validation/test
tables contain inputs only.  Their labels are written below ``private``.

Downloaded pickle, joblib, and object-array NPZ files are deliberately not
deserialized by this builder.  Every new adapter reads JSON, CSV, or XLSX.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from matrecall.build_crystal_reference import file_sha256
from matrecall.build_process_tasks import (
    assign_group_splits,
    canonical_json,
    connect_source_and_input_groups,
)


RELEASE_ID = "matrecall-unified-tasks-base"
TASK_FAMILY = "target_process_to_precursor_steps_conditions"
SPLITS = ("train", "validation", "test")
DIFFSYN_RATIO_FIELDS = (
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
)
DIFFSYN_RATIO_QUANTILES = {
    "Si/Al": 0.98,
    "Al/P": 0.99,
    "Si/Ge": 0.995,
    "Si/B": 0.99,
    "Na/T": 0.95,
    "K/T": 0.99,
    "OH/T": 0.97,
    "F/T": 0.95,
    "H2O/T": 0.99,
    "sda1/T": 0.98,
}
DIFFSYN_HETEROATOM_FIELDS = ("Si", "Al", "P", "Ge", "B", "V", "Be", "Ga", "Ti")
DIFFSYN_CONDITION_FIELDS = ("cryst_temp", "cryst_time")
DIFFSYN_GEL_FIELDS = (
    "Si", "Al", "P", "Na", "K", "Li", "Sr", "Rb", "Cs", "Ba",
    "Ca", "F", "Ge", "Ti", "In", "B", "Mg", "Ga", "Ni", "Mn",
    "Fe", "Co", "Cr", "Zn", "Nb", "Be", "W", "Ce", "Cu", "Sn",
    "Gd", "La", "Y", "Dy", "Sm", "Ag", "Cd", "Zr", "V", "H2O",
    "solvent", "sda1", "sda2", "sda3", "acid", "Ta", "Ru", "Hf",
    "Yb", "Tl", "As", "OH",
)
SYNTMTE_TARGETS = (
    ("Sintering Temperature", "sintering_temperature_c"),
    ("Sintering Time", "sintering_time_h"),
    ("Calcination Temperature", "calcination_temperature_c"),
    ("Calcination Time", "calcination_time_h"),
)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(canonical_json(row) + "\n")


def _write_frame(base: Path, name: str, frame: pd.DataFrame) -> list[Path]:
    base.mkdir(parents=True, exist_ok=True)
    parquet = base / f"{name}.parquet"
    jsonl = base / f"{name}.jsonl"
    frame.to_parquet(parquet, index=False)
    _write_jsonl(jsonl, frame)
    return [parquet, jsonl]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_record(path: Path, root: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list literal, got {value!r}")
    return parsed


def _formula_from_material(material: dict[str, Any]) -> str | None:
    formula = material.get("material_formula")
    if formula:
        return str(formula).strip()
    for composition in material.get("composition") or []:
        formula = composition.get("formula")
        if formula:
            return str(formula).strip()
    return None


def _new_task_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty task directory: {path}")
    (path / "public").mkdir(parents=True, exist_ok=True)
    (path / "private").mkdir(parents=True, exist_ok=True)
    (path / "audit").mkdir(parents=True, exist_ok=True)


def _finalize_task(
    task_dir: Path,
    manifest: dict[str, Any],
    written: Iterable[tuple[Path, int | None]],
) -> dict[str, Any]:
    manifest["files"] = [
        _file_record(path, task_dir, rows) for path, rows in written
    ]
    _write_json(task_dir / "manifest.json", manifest)
    return manifest


def _standard_tables(
    task_dir: Path,
    memory: pd.DataFrame,
    validation_queries: pd.DataFrame,
    test_queries: pd.DataFrame,
    validation_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    membership: pd.DataFrame,
) -> list[tuple[Path, int | None]]:
    outputs: list[tuple[Path, int | None]] = []
    for base, name, frame in (
        (task_dir / "public", "memory_episodes", memory),
        (task_dir / "public", "validation_queries", validation_queries),
        (task_dir / "public", "test_queries", test_queries),
        (task_dir / "private", "validation_labels", validation_labels),
        (task_dir / "private", "test_labels", test_labels),
        (task_dir / "audit", "split_membership", membership),
    ):
        outputs.extend((path, len(frame)) for path in _write_frame(base, name, frame))
    return outputs


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def load_synthesis_similarity_rows(path: Path) -> pd.DataFrame:
    """Read the public JSONL without loading the released object-array NPZ."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_index, line in enumerate(handle):
            record = json.loads(line)
            precursor_formulas = sorted(
                {
                    formula
                    for material in record.get("precursors") or []
                    if (formula := _formula_from_material(material))
                }
            )
            targets = [str(value).strip() for value in record.get("targets_string") or [] if str(value).strip()]
            if not targets:
                fallback = _formula_from_material(record.get("target") or {})
                targets = [fallback] if fallback else []
            if not (1 <= len(precursor_formulas) <= 5):
                continue
            year = pd.to_numeric(record.get("year"), errors="coerce")
            doi = str(record.get("doi") or "").strip()
            if not targets or pd.isna(year) or not doi:
                continue
            source_id = str(record.get("_id") or raw_index)
            for target_index, target_formula in enumerate(dict.fromkeys(targets)):
                rows.append(
                    {
                        "source_record_id": source_id,
                        "source_raw_index": raw_index,
                        "target_index": target_index,
                        "doi": doi,
                        "year": int(year),
                        "synthesis_type": record.get("synthesis_type"),
                        "target_formula": target_formula,
                        "precursor_formulas_json": canonical_json(precursor_formulas),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No eligible SynthesisSimilarity rows in {path}")
    return frame


def assign_synthesis_similarity_splits(frame: pd.DataFrame) -> pd.Series:
    """Chronological 85/5/10 split with DOI and target groups kept intact."""
    union = _UnionFind()
    row_nodes: list[str] = []
    for row in frame.itertuples(index=False):
        row_node = f"row:{row.source_raw_index}"
        union.union(row_node, f"doi:{row.doi.lower()}")
        union.union(row_node, f"target:{row.target_formula}")
        row_nodes.append(row_node)
    components = pd.Series([union.find(node) for node in row_nodes], index=frame.index)
    anchor_year = frame.assign(_component=components).groupby("_component")["year"].min()
    unique_years = sorted(anchor_year.unique())
    if len(unique_years) < 3:
        raise ValueError("Need at least three years for chronological splitting")
    train_end = int(len(unique_years) * 0.85)
    validation_end = int(len(unique_years) * 0.90)
    train_years = set(unique_years[:train_end])
    validation_years = set(unique_years[train_end:validation_end])
    mapping = {
        component: (
            "train"
            if year in train_years
            else "validation"
            if year in validation_years
            else "test"
        )
        for component, year in anchor_year.items()
    }
    return components.map(mapping)


def build_synthesis_similarity(
    source: Path, task_dir: Path, source_revision: str
) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame = load_synthesis_similarity_rows(source)
    frame["split"] = assign_synthesis_similarity_splits(frame)
    frame["record_id"] = frame.apply(
        lambda row: f"synthesis_similarity:{row.source_record_id}:{row.target_index}",
        axis=1,
    )
    frame["input_signature"] = frame["target_formula"].map(_stable_hash)
    frame["query_id"] = frame["record_id"]
    frame["task"] = "target_composition_to_precursor_set"
    frame["task_family"] = TASK_FAMILY
    frame["input_json"] = frame.apply(
        lambda row: canonical_json(
            {
                "target_formula": row.target_formula,
                "synthesis_type": row.synthesis_type,
            }
        ),
        axis=1,
    )
    frame["target_json"] = frame["precursor_formulas_json"].map(
        lambda value: canonical_json({"precursor_formulas": json.loads(value)})
    )
    common = [
        "query_id", "task", "task_family", "input_json", "target_formula",
        "synthesis_type", "doi", "year", "source_record_id",
    ]
    query_columns = common
    memory = frame.loc[frame.split == "train", common + ["target_json", "precursor_formulas_json"]].copy()
    memory = memory.rename(columns={"query_id": "episode_id"})
    val = frame.loc[frame.split == "validation", query_columns].copy()
    test = frame.loc[frame.split == "test", query_columns].copy()
    val_labels = frame.loc[frame.split == "validation", ["query_id", "target_json", "precursor_formulas_json"]].copy()
    test_labels = frame.loc[frame.split == "test", ["query_id", "target_json", "precursor_formulas_json"]].copy()
    membership = frame[
        ["record_id", "split", "source_record_id", "source_raw_index", "target_index", "doi", "year", "target_formula", "input_signature"]
    ].copy()
    written = _standard_tables(task_dir, memory, val, test, val_labels, test_labels, membership)
    counts = frame["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict()
    overlaps = {
        "doi": max(
            len(set(frame.loc[frame.split == left, "doi"]) & set(frame.loc[frame.split == right, "doi"]))
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ),
        "target_formula": max(
            len(set(frame.loc[frame.split == left, "target_formula"]) & set(frame.loc[frame.split == right, "target_formula"]))
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"SynthesisSimilarity group leakage: {overlaps}")
    return _finalize_task(
        task_dir,
        {
            "dataset": "SynthesisSimilarity",
            "task": "target_composition_to_precursor_set",
            "source_revision": source_revision,
            "source_file": str(source),
            "source_sha256": file_sha256(source),
            "split_protocol": "rebuilt_chronological_85_5_10_grouped_by_doi_and_target",
            "official_object_npz_loaded": False,
            "reason_for_rebuild": "released split NPZ stores Python object arrays; this builder only reads JSONL",
            "released_object_npz_header_counts": {
                "train": 44736,
                "validation": 2254,
                "test": 2934,
            },
            "membership_difference": "This release expands the public raw JSONL with transparent formula/count filters and does not reproduce the upstream oxidation-state and prototype preprocessing; its membership differs from the released object NPZ",
            "paper_pretrained_models_directly_comparable_to_main_split": False,
            "baseline_requirement": "retrain SynthesisSimilarity or a simpler model on the released training membership",
            "counts": counts,
            "cross_split_overlap_counts": overlaps,
            "metrics": ["precursor set exact match", "precursor precision/recall/F1", "top-k set recall"],
            "redistribution_status": "upstream repository has no declared license file; review before redistribution",
        },
        written,
    )


def load_syntmte_split(split_dir: Path) -> pd.DataFrame:
    frames = []
    for split, filename in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        current = pd.read_csv(split_dir / filename)
        current.insert(0, "official_split", split)
        current.insert(1, "source_row", np.arange(len(current)))
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def build_syntmte(split_dir: Path, task_dir: Path, source_revision: str) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame = load_syntmte_split(split_dir)
    frame["precursor_formulas_json"] = frame["precursor_formulas"].map(
        lambda value: canonical_json(sorted(map(str, _parse_list(value))))
    )
    frame["input_json"] = frame.apply(
        lambda row: canonical_json(
            {
                "target_formula": row.target_formula,
                "precursor_formulas": json.loads(row.precursor_formulas_json),
            }
        ), axis=1,
    )
    frame["input_signature"] = frame["input_json"].map(_stable_hash)
    frame["source_doi"] = frame["DOI"].astype(str).str.strip().str.lower()
    heldout_signatures = set(frame.loc[frame.official_split != "train", "input_signature"])
    heldout_dois = set(frame.loc[frame.official_split != "train", "source_doi"])
    frame["memory_exclusion_reason"] = None
    train_mask = frame.official_split == "train"
    frame.loc[train_mask & frame.input_signature.isin(heldout_signatures), "memory_exclusion_reason"] = "exact_input_in_validation_or_test"
    frame.loc[train_mask & frame.source_doi.isin(heldout_dois), "memory_exclusion_reason"] = "doi_in_validation_or_test"
    frame["record_id"] = frame.apply(
        lambda row: f"syntmte:{row.official_split}:{row.source_row}", axis=1
    )
    frame["query_id"] = frame["record_id"]
    frame["task"] = "target_and_precursors_to_heating_conditions"
    frame["task_family"] = TASK_FAMILY
    for source_col, output_col in SYNTMTE_TARGETS:
        frame[output_col] = pd.to_numeric(frame[source_col], errors="coerce")
    output_columns = [output for _, output in SYNTMTE_TARGETS]
    frame["target_json"] = frame.apply(
        lambda row: canonical_json({column: row[column] for column in output_columns}),
        axis=1,
    )
    common = [
        "query_id", "task", "task_family", "input_json", "target_formula",
        "precursor_formulas_json", "source_doi", "Year", "source_row",
    ]
    memory_mask = train_mask & frame.memory_exclusion_reason.isna()
    memory = frame.loc[memory_mask, common + ["target_json", *output_columns]].copy().rename(columns={"query_id": "episode_id"})
    val = frame.loc[frame.official_split == "validation", common].copy()
    test = frame.loc[frame.official_split == "test", common].copy()
    val_labels = frame.loc[frame.official_split == "validation", ["query_id", "target_json", *output_columns]].copy()
    test_labels = frame.loc[frame.official_split == "test", ["query_id", "target_json", *output_columns]].copy()
    membership = frame[
        ["record_id", "official_split", "source_row", "source_doi", "Year", "input_signature", "memory_exclusion_reason"]
    ].copy()
    written = _standard_tables(task_dir, memory, val, test, val_labels, test_labels, membership)
    source_files = [split_dir / name for name in ("train.csv", "val.csv", "test.csv")]
    return _finalize_task(
        task_dir,
        {
            "dataset": "SyntMTE",
            "task": "target_and_precursors_to_heating_conditions",
            "source_revision": source_revision,
            "source_files": [{"path": str(path), "sha256": file_sha256(path)} for path in source_files],
            "split_protocol": "upstream_year_split",
            "official_counts": frame["official_split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict(),
            "memory_rows_after_heldout_input_and_doi_exclusion": int(memory_mask.sum()),
            "memory_exclusions": frame["memory_exclusion_reason"].value_counts(dropna=True).astype(int).to_dict(),
            "targets": output_columns,
            "primary_upstream_trained_heads": ["sintering_temperature_c", "calcination_temperature_c"],
            "exploratory_heads": ["sintering_time_h", "calcination_time_h"],
            "metrics": "per-head MAE and RMSE in source units; do not average raw errors across units",
            "redistribution_status": "review upstream data and repository terms before redistribution",
        },
        written,
    )


def _spende_formulas(materials: list[dict[str, Any]]) -> list[str]:
    formulas: list[str] = []
    for material in materials:
        for composition in material.get("composition") or []:
            if composition.get("formula"):
                formulas.append(str(composition["formula"]))
    return formulas


def load_spende_rows(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for source_row, record in enumerate(data):
        target_formulas = [
            str(composition["formula"])
            for composition in (record.get("target") or {}).get("composition") or []
            if composition.get("formula")
        ]
        precursor_formulas = _spende_formulas(record.get("precursors") or [])
        operations = [
            {
                "name": operation.get("string"),
                "type": operation.get("type"),
                "conditions": _clean(operation.get("conditions")),
            }
            for operation in record.get("operations") or []
        ]
        if not target_formulas or not precursor_formulas or not operations:
            continue
        input_value = {
            "target_formulas": target_formulas,
            "precursor_formulas": precursor_formulas,
        }
        rows.append(
            {
                "source_row": source_row,
                "source_doi": str(record.get("doi") or f"missing-doi:{source_row}").strip().lower(),
                "target_formulas_json": canonical_json(target_formulas),
                "precursor_formulas_json": canonical_json(precursor_formulas),
                "input_json": canonical_json(input_value),
                "input_signature": _stable_hash(input_value),
                "operation_sequence_json": canonical_json([operation["type"] for operation in operations]),
                "operation_steps_json": canonical_json(operations),
                "material_property_json": canonical_json(record.get("material property") or {}),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No eligible SPENDE rows in {path}")
    return frame


def build_spende(source: Path, task_dir: Path, source_revision: str) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame = load_spende_rows(source)
    frame["source_group"] = frame["source_doi"]
    frame["leakage_group"] = connect_source_and_input_groups(frame)
    split, split_audit = assign_group_splits(frame, ())
    frame["split"] = split
    rng = np.random.RandomState(0)
    fold_arrays = np.array_split(rng.permutation(len(frame)), 5)
    frame["upstream_outer_fold"] = -1
    for fold, indices in enumerate(fold_arrays):
        frame.loc[indices, "upstream_outer_fold"] = fold
    frame["record_id"] = frame["source_row"].map(lambda value: f"spende:{value}")
    frame["query_id"] = frame["record_id"]
    frame["task"] = "target_and_precursors_to_operations_and_conditions"
    frame["task_family"] = TASK_FAMILY
    frame["target_json"] = frame.apply(
        lambda row: canonical_json(
            {
                "operation_sequence": json.loads(row.operation_sequence_json),
                "operation_steps": json.loads(row.operation_steps_json),
            }
        ), axis=1,
    )
    common = [
        "query_id", "task", "task_family", "input_json", "target_formulas_json",
        "precursor_formulas_json", "source_doi", "source_row",
    ]
    memory = frame.loc[frame.split == "train", common + ["target_json", "operation_sequence_json", "operation_steps_json"]].copy().rename(columns={"query_id": "episode_id"})
    val = frame.loc[frame.split == "validation", common].copy()
    test = frame.loc[frame.split == "test", common].copy()
    val_labels = frame.loc[frame.split == "validation", ["query_id", "target_json", "operation_sequence_json", "operation_steps_json"]].copy()
    test_labels = frame.loc[frame.split == "test", ["query_id", "target_json", "operation_sequence_json", "operation_steps_json"]].copy()
    membership = frame[
        ["record_id", "split", "source_row", "source_doi", "input_signature", "leakage_group", "upstream_outer_fold"]
    ].copy()
    written = _standard_tables(task_dir, memory, val, test, val_labels, test_labels, membership)
    counts = frame["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict()
    return _finalize_task(
        task_dir,
        {
            "dataset": "SPENDE",
            "task": "target_and_precursors_to_operations_and_conditions",
            "source_revision": source_revision,
            "source_file": str(source),
            "source_sha256": file_sha256(source),
            "split_protocol": "group_disjoint_by_doi_and_exact_input_70_15_15",
            "counts": counts,
            "split_search": split_audit,
            "upstream_protocol": "five row-random folds from numpy seed 0",
            "upstream_fold_assignment_preserved_in_audit": True,
            "paper_pretrained_models_directly_comparable_to_main_split": False,
            "baseline_requirement": "retrain SPENDE or RF/XGBoost on the released training membership",
            "metrics": ["operation-sequence Jaccard", "exact sequence accuracy", "condition-specific MAE/F1"],
            "redistribution_status": "upstream repository has no declared license file; review before redistribution",
        },
        written,
    )


def normalize_zeolite_code(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Dense/Amorphous"
    code = str(value)
    if code.startswith("*"):
        code = code[1:]
    if code.startswith("-"):
        code = code[1:]
    return code


def load_diffsyn_routes(
    workbook: Path,
    osda_descriptors: Path,
    zeolite_descriptors: Path,
    cif_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_excel(workbook, sheet_name="Sheet1")
    counts = {"xlsx_rows": int(len(raw))}
    frame = raw.loc[raw["doi"].notna()].copy()
    counts["doi_nonmissing_rows"] = int(len(frame))
    frame["T"] = frame[list(DIFFSYN_HETEROATOM_FIELDS)].sum(axis=1)
    for ratio, quantile in DIFFSYN_RATIO_QUANTILES.items():
        numerator, denominator = ratio.split("/")
        values = frame[numerator] / frame[denominator]
        values = values.fillna(0.0)
        finite_nonzero = values.loc[np.isfinite(values) & (values != 0.0)]
        if finite_nonzero.empty:
            raise ValueError(f"Cannot cap DiffSyn ratio without finite values: {ratio}")
        high_value = float(np.quantile(finite_nonzero, quantile))
        values = values.replace([np.inf, -np.inf], high_value)
        frame[ratio] = values.clip(upper=high_value)
    for condition in DIFFSYN_CONDITION_FIELDS:
        frame[f"{condition}_was_missing"] = frame[condition].isna()
    imputer_columns = [*DIFFSYN_RATIO_FIELDS, *DIFFSYN_CONDITION_FIELDS]
    imputer = IterativeImputer(
        min_value=0.0,
        sample_posterior=True,
        skip_complete=True,
        random_state=0,
    )
    imputed = pd.DataFrame(
        imputer.fit_transform(frame[imputer_columns]),
        columns=imputer_columns,
        index=frame.index,
    )
    for condition in DIFFSYN_CONDITION_FIELDS:
        frame[condition] = imputed[condition]
    frame["zeolite_code"] = frame["Code1"].map(normalize_zeolite_code)
    frame["osda_smiles"] = frame["osda1 smiles"].fillna("OSDA-free").astype(str)
    osda_codes = set(pd.read_csv(osda_descriptors)["osda smiles"].astype(str)) | {"OSDA-free"}
    zeolite_codes = set(pd.read_csv(zeolite_descriptors)["Code"].astype(str)) | {"Dense/Amorphous"}
    graph_codes = {path.stem for path in cif_dir.glob("*.cif")} | {"Dense/Amorphous"}
    frame["osda_features_present"] = frame["osda_smiles"].isin(osda_codes)
    frame["zeolite_features_present"] = frame["zeolite_code"].isin(zeolite_codes)
    frame["zeolite_graph_present"] = frame["zeolite_code"].isin(graph_codes)
    counts["missing_osda_features"] = int((~frame.osda_features_present).sum())
    counts["missing_zeolite_features"] = int((~frame.zeolite_features_present).sum())
    counts["missing_zeolite_graph"] = int((~frame.zeolite_graph_present).sum())
    eligible = frame.loc[
        frame.osda_features_present & frame.zeolite_features_present & frame.zeolite_graph_present
    ].copy()
    counts["model_eligible_rows"] = int(len(eligible))
    return eligible, counts


def assign_diffsyn_system_splits(systems: list[list[str]]) -> dict[tuple[str, str], str]:
    tuples = [tuple(map(str, system)) for system in systems]
    train_systems, test_systems = train_test_split(tuples, test_size=0.2, random_state=0)
    train_systems, validation_systems = train_test_split(
        train_systems, test_size=0.125, random_state=0
    )
    result = {system: "train" for system in train_systems}
    result.update({system: "validation" for system in validation_systems})
    result.update({system: "test" for system in test_systems})
    return result


def build_diffsyn(
    workbook: Path,
    system_list_path: Path,
    osda_descriptors: Path,
    zeolite_descriptors: Path,
    cif_dir: Path,
    task_dir: Path,
    source_revision: str,
    split_code_revision: str,
) -> dict[str, Any]:
    _new_task_dir(task_dir)
    frame, filter_counts = load_diffsyn_routes(
        workbook, osda_descriptors, zeolite_descriptors, cif_dir
    )
    if filter_counts != {
        "xlsx_rows": 30164,
        "doi_nonmissing_rows": 23961,
        "missing_osda_features": 1385,
        "missing_zeolite_features": 588,
        "missing_zeolite_graph": 558,
        "model_eligible_rows": 22024,
    }:
        raise ValueError(f"Unexpected DiffSyn filtering counts: {filter_counts}")
    systems = json.loads(system_list_path.read_text(encoding="utf-8"))
    mapping = assign_diffsyn_system_splits(systems)
    frame["system"] = list(zip(frame["zeolite_code"], frame["osda_smiles"]))
    missing = sorted(set(frame["system"]) - set(mapping))
    if missing:
        raise ValueError(f"Eligible DiffSyn systems absent from system list: {missing[:10]}")
    frame["split"] = frame["system"].map(mapping)
    frame["source_record_id"] = frame["Unnamed: 0"].astype(int).astype(str)
    frame["query_id"] = frame.apply(
        lambda row: "diffsyn:" + _stable_hash([row.zeolite_code, row.osda_smiles])[:20], axis=1
    )
    frame["route_id"] = frame["source_record_id"].map(lambda value: f"diffsyn-route:{value}")
    frame["precursors_text"] = frame["precursors"].map(
        lambda value: None if pd.isna(value) else str(value)
    )
    frame["source_doi"] = frame["doi"].map(
        lambda value: None if pd.isna(value) else str(value)
    )
    frame["source_year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["task"] = "zeolite_and_osda_to_synthesis_route_distribution"
    frame["task_family"] = TASK_FAMILY
    frame["input_json"] = frame.apply(
        lambda row: canonical_json({"zeolite_code": row.zeolite_code, "osda_smiles": row.osda_smiles}),
        axis=1,
    )
    output_fields = [*DIFFSYN_RATIO_FIELDS, *DIFFSYN_CONDITION_FIELDS]
    frame["route_json"] = frame.apply(
        lambda row: canonical_json(
            {
                "gel_fractions": {field: row[field] for field in DIFFSYN_GEL_FIELDS},
                "ratios": {field: row[field] for field in DIFFSYN_RATIO_FIELDS},
                "crystallization_temperature_c": row["cryst_temp"],
                "crystallization_time_h": row["cryst_time"],
                "crystallization_temperature_was_imputed": row["cryst_temp_was_missing"],
                "crystallization_time_was_imputed": row["cryst_time_was_missing"],
                "precursors_text": row["precursors_text"],
            }
        ), axis=1,
    )
    route_common = [
        "query_id", "route_id", "task", "task_family", "input_json",
        "zeolite_code", "osda_smiles", "route_json", *output_fields,
        "precursors_text", "source_doi", "source_year", "source_record_id",
        "cryst_temp_was_missing", "cryst_time_was_missing",
    ]
    memory = frame.loc[frame.split == "train", route_common].copy().rename(columns={"route_id": "episode_id"})
    query_columns = ["query_id", "task", "task_family", "input_json", "zeolite_code", "osda_smiles"]
    queries = frame[query_columns + ["split"]].drop_duplicates("query_id")
    val = queries.loc[queries.split == "validation", query_columns].copy()
    test = queries.loc[queries.split == "test", query_columns].copy()
    val_labels = frame.loc[frame.split == "validation", route_common].copy()
    test_labels = frame.loc[frame.split == "test", route_common].copy()
    membership = frame[
        ["route_id", "query_id", "split", "zeolite_code", "osda_smiles", "source_record_id", "source_doi", "source_year"]
    ].copy()
    written = _standard_tables(task_dir, memory, val, test, val_labels, test_labels, membership)
    route_counts = frame["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict()
    system_counts = queries["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict()
    expected_systems = {"train": 1624, "validation": 232, "test": 464}
    expected_routes = {"train": 14788, "validation": 2068, "test": 5168}
    if system_counts != expected_systems or route_counts != expected_routes:
        raise ValueError(
            f"Unexpected DiffSyn split counts: systems={system_counts}, routes={route_counts}"
        )
    return _finalize_task(
        task_dir,
        {
            "dataset": "DiffSyn / ZeoSyn",
            "task": "zeolite_and_osda_to_synthesis_route_distribution",
            "source_revision": source_revision,
            "split_code_revision": split_code_revision,
            "source_files": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in (workbook, system_list_path, osda_descriptors, zeolite_descriptors)
            ],
            "split_protocol": "historical_upstream_system_disjoint_70_10_20_seed_0_with_ordered_system_list",
            "filter_counts": filter_counts,
            "system_counts": system_counts,
            "route_counts": route_counts,
            "current_head_difference": "current utility keeps test system-disjoint but selects validation rows randomly; this release uses the earlier fully system-disjoint implementation",
            "label_semantics": "each query has a distribution of route rows, not one exact recipe",
            "condition_preprocessing": "upstream IterativeImputer with sample_posterior=True, min_value=0, skip_complete=True, random_state=0; missingness flags are retained",
            "metrics": ["Wasserstein distance", "maximum mean discrepancy", "per-field MAE of distribution summaries"],
            "redistribution_status": "MIT code repository; review ZeoSyn source-data rights before redistribution",
        },
        written,
    )


def _copy_release(source: Path, destination: Path) -> None:
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing release manifest: {source / 'manifest.json'}")
    shutil.copytree(source, destination)


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    crystal_destination = output / "crystal_to_property" / "alignn_jarvis"
    process_destination = output / "process_to_outcome_property" / "specialized_models"
    crystal_destination.parent.mkdir(parents=True, exist_ok=True)
    process_destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_release(args.crystal_release, crystal_destination)
    _copy_release(args.process_release, process_destination)

    inverse_root = output / TASK_FAMILY
    inverse_root.mkdir(parents=True)
    task_manifests = {
        "synthesis_similarity_precursors": build_synthesis_similarity(
            args.synthesis_similarity_jsonl,
            inverse_root / "synthesis_similarity_precursors",
            args.synthesis_similarity_revision,
        ),
        "syntmte_conditions": build_syntmte(
            args.syntmte_year_split,
            inverse_root / "syntmte_conditions",
            args.syntmte_revision,
        ),
        "spende_operations_conditions": build_spende(
            args.spende_json,
            inverse_root / "spende_operations_conditions",
            args.spende_revision,
        ),
        "diffsyn_zeolite_routes": build_diffsyn(
            args.diffsyn_workbook,
            args.diffsyn_system_list,
            args.diffsyn_osda_descriptors,
            args.diffsyn_zeolite_descriptors,
            args.diffsyn_cif_dir,
            inverse_root / "diffsyn_zeolite_routes",
            args.diffsyn_revision,
            args.diffsyn_split_code_revision,
        ),
    }

    registry = {
        "release_id": RELEASE_ID,
        "task_families": {
            "crystal_to_property": "alignn_jarvis",
            "process_to_outcome_property": "specialized_models",
            TASK_FAMILY: list(task_manifests),
        },
        "excluded_initially": {
            "MSP-LLM": "requires LLM fine-tuning; deferred by study design",
            "ARROWS3": "excluded by user request",
        },
    }
    _write_json(output / "source_registry.json", registry)
    readme = """# MatRecall unified specialized-task release (base)

This directory keeps all task families under one root.  Public
validation/test query files contain no target labels; matching labels live in
each task's `private/` directory.  `memory_episodes` is constructed only from
the training membership.

## Families

- `crystal_to_property/alignn_jarvis`: five official ALIGNN/JARVIS property tasks.
- `process_to_outcome_property/specialized_models`: nine synthesis/process outcome tasks.
- `target_process_to_precursor_steps_conditions`: SynthesisSimilarity, SyntMTE,
  SPENDE, and DiffSyn adapters.

Read each task manifest before evaluation.  SynthesisSimilarity and SPENDE use
explicit rebuilt splits; SyntMTE uses its upstream year split; DiffSyn
uses the historical fully system-disjoint upstream split.  Third-party data are
not relicensed by MatRecall.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "release_id": RELEASE_ID,
        "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "unified specialized-model input/output evaluation datasets",
        "copied_releases": {
            "crystal_to_property": json.loads((crystal_destination / "manifest.json").read_text()),
            "process_to_outcome_property": json.loads((process_destination / "manifest.json").read_text()),
        },
        "inverse_synthesis_tasks": {
            name: {
                key: value
                for key, value in task.items()
                if key in {"dataset", "task", "split_protocol", "counts", "official_counts", "route_counts", "system_counts"}
            }
            for name, task in task_manifests.items()
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "safety": "no pickle, joblib, torch checkpoint, or object-array NPZ was deserialized",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crystal-release", type=Path, required=True)
    parser.add_argument("--process-release", type=Path, required=True)
    parser.add_argument("--synthesis-similarity-jsonl", type=Path, required=True)
    parser.add_argument("--synthesis-similarity-revision", required=True)
    parser.add_argument("--syntmte-year-split", type=Path, required=True)
    parser.add_argument("--syntmte-revision", required=True)
    parser.add_argument("--spende-json", type=Path, required=True)
    parser.add_argument("--spende-revision", required=True)
    parser.add_argument("--diffsyn-workbook", type=Path, required=True)
    parser.add_argument("--diffsyn-system-list", type=Path, required=True)
    parser.add_argument("--diffsyn-osda-descriptors", type=Path, required=True)
    parser.add_argument("--diffsyn-zeolite-descriptors", type=Path, required=True)
    parser.add_argument("--diffsyn-cif-dir", type=Path, required=True)
    parser.add_argument("--diffsyn-revision", required=True)
    parser.add_argument("--diffsyn-split-code-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    manifest = build_release(make_parser().parse_args())
    print(json.dumps(manifest["inverse_synthesis_tasks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
