"""Build synthesis/process-to-outcome evaluation datasets.

The release keeps labels out of public validation/test query tables and builds
MatRecall reference memory from training rows only.  Rows are split by source
group (DOI, paper, or organic-template family), with exact input duplicates
connected across source groups before splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import runpy
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from matrecall.build_crystal_reference import file_sha256


RELEASE_ID = "matrecall-process-tasks"
SPLITS = ("train", "validation", "test")
SOURCE_URLS = {
    "raccuglia": "https://doi.org/10.1038/nature17439",
    "inp": "https://github.com/Cossairt-Lab/Indium-Phosphide",
    "meltpool": "https://github.com/BaratiLab/MeltpoolNet",
    "mgt": "https://github.com/s-a-malik/inorg-synth-graph",
    "iron_oxide": "https://doi.org/10.1038/s41467-023-42329-9",
}
SOURCE_REVISIONS = {
    "inp": "23a685dc383dca60bec8ffa370e5a38eeeab4255",
    "meltpool": "6d68e2acaad8d074134a7f3d843278649774d661",
    "mgt": "f270a5b609c80859aa59582482d507e2d7a68abd",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    target_name: str
    unit: str
    kind: str
    metric: str


TASK_SPECS = {
    "raccuglia_crystallization_success": TaskSpec(
        "raccuglia_crystallization_success",
        "hydrothermal_crystallization",
        "crystallization_success",
        "binary",
        "binary_classification",
        "balanced_accuracy, F1, AUROC, and accuracy",
    ),
    "inp_absorption_peak_nm": TaskSpec(
        "inp_absorption_peak_nm",
        "colloidal_nanocrystal_synthesis",
        "absorption_peak",
        "nm",
        "regression",
        "MAE and RMSE in nm",
    ),
    "inp_emission_peak_nm": TaskSpec(
        "inp_emission_peak_nm",
        "colloidal_nanocrystal_synthesis",
        "emission_peak",
        "nm",
        "regression",
        "MAE and RMSE in nm",
    ),
    "inp_particle_diameter_nm": TaskSpec(
        "inp_particle_diameter_nm",
        "colloidal_nanocrystal_synthesis",
        "particle_diameter",
        "nm",
        "regression",
        "MAE and RMSE in nm",
    ),
    "meltpool_depth_um": TaskSpec(
        "meltpool_depth_um",
        "additive_manufacturing",
        "melt_pool_depth",
        "um",
        "regression",
        "MAE and RMSE in um",
    ),
    "meltpool_width_um": TaskSpec(
        "meltpool_width_um",
        "additive_manufacturing",
        "melt_pool_width",
        "um",
        "regression",
        "MAE and RMSE in um",
    ),
    "meltpool_length_um": TaskSpec(
        "meltpool_length_um",
        "additive_manufacturing",
        "melt_pool_length",
        "um",
        "regression",
        "MAE and RMSE in um",
    ),
    "meltpool_shape": TaskSpec(
        "meltpool_shape",
        "additive_manufacturing",
        "melt_pool_shape",
        "class_label",
        "multiclass_classification",
        "macro-F1, balanced accuracy, and accuracy",
    ),
    "mgt_reported_product": TaskSpec(
        "mgt_reported_product",
        "solid_state_synthesis",
        "reported_product_formula",
        "formula",
        "structured_prediction",
        "exact formula accuracy and element-set F1",
    ),
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
            return None
        return text
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(canonical_json(row) + "\n")


def _file_record(path: Path, root: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signature(value: Any) -> str:
    return _stable_hash(canonical_json(value))


def _normal_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "-1", "-1.0", "0", "0.0"}:
        return None
    return text


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
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def connect_source_and_input_groups(
    frame: pd.DataFrame, source_group_col: str = "source_group"
) -> pd.Series:
    """Connect rows sharing either a source group or an exact input signature."""
    union = _UnionFind()
    nodes: list[str] = []
    for row in frame[[source_group_col, "input_signature"]].itertuples(index=False):
        source_node = f"source:{getattr(row, source_group_col)}"
        input_node = f"input:{row.input_signature}"
        union.union(source_node, input_node)
        nodes.append(source_node)
    members: dict[str, list[str]] = {}
    for node in union.parent:
        members.setdefault(union.find(node), []).append(node)
    labels = {
        root: "group:" + _stable_hash("|".join(sorted(values)))[:20]
        for root, values in members.items()
    }
    return pd.Series([labels[union.find(node)] for node in nodes], index=frame.index)


def assign_group_splits(
    frame: pd.DataFrame,
    balance_columns: Iterable[str],
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20261021,
    candidates: int = 128,
) -> tuple[pd.Series, dict[str, Any]]:
    """Choose a deterministic group-disjoint split with count balance."""
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    balance = list(balance_columns)
    aggregate = {"_rows": ("leakage_group", "size")}
    aggregate.update({column: (column, "sum") for column in balance})
    groups = frame.groupby("leakage_group", sort=True).agg(**aggregate)
    targets = {
        column: float(groups[column].sum()) for column in ["_rows", *balance]
    }
    active_splits = [
        split for split, fraction in zip(SPLITS, fractions) if fraction > 0
    ]
    best: tuple[float, int, dict[str, str]] | None = None
    for offset in range(candidates):
        mapping: dict[str, str] = {}
        for group in groups.index.astype(str):
            number = int(_stable_hash(f"{seed + offset}:{group}")[:16], 16) / 16**16
            if number < fractions[0]:
                split = "train"
            elif number < fractions[0] + fractions[1]:
                split = "validation"
            else:
                split = "test"
            mapping[group] = split
        assigned = pd.Series(mapping)
        if any((assigned == split).sum() == 0 for split in active_splits):
            continue
        score = 0.0
        for split, fraction in zip(SPLITS, fractions):
            if fraction == 0:
                continue
            selected = groups.loc[assigned[assigned == split].index]
            for column, total in targets.items():
                if total <= 0:
                    continue
                observed = float(selected[column].sum())
                score += abs(observed - fraction * total) / total
        candidate = (score, seed + offset, mapping)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("Could not construct a non-empty group-disjoint split")
    score, chosen_seed, mapping = best
    split_series = frame["leakage_group"].map(mapping)
    if split_series.isna().any():
        raise ValueError("Some rows were not assigned to a split")
    return split_series, {
        "method": "best of deterministic SHA-256 group assignments",
        "base_seed": seed,
        "candidate_count": candidates,
        "chosen_seed": chosen_seed,
        "balance_score": score,
        "fractions": dict(zip(SPLITS, fractions)),
        "balance_columns": balance,
    }


def _base_record(
    *,
    source_record_id: str,
    source_group: str,
    input_value: dict[str, Any],
    input_text: str,
    feature_value: dict[str, Any],
    context_value: dict[str, Any],
    protocol: str,
    source_dataset: str,
    source_revision: str,
    source_url: str,
    source_sha256: str,
    quality_value: dict[str, Any],
    target_numeric: float | None = None,
    target_text: str | None = None,
) -> dict[str, Any]:
    return {
        "source_record_id": source_record_id,
        "source_group": source_group,
        "input_json": canonical_json(input_value),
        "input_text": input_text,
        "feature_json": canonical_json(feature_value),
        "context_json": canonical_json(context_value),
        "protocol": protocol,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_url": source_url,
        "source_file_sha256": source_sha256,
        "epistemic_status": "experimental",
        "quality_json": canonical_json(quality_value),
        "target_value_numeric": target_numeric,
        "target_value_text": target_text,
        "input_signature": _signature(input_value),
    }


def _raccuglia_input(row: pd.Series) -> dict[str, Any]:
    components = []
    for prefix, role in (
        ("XXXinorg1", "inorganic"),
        ("XXXinorg2", "inorganic"),
        ("XXXinorg3", "inorganic"),
        ("XXXorg1", "organic"),
        ("XXXorg2", "organic"),
        ("XXXoxlike1", "oxide_like"),
    ):
        name = _normal_text(row.get(prefix))
        if name is None:
            continue
        components.append(
            {
                "role": role,
                "name": name,
                "mass": _clean(row.get(prefix + "mass")),
                "moles": _clean(row.get(prefix + "moles")),
            }
        )
    return {
        "components": components,
        "temperature": _clean(row.get("temp")),
        "time": _clean(row.get("time")),
        "slow_cool": _clean(row.get("slowCool")),
        "pH": _clean(row.get("pH")),
        "leak": _clean(row.get("leak")),
    }


def _raccuglia_group(row: pd.Series) -> str:
    organics = sorted(
        value
        for value in (_normal_text(row.get("XXXorg1")), _normal_text(row.get("XXXorg2")))
        if value is not None
    )
    return "organic:" + canonical_json([value.lower() for value in organics])


def load_raccuglia(
    historical_path: Path, prospective_path: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    historical_raw = pd.read_csv(historical_path)
    prospective = pd.read_csv(prospective_path)
    historical_raw.insert(0, "_source_row", range(len(historical_raw)))
    prospective.insert(0, "_source_row", range(len(prospective)))
    duplicate_mask = historical_raw.drop(columns="_source_row").duplicated(keep="first")
    exclusions = [
        {
            "source_dataset": "Raccuglia 2016 historical",
            "source_record_id": f"raccuglia-historical-{int(row):06d}",
            "reason": "exact_duplicate_row",
        }
        for row in historical_raw.loc[duplicate_mask, "_source_row"]
    ]
    historical = historical_raw.loc[~duplicate_mask].copy()
    historical["source_group"] = historical.apply(_raccuglia_group, axis=1)
    prospective["source_group"] = prospective.apply(_raccuglia_group, axis=1)
    test_groups = set(prospective["source_group"])
    overlap_mask = historical["source_group"].isin(test_groups)
    for _, row in historical.loc[overlap_mask].iterrows():
        exclusions.append(
            {
                "source_dataset": "Raccuglia 2016 historical",
                "source_record_id": f"raccuglia-historical-{int(row['_source_row']):06d}",
                "reason": "organic_group_present_in_fixed_prospective_test",
            }
        )
    historical = historical.loc[~overlap_mask].copy()
    shared_features = [
        column
        for column in historical.columns
        if not column.startswith("_")
        and column not in {"XXXtitle", "outcome", "purity", "source_group"}
    ]
    source_sha = {
        "historical": file_sha256(historical_path),
        "prospective": file_sha256(prospective_path),
    }

    def convert(frame: pd.DataFrame, source: str, label_column: str) -> pd.DataFrame:
        rows = []
        for row_index, row in frame.iterrows():
            input_value = _raccuglia_input(row)
            target_class = int(row[label_column])
            success = int(target_class in {3, 4})
            components = ", ".join(
                item["name"] for item in input_value["components"]
            )
            record_id = f"raccuglia-{source}-{int(row['_source_row']):06d}"
            record = _base_record(
                source_record_id=record_id,
                source_group=str(row["source_group"]),
                input_value=input_value,
                input_text=(
                    f"Hydrothermal reaction with {components}; "
                    f"T={input_value['temperature']} C, time={input_value['time']}, "
                    f"pH={input_value['pH']}, slow cool={input_value['slow_cool']}."
                ),
                feature_value={column: row.get(column) for column in shared_features},
                context_value={"source_partition": source},
                protocol="Raccuglia four-class outcome; classes 3/4 mapped to success",
                source_dataset=f"Raccuglia 2016 {source}",
                source_revision=historical_path.name if source == "historical" else prospective_path.name,
                source_url=SOURCE_URLS["raccuglia"],
                source_sha256=source_sha[source],
                quality_value={"label_source": "reported experimental outcome"},
                target_numeric=float(success),
                target_text=str(success),
            )
            record["_balance_positive"] = success
            rows.append(record)
        return pd.DataFrame(rows)

    historical_records = convert(historical, "historical", "outcome")
    historical_records["leakage_group"] = historical_records["source_group"]
    historical_records["split"], split_meta = assign_group_splits(
        historical_records,
        ["_balance_positive"],
        fractions=(0.82, 0.18, 0.0),
    )
    prospective_records = convert(prospective, "prospective", "outcome (actual)")
    prospective_records["leakage_group"] = prospective_records["source_group"]
    prospective_records["split"] = "test"
    records = pd.concat([historical_records, prospective_records], ignore_index=True)
    return (
        {"raccuglia_crystallization_success": records},
        pd.DataFrame(exclusions, columns=("source_dataset", "source_record_id", "reason")),
        {
            "split": split_meta,
            "fixed_test": "published prospective_reactions.csv",
            "historical_raw_rows": len(historical_raw),
            "prospective_rows": len(prospective),
            "historical_exact_duplicates_removed": int(duplicate_mask.sum()),
            "historical_rows_excluded_for_prospective_organic_group_overlap": int(overlap_mask.sum()),
        },
    )


def load_inp(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    raw.insert(0, "_source_row", range(len(raw)))
    duplicate_mask = raw.drop(columns="_source_row").duplicated(keep="first")
    exclusions = pd.DataFrame(
        [
            {
                "source_dataset": "InP nanocrystal synthesis",
                "source_record_id": f"inp-{int(row):06d}",
                "reason": "exact_duplicate_row",
            }
            for row in raw.loc[duplicate_mask, "_source_row"]
        ],
        columns=("source_dataset", "source_record_id", "reason"),
    )
    frame = raw.loc[~duplicate_mask].copy()
    input_columns = [
        "in_source", "in_amount_mmol", "p_source", "p_amount_mmol",
        "first_sol", "first_sol_amount_ml", "second_sol", "second_sol_amount_ml",
        "acid", "acid_amount_mmol", "ligand_source", "ligand_amount_mmol",
        "other_1", "other_1_amount_mmol", "other_2", "other_2_amount_mmol",
        "total_volume_ml", "temp_c", "time_min",
    ]
    numeric_input_columns = [
        "in_amount_mmol", "p_amount_mmol", "first_sol_amount_ml",
        "second_sol_amount_ml", "acid_amount_mmol", "ligand_amount_mmol",
        "other_1_amount_mmol", "other_2_amount_mmol", "total_volume_ml",
        "temp_c", "time_min",
    ]
    for column in numeric_input_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source_group"] = frame["doi"].fillna("missing-doi").astype(str)
    frame["_input_value"] = frame.apply(
        lambda row: {column: _clean(row[column]) for column in input_columns}, axis=1
    )
    frame["input_signature"] = frame["_input_value"].map(_signature)
    frame["leakage_group"] = connect_source_and_input_groups(frame)
    targets = {
        "inp_absorption_peak_nm": "abs_nm",
        "inp_emission_peak_nm": "emission_nm",
        "inp_particle_diameter_nm": "diameter_nm",
    }
    numeric_targets = {column: pd.to_numeric(frame[column], errors="coerce") for column in targets.values()}
    for column, values in numeric_targets.items():
        frame[f"_has_{column}"] = values.notna().astype(int)
    frame["split"], split_meta = assign_group_splits(
        frame,
        [f"_has_{column}" for column in targets.values()],
    )
    source_sha = file_sha256(path)
    task_frames: dict[str, pd.DataFrame] = {}
    for task, column in targets.items():
        values = numeric_targets[column]
        rows = []
        for row_index, row in frame.loc[values.notna()].iterrows():
            input_value = row["_input_value"]
            record = _base_record(
                source_record_id=f"inp-{int(row['_source_row']):06d}",
                source_group=str(row["source_group"]),
                input_value=input_value,
                input_text=(
                    f"InP synthesis using {input_value['in_source']} and "
                    f"{input_value['p_source']} at {input_value['temp_c']} C for "
                    f"{input_value['time_min']} min."
                ),
                feature_value=input_value,
                context_value={"doi": row.get("doi")},
                protocol="literature-extracted InP nanocrystal synthesis",
                source_dataset="InP nanocrystal synthesis",
                source_revision=f"{SOURCE_REVISIONS['inp']}:{path.name}",
                source_url=SOURCE_URLS["inp"],
                source_sha256=source_sha,
                quality_value={"label_observed": True, "imputation_used": False},
                target_numeric=float(values.loc[row_index]),
            )
            record["leakage_group"] = row["leakage_group"]
            record["split"] = row["split"]
            rows.append(record)
        task_frames[task] = pd.DataFrame(rows)
    return task_frames, exclusions, {
        "split": split_meta,
        "raw_rows": len(raw),
        "exact_duplicates_removed": int(duplicate_mask.sum()),
        "unique_doi": int(frame["source_group"].nunique()),
        "observed_target_rows": {
            task: int(numeric_targets[column].notna().sum())
            for task, column in targets.items()
        },
    }


def _meltpool_input(row: pd.Series) -> dict[str, Any]:
    columns = [
        "Material", "Process", "Sub-process", "Power", "Velocity",
        "powder flowrate", "layer thickness", "beam D", "Hatch spacing",
        "E (J/mm)", "E (J/mm3)", "absorption coefficient",
        "absorption coefficient 2", "density", "Cp", "k", "melting T",
        "minimum absorptivity", "Y (wt%)", "Zn (wt%)", "Mg (wt%)",
        "Si (wt%)", "Al (wt%)", "Sn (wt%)", "Zr (wt%)", "W (wt%)",
        "Ti (wt%)", "V (wt%)", "Co (wt%)", "Cu (wt%)", "Ta (wt%)",
        "Nb (wt%)", "Ni (wt.%)", "Cr (wt.%)", "Fe (wt.%)", "Mn (wt%)",
        "Mo (wt.%)", "D10", "D50", "D90",
    ]
    return {column: _clean(row.get(column)) for column in columns}


def load_meltpool(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    raw.insert(0, "_source_row", range(len(raw)))
    duplicate_mask = raw.drop(columns="_source_row").duplicated(keep="first")
    exclusions = pd.DataFrame(
        [
            {
                "source_dataset": "MeltpoolNet regression table",
                "source_record_id": f"meltpool-{int(row):06d}",
                "reason": "exact_duplicate_row",
            }
            for row in raw.loc[duplicate_mask, "_source_row"]
        ],
        columns=("source_dataset", "source_record_id", "reason"),
    )
    frame = raw.loc[~duplicate_mask].copy()
    frame["source_group"] = frame["paper ID"].astype(str)
    frame["_input_value"] = frame.apply(_meltpool_input, axis=1)
    frame["input_signature"] = frame["_input_value"].map(_signature)
    frame["leakage_group"] = connect_source_and_input_groups(frame)
    targets = {
        "meltpool_depth_um": "depth of meltpool",
        "meltpool_width_um": "width of melt pool",
        "meltpool_length_um": "length of melt pool",
        "meltpool_shape": "meltpool shape",
    }
    numeric = {
        task: pd.to_numeric(frame[column], errors="coerce")
        for task, column in targets.items()
        if task != "meltpool_shape"
    }
    balance_columns = []
    for task, column in targets.items():
        indicator = f"_has_{task}"
        if task == "meltpool_shape":
            frame[indicator] = frame[column].notna().astype(int)
        else:
            frame[indicator] = numeric[task].notna().astype(int)
        balance_columns.append(indicator)
    for label in sorted(frame["meltpool shape"].dropna().astype(str).unique()):
        column = "_shape_" + _stable_hash(label)[:8]
        frame[column] = (frame["meltpool shape"].astype(str) == label).astype(int)
        balance_columns.append(column)
    frame["split"], split_meta = assign_group_splits(frame, balance_columns)
    source_sha = file_sha256(path)
    task_frames: dict[str, pd.DataFrame] = {}
    for task, target_column in targets.items():
        observed = frame[target_column].notna()
        if task != "meltpool_shape":
            observed = numeric[task].notna()
        rows = []
        for row_index, row in frame.loc[observed].iterrows():
            input_value = row["_input_value"]
            target_numeric = None
            target_text = None
            if task == "meltpool_shape":
                target_text = str(row[target_column])
            else:
                target_numeric = float(numeric[task].loc[row_index])
            record = _base_record(
                source_record_id=f"meltpool-{int(row['_source_row']):06d}",
                source_group=str(row["source_group"]),
                input_value=input_value,
                input_text=(
                    f"{input_value['Process']} / {input_value['Sub-process']} of "
                    f"{input_value['Material']} at power={input_value['Power']} W and "
                    f"velocity={input_value['Velocity']}."
                ),
                feature_value=input_value,
                context_value={"paper_id": row.get("paper ID"), "paper": row.get("paper")},
                protocol="literature-collected additive-manufacturing experiment",
                source_dataset="MeltpoolNet regression table",
                source_revision=f"{SOURCE_REVISIONS['meltpool']}:{path.name}",
                source_url=SOURCE_URLS["meltpool"],
                source_sha256=source_sha,
                quality_value={"label_observed": True},
                target_numeric=target_numeric,
                target_text=target_text,
            )
            record["leakage_group"] = row["leakage_group"]
            record["split"] = row["split"]
            rows.append(record)
        task_frames[task] = pd.DataFrame(rows)
    return task_frames, exclusions, {
        "split": split_meta,
        "raw_rows": len(raw),
        "exact_duplicates_removed": int(duplicate_mask.sum()),
        "unique_paper_ids": int(frame["source_group"].nunique()),
        "observed_target_rows": {
            task: int(frame[column].notna().sum()) for task, column in targets.items()
        },
        "excluded_columns": [
            "d/l", "d/w", "l/w", "spatter", "porosity", "relative density",
            "comment", "Unnamed: 53",
        ],
    }


def _mgt_filter(raw: list[dict[str, Any]], exclude_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    namespace = runpy.run_path(str(exclude_path))
    bad_doi = set(namespace["BAD_DOI"])
    bad_precursors = set(namespace["BAD_PRECURSORS"])
    bad_targets = set(namespace["BAD_TARGETS"])
    kept = []
    reasons = {"precursor_count": 0, "bad_doi": 0, "bad_precursor": 0, "bad_target": 0}
    for row in raw:
        if not 2 <= len(row["precursors"]) <= 10:
            reasons["precursor_count"] += 1
        elif row["doi"] in bad_doi:
            reasons["bad_doi"] += 1
        elif any(value in bad_targets for value in row["targets_string"]):
            reasons["bad_target"] += 1
        elif any(item["material_formula"] in bad_precursors for item in row["precursors"]):
            reasons["bad_precursor"] += 1
        else:
            kept.append(row)
    return kept, reasons


def _mgt_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "precursors": [
            {
                "material_string": item.get("material_string"),
                "material_formula": item.get("material_formula"),
            }
            for item in row["precursors"]
        ],
        "reactant_stoichiometry": row.get("reaction", {}).get("left_side", []),
        "operations": row.get("operations", []),
    }


def load_mgt(
    path: Path, exclude_path: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    kept, filter_counts = _mgt_filter(raw, exclude_path)
    source_sha = file_sha256(path)
    rows = []
    raw_positions = {id(row): index for index, row in enumerate(raw)}
    for row in kept:
        source_row = raw_positions[id(row)]
        input_value = _mgt_input(row)
        target = str(row["target"]["material_formula"])
        record = _base_record(
            source_record_id=f"mgt-{source_row:06d}",
            source_group=str(row["doi"]),
            input_value=input_value,
            input_text=(
                "Solid-state synthesis from "
                + ", ".join(item["material_formula"] for item in row["precursors"])
                + "; operations: "
                + ", ".join(item.get("string", item.get("type", "")) for item in row["operations"])
                + "."
            ),
            feature_value=input_value,
            context_value={"doi": row["doi"]},
            protocol="text-mined solid-state synthesis procedure",
            source_dataset="MGT solid-state synthesis dataset",
            source_revision=f"{SOURCE_REVISIONS['mgt']}:{path.name}",
            source_url=SOURCE_URLS["mgt"],
            source_sha256=source_sha,
            quality_value={"source": "text-mined literature record"},
            target_text=target,
        )
        rows.append(record)
    frame = pd.DataFrame(rows)
    duplicate_mask = frame.duplicated(subset=["source_group", "input_signature", "target_value_text"], keep="first")
    exclusions = pd.DataFrame(
        [
            {
                "source_dataset": "MGT solid-state synthesis dataset",
                "source_record_id": source_record_id,
                "reason": "duplicate_doi_input_target",
            }
            for source_record_id in frame.loc[duplicate_mask, "source_record_id"]
        ],
        columns=("source_dataset", "source_record_id", "reason"),
    )
    frame = frame.loc[~duplicate_mask].copy()
    frame["leakage_group"] = connect_source_and_input_groups(frame)
    frame["_row"] = 1
    frame["split"], split_meta = assign_group_splits(frame, ["_row"])
    return {"mgt_reported_product": frame}, exclusions, {
        "split": split_meta,
        "raw_rows": len(raw),
        "published_filter_exclusions": filter_counts,
        "rows_after_published_filter": len(kept),
        "duplicate_doi_input_target_rows_removed": int(duplicate_mask.sum()),
        "rows_in_release": len(frame),
        "unique_doi": int(frame["source_group"].nunique()),
        "exclude_code_sha256": file_sha256(exclude_path),
    }


def _task_tables(frame: pd.DataFrame, spec: TaskSpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    common = [
        "source_record_id", "source_group", "leakage_group", "input_json",
        "input_text", "feature_json", "context_json", "protocol", "source_dataset",
        "source_revision", "source_url", "source_file_sha256", "epistemic_status",
        "quality_json", "target_value_numeric", "target_value_text", "split",
    ]
    missing = sorted(set(common) - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.name} missing columns: {missing}")
    work = frame[common].copy()
    work.insert(0, "task", spec.name)
    work.insert(1, "task_family", spec.family)
    work.insert(2, "target_name", spec.target_name)
    work.insert(3, "target_unit", spec.unit)
    work["target_value_json"] = work.apply(
        lambda row: canonical_json(
            row["target_value_numeric"]
            if pd.notna(row["target_value_numeric"])
            else row["target_value_text"]
        ),
        axis=1,
    )
    train = work.loc[work["split"] == "train"].copy()
    train.insert(0, "episode_id", train["source_record_id"].map(lambda value: f"episode:{spec.name}:{value}"))
    train.insert(1, "episode_kind", "reference")
    queries: dict[str, pd.DataFrame] = {}
    labels: dict[str, pd.DataFrame] = {}
    target_columns = {"target_value_numeric", "target_value_text", "target_value_json"}
    for split in ("validation", "test"):
        heldout = work.loc[work["split"] == split].copy()
        heldout.insert(0, "query_id", heldout["source_record_id"].map(lambda value: f"query:{spec.name}:{value}"))
        label_columns = [
            "query_id", "task", "split", "source_record_id", "target_name",
            "target_unit", "target_value_numeric", "target_value_text", "target_value_json",
            "protocol", "source_dataset", "source_revision", "source_file_sha256",
        ]
        labels[split] = heldout[label_columns].copy()
        queries[split] = heldout.drop(columns=sorted(target_columns)).copy()
    return train, queries, labels


def _integrity(
    memory: pd.DataFrame,
    queries: dict[str, pd.DataFrame],
    labels: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    memory_groups = set(memory["leakage_group"])
    validation_groups = set(queries["validation"]["leakage_group"])
    test_groups = set(queries["test"]["leakage_group"])
    leaked_columns = sorted(
        {
            column
            for split in ("validation", "test")
            for column in queries[split].columns
            if column.startswith("target_value")
        }
    )
    result = {
        "public_query_label_columns": leaked_columns,
        "memory_validation_group_overlap": len(memory_groups & validation_groups),
        "memory_test_group_overlap": len(memory_groups & test_groups),
        "validation_test_group_overlap": len(validation_groups & test_groups),
        "duplicate_memory_episode_ids": int(memory["episode_id"].duplicated().sum()),
        "duplicate_validation_query_ids": int(queries["validation"]["query_id"].duplicated().sum()),
        "duplicate_test_query_ids": int(queries["test"]["query_id"].duplicated().sum()),
        "query_label_alignment": all(
            list(queries[split]["query_id"]) == list(labels[split]["query_id"])
            for split in ("validation", "test")
        ),
    }
    result["passed"] = (
        not result["public_query_label_columns"]
        and result["memory_validation_group_overlap"] == 0
        and result["memory_test_group_overlap"] == 0
        and result["validation_test_group_overlap"] == 0
        and result["duplicate_memory_episode_ids"] == 0
        and result["duplicate_validation_query_ids"] == 0
        and result["duplicate_test_query_ids"] == 0
        and result["query_label_alignment"]
    )
    return result


def build_release(
    *,
    raccuglia_historical: Path,
    raccuglia_prospective: Path,
    inp_csv: Path,
    meltpool_csv: Path,
    mgt_json: Path,
    mgt_exclude_py: Path,
    iron_oxide_paper: Path,
    output: Path,
) -> dict[str, Any]:
    source_paths = [
        raccuglia_historical, raccuglia_prospective, inp_csv, meltpool_csv,
        mgt_json, mgt_exclude_py, iron_oxide_paper,
    ]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    loaders: list[tuple[str, Callable[[], Any]]] = [
        ("raccuglia", lambda: load_raccuglia(raccuglia_historical, raccuglia_prospective)),
        ("inp", lambda: load_inp(inp_csv)),
        ("meltpool", lambda: load_meltpool(meltpool_csv)),
        ("mgt", lambda: load_mgt(mgt_json, mgt_exclude_py)),
    ]
    all_tasks: dict[str, pd.DataFrame] = {}
    source_exploration: dict[str, Any] = {}
    source_exclusions: dict[str, pd.DataFrame] = {}
    for source, loader in loaders:
        print(f"loading source={source}", flush=True)
        tasks, exclusions, exploration = loader()
        overlap = sorted(set(all_tasks) & set(tasks))
        if overlap:
            raise ValueError(f"Duplicate task names: {overlap}")
        all_tasks.update(tasks)
        source_exclusions[source] = exclusions
        source_exploration[source] = exploration

    release_files: list[dict[str, Any]] = []
    task_index: dict[str, Any] = {}
    for task, frame in all_tasks.items():
        spec = TASK_SPECS[task]
        task_root = output / "tasks" / task
        public = task_root / "public"
        private = task_root / "private"
        audit = task_root / "audit"
        for directory in (public, private, audit):
            directory.mkdir(parents=True, exist_ok=True)
        memory, queries, labels = _task_tables(frame, spec)
        integrity = _integrity(memory, queries, labels)
        if not integrity["passed"]:
            raise ValueError(f"{task} integrity failed: {integrity}")
        task_files = []
        memory_path = public / "memory_episodes.parquet"
        memory_jsonl = public / "memory_episodes.jsonl"
        memory.to_parquet(memory_path, index=False)
        _write_jsonl(memory_jsonl, memory)
        task_files.extend((
            _file_record(memory_path, output, len(memory)),
            _file_record(memory_jsonl, output, len(memory)),
        ))
        for split in ("validation", "test"):
            query_path = public / f"{split}_queries.parquet"
            query_jsonl = public / f"{split}_queries.jsonl"
            label_path = private / f"{split}_labels.parquet"
            queries[split].to_parquet(query_path, index=False)
            _write_jsonl(query_jsonl, queries[split])
            labels[split].to_parquet(label_path, index=False)
            task_files.extend((
                _file_record(query_path, output, len(queries[split])),
                _file_record(query_jsonl, output, len(queries[split])),
                _file_record(label_path, output, len(labels[split])),
            ))
        membership = frame[["source_record_id", "source_group", "leakage_group", "split"]].copy()
        membership_path = audit / "split_membership.parquet"
        membership.to_parquet(membership_path, index=False)
        task_files.append(_file_record(membership_path, output, len(membership)))
        source = "raccuglia" if task.startswith("raccuglia") else "inp" if task.startswith("inp") else "meltpool" if task.startswith("meltpool") else "mgt"
        exclusions = source_exclusions[source]
        exclusions_path = audit / "source_exclusions.parquet"
        exclusions.to_parquet(exclusions_path, index=False)
        task_files.append(_file_record(exclusions_path, output, len(exclusions)))
        counts = {
            split: int((frame["split"] == split).sum()) for split in SPLITS
        }
        task_manifest = {
            "task": task,
            "task_family": spec.family,
            "target_name": spec.target_name,
            "target_unit": spec.unit,
            "task_kind": spec.kind,
            "metric": spec.metric,
            "counts": counts,
            "memory_policy": "reference episodes from the training split only",
            "split_policy": source_exploration[source]["split"],
            "integrity": integrity,
            "files": task_files,
        }
        manifest_path = task_root / "manifest.json"
        _write_json(manifest_path, task_manifest)
        task_files.append(_file_record(manifest_path, output))
        release_files.extend(task_files)
        task_index[task] = {
            "manifest": str(manifest_path.relative_to(output)),
            "counts": counts,
            "integrity": integrity,
        }

    source_registry = {
        "candidate_count_after_removing_ARROWS3": 5,
        "retained": {
            "raccuglia_2016": {
                "public_data": True,
                "public_code_or_model_parameters": True,
                "license_note": "publisher supplementary files; no explicit reusable-data license found in the downloaded files",
                "source_url": SOURCE_URLS["raccuglia"],
            },
            "inp_nanocrystal": {
                "public_data": True,
                "public_code": True,
                "repository_license": "MIT",
                "repository_commit": SOURCE_REVISIONS["inp"],
                "source_url": SOURCE_URLS["inp"],
            },
            "meltpoolnet": {
                "public_data": True,
                "public_code": True,
                "license_note": "no LICENSE file found in downloaded repository; redistribution requires review",
                "repository_commit": SOURCE_REVISIONS["meltpool"],
                "source_url": SOURCE_URLS["meltpool"],
            },
            "mgt_2021": {
                "public_data": True,
                "public_code": True,
                "repository_license": "MIT",
                "repository_commit": SOURCE_REVISIONS["mgt"],
                "source_url": SOURCE_URLS["mgt"],
            },
        },
        "excluded": {
            "iron_oxide_nanoparticle_2023": {
                "reason": "paper states that models, code, and dataset are available by request; no public row-level download was identified",
                "paper_sha256": file_sha256(iron_oxide_paper),
                "source_url": SOURCE_URLS["iron_oxide"],
            }
        },
    }
    registry_path = output / "source_registry.json"
    exploration_path = output / "source_exploration.json"
    _write_json(registry_path, source_registry)
    _write_json(exploration_path, source_exploration)
    release_files.extend((
        _file_record(registry_path, output),
        _file_record(exploration_path, output),
    ))
    manifest = {
        "release_id": RELEASE_ID,
        "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthesis/process-to-outcome/property evaluation",
        "frozen": True,
        "test_access_count": 0,
        "evaluation_policy": {
            "memory": "training rows only",
            "development": "public validation queries plus private validation labels",
            "final": "run frozen systems once on public test queries and score with private test labels",
            "matched_queries": "LLM-only, MatRecall, and specialized/simple baselines use identical query IDs per task",
            "grouping": "source groups plus connected exact-input duplicates are split together",
        },
        "tasks": task_index,
        "source_registry": str(registry_path.relative_to(output)),
        "source_exploration": str(exploration_path.relative_to(output)),
        "files": sorted(release_files, key=lambda item: item["path"]),
        "builder_code_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raccuglia-historical", type=Path, required=True)
    parser.add_argument("--raccuglia-prospective", type=Path, required=True)
    parser.add_argument("--inp-csv", type=Path, required=True)
    parser.add_argument("--meltpool-csv", type=Path, required=True)
    parser.add_argument("--mgt-json", type=Path, required=True)
    parser.add_argument("--mgt-exclude-py", type=Path, required=True)
    parser.add_argument("--iron-oxide-paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_release(**vars(args))
    print(json.dumps({task: value["counts"] for task, value in manifest["tasks"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
