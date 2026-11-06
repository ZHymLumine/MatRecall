"""Build the matched crystal-property task release (ALIGNN and MatRecall).

Each ALIGNN target keeps the checkpoint's published train/validation/test
membership.  Reference episodes from the official training split form the
MatRecall memory, except for training structures that duplicate a validation or
test query by material ID, serialized structure, or a species-aware
``StructureMatcher`` comparison.  Validation and test labels are written under
``private`` and are never included in the query tables.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pymatgen
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition

from matrecall.build_crystal_reference import file_sha256


RELEASE_ID = "matrecall-crystal-tasks"
SPLITS = ("train", "validation", "test")
MATCHER_KWARGS = {
    "ltol": 0.2,
    "stol": 0.3,
    "angle_tol": 5,
    "primitive_cell": True,
    "scale": True,
    "attempt_supercell": False,
    "allow_subset": False,
}


@dataclass(frozen=True)
class TaskSpec:
    model_name: str
    reference_target: str
    unit: str


TASK_SPECS = {
    "formation_energy_peratom": TaskSpec(
        "formation_energy_peratom_radius", "formation_energy_per_atom", "eV/atom"
    ),
    "optb88vdw_bandgap": TaskSpec(
        "optb88vdw_bandgap_radius", "optb88vdw_band_gap", "eV"
    ),
    "mbj_bandgap": TaskSpec("mbj_bandgap_radius", "tb_mbj_band_gap", "eV"),
    "bulk_modulus_kv": TaskSpec(
        "bulk_modulus_kv_radius", "voigt_bulk_modulus", "GPa"
    ),
    "magmom_oszicar": TaskSpec(
        "magmom_oszicar_radius",
        "total_magnetic_moment",
        "mu_B/simulation_cell",
    ),
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _composition_metadata(formula: str) -> tuple[str, str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        composition = Composition(formula)
        reduced_formula = composition.reduced_formula
        anonymous_formula = composition.anonymized_formula
    chemical_system = "-".join(sorted(str(element) for element in composition.elements))
    return reduced_formula, chemical_system, anonymous_formula


def add_structure_metadata(
    structures: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    """Add composition and a transparent structure-family proxy."""
    required = {
        "material_id",
        "structure_ref",
        "formula",
        "atoms_json",
        "structure_sha256",
    }
    missing = sorted(required - set(structures.columns))
    if missing:
        raise ValueError(f"Structure table is missing required columns: {missing}")
    rows = []
    for formula in structures["formula"].astype(str):
        try:
            rows.append(_composition_metadata(formula))
        except Exception as exc:
            raise ValueError(f"Cannot parse JARVIS formula {formula!r}") from exc
    result = structures.copy()
    result[["reduced_formula", "chemical_system", "anonymous_formula"]] = pd.DataFrame(
        rows, index=result.index
    )

    context_rows = episodes[["material_id", "context_json"]].drop_duplicates(
        "material_id"
    )
    context_rows = context_rows.copy()
    context_rows["space_group_number"] = context_rows["context_json"].map(
        lambda value: json.loads(value).get("space_group_number")
    )
    result = result.merge(
        context_rows[["material_id", "space_group_number"]],
        on="material_id",
        how="left",
        validate="one_to_one",
    )
    result["structure_family_proxy"] = result.apply(
        lambda row: (
            f"{row.anonymous_formula}|sg:{int(row.space_group_number)}"
            if pd.notna(row.space_group_number)
            else None
        ),
        axis=1,
    )
    return result


def load_official_splits(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {"id_train", "id_val", "id_test"}
    if set(data) != expected:
        raise ValueError(
            f"Unexpected split keys in {path}: {sorted(data)}; expected {sorted(expected)}"
        )
    result = {
        "train": [str(value) for value in data["id_train"]],
        "validation": [str(value) for value in data["id_val"]],
        "test": [str(value) for value in data["id_test"]],
    }
    duplicates = {
        split: len(ids) - len(set(ids)) for split, ids in result.items()
    }
    if any(duplicates.values()):
        raise ValueError(f"Duplicate IDs within official splits: {duplicates}")
    return result


def cross_split_id_overlaps(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    sets = {split: set(values) for split, values in splits.items()}
    return {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }


def _parse_structure(atoms_json: str):
    from jarvis.core.atoms import Atoms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Atoms.from_dict(json.loads(atoms_json)).pymatgen_converter()


def find_memory_exclusions(
    structures: pd.DataFrame,
    splits: dict[str, list[str]],
    *,
    use_structure_matcher: bool = True,
) -> pd.DataFrame:
    """Return training rows that cannot enter memory for held-out queries."""
    by_id = structures.set_index("material_id", drop=False)
    heldout_rows = []
    for split in ("validation", "test"):
        current = by_id.loc[splits[split]].copy()
        current["heldout_split"] = split
        heldout_rows.append(current)
    heldout = pd.concat(heldout_rows, ignore_index=True)
    train = by_id.loc[splits["train"]].copy()
    train_by_formula = {
        str(formula): group for formula, group in train.groupby("reduced_formula")
    }

    matcher = StructureMatcher(**MATCHER_KWARGS)
    parsed: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    def parsed_structure(row: Any):
        material_id = str(row.material_id)
        if material_id not in parsed:
            parsed[material_id] = _parse_structure(row.atoms_json)
        return parsed[material_id]

    def record(train_row: Any, query_row: Any, reason: str) -> None:
        key = (str(train_row.material_id), str(query_row.material_id), reason)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        exclusions.append(
            {
                "training_material_id": str(train_row.material_id),
                "heldout_material_id": str(query_row.material_id),
                "heldout_split": str(query_row.heldout_split),
                "reason": reason,
                "reduced_formula": str(query_row.reduced_formula),
            }
        )

    for number, query_row in enumerate(heldout.itertuples(index=False), start=1):
        candidates = train_by_formula.get(str(query_row.reduced_formula))
        if candidates is None:
            continue
        for train_row in candidates.itertuples(index=False):
            if str(train_row.material_id) == str(query_row.material_id):
                record(train_row, query_row, "same_material_id")
                continue
            if str(train_row.structure_sha256) == str(query_row.structure_sha256):
                record(train_row, query_row, "same_serialized_structure")
                continue
            if not use_structure_matcher:
                continue
            try:
                train_structure = parsed_structure(train_row)
                query_structure = parsed_structure(query_row)
                if matcher.fit(train_structure, query_structure, symmetric=True):
                    record(train_row, query_row, "pymatgen_structure_matcher")
            except Exception as exc:
                raise ValueError(
                    "StructureMatcher failed for "
                    f"{train_row.material_id} and {query_row.material_id}"
                ) from exc
        if number % 1000 == 0:
            print(
                f"near-duplicate audit heldout={number}/{len(heldout)} "
                f"matches={len(exclusions)}",
                flush=True,
            )
    columns = (
        "training_material_id",
        "heldout_material_id",
        "heldout_split",
        "reason",
        "reduced_formula",
    )
    return pd.DataFrame(exclusions, columns=columns).sort_values(list(columns)).reset_index(
        drop=True
    )


def query_table(
    task: str,
    spec: TaskSpec,
    split: str,
    split_ids: list[str],
    structures: pd.DataFrame,
    official_train_ids: set[str],
) -> pd.DataFrame:
    by_id = structures.set_index("material_id", drop=False)
    frame = by_id.loc[split_ids].copy().reset_index(drop=True)
    official_train = by_id.loc[sorted(official_train_ids)]
    train_formulas = set(official_train["reduced_formula"].dropna())
    train_systems = set(official_train["chemical_system"].dropna())
    train_families = set(official_train["structure_family_proxy"].dropna())
    frame.insert(0, "query_id", frame["material_id"].map(lambda value: f"{task}:{value}"))
    frame.insert(1, "task", task)
    frame.insert(2, "split", split)
    frame.insert(3, "target_name", spec.reference_target)
    frame.insert(4, "target_unit", spec.unit)
    frame["challenge_new_reduced_formula"] = ~frame["reduced_formula"].isin(
        train_formulas
    )
    frame["challenge_new_chemical_system"] = ~frame["chemical_system"].isin(
        train_systems
    )
    frame["challenge_new_structure_family_proxy"] = frame[
        "structure_family_proxy"
    ].notna() & ~frame["structure_family_proxy"].isin(train_families)
    frame["challenge_any"] = frame[
        [
            "challenge_new_reduced_formula",
            "challenge_new_chemical_system",
            "challenge_new_structure_family_proxy",
        ]
    ].any(axis=1)
    return frame[
        [
            "query_id",
            "task",
            "split",
            "target_name",
            "target_unit",
            "material_id",
            "formula",
            "reduced_formula",
            "chemical_system",
            "anonymous_formula",
            "structure_family_proxy",
            "structure_ref",
            "structure_sha256",
            "atoms_json",
            "challenge_new_reduced_formula",
            "challenge_new_chemical_system",
            "challenge_new_structure_family_proxy",
            "challenge_any",
        ]
    ]


def label_table(
    task: str,
    split: str,
    split_ids: list[str],
    target_episodes: pd.DataFrame,
) -> pd.DataFrame:
    labels = target_episodes.set_index("material_id", drop=False).loc[split_ids].copy()
    labels.insert(0, "query_id", labels["material_id"].map(lambda value: f"{task}:{value}"))
    labels.insert(1, "task", task)
    labels.insert(2, "split", split)
    return labels[
        [
            "query_id",
            "task",
            "split",
            "material_id",
            "target_name",
            "target_value",
            "target_unit",
            "episode_id",
            "protocol",
            "source_dataset",
            "source_record_id",
            "source_revision",
        ]
    ].rename(columns={"episode_id": "source_episode_id"})


def _file_record(path: Path, root: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _checkpoint_metadata(model_dir: Path) -> dict[str, Any]:
    result = {"model_directory": str(model_dir)}
    for name in ("config.json", "best_model.pt", "ids_train_val_test.json"):
        path = model_dir / name
        result[name] = {
            "path": str(path),
            "sha256": file_sha256(path) if path.is_file() else None,
        }
    return result


def _validate_source_coverage(
    task: str,
    splits: dict[str, list[str]],
    structures: pd.DataFrame,
    target_episodes: pd.DataFrame,
) -> None:
    structure_ids = set(structures["material_id"].astype(str))
    label_ids = set(target_episodes["material_id"].astype(str))
    for split, ids in splits.items():
        missing_structures = sorted(set(ids) - structure_ids)
        missing_labels = sorted(set(ids) - label_ids)
        if missing_structures or missing_labels:
            raise ValueError(
                f"{task} {split} source coverage failed: "
                f"missing structures={missing_structures[:10]}, "
                f"missing labels={missing_labels[:10]}"
            )


def build_release(
    reference_dir: Path,
    alignn_model_root: Path,
    output: Path,
    tasks: Iterable[str],
    *,
    use_structure_matcher: bool = True,
) -> dict[str, Any]:
    selected_tasks = list(tasks)
    unknown = sorted(set(selected_tasks) - set(TASK_SPECS))
    if unknown:
        raise ValueError(f"Unsupported tasks: {unknown}")
    if len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("Task names must be unique")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    structure_source = reference_dir / "structures.parquet"
    episode_source = reference_dir / "reference_episodes.parquet"
    structures = pd.read_parquet(structure_source)
    episodes = pd.read_parquet(episode_source)
    structures = add_structure_metadata(structures, episodes)

    task_manifests: dict[str, Any] = {}
    release_files: list[dict[str, Any]] = []
    used_structure_ids: set[str] = set()
    exclusion_cache: dict[str, pd.DataFrame] = {}

    for task in selected_tasks:
        print(f"building task={task}", flush=True)
        spec = TASK_SPECS[task]
        model_dir = alignn_model_root / spec.model_name
        split_path = model_dir / "ids_train_val_test.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing official ALIGNN split: {split_path}")
        splits = load_official_splits(split_path)
        target_episodes = episodes[episodes["target_name"] == spec.reference_target].copy()
        _validate_source_coverage(task, splits, structures, target_episodes)
        for ids in splits.values():
            used_structure_ids.update(ids)

        split_hash = file_sha256(split_path)
        if split_hash not in exclusion_cache:
            exclusion_cache[split_hash] = find_memory_exclusions(
                structures, splits, use_structure_matcher=use_structure_matcher
            )
        exclusions = exclusion_cache[split_hash].copy()
        excluded_train_ids = set(exclusions["training_material_id"].astype(str))
        memory_ids = [value for value in splits["train"] if value not in excluded_train_ids]
        memory = target_episodes.set_index("material_id", drop=False).loc[memory_ids].copy()
        memory.insert(0, "task", task)
        memory.insert(1, "split", "train")

        task_root = output / "tasks" / task
        public = task_root / "public"
        private = task_root / "private"
        audit = task_root / "audit"
        for directory in (public, private, audit):
            directory.mkdir(parents=True, exist_ok=True)

        task_files: list[dict[str, Any]] = []
        memory_path = public / "memory_episodes.parquet"
        memory.to_parquet(memory_path, index=False)
        task_files.append(_file_record(memory_path, output, len(memory)))

        official_train_ids = set(splits["train"])
        query_frames: dict[str, pd.DataFrame] = {}
        label_frames: dict[str, pd.DataFrame] = {}
        for split in ("validation", "test"):
            queries = query_table(
                task,
                spec,
                split,
                splits[split],
                structures,
                official_train_ids,
            )
            labels = label_table(task, split, splits[split], target_episodes)
            if list(queries["query_id"]) != list(labels["query_id"]):
                raise ValueError(f"{task} {split} query/label order mismatch")
            query_frames[split] = queries
            label_frames[split] = labels
            query_path = public / f"{split}_queries.parquet"
            label_path = private / f"{split}_labels.parquet"
            queries.to_parquet(query_path, index=False)
            labels.to_parquet(label_path, index=False)
            task_files.extend(
                (
                    _file_record(query_path, output, len(queries)),
                    _file_record(label_path, output, len(labels)),
                )
            )

        membership = pd.DataFrame(
            [
                {"material_id": material_id, "official_split": split}
                for split in SPLITS
                for material_id in splits[split]
            ]
        )
        membership_path = audit / "official_split_membership.parquet"
        exclusions_path = audit / "memory_exclusions.parquet"
        membership.to_parquet(membership_path, index=False)
        exclusions.to_parquet(exclusions_path, index=False)
        task_files.extend(
            (
                _file_record(membership_path, output, len(membership)),
                _file_record(exclusions_path, output, len(exclusions)),
            )
        )

        overlap = cross_split_id_overlaps(splits)
        memory_id_set = set(memory["material_id"].astype(str))
        heldout_id_set = set(splits["validation"]) | set(splits["test"])
        integrity = {
            "query_label_alignment": True,
            "missing_query_values_in_public_tables": {
                split: int("target_value" in query_frames[split].columns)
                for split in ("validation", "test")
            },
            "memory_heldout_material_id_overlap": len(memory_id_set & heldout_id_set),
            "duplicate_memory_episode_ids": int(memory["episode_id"].duplicated().sum()),
            "duplicate_validation_query_ids": int(
                query_frames["validation"]["query_id"].duplicated().sum()
            ),
            "duplicate_test_query_ids": int(
                query_frames["test"]["query_id"].duplicated().sum()
            ),
            "memory_exclusions_applied": int(len(excluded_train_ids)),
        }
        integrity["passed"] = (
            not any(integrity["missing_query_values_in_public_tables"].values())
            and integrity["memory_heldout_material_id_overlap"] == 0
            and integrity["duplicate_memory_episode_ids"] == 0
            and integrity["duplicate_validation_query_ids"] == 0
            and integrity["duplicate_test_query_ids"] == 0
        )
        if not integrity["passed"]:
            raise ValueError(f"{task} integrity failed: {integrity}")

        counts = {
            "official": {split: len(splits[split]) for split in SPLITS},
            "memory_episodes": int(len(memory)),
            "memory_training_ids_excluded": int(len(excluded_train_ids)),
            "memory_exclusion_edges_by_reason": {
                str(key): int(value)
                for key, value in exclusions["reason"].value_counts().sort_index().items()
            },
            "validation_challenge": {
                column: int(query_frames["validation"][column].sum())
                for column in query_frames["validation"].columns
                if column.startswith("challenge_")
            },
            "test_challenge": {
                column: int(query_frames["test"][column].sum())
                for column in query_frames["test"].columns
                if column.startswith("challenge_")
            },
        }
        task_manifest = {
            "task": task,
            "reference_target": spec.reference_target,
            "unit": spec.unit,
            "specialized_model": _checkpoint_metadata(model_dir),
            "split_policy": (
                "ALIGNN checkpoint ids_train_val_test.json; no new random hold-out"
            ),
            "memory_policy": (
                "official training episodes only, with validation/test material-ID, "
                "serialized-structure and StructureMatcher duplicates removed"
            ),
            "challenge_policy": (
                "predefined views of the official held-out split; no resampling"
            ),
            "structure_family_proxy_definition": (
                "pymatgen anonymized formula plus source-reported space-group number; "
                "this is a transparent grouping proxy, not an AFLOW prototype label"
            ),
            "official_cross_split_id_overlaps": overlap,
            "counts": counts,
            "integrity": integrity,
            "files": task_files,
        }
        manifest_path = task_root / "manifest.json"
        _write_json(manifest_path, task_manifest)
        task_files.append(_file_record(manifest_path, output))
        release_files.extend(task_files)
        task_manifests[task] = {
            "manifest": str(manifest_path.relative_to(output)),
            "counts": counts,
            "integrity": integrity,
        }

    release_structures = structures[
        structures["material_id"].astype(str).isin(used_structure_ids)
    ].sort_values("material_id")
    structures_path = output / "structures.parquet"
    release_structures.to_parquet(structures_path, index=False)
    release_files.append(_file_record(structures_path, output, len(release_structures)))

    manifest = {
        "release_id": RELEASE_ID,
        "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "test_access_count": 0,
        "scope": (
            "matched crystal-structure-to-property evaluation for "
            "LLM-only, MatRecall and ALIGNN 2.0"
        ),
        "evaluation_policy": {
            "development": "official validation queries and private validation labels",
            "final": "official test queries; run the frozen systems once",
            "matched_queries": (
                "LLM-only, MatRecall and the specialized predictor must be scored "
                "on identical query IDs for each target"
            ),
            "metric": "per-task MAE in the target's native unit; do not average raw MAE across targets",
        },
        "source": {
            "reference_directory": str(reference_dir),
            "structures_sha256": file_sha256(structure_source),
            "reference_episodes_sha256": file_sha256(episode_source),
            "alignn_model_root": str(alignn_model_root),
        },
        "near_duplicate_policy": {
            "structure_matcher_enabled": use_structure_matcher,
            "matcher": MATCHER_KWARGS if use_structure_matcher else None,
        },
        "tasks": task_manifests,
        "files": sorted(release_files, key=lambda item: item["path"]),
        "builder_code_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "pymatgen": getattr(pymatgen, "__version__", "unknown"),
            "platform": platform.platform(),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--alignn-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(TASK_SPECS), default=list(TASK_SPECS)
    )
    parser.add_argument(
        "--skip-structure-matcher",
        action="store_true",
        help="Smoke-test only: check IDs and exact serialized structures, not near duplicates",
    )
    args = parser.parse_args()
    manifest = build_release(
        args.reference_dir,
        args.alignn_model_root,
        args.output,
        args.tasks,
        use_structure_matcher=not args.skip_structure_matcher,
    )
    print(
        json.dumps(
            {
                task: details["counts"]
                for task, details in manifest["tasks"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
