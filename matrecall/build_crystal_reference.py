"""Build source-resolved MatRecall episodes without relabelling model output as truth.

The first release adapter covers JARVIS-DFT structure--property records.  It
keeps structures in a separate table and writes one scalar-outcome episode per
material and target.  A later inference job can write the same schema under a
``model_generated`` namespace, but must not append those rows to the reference
table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SOURCE_DATASET = "JARVIS-DFT 3D"
SOURCE_URL = "https://jarvis.nist.gov/jarvisdft/"
RELEASE_ID = "matrecall-crystal-reference"


@dataclass(frozen=True)
class TargetSpec:
    source_field: str
    name: str
    unit: str
    protocol: str
    nominal_min: float | None = None
    nominal_max: float | None = None


TARGET_SPECS = (
    TargetSpec(
        "formation_energy_peratom",
        "formation_energy_per_atom",
        "eV/atom",
        "JARVIS-DFT reported formation_energy_peratom",
    ),
    TargetSpec(
        "optb88vdw_bandgap",
        "optb88vdw_band_gap",
        "eV",
        "JARVIS-DFT OptB88vdW band gap",
        nominal_min=0.0,
    ),
    TargetSpec(
        "mbj_bandgap",
        "tb_mbj_band_gap",
        "eV",
        "JARVIS-DFT TB-mBJ band gap",
        nominal_min=0.0,
    ),
    TargetSpec(
        "bulk_modulus_kv",
        "voigt_bulk_modulus",
        "GPa",
        "JARVIS-DFT Voigt bulk modulus",
        nominal_min=0.0,
    ),
    TargetSpec(
        "magmom_oszicar",
        "total_magnetic_moment",
        "mu_B/simulation_cell",
        "JARVIS-DFT total magnetic moment reported from VASP OSZICAR",
        nominal_min=0.0,
    ),
    TargetSpec(
        "poisson",
        "poisson_ratio",
        "dimensionless",
        "JARVIS-DFT reported Poisson ratio",
        nominal_min=-1.0,
        nominal_max=0.5,
    ),
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def nominal_validity(value: float, spec: TargetSpec) -> str:
    if spec.nominal_min is not None and value < spec.nominal_min:
        return "outside_nominal_range"
    if spec.nominal_max is not None and value > spec.nominal_max:
        return "outside_nominal_range"
    return "within_nominal_range"


def _record_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dimensionality": record.get("dimensionality"),
        "functional": record.get("func"),
        "space_group_number": record.get("spg_number"),
        "space_group_symbol": record.get("spg_symbol"),
    }


def build_jarvis_tables(
    records: Iterable[dict[str, Any]],
    source_revision: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structures: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        material_id = str(record.get("jid", "")).strip()
        atoms = record.get("atoms")
        if not material_id or not isinstance(atoms, dict):
            continue
        if material_id in seen_ids:
            raise ValueError(f"Duplicate JARVIS material identifier: {material_id}")
        seen_ids.add(material_id)
        structure_ref = f"jarvis-dft-3d:{material_id}"
        atoms_json = canonical_json(atoms)
        structures.append(
            {
                "structure_ref": structure_ref,
                "material_id": material_id,
                "formula": str(record.get("formula", "")),
                "atoms_json": atoms_json,
                "structure_sha256": hashlib.sha256(atoms_json.encode("utf-8")).hexdigest(),
                "source_dataset": SOURCE_DATASET,
                "source_record_id": material_id,
                "source_revision": source_revision,
                "source_url": SOURCE_URL,
                "evidence_type": "first_principles",
            }
        )
        context_json = canonical_json(_record_context(record))
        for spec in TARGET_SPECS:
            value = finite_float(record.get(spec.source_field))
            if value is None:
                continue
            episodes.append(
                {
                    "episode_id": (
                        f"jarvis-dft-3d:{material_id}:"
                        f"crystal_structure_to_property:{spec.name}"
                    ),
                    "episode_kind": "reference",
                    "task_family": "crystal_structure_to_property",
                    "input_modality": "composition_and_crystal_structure",
                    "material_id": material_id,
                    "formula": str(record.get("formula", "")),
                    "structure_ref": structure_ref,
                    "context_json": context_json,
                    "target_name": spec.name,
                    "target_value": value,
                    "target_unit": spec.unit,
                    "protocol": spec.protocol,
                    "source_field": spec.source_field,
                    "source_dataset": SOURCE_DATASET,
                    "source_record_id": material_id,
                    "source_revision": source_revision,
                    "source_url": SOURCE_URL,
                    "evidence_type": "first_principles",
                    "validity_flag": nominal_validity(value, spec),
                }
            )
    structure_frame = pd.DataFrame(structures)
    episode_frame = pd.DataFrame(episodes)
    return structure_frame, episode_frame


def integrity_report(structures: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, Any]:
    episode_ids = episodes.get("episode_id", pd.Series(dtype=str))
    structure_refs = structures.get("structure_ref", pd.Series(dtype=str))
    episode_structure_refs = episodes.get("structure_ref", pd.Series(dtype=str))
    report = {
        "structures": int(len(structures)),
        "episodes": int(len(episodes)),
        "duplicate_structure_refs": int(structure_refs.duplicated().sum()),
        "duplicate_episode_ids": int(episode_ids.duplicated().sum()),
        "dangling_structure_refs": int((~episode_structure_refs.isin(structure_refs)).sum()),
        "missing_target_values": int(
            episodes.get("target_value", pd.Series(dtype=float)).isna().sum()
        ),
        "episode_counts_by_target": {
            str(key): int(value)
            for key, value in episodes.get(
                "target_name", pd.Series(dtype=str)
            ).value_counts().sort_index().items()
        },
        "validity_counts": {
            str(key): int(value)
            for key, value in episodes.get(
                "validity_flag", pd.Series(dtype=str)
            ).value_counts().sort_index().items()
        },
    }
    report["passed"] = not any(
        report[key]
        for key in (
            "duplicate_structure_refs",
            "duplicate_episode_ids",
            "dangling_structure_refs",
            "missing_target_values",
        )
    )
    return report


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_release(
    input_path: Path,
    output_path: Path,
    sample_size: int,
    limit: int | None = None,
) -> dict[str, Any]:
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("JARVIS input must be a JSON list of records")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        records = records[:limit]
    source_revision = input_path.name
    structures, episodes = build_jarvis_tables(records, source_revision)
    integrity = integrity_report(structures, episodes)
    if not integrity["passed"]:
        raise ValueError(f"Episode integrity checks failed: {integrity}")

    structures_path = output_path / "structures.parquet"
    episodes_path = output_path / "reference_episodes.parquet"
    structures.to_parquet(structures_path, index=False)
    episodes.to_parquet(episodes_path, index=False)

    sample_path = output_path / "reference_episode_sample.jsonl"
    sample = episodes.sort_values(["target_name", "material_id"]).head(sample_size)
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in sample.to_dict(orient="records"):
            handle.write(canonical_json(row) + "\n")

    manifest = {
        "release_id": RELEASE_ID,
        "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "JARVIS-DFT crystal-structure-to-property reference episodes",
        "epistemic_boundary": (
            "These rows preserve source-reported first-principles values. "
            "Predictions from pretrained models must be written to a separate "
            "model_generated release and are not reference labels."
        ),
        "source": {
            "dataset": SOURCE_DATASET,
            "url": SOURCE_URL,
            "input_path": str(input_path),
            "input_sha256": file_sha256(input_path),
            "source_revision": source_revision,
            "licence_review": "required_before_redistribution",
        },
        "record_limit": limit,
        "target_specs": [spec.__dict__ for spec in TARGET_SPECS],
        "integrity": integrity,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (structures_path, episodes_path, sample_path)
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jarvis-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N records for a smoke test",
    )
    args = parser.parse_args()
    manifest = write_release(
        args.jarvis_json,
        args.output,
        args.sample_size,
        limit=args.limit,
    )
    print(json.dumps(manifest["integrity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
