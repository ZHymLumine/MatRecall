"""Build the JARVIS structure-descriptor index for the crystal-property structures.

Composition, local-environment, XRD and SOAP descriptors are computed for every
structure in the supplied structure table.  The output must cover every
structure; partial indexes are rejected.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms

from matrecall.build_crystal_reference import file_sha256
from matrecall.structure_descriptors import process_structure


VECTOR_FIELDS = ("composition", "local_environment", "xrd", "soap_mu2")


def compute_record(record: dict[str, Any]) -> dict[str, Any]:
    try:
        structure = Atoms.from_dict(json.loads(record["atoms_json"])).pymatgen_converter()
        return process_structure(
            material_id=str(record["material_id"]),
            structure=structure,
            source="jarvis_dft_3d",
            reference=str(record["structure_ref"]),
            formation_energy=None,
            e_above_hull=None,
        )
    except Exception as exc:
        return {
            "ok": False,
            "material_id": str(record.get("material_id", "")),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    structures = pd.read_parquet(args.structures)
    if structures["material_id"].duplicated().any():
        raise ValueError("Duplicate material IDs in frozen structure table")
    desired_ids = structures["material_id"].astype(str).tolist()

    computed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for number, result in enumerate(
            executor.map(compute_record, structures.to_dict("records"), chunksize=4), start=1
        ):
            if result["ok"]:
                computed[result["material_id"]] = result
            else:
                failures.append(result)
            if number % 250 == 0 or number == len(structures):
                print(json.dumps({
                    "event": "descriptor-progress", "processed": number,
                    "total": len(structures), "success": len(computed),
                    "failed": len(failures),
                }), flush=True)
    pd.DataFrame(failures).to_parquet(args.output / "failures.parquet", index=False)
    if failures:
        raise ValueError(f"Descriptor construction failed for {len(failures)} frozen structures")

    output_vectors: dict[str, list[np.ndarray]] = {field: [] for field in VECTOR_FIELDS}
    metadata_rows: list[dict[str, Any]] = []
    for material_id in desired_ids:
        row = computed[material_id]
        for field in VECTOR_FIELDS:
            output_vectors[field].append(np.asarray(row.pop(field), dtype=np.float32))
        row.pop("ok", None)
        metadata_rows.append(row)
    ids = np.asarray(desired_ids)
    for field in VECTOR_FIELDS:
        np.savez_compressed(
            args.output / f"{field}.npz",
            material_ids=ids,
            vectors=np.stack(output_vectors[field]).astype(np.float32),
        )
    metadata = pd.DataFrame(metadata_rows)
    metadata["material_id"] = desired_ids
    metadata.to_parquet(args.output / "metadata.parquet", index=False)
    if len(metadata) != len(structures):
        raise ValueError("Full index row count mismatch")
    files = [
        args.output / "metadata.parquet", args.output / "failures.parquet",
        *[args.output / f"{field}.npz" for field in VECTOR_FIELDS],
    ]
    manifest = {
        "index_version": "matrecall-structure-index",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_records": len(structures),
        "computed_records": len(computed),
        "failures": len(failures),
        "source_sha256": {"structures": file_sha256(args.structures)},
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        },
        "builder_sha256": file_sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
