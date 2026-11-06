"""Composition, local-environment, XRD and SOAP descriptors for crystal structures.

The local-environment descriptor is not SOAP: it concatenates a nearest-shell
coordination histogram with a periodic radial distribution histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from ase.data import chemical_symbols
from dscribe.descriptors import SOAP
import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.prototypes import AflowPrototypeMatcher
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


N_ELEMENTS = 118
LOCAL_CN_BINS = 17
LOCAL_RDF_BINS = 60
LOCAL_CUTOFF_ANGSTROM = 6.0
XRD_MIN_2THETA = 5.0
XRD_MAX_2THETA = 90.0
XRD_STEP_2THETA = 0.1
SOAP_CONFIG = {
    "periodic": True,
    "r_cut": 5.0,
    "n_max": 4,
    "l_max": 3,
    "sigma": 0.5,
    "average": "inner",
    "compression": {"mode": "mu2"},
    "dtype": "float32",
}
SYMMETRY_CONFIG = {"symprec": 0.1, "angle_tolerance": 5.0}

_SOAP: SOAP | None = None
_AFLOW: AflowPrototypeMatcher | None = None


@dataclass(frozen=True)
class FormulaFeatures:
    reduced_formula: str
    anonymized_formula: str
    chemical_system: str
    vector: np.ndarray


def formula_features(formula: str) -> FormulaFeatures:
    """Return canonical formula fields and a normalized 118-element vector."""
    composition = Composition(formula)
    fractional = composition.fractional_composition
    vector = np.zeros(N_ELEMENTS, dtype=np.float32)
    for element, amount in fractional.items():
        vector[Element(element).Z - 1] = float(amount)
    return FormulaFeatures(
        reduced_formula=composition.reduced_formula,
        anonymized_formula=composition.anonymized_formula,
        chemical_system="-".join(sorted(str(e) for e in composition.elements)),
        vector=vector,
    )


def local_environment_descriptor(structure: Structure) -> np.ndarray:
    """Average nearest-shell coordination plus periodic RDF descriptor.

    For each atomic site, the nearest shell is defined as distances no greater
    than 1.25 times that site's nearest-neighbour distance.  The RDF covers
    0--6 Å in 0.1 Å bins.  Both blocks are independently L1-normalized.
    """
    cn_hist = np.zeros(LOCAL_CN_BINS, dtype=np.float32)
    rdf_hist = np.zeros(LOCAL_RDF_BINS, dtype=np.float32)
    rdf_edges = np.linspace(0.0, LOCAL_CUTOFF_ANGSTROM, LOCAL_RDF_BINS + 1)

    for site in structure:
        neighbours = structure.get_neighbors(site, LOCAL_CUTOFF_ANGSTROM)
        distances = np.asarray([float(n.nn_distance) for n in neighbours], dtype=float)
        distances = distances[distances > 1e-6]
        if not len(distances):
            cn_hist[0] += 1
            continue
        nearest = float(distances.min())
        coordination = int(np.count_nonzero(distances <= 1.25 * nearest))
        cn_hist[min(coordination, LOCAL_CN_BINS - 1)] += 1
        rdf_hist += np.histogram(distances, bins=rdf_edges)[0].astype(np.float32)

    if cn_hist.sum():
        cn_hist /= cn_hist.sum()
    if rdf_hist.sum():
        rdf_hist /= rdf_hist.sum()
    return np.concatenate([cn_hist, rdf_hist]).astype(np.float32)


def xrd_descriptor(structure: Structure) -> np.ndarray:
    """Return a max-normalized ideal Cu K-alpha powder XRD vector."""
    size = int(round((XRD_MAX_2THETA - XRD_MIN_2THETA) / XRD_STEP_2THETA)) + 1
    vector = np.zeros(size, dtype=np.float32)
    pattern = XRDCalculator(wavelength="CuKa").get_pattern(
        structure, two_theta_range=(XRD_MIN_2THETA, XRD_MAX_2THETA)
    )
    for angle, intensity in zip(pattern.x, pattern.y):
        index = int(round((float(angle) - XRD_MIN_2THETA) / XRD_STEP_2THETA))
        if 0 <= index < size:
            vector[index] = max(vector[index], float(intensity))
    maximum = float(vector.max())
    if maximum:
        vector /= maximum
    return vector


def soap_descriptor() -> SOAP:
    global _SOAP
    if _SOAP is None:
        _SOAP = SOAP(species=chemical_symbols[1:119], **SOAP_CONFIG)
    return _SOAP


def aflow_matcher() -> AflowPrototypeMatcher:
    global _AFLOW
    if _AFLOW is None:
        _AFLOW = AflowPrototypeMatcher()
    return _AFLOW


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def wyckoff_signatures(structure: Structure) -> tuple[int, str, str]:
    analyzer = SpacegroupAnalyzer(structure, **SYMMETRY_CONFIG)
    dataset = analyzer.get_symmetry_dataset()
    equivalent = np.asarray(dataset.equivalent_atoms)
    wyckoffs = list(dataset.wyckoffs)
    anonymous_tokens = []
    species_tokens = []
    for orbit in sorted(set(equivalent.tolist())):
        indices = np.flatnonzero(equivalent == orbit)
        first = int(indices[0])
        token = f"{len(indices)}{wyckoffs[first]}"
        anonymous_tokens.append(token)
        species_tokens.append(f"{structure[first].specie.symbol}:{token}")
    number = int(dataset.number)
    return (
        number,
        f"sg{number}|" + ".".join(sorted(anonymous_tokens)),
        f"sg{number}|" + ".".join(sorted(species_tokens)),
    )


def aflow_labels(structure: Structure) -> tuple[str, str, str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matches = aflow_matcher().get_prototypes(structure)
    if not matches:
        return "", "", "", ""
    tags = matches[0].get("tags", {})
    return (
        str(tags.get("aflow", "") or ""),
        str(tags.get("pearson", "") or ""),
        str(tags.get("strukturbericht", "") or ""),
        str(tags.get("mineral", "") or ""),
    )


def process_structure(
    material_id: str,
    structure: Structure,
    source: str,
    reference: str,
    formation_energy: Any,
    e_above_hull: Any,
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = formula_features(structure.composition.reduced_formula)
        symmetry_number, wyckoff_anonymous, wyckoff_species = wyckoff_signatures(structure)
        aflow, pearson, strukturbericht, mineral = aflow_labels(structure)
        soap = l2_normalize(
            soap_descriptor().create(AseAtomsAdaptor.get_atoms(structure))
        )
        local = l2_normalize(local_environment_descriptor(structure))
        xrd = l2_normalize(xrd_descriptor(structure))
    return {
        "ok": True,
        "material_id": material_id,
        "source": source,
        "reference": reference,
        "reduced_formula": features.reduced_formula,
        "anonymized_formula": features.anonymized_formula,
        "chemical_system": features.chemical_system,
        "spacegroup_number": symmetry_number,
        "site_count": len(structure),
        "formation_energy_per_atom": pd.to_numeric(formation_energy, errors="coerce"),
        "e_above_hull": pd.to_numeric(e_above_hull, errors="coerce"),
        "wyckoff_anonymous": wyckoff_anonymous,
        "wyckoff_species": wyckoff_species,
        "aflow_prototype": aflow,
        "pearson_symbol": pearson,
        "strukturbericht": strukturbericht,
        "mineral_name": mineral,
        "composition": l2_normalize(features.vector),
        "local_environment": local,
        "xrd": xrd,
        "soap_mu2": soap,
    }
