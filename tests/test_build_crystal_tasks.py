import json

import pandas as pd

from matrecall.build_crystal_tasks import (
    TaskSpec,
    cross_split_id_overlaps,
    find_memory_exclusions,
    label_table,
    load_official_splits,
    query_table,
)


def _atoms(elements):
    return json.dumps(
        {
            "lattice_mat": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
            "elements": elements,
            "cartesian": False,
        },
        sort_keys=True,
    )


def _structures():
    rows = [
        ("train-same-id", "NaCl", _atoms(["Na", "Cl"]), "h1", "AB|sg:225"),
        ("train-same-structure", "NaCl", _atoms(["Na", "Cl"]), "h2", "AB|sg:225"),
        ("train-safe", "LiF", _atoms(["Li", "F"]), "h3", "AB|sg:225"),
        ("test-duplicate", "NaCl", _atoms(["Na", "Cl"]), "h2", "AB|sg:225"),
        ("new-system", "KBr", _atoms(["K", "Br"]), "h4", "AB|sg:225"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=(
            "material_id",
            "formula",
            "atoms_json",
            "structure_sha256",
            "structure_family_proxy",
        ),
    )
    frame["structure_ref"] = frame["material_id"].map(lambda value: f"ref:{value}")
    frame["reduced_formula"] = frame["formula"]
    frame["chemical_system"] = frame["formula"].map(
        {"NaCl": "Cl-Na", "LiF": "F-Li", "KBr": "Br-K"}
    )
    frame["anonymous_formula"] = "AB"
    return frame


def test_official_split_loader_preserves_and_reports_cross_split_ids(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(
        json.dumps(
            {
                "id_train": ["a", "b"],
                "id_val": ["c"],
                "id_test": ["b", "d"],
            }
        )
    )
    splits = load_official_splits(path)
    assert splits["validation"] == ["c"]
    assert cross_split_id_overlaps(splits)["train_test"] == ["b"]


def test_memory_exclusions_remove_id_and_exact_structure_leakage():
    structures = _structures()
    splits = {
        "train": ["train-same-id", "train-same-structure", "train-safe"],
        "validation": ["train-same-id"],
        "test": ["test-duplicate", "new-system"],
    }
    exclusions = find_memory_exclusions(
        structures, splits, use_structure_matcher=False
    )
    assert set(exclusions["training_material_id"]) == {
        "train-same-id",
        "train-same-structure",
    }
    assert set(exclusions["reason"]) == {
        "same_material_id",
        "same_serialized_structure",
    }


def test_queries_hide_labels_and_mark_predefined_challenge_views():
    structures = _structures()
    spec = TaskSpec("model", "formation_energy_per_atom", "eV/atom")
    queries = query_table(
        "formation_energy_peratom",
        spec,
        "test",
        ["test-duplicate", "new-system"],
        structures,
        {"train-same-id", "train-same-structure", "train-safe"},
    )
    assert "target_value" not in queries.columns
    duplicate = queries.set_index("material_id").loc["test-duplicate"]
    novel = queries.set_index("material_id").loc["new-system"]
    assert not bool(duplicate["challenge_new_reduced_formula"])
    assert bool(novel["challenge_new_reduced_formula"])
    assert bool(novel["challenge_new_chemical_system"])


def test_private_labels_keep_source_provenance():
    episodes = pd.DataFrame(
        [
            {
                "material_id": "test-1",
                "target_name": "formation_energy_per_atom",
                "target_value": -1.0,
                "target_unit": "eV/atom",
                "episode_id": "episode:test-1",
                "protocol": "DFT",
                "source_dataset": "JARVIS-DFT 3D",
                "source_record_id": "test-1",
                "source_revision": "source.json",
            }
        ]
    )
    labels = label_table(
        "formation_energy_peratom", "test", ["test-1"], episodes
    )
    assert labels.iloc[0]["query_id"] == "formation_energy_peratom:test-1"
    assert labels.iloc[0]["source_episode_id"] == "episode:test-1"
