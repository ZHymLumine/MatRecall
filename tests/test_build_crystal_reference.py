import json

from matrecall.build_crystal_reference import (
    build_jarvis_tables,
    integrity_report,
)


def test_jarvis_episode_builder_separates_structures_and_scalar_outcomes():
    records = [
        {
            "jid": "JVASP-1",
            "formula": "NaCl",
            "atoms": {
                "lattice_mat": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
                "elements": ["Na", "Cl"],
            },
            "func": "OPT",
            "spg_number": 225,
            "spg_symbol": "Fm-3m",
            "formation_energy_peratom": -1.2,
            "optb88vdw_bandgap": 4.5,
            "mbj_bandgap": "na",
            "bulk_modulus_kv": 25.0,
            "magmom_oszicar": 0.0,
            "poisson": 0.25,
        }
    ]
    structures, episodes = build_jarvis_tables(records, "fixture.json")
    assert len(structures) == 1
    assert len(episodes) == 5
    assert set(episodes["episode_kind"]) == {"reference"}
    assert set(episodes["evidence_type"]) == {"first_principles"}
    assert json.loads(episodes.iloc[0]["context_json"])["space_group_number"] == 225
    assert integrity_report(structures, episodes)["passed"]


def test_nominal_range_is_flagged_without_deleting_source_value():
    records = [
        {
            "jid": "JVASP-2",
            "formula": "X",
            "atoms": {"lattice_mat": [], "coords": [], "elements": []},
            "poisson": 4.2,
        }
    ]
    _, episodes = build_jarvis_tables(records, "fixture.json")
    assert episodes.iloc[0]["target_value"] == 4.2
    assert episodes.iloc[0]["validity_flag"] == "outside_nominal_range"
