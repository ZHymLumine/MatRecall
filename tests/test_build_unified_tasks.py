from __future__ import annotations

from pathlib import Path

import pandas as pd

from matrecall.build_unified_tasks import (
    FECONI_ELEMENTS,
    HEA_ELEMENTS,
    build_feconi,
    build_hea,
    build_pcmsp,
    build_sg_predict,
    parse_webanno_tsv,
)


def _assert_hidden_labels(task_dir: Path) -> None:
    for split in ("validation", "test"):
        queries = pd.read_parquet(task_dir / "public" / f"{split}_queries.parquet")
        labels = pd.read_parquet(task_dir / "private" / f"{split}_labels.parquet")
        assert "target_json" not in queries
        assert queries["query_id"].is_unique
        assert labels["query_id"].is_unique
        assert set(queries["query_id"]) == set(labels["query_id"])


def test_webanno_parser_preserves_typed_nodes_and_relation_direction(tmp_path):
    source = tmp_path / "example.tsv"
    source.write_text(
        """#FORMAT=WebAnno TSV 3.3
#T_SP=webanno.custom.DemoNERRE|NamedentityTags
#T_RL=webanno.custom.DemoRE|NamedentityRelations|BT_webanno.custom.DemoNERRE

#Text=The powder was heated at 900 C with a 2\\\x1c scan.
1-1\t0-3\tThe\t_\t_\t
1-2\t4-10\tpowder\tMaterial-intermedium\t_\t_\t
1-3\t11-14\twas\t_\t_\t
1-4\t15-21\theated\tOperation\tParticipant-material\t1-2\t
1-5\t22-24\tat\t_\t_\t
1-6\t25-28\t900\tProperty-temperature[1]\tCondition-of\t1-4\t
1-7\t29-30\tC\tProperty-temperature[1]\t_\t_\t
1-8\t30-31\t.\t_\t_\t

""",
        encoding="utf-8",
    )
    records = parse_webanno_tsv(source)
    assert len(records) == 1
    assert "\x1c" in records[0]["text"]
    assert [node["type"] for node in records[0]["nodes"]] == [
        "Material-intermedium",
        "Operation",
        "Property-temperature",
    ]
    assert records[0]["nodes"][2]["text"] == "900 C"
    assert records[0]["edges"] == [
        {"source": "n0", "target": "n1", "type": "Participant-material"},
        {"source": "n1", "target": "n2", "type": "Condition-of"},
    ]


def test_sg_predict_builder_groups_identical_compositions(tmp_path):
    source = tmp_path / "cubic.csv"
    rows = []
    for index in range(60):
        a = index + 1
        rows.append(
            {
                "mp-id": f"mp-{index}",
                "space_group": 195 + index % 3,
                "formula": f"Na{a}Cl1",
                "full_formula": f"Na{a}Cl1",
                "composition": f"Na{a} Cl1",
            }
        )
    # One composition has two admissible space-group labels.
    rows.append({**rows[0], "mp-id": "mp-conflict", "space_group": 225})
    pd.DataFrame(rows).to_csv(source, index=False)

    task_dir = tmp_path / "task"
    manifest = build_sg_predict(source, task_dir, "test-revision")
    membership = pd.read_parquet(task_dir / "audit" / "split_membership.parquet")
    assert int(membership.groupby("input_signature")["split"].nunique().max()) == 1
    assert manifest["counts"]["conflicting_compositions"] == 1
    _assert_hidden_labels(task_dir)


def _write_pcmsp_document(path: Path, temperature: int) -> None:
    path.write_text(
        f"""#FORMAT=WebAnno TSV 3.3
#T_SP=webanno.custom.DemoNERRE|NamedentityTags
#T_RL=webanno.custom.DemoRE|NamedentityRelations|BT_webanno.custom.DemoNERRE

#Text=The powder was heated at {temperature} C.
1-1\t0-3\tThe\t_\t_\t
1-2\t4-10\tpowder\tMaterial-intermedium\t_\t_\t
1-3\t11-14\twas\t_\t_\t
1-4\t15-21\theated\tOperation\tParticipant-material\t1-2\t
1-5\t22-24\tat\t_\t_\t
1-6\t25-28\t{temperature}\tProperty-temperature[1]\tCondition-of\t1-4\t
1-7\t29-30\tC\tProperty-temperature[1]\t_\t_\t
1-8\t30-31\t.\t_\t_\t

""",
        encoding="utf-8",
    )


def test_pcmsp_builder_uses_document_split_and_hides_graphs(tmp_path):
    root = tmp_path / "original"
    for name in ("train", "dev", "test"):
        (root / name).mkdir(parents=True)
    for index in range(4):
        _write_pcmsp_document(root / "train" / f"train-{index}.tsv", 800 + index)
    _write_pcmsp_document(root / "dev" / "dev.tsv", 900)
    # The public release has two document IDs in both train and test.  The
    # builder must preserve the evaluation copy and exclude the train copy.
    _write_pcmsp_document(root / "test" / "train-0.tsv", 1000)

    task_dir = tmp_path / "task"
    manifest = build_pcmsp(root, task_dir, "test-revision")
    membership = pd.read_parquet(task_dir / "audit" / "split_membership.parquet")
    assert int(membership.groupby("document_id")["split"].nunique().max()) == 2
    assert manifest["counts"]["source_files"] == 6
    assert manifest["counts"]["unique_documents"] == 5
    assert manifest["counts"]["memory_document_overlap_exclusions"] == 1
    memory = pd.read_parquet(task_dir / "public" / "memory_episodes.parquet")
    assert "train-0" not in set(memory["document_id"])
    _assert_hidden_labels(task_dir)


def test_feconi_builder_connects_composition_and_property_duplicates(tmp_path):
    raw_rows = []
    imputed_rows = []
    for index in range(60):
        row = {element: 0.0 for element in FECONI_ELEMENTS}
        row.update({"Fe": float(index + 1), "Co": float(61 - index), "Ni": 40.0})
        row.update(
            {
                "Ms": 1.0 + index / 100,
                "Hc": 10.0 + index,
                "Tc": 700.0 + index,
                "ER": 8.0 + index / 10,
                "Elong": 20.0 + index / 10,
                "HV": 100.0 + index,
                "Cost": 2.0 + index / 20,
            }
        )
        imputed_rows.append(dict(row))
        raw_row = dict(row)
        if index % 2:
            raw_row["Hc"] = None
        raw_rows.append(raw_row)
    # An exact composition duplicate must stay in one split.
    duplicate = dict(imputed_rows[0])
    duplicate["Ms"] = 2.5
    imputed_rows.append(duplicate)
    raw_rows.append(dict(duplicate))
    imputed = tmp_path / "imputed.csv"
    raw = tmp_path / "raw.csv"
    pd.DataFrame(imputed_rows).to_csv(imputed, index=False)
    pd.DataFrame(raw_rows).to_csv(raw, index=False)

    task_dir = tmp_path / "task"
    build_feconi(imputed, raw, task_dir, "test-revision")
    membership = pd.read_parquet(task_dir / "audit" / "split_membership.parquet")
    assert int(membership.groupby("composition_signature")["split"].nunique().max()) == 1
    assert int(membership.groupby("property_signature")["split"].nunique().max()) == 1
    _assert_hidden_labels(task_dir)


def test_hea_builder_uses_set_valued_conflict_labels(tmp_path):
    rows = []
    for index in range(60):
        row = {element: 0.0 for element in HEA_ELEMENTS}
        row.update(
            {
                "index": f"Fe{index + 1}Co{61 - index}",
                "Fe": float(index + 1),
                "Co": float(61 - index),
                "Phase": "BCC" if index % 2 else "FCC+IM",
            }
        )
        rows.append(row)
    conflict = dict(rows[0])
    conflict["Phase"] = "BCC+FCC"
    rows.append(conflict)
    # The released source table omits several late-period element columns.
    # The reported formula remains parseable and is the authoritative input.
    omitted_element_row = {element: 0.0 for element in HEA_ELEMENTS}
    omitted_element_row.update({"index": "Au2Bi", "Phase": "IM"})
    rows.append(omitted_element_row)
    source = tmp_path / "hea.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    task_dir = tmp_path / "task"
    manifest = build_hea(source, task_dir, "test-revision")
    membership = pd.read_parquet(task_dir / "audit" / "split_membership.parquet")
    assert int(membership.groupby("composition_signature")["split"].nunique().max()) == 1
    assert manifest["counts"]["conflicting_compositions"] == 1
    assert manifest["counts"]["source_composition_discrepancies"] == 1
    assert manifest["counts"]["source_columns_omitted_elements"] == ["Au", "Bi"]
    discrepancies = pd.read_parquet(
        task_dir / "audit" / "source_composition_discrepancies.parquet"
    )
    assert discrepancies.iloc[0]["reported_formula"] == "Au2Bi"
    _assert_hidden_labels(task_dir)

