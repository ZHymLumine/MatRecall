from matrecall.static_retrieval import digest
from matrecall.prepare_cross_task_transfer import jointly_disjoint_offsets


def test_cross_task_selection_uses_common_source_disjoint_target_cohort():
    cases = [
        {"query_visible": {"formula": "A"}},
        {"query_visible": {"formula": "B"}},
        {"query_visible": {"formula": "C"}},
    ]
    index = [
        {"transfer_group_id": "chemical_system:A-O"},
        {"transfer_group_id": "chemical_system:B-O"},
        {"transfer_group_id": "chemical_system:C-O"},
    ]
    sources = {
        "related": {
            "groups": {"chemical_system:A-O"},
            "group_namespaces": ["chemical_system"],
            "query_hashes": set(),
        },
        "unrelated": {
            "groups": set(),
            "group_namespaces": ["chemical_system"],
            "query_hashes": {digest({"formula": "B"})},
        },
    }
    offsets, group_exclusions, query_exclusions = jointly_disjoint_offsets(
        cases, index, sources, "chemical_system"
    )
    assert offsets == [2]
    assert group_exclusions == 1
    assert query_exclusions == 1


def test_noncomparable_source_namespace_does_not_create_false_group_overlap():
    cases = [{"query_visible": {"target_formula": "A"}}]
    index = [{"transfer_group_id": "doi:example"}]
    sources = {
        "related": {
            "groups": set(), "group_namespaces": ["doi"], "query_hashes": set(),
        },
        "unrelated": {
            "groups": {"chemical_system:A"},
            "group_namespaces": ["chemical_system"], "query_hashes": set(),
        },
    }
    offsets, group_exclusions, query_exclusions = jointly_disjoint_offsets(
        cases, index, sources, "doi"
    )
    assert offsets == [0]
    assert group_exclusions == 0
    assert query_exclusions == 0
