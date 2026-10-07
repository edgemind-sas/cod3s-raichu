"""Independent structural and numerical fault-tree outputs."""

import copy
import math

import pyraichu
import pytest
from test_fault_tree import model, nok, unit


def pair():
    return model(unit("A"), unit("B")), {
        "op": "bool",
        "bool_op": "or",
        "args": [nok("A"), nok("B")],
    }


def test_omitted_extraction_ignores_too_small_budget():
    source, top = pair()
    with pytest.raises(pyraichu.SimulationError, match="cut"):
        pyraichu.fault_tree(source, top, cut_set_limit=1)
    tree = pyraichu.fault_tree(source, top, cut_set_limit=1, cut_sets=False)
    assert tree.minimal_cut_sets is None
    structural = tree.structure()
    assert structural["format"] == "raichu.fault_tree.structure"
    assert structural["version"] == 1
    assert structural["minimal_cut_sets"] is None
    assert structural["cut_sets_omitted"] == "not requested"
    assert "instants" not in structural and "horizon" not in structural
    assert pyraichu.read_fault_tree_structure(structural) == structural
    result = tree.envelope([1000.0], cut_sets=False, cut_set_limit=1)
    assert result["version"] == 1
    assert result["horizon"]["exact"]
    assert result["horizon"]["minimal_cut_sets"] is None
    assert result["horizon"]["probability"] == pytest.approx(1 - math.exp(-2))


def test_default_extraction_and_structural_roundtrip(tmp_path):
    source, top = pair()
    tree = pyraichu.fault_tree(source, top)
    assert tree.minimal_cut_sets == [["A.health.fail"], ["B.health.fail"]]
    structural = tree.structure()
    assert structural["minimal_cut_sets"] == tree.minimal_cut_sets
    assert structural["cut_sets_omitted"] is None
    assert structural["provenance"]["source"]["generation"] == {
        "max_nodes": 1_000_000,
        "profile": [],
    }
    import json

    path = tmp_path / "tree.json"
    path.write_text(json.dumps(structural))
    assert pyraichu.read_fault_tree_structure(path) == structural
    assert pyraichu.read_fault_tree_structure(json.dumps(structural)) == structural


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("format", "unknown", "format"),
        ("version", 2, "version"),
        ("tree", {}, "JSON"),
        ("minimal_cut_sets", [["unknown"]], "unknown"),
    ],
)
def test_structural_reader_refuses_unknown_or_invalid_document(field, value, match):
    source, top = pair()
    structural = copy.deepcopy(pyraichu.fault_tree(source, top).structure())
    structural[field] = value
    with pytest.raises(pyraichu.SimulationError, match=match):
        pyraichu.read_fault_tree_structure(structural)


def test_deep_generated_tree_reads_back_without_an_implicit_depth_cap():
    components = [unit("base")]
    previous = "base"
    for i in range(125):
        side, current = f"side{i}", f"step{i}"
        components.extend(
            [
                unit(side),
                unit(
                    current,
                    guard={
                        "op": "bool",
                        "bool_op": "or",
                        "args": [nok(previous), nok(side)],
                    },
                ),
            ]
        )
        previous = current
    tree = pyraichu.fault_tree(model(*components), nok(previous), cut_sets=False)
    document = tree.structure()
    assert len(tree.basic_events) == 251
    assert pyraichu.read_fault_tree_structure(document) == document
    original_warnings = list(document["tree"]["warnings"])
    document["tree"]["basic_events"][0]["name"] = "changed"
    document["tree"]["warnings"].append("changed")
    assert tree.structure()["tree"]["basic_events"][0]["name"] != "changed"
    assert tree.basic_events[0]["name"] != "changed"
    assert tree.structure()["tree"]["warnings"] == original_warnings


def test_legacy_generation_retains_event_index_order():
    tree = pyraichu.fault_tree(
        model(unit("Z"), unit("A")),
        {"op": "bool", "bool_op": "or", "args": [nok("Z"), nok("A")]},
    )
    assert tree.minimal_cut_sets == [["Z.health.fail"], ["A.health.fail"]]
    assert tree.structure()["minimal_cut_sets"] == [
        ["A.health.fail"],
        ["Z.health.fail"],
    ]


def test_structural_provenance_identifies_model_generation_and_dependencies():
    import hashlib
    import time
    from pathlib import Path

    source, top = pair()
    before = int(time.time() * 1000)
    tree = pyraichu.fault_tree(
        source, top, profile={"A.x": 2.0}, max_nodes=500, cut_sets=False
    )
    provenance = tree.structure()["provenance"]
    assert provenance["source"]["kind"] == "model"
    assert provenance["source"]["model_hash"].startswith("sha256:")
    assert provenance["source"]["generation"]["max_nodes"] == 500
    assert provenance["source"]["generation"]["profile"] == [
        ["A.x", {"kind": "float", "value": 2.0}]
    ]
    assert before <= provenance["generated_at_unix_ms"] <= int(time.time() * 1000)
    lock = Path(__file__).resolve().parents[3] / "Cargo.lock"
    assert (
        provenance["dependency_lock_hash"]
        == "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    assert pyraichu.read_fault_tree_structure(tree.structure()) == tree.structure()
    changed_source = pyraichu.load_model(
        {"name": "other", "components": [unit("A"), unit("B")]}
    )
    changed = pyraichu.fault_tree(changed_source, top, cut_sets=False).structure()
    assert (
        changed["provenance"]["source"]["model_hash"]
        != provenance["source"]["model_hash"]
    )
