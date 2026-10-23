"""The schema witness (R24, U9): the published JSON Schemas validate the
witness documents, and malformed documents are refused by the loader with
a message naming the fault; the structural faults are refused by the
schema as well."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import pyraichu
from pyraichu import ModelError, library

jsonschema = pytest.importorskip("jsonschema")

REPO = Path(__file__).resolve().parents[3]
PUBLISHED = REPO / "docs" / "reference" / "schema"
CYBER = Path(__file__).parent / "library_fixtures" / "cyber_3comp"
KINDS = ("library", "system", "study")


def validator(kind: str):
    return jsonschema.Draft202012Validator(library.schema(kind))


def cyber() -> dict:
    return {kind: json.loads((CYBER / f"{kind}.json").read_text()) for kind in KINDS}


@pytest.mark.parametrize("kind", KINDS)
def test_the_packaged_schema_is_the_published_one(kind):
    published = json.loads((PUBLISHED / f"raichu-{kind}.schema.json").read_text())
    assert library.schema(kind) == published


@pytest.mark.parametrize("kind", KINDS)
def test_the_schemas_are_valid_draft_2020_12(kind):
    jsonschema.Draft202012Validator.check_schema(library.schema(kind))


@pytest.mark.parametrize("kind", KINDS)
def test_the_cyber_witness_documents_validate(kind):
    errors = list(validator(kind).iter_errors(cyber()[kind]))
    assert not errors, [e.message for e in errors[:3]]


def test_the_standard_library_validates():
    errors = list(validator("library").iter_errors(pyraichu.standard_library()))
    assert not errors, [e.message for e in errors[:3]]


def mutate(path_kind: str, edit):
    documents = copy.deepcopy(cyber())
    edit(documents[path_kind])
    return documents


def library_types(doc):
    return doc["library"]["types"]


#: (case, document kind, mutation, message fragment, refused by the schema)
CASES = [
    ("unknown field", "system", lambda d: d["system"].update({"instancez": []}), "instancez", True),
    ("wrong envelope", "study", lambda d: d.update({"raichu_model": d.pop("raichu_study")}), "raichu_study", True),
    (
        "misspelled key inside a type body",
        "library",
        lambda d: library_types(d)[0].update(
            {"body": {"attributes": [{"name": "x", "kind": "bool", "init": {"kind": "bool", "value": True}, "intt": 1}]}}
        ),
        "intt",
        True,
    ),
    ("name with a dot", "system", lambda d: d["system"]["instances"][0].update({"name": "Al.im"}), "`.`", False),
    (
        "unknown enum value",
        "library",
        lambda d: library_types(d)[1]["uses"][0]["with"].update({"logic": "xor"}),
        "`xor`",
        False,
    ),
    (
        "missing required parameter",
        "library",
        lambda d: library_types(d)[0]["uses"][0]["with"].pop("name"),
        "required parameter `name`",
        False,
    ),
    (
        "mode without rule",
        "library",
        lambda d: library_types(d)[1]["modes"][1]["effects"].append({"attribute": "power_fed_in", "value": True}),
        "declares no combination rule",
        False,
    ),
    (
        "rule and kind mismatch",
        "library",
        lambda d: library_types(d)[1]["modes"][1]["effects"].__setitem__(
            0, {"attribute": "f_service_prod_available", "value": 1.5}
        ),
        "`any_activates`",
        False,
    ),
    ("version mismatch", "system", lambda d: d["system"]["library"].update({"version": "9.9.9"}), "`9.9.9`", False),
    (
        "condition on an inactive mode",
        "study",
        lambda d: d["study"]["modes"].pop(0),
        "which the study does not activate",
        False,
    ),
    (
        "duplicate fragment attribute",
        "library",
        lambda d: library_types(d)[0]["uses"].append({"fragment": "std.bool_source_out", "with": {"name": "power"}}),
        "declared by both",
        False,
    ),
]


@pytest.mark.parametrize("case, kind, edit, fragment, by_schema", CASES, ids=[c[0] for c in CASES])
def test_a_malformed_document_is_refused_naming_the_fault(case, kind, edit, fragment, by_schema):
    documents = mutate(kind, edit)
    with pytest.raises(ModelError) as refused:
        pyraichu.load_system(documents["system"], [documents["library"]], documents["study"])
    assert fragment in str(refused.value), str(refused.value)
    if by_schema:
        assert list(validator(kind).iter_errors(documents[kind])), f"the schema accepts: {case}"
