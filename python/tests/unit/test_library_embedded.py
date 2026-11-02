"""The libraries shipped with the engine, from Python."""

import pyraichu
import pytest


def test_the_embedded_std_is_the_standard_library():
    assert pyraichu.embedded_library("std") == pyraichu.standard_library()


def test_the_muscadet_library_is_embedded_and_imports_nothing():
    muscadet = pyraichu.embedded_library("muscadet")
    assert muscadet["library"]["name"] == "muscadet"
    assert not muscadet["library"].get("imports")


def test_an_unknown_embedded_library_is_refused_naming_the_registry():
    with pytest.raises(pyraichu.ModelError, match="std, muscadet"):
        pyraichu.embedded_library("nowhere")


def plant(library: str) -> dict:
    return {"raichu_library": {"format": 1}, "library": {
        "name": "plant", "version": "1.0.0", "imports": [library],
        "types": [{"name": "Source", "uses": [
            {"fragment": f"{library}.bool_source_out", "with": {"name": "power"}}]}]}}


SYSTEM = {"raichu_system": {"format": 1}, "system": {
    "name": "s", "library": {"name": "plant", "version": "1.0.0"},
    "instances": [{"name": "A", "type": "Source"}]}}


@pytest.mark.parametrize(("library", "recorded"), [
    ("muscadet", {"muscadet"}),
    # `std` forwards the name to `muscadet`, so both supplied the model.
    ("std", {"std", "muscadet"}),
])
def test_an_expanded_model_records_the_embedded_versions_it_used(library, recorded):
    model = pyraichu.model_body(pyraichu.expand_documents(SYSTEM, [plant(library)])["model"])
    assert model["source"]["embedded"] == dict.fromkeys(recorded, pyraichu.__version__)
