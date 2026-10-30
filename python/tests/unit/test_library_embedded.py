"""The libraries shipped with the engine, from Python."""

import pyraichu
import pytest


def test_the_embedded_std_is_the_standard_library():
    assert pyraichu.embedded_library("std") == pyraichu.standard_library()


def test_an_unknown_embedded_library_is_refused_naming_the_registry():
    with pytest.raises(pyraichu.ModelError, match="std"):
        pyraichu.embedded_library("nowhere")


def test_an_expanded_model_records_the_embedded_versions_it_used():
    library = {"raichu_library": {"format": 1}, "library": {
        "name": "plant", "version": "1.0.0", "imports": ["std"],
        "types": [{"name": "Source", "uses": [
            {"fragment": "std.bool_source_out", "with": {"name": "power"}}]}]}}
    system = {"raichu_system": {"format": 1}, "system": {
        "name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "A", "type": "Source"}]}}
    model = pyraichu.model_body(pyraichu.expand_documents(system, [library])["model"])
    assert model["source"]["embedded"] == {"std": pyraichu.__version__}
