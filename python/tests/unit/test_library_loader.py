"""The component type library loader: library, system and study documents
expanded by the engine and loaded from Python (unit U7)."""

from __future__ import annotations

import json

import pytest

import pyraichu
from pyraichu import ModelError, library


def lamp_library() -> dict:
    return {
        "raichu_library": {"format": 1},
        "library": {
            "name": "home",
            "version": "1.0.0",
            "imports": ["std"],
            "types": [
                {
                    "name": "Switch",
                    "uses": [{"fragment": "std.bool_source_out", "with": {"name": "power"}}],
                    "modes": [
                        {
                            "name": "stuck_off",
                            "nature": "failure",
                            "effects": [{"attribute": "power_fed_available_out", "value": False}],
                        }
                    ],
                },
                {
                    "name": "Lamp",
                    "uses": [
                        {"fragment": "std.bool_flow_in", "with": {"name": "power"}},
                        {
                            "fragment": "std.bool_flow_out",
                            "with": {"name": "light", "prod_cond": [["power_fed_in"]]},
                        },
                    ],
                },
            ],
        },
    }


def room_system(version: str = "1.0.0") -> dict:
    return {
        "raichu_system": {"format": 1},
        "system": {
            "name": "room",
            "library": {"name": "home", "version": version},
            "instances": [{"name": "S", "type": "Switch"}, {"name": "L", "type": "Lamp"}],
            "connections": [{"from": "S.power_out", "to": "L.power_in"}],
            "indicators": [
                {
                    "name": "lit",
                    "target": "attribute",
                    "attr": {"component": "L", "attribute": "light_fed_out"},
                }
            ],
        },
    }


def room_study() -> dict:
    return {
        "raichu_study": {"format": 1},
        "study": {
            "name": "outage",
            "system": "room",
            "modes": [
                {
                    "mode": "stuck_off",
                    "instances": ["S"],
                    "occurrence": {"distrib": "delay", "time": 2.0},
                }
            ],
            "feared_events": [
                {
                    "name": "dark",
                    "target": True,
                    "condition": {
                        "op": "cmp",
                        "cmp": "eq",
                        "lhs": {"op": "attr", "attr": {"component": "L", "attribute": "light_fed_out"}},
                        "rhs": {"op": "const", "value": {"kind": "bool", "value": False}},
                    },
                }
            ],
            "simulation": {"method": "monte_carlo", "nb_runs": 3, "horizon": 5.0, "instants": [1.0, 3.0]},
        },
    }


def test_a_yaml_library_a_json_system_and_a_dict_study_load_and_run(tmp_path):
    yaml = pytest.importorskip("yaml")
    library_path = tmp_path / "home.yaml"
    library_path.write_text(yaml.safe_dump(lamp_library()), encoding="utf-8")
    system_path = tmp_path / "room.json"
    system_path.write_text(json.dumps(room_system()), encoding="utf-8")
    model = pyraichu.load_system(system_path, [library_path], room_study())
    assert model.name == "room/outage"
    result = pyraichu.simulate(model, t_max=5.0)
    assert result.indicators["lit"] == [(0.0, True), (2.0, False)]


def test_a_library_version_mismatch_names_both_versions():
    with pytest.raises(ModelError, match=r"`2\.0\.0`.*`1\.0\.0`"):
        pyraichu.load_system(room_system("2.0.0"), [lamp_library()])


def test_load_model_routes_a_system_document_to_the_library_loader():
    model = pyraichu.load_model(room_system(), libraries=[lamp_library()])
    assert model.name == "room"
    assert model.provenance["attribute:L.light_fed_out"]["fragment"] == "std.bool_flow_out"


def test_a_flat_model_still_loads_and_runs_unchanged():
    flat = {
        "name": "flat",
        "components": [
            {
                "name": "A",
                "attributes": [{"name": "x", "kind": "bool", "init": {"kind": "bool", "value": True}}],
                "automata": [
                    {
                        "name": "life",
                        "states": ["ok", "ko"],
                        "init": "ok",
                        "transitions": [
                            {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "delay", "time": 1.0}
                        ],
                    }
                ],
            }
        ],
        "indicators": [{"name": "ko", "target": "state", "component": "A", "automaton": "life", "state": "ko"}],
    }
    model = pyraichu.load_model(flat)
    assert model.provenance is None
    assert pyraichu.simulate(model, t_max=2.0).indicators["ko"] == [(0.0, False), (1.0, True)]


def test_provenance_maps_a_public_mode_name_back_to_its_activation():
    model = pyraichu.load_system(room_system(), [lamp_library()], room_study())
    origin = model.provenance["automaton:S.stuck_off"]
    assert origin["document"] == "study"
    assert origin["pointer"] == "/study/modes/0"
    assert origin["nature"] == "failure"


def test_a_refused_document_raises_with_its_pointer():
    system = room_system()
    system["system"]["instances"][0]["typo"] = 1
    with pytest.raises(ModelError, match=r"/system/instances/0/typo"):
        pyraichu.load_system(system, [lamp_library()])


def test_run_study_uses_the_study_settings():
    model = pyraichu.load_system(room_system(), [lamp_library()], room_study())
    assert model.simulation["nb_runs"] == 3
    estimates = pyraichu.run_study(model)
    assert estimates is not None


def test_run_study_without_settings_is_refused():
    model = pyraichu.load_system(room_system(), [lamp_library()])
    with pytest.raises(ModelError, match="no run settings"):
        pyraichu.run_study(model)


def test_the_standard_library_and_check_library_are_reachable():
    std = library.standard_library()
    assert std["library"]["name"] == "std"
    assert std["library"]["version"] == pyraichu.__version__
    report = pyraichu.check_library(lamp_library())
    assert [entry["status"] for entry in report] == ["valid", "valid"]


def test_an_unquoted_yaml_boolean_name_reads_as_a_boolean(tmp_path):
    yaml = pytest.importorskip("yaml")
    doc = lamp_library()
    doc["library"]["types"][0]["modes"][0]["name"] = "on"
    path = tmp_path / "lib.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    # safe_dump quotes the name, so it survives the round trip; an author
    # writing `name: on` by hand gets the boolean `True` instead.
    assert library.read_document(path)["library"]["types"][0]["modes"][0]["name"] == "on"
    assert yaml.safe_load("name: on")["name"] is True


def test_load_model_refuses_libraries_or_a_study_for_another_document():
    flat = {"name": "flat", "components": []}
    with pytest.raises(ModelError, match="raichu_system"):
        pyraichu.load_model(flat, libraries=[lamp_library()])
    with pytest.raises(ModelError, match="raichu_system"):
        pyraichu.load_model(flat, study=room_study())
