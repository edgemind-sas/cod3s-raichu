"""Model-level constructs written in a system document.

A mixed-integer program, an FMU unit and an interface connection each bind
several components, so no type body can carry them. A system document
declares programs and FMU units in the core form, naming instances where
the core names components (an instance becomes the component of the same
name), and interface connections as `Instance.interface`, as it writes
port connections. Each test
rebuilds a flat model of the public suite as a library and a system, and
asks both the same question.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pyraichu
import pytest

from test_interface_connections import BY_INTERFACE, two_way
from test_program import dispatch_model

LIBRARY = {"name": "flat", "version": "1.0.0"}
BODY_KEYS = ("attributes", "ports", "interfaces", "automata", "sensitive_functions", "equations", "allocations")
FIXTURES = Path(__file__).resolve().parents[3] / "crates/raichu-fmi/tests/fixtures/reference-fmus/3.0"


def as_self(value, name):
    """A component's own references, written `$self` as a type body writes them."""
    if isinstance(value, dict):
        return {
            k: ("$self" if k == "component" and v == name else as_self(v, name))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [as_self(v, name) for v in value]
    return value


def as_documents(flat):
    """A flat model body as a library (one type per component) and a system."""
    types = [
        {"name": f"T_{c['name']}", "body": as_self({k: c[k] for k in BODY_KEYS if k in c}, c["name"])}
        for c in flat["components"]
    ]
    system = {
        "name": flat["name"],
        "library": LIBRARY,
        "instances": [{"name": c["name"], "type": f"T_{c['name']}"} for c in flat["components"]],
        "connections": [
            {"from": f"{c['from']['component']}.{c['from']['port']}",
             "to": f"{c['to']['component']}.{c['to']['port']}"}
            for c in flat.get("connections", [])
        ],
        "indicators": flat.get("indicators", []),
        "targets": flat.get("targets", []),
    }
    for key in ("programs", "fmu_units"):
        if key in flat:
            system[key] = copy.deepcopy(flat[key])
    if "interface_connections" in flat:
        system["interface_connections"] = [
            {"from": f"{c['from']['component']}.{c['from']['interface']}",
             "to": f"{c['to']['component']}.{c['to']['interface']}"}
            for c in flat["interface_connections"]
        ]
    library = {"raichu_library": {"format": 1}, "library": {**LIBRARY, "types": types}}
    return library, {"raichu_system": {"format": 1}, "system": system}


def same_run(flat_model, library_model, t_max):
    flat = pyraichu.simulate(flat_model, t_max=t_max)
    built = pyraichu.simulate(library_model, t_max=t_max)
    assert [(e.time, e.transition) for e in built.events] == [(e.time, e.transition) for e in flat.events]
    assert built.indicators == flat.indicators


def test_a_program_declared_in_the_system_dispatches_as_the_flat_one():
    flat = dispatch_model()
    library, system = as_documents(flat)
    expanded = pyraichu.model_body(pyraichu.expand_documents(system, [library])["model"])
    assert expanded["programs"] == flat["programs"]
    same_run(pyraichu.load_model(flat), pyraichu.load_model(system, libraries=[library]), t_max=16)


def test_an_interface_connection_declared_in_the_system_carries_both_directions():
    flat = json.loads(two_way(BY_INTERFACE))
    library, system = as_documents(flat)
    expanded = pyraichu.model_body(pyraichu.expand_documents(system, [library])["model"])
    assert expanded["interface_connections"] == flat["interface_connections"]
    same_run(pyraichu.load_model(flat), pyraichu.load_model(system, libraries=[library]), t_max=1)


def feedthrough(path):
    return {
        "name": "fmi-library",
        "components": [{
            "name": "plant",
            "attributes": [
                {"name": "input", "kind": "float", "init": {"kind": "float", "value": 2.0}},
                {"name": "output", "kind": "float", "init": {"kind": "float", "value": 0.0}},
            ],
        }],
        "indicators": [{"name": "out", "target": "attribute",
                        "attr": {"component": "plant", "attribute": "output"}}],
        "fmu_units": [{
            "name": "feed", "path": path, "step": 0.1,
            "inputs": [{"attribute": {"component": "plant", "attribute": "input"},
                        "variable": "Float64_continuous_input"}],
            "outputs": [{"attribute": {"component": "plant", "attribute": "output"},
                         "variable": "Float64_continuous_output"}],
        }],
    }


def test_an_fmu_unit_declared_in_the_system_runs_next_to_its_document(tmp_path):
    shutil.copy(FIXTURES / "Feedthrough.fmu", tmp_path / "feed.fmu")
    flat = feedthrough("feed.fmu")
    library, system = as_documents(flat)
    (tmp_path / "flat.json").write_text(json.dumps(flat))
    (tmp_path / "system.json").write_text(json.dumps(system))
    built = pyraichu.load_model(tmp_path / "system.json", libraries=[library], allow_fmu_import=True)
    assert built.base_dir == tmp_path
    assert built.allow_fmu_import
    # The path stays as written: it resolves against the document at run time.
    assert pyraichu.model_body(json.loads(built.json))["fmu_units"][0]["path"] == "feed.fmu"
    same_run(pyraichu.load_model(tmp_path / "flat.json", allow_fmu_import=True), built, t_max=0.2)


def test_an_fmu_unit_declared_in_the_system_needs_the_import_permission(tmp_path):
    shutil.copy(FIXTURES / "Feedthrough.fmu", tmp_path / "feed.fmu")
    library, system = as_documents(feedthrough("feed.fmu"))
    with pytest.raises(pyraichu.ModelError, match="feed.*allow_fmu_import"):
        pyraichu.load_model(system, libraries=[library])


def test_a_program_naming_a_missing_instance_is_refused_with_its_name():
    flat = dispatch_model()
    flat["programs"][0]["feasible"] = {"component": "nowhere", "attribute": "feasible"}
    library, system = as_documents(flat)
    with pytest.raises(pyraichu.ModelError, match="nowhere"):
        pyraichu.load_model(system, libraries=[library])


def test_an_interface_connection_written_without_an_instance_is_refused():
    library, system = as_documents(json.loads(two_way(BY_INTERFACE)))
    system["system"]["interface_connections"][0]["from"] = "bus"
    with pytest.raises(pyraichu.ModelError, match="interface_connections"):
        pyraichu.expand_documents(system, [library])


def test_load_system_resolves_relative_fmu_paths_next_to_a_system_file(tmp_path, monkeypatch):
    library, system = as_documents(feedthrough("feed.fmu"))
    (tmp_path / "system.json").write_text(json.dumps(system))
    from_file = pyraichu.library.load_system(tmp_path / "system.json", [library], allow_fmu_import=True)
    assert from_file.base_dir == tmp_path
    monkeypatch.chdir(tmp_path.parent)
    from_dict = pyraichu.library.load_system(system, [library], allow_fmu_import=True)
    assert from_dict.base_dir == tmp_path.parent


def test_a_system_with_a_program_and_no_fmu_unit_needs_no_import_permission():
    library, system = as_documents(dispatch_model())
    assert not pyraichu.load_model(system, libraries=[library]).allow_fmu_import
