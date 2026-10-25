"""The behaviour record and the flat-model loaders the example libraries rest on."""

from __future__ import annotations

import copy

import pyraichu
from library_examples_support import (
    differences,
    example_module,
    page_namespace,
    record,
)


def _two_state(rate: float) -> dict:
    return {
        "name": "unit",
        "components": [
            {
                "name": "pump",
                "attributes": [],
                "automata": [
                    {
                        "name": "health",
                        "states": ["ok", "failed"],
                        "init": "ok",
                        "transitions": [
                            {"name": "fail", "source": "ok", "targets": ["failed"],
                             "distrib": "exp", "rate": rate},
                            {"name": "repair", "source": "failed", "targets": ["ok"],
                             "distrib": "exp", "rate": 1.0},
                        ],
                    }
                ],
            }
        ],
        "indicators": [
            {"name": "down", "target": "state", "component": "pump",
             "automaton": "health", "state": "failed"}
        ],
    }


INSTANTS = [0.0, 5.0, 10.0]


def _record(document: dict, **kwargs) -> dict:
    return record(pyraichu.load_model(document), t_max=10.0, instants=INSTANTS, seed=7, **kwargs)


def test_a_model_recorded_twice_with_one_seed_agrees_with_itself():
    assert differences(_record(_two_state(0.5)), _record(_two_state(0.5))) == []


def test_a_changed_rate_is_named_by_the_comparison():
    lines = differences(_record(_two_state(0.5)), _record(_two_state(0.25)))
    assert lines
    assert any("pump.health.fail" in line or "event" in line for line in lines)


def test_a_rename_map_makes_a_renamed_copy_compare_equal():
    renamed = copy.deepcopy(_two_state(0.5))
    automaton = renamed["components"][0]["automata"][0]
    automaton["name"] = "failure"
    automaton["transitions"][0]["name"] = "occ"
    automaton["transitions"][1]["name"] = "rep"
    renamed["indicators"][0]["automaton"] = "failure"
    observed = _record(
        renamed,
        rename={"pump.failure.occ": "pump.health.fail", "pump.failure.rep": "pump.health.repair"},
    )
    assert differences(_record(_two_state(0.5)), observed) == []


def test_the_page_loader_rebuilds_the_heated_room_model():
    namespace = page_namespace("examples/heated-room.md")
    model = namespace["model"]
    assert isinstance(model, pyraichu.Model)
    import json

    names = {c["name"] for c in pyraichu.model_body(json.loads(model.json))["components"]}
    assert names == {"room", "main", "backup"}


def test_both_example_generators_import_in_one_process():
    gas = example_module("gas_production")
    approdyn = example_module("approdyn")
    assert gas is not approdyn
    assert callable(gas.build) and callable(approdyn.build)


def test_a_page_block_reads_a_committed_example_by_relative_path(tmp_path, monkeypatch):
    from pathlib import Path

    import library_examples_support as support

    page = tmp_path / "page.md"
    page.write_text(
        "```python\n"
        "from pathlib import Path\n"
        "seen = Path('examples/gas_production/model.py').is_file()\n"
        "```\n"
    )
    monkeypatch.setattr(support, "DOCS", tmp_path)
    assert support.page_namespace("page.md")["seen"] is True
    assert Path.cwd() != tmp_path


def test_a_shorter_sample_series_is_a_difference():
    expected = _record(_two_state(0.5))
    observed = _record(_two_state(0.5))
    observed["samples"]["down"] = observed["samples"]["down"][:-1]
    lines = differences(expected, observed)
    assert any("down: expected 3 samples, got 2" in line for line in lines), lines


def test_a_block_nested_in_a_tab_runs_after_the_json_before_it(tmp_path, monkeypatch):
    import library_examples_support as support

    page = tmp_path / "page.md"
    page.write_text(
        '=== "Data"\n\n'
        "    ```json\n"
        '    {"a": 1}\n'
        "    ```\n\n"
        '=== "Code"\n\n'
        "    ```python\n"
        "    nested = True\n"
        "    ```\n\n"
        "```python\n"
        "after = nested\n"
        "```\n"
    )
    monkeypatch.setattr(support, "DOCS", tmp_path)
    namespace = support.page_namespace("page.md")
    assert namespace["after"] is True
