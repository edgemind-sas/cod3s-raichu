"""Public FMI import entry-point tests using Reference FMU fixtures."""

import json
from pathlib import Path

import pyraichu
import pytest


def _document() -> dict:
    return {
        "name": "fmi-permission",
        "components": [],
        "fmu_units": [
            {"name": "plant", "path": "plant.fmu", "step": 0.1}
        ],
    }


def test_fmu_import_requires_loader_permission() -> None:
    with pytest.raises(pyraichu.ModelError, match="plant.*allow_fmu_import"):
        pyraichu.load_model(_document())


def test_fmu_import_requires_permission_after_plugin_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {"name": "generated-fmu", "components": [], "plugins": []}
    monkeypatch.setattr(pyraichu, "expand_model", lambda _source: _document())

    with pytest.raises(pyraichu.ModelError, match="plant.*allow_fmu_import"):
        pyraichu.load_model(source)


def test_fmu_loader_seals_feature_and_keeps_base_directory(tmp_path: Path) -> None:
    document_path = tmp_path / "model.json"
    document_path.write_text(json.dumps(_document()), encoding="utf-8")

    model = pyraichu.load_model(document_path, allow_fmu_import=True)

    assert model.base_dir == tmp_path
    assert model.allow_fmu_import
    assert "fmi" in json.loads(model.json)["raichu_model"]["requires"]


def test_feedthrough_runs_and_replays_interactively() -> None:
    fixture = (
        Path(__file__).resolve().parents[3]
        / "crates/raichu-fmi/tests/fixtures/reference-fmus/3.0/Feedthrough.fmu"
    )
    document = {
        "name": "fmi-python",
        "components": [{
            "name": "plant",
            "attributes": [
                {"name": "input", "kind": "float", "init": {"kind": "float", "value": 2.0}},
                {"name": "output", "kind": "float", "init": {"kind": "float", "value": 0.0}},
            ],
        }],
        "fmu_units": [{
            "name": "feed", "path": str(fixture), "step": 0.1,
            "inputs": [{
                "attribute": {"component": "plant", "attribute": "input"},
                "variable": "Float64_continuous_input",
            }],
            "outputs": [{
                "attribute": {"component": "plant", "attribute": "output"},
                "variable": "Float64_continuous_output",
            }],
        }],
    }
    model = pyraichu.load_model(document, allow_fmu_import=True)
    with pytest.raises(pyraichu.SimulationError, match="finite"):
        pyraichu.simulate(model)
    batch = pyraichu.simulate(model, t_max=0.2)
    assert len(batch.provenance["fmu_units"]) == 1
    assert batch.provenance["fmu_units"][0]["content_hash"].startswith("sha256:")
    estimates = pyraichu.monte_carlo(model, nb_runs=2, t_max=0.2, samples=[0.1, 0.2])
    assert estimates.fmu_units == batch.provenance["fmu_units"]

    session = pyraichu.interactive(model, t_max=0.2)
    first = session.step()
    assert first is not None and first.time == pytest.approx(0.1)
    first_value = session.attribute("plant.output")
    snap = session.snapshot()
    second = session.step()
    assert second is not None and second.time == pytest.approx(0.2)
    session.restore(snap)
    assert session.attribute("plant.output") == first_value
    replay = session.step()
    assert replay is not None and replay.time == pytest.approx(second.time)
    assert session.attribute("plant.output") == pytest.approx(2.0)


def test_discretised_exploration_uses_fmu_snapshots() -> None:
    fixture = (
        Path(__file__).resolve().parents[3]
        / "crates/raichu-fmi/tests/fixtures/reference-fmus/3.0/Dahlquist.fmu"
    )
    document = {
        "name": "fmi-exploration-python",
        "components": [
            {"name": "plant", "attributes": [
                {"name": "x", "kind": "float", "init": {"kind": "float", "value": 1.0}},
            ]},
            {"name": "sys", "automata": [{
                "name": "watch", "states": ["ok", "down"], "init": "ok",
                "transitions": [{
                    "name": "down", "source": "ok", "targets": ["down"],
                    "distrib": "inst", "probs": [],
                    "guard": {
                        "op": "cmp", "cmp": "le",
                        "lhs": {"op": "attr", "attr": {"component": "plant", "attribute": "x"}},
                        "rhs": {"op": "const", "value": {"kind": "float", "value": 0.7}},
                    },
                }],
            }]},
        ],
        "targets": [{"name": "down", "component": "sys", "automaton": "watch", "state": "down"}],
        "fmu_units": [{
            "name": "physics", "path": str(fixture), "step": 0.1,
            "outputs": [{
                "attribute": {"component": "plant", "attribute": "x"},
                "variable": "x",
            }],
        }],
    }
    model = pyraichu.load_model(document, allow_fmu_import=True)
    result = pyraichu.explore(model, "down", 1.0, algorithm="discretised", level=2, refine=False)
    assert result.lower == result.upper == 1.0
