"""Independent FMI 3 execution and RAICHU import of exported models."""

import os
import subprocess
from pathlib import Path

import pyraichu
import pytest
from pyraichu import fmi


def _hybrid_model() -> pyraichu.Model:
    return pyraichu.load_model(
        {
            "name": "hybrid-export",
            "components": [
                {
                    "name": "plant",
                    "attributes": [
                        {
                            "name": "x",
                            "kind": "float",
                            "init": {"kind": "float", "value": 0.0},
                        },
                        {
                            "name": "failed",
                            "kind": "bool",
                            "init": {"kind": "bool", "value": False},
                        },
                        {
                            "name": "failure_date",
                            "kind": "float",
                            "init": {"kind": "float", "value": 0.0},
                        },
                    ],
                    "equations": [
                        {
                            "target": "x",
                            "kind": "ode",
                            "expr": {
                                "op": "const",
                                "value": {"kind": "float", "value": 1.0},
                            },
                        }
                    ],
                    "automata": [
                        {
                            "name": "failure",
                            "states": ["ok", "down"],
                            "init": "ok",
                            "transitions": [
                                {
                                    "name": "fail",
                                    "source": "ok",
                                    "targets": ["down"],
                                    "distrib": "exp",
                                    "rate": 1.0,
                                    "effects": [
                                        {
                                            "target": {
                                                "component": "plant",
                                                "attribute": "failed",
                                            },
                                            "value": {
                                                "op": "const",
                                                "value": {
                                                    "kind": "bool",
                                                    "value": True,
                                                },
                                            },
                                        },
                                        {
                                            "target": {
                                                "component": "plant",
                                                "attribute": "failure_date",
                                            },
                                            "value": {"op": "time"},
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def _delay_model() -> pyraichu.Model:
    return pyraichu.load_model(
        {
            "name": "delay-export",
            "components": [
                {
                    "name": "plant",
                    "attributes": [
                        {
                            "name": "alarm",
                            "kind": "bool",
                            "init": {"kind": "bool", "value": False},
                        },
                        {
                            "name": "alarm_date",
                            "kind": "float",
                            "init": {"kind": "float", "value": 0.0},
                        },
                    ],
                    "automata": [
                        {
                            "name": "clock",
                            "states": ["idle", "fired"],
                            "init": "idle",
                            "transitions": [
                                {
                                    "name": "fire",
                                    "source": "idle",
                                    "targets": ["fired"],
                                    "distrib": "delay",
                                    "time": 0.6,
                                    "effects": [
                                        {
                                            "target": {
                                                "component": "plant",
                                                "attribute": "alarm",
                                            },
                                            "value": {
                                                "op": "const",
                                                "value": {
                                                    "kind": "bool",
                                                    "value": True,
                                                },
                                            },
                                        },
                                        {
                                            "target": {
                                                "component": "plant",
                                                "attribute": "alarm_date",
                                            },
                                            "value": {"op": "time"},
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


@pytest.fixture(scope="module")
def independent_importer():
    fmpy = pytest.importorskip("fmpy")
    return fmpy


def _validate(path: Path, fmpy) -> None:
    from fmpy.validation import validate_fmu

    assert validate_fmu(str(path)) == []
    if executable := os.environ.get("RAICHU_FMUSIM"):
        subprocess.run([executable, "validate", str(path)], check=True)


def _series(path: Path, fmpy, outputs: list[str], seed: int = 0):
    return fmpy.simulate_fmu(
        str(path),
        start_time=0,
        stop_time=2,
        step_size=0.5,
        output_interval=0.5,
        output=outputs,
        start_values={"seed": seed},
        validate=True,
    )


def test_delay_fmu_matches_native_points_and_event_date(
    tmp_path: Path,
    independent_importer,
) -> None:
    model = _delay_model()
    path = fmi.export(
        model, tmp_path / "delay.fmu", outputs=["plant.alarm", "plant.alarm_date"]
    )
    _validate(path, independent_importer)
    series = _series(path, independent_importer, ["plant.alarm", "plant.alarm_date"])
    native = pyraichu.interactive(model, t_max=2)
    for row in series:
        native.advance_to(float(row["time"]))
        assert bool(row["plant.alarm"]) == native.attribute("plant.alarm")
        assert float(row["plant.alarm_date"]) == native.attribute("plant.alarm_date")
    assert [event.time for event in native.history()] == [0.6]
    assert bool(series[1]["plant.alarm"]) is False
    assert bool(series[2]["plant.alarm"]) is True


def test_fmpy_drives_exported_input_at_communication_points(
    tmp_path: Path,
    independent_importer,
) -> None:
    model = pyraichu.load_model(
        {
            "name": "driven-export",
            "components": [
                {
                    "name": "plant",
                    "attributes": [
                        {"name": "drive", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                        {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                    ],
                    "equations": [
                        {
                            "target": "x",
                            "kind": "ode",
                            "expr": {
                                "op": "attr",
                                "attr": {"component": "plant", "attribute": "drive"},
                            },
                        }
                    ],
                }
            ],
        }
    )
    path = fmi.export(
        model,
        tmp_path / "driven.fmu",
        inputs=["plant.drive"],
        outputs=["plant.x"],
    )
    _validate(path, independent_importer)
    description = independent_importer.read_model_description(str(path))
    variables = {variable.name: variable.valueReference for variable in description.modelVariables}
    extracted = independent_importer.extract(str(path), unzipdir=str(tmp_path / "extracted"))
    from fmpy.fmi3 import FMU3Slave

    slave = FMU3Slave(
        guid=description.guid,
        modelIdentifier=description.coSimulation.modelIdentifier,
        unzipDirectory=extracted,
        instanceName="driven",
    )
    native = pyraichu.interactive(model, t_max=2)
    try:
        slave.instantiate()
        slave.enterInitializationMode(startTime=0.0, stopTime=2.0)
        slave.exitInitializationMode()
        for time, drive in ((0.0, 1.0), (0.5, 2.0), (1.0, -1.0), (1.5, 0.0)):
            slave.setFloat64([variables["plant.drive"]], [drive])
            native.set_input("plant.drive", drive, ["plant.drive"])
            slave.doStep(currentCommunicationPoint=time, communicationStepSize=0.5)
            native.advance_to(time + 0.5)
            assert slave.getFloat64([variables["plant.x"]])[0] == pytest.approx(
                native.attribute("plant.x"), abs=1e-7, rel=1e-7
            )
        slave.terminate()
    finally:
        slave.freeInstance()


def test_hybrid_fmu_replays_and_round_trips(
    tmp_path: Path,
    independent_importer,
) -> None:
    model = _hybrid_model()
    path = fmi.export(
        model,
        tmp_path / "hybrid.fmu",
        outputs=[
            "plant.failed",
            "plant.x",
            "plant.failure_date",
        ],
    )
    _validate(path, independent_importer)
    outputs = ["plant.failed", "plant.x", "plant.failure_date"]
    first = _series(path, independent_importer, outputs, 42)
    second = _series(path, independent_importer, outputs, 42)
    third = _series(path, independent_importer, outputs, 43)
    assert first.tobytes() == second.tobytes()
    assert first["plant.failed"].tobytes() != third["plant.failed"].tobytes()

    native = pyraichu.interactive(model, t_max=2, seed=42)
    for row in first:
        native.advance_to(float(row["time"]))
        assert bool(row["plant.failed"]) == native.attribute("plant.failed")
        assert float(row["plant.failure_date"]) == native.attribute(
            "plant.failure_date"
        )
        assert float(row["plant.x"]) == pytest.approx(
            native.attribute("plant.x"),
            abs=1e-7,
            rel=1e-7,
        )

    imported = pyraichu.load_model(
        {
            "name": "export-roundtrip",
            "components": [
                {
                    "name": "sink",
                    "attributes": [
                        {
                            "name": "failed",
                            "kind": "bool",
                            "init": {"kind": "bool", "value": False},
                        },
                        {
                            "name": "x",
                            "kind": "float",
                            "init": {"kind": "float", "value": 0.0},
                        },
                        {
                            "name": "failure_date",
                            "kind": "float",
                            "init": {"kind": "float", "value": 0.0},
                        },
                    ],
                }
            ],
            "indicators": [
                {
                    "name": "failed",
                    "target": "attribute",
                    "attr": {"component": "sink", "attribute": "failed"},
                },
                {
                    "name": "x",
                    "target": "attribute",
                    "attr": {"component": "sink", "attribute": "x"},
                },
                {
                    "name": "failure_date",
                    "target": "attribute",
                    "attr": {"component": "sink", "attribute": "failure_date"},
                },
            ],
            "fmu_units": [
                {
                    "name": "exported",
                    "path": str(path),
                    "step": 0.5,
                    "outputs": [
                        {
                            "attribute": {"component": "sink", "attribute": "failed"},
                            "variable": "plant.failed",
                        },
                        {
                            "attribute": {"component": "sink", "attribute": "x"},
                            "variable": "plant.x",
                        },
                        {
                            "attribute": {
                                "component": "sink",
                                "attribute": "failure_date",
                            },
                            "variable": "plant.failure_date",
                        },
                    ],
                }
            ],
        },
        allow_fmu_import=True,
    )
    zero_seed = _series(path, independent_importer, outputs)
    result = pyraichu.simulate(imported, t_max=2, samples=[0.5, 1.0, 1.5, 2.0])
    for name, source in (
        ("failed", "plant.failed"),
        ("x", "plant.x"),
        ("failure_date", "plant.failure_date"),
    ):
        actual = result.samples[name]
        for (date, value), row in zip(actual, zero_seed[1:], strict=True):
            assert date == float(row["time"])
            expected = row[source]
            if name == "x":
                assert value == pytest.approx(float(expected), abs=1e-7)
            else:
                assert value == expected
