"""The Python interactive facade exposes communication-point controls."""

import pyraichu
import pytest


def test_advance_and_declared_input():
    model = {
        "name": "clock_input",
        "components": [{
            "name": "C",
            "attributes": [{
                "name": "input",
                "kind": "bool",
                "init": {"kind": "bool", "value": False},
            }],
            "ports": [],
            "interfaces": [],
            "equations": [],
            "automata": [{
                "name": "A",
                "states": ["idle", "done"],
                "init": "idle",
                "transitions": [{
                    "name": "finish",
                    "source": "idle",
                    "targets": ["done"],
                    "on_interruption": "reset",
                    "distrib": "delay",
                    "time": 2.0,
                }],
            }],
            "sensitive_functions": [],
        }],
        "connections": [],
        "indicators": [],
    }
    simulation = pyraichu.interactive(model, t_max=3.0)
    with pytest.raises(pyraichu.SimulationError, match="C.input"):
        simulation.set_input("C.input", True, [])
    simulation.set_input("C.input", True, ["C.input"])
    assert simulation.attribute("C.input") is True
    simulation.advance_to(1.0)
    assert simulation.time == 1.0
    simulation.advance_to(2.0)
    assert simulation.state("C.A") == "done"
    assert [event.time for event in simulation.history()] == [2.0]
