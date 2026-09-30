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


def _operator_model():
    return {
        "name": "operator_stock",
        "components": [
            {
                "name": "stock",
                "attributes": [
                    {
                        "name": "level",
                        "kind": "float",
                        "init": {"kind": "float", "value": 4.0},
                    }
                ],
                "equations": [
                    {
                        "target": "level",
                        "kind": "ode",
                        "expr": {
                            "op": "if",
                            "cond": {
                                "op": "state_active",
                                "state": {
                                    "component": "stock",
                                    "automaton": "bound",
                                    "state": "full",
                                },
                            },
                            "then": {
                                "op": "const",
                                "value": {"kind": "float", "value": -1.0},
                            },
                            "otherwise": {
                                "op": "const",
                                "value": {"kind": "float", "value": 0.0},
                            },
                        },
                    }
                ],
                "automata": [
                    {
                        "name": "bound",
                        "states": ["full", "empty"],
                        "init": "full",
                        "transitions": [
                            {
                                "name": "empty",
                                "source": "full",
                                "targets": ["empty"],
                                "distrib": "watched",
                                "guard": {
                                    "op": "cmp",
                                    "cmp": "le",
                                    "lhs": {
                                        "op": "attr",
                                        "attr": {
                                            "component": "stock",
                                            "attribute": "level",
                                        },
                                    },
                                    "rhs": {
                                        "op": "const",
                                        "value": {"kind": "float", "value": 0.0},
                                    },
                                },
                            }
                        ],
                    }
                ],
            },
            {
                "name": "failure",
                "automata": [
                    {
                        "name": "mode",
                        "states": ["ok", "failed"],
                        "init": "ok",
                        "transitions": [
                            {
                                "name": "fail",
                                "source": "ok",
                                "targets": ["failed"],
                                "distrib": "exp",
                                "rate": 0.5,
                            }
                        ],
                    }
                ],
            },
        ],
    }


def test_operator_clock_stops_at_boundary_and_keeps_explicit_date():
    simulation = pyraichu.interactive(
        _operator_model(), t_max=20.0, operator_control=True
    )
    simulation.set_date("failure.mode.fail", 8.0)
    before = simulation.snapshot()
    boundary = simulation.advance_operator_to(10.0)
    assert isinstance(boundary, pyraichu.OperatorAdvance)
    assert boundary.requested_time == 10.0
    assert boundary.reached_time == pytest.approx(4.0, abs=1e-8)
    assert boundary.stop == "event"
    assert [event.transition for event in boundary.events] == ["stock.bound.empty"]
    assert simulation.attribute("stock.level") == pytest.approx(0.0, abs=1e-8)
    assert [(f.transition, f.date) for f in simulation.fireable()] == [
        ("failure.mode.fail", 8.0)
    ]
    failure = simulation.advance_operator_to(10.0)
    assert failure.reached_time == 8.0
    assert failure.events[0].transition == "failure.mode.fail"
    simulation.restore(before)
    assert simulation.advance_operator_to(10.0) == boundary
    assert simulation.advance_operator_to(10.0) == failure
    assert simulation.advance_operator_to(10.0).stop == "target"
    assert simulation.time == 10.0
    simulation.reset()
    assert simulation.time == 0.0
    assert simulation.history() == []
    assert (
        next(
            f for f in simulation.fireable() if f.transition == "failure.mode.fail"
        ).date
        is None
    )


def test_operator_advancing_without_programming_never_fires_stochastic_clock():
    simulation = pyraichu.interactive(
        _operator_model(), t_max=20.0, operator_control=True
    )
    simulation.advance_operator_to(10.0)
    assert simulation.advance_operator_to(10.0).stop == "target"
    assert simulation.state("failure.mode") == "ok"
    before = simulation.snapshot()
    history = simulation.history()
    for date in (float("nan"), float("inf"), 9.0, 21.0):
        with pytest.raises(pyraichu.SimulationError):
            simulation.advance_operator_to(date)
        with pytest.raises(pyraichu.SimulationError):
            simulation.set_date("failure.mode.fail", date)
        assert simulation.time == 10.0
        assert simulation.history() == history
    simulation.restore(before)
    automatic = pyraichu.interactive(_operator_model(), t_max=20.0)
    with pytest.raises(pyraichu.SimulationError, match="policy"):
        simulation.restore(automatic.snapshot())
    assert simulation.time == 10.0


def test_operator_choice_is_inspected_before_selection():
    model = {
        "name": "choice",
        "components": [
            {
                "name": "C",
                "automata": [
                    {
                        "name": "A",
                        "states": ["pending", "ok", "failed"],
                        "init": "pending",
                        "transitions": [
                            {
                                "name": "resolve",
                                "source": "pending",
                                "targets": ["ok", "failed"],
                                "distrib": "inst",
                                "probs": [0.7],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    simulation = pyraichu.interactive(model, t_max=10.0, operator_control=True)
    result = simulation.advance_operator_to(5.0)
    assert result.stop == "choice"
    assert result.choice == "C.A.resolve"
    assert result.reached_time == 0.0
    assert result.events == []
    with pytest.raises(pyraichu.SimulationError, match="explicit destination"):
        simulation.fire("C.A.resolve")
    assert simulation.history() == []
    assert simulation.fire("C.A.resolve", to="failed").to_state == "failed"
    assert simulation.advance_operator_to(5.0).reached_time == 5.0


def test_operator_foreign_model_snapshot_is_rejected_before_commit():
    session = pyraichu.interactive(_operator_model(), t_max=20.0, operator_control=True)
    session.advance_operator_to(2.0)
    foreign = pyraichu.interactive(
        {"name": "different", "components": []}, t_max=20.0, operator_control=True
    )
    with pytest.raises(pyraichu.SimulationError, match="another compiled model"):
        session.restore(foreign.snapshot())
    assert session.time == 2.0
    assert session.attribute("stock.level") == pytest.approx(2.0)
