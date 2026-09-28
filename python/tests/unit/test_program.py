"""Public tests for the mixed-integer program document surface."""

import pyraichu
import pytest


def ref(name):
    return {"component": "plant", "attribute": name}


def read(name):
    return {"op": "attr", "attr": ref(name)}


def number(value):
    return {"op": "const", "value": {"kind": "float", "value": value}}


def dispatch_model():
    return {
        "name": "dispatch",
        "components": [
            {
                "name": "plant",
                "attributes": [
                    {
                        "name": name,
                        "kind": "float",
                        "init": {"kind": "float", "value": value},
                    }
                    for name, value in (
                        ("cheap", 0.0),
                        ("expensive", 0.0),
                        ("demand", 80.0),
                        ("cost", 0.0),
                    )
                ]
                + [
                    {
                        "name": "feasible",
                        "kind": "bool",
                        "init": {"kind": "bool", "value": False},
                    }
                ],
                "automata": [
                    {
                        "name": "mode",
                        "states": ["up", "down"],
                        "init": "up",
                        "transitions": [
                            {
                                "name": "fail",
                                "source": "up",
                                "targets": ["down"],
                                "distrib": "delay",
                                "time": 5.0,
                            },
                            {
                                "name": "repair",
                                "source": "down",
                                "targets": ["up"],
                                "distrib": "delay",
                                "time": 10.0,
                            },
                        ],
                    }
                ],
            }
        ],
        "programs": [
            {
                "name": "least_cost",
                "variables": [
                    {
                        "attribute": ref("cheap"),
                        "lower": number(0),
                        "upper": {
                            "op": "if",
                            "cond": {
                                "op": "state_active",
                                "state": {
                                    "component": "plant",
                                    "automaton": "mode",
                                    "state": "up",
                                },
                            },
                            "then": number(60),
                            "otherwise": number(0),
                        },
                        "on_infeasible": {"kind": "float", "value": 0.0},
                    },
                    {
                        "attribute": ref("expensive"),
                        "lower": number(0),
                        "upper": number(50),
                        "on_infeasible": {"kind": "float", "value": 0.0},
                    },
                ],
                "sense": "minimize",
                "objective": {
                    "op": "add",
                    "args": [
                        read("cheap"),
                        {"op": "mul", "args": [number(2), read("expensive")]},
                    ],
                },
                "constraints": [
                    {
                        "name": "supply",
                        "expr": {
                            "op": "add",
                            "args": [read("cheap"), read("expensive")],
                        },
                        "lower": read("demand"),
                    }
                ],
                "feasible": ref("feasible"),
                "objective_value": ref("cost"),
            }
        ],
        "indicators": [
            {"name": name, "target": "attribute", "attr": ref(name)}
            for name in ("cheap", "expensive", "feasible", "cost")
        ],
    }


def test_dispatch_failure_and_repair():
    model = pyraichu.load_model(dispatch_model())
    result = pyraichu.simulate(model, t_max=16, journal=True)
    assert result.indicators["cheap"] == [(0.0, 60.0), (5.0, 0.0), (15.0, 60.0)]
    assert result.indicators["expensive"] == [(0.0, 20.0), (5.0, 0.0), (15.0, 20.0)]
    assert result.indicators["feasible"] == [(0.0, True), (5.0, False), (15.0, True)]
    assert [
        (record["time"], record["status"])
        for record in result.journal
        if record["record"] == "program_solved"
    ] == [(0.0, "optimal"), (5.0, "infeasible"), (15.0, "optimal")]


def test_nonlinear_program_is_refused_with_its_name():
    body = dispatch_model()
    body["programs"][0]["objective"] = {
        "op": "mul",
        "args": [read("cheap"), read("expensive")],
    }
    with pytest.raises(pyraichu.ModelError, match="least_cost.*nonlinear"):
        pyraichu.load_model(body)


def test_penalised_shortfall_keeps_a_degraded_dispatch_feasible():
    body = dispatch_model()
    body["components"][0]["attributes"].append(
        {"name": "shortfall", "kind": "float", "init": {"kind": "float", "value": 0.0}}
    )
    program = body["programs"][0]
    program["variables"][0]["upper"] = number(0)
    program["variables"].append(
        {
            "attribute": ref("shortfall"),
            "lower": number(0),
            "on_infeasible": {"kind": "float", "value": 0.0},
        }
    )
    program["objective"]["args"].append(
        {"op": "mul", "args": [number(10), read("shortfall")]}
    )
    program["constraints"][0]["expr"]["args"].append(read("shortfall"))
    body["indicators"].append(
        {"name": "shortfall", "target": "attribute", "attr": ref("shortfall")}
    )
    result = pyraichu.simulate(pyraichu.load_model(body), t_max=1.0)
    assert result.indicators["cheap"][0] == (0.0, 0.0)
    assert result.indicators["expensive"][0] == (0.0, 50.0)
    assert result.indicators["shortfall"][0] == (0.0, 30.0)
    assert result.indicators["feasible"][0] == (0.0, True)
