"""Competing boundary transitions follow crossing time, regardless of declaration order."""

import pyraichu
import pytest


def constant(value):
    return {"op": "const", "value": {"kind": "float", "value": value}}


@pytest.mark.parametrize("reverse", [False, True])
def test_earliest_boundary_transition_wins(reverse):
    thresholds = [("late", 0.004), ("early", 0.002)]
    if reverse:
        thresholds.reverse()
    level = {"op": "attr", "attr": {"component": "c", "attribute": "x"}}
    model = pyraichu.load_model(
        {
            "name": "competing_crossings",
            "components": [
                {
                    "name": "c",
                    "attributes": [
                        {"name": "x", "kind": "float", "init": constant(0.0)["value"]}
                    ],
                    "automata": [
                        {
                            "name": "a",
                            "states": ["initial", "late", "early"],
                            "init": "initial",
                            "transitions": [
                                {
                                    "name": name,
                                    "source": "initial",
                                    "targets": [name],
                                    "distrib": "boundary",
                                    "guard": {
                                        "op": "cmp",
                                        "cmp": "ge",
                                        "lhs": level,
                                        "rhs": constant(threshold),
                                    },
                                }
                                for name, threshold in thresholds
                            ],
                        }
                    ],
                    "equations": [
                        {"target": "x", "kind": "ode", "expr": constant(1.0)}
                    ],
                }
            ],
            "indicators": [
                {
                    "name": "early",
                    "target": "state",
                    "component": "c",
                    "automaton": "a",
                    "state": "early",
                }
            ],
        }
    )
    result = pyraichu.simulate(model, t_max=0.1, samples=[0.001, 0.003, 0.1])
    assert [event.transition for event in result.events] == ["c.a.early"]
    assert result.events[0].time == pytest.approx(0.002, abs=1e-10, rel=0)
    assert result.samples["early"] == [(0.001, 0.0), (0.003, 1.0), (0.1, 1.0)]


def test_the_former_spellings_load_the_same_model():
    """`sensitive_functions` and `"distrib": "watched"`, the spellings the
    schema used up to 0.145, are still read and give the same trajectory."""
    level = {"op": "attr", "attr": {"component": "c", "attribute": "x"}}

    def document(functions_key, law):
        return {
            "name": "former_spellings",
            "components": [
                {
                    "name": "c",
                    "attributes": [
                        {"name": "x", "kind": "float", "init": constant(0.0)["value"]},
                        {"name": "seen", "kind": "bool", "init": {"kind": "bool", "value": False}},
                    ],
                    "automata": [
                        {
                            "name": "a",
                            "states": ["low", "high"],
                            "init": "low",
                            "transitions": [
                                {
                                    "name": "cross",
                                    "source": "low",
                                    "targets": ["high"],
                                    "distrib": law,
                                    "guard": {
                                        "op": "cmp",
                                        "cmp": "ge",
                                        "lhs": level,
                                        "rhs": constant(2.0),
                                    },
                                }
                            ],
                        }
                    ],
                    functions_key: [
                        {
                            "name": "mark",
                            "effects": [
                                {
                                    "target": {"component": "c", "attribute": "seen"},
                                    "value": {
                                        "op": "state_active",
                                        "state": {"component": "c", "automaton": "a", "state": "high"},
                                    },
                                }
                            ],
                        }
                    ],
                    "equations": [{"target": "x", "kind": "ode", "expr": constant(1.0)}],
                }
            ],
            "indicators": [
                {
                    "name": "seen",
                    "target": "attribute",
                    "attr": {"component": "c", "attribute": "seen"},
                }
            ],
        }

    def run(doc):
        result = pyraichu.simulate(pyraichu.load_model(doc), t_max=4.0, samples=[1.0, 3.0])
        return [event.transition for event in result.events], result.samples["seen"]

    current = run(document("propagations", "boundary"))
    former = run(document("sensitive_functions", "watched"))
    assert current == (["c.a.cross"], [(1.0, 0.0), (3.0, 1.0)])
    assert former == current
