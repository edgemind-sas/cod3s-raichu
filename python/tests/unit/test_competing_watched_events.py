"""Competing watched transitions follow crossing time, regardless of declaration order."""

import pyraichu
import pytest


def constant(value):
    return {"op": "const", "value": {"kind": "float", "value": value}}


@pytest.mark.parametrize("reverse", [False, True])
def test_earliest_watched_transition_wins(reverse):
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
                                    "distrib": "watched",
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
