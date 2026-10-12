"""The causal journal numbers the fixpoint iterations (`evolA`).

One iteration applies one sensitive function: the one with the lowest
index among those pending. Every write it makes carries its iteration
index, counted from 1 and restarted at every fixpoint. A fired
transition's own edge effects are written before the fixpoint and carry
0.
"""

import pyraichu


def _attr(name, init):
    return {"name": name, "kind": "bool", "init": {"kind": "bool", "value": init}}


def _ref(attribute):
    return {"component": "C", "attribute": attribute}


def _on():
    return {"op": "state_active",
            "state": {"component": "C", "automaton": "sw", "state": "on"}}


def _model():
    """`flip` fires at t = 3 and writes `latched` itself (edge effect).

    The state change triggers `high` and `other`; `high` writes `b`, which
    triggers `low`. Declared in the order low, high, other, `low` has the
    lowest index, so it is popped before `other` although `other` was
    pending first.
    """
    return pyraichu.load_model({
        "name": "iterations",
        "components": [{
            "name": "C",
            "attributes": [_attr("latched", False), _attr("b", False),
                           _attr("c", False), _attr("d", False)],
            "automata": [{
                "name": "sw", "states": ["off", "on"], "init": "off",
                "transitions": [{
                    "name": "flip", "source": "off", "targets": ["on"],
                    "distrib": "delay", "time": 3.0,
                    "effects": [{"target": _ref("latched"),
                                 "value": {"op": "const",
                                           "value": {"kind": "bool", "value": True}}}],
                }],
            }],
            "sensitive_functions": [
                {"name": "low", "effects": [{"target": _ref("c"),
                                             "value": {"op": "attr", "attr": _ref("b")}}]},
                {"name": "high", "effects": [{"target": _ref("b"), "value": _on()}]},
                {"name": "other", "effects": [{"target": _ref("d"), "value": _on()}]},
            ],
        }],
    })


def _records_at(journal, time):
    return [r for r in journal
            if r["record"] in ("function_triggered", "attribute_changed")
            and r["time"] == time]


def test_writes_follow_the_engine_pop_order_with_their_iteration():
    result = pyraichu.simulate(_model(), t_max=5.0, journal=True)
    seen = [
        (r["record"], r.get("function") or r["attribute"], r["iteration"])
        for r in _records_at(result.journal, 3.0)
    ]
    assert seen == [
        ("attribute_changed", "C.latched", 0),
        ("function_triggered", "C.high", 1),
        ("attribute_changed", "C.b", 1),
        ("function_triggered", "C.low", 2),
        ("attribute_changed", "C.c", 2),
        ("function_triggered", "C.other", 3),
        ("attribute_changed", "C.d", 3),
    ]


def test_cascade_after_carries_the_iteration_of_each_write():
    result = pyraichu.simulate(_model(), t_max=5.0, journal=True)
    cascade = pyraichu.JournalQuery(result.journal).cascade_after("C.sw.flip")
    assert [(c.attribute, c.iteration) for c in cascade.changes] == [
        ("C.latched", 0), ("C.b", 1), ("C.c", 2), ("C.d", 3),
    ]
