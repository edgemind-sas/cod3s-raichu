"""Fault-tree generation by backward chaining (``pyraichu.fault_tree``).

The explanation of a top expression over states: an attribute nothing
computes keeps its initial value (or a profile's), only states move, a state
is entered by a transition whose draw is a basic event. Checked against block diagrams whose
minimal cut sets are known by construction.
"""

from __future__ import annotations

import pytest

import pyraichu


def unit(name, rate=1e-3, guard=None, repair=None):
    fail = {
        "name": "fail",
        "source": "ok",
        "targets": ["nok"],
        "distrib": "exp",
        "rate": rate,
    }
    if guard is not None:
        fail["guard"] = guard
    transitions = [fail]
    if repair is not None:
        transitions.append(
            {
                "name": "repair",
                "source": "nok",
                "targets": ["ok"],
                "distrib": "exp",
                "rate": repair,
            }
        )
    return {
        "name": name,
        "attributes": [
            {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ],
        "automata": [
            {
                "name": "health",
                "states": ["ok", "nok"],
                "init": "ok",
                "transitions": transitions,
            }
        ],
    }


def nok(name):
    return {
        "op": "state_active",
        "state": {"component": name, "automaton": "health", "state": "nok"},
    }


def model(*components):
    return pyraichu.load_model({"name": "ft", "components": list(components)})


def test_a_parallel_pair_has_one_cut_of_two():
    tree = pyraichu.fault_tree(
        model(unit("A"), unit("B")),
        {"op": "bool", "bool_op": "and", "args": [nok("A"), nok("B")]},
    )
    assert tree.minimal_cut_sets == [["A.health.fail", "B.health.fail"]]
    assert {e["law"] for e in tree.basic_events} == {"exponential"}
    assert "<define-fault-tree" in tree.open_psa


def test_a_cascade_is_explained_through_the_guard():
    tree = pyraichu.fault_tree(model(unit("A"), unit("B", guard=nok("A"))), nok("B"))
    assert tree.minimal_cut_sets == [["A.health.fail", "B.health.fail"]]


def test_a_repair_loop_adds_no_event():
    tree = pyraichu.fault_tree(model(unit("A", repair=0.1)), nok("A"))
    assert [e["name"] for e in tree.basic_events] == ["A.health.fail"]


def test_a_profile_holds_an_attribute_at_another_value():
    guard = {
        "op": "cmp",
        "cmp": "gt",
        "lhs": {"op": "attr", "attr": {"component": "B", "attribute": "x"}},
        "rhs": {"op": "const", "value": {"kind": "float", "value": 5.0}},
    }
    m = model(unit("B", guard=guard))
    # Nothing writes `x`: the failure can never fire, and a tree that is the
    # constant false is refused rather than returned empty.
    with pytest.raises(pyraichu.SimulationError, match="degenerate"):
        pyraichu.fault_tree(m, nok("B"))
    held = pyraichu.fault_tree(m, nok("B"), profile={"B.x": 10.0})
    assert held.minimal_cut_sets == [["B.health.fail"]]


def ok(name):
    return {
        "op": "state_active",
        "state": {"component": name, "automaton": "health", "state": "ok"},
    }


def test_a_negated_state_is_one_of_the_other_states():
    tree = pyraichu.fault_tree(
        model(unit("A")), {"op": "bool", "bool_op": "not", "args": [ok("A")]}
    )
    assert tree.minimal_cut_sets == [["A.health.fail"]]


def test_a_state_required_to_persist_is_refused_by_name():
    with pytest.raises(pyraichu.SimulationError, match="A.health.ok"):
        pyraichu.fault_tree(
            model(unit("A"), unit("B")),
            {"op": "bool", "bool_op": "and", "args": [ok("A"), nok("B")]},
        )


def _gates(node):
    if node["node"] != "gate":
        return []
    found = [node]
    for child in node["children"]:
        found.extend(_gates(child))
    return found


def test_gates_carry_their_identifier_and_origin():
    # `up` needs both automata of each block in `ok`: lost when either
    # block is down, each a gate of the attribute it stands for.
    def block(name):
        def health(automaton):
            return {
                "name": automaton,
                "states": ["ok", "nok"],
                "init": "ok",
                "transitions": [
                    {
                        "name": "fail",
                        "source": "ok",
                        "targets": ["nok"],
                        "distrib": "exp",
                        "rate": 1e-3,
                    }
                ],
            }

        def ok(automaton):
            return {
                "op": "state_active",
                "state": {"component": name, "automaton": automaton, "state": "ok"},
            }

        return {
            "name": name,
            "attributes": [
                {"name": "up", "kind": "bool", "init": {"kind": "bool", "value": True}}
            ],
            "automata": [health("h1"), health("h2")],
            "sensitive_functions": [
                {
                    "name": "update",
                    "effects": [
                        {
                            "target": {"component": name, "attribute": "up"},
                            "value": {
                                "op": "bool",
                                "bool_op": "and",
                                "args": [ok("h1"), ok("h2")],
                            },
                        }
                    ],
                }
            ],
        }

    def down(name):
        return {
            "op": "bool",
            "bool_op": "not",
            "args": [{"op": "attr", "attr": {"component": name, "attribute": "up"}}],
        }

    tree = pyraichu.fault_tree(
        model(block("A"), block("B")),
        {"op": "bool", "bool_op": "or", "args": [down("A"), down("B")]},
    )
    gates = _gates(tree.top)
    assert [g["id"] for g in gates] == ["top", "G1", "G2"]
    assert gates[0].get("origin") is None
    assert [g["origin"] for g in gates[1:]] == [
        {"kind": "attribute", "component": "A", "name": "up", "negated": True},
        {"kind": "attribute", "component": "B", "name": "up", "negated": True},
    ]
    assert _gates(tree.structure()["tree"]["top"]) == gates
    for gate in gates[1:]:
        assert f'<define-gate name="{gate["id"]}">' in tree.open_psa
