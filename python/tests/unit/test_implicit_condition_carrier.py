"""A condition leaf that names no `obj` reads the component carrying it.

``obj`` is optional in a cod3s condition leaf, and the omission is not a
shorthand for "anywhere": it names the component the condition is carried
by, which cod3s resolves by handing that component to `prepare_attr_tree`
as its `obj_default` (`cod3s/pycatshoo/common.py`). For a failure mode
that component is its TARGET, and a common-cause mode resolves one per
combination, requiring the tree on every target the combination acts on
(`ObjFM.get_failure_cond` compiles one tree per `target_comps` and
`all()`s them).

The plugin used to read `leaf["obj"]` with no default, so a study written
in the implicit form (the form the COD3S platform emits) was translated
and then died at build time on `KeyError: 'obj'`. These tests pin the
fallback on both leaf shapes, the variable read (``attr``) and the state
read (``automaton``/``state``), and pin the named refusal where there is
no carrier to fall back on: an event observes arbitrary components and a
logic gate names its sources.
"""

import pytest

import pyraichu
from pyraichu.plugins import expand_model


def _watched_component(expression):
    """The `component` of the single leaf a one-leaf condition tree reads.

    The tree is `OR(AND(cmp))`, so the comparison sits two boolean levels
    down; its left-hand side is an `attr` or a `state_active` read, and
    both carry the component under the same key.
    """
    leaves = _leaves(expression)
    assert len(leaves) == 1, f"expected one leaf, got {leaves}"
    return leaves[0]["component"]


def _leaves(expression):
    """Every `(component, …)` reference an expression reads, in order."""
    if not isinstance(expression, dict):
        return []
    if expression.get("op") == "attr":
        return [expression["attr"]]
    if expression.get("op") == "state_active":
        return [expression["state"]]
    found = []
    for key in ("args", "cond", "then", "otherwise", "lhs", "rhs"):
        value = expression.get(key)
        for item in value if isinstance(value, list) else [value]:
            found += _leaves(item)
    return found


def _target(name, automaton=None):
    """A plain core component: one boolean attribute a mode can fail, and
    optionally one automaton a condition can read a state of."""
    return {
        "name": name,
        "attributes": [
            {"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}
        ],
        "ports": [],
        "interfaces": [],
        "automata": [automaton] if automaton else [],
        "sensitive_functions": [],
        "equations": [],
    }


#: A monitor automaton a condition leaf can name a state of. It never
#: fires (no transition): what is under test is which component the state
#: is resolved on, not when it is active.
_MONITOR = {"name": "mon", "states": ["up", "down"], "init": "up", "transitions": []}


def _mode(name, targets, **extra):
    spec = {
        "type": "ObjFM",
        "name": name,
        "targets": list(targets),
        "failure": [{"law": "delay", "time": 5.0}] + [None] * (len(targets) - 1),
        "repair": [{"law": "delay", "time": 3.0}] + [None] * (len(targets) - 1),
        "failure_effects": {"flow": False},
    }
    spec.update(extra)
    return spec


def _model(objects, components):
    return {
        "name": "implicit_carrier",
        "plugins": {"muscadet": {"objects": list(objects)}},
        "components": list(components),
        "connections": [],
        "indicators": [],
    }


def _transitions(expanded, component, automaton):
    aut = next(
        a
        for c in expanded["components"]
        if c["name"] == component
        for a in c["automata"]
        if a["name"] == automaton
    )
    return {t["name"]: t for t in aut["transitions"]}


# --- the fallback, on both leaf shapes ----------------------------------------


def test_an_attr_leaf_with_no_object_reads_the_modes_target():
    """The `attr` branch: `{"attr": …}` with no `obj` is the target's
    variable, which is what the platform emits for a mode conditioned on
    its own component."""
    expanded = expand_model(
        _model(
            [_mode("crack", ["A"], failure_cond=[[{"attr": "flow", "value": True}]])],
            [_target("A")],
        )
    )
    guard = _transitions(expanded, "crack", "fm")["failure"]["guard"]
    assert _watched_component(guard) == "A"


def test_a_state_leaf_with_no_object_reads_the_modes_target():
    """The `automaton`/`state` branch: the same omission, resolved the
    same way. The two branches read the one `obj` cod3s defaults."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "crack",
                    ["A"],
                    repair_cond=[[{"automaton": "mon", "state": "up", "value": True}]],
                )
            ],
            [_target("A", _MONITOR)],
        )
    )
    guard = _transitions(expanded, "crack", "fm")["repair"]["guard"]
    assert _leaves(guard) == [{"component": "A", "automaton": "mon", "state": "up"}]


def test_a_leaf_that_names_its_object_keeps_naming_it():
    """The fallback is a default, not an override: a leaf pointing at
    ANOTHER component still reads that component."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "crack",
                    ["A"],
                    failure_cond=[[{"obj": "B", "attr": "flow", "value": True}]],
                )
            ],
            [_target("A"), _target("B")],
        )
    )
    guard = _transitions(expanded, "crack", "fm")["failure"]["guard"]
    assert _watched_component(guard) == "B"


def test_an_on_demand_mode_resolves_the_leaf_the_same_way():
    """`ObjFMInst` is the same mode under the historical vocabulary, and
    its demand is a condition tree like any other."""
    expanded = expand_model(
        _model(
            [
                {
                    "type": "ObjFMInst",
                    "name": "crack",
                    "targets": ["A"],
                    "failure": {"prob": 0.3},
                    "repair": 0.0,
                    "failure_cond": [[{"attr": "flow", "value": True}]],
                    "failure_effects": {"flow": False},
                }
            ],
            [_target("A")],
        )
    )
    draw = _transitions(expanded, "crack", "fm")["occ"]
    assert _watched_component(draw["guard"]) == "A"


# --- common cause: one carrier per combination --------------------------------


def test_a_common_cause_combination_reads_every_target_it_acts_on():
    """cod3s compiles the tree once per target of the combination and
    requires it on all of them. An order-1 combination therefore reads
    its own target; the order-2 one reads both, under a conjunction."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "crack",
                    ["A", "B"],
                    failure=[{"law": "delay", "time": 5.0}] * 2,
                    repair=[{"law": "delay", "time": 3.0}] * 2,
                    failure_cond=[[{"attr": "flow", "value": True}]],
                )
            ],
            [_target("A"), _target("B")],
        )
    )
    first = _transitions(expanded, "crack", "fm__cc_1")["occ__cc_1"]["guard"]
    second = _transitions(expanded, "crack", "fm__cc_2")["occ__cc_2"]["guard"]
    both = _transitions(expanded, "crack", "fm__cc_1_2")["occ__cc_1_2"]["guard"]
    assert [leaf["component"] for leaf in _leaves(first)] == ["A"]
    assert [leaf["component"] for leaf in _leaves(second)] == ["B"]
    assert [leaf["component"] for leaf in _leaves(both)] == ["A", "B"]


def test_an_explicit_common_cause_condition_is_emitted_as_it_was():
    """The per-carrier conjunction collapses when every leaf names its
    `obj`: a tree that names no carrier is one expression whatever the
    combination, so an explicit condition keeps the guard it had."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "crack",
                    ["A", "B"],
                    failure=[{"law": "delay", "time": 5.0}] * 2,
                    repair=[{"law": "delay", "time": 3.0}] * 2,
                    failure_cond=[[{"obj": "A", "attr": "flow", "value": True}]],
                )
            ],
            [_target("A"), _target("B")],
        )
    )
    both = _transitions(expanded, "crack", "fm__cc_1_2")["occ__cc_1_2"]["guard"]
    assert [leaf["component"] for leaf in _leaves(both)] == ["A"]


# --- where there is no carrier ------------------------------------------------


def test_an_event_leaf_with_no_object_is_refused_by_naming_it():
    """An `ObjEvent` has no target, and cod3s compiles its tree with no
    `obj_default`: a leaf has to name what it watches, and the refusal
    says so rather than dying on a missing key."""
    with pytest.raises(ValueError, match=r"ObjEvent `ER`: `cond`.*names no `obj`"):
        expand_model(
            _model(
                [
                    {
                        "type": "ObjEvent",
                        "name": "ER",
                        "cond": [[{"attr": "flow", "value": False}]],
                    }
                ],
                [_target("A")],
            )
        )


def test_a_logic_gate_leaf_with_no_object_is_refused_by_naming_it():
    """A gate is a combinational function of the sources its leaves name,
    and holds nothing a leaf could read instead."""
    with pytest.raises(ValueError, match=r"ObjLogicGate `G`: `cond`.*names no `obj`"):
        expand_model(
            _model(
                [
                    {
                        "type": "ObjLogicGate",
                        "name": "G",
                        "kind": "or",
                        "cond": [[{"attr": "flow", "value": True}]],
                    }
                ],
                [_target("A")],
            )
        )


# --- end to end ---------------------------------------------------------------


def _rail_study():
    """The running example in miniature: a fed segment whose crack mode
    is conditioned on its OWN input flow, written in the implicit form
    the platform emits (`{"attr": "O_fed_in", "ope": "==", …}`)."""
    return {
        "name": "rail",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "SRC",
                        "flows_out": [{"name": "O", "var_prod_default": True}],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "SEG",
                        "flows_in": [{"name": "O", "logic": "or"}],
                        "flows_out": [{"name": "O", "var_prod_default": True}],
                    },
                    {
                        "type": "ObjFM",
                        "name": "crack",
                        "targets": ["SEG"],
                        "failure": {"distrib": "delay", "time": 5.0},
                        "repair": [None],
                        "failure_cond": [
                            [{"attr": "O_fed_in", "ope": "==", "value": True}]
                        ],
                        "failure_effects": {"O_prod_available": False},
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {
                "from": {"component": "SRC", "port": "O_out"},
                "to": {"component": "SEG", "port": "O_in"},
            }
        ],
        "indicators": [
            {
                "name": "seg_out",
                "target": "attribute",
                "attr": {"component": "SEG", "attribute": "O_fed_out"},
            }
        ],
    }


def test_a_study_carrying_an_implicit_condition_builds_and_runs():
    """The whole route, which is where the missing default showed: the
    study is built (no `KeyError`) and simulated, and the mode gated by
    its own input flow fires on it."""
    result = pyraichu.simulate(
        pyraichu.load_model(_rail_study()), t_max=20.0, samples=[0.0, 4.0, 6.0]
    )
    assert result.samples["seg_out"] == [(0.0, True), (4.0, True), (6.0, False)]
    assert [(event.time, event.transition) for event in result.events] == [
        (5.0, "crack.fm.failure")
    ]
