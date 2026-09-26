"""Declared transition kind on the edges the muscadet plugin emits.

A driver bounding the number of failures along a sequence (the
failure-count cut-off of the sequence-tree exploration) reads a
DECLARED kind, `failure` or `repair`, rather than guessing from state
names. The ObjFM expansions declare it on the edges that carry the mode's
occurrence and return:

- an internal mode, on its own `fm*` automaton;
- an external mode, on the target's mirror automaton only, which is where
  its sequence events live: the mode's own automaton stays undeclared, so
  one occurrence is counted once per target and never twice;
- an on-demand draw is ONE instantaneous transition towards
  `[failed, parked]`: it is declared `failure` and the failure state is its
  first declared target, the only branch the kind applies to. Its re-arm
  is structure and stays undeclared.
"""

import json

import pyraichu
import pytest
from pyraichu.plugins import expand_model

DEMAND = [
    [{"obj": "GEN", "automaton": "sig", "state": "high", "ope": "==", "value": True}]
]


def _target(name):
    return {
        "name": name,
        "attributes": [
            {"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}
        ],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [],
        "equations": [],
    }


def _clock():
    """A solicitation that rises at 2.0, for the on-demand draws."""
    return {
        "name": "GEN",
        "attributes": [],
        "ports": [],
        "interfaces": [],
        "equations": [],
        "sensitive_functions": [],
        "automata": [
            {
                "name": "sig",
                "states": ["low", "high"],
                "init": "low",
                "transitions": [
                    {
                        "name": "rise",
                        "source": "low",
                        "targets": ["high"],
                        "distrib": "delay",
                        "time": 2.0,
                    }
                ],
            }
        ],
    }


def _model(objects, targets=("A",)):
    return {
        "name": "kinds",
        "plugins": {"muscadet": {"objects": list(objects)}},
        "components": [_target(name) for name in targets] + [_clock()],
        "connections": [],
        "indicators": [],
    }


def _mode(
    name, targets, failure, repair, *, behaviour="internal", mode_type="ObjFM", **extra
):
    spec = {
        "type": mode_type,
        "name": name,
        "targets": list(targets),
        "failure": failure,
        "repair": repair,
        "failure_effects": {"flow": False},
        "behaviour": behaviour,
    }
    spec.update(extra)
    return spec


def _automaton(expanded, component, automaton):
    for comp in expanded["components"]:
        if comp["name"] == component:
            for aut in comp.get("automata", []):
                if aut["name"] == automaton:
                    return aut
    raise AssertionError(f"no automaton `{component}.{automaton}`")


def _by_name(automaton):
    return {t["name"]: t for t in automaton["transitions"]}


def _loads(expanded):
    """The expanded body loads through the engine: the kind is a baseline
    construct, so no feature envelope is needed."""
    pyraichu.load_model(json.dumps(expanded))


# --- internal modes -------------------------------------------------------------


def test_an_internal_timed_mode_declares_its_failure_and_repair():
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A"],
                    [{"law": "exp", "rate": 1e-3}],
                    [{"law": "exp", "rate": 0.1}],
                )
            ]
        )
    )
    edges = _by_name(_automaton(expanded, "fm", "fm"))
    assert edges["failure"]["kind"] == "failure"
    assert edges["repair"]["kind"] == "repair"
    _loads(expanded)


def test_every_ccf_combination_declares_its_edges():
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A", "B"],
                    [{"law": "exp", "rate": 1e-3}, {"law": "exp", "rate": 1e-4}],
                    [{"law": "exp", "rate": 0.1}, {"law": "exp", "rate": 0.1}],
                )
            ],
            targets=("A", "B"),
        )
    )
    fm_component = next(c for c in expanded["components"] if c["name"] == "fm")
    kinds = [
        (t["source"], t["targets"][0], t.get("kind"))
        for aut in fm_component["automata"]
        for t in aut["transitions"]
    ]
    assert len(fm_component["automata"]) == 3  # A, B and the common cause AB
    for source, target, kind in kinds:
        expected = "failure" if target.startswith("occ") else "repair"
        assert kind == expected, (source, target, kind)
    _loads(expanded)


def test_a_non_repairable_mode_declares_its_failure_only():
    expanded = expand_model(
        _model([_mode("fm", ["A"], [{"law": "exp", "rate": 1e-3}], [None])])
    )
    edges = _by_name(_automaton(expanded, "fm", "fm"))
    assert set(edges) == {"failure"}
    assert edges["failure"]["kind"] == "failure"


# --- on-demand draws ----------------------------------------------------------------


def _assert_draw(automaton, source, failed, parked, kind):
    """The draw out of `source` carries `kind`, lists `failed` FIRST, and its
    re-arm out of `parked` carries no kind."""
    draws = [t for t in automaton["transitions"] if t["source"] == source]
    assert len(draws) == 1
    draw = draws[0]
    assert draw["distrib"] == "inst"
    assert draw["targets"] == [failed, parked]
    assert draw["kind"] == kind
    rearm = next(t for t in automaton["transitions"] if t["source"] == parked)
    assert "kind" not in rearm


def test_an_internal_on_demand_draw_is_a_failure_on_its_first_target():
    """The unified expander, an `inst` occurrence with an `inst` return."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A"],
                    [{"law": "inst", "prob": 0.3}],
                    [{"law": "inst", "prob": 0.5}],
                    failure_cond=DEMAND,
                )
            ]
        )
    )
    aut = _automaton(expanded, "fm", "fm")
    parked_occ = next(
        t["targets"][1] for t in aut["transitions"] if t["source"] == "rep"
    )
    _assert_draw(aut, "rep", "occ", parked_occ, "failure")
    parked_rep = next(
        t["targets"][1] for t in aut["transitions"] if t["source"] == "occ"
    )
    _assert_draw(aut, "occ", "rep", parked_rep, "repair")
    _loads(expanded)


def test_the_dedicated_on_demand_expander_declares_its_draw_too():
    """`ObjFMInst`, the historical on-demand expander, with a timed return."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A"],
                    [0.3],
                    [{"law": "exp", "rate": 0.1}],
                    mode_type="ObjFMInst",
                    failure_cond=DEMAND,
                )
            ]
        )
    )
    aut = _automaton(expanded, "fm", "fm")
    parked = next(t["targets"][1] for t in aut["transitions"] if t["source"] == "rep")
    _assert_draw(aut, "rep", "occ", parked, "failure")
    repair = next(t for t in aut["transitions"] if t["source"] == "occ")
    assert repair["kind"] == "repair"
    _loads(expanded)

    # An on-demand return (`inst` repair law): its draw is the repair, on
    # its first target, and its re-arm carries no kind.
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A"],
                    [0.3],
                    [{"law": "inst", "prob": 0.5}],
                    mode_type="ObjFMInst",
                    failure_cond=DEMAND,
                )
            ]
        )
    )
    aut = _automaton(expanded, "fm", "fm")
    parked_occ = next(
        t["targets"][1] for t in aut["transitions"] if t["source"] == "rep"
    )
    _assert_draw(aut, "rep", "occ", parked_occ, "failure")
    parked_rep = next(
        t["targets"][1] for t in aut["transitions"] if t["source"] == "occ"
    )
    _assert_draw(aut, "occ", "rep", parked_rep, "repair")
    _loads(expanded)


def test_a_drawn_failure_counts_on_its_failure_branch_only():
    """What the declaration means once run: the draw fires into either
    branch, and only the one entering its first target is the failure.
    Both branches are reached over seeds, so the order is not vacuous."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A"],
                    [{"law": "inst", "prob": 0.5}],
                    [None],
                    failure_cond=DEMAND,
                )
            ]
        )
    )
    aut = _automaton(expanded, "fm", "fm")
    draw = next(t for t in aut["transitions"] if t["source"] == "rep")
    failed, parked = draw["targets"]
    expanded["indicators"] = [
        {
            "name": "failed",
            "target": "state",
            "component": "fm",
            "automaton": "fm",
            "state": failed,
        },
        {
            "name": "parked",
            "target": "state",
            "component": "fm",
            "automaton": "fm",
            "state": parked,
        },
    ]
    model = pyraichu.load_model(json.dumps(expanded))
    outcomes = set()
    for seed in range(20):
        result = pyraichu.simulate(model, t_max=5.0, seed=seed)
        final = {
            name: bool(series[-1][1]) for name, series in result.indicators.items()
        }
        outcomes.add((final["failed"], final["parked"]))
    assert outcomes == {(True, False), (False, True)}


# --- external modes -------------------------------------------------------------


@pytest.mark.parametrize("behaviour", ["external", "external_rep_indep"])
def test_an_external_mode_declares_its_target_mirror_only(behaviour):
    expanded = expand_model(
        _model(
            [
                _mode(
                    "fm",
                    ["A", "B"],
                    [{"law": "exp", "rate": 1e-3}, None],
                    [{"law": "exp", "rate": 0.1}, None],
                    behaviour=behaviour,
                )
            ],
            targets=("A", "B"),
        )
    )
    for target in ("A", "B"):
        mirror = _automaton(expanded, target, "fm")
        failure = next(t for t in mirror["transitions"] if t["targets"] == ["occ"])
        repair = next(t for t in mirror["transitions"] if t["targets"] == ["rep"])
        assert failure["kind"] == "failure"
        assert repair["kind"] == "repair"
    # The mode's own automata drive the mirrors: declaring them too would
    # count one occurrence twice.
    own = next(c for c in expanded["components"] if c["name"] == "fm")
    assert all("kind" not in t for aut in own["automata"] for t in aut["transitions"])
    _loads(expanded)


# --- what is not a failure ------------------------------------------------------


def test_a_feared_event_declares_no_kind():
    event = {
        "type": "ObjEvent",
        "name": "ER",
        "cond": [[{"obj": "A", "attr": "flow", "ope": "==", "value": False}]],
    }
    expanded = expand_model(
        _model([_mode("fm", ["A"], [{"law": "exp", "rate": 1e-3}], [None]), event])
    )
    ev = _automaton(expanded, "ER", "ev")
    assert all("kind" not in t for t in ev["transitions"])


def test_an_unknown_kind_is_refused_by_the_engine_naming_it():
    expanded = expand_model(
        _model([_mode("fm", ["A"], [{"law": "exp", "rate": 1e-3}], [None])])
    )
    _automaton(expanded, "fm", "fm")["transitions"][0]["kind"] = "breakdown"
    with pytest.raises(Exception, match="breakdown"):
        pyraichu.load_model(json.dumps(expanded))
