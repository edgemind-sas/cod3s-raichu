"""An interactive session hands back its causal journal.

This is what records a forced branch with its full cascade: run to the
branch point, snapshot, fire each outcome from the same prefix.
"""

import pytest

import pyraichu


def _bool(value):
    return {"op": "const", "value": {"kind": "bool", "value": value}}


def _state(automaton, state):
    return {"op": "state_active",
            "state": {"component": "B", "automaton": automaton, "state": state}}


def _model():
    """A demand at t = 2 solicits the backup, which starts (90 %) or fails
    on demand (10 %); `running` follows its state through a function."""
    return pyraichu.load_model({
        "name": "solicited_backup",
        "components": [{
            "name": "B",
            "attributes": [{"name": "running", "kind": "bool",
                            "init": {"kind": "bool", "value": False}}],
            "automata": [
                {"name": "need", "states": ["no", "yes"], "init": "no",
                 "transitions": [{"name": "ask", "source": "no", "targets": ["yes"],
                                  "distrib": "delay", "time": 2.0}]},
                {"name": "unit", "states": ["idle", "on", "failed"], "init": "idle",
                 "transitions": [{"name": "start", "source": "idle",
                                  "targets": ["on", "failed"], "distrib": "inst",
                                  "probs": [0.9], "guard": _state("need", "yes")}]},
            ],
            "sensitive_functions": [{"name": "follow", "effects": [{
                "target": {"component": "B", "attribute": "running"},
                "value": _state("unit", "on")}]}],
        }],
    })


def _at_branch_point(seed=0):
    session = pyraichu.Interactive(_model(), t_max=10.0, journal=True, seed=seed)
    session.step()  # B.need.ask at t = 2: the start is now armed
    assert [f.transition for f in session.fireable()] == ["B.unit.start"]
    return session


def _shape(records):
    """What a page shows of a cascade: no dates."""
    kept = ("transition_fired", "function_triggered", "attribute_changed")
    return [{k: v for k, v in r.items() if k != "time"}
            for r in records if r["record"] in kept]


def test_the_journal_carries_the_fired_transition_and_its_cascade():
    session = _at_branch_point()
    session.fire("B.unit.start", to="on")
    journal = session.journal()
    fired = [r["transition"] for r in journal if r["record"] == "transition_fired"]
    assert fired == ["B.need.ask", "B.unit.start"]
    start = max(i for i, r in enumerate(journal) if r["record"] == "transition_fired")
    last = journal[start:]
    assert [(r["record"], r.get("function") or r.get("attribute")) for r in _shape(last)] == [
        ("transition_fired", None),
        ("function_triggered", "B.follow"),
        ("attribute_changed", "B.running"),
    ]


def test_two_branches_from_one_snapshot_share_the_prefix():
    session = _at_branch_point()
    checkpoint = session.snapshot()
    prefix = session.journal()
    session.fire("B.unit.start", to="on")
    started = session.journal()
    session.restore(checkpoint)
    assert session.journal() == prefix
    session.fire("B.unit.start", to="failed")
    failed = session.journal()
    assert started[: len(prefix)] == failed[: len(prefix)] == prefix
    assert started[len(prefix):] != failed[len(prefix):]


def test_a_forced_outcome_writes_what_the_sampled_one_writes():
    sampled = None
    for seed in range(200):
        session = _at_branch_point(seed)
        session.step()
        if session.state("B.unit") == "failed":
            sampled = session.journal()
            break
    assert sampled is not None, "no seed below 200 drew the failure on demand"
    forced = _at_branch_point()
    forced.fire("B.unit.start", to="failed")
    assert _shape(forced.journal()) == _shape(sampled)


def test_a_session_without_the_journal_says_so():
    session = pyraichu.Interactive(_model(), t_max=10.0)
    with pytest.raises(pyraichu.SimulationError, match="journal=True"):
        session.journal()
