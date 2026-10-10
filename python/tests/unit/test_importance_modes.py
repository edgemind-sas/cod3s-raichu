"""Importance measures per failure mode, per component and per group,
against **closed forms**.

A component routinely carries several failure modes, and in the
``external`` behaviour each mode is an automaton grafted on the component
that reaches a state named like every other mode's. A basic event is
therefore named by its automaton too (``component.automaton.state``): the
modes of one pump stay distinct in the cuts, and the pump is measured as
the unit that gathers them. Birnbaum and criticality do not add up over
the modes, so the component level is computed, not summed.

Three properties of the basic events are checked besides:

- a common cause counts for each of its targets;
- an intermediate observer is never a failure, so it sits in no cut;
- an on-demand (``inst``) failure is a failure like any other: a lost
  draw followed by a won one still reads as failed, and the cut structure
  then reproduces the recorded feared-event probability exactly.
"""

from __future__ import annotations

import math

import pytest

import pyraichu

#: The tightest quantity compared is a conditional proportion over the
#: failed replicas; at this count its standard error is ≈ 0.003, so `TOL`
#: is roughly six of them.
NB_RUNS = 40_000
TOL = 0.02
INSTANTS = [10.0, 20.0, 40.0]


def _leaf(obj: str, attr: str) -> dict:
    return {"obj": obj, "attr": attr, "ope": "==", "value": False}


def _external(name: str, target: str, rate, attribute: str) -> dict:
    """A non-repairable ``external`` mode of `target` writing `attribute`."""
    failure = rate if isinstance(rate, list) else [{"law": "exp", "rate": rate}]
    return {
        "type": "ObjFM",
        "name": name,
        "behaviour": "external",
        "targets": [target] if isinstance(target, str) else target,
        "failure": failure,
        "repair": None,
        "failure_effects": {attribute: False},
    }


def _blocks(attributes: dict[str, list[str]]) -> list[dict]:
    return [
        {
            "name": block,
            "attributes": [
                {"name": a, "kind": "bool", "init": {"kind": "bool", "value": True}}
                for a in names
            ],
        }
        for block, names in attributes.items()
    ]


def _load(objects: list[dict], attributes: dict[str, list[str]]):
    return pyraichu.load_model(
        {
            "name": "modes",
            "plugins": {"muscadet": {"objects": objects}},
            "components": _blocks(attributes),
            "indicators": [],
        }
    )


# --- A in series with (B ∥ C), two modes on A --------------------------------

RATES = {"A1": 0.01, "A2": 0.015, "B": 0.05, "C": 0.08}


def two_mode_model():
    """`Φ = A1 ∨ A2 ∨ (B ∧ C)`: the two modes of A are two external modes
    grafted on A, each writing its own attribute."""
    objects = [
        _external("fm1", "A", RATES["A1"], "ok1"),
        _external("fm2", "A", RATES["A2"], "ok2"),
        _external("fm_B", "B", RATES["B"], "ok"),
        _external("fm_C", "C", RATES["C"], "ok"),
        {
            "type": "ObjEvent",
            "name": "system_down",
            "target": True,
            "cond": [
                [_leaf("A", "ok1")],
                [_leaf("A", "ok2")],
                [_leaf("B", "ok"), _leaf("C", "ok")],
            ],
        },
    ]
    return _load(objects, {"A": ["ok1", "ok2"], "B": ["ok"], "C": ["ok"]})


def _q(instant: float) -> dict[str, float]:
    return {k: 1.0 - math.exp(-rate * instant) for k, rate in RATES.items()}


@pytest.fixture(scope="module")
def two_modes():
    return pyraichu.importance(
        two_mode_model(),
        nb_runs=NB_RUNS,
        t_max=40.0,
        instants=INSTANTS,
        seed=42,
        groups={"pair": ["B.fm_B.occ", "C.fm_C.occ"], "pump": ["A.fm1.occ", "A.fm2.occ"]},
    )


def test_two_modes_of_one_component_are_two_basic_events(two_modes):
    found = {frozenset(cut.events) for cut in two_modes.cuts}
    assert found == {
        frozenset({"A.fm1.occ"}),
        frozenset({"A.fm2.occ"}),
        frozenset({"B.fm_B.occ", "C.fm_C.occ"}),
    }
    assert two_modes.components["A"].events == ["fm1.occ", "fm2.occ"]
    assert two_modes.q_cuts == pytest.approx(two_modes.q_target, abs=1e-12)


def test_each_mode_matches_its_closed_form(two_modes):
    for k, instant in enumerate(INSTANTS):
        q = _q(instant)
        pair = q["B"] * q["C"]
        top = 1.0 - (1.0 - q["A1"]) * (1.0 - q["A2"]) * (1.0 - pair)
        for mode, other in (("A1", "A2"), ("A2", "A1")):
            measured = two_modes.basic_events[f"A.fm{mode[-1]}.occ"]
            assert measured.component == "A"
            assert measured.automaton == f"fm{mode[-1]}"
            assert measured.state == "occ"
            # Critical for one mode: the other mode and the pair both hold.
            birnbaum = (1.0 - q[other]) * (1.0 - pair)
            assert measured.birnbaum[k] == pytest.approx(birnbaum, abs=TOL)
            assert measured.unavailability[k] == pytest.approx(q[mode], abs=TOL)
            assert measured.fussell_vesely[k] == pytest.approx(q[mode] / top, abs=TOL)
            assert measured.criticality[k] == pytest.approx(
                birnbaum * q[mode] / top, abs=TOL
            )


def test_the_component_is_measured_as_one_unit(two_modes):
    """Birnbaum of the pump is `1 − q_B q_C`, which is not the sum of the
    Birnbaum of its two modes: computed, never added."""
    for k, instant in enumerate(INSTANTS):
        q = _q(instant)
        pair = q["B"] * q["C"]
        a_down = 1.0 - (1.0 - q["A1"]) * (1.0 - q["A2"])
        top = 1.0 - (1.0 - a_down) * (1.0 - pair)
        pump = two_modes.components["A"]
        assert pump.birnbaum[k] == pytest.approx(1.0 - pair, abs=TOL)
        assert pump.unavailability[k] == pytest.approx(a_down, abs=TOL)
        assert pump.fussell_vesely[k] == pytest.approx(a_down / top, abs=TOL)
        assert pump.criticality[k] == pytest.approx(
            (1.0 - pair) * a_down / top, abs=TOL
        )
        assert pump.q_system_failed[k] == pytest.approx(1.0, abs=TOL)
        assert pump.q_system_intact[k] == pytest.approx(pair, abs=TOL)
    # At t = 10 the sum of the modes is 1.38: not even a probability.
    modes = sum(two_modes.basic_events[f"A.fm{i}.occ"].birnbaum[0] for i in (1, 2))
    assert modes > 1.0 > two_modes.components["A"].birnbaum[0]


def test_a_declared_group_is_measured_as_one_unit(two_modes):
    """A group of the two modes is the pump; a group of the pair is
    critical wherever A holds."""
    pump = two_modes.groups["pump"]
    component = two_modes.components["A"]
    assert pump.birnbaum == component.birnbaum
    assert pump.fussell_vesely == component.fussell_vesely
    pair = two_modes.groups["pair"]
    assert pair.events == ["B.fm_B.occ", "C.fm_C.occ"]
    for k, instant in enumerate(INSTANTS):
        q = _q(instant)
        holds = (1.0 - q["A1"]) * (1.0 - q["A2"])
        assert pair.birnbaum[k] == pytest.approx(holds, abs=TOL)
        assert pair.unavailability[k] == pytest.approx(
            1.0 - (1.0 - q["B"]) * (1.0 - q["C"]), abs=TOL
        )
    assert pair.risk_achievement(two_modes)[-1] == pytest.approx(
        1.0 / two_modes.q_cuts[-1]
    )


def test_a_group_naming_an_unknown_basic_event_is_refused():
    with pytest.raises(pyraichu.SimulationError) as excinfo:
        pyraichu.importance(
            two_mode_model(),
            nb_runs=10,
            t_max=1.0,
            instants=[1.0],
            groups={"pump": ["A.fm3.occ"]},
        )
    assert "group `pump` names `A.fm3.occ`" in str(excinfo.value)


def test_a_malformed_basic_event_name_is_refused_before_the_run():
    with pytest.raises(ValueError, match="component.automaton.state"):
        pyraichu.importance(
            two_mode_model(), nb_runs=10, t_max=1.0, instants=[1.0], basic_events=["A.occ"]
        )


def test_a_listed_basic_event_must_be_recorded():
    with pytest.raises(pyraichu.SimulationError) as excinfo:
        pyraichu.importance(
            two_mode_model(),
            nb_runs=10,
            t_max=1.0,
            instants=[1.0],
            basic_events=["A.fm1.rep_never"],
        )
    assert "is not a state the model records" in str(excinfo.value)


# --- Common cause and intermediate observer -----------------------------------

CCF = {"B": 0.05, "C": 0.08, "cc": 0.02}


def common_cause_model():
    """`Φ = B ∧ C`, each block with its own mode and a common cause that
    takes both down at once, and an intermediate observer of B that is
    not the feared event."""
    down = {
        "B": [_leaf("B", "ind"), _leaf("B", "cc")],
        "C": [_leaf("C", "ind"), _leaf("C", "cc")],
    }
    objects = [
        _external("fm_B", "B", CCF["B"], "ind"),
        _external("fm_C", "C", CCF["C"], "ind"),
        # Order 1 inactive, order 2 active: the common cause proper.
        _external("ccf", ["B", "C"], [None, {"law": "exp", "rate": CCF["cc"]}], "cc"),
        {"type": "ObjEvent", "name": "b_down", "cond": [[leaf] for leaf in down["B"]]},
        {
            "type": "ObjEvent",
            "name": "system_down",
            "target": True,
            "cond": [[b, c] for b in down["B"] for c in down["C"]],
        },
    ]
    return _load(objects, {"B": ["ind", "cc"], "C": ["ind", "cc"]})


@pytest.fixture(scope="module")
def common_cause():
    return pyraichu.importance(
        common_cause_model(), nb_runs=NB_RUNS, t_max=40.0, instants=INSTANTS, seed=3
    )


def test_a_common_cause_counts_for_each_of_its_targets(common_cause):
    found = {frozenset(cut.events) for cut in common_cause.cuts}
    assert frozenset({"B.ccf.occ", "C.ccf.occ"}) in found
    assert frozenset({"B.fm_B.occ", "C.fm_C.occ"}) in found
    assert "ccf.occ" in common_cause.components["B"].events
    assert "ccf.occ" in common_cause.components["C"].events
    for k, instant in enumerate(INSTANTS):
        q = {k_: 1.0 - math.exp(-rate * instant) for k_, rate in CCF.items()}
        # B with both its events failed: the system is down iff C is.
        c_down = 1.0 - (1.0 - q["C"]) * (1.0 - q["cc"])
        assert common_cause.components["B"].birnbaum[k] == pytest.approx(c_down, abs=TOL)
        # Every cut meets B: it carries all of the risk.
        assert common_cause.components["B"].fussell_vesely[k] == pytest.approx(1.0)


def test_an_intermediate_observer_is_in_no_cut(common_cause):
    """`b_down` records its entry before the feared event, as every
    observer does; it is not a failure, so it is not a basic event."""
    for cut in common_cause.cuts:
        assert not any(event.startswith("b_down.") for event in cut.events)
    assert "b_down" not in common_cause.components
    assert common_cause.q_cuts == pytest.approx(common_cause.q_target, abs=1e-12)


def test_selecting_every_recorded_state_lets_the_observer_in():
    """The contrast: under `monitored` the observer is a basic event, and
    it then stands in for B in the cuts."""
    analysis = pyraichu.importance(
        common_cause_model(),
        nb_runs=4000,
        t_max=40.0,
        instants=[40.0],
        seed=3,
        basic_events="monitored",
    )
    assert any(
        event.startswith("b_down.") for cut in analysis.cuts for event in cut.events
    )


# --- On-demand failures -------------------------------------------------------


def _on_demand(kind: str) -> dict:
    """The on-demand failure of D, solicited while the grid G is down, in
    each of the three dialects that build one."""
    demand = [[{"obj": "G", "attr": "ok", "ope": "==", "value": False}]]
    if kind == "ObjFMInst":
        return {
            "type": "ObjFMInst",
            "name": "start",
            "targets": ["D"],
            "failure": {"prob": 0.3},
            "repair": 0.2,
            "failure_cond": demand,
            "failure_effects": {"ok": False},
        }
    return {
        "type": "ObjFM",
        "name": "start",
        "behaviour": kind,
        "targets": ["D"],
        "failure": [{"law": "inst", "prob": 0.3}],
        "repair": [{"law": "exp", "rate": 0.2}],
        "failure_cond": demand,
        "failure_effects": {"ok": False},
    }


def on_demand_model(kind: str):
    """`Φ = G ∧ D`: a repairable grid G, and a diesel D that fails to
    start with probability 0.3 on each loss of the grid. The grid is lost
    and restored many times over the horizon, so most draws are lost
    before one is won."""
    objects = [
        {
            "type": "ObjFM",
            "name": "grid",
            "targets": ["G"],
            "failure": [{"law": "exp", "rate": 0.2}],
            "repair": [{"law": "exp", "rate": 1.0}],
            "failure_effects": {"ok": False},
        },
        _on_demand(kind),
        {
            "type": "ObjEvent",
            "name": "blackout",
            "target": True,
            "cond": [[_leaf("G", "ok"), _leaf("D", "ok")]],
        },
    ]
    return _load(objects, {"G": ["ok"], "D": ["ok"]})


@pytest.mark.parametrize("kind", ["ObjFMInst", "internal", "external"])
def test_an_on_demand_failure_is_a_basic_event(kind):
    """The draw is in the cut beside the grid, and the structure then
    judges the system down exactly where the trajectories recorded it.
    Before, the on-demand mode was missing from the cut and the grid
    alone read as a blackout: `q_cuts` far above `q_target`."""
    analysis = pyraichu.importance(
        on_demand_model(kind), nb_runs=4000, t_max=50.0, instants=[10.0, 50.0], seed=9
    )
    assert analysis.q_target[-1] > 0.05
    assert analysis.q_cuts == pytest.approx(analysis.q_target, abs=1e-12)
    for cut in analysis.cuts:
        assert len(cut.events) == 2, cut
        assert any(event.startswith("D.") or event.startswith("start.") for event in cut.events)
    assert analysis.basic_events, "the on-demand mode carries a measure"
