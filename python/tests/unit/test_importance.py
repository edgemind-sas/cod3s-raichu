"""Native importance measures, against a **closed-form** reliability
block diagram.

There is no reference oracle to differ from here, because no engine in
this family computes importance measures: the rule for that case is an
analytic reference. The diagram used throughout is the smallest one with
two cuts of different sizes,

    Φ = A ∨ (B ∧ C)

one series block and one redundant pair, every block non-repairable with
its own exponential failure law. Everything the analysis reports then has
a textbook expression in the block unavailabilities ``q_i(t) = 1 −
e^{−λ_i t}``, written out in `_expected` below and asserted term by term.

A second family checks the two degenerate diagrams (pure series, pure
parallel), a third the repairable steady state, and the last one the
computational cost, which is the property the measures live or die by:
one campaign, not two.
"""

from __future__ import annotations

import math
import time

import pytest

import pyraichu

#: Failure rates of the three blocks. Spread apart on purpose: measures
#: that only agreed on symmetric blocks would agree for the wrong reason.
RATES = {"A": 0.02, "B": 0.05, "C": 0.08}

#: Replicas of the analytic comparisons. The tightest quantity compared
#: is a conditional proportion over the failed replicas, whose standard
#: error at this count is ≈ 0.003, so `TOL` below is roughly six of them.
NB_RUNS = 40_000
TOL = 0.02

INSTANTS = [10.0, 20.0, 40.0]


def _model(rates: dict[str, float], cond: list[list[dict]], repair: float | None = None):
    """A block diagram: one non-repairable failure mode per block, and an
    `ObjEvent` feared event over the blocks' `flow` attributes."""
    repair_law = None if repair is None else [{"law": "exp", "rate": repair}]
    objects = [
        {
            "type": "ObjFM",
            "name": f"fm_{block}",
            "targets": [block],
            "failure": [{"law": "exp", "rate": rate}],
            "repair": repair_law,
            "failure_effects": {"flow": False},
        }
        for block, rate in rates.items()
    ] + [
        {"type": "ObjEvent", "name": "system_down", "target": True, "cond": cond},
    ]
    return pyraichu.load_model(
        {
            "name": "rbd",
            "plugins": {"muscadet": {"objects": objects}},
            "components": [
                {
                    "name": block,
                    "attributes": [
                        {
                            "name": "flow",
                            "kind": "bool",
                            "init": {"kind": "bool", "value": True},
                        }
                    ],
                }
                for block in rates
            ],
            "indicators": [
                {
                    "name": "system_down_occ",
                    "target": "state",
                    "component": "system_down",
                    "automaton": "ev",
                    "state": "occ",
                }
            ],
        }
    )


def _down(block: str) -> dict:
    return {"obj": block, "attr": "flow", "ope": "==", "value": False}


def series_parallel():
    """`Φ = A ∨ (B ∧ C)`."""
    return _model(RATES, [[_down("A")], [_down("B"), _down("C")]])


def _expected(instant: float) -> dict:
    """The closed form of every reported quantity at `instant`."""
    q = {b: 1.0 - math.exp(-rate * instant) for b, rate in RATES.items()}
    # Φ = A ∨ (B ∧ C), independent blocks.
    top = 1.0 - (1.0 - q["A"]) * (1.0 - q["B"] * q["C"])
    birnbaum = {
        # E[Φ(1_A, ·) − Φ(0_A, ·)] = 1 − q_B q_C.
        "A": 1.0 - q["B"] * q["C"],
        # E[(A ∨ C) − A] = (1 − q_A) q_C, and symmetrically for C.
        "B": (1.0 - q["A"]) * q["C"],
        "C": (1.0 - q["A"]) * q["B"],
    }
    # P(a realized cut contains the block | system down).
    fussell_vesely = {
        "A": q["A"] / top,
        "B": q["B"] * q["C"] / top,
        "C": q["B"] * q["C"] / top,
    }
    return {
        "q": q,
        "top": top,
        "birnbaum": birnbaum,
        "fussell_vesely": fussell_vesely,
        "criticality": {b: birnbaum[b] * q[b] / top for b in RATES},
        # The risk with the block certainly failed, and with it perfect.
        "q_system_failed": {
            "A": 1.0,
            "B": 1.0 - (1.0 - q["A"]) * (1.0 - q["C"]),
            "C": 1.0 - (1.0 - q["A"]) * (1.0 - q["B"]),
        },
        "q_system_intact": {
            "A": q["B"] * q["C"],
            "B": q["A"],
            "C": q["A"],
        },
    }


@pytest.fixture(scope="module")
def analysis():
    return pyraichu.importance(
        series_parallel(), nb_runs=NB_RUNS, t_max=40.0, instants=INSTANTS, seed=42
    )


def test_the_minimal_cuts_are_the_diagram_cuts(analysis):
    """`{A}` and `{B, C}`, and nothing else: the two orders of the
    redundant pair are one cut set, not two."""
    found = {frozenset(cut.events) for cut in analysis.cuts}
    assert found == {
        frozenset({"fm_A.occ"}),
        frozenset({"fm_B.occ", "fm_C.occ"}),
    }


def test_the_cut_structure_reproduces_the_recorded_probability(analysis):
    """The completeness diagnostic: the reconstructed structure judges the
    system down exactly where the trajectories recorded the feared event.
    A cut no replica walked would show up here as a gap."""
    assert analysis.q_cuts == pytest.approx(analysis.q_target, abs=1e-12)


def test_the_feared_event_probability_matches_the_closed_form(analysis):
    for k, instant in enumerate(analysis.instants):
        assert analysis.q_cuts[k] == pytest.approx(_expected(instant)["top"], abs=TOL)


@pytest.mark.parametrize("block", ["A", "B", "C"])
def test_birnbaum_matches_the_closed_form(analysis, block):
    """`I^B_i = P(the system is critical for i)`, which on this diagram is
    `1 − q_B q_C` for the series block and `(1 − q_A) q_other` for each
    member of the redundant pair."""
    measured = analysis.components[f"fm_{block}"].birnbaum
    for k, instant in enumerate(analysis.instants):
        assert measured[k] == pytest.approx(_expected(instant)["birnbaum"][block], abs=TOL)


@pytest.mark.parametrize("block", ["A", "B", "C"])
def test_fussell_vesely_matches_the_closed_form(analysis, block):
    """`FV_i = P(cut ∋ i realized) / P(top)`: `q_A / Q` for the series
    block, `q_B q_C / Q` for either member of the pair. The three do not
    sum to one, and must not: two cuts can be realized at once."""
    measured = analysis.components[f"fm_{block}"].fussell_vesely
    for k, instant in enumerate(analysis.instants):
        assert measured[k] == pytest.approx(
            _expected(instant)["fussell_vesely"][block], abs=TOL
        )


@pytest.mark.parametrize("block", ["A", "B", "C"])
def test_criticality_and_unavailability_match_the_closed_form(analysis, block):
    component = analysis.components[f"fm_{block}"]
    for k, instant in enumerate(analysis.instants):
        expected = _expected(instant)
        assert component.unavailability[k] == pytest.approx(expected["q"][block], abs=TOL)
        assert component.criticality[k] == pytest.approx(
            expected["criticality"][block], abs=TOL
        )


@pytest.mark.parametrize("block", ["A", "B", "C"])
def test_the_pivotal_risk_levels_match_the_closed_form(analysis, block):
    """The two levels risk-achievement and risk-reduction worth are built
    from: the risk with the block certainly failed, and with it perfect."""
    component = analysis.components[f"fm_{block}"]
    for k, instant in enumerate(analysis.instants):
        expected = _expected(instant)
        assert component.q_system_failed[k] == pytest.approx(
            expected["q_system_failed"][block], abs=TOL
        )
        assert component.q_system_intact[k] == pytest.approx(
            expected["q_system_intact"][block], abs=TOL
        )


def test_the_worths_derive_from_the_two_pivotal_levels(analysis):
    """Risk-achievement worth is `Q⁺/Q` and risk-reduction worth `Q/Q⁻`:
    reported as their two levels rather than as ratios, because both
    denominators legitimately reach zero."""
    component = analysis.components["fm_A"]
    raw = component.risk_achievement(analysis)
    rrw = component.risk_reduction(analysis)
    for k in range(len(analysis.instants)):
        assert raw[k] == pytest.approx(component.q_system_failed[k] / analysis.q_cuts[k])
        assert rrw[k] == pytest.approx(analysis.q_cuts[k] / component.q_system_intact[k])
        # A is the series block: failing it is certain top, so its
        # achievement worth is the reciprocal of the risk itself.
        assert raw[k] == pytest.approx(1.0 / analysis.q_cuts[k])


def test_the_components_come_back_ranked(analysis):
    """The reported order is the answer to "where do I invest": the share
    of risk carried, descending. On this diagram at t = 40 the closed form
    puts the redundant pair ahead of the series block (0.90 against 0.60),
    which is the whole point of computing it rather than guessing: the
    single point of failure is *not* the first place to invest here."""
    ranked = list(analysis.components)
    shares = [analysis.components[name].fussell_vesely[-1] for name in ranked]
    assert shares == sorted(shares, reverse=True)
    assert ranked[-1] == "fm_A"
    last = _expected(analysis.instants[-1])["fussell_vesely"]
    assert last["B"] > last["A"]
    assert analysis.components["fm_B"].fussell_vesely[-1] == pytest.approx(
        last["B"], abs=TOL
    )


def test_a_redundant_pair_carries_all_of_the_risk():
    """A pure parallel pair has one cut, so both members have
    Fussell-Vesely 1, and the Birnbaum of one **is** the unavailability of
    the other: it is critical exactly when its partner is already down."""
    rates = {"B": 0.05, "C": 0.08}
    model = _model(rates, [[_down("B"), _down("C")]])
    analysis = pyraichu.importance(
        model, nb_runs=NB_RUNS, t_max=40.0, instants=INSTANTS, seed=7
    )
    assert len(analysis.cuts) == 1
    for block, partner in (("B", "C"), ("C", "B")):
        component = analysis.components[f"fm_{block}"]
        assert component.fussell_vesely == pytest.approx([1.0] * len(INSTANTS))
        for k, instant in enumerate(INSTANTS):
            assert component.birnbaum[k] == pytest.approx(
                1.0 - math.exp(-rates[partner] * instant), abs=TOL
            )


def test_a_series_chain_is_critical_through_the_survival_of_the_others():
    """Every block of a series chain is its own cut, so its Birnbaum is
    the probability that all the others hold, and its Fussell-Vesely the
    share of failures its own cut explains."""
    rates = {"A": 0.02, "B": 0.05, "C": 0.08}
    model = _model(rates, [[_down(b)] for b in rates])
    analysis = pyraichu.importance(
        model, nb_runs=NB_RUNS, t_max=40.0, instants=INSTANTS, seed=7
    )
    assert len(analysis.cuts) == 3
    for k, instant in enumerate(INSTANTS):
        q = {b: 1.0 - math.exp(-rate * instant) for b, rate in rates.items()}
        top = 1.0 - math.prod(1.0 - value for value in q.values())
        for block in rates:
            component = analysis.components[f"fm_{block}"]
            others = math.prod(1.0 - q[o] for o in rates if o != block)
            assert component.birnbaum[k] == pytest.approx(others, abs=TOL)
            assert component.fussell_vesely[k] == pytest.approx(q[block] / top, abs=TOL)


def test_a_repairable_block_reaches_its_steady_state_unavailability():
    """With repair the states cycle, and the reconstruction has to follow
    them: a single repairable block settles at `λ / (λ + μ)`."""
    model = _model({"A": 0.1}, [[_down("A")]], repair=0.4)
    analysis = pyraichu.importance(
        model, nb_runs=20_000, t_max=200.0, instants=[100.0, 200.0], seed=11
    )
    component = analysis.components["fm_A"]
    for k in range(2):
        assert component.unavailability[k] == pytest.approx(0.1 / 0.5, abs=0.01)
        assert analysis.q_cuts[k] == pytest.approx(0.1 / 0.5, abs=0.01)
        # The only cut: the block is critical in every state, and carries
        # every failure.
        assert component.birnbaum[k] == pytest.approx(1.0)
        assert component.fussell_vesely[k] == pytest.approx(1.0)


def test_a_campaign_that_never_reached_the_event_says_so():
    model = _model({"A": 1e-9}, [[_down("A")]])
    analysis = pyraichu.importance(
        model, nb_runs=200, t_max=1.0, instants=[1.0], seed=3
    )
    assert analysis.cuts == []
    assert analysis.components == {}
    assert analysis.q_cuts == [0.0]


def test_the_result_is_reproducible_and_thread_invariant():
    model = series_parallel()
    kwargs = dict(nb_runs=4000, t_max=40.0, instants=INSTANTS, seed=5)
    one = pyraichu.importance(model, threads=1, **kwargs)
    many = pyraichu.importance(model, threads=4, **kwargs)
    again = pyraichu.importance(model, threads=1, **kwargs)
    assert one == again
    assert one == many


def test_an_unrecorded_feared_event_is_refused_rather_than_left_empty():
    """A target whose entry transition is not `monitored` never appears in
    a trajectory, so the analysis would find no cut and hand back an empty
    ranking with nothing saying why."""
    body = pyraichu.model_body(pyraichu.expand_model(series_parallel().json))
    for component in body["components"]:
        if component["name"] == "system_down":
            for automaton in component["automata"]:
                for transition in automaton["transitions"]:
                    transition["monitored"] = False
    silent = pyraichu.load_model(body)
    with pytest.raises(pyraichu.SimulationError) as excinfo:
        pyraichu.importance(silent, nb_runs=10, t_max=1.0, instants=[1.0])
    assert "never recorded in a sequence" in str(excinfo.value)


def test_the_analysis_carries_its_provenance(analysis):
    assert analysis.nb_runs == NB_RUNS
    assert analysis.engine_version == pyraichu.__version__


def test_naming_an_unknown_feared_event_names_the_declared_ones():
    with pytest.raises(pyraichu.SimulationError) as excinfo:
        pyraichu.importance(
            series_parallel(), nb_runs=10, t_max=1.0, instants=[1.0], target="nope"
        )
    assert "system_down" in str(excinfo.value)


def test_a_model_with_two_feared_events_asks_which_one():
    model = _model(
        RATES,
        [[_down("A")], [_down("B"), _down("C")]],
    )
    body = pyraichu.model_body(pyraichu.expand_model(model.json))
    # A second declared target makes the choice ambiguous.
    body["targets"] = body["targets"] + [
        {
            "name": "b_alone",
            "component": "fm_B",
            "automaton": "fm",
            "state": "occ",
        }
    ]
    two = pyraichu.load_model(body)
    with pytest.raises(pyraichu.SimulationError) as excinfo:
        pyraichu.importance(two, nb_runs=10, t_max=1.0, instants=[1.0])
    assert "name the one to measure" in str(excinfo.value)
    # Named, either one is measurable.
    picked = pyraichu.importance(
        two, nb_runs=2000, t_max=40.0, instants=[40.0], target="b_alone", seed=1
    )
    assert picked.target == "fm_B.occ"
