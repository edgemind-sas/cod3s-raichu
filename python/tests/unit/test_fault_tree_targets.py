"""A fault tree generated from the feared event of a muscadet-shaped model.

The declaration is the one ``muscadet.declare.system_spec`` writes for a
reliability block diagram, field for field, written out so this suite needs
neither muscadet nor the reference engine: a source, blocks failing on an
exponential law, a target, and the feared event ``T_lost`` observing that
the target is no longer fed, declared as the model's target the way a study
declares it.

What is pinned:

1. the tree of the feared event is the block diagram's, down to the
   failures: its probability is the closed form (the two-block case
   answered 1.0 before the flow variables were unrolled);
2. on a model without repair and without dynamics, the tree and RAICHU's
   own Monte-Carlo of the same model agree: the tree's probability lies in
   the simulation's confidence interval, seed fixed;
3. with repairs, the tree is an upper bound of the first-occurrence
   probability, and says why;
4. a tree that would be degenerate, or that would take the observer for a
   failure, is refused with its cause named;
5. the ``raichu.fault_tree`` envelope has the shape its reference states.
"""

from __future__ import annotations

import json
import math

import pyraichu
import pytest
from pyraichu.muscadet import declare, engine

FLOW = "is_ok"
MISSION = 1000.0
RATES = {"B1": 1e-3, "B2": 2e-3, "B3": 3e-3}


def _flow_in(logic="or"):
    return {
        "name": FLOW,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_in_default": False,
        "var_available_in_default": True,
        "logic": logic,
        "cls": "FlowIn",
    }


def _flow_out(prod_cond=None, prod_default=False):
    entry = {
        "name": FLOW,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_is_active_default": True,
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_cond_inner_mode": "or",
        "var_prod_default": prod_default,
        "negate": False,
        "cls": "FlowOut",
    }
    if prod_cond is not None:
        entry["var_prod_cond"] = prod_cond
    return entry


def _component(name, source_cls, flows, failure_modes=()):
    return {
        "name": name,
        "cls": "ObjFlow",
        "source_cls": source_cls,
        "flows": list(flows),
        "capacities": [],
        "measurements_in": [],
        "measurements_out": [],
        "rules": [],
        "transfers": [],
        "mixtures": [],
        "automata": [],
        "failure_modes": list(failure_modes),
    }


def _exp_mode(rate, repair):
    return {
        "cls": "exp",
        "name": "frun",
        "failure_state": "occ",
        "failure_cond": True,
        "failure_rate": rate,
        "failure_effects": [[f"{FLOW}_fed_available_out", False]],
        "failure_param_name": "lambda",
        "repair_state": "rep",
        "repair_cond": True,
        "repair_rate": repair,
        "repair_effects": [],
        "repair_param_name": "mu",
    }


def _connection(source, target):
    return {
        "source": source,
        "source_box": f"{FLOW}_out",
        "target": target,
        "target_box": f"{FLOW}_in",
        "flow": FLOW,
    }


def declaration(shape, repair=0.0, **event):
    """``parallel`` (B1, B2), ``series`` (S -> B1 -> B2 -> T) or
    ``two_of_three`` (B1, B2, B3 into a target needing two), and the feared
    event ``T_lost``: the target is not fed."""
    blocks = ["B1", "B2", "B3"] if shape == "two_of_three" else ["B1", "B2"]
    produced = [[{"name": FLOW, "port": "in"}]]
    components = {
        "S": _component("S", "Source", [_flow_out(prod_default=True)]),
        "T": _component(
            "T", "Target", [_flow_in(logic=2 if shape == "two_of_three" else "or")]
        ),
        "T_lost": dict(
            {
                "name": "T_lost",
                "kind": "two_state_mode",
                "cls": "ObjEvent",
                "cond": [[{"obj": "T", "attr": f"{FLOW}_fed_in", "value": False}]],
            },
            **event,
        ),
    }
    for name in blocks:
        components[name] = _component(
            name,
            "Block",
            [_flow_in(), _flow_out(prod_cond=produced)],
            [_exp_mode(RATES[name], repair)],
        )
    if shape == "series":
        wiring = [("S", "B1"), ("B1", "B2"), ("B2", "T")]
    else:
        wiring = [("S", b) for b in blocks] + [(b, "T") for b in blocks]
    return {
        "version": "1.0.0",
        "name": shape,
        "components": components,
        "connections": [_connection(s, t) for s, t in wiring],
        "indicators": [],
        "generated_indicators": True,
    }


def study_model(document):
    """The model, with ``T_lost`` declared as its target and observed by an
    indicator: what a study launching a target campaign hands the engine."""
    model = engine.build_model(document)
    raw = json.loads(model.json)
    body = pyraichu.model_body(raw)
    spec = document["components"]["T_lost"]
    automaton, _ = declare.event_automaton(spec)
    state = declare.event_occurrence_state(spec)
    body.setdefault("targets", []).append(
        {
            "name": "T_lost",
            "component": "T_lost",
            "automaton": automaton,
            "state": state,
        }
    )
    body.setdefault("indicators", []).append(
        {
            "name": "T_lost_occ",
            "target": "state",
            "component": "T_lost",
            "automaton": automaton,
            "state": state,
        }
    )
    return pyraichu.load_model(json.dumps(pyraichu.seal(raw)))


def failed(name, t=MISSION):
    return 1.0 - math.exp(-RATES[name] * t)


CLOSED_FORMS = {
    "parallel": lambda: failed("B1") * failed("B2"),
    "series": lambda: 1.0 - (1.0 - failed("B1")) * (1.0 - failed("B2")),
    "two_of_three": lambda: (
        failed("B1") * failed("B2")
        + failed("B1") * failed("B3")
        + failed("B2") * failed("B3")
        - 2.0 * failed("B1") * failed("B2") * failed("B3")
    ),
}

#: The quantifier's own exactness, not a Monte-Carlo band: decision
#: diagrams sum a handful of products of the laws' distributions, so the
#: result is the closed form to a few units in the last place.
CLOSED_FORM_RTOL = 1e-12


# --- 1. the feared event is explained down to the failures ---------------


def test_the_two_block_parallel_target_is_its_closed_form():
    """The case measured on 2026-10-04: 1.0 then, the closed form now."""
    tree = pyraichu.fault_tree(
        study_model(declaration("parallel", repair=0.1)), targets=["T_lost"]
    )
    assert tree.minimal_cut_sets == [["B1.frun.failure", "B2.frun.failure"]]
    result = tree.quantify(mission_time=MISSION)
    exact = (1 - math.exp(-1)) * (1 - math.exp(-2))
    assert result.probability == pytest.approx(exact, rel=CLOSED_FORM_RTOL)
    assert round(result.probability, 4) == 0.5466
    assert result.method == "bdd" and result.exact
    # Repaired blocks: the tree reads the probability without repair, and
    # says so per automaton.
    assert len(tree.warnings) == 2
    assert all("without repair" in w for w in tree.warnings)


@pytest.mark.parametrize("shape", sorted(CLOSED_FORMS))
def test_a_tree_without_repair_is_the_closed_form_and_the_simulation(shape):
    model = study_model(declaration(shape))
    tree = pyraichu.fault_tree(model, targets=["T_lost"])
    assert tree.warnings == [], "no repair, no dynamics: the tree is exact"
    probability = tree.quantify(mission_time=MISSION).probability
    assert probability == pytest.approx(CLOSED_FORMS[shape](), rel=CLOSED_FORM_RTOL)

    # The independent witness: RAICHU's own simulation of the same model,
    # each trajectory stopped at the feared event.
    estimates = pyraichu.monte_carlo(
        model,
        nb_runs=20_000,
        t_max=MISSION,
        samples=[MISSION],
        seed=17,
        stop_at_targets=True,
        confidence=0.99,
    )
    lost = estimates.indicators["T_lost_occ"]
    assert lost.ci.low[0] <= probability <= lost.ci.high[0], (
        probability,
        lost.mean[0],
        lost.ci.low[0],
        lost.ci.high[0],
    )


def test_with_repairs_the_tree_bounds_the_first_occurrence_from_above():
    model = study_model(declaration("parallel", repair=0.1))
    tree = pyraichu.fault_tree(model, targets=["T_lost"])
    probability = tree.quantify(mission_time=MISSION).probability
    estimates = pyraichu.monte_carlo(
        model,
        nb_runs=20_000,
        t_max=MISSION,
        samples=[MISSION],
        seed=17,
        stop_at_targets=True,
        confidence=0.99,
    )
    lost = estimates.indicators["T_lost_occ"]
    # Repaired within ten hours, the blocks are rarely down together: the
    # first occurrence is far below the probability without repair.
    assert probability > lost.ci.high[0], (probability, lost.mean[0])


def test_the_observer_state_as_a_top_is_crossed_not_counted():
    """The second half of the 2026-10-04 defect: the observer's own state,
    given as the top, used to come back as a basic event of probability
    one."""
    model = study_model(declaration("parallel"))
    occ = {
        "op": "state_active",
        "state": {"component": "T_lost", "automaton": "ev", "state": "occ"},
    }
    tree = pyraichu.fault_tree(model, occ)
    assert tree.minimal_cut_sets == [["B1.frun.failure", "B2.frun.failure"]]
    assert all(e["component"] != "T_lost" for e in tree.basic_events)


def test_a_zero_repair_rate_is_a_mode_without_repair():
    """muscadet's default repair rate is zero, and the reference engine never
    draws it; this layer builds the mode without its repair edge."""
    body = pyraichu.model_body(
        json.loads(engine.build_model(declaration("parallel")).json)
    )
    b1 = next(c for c in body["components"] if c["name"] == "B1")
    (frun,) = b1["automata"]
    assert [t["name"] for t in frun["transitions"]] == ["failure"]


# --- 2. what is refused, and why -----------------------------------------


def test_a_feared_event_no_failure_can_cause_is_a_degenerate_tree():
    document = declaration("parallel")
    for name in ("B1", "B2"):
        document["components"][name]["failure_modes"] = []
    with pytest.raises(pyraichu.SimulationError, match="degenerate") as refused:
        pyraichu.fault_tree(study_model(document), targets=["T_lost"])
    message = str(refused.value)
    assert "T_lost.ev.occ" in message and "reads no state" in message


def test_an_observer_that_waits_is_refused_not_taken_for_a_failure():
    model = study_model(declaration("parallel", tempo_occ=5.0))
    with pytest.raises(pyraichu.SimulationError, match="observer") as refused:
        pyraichu.fault_tree(model, targets=["T_lost"])
    assert "lags its condition" in str(refused.value)


def test_an_attribute_that_cannot_be_unrolled_is_refused_not_frozen():
    """An attribute a transition's effect writes keeps no expression of the
    states: frozen at its initial value it would make the tree a constant."""
    model = pyraichu.load_model(
        {
            "name": "counter",
            "components": [
                {
                    "name": "A",
                    "attributes": [
                        {
                            "name": "count",
                            "kind": "int",
                            "init": {"kind": "int", "value": 0},
                        }
                    ],
                    "automata": [
                        {
                            "name": "health",
                            "states": ["ok", "nok"],
                            "init": "ok",
                            "transitions": [
                                {
                                    "name": "fail",
                                    "source": "ok",
                                    "targets": ["nok"],
                                    "distrib": "exp",
                                    "rate": 1e-3,
                                    "effects": [
                                        {
                                            "target": {
                                                "component": "A",
                                                "attribute": "count",
                                            },
                                            "value": {
                                                "op": "const",
                                                "value": {"kind": "int", "value": 1},
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    top = {
        "op": "cmp",
        "cmp": "ge",
        "lhs": {"op": "attr", "attr": {"component": "A", "attribute": "count"}},
        "rhs": {"op": "const", "value": {"kind": "int", "value": 1}},
    }
    with pytest.raises(pyraichu.SimulationError, match="A.count") as refused:
        pyraichu.fault_tree(model, top)
    assert "cannot unroll" in str(refused.value)


def test_an_unknown_target_and_an_ambiguous_top_are_refused():
    model = study_model(declaration("parallel"))
    with pytest.raises(pyraichu.SimulationError, match="T_lost"):
        pyraichu.fault_tree(model, targets=["nope"])
    with pytest.raises(pyraichu.SimulationError, match="either"):
        pyraichu.fault_tree(model)


# --- 3. the envelope ----------------------------------------------------


ENVELOPE_KEYS = {
    "format",
    "version",
    "provenance",
    "top",
    "measure",
    "generation",
    "settings",
    "instants",
    "horizon",
}
HORIZON_KEYS = {
    "mission_time",
    "probability",
    "method",
    "exact",
    "upper_bound",
    "coherent",
    "cut_set_count",
    "cut_sets_complete",
    "cut_sets_omitted",
    "minimal_cut_sets",
    "importance",
    "warnings",
    "provenance",
}
IMPORTANCE_KEYS = {
    "event",
    "probability",
    "birnbaum",
    "criticality",
    "fussell_vesely",
    "diagnostic",
    "risk_achievement_worth",
    "risk_reduction_worth",
}


def test_the_envelope_has_the_documented_shape():
    tree = pyraichu.fault_tree(study_model(declaration("parallel")), targets=["T_lost"])
    times = [100.0, 500.0, MISSION]
    envelope = tree.envelope(times)
    assert set(envelope) == ENVELOPE_KEYS
    assert envelope["format"] == pyraichu.FAULT_TREE_FORMAT == "raichu.fault_tree"
    assert envelope["version"] == pyraichu.FAULT_TREE_VERSION == 1
    assert envelope["provenance"]["engine_version"] == pyraichu.__version__
    assert envelope["top"] == {"kind": "targets", "targets": ["T_lost"]}
    assert envelope["measure"] == "probability_without_repair"
    generation = envelope["generation"]
    assert generation["exact"] is True and generation["warnings"] == []
    assert [e["name"] for e in generation["basic_events"]] == [
        "B1.frun.failure",
        "B2.frun.failure",
    ]
    assert generation["basic_events"][0]["law"] == {"law": "exponential", "rate": 1e-3}
    assert [i["mission_time"] for i in envelope["instants"]] == times
    for instant in envelope["instants"]:
        t = instant["mission_time"]
        assert instant["probability"] == pytest.approx(
            failed("B1", t) * failed("B2", t), rel=CLOSED_FORM_RTOL
        )
        assert instant["method"] == "bdd" and instant["exact"] is True
    horizon = envelope["horizon"]
    assert set(horizon) == HORIZON_KEYS
    assert horizon["mission_time"] == MISSION
    assert horizon["probability"] == envelope["instants"][-1]["probability"]
    assert horizon["cut_set_count"] == 1 and horizon["cut_sets_complete"] is True
    (cut,) = horizon["minimal_cut_sets"]
    assert cut == {
        "events": ["B1.frun.failure", "B2.frun.failure"],
        "order": 2,
        "probability": pytest.approx(horizon["probability"], rel=1e-15),
    }
    assert [set(i) for i in horizon["importance"]] == [IMPORTANCE_KEYS] * 2
    # Read back through the engine's own reader.
    assert pyraichu.read_fault_tree(json.dumps(envelope)) == envelope


def test_an_envelope_of_another_version_or_too_many_instants_is_refused():
    tree = pyraichu.fault_tree(study_model(declaration("parallel")), targets=["T_lost"])
    envelope = tree.envelope([MISSION])
    later = dict(envelope, version=pyraichu.FAULT_TREE_VERSION + 1)
    with pytest.raises(pyraichu.SimulationError, match="version"):
        pyraichu.read_fault_tree(later)
    times = [float(t) for t in range(1, pyraichu.MAX_MISSION_TIMES + 2)]
    with pytest.raises(pyraichu.SimulationError, match="at most 20"):
        tree.envelope(times)
    with pytest.raises(pyraichu.SimulationError, match="strictly increasing"):
        tree.envelope([MISSION, 100.0])
