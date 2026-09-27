"""Exact fault-tree quantification (``pyraichu.quantify``).

The Rust suite pins the decision-diagram kernel against closed forms and
against the truth table of random trees; this pins what the binding hands
a user: a tree generated from a model, quantified at a mission time, and
the same number drawn by the simulator on that model.
"""

from __future__ import annotations

import math

import pyraichu
import pytest


def unit(name, rate):
    return {
        "name": name,
        "attributes": [],
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
                        "rate": rate,
                    }
                ],
            }
        ],
    }


def nok(name):
    return {
        "op": "state_active",
        "state": {"component": name, "automaton": "health", "state": "nok"},
    }


# A pump backed by two redundant ones: the plant is lost when A fails and
# at least one of B and C has failed too.
TOP = {
    "op": "bool",
    "bool_op": "and",
    "args": [nok("A"), {"op": "bool", "bool_op": "or", "args": [nok("B"), nok("C")]}],
}
RATES = {"A": 1e-3, "B": 2e-3, "C": 5e-4}


def plant(with_observer=False):
    components = [unit(name, rate) for name, rate in RATES.items()]
    indicators = []
    if with_observer:
        # The top expression as a state the simulator can be asked about.
        components.append(
            {
                "name": "plant",
                "attributes": [],
                "automata": [
                    {
                        "name": "status",
                        "states": ["up", "lost"],
                        "init": "up",
                        "transitions": [
                            {
                                "name": "trip",
                                "source": "up",
                                "targets": ["lost"],
                                "distrib": "inst",
                                "probs": [],
                                "guard": TOP,
                            }
                        ],
                    }
                ],
            }
        )
        indicators.append(
            {
                "name": "lost",
                "target": "state",
                "component": "plant",
                "automaton": "status",
                "state": "lost",
            }
        )
    return pyraichu.load_model(
        {"name": "plant", "components": components, "indicators": indicators}
    )


def occurred(rate, t):
    return 1.0 - math.exp(-rate * t)


def test_a_generated_tree_is_quantified_at_its_mission_time():
    t = 500.0
    result = pyraichu.fault_tree(plant(), TOP).quantify(mission_time=t)
    a, b, c = (occurred(RATES[n], t) for n in "ABC")
    assert result.probability == pytest.approx(a * (1 - (1 - b) * (1 - c)), rel=1e-14)
    assert result.method == "bdd" and result.exact and result.coherent
    assert result.minimal_cut_sets == [
        ["A.health.fail", "B.health.fail"],
        ["A.health.fail", "C.health.fail"],
    ]
    assert result.cut_set_count == 2
    by_name = {i.name: i for i in result.importance}
    # A is in every cut set: making it impossible removes the whole risk.
    assert by_name["A.health.fail"].risk_reduction_worth is None
    assert by_name["A.health.fail"].fussell_vesely == pytest.approx(1.0)
    assert by_name["A.health.fail"].birnbaum == pytest.approx(1 - (1 - b) * (1 - c))


def test_the_exact_probability_is_what_the_simulator_draws():
    """An independent witness: the same model, simulated, with no repair."""
    t = 500.0
    exact = pyraichu.fault_tree(plant(), TOP).quantify(mission_time=t).probability
    runs = 40_000
    estimates = pyraichu.monte_carlo(
        plant(with_observer=True), nb_runs=runs, t_max=t, samples=[t], seed=11
    )
    drawn = estimates.indicators["lost"].mean[0]
    sigma = math.sqrt(exact * (1 - exact) / runs)
    assert abs(drawn - exact) < 4 * sigma, (drawn, exact, sigma)


def test_a_timed_tree_without_a_mission_time_is_refused_by_name():
    tree = pyraichu.fault_tree(plant(), TOP)
    with pytest.raises(pyraichu.SimulationError, match="mission time"):
        tree.quantify()


def test_an_open_psa_document_is_quantified_from_text_or_path(tmp_path):
    document = """<?xml version="1.0"?>
<opsa-mef>
  <define-fault-tree name="demo">
    <define-gate name="TOP">
      <or><basic-event name="V"/><not><basic-event name="S"/></not></or>
    </define-gate>
  </define-fault-tree>
  <model-data>
    <define-basic-event name="V"><float value="0.01"/></define-basic-event>
    <define-basic-event name="S"><float value="0.9"/></define-basic-event>
  </model-data>
</opsa-mef>"""
    from_text = pyraichu.quantify(document)
    path = tmp_path / "demo.xml"
    path.write_text(document, encoding="utf-8")
    from_path = pyraichu.quantify(path)
    exact = 1 - (1 - 0.01) * 0.9
    assert from_text.probability == pytest.approx(exact, rel=1e-14)
    assert from_path.probability == from_text.probability
    # The negation survives: exact probability, no cut sets, and why.
    assert not from_text.coherent
    assert from_text.minimal_cut_sets is None
    assert "not monotone" in from_text.cut_sets_omitted


def test_cut_sets_can_be_skipped():
    tree = pyraichu.fault_tree(plant(), TOP)
    full = tree.quantify(mission_time=100.0)
    light = tree.quantify(mission_time=100.0, cut_sets=False)
    assert light.probability == full.probability
    assert light.importance == full.importance
    assert light.minimal_cut_sets is None and light.cut_set_count is None
    assert light.cut_sets_omitted == "not requested"


def test_the_provenance_records_the_variable_order():
    result = pyraichu.fault_tree(plant(), TOP).quantify(mission_time=100.0)
    provenance = result.provenance
    assert provenance["variable_order"] == "depth_first_left_most"
    variables = [v for m in provenance["modules"] for v in m["variables"]]
    assert "A.health.fail" in variables
