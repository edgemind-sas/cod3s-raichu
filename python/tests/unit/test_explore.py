"""Sequence-tree exploration from Python.

`pyraichu.explore` runs the exact or the discretised exploration driver
on the same model objects the Monte-Carlo entry points take, and returns
an `Exploration`:
the retained sequences to the feared event with their probabilities at
the horizon, the lower and upper bounds, what each cut-off discarded, and
the inconclusive flag. The result is saved and reloaded in its open
format, `raichu.exploration` v1, and reduces to minimal sequences through
the engine's own reduction. These tests pin the Python surface; the Rust
suite of `raichu-explore` pins the algorithm.
"""

import copy
import json
import math

import pyraichu
import pytest
from pyraichu.plugins import expand_model

RATE_A = 0.1
RATE_B = 0.03
REPAIR = 0.5
HORIZON = 10.0


def _objfm(name, target, rate, repair):
    return {
        "type": "ObjFM",
        "name": name,
        "targets": [target],
        "failure": [{"law": "exp", "rate": rate}],
        "repair": [repair],
        "failure_effects": {"flow": False},
    }


def _pair(repair_a=None, repair_b=None, rate_a=None, rate_b=None, repair=None):
    """Two components A and B, one ObjFM each, feared event: both down.
    The rates default to the module constants."""
    exp_repair = {"law": "exp", "rate": REPAIR if repair is None else repair}
    return {
        "name": "pair",
        "plugins": {
            "muscadet": {
                "objects": [
                    _objfm("fa", "A", RATE_A if rate_a is None else rate_a, repair_a or exp_repair),
                    _objfm("fb", "B", RATE_B if rate_b is None else rate_b, repair_b or exp_repair),
                    {
                        "type": "ObjEvent",
                        "name": "system_down",
                        "target": True,
                        "cond": [
                            [
                                {
                                    "obj": "A",
                                    "attr": "flow",
                                    "ope": "==",
                                    "value": False,
                                },
                                {
                                    "obj": "B",
                                    "attr": "flow",
                                    "ope": "==",
                                    "value": False,
                                },
                            ]
                        ],
                    },
                ]
            }
        },
        "components": [
            {
                "name": n,
                "attributes": [
                    {
                        "name": "flow",
                        "kind": "bool",
                        "init": {"kind": "bool", "value": True},
                    }
                ],
            }
            for n in ("A", "B")
        ],
    }


def _non_repairable():
    """The pair with its repair edges removed: two independent
    non-repairable exponential components, whose closed forms are known."""
    body = expand_model(_pair())
    for component in body["components"]:
        for automaton in component.get("automata", []):
            automaton["transitions"] = [
                t for t in automaton["transitions"] if t.get("kind") != "repair"
            ]
    return pyraichu.load_model(body)


def _hypoexp_cdf(q1, q2, t):
    """P(Exp(q1) + Exp(q2) <= t), distinct rates."""
    return 1.0 - (q2 * math.exp(-q1 * t) - q1 * math.exp(-q2 * t)) / (q2 - q1)


def _first_failure(sequence):
    return sequence.events[0]["obj"]


def test_two_independent_components_give_both_orderings_with_closed_forms():
    result = pyraichu.explore(_non_repairable(), "system_down", HORIZON)
    assert isinstance(result, pyraichu.Exploration)
    assert result.algorithm == "exact"
    assert result.target == "system_down"
    assert result.horizon == HORIZON
    assert [_first_failure(s) for s in result.sequences] == ["fa", "fb"]

    total = RATE_A + RATE_B
    a_then_b = RATE_A / total * _hypoexp_cdf(total, RATE_B, HORIZON)
    b_then_a = RATE_B / total * _hypoexp_cdf(total, RATE_A, HORIZON)
    for sequence, expected in zip(result.sequences, (a_then_b, b_then_a)):
        assert sequence.end_cause == "system_down"
        assert not sequence.imprecise
        assert sequence.probability == pytest.approx(expected, rel=1e-9)
        assert [e["obj"] for e in sequence.events][:2] in (["fa", "fb"], ["fb", "fa"])

    both_failed = (1 - math.exp(-RATE_A * HORIZON)) * (1 - math.exp(-RATE_B * HORIZON))
    assert result.lower == pytest.approx(both_failed, rel=1e-9)
    assert result.upper == result.lower
    assert not result.inconclusive
    assert result.cutoff_tallies["min_probability"]["pruned_nodes"] == 0


def test_a_repairable_pair_under_a_length_cut_keeps_the_direct_paths_and_bounds_the_rest():
    model = pyraichu.load_model(_pair())
    free = pyraichu.explore(model, "system_down", HORIZON, max_length=3)
    direct = [s for s in free.sequences if len(s.transitions) == 3]
    assert [_first_failure(s) for s in direct] == ["fa", "fb"]
    total = RATE_A + RATE_B
    expected = (
        RATE_A
        / total
        * RATE_B
        / (RATE_B + REPAIR)
        * _hypoexp_cdf(total, RATE_B + REPAIR, HORIZON)
    )
    assert direct[0].probability == pytest.approx(expected, rel=1e-9)
    tally = free.cutoff_tallies["max_length"]
    assert tally["pruned_nodes"] > 0
    assert free.upper == pytest.approx(free.lower + tally["mass"], rel=1e-12)
    assert free.upper > free.lower
    assert free.inconclusive
    assert free.cutoffs["max_length"] == 3


def test_a_law_armed_outside_the_exact_domain_names_the_transition():
    # A's repair is a fixed delay: armed only once A has failed.
    model = pyraichu.load_model(_pair(repair_a={"law": "delay", "time": 2.0}))
    with pytest.raises(pyraichu.SimulationError, match=r"fa\.fm\.repair") as refused:
        pyraichu.explore(model, "system_down", HORIZON)
    assert "[fa.fm.failure]" in str(refused.value)


def test_the_json_round_trip_reproduces_an_equal_result(tmp_path):
    result = pyraichu.explore(
        pyraichu.load_model(_pair()), "system_down", HORIZON, max_length=5
    )
    text = result.to_json()
    document = json.loads(text)
    assert document["format"] == "raichu.exploration"
    assert document["version"] == 1
    assert document["engine_version"] == pyraichu.__version__
    assert pyraichu.read_exploration(text) == result

    path = tmp_path / "exploration.json"
    result.to_json(path)
    assert pyraichu.read_exploration(path) == result
    assert pyraichu.read_exploration(str(path)) == result


def test_the_result_is_written_in_its_compact_form_and_read_back_exactly():
    result = pyraichu.explore(
        pyraichu.load_model(_pair()), "system_down", HORIZON, max_length=7
    )
    text = result.to_json()
    document = json.loads(text)
    # One table of distinct steps; each sequence lists indices into it.
    table = document["steps"]
    assert all(set(step) == {"transition", "from", "to", "event"} for step in table)
    for sequence in document["sequences"]:
        assert set(sequence) == {"steps", "end_cause", "probability", "error_bound", "imprecise"}
        assert all(isinstance(i, int) and 0 <= i < len(table) for i in sequence["steps"])
    assert pyraichu.read_exploration(text).to_json() == text


def test_the_step_table_is_deduplicated_and_thread_independent():
    model = pyraichu.load_model(_pair())
    one = pyraichu.explore(model, "system_down", HORIZON, max_length=7, threads=1)
    four = pyraichu.explore(model, "system_down", HORIZON, max_length=7, threads=4)
    assert one.steps == four.steps
    distinct = {
        (t["transition"], t["to"]) for s in one.sequences for t in s.transitions
    }
    assert len(one.steps) == len(distinct)
    fired = sum(len(s.steps) for s in one.sequences)
    assert fired > 10 * len(one.steps)


def test_resolved_transitions_and_events_keep_their_shapes():
    result = pyraichu.explore(_non_repairable(), "system_down", HORIZON)
    sequence = result.sequences[0]
    assert all(set(t) == {"transition", "from", "to"} for t in sequence.transitions)
    assert sequence.events
    assert all(set(e) == {"obj", "attr", "cycle_group"} for e in sequence.events)
    assert len(sequence.transitions) == len(sequence.steps)
    # Monitored steps only.
    monitored = [result.steps[i] for i in sequence.steps if result.steps[i]["event"]]
    assert sequence.events == [step["event"] for step in monitored]


def test_a_deep_exploration_result_stays_small():
    """Regression: a result used to repeat every step as full strings in
    every sequence, so its size grew with the square of the depth (67 MB
    of JSON here). With the step table it is a few megabytes."""
    model = pyraichu.load_model(
        expand_model(
            _pair(
                repair_b={"law": "exp", "rate": 1e-12},
                rate_a=1.0,
                rate_b=1e-6,
                repair=1.0,
            )
        )
    )
    result = pyraichu.explore(model, "system_down", 1000.0, min_probability=1e-12)
    fired = sum(len(s.steps) for s in result.sequences)
    assert fired > 500_000, fired
    assert len(result.steps) <= 10
    assert len(result.to_json()) < 10_000_000


def test_reading_refuses_another_format():
    result = pyraichu.explore(_non_repairable(), "system_down", HORIZON)
    document = json.loads(result.to_json())
    document["format"] = "raichu.sequences"
    with pytest.raises(pyraichu.SimulationError, match="raichu.exploration"):
        pyraichu.read_exploration(json.dumps(document))
    document["format"] = "raichu.exploration"
    document["version"] = 3
    with pytest.raises(pyraichu.SimulationError, match="version"):
        pyraichu.read_exploration(json.dumps(document))
    document["version"] = 1
    document["sequences"][0]["steps"].append(len(document["steps"]))
    with pytest.raises(pyraichu.SimulationError, match="step table"):
        pyraichu.read_exploration(json.dumps(document))


def test_minimal_sequences_preserve_the_total_probability():
    result = pyraichu.explore(
        pyraichu.load_model(_pair()), "system_down", HORIZON, max_length=7
    )
    assert len(result.sequences) > 2
    minimal = result.minimal_sequences()
    assert 0 < len(minimal) < len(result.sequences)
    assert sum(s["weight"] for s in minimal) == pytest.approx(result.lower, rel=1e-12)
    assert all(s["end_cause"] == "system_down" for s in minimal)


def test_an_invalid_setting_raises_before_exploration():
    model = _non_repairable()
    with pytest.raises(pyraichu.SimulationError, match="horizon"):
        pyraichu.explore(model, "system_down", -1.0)
    with pytest.raises(pyraichu.SimulationError, match="min_probability"):
        pyraichu.explore(model, "system_down", HORIZON, min_probability=0.0)
    with pytest.raises(pyraichu.SimulationError, match="target"):
        pyraichu.explore(model, "no_such_event", HORIZON)


def test_an_algorithm_not_provided_is_refused_by_name():
    with pytest.raises(pyraichu.SimulationError, match="sampling"):
        pyraichu.explore(
            _non_repairable(), "system_down", HORIZON, algorithm="sampling"
        )


def test_the_thread_count_does_not_change_the_result():
    model = pyraichu.load_model(_pair())
    one = pyraichu.explore(model, "system_down", HORIZON, max_length=7, threads=1)
    four = pyraichu.explore(model, "system_down", HORIZON, max_length=7, threads=4)
    assert one == four


def test_the_domain_report_lists_an_ode_and_exploring_refuses_it():
    document = copy.deepcopy(_pair())
    component = document["components"][0]
    component["attributes"].append(
        {"name": "level", "kind": "float", "init": {"kind": "float", "value": 0.0}}
    )
    component["equations"] = [
        {
            "target": "level",
            "kind": "ode",
            "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}},
        }
    ]
    model = pyraichu.load_model(document)
    report = pyraichu.exploration_domain(model)
    assert [(v["kind"], v["attribute"]) for v in report] == [("ode", "A.level")]
    assert "A.level" in report[0]["message"]
    with pytest.raises(pyraichu.SimulationError, match="A.level"):
        pyraichu.explore(model, "system_down", HORIZON)


def test_a_model_inside_the_domain_has_an_empty_report():
    assert pyraichu.exploration_domain(_non_repairable()) == []


# ---- discretised algorithm ----------------------------------------------------

WEIBULL_SHAPE = 1.5
WEIBULL_SCALE = 3.0
WEIBULL_HORIZON = 2.0


def _unit(name, law, guard=None):
    """A non-repairable unit `name.fail` (`ok`, `nok`) failing under `law`,
    in the core schema (the muscadet plugin carries no Weibull law)."""
    occ = {"name": "occ", "source": "ok", "targets": ["nok"], "monitored": True, **law}
    if guard is not None:
        occ["guard"] = guard
    return {
        "name": name,
        "automata": [
            {"name": "fail", "states": ["ok", "nok"], "init": "ok", "transitions": [occ]}
        ],
    }


def _weibull(shape, scale):
    return {"distrib": "weibull", "shape": shape, "scale": scale}


def _down(name):
    return {
        "op": "state_active",
        "state": {"component": name, "automaton": "fail", "state": "nok"},
    }


def _single_weibull():
    return pyraichu.load_model(
        {
            "name": "single_weibull",
            "components": [_unit("W", _weibull(WEIBULL_SHAPE, WEIBULL_SCALE))],
            "targets": [{"name": "w_down", "component": "W", "automaton": "fail", "state": "nok"}],
        }
    )


def _two_weibulls():
    """A (Weibull 2, 1.5), then B (Weibull 2, 1) armed once A has failed;
    target: B down. The instant A fails at decides how long B has, so the
    discretisation error is not zero."""
    return pyraichu.load_model(
        {
            "name": "two_weibulls",
            "components": [
                _unit("A", _weibull(2.0, 1.5)),
                _unit("B", _weibull(2.0, 1.0), guard=_down("A")),
            ],
            "targets": [{"name": "b_down", "component": "B", "automaton": "fail", "state": "nok"}],
        }
    )


def _two_weibulls_closed_form(t, n=20_000):
    """P(T_A + T_B <= t) by Simpson's rule on int_0^t f_A(s) F_B(t - s) ds."""

    def f_a(s):
        return (2.0 * s / 2.25) * math.exp(-((s / 1.5) ** 2))

    def cdf_b(s):
        return -math.expm1(-(max(s, 0.0) ** 2))

    h = t / n
    total = f_a(0.0) * cdf_b(t) + f_a(t) * cdf_b(0.0)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f_a(i * h) * cdf_b(t - i * h)
    return total * h / 3.0


def test_a_weibull_failure_explored_discretised_matches_its_closed_form():
    result = pyraichu.explore(
        _single_weibull(), "w_down", WEIBULL_HORIZON, algorithm="discretised"
    )
    assert result.algorithm == "discretised"
    assert result.discretisation["level"] == 16
    assert result.discretisation["refinement"]["base_level"] == 8
    expected = -math.expm1(-((WEIBULL_HORIZON / WEIBULL_SCALE) ** WEIBULL_SHAPE))
    estimate = result.error_estimate
    assert estimate is not None
    assert len(result.sequences) == 1
    assert [t["transition"] for t in result.sequences[0].transitions] == ["W.fail.occ"]
    assert abs(result.lower - expected) <= max(2.0 * estimate, 1e-9)
    assert result.upper == result.lower


def test_a_sequence_of_two_weibulls_lies_within_twice_its_error_estimate():
    t = 2.0
    result = pyraichu.explore(_two_weibulls(), "b_down", t, algorithm="discretised")
    estimate = result.error_estimate
    assert estimate > 0
    assert not result.discretisation["refinement"]["truncation_dominated"]
    expected = _two_weibulls_closed_form(t)
    assert abs(result.lower - expected) <= max(2.0 * estimate, 1e-9)


def test_the_exact_algorithm_refuses_the_weibull_model_by_name():
    with pytest.raises(pyraichu.SimulationError, match=r"W\.fail\.occ"):
        pyraichu.explore(_single_weibull(), "w_down", WEIBULL_HORIZON)


def test_the_json_round_trip_keeps_the_level_and_the_error_estimate(tmp_path):
    result = pyraichu.explore(
        _two_weibulls(), "b_down", 2.0, algorithm="discretised", level=4
    )
    text = result.to_json()
    document = json.loads(text)
    assert document["algorithm"] == "discretised"
    # Format version 2 introduced the discretised algorithm; an exact result
    # stays at version 1.
    assert document["version"] == 2
    assert document["discretisation"]["level"] == 8
    assert document["discretisation"]["refinement"]["base_level"] == 4
    again = pyraichu.read_exploration(text)
    assert again == result
    assert again.discretisation == result.discretisation
    assert again.error_estimate == result.error_estimate
    assert again.to_json() == text

    path = tmp_path / "discretised.json"
    result.to_json(path)
    assert pyraichu.read_exploration(path) == result

    # A discretised document declaring version 1 is inconsistent: a version-1
    # writer knew no such algorithm.
    document["version"] = 1
    with pytest.raises(pyraichu.SimulationError, match="inconsistent"):
        pyraichu.read_exploration(json.dumps(document))


def test_switching_the_refinement_off_yields_no_estimate():
    result = pyraichu.explore(
        _two_weibulls(), "b_down", 2.0, algorithm="discretised", level=4, refine=False
    )
    assert result.discretisation == {"level": 4, "refinement": None}
    assert result.error_estimate is None
    document = json.loads(result.to_json())
    assert document["discretisation"]["refinement"] is None
    assert pyraichu.read_exploration(result.to_json()).error_estimate is None


def test_an_exact_result_carries_no_discretisation():
    result = pyraichu.explore(_non_repairable(), "system_down", HORIZON)
    assert result.discretisation is None
    assert result.error_estimate is None
    text = result.to_json()
    assert "discretisation" not in json.loads(text)
    assert pyraichu.read_exploration(text).to_json() == text


def test_settings_of_one_algorithm_are_refused_by_the_other():
    model = _single_weibull()
    with pytest.raises(pyraichu.SimulationError, match="level"):
        pyraichu.explore(_non_repairable(), "system_down", HORIZON, level=4)
    with pytest.raises(pyraichu.SimulationError, match="refine"):
        pyraichu.explore(_non_repairable(), "system_down", HORIZON, refine=False)
    with pytest.raises(pyraichu.SimulationError, match="rel_precision"):
        pyraichu.explore(
            model, "w_down", WEIBULL_HORIZON, algorithm="discretised", rel_precision=1e-9
        )
    with pytest.raises(pyraichu.SimulationError, match="max_terms"):
        pyraichu.explore(
            model, "w_down", WEIBULL_HORIZON, algorithm="discretised", max_terms=100
        )
    with pytest.raises(pyraichu.SimulationError, match="level"):
        pyraichu.explore(model, "w_down", WEIBULL_HORIZON, algorithm="discretised", level=0)


def test_the_discretised_result_does_not_depend_on_the_thread_count():
    model = _two_weibulls()
    one = pyraichu.explore(model, "b_down", 2.0, algorithm="discretised", level=4, threads=1)
    four = pyraichu.explore(model, "b_down", 2.0, algorithm="discretised", level=4, threads=4)
    assert one == four
