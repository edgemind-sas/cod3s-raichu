"""The uniform occurrence law on the native `cls: ObjMode2S` wire.

cod3s 1.24.0 adds `{"cls": "uniform", "low": ..., "high": ...}`: a delay
drawn uniformly in [low, high] each time the direction arms, the first law
with two parameters per common-cause order. The engine already draws a
`{"distrib": "uniform", "low", "high"}` transition; this module locks the
reading of the wire and the drawn distribution against its closed form
(mean `(low + high) / 2`, CDF `(t - low) / (high - low)`).
"""

from __future__ import annotations

import math

import pyraichu
import pytest
from pyraichu.mode_objects import TranslationError, failure_mode_object
from pyraichu.plugins import expand_model

LOW, HIGH = 2.0, 12.0


def store(name: str) -> dict:
    return {
        "name": name,
        "attributes": [
            {"name": "working", "kind": "bool", "init": {"kind": "bool", "value": True}}
        ],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [],
        "equations": [],
    }


def mode(**fields):
    spec = {
        "cls": "ObjMode2S",
        "fm_name": "late",
        "targets": ["S"],
        "behaviour": "internal",
        "occ_state": "occ",
        "not_occ_state": "rep",
        "occ_law": {"cls": "uniform", "low": [LOW], "high": [HIGH]},
        "occ_param_name": ["ttf_low", "ttf_high"],
        "not_occ_law": {"cls": "exp", "rate": [0.0]},
        "not_occ_param_name": ["mu"],
        "occ_effects": {"working": False},
    }
    spec.update(fields)
    return failure_mode_object(spec)


def test_the_uniform_law_reads_into_a_uniform_transition():
    obj = mode()
    assert obj["failure"] == [{"law": "uniform", "low": LOW, "high": HIGH}]


def test_a_common_cause_vector_reads_one_pair_per_order():
    obj = mode(
        targets=["S", "T"],
        occ_law={"cls": "uniform", "low": [1.0, None], "high": [3.0, None]},
        not_occ_law={"cls": "exp", "rate": [0.0, 0.0]},
    )
    assert obj["failure"] == [{"law": "uniform", "low": 1.0, "high": 3.0}, None]


@pytest.mark.parametrize(
    "law, message",
    [
        ({"cls": "uniform", "low": [5.0], "high": [2.0]}, "low < high"),
        ({"cls": "uniform", "low": [2.0], "high": [2.0]}, "low < high"),
        ({"cls": "uniform", "low": [-1.0], "high": [2.0]}, "low < high"),
        ({"cls": "uniform", "low": [1.0]}, "'high'"),
        ({"cls": "uniform", "low": [1.0, 2.0], "high": [3.0]}, "differ in length"),
    ],
)
def test_a_malformed_uniform_law_is_refused(law, message):
    with pytest.raises(TranslationError, match=message):
        mode(occ_law=law)


def test_the_parameter_names_follow_the_law():
    with pytest.raises(TranslationError, match="disagrees"):
        mode(occ_param_name=["ttf"])


def test_the_drawn_date_matches_the_closed_form():
    body = expand_model(
        {
            "name": "uniform",
            "components": [store("S")],
            "connections": [],
            "indicators": [],
            "plugins": {"muscadet": {"objects": [mode()]}},
        }
    )
    body["indicators"] = [
        {
            "name": "S_working",
            "target": "attribute",
            "attr": {"component": "S", "attribute": "working"},
        }
    ]
    schedule = [0.0, 1.0] + [LOW + k * (HIGH - LOW) / 10 for k in range(11)]
    nb_runs = 20000
    result = pyraichu.monte_carlo(
        pyraichu.load_model(body),
        nb_runs=nb_runs,
        t_max=schedule[-1],
        samples=schedule,
        seed=20261109,
    )
    up = result.indicators["S_working"].mean
    cdf = [1.0 - u for u in up]
    for t, f in zip(schedule, cdf):
        expected = min(1.0, max(0.0, (t - LOW) / (HIGH - LOW)))
        tolerance = max(4.0 * math.sqrt(expected * (1.0 - expected) / nb_runs), 2e-3)
        assert abs(f - expected) <= tolerance, (t, f, expected)
    grid = schedule[2:]
    mean = LOW + sum(
        (1.0 - (cdf[2 + i] + cdf[3 + i]) / 2.0) * (grid[i + 1] - grid[i])
        for i in range(len(grid) - 1)
    )
    assert abs(mean - (LOW + HIGH) / 2.0) <= 4.0 * (HIGH - LOW) / math.sqrt(
        12.0 * nb_runs
    )


def _operator_session():
    obj = mode(
        not_occ_law={"cls": "uniform", "low": [1.0], "high": [3.0]},
        not_occ_param_name=["ttr_low", "ttr_high"],
    )
    body = expand_model(
        {
            "name": "uniform",
            "components": [store("S")],
            "connections": [],
            "indicators": [],
            "plugins": {"muscadet": {"objects": [obj]}},
        }
    )
    return pyraichu.Interactive(
        pyraichu.load_model(body), t_max=100.0, seed=1, operator_control=True
    )


def test_an_operator_session_reports_when_a_uniform_transition_armed():
    """The window of a uniform transition is `[armed_at + low, armed_at + high]`:
    an operator session draws nothing, so the arming instant and the law are
    what bound the date the operator may give."""
    sim = _operator_session()
    (armed,) = sim.deferred()
    assert armed["transition"] == "late.fm.failure"
    assert armed["armed_at"] == 0.0
    assert armed["law"] == {"distrib": "uniform", "low": LOW, "high": HIGH}

    sim.set_date("late.fm.failure", 5.0)
    # Still listed once dated: the date is the operator's, the arming is not.
    assert [d["armed_at"] for d in sim.deferred()] == [0.0]

    sim.fire("late.fm.failure")
    (repair,) = sim.deferred()
    assert repair["transition"] == "late.fm.repair"
    assert repair["armed_at"] == 5.0
    assert repair["law"] == {"distrib": "uniform", "low": 1.0, "high": 3.0}
