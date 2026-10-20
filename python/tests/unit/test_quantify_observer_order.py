"""A zero-delay repair must not erase an observed failure in exploration."""

import math

import pyraichu
import pytest


def observed_failure(observer_first=False):
    plant = {
        "name": "plant",
        "automata": [
            {
                "name": "a",
                "states": ["ok", "failed", "repaired"],
                "init": "ok",
                "transitions": [
                    {
                        "name": "fail",
                        "source": "ok",
                        "targets": ["failed"],
                        "distrib": "exp",
                        "rate": 1.0,
                    },
                    {
                        "name": "repair",
                        "source": "failed",
                        "targets": ["repaired"],
                        "distrib": "delay",
                        "time": 0.0,
                    },
                ],
            }
        ],
    }
    observer = {
        "name": "observer",
        "automata": [
            {
                "name": "a",
                "states": ["off", "on"],
                "init": "off",
                "transitions": [
                    {
                        "name": "detect",
                        "source": "off",
                        "targets": ["on"],
                        "distrib": "delay",
                        "time": 0.0,
                        "kind": "observation",
                        "guard": {
                            "op": "state_active",
                            "state": {
                                "component": "plant",
                                "automaton": "a",
                                "state": "failed",
                            },
                        },
                    }
                ],
            }
        ],
    }
    return pyraichu.load_model(
        pyraichu.seal(
            {
                "name": "zero_repair_observer",
                "components": (
                    [observer, plant] if observer_first else [plant, observer]
                ),
                "targets": [
                    {
                        "name": "lost",
                        "component": "observer",
                        "automaton": "a",
                        "state": "on",
                    }
                ],
            }
        )
    )


@pytest.mark.parametrize("observer_first", [False, True])
@pytest.mark.parametrize(
    ("method", "level"),
    [("exact", None), ("discretised", 1), ("discretised", 3), ("discretised", 6)],
)
def test_exploration_observes_a_failure_before_its_immediate_repair(
    observer_first, method, level
):
    study = pyraichu.Study("lost", 1.0, seed=7)
    settings = {} if level is None else {"level": level}
    probability = pyraichu.quantify(
        observed_failure(observer_first), study, method=method, **settings
    ).probability
    expected = -math.expm1(-1.0)
    assert probability.low == pytest.approx(expected, abs=1e-12)
    assert probability.high == pytest.approx(expected, abs=1e-12)
    assert probability.inconclusive is False
    if method == "discretised":
        # This constant-rate case is already exact on the coarsest mesh.
        assert probability.error_estimate == pytest.approx(0.0, abs=1e-12)


def test_simulation_records_the_failure_even_though_its_duration_is_zero():
    model = observed_failure()
    events = pyraichu.simulate(model, t_max=1.0, seed=7).events
    assert [event.transition for event in events] == [
        "plant.a.fail",
        "observer.a.detect",
        "plant.a.repair",
    ]
    assert events[0].time == events[1].time == events[2].time
    probability = pyraichu.quantify(
        model,
        pyraichu.Study("lost", 1.0, seed=7),
        method="monte_carlo",
        nb_runs=10_000,
        confidence=0.99,
    ).probability
    assert probability.low <= -math.expm1(-1.0) <= probability.high
