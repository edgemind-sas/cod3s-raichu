"""A fixed exponential rate of 0 is a transition that never fires.

It used to be refused when the model was loaded, so a study neutralising
a failure mode by setting its rate to 0 did not run.
"""

import math

import pyraichu

HORIZON = 100.0
RATE = 1e-2


def _unit(name: str, rate: float) -> dict:
    return {
        "name": name,
        "automata": [
            {
                "name": "health",
                "states": ["ok", "ko"],
                "init": "ok",
                "transitions": [
                    {
                        "name": "fail",
                        "source": "ok",
                        "targets": ["ko"],
                        "distrib": "exp",
                        "rate": rate,
                        "kind": "failure",
                    }
                ],
            }
        ],
    }


def _model(*units: dict) -> pyraichu.Model:
    return pyraichu.load_model(
        {
            "name": "dormant",
            "components": list(units),
            "indicators": [
                {
                    "name": f"{unit['name']}_down",
                    "target": "state",
                    "component": unit["name"],
                    "automaton": "health",
                    "state": "ko",
                }
                for unit in units
            ],
            "targets": [
                {"name": "b_down", "component": "b", "automaton": "health", "state": "ko"}
            ],
        }
    )


def test_a_dormant_transition_never_fires_and_leaves_the_others_alone():
    samples = [HORIZON / 2, HORIZON]
    alone = pyraichu.monte_carlo(
        _model(_unit("b", RATE)), nb_runs=500, t_max=HORIZON, samples=samples, seed=5
    )
    beside = pyraichu.monte_carlo(
        _model(_unit("a", 0.0), _unit("b", RATE)),
        nb_runs=500,
        t_max=HORIZON,
        samples=samples,
        seed=5,
    )
    assert list(beside.indicators["a_down"].mean) == [0.0, 0.0]
    # Not one random number went to the dormant transition.
    assert list(beside.indicators["b_down"].mean) == list(alone.indicators["b_down"].mean)


def test_the_exact_exploration_does_not_branch_on_it():
    result = pyraichu.explore(_model(_unit("a", 0.0), _unit("b", RATE)), "b_down", HORIZON)
    assert len(result.sequences) == 1
    expected = 1.0 - math.exp(-RATE * HORIZON)
    assert abs(result.lower - expected) < 1e-12
    assert abs(result.upper - expected) < 1e-12
