"""The normal law of a transition, truncated to its interval.

A duration cannot be negative, so the law takes a lower bound (0 unless
stated) and an optional upper one, and draws from the normal law
conditioned on that interval. The closed form is written out here with
``math.erf`` so the engine is compared against the formula and not
against itself.
"""

import math

import pytest

import pyraichu


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncated_cdf(t, mu, sigma, low, high):
    def at(x):
        return _phi((x - mu) / sigma)

    upper = 1.0 if high is None else at(high)
    clamped = max(t, low) if high is None else min(max(t, low), high)
    return (at(clamped) - at(low)) / (upper - at(low))


def _model(law: dict) -> dict:
    return {
        "name": "normal",
        "components": [
            {
                "name": "C",
                "automata": [
                    {
                        "name": "aut",
                        "states": ["ok", "nok"],
                        "init": "ok",
                        "transitions": [
                            {"name": "fire", "source": "ok", "targets": ["nok"], **law}
                        ],
                    }
                ],
            }
        ],
        "indicators": [
            {
                "name": "fired",
                "target": "state",
                "component": "C",
                "automaton": "aut",
                "state": "nok",
            }
        ],
        "targets": [
            {"name": "done", "component": "C", "automaton": "aut", "state": "nok"}
        ],
    }


@pytest.mark.parametrize(
    ("law", "bounds", "instants"),
    [
        # Default bounds: 15.9 % of the untruncated mass lies below 0.
        ({"mu": 10.0, "sigma": 10.0}, (0.0, None), [1.0, 10.0, 25.0]),
        ({"mu": 2.0, "sigma": 3.0, "low": 1.0, "high": 6.0}, (1.0, 6.0), [2.0, 4.0, 7.0]),
    ],
)
def test_the_firing_probability_follows_the_truncated_law(law, bounds, instants):
    model = pyraichu.load_model(_model({"distrib": "normal", **law}))
    runs = 20_000
    estimates = pyraichu.monte_carlo(
        model, nb_runs=runs, t_max=instants[-1], samples=instants, seed=11
    )
    for t, mean in zip(instants, estimates.indicators["fired"].mean):
        expected = _truncated_cdf(t, law["mu"], law["sigma"], *bounds)
        tolerance = 4.0 * math.sqrt(max(expected * (1.0 - expected), 1e-4) / runs)
        assert abs(mean - expected) < tolerance, (t, mean, expected)


def test_the_discretised_exploration_reads_the_same_law():
    law = {"distrib": "normal", "mu": 10.0, "sigma": 10.0}
    model = pyraichu.load_model(_model(law))
    result = pyraichu.explore(model, "done", 25.0, algorithm="discretised")
    expected = _truncated_cdf(25.0, 10.0, 10.0, 0.0, None)
    assert result.lower - 1e-9 <= expected <= result.upper + 1e-9
    assert result.upper - result.lower < 0.05


@pytest.mark.parametrize(
    "law",
    [
        {"mu": 10.0, "sigma": 0.0},
        {"mu": 10.0, "sigma": 2.0, "low": -1.0},
        {"mu": 10.0, "sigma": 2.0, "low": 5.0, "high": 5.0},
        # Fifty deviations above the mean: nothing to renormalise.
        {"mu": 0.0, "sigma": 1.0, "low": 50.0},
    ],
)
def test_an_invalid_law_is_refused_when_the_model_is_loaded(law):
    with pytest.raises(Exception, match="normal"):
        pyraichu.load_model(_model({"distrib": "normal", **law}))
