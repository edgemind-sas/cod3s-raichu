"""An offset must not destroy an indicator's measured dispersion."""

import math

import pyraichu
import pytest


def offset_model(offset):
    return pyraichu.load_model(
        {
            "name": "offset_variance",
            "components": [
                {
                    "name": "c",
                    "attributes": [
                        {
                            "name": "x",
                            "kind": "float",
                            "init": {"kind": "float", "value": offset},
                        }
                    ],
                    "automata": [
                        {
                            "name": "a",
                            "states": ["off", "on"],
                            "init": "off",
                            "transitions": [
                                {
                                    "name": "enter",
                                    "source": "off",
                                    "targets": ["on"],
                                    "distrib": "uniform",
                                    "low": 1.0,
                                    "high": 3.0,
                                    "effects": [
                                        {
                                            "target": {
                                                "component": "c",
                                                "attribute": "x",
                                            },
                                            "value": {
                                                "op": "const",
                                                "value": {
                                                    "kind": "float",
                                                    "value": offset + 1.0,
                                                },
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
            "indicators": [
                {
                    "name": "x",
                    "target": "attribute",
                    "attr": {"component": "c", "attribute": "x"},
                },
                {
                    "name": "on",
                    "target": "state",
                    "component": "c",
                    "automaton": "a",
                    "state": "on",
                },
            ],
        }
    )


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e8, 1e9, 1e12])
def test_offset_preserves_variance_and_normal_interval(offset):
    n = 2000
    estimates = pyraichu.monte_carlo(
        offset_model(offset),
        nb_runs=n,
        seed=17,
        t_max=2.0,
        samples=[0.0, 2.0],
        threads=1,
    )
    x = estimates.indicators["x"]
    p = estimates.indicators["on"].mean[1]
    # The state counts the same trials; the reference is the Bernoulli
    # sample-variance identity, not the engine's float variance.
    expected = math.sqrt(n / (n - 1) * p * (1 - p))
    assert x.std[1] == pytest.approx(expected, rel=1e-12)
    assert not x.ci.constant_sample[1]
    expected_half = 1.959963984540054 * expected / math.sqrt(n)
    # The endpoints themselves round on the offset's floating-point grid.
    assert x.ci.high[1] - x.mean[1] == pytest.approx(
        expected_half,
        abs=max(1e-12, math.ulp(offset)),
    )
    assert x.mean[1] - x.ci.low[1] == pytest.approx(
        expected_half,
        abs=max(1e-12, math.ulp(offset)),
    )
    assert x.std[0] == 0.0
    assert x.ci.constant_sample[0]


def test_offset_campaign_is_identical_across_threads_and_chunk_boundaries():
    model = offset_model(1e12)
    options = {
        "nb_runs": 70_000,
        "seed": 17,
        "t_max": 2.0,
        "samples": [0.0, 2.0],
        "quantiles": [0.0, 0.5, 1.0],
    }
    one = pyraichu.monte_carlo(model, threads=1, **options)
    many = pyraichu.monte_carlo(model, threads=4, **options)
    assert one.indicators == many.indicators
    assert one.indicators["x"].std[1] > 0.49
    assert one.indicators["x"].extremes.min[1] == 1e12
    assert one.indicators["x"].extremes.max[1] == 1e12 + 1.0
