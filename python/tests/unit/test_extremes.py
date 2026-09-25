"""The ``min`` and ``max`` statistics cross into Python as ``*_extremes``.

The smallest and the largest value each measure took across the replicas, at
each instant. The Rust suite pins their order statistics; this pins the
estimate the binding hands a launcher.
"""

import pyraichu

UNIFORM_ENTRY = {
    "name": "extremes",
    "components": [
        {
            "name": "c",
            "attributes": [],
            "ports": [],
            "automata": [
                {
                    "name": "aut",
                    "states": ["off", "on"],
                    "init": "off",
                    "transitions": [
                        {"name": "enter", "source": "off", "targets": ["on"], "distrib": "uniform", "low": 1.0, "high": 3.0}
                    ],
                }
            ],
        }
    ],
    "indicators": [{"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"}],
}


def test_the_extremes_of_a_sojourn_bound_its_draws():
    estimates = pyraichu.monte_carlo(pyraichu.load_model(UNIFORM_ENTRY), nb_runs=2_000, t_max=4.0, samples=[2.0, 4.0])
    indicator = estimates.indicators["c_on"]
    assert isinstance(indicator.extremes, pyraichu.Extremes)
    assert (indicator.extremes.min[0], indicator.extremes.max[0]) == (0.0, 1.0)
    low, high = indicator.sojourn_extremes.min[1], indicator.sojourn_extremes.max[1]
    assert 1.0 <= low < 1.02 and 2.98 < high <= 3.0
    assert indicator.sojourn_extremes.min[1] <= indicator.sojourn_mean[1] <= indicator.sojourn_extremes.max[1]
