"""The "had value" measure crosses into Python as ``reached_*``.

Per trajectory it is 1 from the first time the indicator is active on, and
stays 1 after it falls back: the reference engine's ``realized`` computation,
whose mean over replicas is the probability of having reached the indicator
by each instant. The Rust suite pins its closed forms; this pins the
estimate the binding hands a launcher.
"""

import pyraichu

ON_THEN_OFF = {
    "name": "reached",
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
                        {"name": "enter", "source": "off", "targets": ["on"], "distrib": "delay", "time": 2.0},
                        {"name": "leave", "source": "on", "targets": ["off"], "distrib": "delay", "time": 2.0},
                    ],
                }
            ],
        }
    ],
    "indicators": [{"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"}],
}


def test_an_indicator_left_after_being_reached_stays_reached():
    estimates = pyraichu.monte_carlo(pyraichu.load_model(ON_THEN_OFF), nb_runs=3, t_max=5.0, samples=[1.0, 3.0, 5.0])
    indicator = estimates.indicators["c_on"]
    assert indicator.mean == [0.0, 1.0, 0.0]
    assert indicator.reached_mean == [0.0, 1.0, 1.0]
    assert indicator.reached_std == [0.0, 0.0, 0.0]
