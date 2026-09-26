"""A study's requested resolution for crossing detection, honoured.

The reference engine steps its PDMP solver on a base step a study
declares (`pdmp_dt`), so a study asking a finer one catches a shorter
episode. RAICHU locates crossings instead: it scans each accepted step at
`sub_samples` points of the dense output and bisects the first sign
change. A margin that holds for less than the scan spacing, `max_step /
sub_samples` at most, can fall between two points and be missed.

`event_resolution` is the widest spacing a study accepts between two scan
points: a floor on resolution. Finer than the engine's own, it adds scan
points and the short episode is seen; coarser, it changes nothing, bit
for bit. muscadet's `pdmp_dt` reaches it through the engine seam.
"""

import math

import pytest

import pyraichu

#: Half the width of the window during which `sin(t)` stays above the
#: threshold: 1e-4 in all, far below the default scan spacing (0.00625).
HALF_WIDTH = 5e-5
HORIZON = 3.0


def _const(value: float) -> dict:
    return {"op": "const", "value": {"kind": "float", "value": value}}


def short_episode_model() -> pyraichu.Model:
    """`x = sin(t)` and a watched transition into `seen` while `x` is within
    `1 - cos(HALF_WIDTH)` of its apex, a margin that holds for 1e-4."""
    return pyraichu.load_model(
        {
            "name": "short_episode",
            "components": [
                {
                    "name": "S",
                    "attributes": [
                        {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
                    ],
                    "equations": [
                        {
                            "target": "x",
                            "kind": "explicit",
                            "expr": {"op": "sin", "arg": {"op": "time"}},
                        }
                    ],
                    "automata": [
                        {
                            "name": "watch",
                            "states": ["waiting", "seen"],
                            "init": "waiting",
                            "transitions": [
                                {
                                    "name": "cross",
                                    "source": "waiting",
                                    "targets": ["seen"],
                                    "distrib": "watched",
                                    "guard": {
                                        "op": "cmp",
                                        "cmp": "ge",
                                        "lhs": {
                                            "op": "attr",
                                            "attr": {"component": "S", "attribute": "x"},
                                        },
                                        "rhs": _const(math.cos(HALF_WIDTH)),
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
            "indicators": [
                {
                    "name": "seen",
                    "target": "state",
                    "component": "S",
                    "automaton": "watch",
                    "state": "seen",
                }
            ],
        }
    )


def seen_at_horizon(**knobs) -> float:
    result = pyraichu.monte_carlo(
        short_episode_model(), nb_runs=1, t_max=HORIZON, samples=[HORIZON], **knobs
    )
    return list(result.indicators["seen"].mean)[-1]


def test_an_episode_shorter_than_the_scan_spacing_is_missed_by_default():
    assert seen_at_horizon() == 0.0


def test_a_finer_resolution_catches_it():
    assert seen_at_horizon(event_resolution=2e-5) == 1.0


def test_a_coarser_resolution_than_the_engine_s_own_changes_nothing():
    fine_sampling = [0.25 * index for index in range(1, 13)]
    default = pyraichu.monte_carlo(
        short_episode_model(), nb_runs=1, t_max=HORIZON, samples=fine_sampling
    )
    coarse = pyraichu.monte_carlo(
        short_episode_model(),
        nb_runs=1,
        t_max=HORIZON,
        samples=fine_sampling,
        event_resolution=1.0,
    )
    assert default == coarse


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_a_resolution_that_is_not_a_positive_time_is_refused(value):
    with pytest.raises(pyraichu.SimulationError, match="event_resolution"):
        seen_at_horizon(event_resolution=value)
