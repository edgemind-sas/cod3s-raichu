"""What the importance measures cost, which is the property they live or
die by: an answer that doubled the duration of a campaign would not be
asked for twice.

Two things are measured here, and they are different claims.

**One campaign, not two.** The measures are a post-processing pass over a
campaign that already had to run, so on a model whose cost is the
simulation, asking for them adds a fraction and not a factor. The
reference used is the project's own hybrid benchmark (room-temperature
ODE, watched thermostats, stochastic failures), annotated for sequence
analysis.

**Linear in the replicas.** A study grows by running more replicas, so
what decides whether the measures survive the team's load levels is the
shape of the growth, not the constant. Quadratic would quadruple the time
when the campaign doubles; the guard below is far from that, and would
have caught the quadratic grouping that this work found and removed.

Timing assertions, so the bounds are deliberately loose: they are there to
catch a change of *regime*, not to police a percentage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import pyraichu

REPO = Path(__file__).resolve().parents[3]
HYBRID = REPO / "benchmarks" / "models" / "heated_room_s3.json"


def _fastest(call, repeats: int = 3) -> float:
    """The fastest of `repeats` runs: the one least polluted by whatever
    else the machine was doing."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _state_is(component: str, state: str) -> dict:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": "dysfunctional", "state": state},
    }


def hybrid_model():
    """The heated-room benchmark, annotated for sequence analysis: both
    heaters KO is the feared event."""
    body = json.loads(HYBRID.read_text())
    for component in body["components"]:
        for automaton in component.get("automata", []):
            if automaton["name"] == "dysfunctional":
                for transition in automaton["transitions"]:
                    transition["monitored"] = True
                    transition["cycle_group"] = "dysfunctional"
    both_ko = {
        "op": "bool",
        "bool_op": "and",
        "args": [_state_is("aMasterHeater", "KO"), _state_is("aSlaveHeater", "KO")],
    }
    body["components"].append(
        {
            "name": "system_down",
            "attributes": [],
            "ports": [],
            "interfaces": [],
            "automata": [
                {
                    "name": "ev",
                    "states": ["not_occ", "occ"],
                    "init": "not_occ",
                    "transitions": [
                        {
                            "name": "occ",
                            "source": "not_occ",
                            "targets": ["occ"],
                            "monitored": True,
                            "cycle_group": "ev",
                            "distrib": "delay",
                            "time": 0.0,
                            "guard": both_ko,
                        },
                        {
                            "name": "not_occ",
                            "source": "occ",
                            "targets": ["not_occ"],
                            "monitored": True,
                            "cycle_group": "ev",
                            "distrib": "delay",
                            "time": 0.0,
                            "guard": {"op": "bool", "bool_op": "not", "args": [both_ko]},
                        },
                    ],
                }
            ],
            "sensitive_functions": [],
            "equations": [],
        }
    )
    body["targets"] = [
        {
            "name": "system_down",
            "component": "system_down",
            "automaton": "ev",
            "state": "occ",
        }
    ]
    return pyraichu.load_model(body)


def discrete_model(pairs: int):
    """`pairs` redundant pairs in series: purely discrete and as cheap to
    simulate as a model gets, so the post-processing is as visible as it
    will ever be. The worst case for the ratio, and the right case for
    measuring the shape of the growth."""
    blocks = [f"{side}{i}" for i in range(pairs) for side in ("a", "b")]
    objects = [
        {
            "type": "ObjFM",
            "name": f"fm_{block}",
            "targets": [block],
            "failure": [{"law": "exp", "rate": 0.01}],
            "repair": [{"law": "exp", "rate": 0.05}],
            "failure_effects": {"flow": False},
        }
        for block in blocks
    ] + [
        {
            "type": "ObjEvent",
            "name": "system_down",
            "target": True,
            "cond": [
                [
                    {"obj": f"a{i}", "attr": "flow", "ope": "==", "value": False},
                    {"obj": f"b{i}", "attr": "flow", "ope": "==", "value": False},
                ]
                for i in range(pairs)
            ],
        },
    ]
    return pyraichu.load_model(
        {
            "name": "plant",
            "plugins": {"muscadet": {"objects": objects}},
            "components": [
                {
                    "name": block,
                    "attributes": [
                        {
                            "name": "flow",
                            "kind": "bool",
                            "init": {"kind": "bool", "value": True},
                        }
                    ],
                }
                for block in blocks
            ],
            "indicators": [],
        }
    )


@pytest.mark.skipif(not HYBRID.exists(), reason="benchmark model not in this checkout")
def test_importance_costs_a_fraction_of_a_hybrid_campaign():
    """On a model whose cost is the simulation, the measures ride on the
    campaign instead of adding one. Measured at ~1.15x when this was
    written; the guard is at 2, the point past which the ticket says the
    answer would stop being asked for."""
    model = hybrid_model()
    instants = [100.0 * k for k in range(1, 11)]
    plain = _fastest(
        lambda: pyraichu.monte_carlo(
            model, nb_runs=500, t_max=1000.0, samples=instants, seed=1
        )
    )
    measured = _fastest(
        # A native model that declares no failure role: every recorded
        # state counts.
        lambda: pyraichu.importance(
            model,
            nb_runs=500,
            t_max=1000.0,
            instants=instants,
            seed=1,
            basic_events="monitored",
        )
    )
    assert measured < 2.0 * plain, (
        f"importance took {measured:.3f}s against {plain:.3f}s for the bare "
        f"campaign ({measured / plain:.2f}x)"
    )


def test_the_analysis_grows_linearly_with_the_replicas():
    """Four times the replicas, at most six times the time. Linear would
    be four and quadratic sixteen, so this separates the two regimes
    without pretending to measure a constant."""
    model = discrete_model(pairs=8)
    instants = [10.0 * k for k in range(1, 11)]

    def campaign(nb_runs: int):
        return lambda: pyraichu.importance(
            model, nb_runs=nb_runs, t_max=100.0, instants=instants, seed=1
        )

    small = _fastest(campaign(4_000))
    large = _fastest(campaign(16_000))
    assert large < 6.0 * small, (
        f"4x the replicas took {large / small:.2f}x the time "
        f"({small:.3f}s then {large:.3f}s): the reduction is not linear"
    )
