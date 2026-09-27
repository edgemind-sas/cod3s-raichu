"""Tutorial 3: a Monte-Carlo unavailability estimate against its closed form.

The repairable pump of the home page (failure rate λ, repair rate μ) has the
exact unavailability U(t) = λ/(λ+μ) · (1 − e^{−(λ+μ)t}). The chart shows the
estimate, its confidence band, and the exact curve on top: the band should
hold it.
"""

from __future__ import annotations

import math

import pyraichu
from _style import Theme, render

LAMBDA, MU = 0.01, 0.1
T_MAX = 200.0
INSTANTS = [2.0 * k for k in range(101)]


def model() -> pyraichu.Model:
    return pyraichu.load_model({
        "name": "pump",
        "components": [{
            "name": "P",
            "automata": [{
                "name": "health", "states": ["working", "failed"], "init": "working",
                "transitions": [
                    {"name": "fail", "source": "working", "targets": ["failed"],
                     "distrib": "exp", "rate": LAMBDA},
                    {"name": "repair", "source": "failed", "targets": ["working"],
                     "distrib": "exp", "rate": MU},
                ],
            }],
        }],
        "indicators": [{"name": "P_failed", "target": "state",
                        "component": "P", "automaton": "health", "state": "failed"}],
    })


def main() -> None:
    est = pyraichu.monte_carlo(model(), nb_runs=2000, t_max=T_MAX,
                               samples=INSTANTS, seed=1)
    failed = est.indicators["P_failed"]
    exact = [LAMBDA / (LAMBDA + MU) * (1 - math.exp(-(LAMBDA + MU) * t))
             for t in failed.instants]

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        ax.fill_between(failed.instants, failed.ci.low, failed.ci.high,
                        color=theme.band, linewidth=0,
                        label=f"{failed.ci.level:.0%} confidence band")
        ax.plot(failed.instants, failed.mean, color=theme.series[0],
                label="Monte-Carlo estimate (2 000 replicas)")
        ax.plot(failed.instants, exact, color=theme.series[1], linestyle="--",
                label="exact closed form")
        ax.set_xlabel("time")
        ax.set_ylabel("unavailability")
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(0, None)
        ax.legend(loc="lower right")

    render("tutorial-unavailability", draw)


if __name__ == "__main__":
    main()
