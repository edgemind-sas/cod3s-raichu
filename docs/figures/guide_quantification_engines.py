"""Quantification: one study, three engines, one closed form.

The unit in series with a redundant pair of the quantification guide, over
10 hours. Each engine's answer is drawn with the uncertainty it states: a
Wilson interval for Monte-Carlo simulation, guaranteed bounds for exact
exploration, bounds widened by the error estimate for discretised
exploration. The dashed line is the closed form.
"""

from __future__ import annotations

import math

import pyraichu
from _style import Theme, render

RATES = {"A": 0.01, "B": 0.1, "C": 0.2}
HORIZON = 10.0


def unit(name, rate):
    return {"name": name, "automata": [{
        "name": "fail", "states": ["ok", "nok"], "init": "ok",
        "transitions": [{"name": "occ", "source": "ok", "targets": ["nok"],
                         "distrib": "exp", "rate": rate, "monitored": True}]}]}


def nok(name):
    return {"op": "state_active",
            "state": {"component": name, "automaton": "fail", "state": "nok"}}


def model() -> pyraichu.Model:
    guard = {"op": "bool", "bool_op": "or", "args": [
        nok("A"), {"op": "bool", "bool_op": "and", "args": [nok("B"), nok("C")]}]}
    return pyraichu.load_model({
        "name": "series_pair",
        "components": [unit(n, r) for n, r in RATES.items()] + [{
            "name": "sys", "automata": [{
                "name": "watch", "states": ["ok", "lost"], "init": "ok",
                "transitions": [{"name": "loss", "source": "ok", "targets": ["lost"],
                                 "distrib": "inst", "probs": [], "guard": guard}]}]}],
        "targets": [{"name": "system_lost", "component": "sys",
                     "automaton": "watch", "state": "lost"}],
    })


def main() -> None:
    p = {n: 1 - math.exp(-r * HORIZON) for n, r in RATES.items()}
    closed = 1 - (1 - p["A"]) * (1 - p["B"] * p["C"])
    m = model()
    study = pyraichu.Study("system_lost", HORIZON, seed=7)
    rows = []
    for label, method, extra in (
        ("Monte-Carlo\n20 000 replicas", "monte_carlo", {"nb_runs": 20_000}),
        ("Monte-Carlo\n2 000 replicas", "monte_carlo", {"nb_runs": 2_000}),
        ("exact\nexploration", "exact", {}),
        ("discretised\nexploration, level 4", "discretised", {"level": 4}),
    ):
        prob = pyraichu.quantify(m, study, method=method, **extra).probability
        err = getattr(prob, "error_estimate", None) or 0.0
        estimate = prob.estimate if prob.kind == "confidence_interval" else 0.5 * (prob.low + prob.high)
        rows.append((label, estimate, prob.low - err, prob.high + err))

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        for i, (label, est, lo, hi) in enumerate(rows):
            ax.errorbar([est], [i], xerr=[[est - lo], [hi - est]], fmt="o",
                        color=theme.series[0], capsize=5, markersize=5)
        ax.axvline(closed, color=theme.series[1], linestyle="--", label="closed form")
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("P(system lost by 10 h)")
        ax.legend(loc="lower right")

    render("guide-quantification-engines", draw, size=(7.0, 3.6))


if __name__ == "__main__":
    main()
