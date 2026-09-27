"""Sequence-tree exploration: the bounds close as the probability cut-off drops.

The repairable pair of the guide (A at 0.1, B at 0.03, both repaired at 0.5,
feared event: both down) explored to a horizon of 10 under a decreasing
``min_probability``. The lower bound is the retained mass, the upper bound
adds what the cut-off discarded; the true probability lies between.
"""

from __future__ import annotations

import pyraichu
from _style import Theme, render

THRESHOLDS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8]


def objfm(name, target, rate):
    return {"type": "ObjFM", "name": name, "targets": [target],
            "failure": [{"law": "exp", "rate": rate}],
            "repair": [{"law": "exp", "rate": 0.5}],
            "failure_effects": {"flow": False}}


def model() -> pyraichu.Model:
    return pyraichu.load_model({
        "name": "repairable_pair",
        "plugins": {"muscadet": {"objects": [
            objfm("fm_A", "A", 0.1),
            objfm("fm_B", "B", 0.03),
            {"type": "ObjEvent", "name": "system_down", "target": True,
             "cond": [[{"obj": "A", "attr": "flow", "ope": "==", "value": False},
                       {"obj": "B", "attr": "flow", "ope": "==", "value": False}]]},
        ]}},
        "components": [
            {"name": n, "attributes": [{"name": "flow", "kind": "bool",
                                        "init": {"kind": "bool", "value": True}}]}
            for n in ("A", "B")
        ],
    })


def main() -> None:
    m = model()
    runs = [pyraichu.explore(m, "system_down", horizon=10.0, min_probability=t)
            for t in THRESHOLDS]
    lower = [r.lower for r in runs]
    upper = [r.upper for r in runs]
    kept = [len(r.sequences) for r in runs]

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        ax.fill_between(THRESHOLDS, lower, upper, color=theme.band, linewidth=0,
                        label="mass the cut-off discarded")
        ax.plot(THRESHOLDS, upper, color=theme.series[1], marker="o", markersize=4,
                label="upper bound")
        ax.plot(THRESHOLDS, lower, color=theme.series[0], marker="o", markersize=4,
                label="lower bound (retained sequences)")
        for t, lo, k in zip(THRESHOLDS[::2], lower[::2], kept[::2]):
            ax.annotate(f"{k}", (t, lo), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=9, color=theme.text)
        ax.set_ylim(min(lower) - 0.008, None)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("min_probability (numbers: sequences retained)")
        ax.set_ylabel("P(both down by t = 10)")
        ax.legend(loc="upper right")

    render("guide-exploration-bounds", draw)


if __name__ == "__main__":
    main()
