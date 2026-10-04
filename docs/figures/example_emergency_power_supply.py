"""Examples, emergency power supply: RAICHU's results against the published
references.

The model is not run here: the figures read the result files the example's
scripts wrote under `examples/emergency_power_supply/results`, so a figure
always shows the numbers the page quotes. The references are the values
printed by Bouissou, Khan, Katoen and Krcal (2020), Tables 1 to 3.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyraichu
from _style import Theme, render

RESULTS = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "emergency_power_supply"
    / "results"
)

#: 2020, Table 3: mission time -> STORM, RiskSpectrum static tree.
STORM = {100.0: 3.495e-6, 1000.0: 7.925e-3, 10000.0: 3.604e-1}
RISKSPECTRUM_STATIC = {100.0: 2.812e-5, 1000.0: 2.740e-2, 10000.0: 4.071e-1}
#: 2020, Tables 1 and 2: (label, estimate, low, high) at 10 000 h; YAMS
#: prints a 90 % half-width.
REFERENCES = {
    "eps_benchmark": [
        ("FIGSEQ, cut-off 1e-10", 3.84e-5, None, None),
        ("YAMS, 2e7 histories (90 %)", 3.80e-5, 3.5e-5, 4.1e-5),
        ("I&AB", 1.35e-4, None, None),
    ],
    "eps_benchmark_fast_line_repair": [
        ("FIGSEQ, cut-off 1e-11", 3.85e-6, None, None),
        ("I&AB", 1.46e-5, None, None),
    ],
}
METHODS = (
    ("monte_carlo", "Monte-Carlo"),
    ("cross_entropy", "cross-entropy"),
    ("splitting", "splitting"),
)


def quantification(name: str):
    path = RESULTS / f"{name}.json"
    return pyraichu.read_quantification(path) if path.exists() else None


def raichu_rows(variant: str) -> list[tuple[str, float, float, float, bool]]:
    rows = []
    for method, label in METHODS:
        q = quantification(f"{variant}_{method}")
        if q is None:
            continue
        p = q.probability
        rows.append(
            (f"RAICHU {label}", p.estimate, p.low, p.high, bool(p.inconclusive))
        )
    seeds = RESULTS / f"{variant}_splitting_seeds.json"
    if seeds.exists():
        sweep = json.loads(seeds.read_text())
        mean, se = sweep["mean"], sweep["standard_error_of_mean"]
        rows.append(
            (
                f"RAICHU splitting, {len(sweep['runs'])} seeds (2 SE)",
                mean,
                mean - 2 * se,
                mean + 2 * se,
                False,
            )
        )
    return rows


def non_repairable() -> None:
    explored = {}
    for t in (100, 1000, 10000):
        q = quantification(f"eps_benchmark_nonrepairable_exact_{t}h")
        if q is not None:
            explored[float(t)] = q.probability
    static = json.loads((RESULTS / "static_tree.json").read_text())
    static_rows = {
        row["mission_time"]: row["top_probability"] for row in static["rows"]
    }

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        times = sorted(static_rows)
        ax.plot(
            times,
            [static_rows[t] for t in times],
            color=theme.series[1],
            marker="o",
            label="RAICHU static fault tree (BDD)",
        )
        ax.plot(
            times,
            [RISKSPECTRUM_STATIC[t] for t in times],
            color=theme.series[1],
            marker="s",
            markerfacecolor="none",
            linestyle="none",
            markersize=10,
            label="RiskSpectrum static tree (2020)",
        )
        ax.plot(
            sorted(explored),
            [explored[t].low for t in sorted(explored)],
            color=theme.series[0],
            marker="o",
            label="RAICHU exact exploration",
        )
        ax.plot(
            times,
            [STORM[t] for t in times],
            color=theme.series[0],
            marker="s",
            markerfacecolor="none",
            linestyle="none",
            markersize=10,
            label="STORM (2020)",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mission time (h)")
        ax.set_ylabel("unreliability")
        ax.legend(loc="lower right", fontsize=9)

    render("example-emergency-power-supply-nonrepairable", draw, size=(7.0, 3.8))


def repairable() -> None:
    #: (variant, title, unit of the axis): the dynamic methods agree within a
    #: factor two, I&AB sits beyond the axis and is drawn as an arrow.
    panels = [
        ("eps_benchmark", "published model", 1e-5),
        ("eps_benchmark_fast_line_repair", "line repair ten times faster", 1e-6),
    ]

    def draw(fig, theme: Theme) -> None:
        axes = fig.subplots(1, len(panels), sharey=False)
        for ax, (variant, title, unit) in zip(axes, panels):
            rows = [(*r, "raichu") for r in raichu_rows(variant)]
            rows += [
                (label, est, low, high, False, "ref")
                for label, est, low, high in REFERENCES[variant]
            ]
            dynamic = [r for r in rows if r[0] != "I&AB"]
            x_max = (
                1.4 * max(r[3] if r[3] is not None else r[1] for r in dynamic) / unit
            )
            for y, (label, est, low, high, inconclusive, source) in enumerate(rows):
                colour = theme.series[0] if source == "raichu" else theme.series[1]
                if est / unit > x_max:
                    ax.annotate(
                        f"{est:.2e}",
                        xy=(x_max, y),
                        xytext=(0.82 * x_max, y),
                        va="center",
                        ha="right",
                        color=colour,
                        fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": colour},
                    )
                    continue
                if low is not None and high is not None:
                    ax.plot(
                        [low / unit, high / unit], [y, y], color=colour, linewidth=2
                    )
                ax.plot(
                    [est / unit],
                    [y],
                    marker="o" if source == "raichu" else "s",
                    color=colour,
                    markerfacecolor="none" if inconclusive else colour,
                )
            ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
            ax.set_ylim(len(rows) - 0.3, -0.7)
            ax.set_xlim(0, x_max)
            exponent = round(math.log10(unit))
            ax.set_xlabel(f"unreliability at 10 000 h (x 1e{exponent})")
            ax.set_title(title)
            ax.grid(axis="y", visible=False)

    render("example-emergency-power-supply-repairable", draw, size=(9.0, 4.2))


def main() -> None:
    non_repairable()
    repairable()


if __name__ == "__main__":
    main()
