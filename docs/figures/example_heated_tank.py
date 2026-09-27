"""Examples, heated tank: one trajectory, the top-event probabilities and
the minimal sequences.

The model is the page's own (`page_namespace`), so the page and its charts
cannot drift apart.
"""

from __future__ import annotations

from collections import Counter

from matplotlib.patches import Patch

import pyraichu
from _style import Theme, page_namespace, render

NB_RUNS = 20_000
T_MAX = 1000.0
EVENTS = ("dryout", "overflow", "overheat")
LABELS = {"dryout": "dry-out", "overflow": "overflow", "overheat": "overheat"}


def trajectory(model) -> None:
    samples = [0.25 * k for k in range(0, 401)]
    run = pyraichu.simulate(model, t_max=100.0, seed=3, samples=samples)
    level = run.samples["level"]
    theta = run.samples["temperature"]
    events = [e for e in run.events if e.time <= 100.0]

    def draw(fig, theme: Theme) -> None:
        top, bottom = fig.subplots(2, 1, sharex=True)
        top.plot([t for t, _ in level], [v for _, v in level], color=theme.series[0])
        for bound, style in ((4, "-"), (6, "--"), (8, "--"), (10, "-")):
            top.axhline(bound, color=theme.series[1] if style == "-" else theme.grid,
                        linestyle=style, linewidth=1)
        top.set_ylabel("level h (m)")
        top.set_ylim(3, 11)
        bottom.plot([t for t, _ in theta], [v for _, v in theta], color=theme.series[0])
        bottom.axhline(100, color=theme.series[1], linewidth=1)
        bottom.set_ylabel("temperature θ (°C)")
        bottom.set_xlabel("time (h)")
        bottom.set_xlim(0, 100)
        for event in events:
            component = event.transition.split(".")[0]
            command = event.to_state in ("on", "off")
            for ax in (top, bottom):
                ax.axvline(event.time, color=theme.text,
                           linestyle="--" if command else ":", linewidth=0.8,
                           alpha=0.5 if command else 1.0)
            if command:
                continue  # a control action, drawn but not labelled
            label = (event.to_state if component == "tank"
                     else f"{component} {event.to_state.replace('_', ' ')}")
            top.annotate(label, (event.time, 10.9), rotation=90, fontsize=9,
                         ha="right", va="top", color=theme.text)

    render("example-heated-tank-trajectory", draw, size=(7.0, 5.2))


def top_events(model) -> None:
    instants = [10.0 * k for k in range(101)]
    estimates = pyraichu.monte_carlo(model, nb_runs=NB_RUNS, t_max=T_MAX,
                                     samples=instants, seed=1)

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        for colour, event in zip(theme.series, EVENTS):
            ind = estimates.indicators[event]
            ax.fill_between(ind.instants, ind.ci.low, ind.ci.high,
                            color=colour, alpha=0.25, linewidth=0)
            ax.plot(ind.instants, ind.mean, color=colour, label=LABELS[event])
        ax.set_xlabel("time (h)")
        ax.set_ylabel("probability of having occurred")
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", title=f"{NB_RUNS:_} replicas, 95 % bands".replace("_", " "))

    render("example-heated-tank-top-events", draw)
    for event in EVENTS:
        ind = estimates.indicators[event]
        print(f"   P({event} by {T_MAX:g} h) = {ind.mean[-1]:.4f} "
              f"[{ind.ci.low[-1]:.4f}, {ind.ci.high[-1]:.4f}]")


def sequences(model) -> None:
    found = pyraichu.analyse_sequences(model, nb_runs=NB_RUNS, t_max=T_MAX, seed=1)
    weights = Counter()
    for seq in found:
        if seq["end_cause"] is None:
            continue
        failures = " → ".join(e["obj"] + " " + e["attr"].replace("_", " ") for e in seq["events"])
        weights[(seq["end_cause"], failures)] += seq["weight"] / NB_RUNS
    shown = weights.most_common(12)[::-1]

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        colours = dict(zip(EVENTS, theme.series))
        ax.barh(range(len(shown)), [w for _, w in shown],
                color=[colours[cause] for (cause, _), _ in shown])
        ax.set_yticks(range(len(shown)), [f for (_, f), _ in shown], fontsize=9)
        ax.set_xlabel("fraction of the trajectories")
        ax.grid(axis="y", visible=False)
        handles = [Patch(color=colours[e], label=LABELS[e]) for e in EVENTS]
        ax.legend(handles=handles, loc="lower right", title="ends in")

    render("example-heated-tank-sequences", draw, size=(7.0, 4.6))
    for (cause, failures), w in weights.most_common(6):
        print(f"   {w:.4f}  {cause:9s} {failures}")
    print(f"   no top event by {T_MAX:g} h: "
          f"{sum(s['weight'] for s in found if s['end_cause'] is None) / NB_RUNS:.4f}")


def main() -> None:
    model = page_namespace("examples/heated-tank.md")["model"]
    trajectory(model)
    top_events(model)
    sequences(model)


if __name__ == "__main__":
    main()
