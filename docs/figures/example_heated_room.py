"""Examples, heated room: one trajectory around a double failure, and a
winter campaign.

The model is the page's own (`page_namespace`).
"""

from __future__ import annotations

import math

import pyraichu
from _style import Theme, page_namespace, render

WINDOW = (170.0, 250.0)
NB_RUNS = 5000
DAYS = 90


def trajectory(ns) -> None:
    model = ns["model"]
    samples = [WINDOW[0] + 0.1 * k for k in range(int((WINDOW[1] - WINDOW[0]) * 10) + 1)]
    run = pyraichu.simulate(model, t_max=WINDOW[1], seed=12, samples=samples)
    temperature = run.samples["temperature"]
    mean, amplitude, peak = ns["OUTSIDE"]
    outside = [mean + amplitude * math.sin(2 * math.pi / 24 * (t - (peak - 6))) for t, _ in temperature]
    down: dict[str, list[tuple[float, float]]] = {"main": [], "backup": []}
    opened: dict[str, float] = {}
    for event in run.events:
        name, automaton, _ = event.transition.split(".")
        if automaton != "health":
            continue
        if event.to_state == "failed":
            opened[name] = event.time
        else:
            down[name].append((opened.pop(name), event.time))
    for name, start in opened.items():
        down[name].append((start, WINDOW[1]))

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        for row, (name, colour) in enumerate((("main", theme.series[0]), ("backup", theme.series[2]))):
            for start, end in down[name]:
                lo, hi = max(start, WINDOW[0]), min(end, WINDOW[1])
                if hi > lo:
                    ax.axvspan(lo, hi, ymin=0.93 - 0.06 * row, ymax=0.98 - 0.06 * row,
                               color=colour, linewidth=0)
                    ax.text(hi + 0.8, 25.4 - 1.5 * row, f"{name} down", fontsize=9,
                            color=colour, va="center")
        ax.plot([t for t, _ in temperature], [v for _, v in temperature],
                color=theme.series[0], label="room")
        ax.plot([t for t, _ in temperature], outside, color=theme.series[4],
                linestyle="--", label="outside")
        ax.axhline(10, color=theme.series[1], linewidth=1)
        ax.text(WINDOW[0] + 1, 10.4, "frozen below 10 °C", fontsize=9, color=theme.series[1])
        ax.set_xlim(*WINDOW)
        ax.set_ylim(-6, 27)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("temperature (°C)")
        ax.legend(loc="lower left")

    render("example-heated-room-trajectory", draw)


def winter(model) -> None:
    days = [24.0 * d for d in range(DAYS + 1)]
    estimates = pyraichu.monte_carlo(model, nb_runs=NB_RUNS, t_max=24.0 * DAYS,
                                     samples=days, seed=1)
    frozen, cold = estimates.indicators["frozen"], estimates.indicators["cold"]
    main = estimates.indicators["main_failed"]
    x = list(range(DAYS + 1))

    def draw(fig, theme: Theme) -> None:
        left, right = fig.subplots(1, 2)
        left.fill_between(x, frozen.reached_ci.low, frozen.reached_ci.high,
                          color=theme.band, linewidth=0)
        left.plot(x, frozen.reached_mean, color=theme.series[0])
        left.set_title("at least one freeze")
        left.set_xlabel("day")
        left.set_ylabel("probability")
        left.set_xlim(0, DAYS)
        left.set_ylim(0, None)
        right.fill_between(x, cold.sojourn_ci.low, cold.sojourn_ci.high,
                           color=theme.band, linewidth=0)
        right.plot(x, cold.sojourn_mean, color=theme.series[0])
        right.set_title("hours under 17 °C")
        right.set_xlabel("day")
        right.set_ylabel("mean cumulated hours")
        right.set_xlim(0, DAYS)
        right.set_ylim(0, None)

    render("example-heated-room-winter", draw, size=(7.0, 3.4))
    lam, mu = 1 / 500, 1 / 24
    exact = lam / (lam + mu) * (1 - math.exp(-(lam + mu) * 24.0 * DAYS))
    print(f"   P(frozen at least once in {DAYS} d) = {frozen.reached_mean[-1]:.4f} "
          f"[{frozen.reached_ci.low[-1]:.4f}, {frozen.reached_ci.high[-1]:.4f}]")
    print(f"   freezes per winter = {frozen.nb_occurrences_mean[-1]:.4f}, "
          f"hours frozen = {frozen.sojourn_mean[-1]:.2f}")
    print(f"   hours under 17 C = {cold.sojourn_mean[-1]:.2f} "
          f"[{cold.sojourn_ci.low[-1]:.2f}, {cold.sojourn_ci.high[-1]:.2f}]")
    print(f"   main heater unavailable at day {DAYS} = {main.mean[-1]:.4f} "
          f"[{main.ci.low[-1]:.4f}, {main.ci.high[-1]:.4f}], exact {exact:.4f}")


def main() -> None:
    ns = page_namespace("examples/heated-room.md")
    trajectory(ns)
    winter(ns["model"])


if __name__ == "__main__":
    main()
