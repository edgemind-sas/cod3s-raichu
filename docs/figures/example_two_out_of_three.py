"""Examples, two-out-of-three pumps: exact values, Monte-Carlo and
sequence-tree exploration on one chart, and the exploration's mass split
by sequence length.

The model and the exact unreliability are the page's own (`page_namespace`).
"""

from __future__ import annotations

import math
from collections import defaultdict

import pyraichu
from _style import Theme, page_namespace, render

NB_RUNS = 20_000
T_MAX = 1000.0
EXPLORED = (100.0, 250.0, 500.0, 750.0, 1000.0)


def main() -> None:
    ns = page_namespace("examples/two-out-of-three.md")
    model, unreliability = ns["model"], ns["unreliability"]
    lam, mu = ns["LAMBDA"], ns["MU"]

    def unavailability(t: float) -> float:
        p = lam / (lam + mu) * (1 - math.exp(-(lam + mu) * t))
        return 3 * p * p * (1 - p) + p**3

    instants = [10.0 * k for k in range(101)]
    campaign = pyraichu.monte_carlo(model, nb_runs=NB_RUNS, t_max=T_MAX,
                                    samples=instants, seed=1)
    down = campaign.indicators["down"]
    explored = {h: pyraichu.explore(model, "system_down", horizon=h,
                                    min_probability=1e-10, max_length=30)
                for h in EXPLORED}

    def draw(fig, theme: Theme) -> None:
        left, right = fig.subplots(1, 2)
        fine = [5.0 * k for k in range(201)]
        left.fill_between(down.instants, down.reached_ci.low, down.reached_ci.high,
                          color=theme.band, linewidth=0, label="Monte-Carlo, 95 % band")
        left.plot(down.instants, down.reached_mean, color=theme.series[0], linewidth=1,
                  label="Monte-Carlo estimate")
        left.plot(fine, [unreliability(t) for t in fine], color=theme.series[1],
                  linestyle="--", label="exact")
        left.plot(list(explored), [r.lower for r in explored.values()], "o",
                  color=theme.series[2], markersize=5, label="exploration bounds")
        left.set_title("unreliability F(t)")
        left.set_xlabel("time (h)")
        left.set_xlim(0, T_MAX)
        left.set_ylim(0, None)
        left.legend(loc="upper left", fontsize=9)
        right.fill_between(down.instants, down.ci.low, down.ci.high,
                           color=theme.band, linewidth=0)
        right.plot(down.instants, down.mean, color=theme.series[0], linewidth=1)
        right.plot(fine, [unavailability(t) for t in fine], color=theme.series[1],
                   linestyle="--")
        right.set_title("unavailability U(t)")
        right.set_xlabel("time (h)")
        right.set_xlim(0, T_MAX)
        right.set_ylim(0, None)

    render("example-two-out-of-three-measures", draw, size=(7.0, 3.6))

    by_length: dict[int, float] = defaultdict(float)
    for seq in explored[250.0].sequences:
        by_length[len(seq.events)] += seq.probability
    lengths = sorted(by_length)

    def draw_lengths(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        ax.bar([str(n) for n in lengths], [by_length[n] for n in lengths],
               color=theme.series[0])
        ax.set_yscale("log")
        ax.set_xlabel("events in the sequence (failures and repairs)")
        ax.set_ylabel("probability by 250 h")
        ax.grid(axis="x", visible=False)

    render("example-two-out-of-three-lengths", draw_lengths, size=(7.0, 3.2))

    for h, r in explored.items():
        print(f"   {h:6.0f} h  exact {unreliability(h):.6f}  explored "
              f"[{r.lower:.6f}, {r.upper:.6f}]  {len(r.sequences)} sequences")
    print(f"   F(1000) Monte-Carlo {down.reached_mean[-1]:.4f} "
          f"[{down.reached_ci.low[-1]:.4f}, {down.reached_ci.high[-1]:.4f}]")
    print(f"   U(1000) exact {unavailability(T_MAX):.5f}, Monte-Carlo {down.mean[-1]:.5f} "
          f"[{down.ci.low[-1]:.5f}, {down.ci.high[-1]:.5f}]")
    for n in lengths:
        print(f"   length {n:2d}: {by_length[n]:.4e}")


if __name__ == "__main__":
    main()
