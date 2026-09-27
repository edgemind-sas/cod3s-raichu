"""Confidence intervals: Wilson against the textbook interval on a rare event.

500 replicas, k of which reached the event. The textbook (Wald) interval
p ± z·sqrt(p(1-p)/n) collapses to [0, 0] at k = 0 and dips below zero for
small k; the Wilson interval RAICHU uses stays in [0, 1] and at k = 0 still
bounds the probability from above.
"""

from __future__ import annotations

import math

from _style import Theme, render

N = 500
Z = 1.959963984540054
K = list(range(0, 16))


def wilson(k: int) -> tuple[float, float]:
    p = k / N
    centre = (p + Z * Z / (2 * N)) / (1 + Z * Z / N)
    half = Z / (1 + Z * Z / N) * math.sqrt(p * (1 - p) / N + Z * Z / (4 * N * N))
    return centre - half, centre + half


def wald(k: int) -> tuple[float, float]:
    p = k / N
    half = Z * math.sqrt(p * (1 - p) / N)
    return p - half, p + half


def main() -> None:
    w = [wilson(k) for k in K]
    t = [wald(k) for k in K]

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        for k, (lo, hi) in zip(K, t):
            ax.plot([k - 0.15, k - 0.15], [lo, hi], color=theme.series[1], linewidth=2.2,
                    label="textbook p ± z·σ" if k == 0 else None)
        for k, (lo, hi) in zip(K, w):
            ax.plot([k + 0.15, k + 0.15], [lo, hi], color=theme.series[0], linewidth=2.2,
                    label="Wilson" if k == 0 else None)
        ax.plot(K, [k / N for k in K], "o", color=theme.text, markersize=3,
                label="estimate k / n")
        ax.axhline(0, color=theme.text, linewidth=0.8)
        ax.set_xlabel(f"replicas that reached the event, out of n = {N}")
        ax.set_ylabel("95 % interval on P")
        ax.legend(loc="upper left")

    render("guide-confidence-wilson", draw)


if __name__ == "__main__":
    main()
