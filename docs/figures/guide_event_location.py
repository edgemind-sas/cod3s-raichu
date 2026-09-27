"""Event location: how a watched boundary is found inside one solver step.

A schematic drawn from a smooth trajectory, not from an engine run: the
solver takes a step, its dense output is scanned at sub-sample points, the
first pair of points bracketing the boundary is kept, and the crossing is
bisected to the event tolerance on the dense output.
"""

from __future__ import annotations

import math

from _style import Theme, render


def x(t: float) -> float:
    return 18.0 + 4.0 * math.sin(1.3 * t) + 0.8 * t


BOUND = 22.5
T0, T1 = 0.0, 1.6
SUB = 8


def root() -> float:
    lo, hi = T0, T1
    scan = [T0 + (T1 - T0) * i / SUB for i in range(SUB + 1)]
    for a, b in zip(scan, scan[1:]):
        if (x(a) - BOUND) * (x(b) - BOUND) <= 0:
            lo, hi = a, b
            break
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (x(lo) - BOUND) * (x(mid) - BOUND) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> None:
    fine = [T0 + (T1 - T0) * i / 400 for i in range(401)]
    scan = [T0 + (T1 - T0) * i / SUB for i in range(SUB + 1)]
    tc = root()
    bracket = next((a, b) for a, b in zip(scan, scan[1:])
                   if (x(a) - BOUND) * (x(b) - BOUND) <= 0)

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        ax.axvspan(*bracket, color=theme.band, linewidth=0, label="bracketing pair")
        ax.plot(fine, [x(t) for t in fine], color=theme.series[0],
                label="dense output over one step")
        ax.axhline(BOUND, color=theme.series[1], linestyle="--", label="guard boundary")
        ax.plot(scan, [x(t) for t in scan], "o", color=theme.series[0], markersize=5,
                label=f"scan points (sub_samples = {SUB})")
        ax.plot([tc], [BOUND], marker="*", markersize=14, color=theme.series[1],
                linestyle="none", label="crossing, bisected to tol_event")
        ax.set_xticks([T0, T1], ["step start", "step end"])
        ax.set_yticks([])
        ax.set_xlabel("time")
        ax.set_ylabel("continuous attribute")
        ax.legend(loc="lower right", fontsize=9)

    render("guide-event-location", draw)


if __name__ == "__main__":
    main()
