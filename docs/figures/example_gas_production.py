"""Examples, gas production: gap between RAICHU and the published
regenerative Monte-Carlo estimate, per case and quantity.

The figure reads the campaign files and the reference table of
`examples/gas_production/compare.py`, so it shows the numbers the page
quotes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from _style import Theme, render

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "gas_production"
sys.path.insert(0, str(EXAMPLE))

import compare

#: Gaps beyond this are drawn at the edge, with their value.
CLIP = 8.0


def main() -> None:
    labels, gaps, cases = [], [], []
    for case in compare.REFERENCE:
        for quantity in compare.QUANTITIES:
            if (case, quantity) in compare.INCONSISTENT:
                continue
            _, _, verdict = compare.gap(case, quantity)
            labels.append(f"case {case}  {quantity}")
            gaps.append(float(verdict))
            cases.append(case)

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        ys = range(len(labels))
        ax.axvspan(-2, 2, color=theme.band, linewidth=0, label="within 2 combined SE")
        for y, g, case in zip(ys, gaps, cases):
            colour = theme.series[1] if case == 4 else theme.series[0]
            shown = max(-CLIP, min(CLIP, g))
            ax.plot([0, shown], [y, y], color=colour, linewidth=2)
            ax.plot([shown], [y], "o", color=colour)
            if not math.isclose(shown, g):
                ax.annotate(
                    f"{g:+.1f}",
                    xy=(shown, y),
                    xytext=(4 if g > 0 else -4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if g > 0 else "right",
                    color=colour,
                    fontsize=9,
                )
        ax.set_yticks(list(ys), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-CLIP - 2, CLIP + 2)
        ax.set_xlabel("(RAICHU - reference) / combined standard error")
        ax.axvline(0, color=theme.text, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=9)

    render("example-gas-production-gaps", draw, size=(7.0, 5.0))


if __name__ == "__main__":
    main()
