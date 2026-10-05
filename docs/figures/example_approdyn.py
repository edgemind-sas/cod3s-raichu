"""Examples, APPRODYN feedwater: probability of a trip over the 18-month
cycle, split by cause, for the published Monte-Carlo and each RAICHU
variant.

The figure reads `examples/approdyn/results` through the example's
`compare.py`, so it shows the numbers the page quotes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _style import Theme, render

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "approdyn"
sys.path.insert(0, str(EXAMPLE))

import compare
import model

CAUSES = ("vvp", "are", "cex", "tpa")
NAMES = {
    "vvp": "VVP header",
    "are": "ARE valves",
    "cex": "CEX pumps",
    "tpa": "TPA turbo-pumps",
}


def main() -> None:
    rows = [("published Monte-Carlo\n(4000 histories)", compare.published())]
    for variant in (*model.VARIANTS, *model.DIAGNOSTIC):
        rows.append(
            (compare.LABELS[variant].replace("; ", "\n"), compare.campaign(variant))
        )

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        for y, (_, values) in enumerate(rows):
            left = 0.0
            for i, cause in enumerate(CAUSES):
                width = 100 * values[cause][0]
                ax.barh(
                    y,
                    width,
                    left=left,
                    color=theme.series[i],
                    height=0.6,
                    label=NAMES[cause] if y == 0 else None,
                    hatch="//" if cause == "tpa" else None,
                    edgecolor=theme.text if cause == "tpa" else "none",
                    linewidth=0,
                )
                left += width
            ax.text(
                left + 1, y, f"{left:.1f} %", va="center", fontsize=9, color=theme.text
            )
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("probability of a trip within the 18-month cycle (%)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=9)

    render("example-approdyn-trips", draw, size=(8.0, 4.4))


if __name__ == "__main__":
    main()
