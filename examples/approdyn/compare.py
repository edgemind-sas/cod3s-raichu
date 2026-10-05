"""Compare the campaigns with the published Monte-Carlo.

Usage: python compare.py   (prints the Markdown table the article quotes)

The reference is the piecewise-deterministic Monte-Carlo of the APPRODYN
report (Aubry et al. 2012, hal-00740181, chapter 6, Table 6.1, p. 101):
4000 histories of scenario 1, stopped at the first trip, 2190 trips, of
which 792 by the VVP header, 1301 by the ARE valves, 50 by the CEX pumps
and 47 by the TPA. Its counting rule makes every ARE failure forcing a
drop to 2 % power a trip (report p. 9).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import model

RESULTS = Path(__file__).resolve().parent / "results"

#: Published trips per cause, out of PUBLISHED_RUNS histories.
PUBLISHED_RUNS = 4000
PUBLISHED = {"vvp": 792, "are": 1301, "cex": 50, "tpa": 47}
ORDER = ("total", "vvp", "are", "cex", "tpa")
LABELS = {
    "approdyn_pdmp": "ARE: every failure trips; common causes",
    "approdyn_pdmp_noccf": "ARE: every failure trips; no common cause",
    "approdyn_tables": "ARE: report tables; common causes",
    "approdyn_pdmp_perfect_tpa": "diagnostic: perfect TPA",
}


def published() -> dict[str, tuple[float, float]]:
    """Published share and its binomial standard error, per cause."""
    counts = {**PUBLISHED, "total": sum(PUBLISHED.values())}
    out = {}
    for key, k in counts.items():
        p = k / PUBLISHED_RUNS
        out[key] = (p, math.sqrt(p * (1 - p) / PUBLISHED_RUNS))
    return out


def campaign(variant: str) -> dict[str, tuple[float, float]]:
    end = json.loads((RESULTS / f"{variant}.json").read_text())["end_of_cycle"]
    out = {"total": (end["total"]["estimate"], end["total"]["standard_error"])}
    for cause, q in end["by_cause"].items():
        out[cause] = (q["estimate"], q["standard_error"])
    return out


def main() -> None:
    reference = published()
    print("| | Trip in 18 months | VVP | ARE | CEX | TPA |")
    print("|---|---|---|---|---|---|")
    row = " | ".join(
        f"{100 * reference[k][0]:.1f} % ± {196 * reference[k][1]:.1f}" for k in ORDER
    )
    print(f"| published Monte-Carlo, 4000 histories | {row} |")
    for variant in (*model.VARIANTS, *model.DIAGNOSTIC):
        values = campaign(variant)
        row = " | ".join(f"{100 * values[k][0]:.1f} %" for k in ORDER)
        print(f"| {LABELS[variant]} | {row} |")


if __name__ == "__main__":
    main()
