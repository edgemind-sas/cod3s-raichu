"""Compare the campaigns with the published reference values.

Usage: python compare.py   (prints the Markdown table the article quotes)

The reference is Eymard and Mercier's regenerative Monte-Carlo estimate
with its 95 % confidence interval (Reliability Engineering & System Safety
93, 2008, Tables 3 to 6, read on the HAL version hal-00693079). The gap is
in combined standard errors, z = (RAICHU - reference) / sqrt(s_R^2 + s_E^2),
with s_E the published half-width over 1.96.

One published cell is not usable: Table 5, case 2 prints 0.11295 for
f_TLP with the interval [0.11367, 0.11692], which does not contain it.
That cell is compared with its interval only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

#: (estimate, low, high) per case and quantity, Eymard and Mercier (2008).
REFERENCE = {
    1: {
        "A": (0.9989852, 0.9989790, 0.9989915),
        "PA": (0.9995232, 0.9995209, 0.9995255),
        "f_LNP": (0.07640, 0.07627, 0.07652),
        "f_TLP": (0.001151, 0.001132, 0.001171),
    },
    2: {
        "A": (0.9898986, 0.9898195, 0.9899777),
        "PA": (0.9952215, 0.9951946, 0.9952484),
        "f_LNP": (0.7526, 0.7514, 0.7538),
        "f_TLP": (0.11295, 0.11367, 0.11692),
    },
    3: {
        "A": (0.9884501, 0.9883955, 0.9885048),
        "PA": (0.9940342, 0.9940010, 0.9940584),
        "f_LNP": (0.8291, 0.8282, 0.8300),
        "f_TLP": (1.2143, 1.2053, 1.2234),
    },
    4: {
        "A": (0.9716373, 0.9712818, 0.9719929),
        "PA": (0.9853242, 0.9851754, 0.9854729),
        "f_LNP": (3.4985, 3.4887, 3.5082),
        "f_TLP": (2.9331, 2.8710, 2.9953),
    },
}
#: The self-inconsistent published cell (see the module docstring).
INCONSISTENT = {(2, "f_TLP")}
QUANTITIES = ("A", "PA", "f_LNP", "f_TLP")


def gap(case: int, quantity: str) -> tuple[float, float, str]:
    """RAICHU's estimate, its 95 % half-width, and the gap to the reference."""
    measured = json.loads((RESULTS / f"case{case}.json").read_text())["quantities"][
        quantity
    ]
    value, s_r = measured["estimate"], measured["standard_error"]
    reference, low, high = REFERENCE[case][quantity]
    if (case, quantity) in INCONSISTENT:
        verdict = (
            "inside the published interval" if low <= value <= high else "outside it"
        )
        return value, 1.96 * s_r, verdict
    s_e = (high - low) / 3.92
    return value, 1.96 * s_r, f"{(value - reference) / math.hypot(s_r, s_e):+.1f}"


def main() -> None:
    print("| Case | Quantity | RAICHU (95 %) | Reference (95 %) | Gap (combined SE) |")
    print("|---|---|---|---|---|")
    for case in REFERENCE:
        for quantity in QUANTITIES:
            value, half, verdict = gap(case, quantity)
            reference, low, high = REFERENCE[case][quantity]
            print(
                f"| {case} | {quantity} | {value:.6g} ± {half:.2g} | "
                f"{reference} [{low}, {high}] | {verdict} |"
            )


if __name__ == "__main__":
    main()
