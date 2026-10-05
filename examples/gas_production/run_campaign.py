"""Stationary production-availability quantities of one case, by Monte-Carlo.

Usage: python run_campaign.py CASE [--runs N] [--horizon T] [--seed S] [--tag TAG]

Every history starts from the perfect state (both units up, reservoir
full), the regeneration state of the published regenerative estimator, and
runs to the horizon T. Over [0, T]:

- A, the availability: the share of time with nominal production, i.e.
  with the reservoir not empty;
- PA, the production availability: delivered over nominal production;
- f_LNP, the annual frequency of loss of nominal production: entries into
  an empty reservoir, per year of 8766 h;
- f_TLP, the annual frequency of total loss of production: entries into
  "both units down and reservoir empty", per year.

Each is a time average over [0, T]; the standard error is the spread
between histories over the square root of their number. Starting from the
regeneration state biases a time average by O(1/T): at an equal budget,
case 2 measures a bias of about three standard errors at T = 1e5 h and
none between 1e6 and 1e7 h (`--tag T1e5`, `T1e7` results), hence the
default horizon of 1e6 h, about 300 to 3000 cycles (343 h to 3342 h,
Eymard and Mercier, Table 13).

The ODE of the reservoir level is linear between two events, so the
integrator's step cap is raised to 100 h (the default is 0.1 h): the
trajectory is the same, located boundary crossings included, at a few
hundredths of the cost.

Writes `results/case<CASE>[_<TAG>].json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import model
import pyraichu

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MAX_STEP = 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=int, choices=model.CASES)
    parser.add_argument("--runs", type=int, default=40_000)
    parser.add_argument("--horizon", type=float, default=1.0e6)
    parser.add_argument("--seed", type=int, default=2008)
    parser.add_argument(
        "--tag", help="suffix of the result name, to keep several campaigns"
    )
    args = parser.parse_args()

    compiled = pyraichu.load_model(model.build(args.case))
    horizon = args.horizon
    start = time.perf_counter()
    estimates = pyraichu.monte_carlo(
        compiled,
        args.runs,
        horizon,
        [horizon],
        seed=args.seed,
        max_step=MAX_STEP,
    )
    elapsed = time.perf_counter() - start
    by = estimates.indicators
    root_n = math.sqrt(args.runs)
    per_year = model.YEAR / horizon

    def quantity(mean: float, std: float, scale: float, offset: float = 0.0) -> dict:
        return {
            "estimate": offset + scale * mean,
            "standard_error": abs(scale) * std / root_n,
        }

    empty, delivered, total_loss = by["empty"], by["delivered"], by["total_loss"]
    result = {
        "format": "gas_example.campaign",
        "case": args.case,
        "quantities": {
            "A": quantity(
                empty.sojourn_mean[-1], empty.sojourn_std[-1], -1 / horizon, 1.0
            ),
            "PA": quantity(
                delivered.sojourn_mean[-1], delivered.sojourn_std[-1], 1 / horizon
            ),
            "f_LNP": quantity(
                empty.nb_occurrences_mean[-1], empty.nb_occurrences_std[-1], per_year
            ),
            "f_TLP": quantity(
                total_loss.nb_occurrences_mean[-1],
                total_loss.nb_occurrences_std[-1],
                per_year,
            ),
        },
        "settings": {
            "runs": args.runs,
            "horizon_h": horizon,
            "seed": args.seed,
            "max_step_h": MAX_STEP,
            "year_h": model.YEAR,
        },
        "run": {
            "pyraichu": pyraichu.__version__,
            "wall_clock_s": round(elapsed, 1),
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
        },
    }
    RESULTS.mkdir(exist_ok=True)
    name = f"case{args.case}" + (f"_{args.tag}" if args.tag else "")
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, q in result["quantities"].items():
        print(
            f"case {args.case} {name}: {q['estimate']:.7g} +- {q['standard_error']:.2g}"
        )
    print(f"{elapsed:.0f} s")


if __name__ == "__main__":
    main()
