"""Which TPA hypothesis reproduces the published TPA share?

Usage: python hypotheses.py [--runs N] [--seed S]

The published Monte-Carlo (report chapter 6, Table 6.1) attributes 47 of
4000 trips to the TPA turbo-pumps, 1.2 %; the printed data and logic give
22 % without common causes and 54 % with them. Each hypothesis below
changes one lever of the TPA (or a combination) on the published counting
rule, everything else as printed, and the campaign measures the TPA share
and the total. A lever that brings the share down to 1.2 % while keeping
the total near 54.75 % is a candidate for what the published run did.

Every lever comes from the report or its sources:

- the TPA common-cause failures (5 % of the running failures, p. 49);
- the turbine failure rate, 5.9e-4 /h (Table 4.4), here tried ten times
  lower, as for a misplaced decimal;
- the turbine refusal on demand, printed three ways (3.9e-3 in Table 4.4,
  3.9e-2 in a worked example, 3.9e-5 in Ionescu's 2016 thesis on the TPA);
- the forcing back to 60 % and the restart, which may fail (p. 50);
- the repair times (Tables 4.8, 4.10), which the report's implementations
  used doubled; here also made instantaneous, as for a model that ignores
  the exposure during repair.

Writes `results/hypotheses.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import model
import pyraichu
from model import TpaData

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

#: name -> (description, TPA data). All without TPA common causes but the
#: first, which is the report as printed.
INSTANT = 1e-6
HYPOTHESES = {
    "printed": ("report as printed, with TPA common causes", TpaData(ccf=True)),
    "no_ccf": ("no TPA common cause", TpaData(ccf=False)),
    "turbine_rate_div10": (
        "no common cause; turbine failure rate / 10",
        TpaData(ccf=False, turbine_rate=model.TPA_T_RATE / 10),
    ),
    "turbine_pfd_thesis": (
        "no common cause; turbine refusal 3.9e-5 (thesis)",
        TpaData(ccf=False, turbine_pfd=3.9e-5),
    ),
    "no_demand_failure": (
        "no common cause; forcing and restarts never fail",
        TpaData(ccf=False, turbine_pfd=0.0, out_of_turbine_pfd=0.0, forcing_pfd=0.0),
    ),
    "repairs_doubled": (
        "no common cause; repair times doubled",
        TpaData(ccf=False, repair_factor=2.0),
    ),
    "repairs_instant": (
        "no common cause; repairs instantaneous",
        TpaData(ccf=False, repair_factor=INSTANT),
    ),
    "rate_div10_pfd_thesis": (
        "no common cause; turbine rate / 10 and refusal 3.9e-5",
        TpaData(ccf=False, turbine_rate=model.TPA_T_RATE / 10, turbine_pfd=3.9e-5),
    ),
    "rate_div10_no_demand_failure": (
        "no common cause; turbine rate / 10, forcing and restarts never fail",
        TpaData(
            ccf=False,
            turbine_rate=model.TPA_T_RATE / 10,
            turbine_pfd=0.0,
            out_of_turbine_pfd=0.0,
            forcing_pfd=0.0,
        ),
    ),
    "instant_repairs_pfd_thesis": (
        "no common cause; repairs instantaneous and refusal 3.9e-5",
        TpaData(ccf=False, repair_factor=INSTANT, turbine_pfd=3.9e-5),
    ),
    "rate_div10_instant_repairs": (
        "no common cause; turbine rate / 10 and repairs instantaneous",
        TpaData(ccf=False, turbine_rate=model.TPA_T_RATE / 10, repair_factor=INSTANT),
    ),
    "rate_div10_instant_repairs_pfd_thesis": (
        "no common cause; turbine rate / 10, repairs instantaneous, refusal 3.9e-5",
        TpaData(
            ccf=False,
            turbine_rate=model.TPA_T_RATE / 10,
            repair_factor=INSTANT,
            turbine_pfd=3.9e-5,
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2012)
    args = parser.parse_args()

    rows = []
    for name, (description, data) in HYPOTHESES.items():
        document = model.build("pdmp", True, f"approdyn_{name}", tpa_data=data)
        start = time.perf_counter()
        estimates = pyraichu.monte_carlo(
            pyraichu.load_model(document),
            args.runs,
            model.HORIZON,
            [model.HORIZON],
            seed=args.seed,
            stop_at_targets=True,
        )
        elapsed = time.perf_counter() - start
        shares = {
            c: estimates.indicators[f"trip_{c}"].reached_mean[-1] for c in model.CAUSES
        }
        total = sum(shares.values())
        rows.append(
            {
                "name": name,
                "description": description,
                "tpa_data": asdict(data),
                "total": total,
                "by_cause": shares,
                "standard_error_total": math.sqrt(total * (1 - total) / args.runs),
                "wall_clock_s": round(elapsed, 1),
            }
        )
        print(f"{name:32s} total {total:.4f}  tpa {shares['tpa']:.4f}")
    RESULTS.mkdir(exist_ok=True)
    out = {
        "format": "approdyn_example.hypotheses",
        "are_trip_rule": "pdmp",
        "cex_common_causes": True,
        "settings": {"runs": args.runs, "seed": args.seed, "horizon_h": model.HORIZON},
        "pyraichu": pyraichu.__version__,
        "hypotheses": rows,
    }
    (RESULTS / "hypotheses.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
