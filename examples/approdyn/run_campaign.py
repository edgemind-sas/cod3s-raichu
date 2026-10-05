"""Probability of a trip over the 18-month cycle, and its cause.

Usage: python run_campaign.py VARIANT [--runs N] [--seed S]

Each history runs scenario 1 from the plant at 0 % power with every
component available, and stops at the first trip (the four causes are
the model's targets). The probability that cause c is the first trip by
time t is the share of histories whose trip automaton reached c by t; the
total is their sum. Standard errors are binomial.

Writes `results/<VARIANT>.json`.
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
#: Reporting instants: every 1000 h, and the end of the cycle.
INSTANTS = [float(t) for t in range(0, int(model.HORIZON), 1000)] + [model.HORIZON]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", choices=[*model.VARIANTS, *model.DIAGNOSTIC])
    parser.add_argument("--runs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2012)
    args = parser.parse_args()

    perfect_tpa = args.variant in model.DIAGNOSTIC
    rule, ccf = {**model.VARIANTS, **model.DIAGNOSTIC}[args.variant]
    compiled = pyraichu.load_model(model.build(rule, ccf, args.variant, perfect_tpa))
    start = time.perf_counter()
    estimates = pyraichu.monte_carlo(
        compiled,
        args.runs,
        model.HORIZON,
        INSTANTS,
        seed=args.seed,
        stop_at_targets=True,
    )
    elapsed = time.perf_counter() - start

    def binomial(p: float) -> dict:
        return {"estimate": p, "standard_error": math.sqrt(p * (1 - p) / args.runs)}

    by_cause = {c: estimates.indicators[f"trip_{c}"].reached_mean for c in model.CAUSES}
    total = [sum(by_cause[c][k] for c in model.CAUSES) for k in range(len(INSTANTS))]
    result = {
        "format": "approdyn_example.campaign",
        "variant": args.variant,
        "are_trip_rule": rule,
        "common_cause_failures": ccf,
        "perfect_tpa": perfect_tpa,
        "end_of_cycle": {
            "total": binomial(total[-1]),
            "by_cause": {c: binomial(by_cause[c][-1]) for c in model.CAUSES},
        },
        "cumulative": {
            "instants_h": INSTANTS,
            "total": total,
            **{c: list(by_cause[c]) for c in model.CAUSES},
        },
        "settings": {"runs": args.runs, "seed": args.seed, "horizon_h": model.HORIZON},
        "run": {
            "pyraichu": pyraichu.__version__,
            "wall_clock_s": round(elapsed, 1),
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    end = result["end_of_cycle"]
    print(
        f"{args.variant}: trip {end['total']['estimate']:.4f} "
        f"+- {end['total']['standard_error']:.4f}  ({elapsed:.0f} s)"
    )
    for c, q in end["by_cause"].items():
        print(f"  {c}: {q['estimate']:.4f}")


if __name__ == "__main__":
    main()
