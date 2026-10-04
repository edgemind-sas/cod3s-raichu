"""Splitting repeated over several seeds: does one interval tell the truth?

Usage: python run_splitting_seeds.py VARIANT [--seeds N]

A splitting estimate comes with a Student interval over independent
batches. When the batch estimates are very skewed, that interval can be
narrower than the spread of the estimator. Repeating the whole run with
different seeds measures the spread directly: the dispersion of the
estimates between seeds, against the half-widths each run announces.

Writes `results/<VARIANT>_splitting_seeds.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics

import common
import pyraichu
import run_rare_event
import translate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant")
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args()

    common.load(args.variant)  # regenerates the model when it is missing
    document = json.loads((translate.MODEL_DIR / f"{args.variant}.json").read_text())
    model = pyraichu.load_model(run_rare_event.with_progress_score(document))
    settings = run_rare_event.SETTINGS["splitting"]
    runs = []
    name = f"{args.variant}_splitting_seeds"
    with common.timed(
        name, particles=settings["particles"], batches=settings["batches"]
    ):
        for seed in range(1, args.seeds + 1):
            study = pyraichu.Study(
                target=common.TARGET, horizon=common.MISSION_TIME, seed=seed
            )
            p = pyraichu.quantify(
                model, study, method="splitting", **settings
            ).probability
            runs.append(
                {"seed": seed, "estimate": p.estimate, "low": p.low, "high": p.high}
            )
            print(f"seed {seed}: {p.estimate:.4e} [{p.low:.4e}, {p.high:.4e}]")
    estimates = [r["estimate"] for r in runs]
    summary = {
        "format": "eps_example.splitting_seeds",
        "variant": args.variant,
        "runs": runs,
        "mean": statistics.fmean(estimates),
        "stdev_between_seeds": statistics.stdev(estimates),
        "standard_error_of_mean": statistics.stdev(estimates) / len(estimates) ** 0.5,
    }
    (common.RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"mean {summary['mean']:.4e}, sd between seeds {summary['stdev_between_seeds']:.2e}"
    )


if __name__ == "__main__":
    main()
