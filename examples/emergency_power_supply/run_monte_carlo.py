"""Plain Monte-Carlo estimate of the unreliability at the mission time.

Usage: python run_monte_carlo.py VARIANT NB_RUNS [--seed N] [--tag TAG]

Writes `results/<VARIANT>_monte_carlo[_<TAG>].json`, a `raichu.quantification`
envelope: the proportion of histories that lose both bus bars before
10 000 h, with its 95 % Wilson interval.
"""

from __future__ import annotations

import argparse

import common
import pyraichu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant")
    parser.add_argument("nb_runs", type=int)
    parser.add_argument("--seed", type=int, default=2017)
    parser.add_argument("--tag", help="suffix of the result name, to keep several campaigns")
    args = parser.parse_args()

    model = common.load(args.variant)
    study = pyraichu.Study(
        target=common.TARGET, horizon=common.MISSION_TIME, seed=args.seed
    )
    name = f"{args.variant}_monte_carlo" + (f"_{args.tag}" if args.tag else "")
    with common.timed(name, nb_runs=args.nb_runs):
        result = pyraichu.quantify(
            model, study, method="monte_carlo", nb_runs=args.nb_runs
        )
    result.to_json(common.RESULTS / f"{name}.json")
    p = result.probability
    print(f"{name}: {p.estimate:.4e} [{p.low:.4e}, {p.high:.4e}]")


if __name__ == "__main__":
    main()
