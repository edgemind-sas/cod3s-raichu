"""Rare-event estimates of the unreliability: cross-entropy and splitting.

Usage: python run_rare_event.py VARIANT {cross_entropy,splitting} [--seed N]

Cross-entropy multiplies the constant exponential rates by factors fitted
over pilot campaigns and weights each history by its likelihood ratio.
Splitting clones the histories that progress furthest towards the feared
event; progress is read on a score this script adds to the model: the
number of supply paths of the bus bars LHA and LHB currently lost
(`PROGRESS_GATES`). The score is an observer, written by a sensitive
function from the gates' `S` attributes: it changes no transition.

Writes `results/<VARIANT>_<method>.json`.
"""

from __future__ import annotations

import argparse
import json

import common
import pyraichu
import translate

#: Gates of the BDMP on the way from a first loss to the loss of both bus
#: bars (2017, appendix, main and diesel pages): the grid connection, the
#: unit, both transformers, each train's supplies, then each bus bar.
PROGRESS_GATES = (
    "loss_of_supply_by_GEV",
    "loss_of_supply_by_UNIT",
    "loss_of_supply_by_TS",
    "loss_of_supply_by_TA",
    "loss_of_supply_by_LGD",
    "loss_of_supply_by_LGF",
    "loss_of_supply_by_DGA",
    "loss_of_supply_by_DGB",
    "loss_of_supply_by_TAC",
    "LHA_lost",
    "LHB_lost",
)
SCORE = {"kind": "attribute", "name": "progress.score"}

#: Settings of each method; the same for every variant, so that a reader
#: can compare the variants run for run.
SETTINGS = {
    "cross_entropy": {"nb_runs": 1_000_000, "pilot_runs": 100_000},
    "splitting": {"importance": SCORE, "particles": 1000, "batches": 100},
}


def with_progress_score(document: dict) -> dict:
    """The model document with the `progress.score` observer added."""
    body = pyraichu.model_body(document)
    one = {"op": "const", "value": {"kind": "float", "value": 1.0}}
    zero = {"op": "const", "value": {"kind": "float", "value": 0.0}}
    terms = [
        {
            "op": "if",
            "cond": {"op": "attr", "attr": {"component": gate, "attribute": "S"}},
            "then": one,
            "otherwise": zero,
        }
        for gate in PROGRESS_GATES
    ]
    body["components"].append(
        {
            "name": "progress",
            "attributes": [
                {
                    "name": "score",
                    "kind": "float",
                    "init": {"kind": "float", "value": 0.0},
                }
            ],
            "sensitive_functions": [
                {
                    "name": "count_lost_supplies",
                    "effects": [
                        {
                            "target": {"component": "progress", "attribute": "score"},
                            "value": {"op": "add", "args": terms},
                        }
                    ],
                }
            ],
        }
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant")
    parser.add_argument("method", choices=sorted(SETTINGS))
    parser.add_argument("--seed", type=int, default=2017)
    args = parser.parse_args()

    common.load(args.variant)  # regenerates the model when it is missing
    document = json.loads((translate.MODEL_DIR / f"{args.variant}.json").read_text())
    model = pyraichu.load_model(with_progress_score(document))
    study = pyraichu.Study(
        target=common.TARGET, horizon=common.MISSION_TIME, seed=args.seed
    )
    settings = SETTINGS[args.method]
    name = f"{args.variant}_{args.method}"
    with common.timed(name, **{k: v for k, v in settings.items() if k != "importance"}):
        result = pyraichu.quantify(model, study, method=args.method, **settings)
    result.to_json(common.RESULTS / f"{name}.json")
    p = result.probability
    print(
        f"{name}: {p.estimate:.4e} [{p.low:.4e}, {p.high:.4e}] inconclusive={p.inconclusive}"
    )


if __name__ == "__main__":
    main()
