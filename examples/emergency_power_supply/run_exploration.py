"""Exact sequence-tree exploration: bounds on the unreliability and the
most probable sequences.

Usage: python run_exploration.py VARIANT HORIZON MIN_PROBABILITY

The exploration enumerates the sequences of events from the perfect state
to the feared event, each with its exact probability by the horizon, and
cuts the branches less probable than MIN_PROBABILITY. The retained
sequences sum to a lower bound; adding the mass cut off gives an upper
bound. This is the brute-force algorithm FIGSEQ calls "NS" (Bouissou,
Khan, Katoen and Krcal 2020, section 2.2).

Writes `results/<VARIANT>_exact_<HORIZON>h.json`, the `raichu.quantification`
envelope with its sequence list cut to the `TOP_SEQUENCES` most probable
ones (the full list runs to thousands of sequences and megabytes; the
bounds are unchanged), and the number of sequences retained in the
`.meta.json` beside it.
"""

from __future__ import annotations

import argparse
import json

import common
import pyraichu
import translate

#: How many of the most probable sequences the results keep.
TOP_SEQUENCES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant")
    parser.add_argument("horizon", type=float)
    parser.add_argument("min_probability", type=float)
    args = parser.parse_args()

    model = common.load(args.variant)
    study = pyraichu.Study(target=common.TARGET, horizon=args.horizon)
    name = f"{args.variant}_exact_{args.horizon:.0f}h"
    with common.timed(name, min_probability=args.min_probability) as meta:
        result = pyraichu.quantify(
            model, study, method="exact", min_probability=args.min_probability
        )
        retained = len(result.detail.sequences)
        meta["retained_sequences"] = retained
        # Does the battery depletion matter? The retained sequences in
        # which one of its stages fails, and their probability.
        battery = [
            s
            for s in result.detail.sequences
            if any(e["obj"] in translate.BATTERY_STAGES for e in s.events)
        ]
        meta["sequences_with_battery"] = len(battery)
        meta["probability_with_battery"] = sum(s.probability for s in battery)
    envelope = json.loads(result.to_json())
    exploration = envelope["detail"]["exploration"]
    exploration["sequences"] = exploration["sequences"][:TOP_SEQUENCES]
    (common.RESULTS / f"{name}.json").write_text(json.dumps(envelope) + "\n")
    p = result.probability
    print(
        f"{name}: [{p.low:.4e}, {p.high:.4e}] {retained} sequences, inconclusive={p.inconclusive}"
    )


if __name__ == "__main__":
    main()
