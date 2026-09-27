# Quantification envelope format

RAICHU answers "what is the probability that this feared event happens by
this horizon" with three engines: **Monte-Carlo simulation**, **exact
exploration** and **discretised exploration**. Each engine keeps its own
settings and its own detailed result; the quantification envelope puts one
shape around all three, so a reader finds the method, the provenance and
the probability with its uncertainty in the same place whichever engine
answered.

The contract lives in the crate `raichu-quantify` (re-exported by the
umbrella crate as `raichu::raichu_quantify`): a `Study` states the
question once, a `Method` names the engine with the settings that belong
to it alone, and `quantify(model, study, method)` returns the envelope.
`read_quantification` reads one back.

For a walk through one study quantified by the three engines, see the
[quantification guide](../guides/quantification.md); this page specifies
the format.

## From Python

`pyraichu.quantify(model, study, method=..., **settings)` is the one entry
point to the three engines; `Quantification.to_json()` writes the envelope
and `pyraichu.read_quantification` reads it back into an equal object.

```python
import math

import pyraichu


def unit(name, rate):
    return {
        "name": name,
        "automata": [
            {
                "name": "fail",
                "states": ["ok", "nok"],
                "init": "ok",
                "transitions": [
                    {"name": "occ", "source": "ok", "targets": ["nok"],
                     "distrib": "exp", "rate": rate, "monitored": True}
                ],
            }
        ],
    }


def nok(name):
    return {"op": "state_active",
            "state": {"component": name, "automaton": "fail", "state": "nok"}}


pair = pyraichu.load_model({
    "name": "parallel_pair",
    "components": [
        unit("A", 0.1),
        unit("B", 0.2),
        {
            "name": "sys",
            "automata": [{
                "name": "watch", "states": ["ok", "down"], "init": "ok",
                "transitions": [{
                    "name": "down", "source": "ok", "targets": ["down"],
                    "distrib": "inst", "probs": [],
                    "guard": {"op": "bool", "bool_op": "and",
                              "args": [nok("A"), nok("B")]},
                }],
            }],
        },
    ],
    "targets": [{"name": "both_down", "component": "sys",
                 "automaton": "watch", "state": "down"}],
})
study = pyraichu.Study("both_down", 5.0, seed=1)
closed_form = (1 - math.exp(-0.1 * 5.0)) * (1 - math.exp(-0.2 * 5.0))

exact = pyraichu.quantify(pair, study, method="exact")
assert math.isclose(exact.probability.low, closed_form, rel_tol=1e-12)

mc = pyraichu.quantify(pair, study, method="monte_carlo", nb_runs=2000)
print(mc.probability.estimate, mc.probability.low, mc.probability.high)

disc = pyraichu.quantify(pair, study, method="discretised", level=4)
assert abs(disc.probability.low - closed_form) <= disc.probability.error_estimate

assert pyraichu.read_quantification(exact.to_json()) == exact
```

## What the probability is

The probability that the study's target is the **first** of the model's
declared targets reached, by the horizon. The engines already behave this
way: a Monte-Carlo trajectory stopped at targets stops at the first one,
and an exploration treats a sequence that reaches another target first as
a leaf contributing nothing. On a model declaring a single target, it is
the probability of reaching it by the horizon.

## The study

| Field | Type | Meaning |
|---|---|---|
| `target` | string | the declared target (feared event) quantified |
| `horizon` | number | finite, nonnegative, in the model's time unit |
| `instants` | array of numbers, optional | reporting instants of the Monte-Carlo detailed result, strictly ascending within `[0, horizon]`; default: the horizon alone |
| `seed` | integer, optional | master seed of a Monte-Carlo campaign; default `0` |
| `threads` | integer, optional | worker threads, at least 1; no result depends on it |

An unknown target is refused before anything runs, naming the targets the
model declares.

## `raichu.quantification`, version 1

One JSON document.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | always `"raichu.quantification"` |
| `version` | integer | `1` |
| `method` | object | `{"name", "settings"}`: the method and the settings it applied, defaults resolved (see [Methods](#methods)) |
| `provenance` | object | where the answer comes from (see [Provenance](#provenance)) |
| `probability` | object | the probability of the target, with its uncertainty (see [Probability](#probability)) |
| `detail` | object | the engine's own result, unchanged (see [Detail](#detail)) |

### Methods

| `name` | Engine | `settings` |
|---|---|---|
| `monte_carlo` | Monte-Carlo simulation | `nb_runs` (at least 1, required), `confidence` (default `0.95`), `quantiles` (default none) |
| `exact` | exact exploration (Markov family) | `min_probability`, `max_length`, `max_failures`, `max_branches` (cut-offs, each optional), `gap_tolerance`, `rel_precision`, `max_terms` |
| `discretised` | discretised exploration | the four cut-offs (`max_branches` defaults to 1 000 000), `gap_tolerance`, `level` (default 8), `refine` (default `true`) |

A setting given to a method it does not belong to is refused, naming the
methods that do take it. The Monte-Carlo campaign always stops each
trajectory at the first target reached, runs to the study's horizon,
samples at the study's instants, and uses the engine's default solver and
flow settings.

### Provenance

| Field | Type | Meaning |
|---|---|---|
| `engine_version` | string | the engine version that quantified |
| `model` | string | the model's name |
| `model_hash` | string | `sha256:` and 64 lowercase hex digits: the SHA-256 of the sealed model document written as canonical JSON (see below) |
| `target` | string | the study's target |
| `horizon` | number | the study's horizon |
| `instants` | array of numbers | the reporting instants; **Monte-Carlo only**, absent otherwise |
| `seed` | integer | the master seed; **Monte-Carlo only**, absent otherwise |

The thread count is never recorded, since no result depends on it. The
exploration methods ignore the seed and the instants and do not record
them, so the same study with another seed yields the same exploration
envelope, byte for byte.

**The content hash.** The model is sealed (written as a complete document
with its format header, every defaulted field spelled out), then written
as canonical JSON (object keys sorted by byte order, no whitespace) and
hashed with SHA-256. Two models that differ in any declared value have
different hashes, and the same model hashes the same whatever the layout
of the document it was read from. Because sealing spells out defaults and
the format revision, an engine release that adds a defaulted field changes
the hash of an unchanged model: compare hashes only between envelopes
carrying the same `engine_version`.

### Probability

A `kind` member says which uncertainty the engine can state.

**`confidence_interval`** (Monte-Carlo simulation):

| Field | Type | Meaning |
|---|---|---|
| `estimate` | number | `reached / replicas` |
| `reached` | integer | replicas whose first target reached was the study's, by the horizon |
| `replicas` | integer | replicas run |
| `level` | number | the confidence level, strictly inside `(0, 1)` |
| `method` | string | `"wilson"`: the estimate is a binomial proportion |
| `low`, `high` | number | the Wilson score interval |

The count comes from how each replica of one stop-at-targets campaign
ended, not from an indicator. On an event no replica reached, the interval
is `[0, z²/(n + z²)]` rather than a point at zero (see the
[confidence intervals guide](../guides/confidence-intervals.md)).

**`bounds`** (exact and discretised exploration):

| Field | Type | Meaning |
|---|---|---|
| `lower` | number | the sum of the retained sequence probabilities |
| `upper` | number | `lower` plus the mass the cut-offs discarded |
| `inconclusive` | boolean | the relative gap exceeds the declared tolerance |
| `error_estimate` | number, optional | the discretisation error estimate by refinement: present for a discretised exploration that refined, absent otherwise. An estimate, not a bound |

For a discretised exploration, the bounds are bounds on the discretised
model.

### Detail

An object with a single key naming the engine kind:

- `{"monte_carlo": {...}}`: the Monte-Carlo estimates of the
  stop-at-targets campaign, identical to a direct Monte-Carlo run with the
  stop at targets on and the same seed, replicas, level, quantiles, horizon
  and instants;
- `{"exploration": {...}}`: the exploration result, a `raichu.exploration`
  document (version 1 for an exact exploration, 2 for a discretised one;
  see the [sequence-tree exploration guide](../guides/sequence-tree-exploration.md)).

### Example

An exact exploration of a parallel pair (the detail shortened):

```json
{
  "format": "raichu.quantification",
  "version": 1,
  "method": {"name": "exact", "settings": {"min_probability": null, "max_length": null,
             "max_failures": null, "max_branches": null, "gap_tolerance": 0.01,
             "rel_precision": 1e-9, "max_terms": 100000}},
  "provenance": {"engine_version": "0.60.0", "model": "parallel_pair",
                 "model_hash": "sha256:e600909d44c8e90c5969ffdb4d02e660b175f9438e826e9c030ddca384a04639",
                 "target": "both_down", "horizon": 5.0},
  "probability": {"kind": "bounds", "lower": 0.24872005926435417,
                  "upper": 0.24872005926435417, "inconclusive": false},
  "detail": {"exploration": {"format": "raichu.exploration", "version": 1, "...": "..."}}
}
```

## Reading

A reader refuses:

- another `format`, by name, before reading anything else;
- a `version` it does not know (the version is raised whenever the format
  changes, a new method included);
- a document whose method, probability and detail do not belong together:
  an exploration detail under the `monte_carlo` method, an exact result
  under the `discretised` method, bounds that are not the exploration's
  own, or an estimate, replica count, level or seed that disagrees with the
  Monte-Carlo detail and the method settings;
- an exploration detail that its own reader refuses.

Numbers are read correctly rounded, so an envelope written by the engine
reads back to the same bits.

## Adding an engine

The three names are not a closed list. A later engine adds a `method`
name with its settings, and a `probability` kind if it states a new sort
of uncertainty; the envelope's other members keep their meaning. Such an
addition raises the format version, so a reader written before it refuses
the new envelopes by their version, as it refuses any version it does not
know, rather than failing on an unknown name.
