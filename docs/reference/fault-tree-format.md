# Fault-tree envelope format

A fault tree generated from a model (see [Fault trees](../guides/fault-tree.md))
is quantified into one document, the **fault-tree envelope**: what the tree
explains, how it was generated, its probability at each mission time asked
for, and at the last of them, the horizon, its minimal cut sets and the
importance of each basic event. RAICHU produces it and versions it, so a
reader refuses a document it does not know rather than showing an empty field.

The contract lives in the crate `raichu-fta` (re-exported by the umbrella crate
as `raichu::raichu_fta`): `fault_tree_envelope(tree, name, top, mission_times,
settings)` writes it, `read_fault_tree_envelope` checks one.

## From Python

`FaultTree.envelope(mission_times, **settings)` returns the envelope as a
dictionary; `pyraichu.read_fault_tree(document)` checks the format and the
version through the engine's own reader before returning one, from a JSON
text, a file path or a dictionary.

```python
import json
import math

import pyraichu


def unit(name, rate):
    return {
        "name": name,
        "automata": [{
            "name": "health", "states": ["ok", "nok"], "init": "ok",
            "transitions": [{"name": "fail", "source": "ok", "targets": ["nok"],
                             "distrib": "exp", "rate": rate}],
        }],
    }


def nok(name):
    return {"op": "state_active",
            "state": {"component": name, "automaton": "health", "state": "nok"}}


model = pyraichu.load_model({"name": "pair", "components": [unit("A", 1e-3), unit("B", 2e-3)]})
tree = pyraichu.fault_tree(model, {"op": "bool", "bool_op": "and", "args": [nok("A"), nok("B")]})
envelope = tree.envelope([100.0, 1000.0])

assert envelope["format"] == pyraichu.FAULT_TREE_FORMAT == "raichu.fault_tree"
assert envelope["version"] == pyraichu.FAULT_TREE_VERSION == 1
assert envelope["top"]["kind"] == "expression"
assert envelope["horizon"]["probability"] == envelope["instants"][-1]["probability"]
p = (1 - math.exp(-1)) * (1 - math.exp(-2))
assert abs(envelope["horizon"]["probability"] - p) < 1e-12
assert pyraichu.read_fault_tree(json.dumps(envelope)) == envelope
```

## Members

Version 1. Every member is present; a member with no value is `null`.

| Member | Type | Content |
|---|---|---|
| `format` | string | always `"raichu.fault_tree"` |
| `version` | integer | `1` |
| `provenance` | object | `engine_version` (the RAICHU version that generated and quantified the tree) and `tree` (its name, the `define-fault-tree` of its OpenPSA form) |
| `top` | object | what the tree explains: `{"kind": "targets", "targets": [names]}` for the model's declared targets, or `{"kind": "expression", "expression": {...}}` for a top expression, as given |
| `measure` | string | what every probability measures: `"probability_without_repair"`, the probability that the top holds at the mission time when no repair is made |
| `generation` | object | see [Generation](#generation) |
| `settings` | object | the quantification settings applied: `max_bdd_nodes`, `cut_set_limit`, `cut_sets`, `engine` (`"auto"`, `"exact"` or `"cut_sets"`), `max_order`, `min_cut_probability`, `max_cut_sets`, `max_expansions` (see the guide's [Quantification](../guides/fault-tree.md#quantification)) |
| `instants` | array | one entry per mission time asked for, in increasing time, the last being the horizon: see [Instants](#instants) |
| `horizon` | object | the quantification at the horizon: see [Horizon](#horizon) |

### Generation

| Member | Type | Content |
|---|---|---|
| `exact` | boolean | whether the tree is the model's exact structure (no warning): its probability without repair is then the model's own |
| `warnings` | array of strings | why the tree may over-estimate, one sentence per transition concerned: a repair ignored, a failure guarded by a condition on states, a draw from a state entered during the mission, a competing transition |
| `basic_events` | array | the basic events, each a transition's draw: `name` (unique in the tree), `component`, `automaton` (qualified), `transition`, `target` (the state it leads to), and `law`, `{"law": <name>, <parameters>}` with `<name>` one of `exponential` (`rate`), `weibull` (`shape`, `scale`), `lognormal` (`mu`, `sigma`), `gamma` (`shape`, `scale`), `uniform` (`low`, `high`), `delay` (`time`), `empirical` (`points`), `probability` (`probability`) |

### Instants

At most 20 mission times, finite, non-negative and strictly increasing. Each
entry:

| Member | Type | Content |
|---|---|---|
| `mission_time` | number | the instant |
| `probability` | number | the top's probability without repair at that instant |
| `method` | string | `"bdd"`, `"cut_sets"` or `"bdd+cut_sets"` |
| `exact` | boolean | whether the quantification is exact (no cutoff, no bound) |
| `upper_bound` | number or null | a guaranteed upper bound, equal to `probability` when exact; `null` when none can be certified |
| `warnings` | array of strings | the quantifier's warnings at that instant (a module that fell back to its cut sets, a cutoff that bit) |

The minimal cut sets and the importance measures are computed at the horizon
only.

### Horizon

| Member | Type | Content |
|---|---|---|
| `mission_time`, `probability`, `method`, `exact`, `upper_bound`, `warnings` | | as for an instant, at the horizon |
| `coherent` | boolean | whether the top is monotone in every basic event |
| `cut_set_count` | integer or null | the number of minimal cut sets (exact when `cut_sets_complete`); `null` when they were not extracted |
| `cut_sets_complete` | boolean | whether no cutoff removed any cut set |
| `cut_sets_omitted` | string or null | why the cut sets are not listed, when they are not |
| `minimal_cut_sets` | array or null | each `{"events": [names, sorted], "order": n, "probability": p}`, by order then names; `probability` is the product of the events' probabilities at the horizon (they are independent), `null` when one has none |
| `importance` | array | per basic event, in the tree's order: `event` (its name), `probability`, `birnbaum`, `criticality`, `fussell_vesely`, `diagnostic`, `risk_achievement_worth`, `risk_reduction_worth`; a ratio whose denominator is zero is `null` |
| `provenance` | object | the quantifier's own record: variable-ordering heuristic, engine, cutoffs, and per module its method, variables, diagram size and estimators |

## Reading and versions

A reader checks `format` first and `version` second, and refuses a document of
another format, or of a version above the one it knows, by name: a version 1
reader refuses a version 2 document rather than reading it with members
missing. Any change to the members makes a new version.

## Independent structural output

`FaultTree.structure()` returns `raichu.fault_tree.structure` version 1.
It is separate from the quantified envelope above, whose version 1 contract
is unchanged. No mission time, probability or importance is computed for
this result. `pyraichu.read_fault_tree_structure` accepts JSON text, a file
path or a dictionary and invokes the Rust producer's reader.

The Rust API is `fault_tree_structure(tree, name, top, settings, source)` and
`read_fault_tree_structure(json)` in `raichu-fta`.

| Member | Content |
|---|---|
| `format`, `version` | `"raichu.fault_tree.structure"`, `1` |
| `provenance` | `engine_version`, `tree` name, `source`, `generated_at_unix_ms`, `dependency_lock_hash` |
| `top` | Same declared expression or targets descriptor as the quantified envelope |
| `tree` | `top` nested nodes, `basic_events` and `warnings`, as generated by the core |
| `settings` | `cut_sets` (boolean), `cut_set_limit` (structural expansion budget) |
| `minimal_cut_sets` | Sorted names within each set, sets by size then name; `null` if omitted |
| `cut_sets_omitted` | `"not requested"` when omitted, otherwise `null` |

A `tree.top` node is `{"node":"basic","event":index}`, or
`{"node":"gate","gate":"and"|"or"|"at_least","children":[...]}`;
`at_least` also carries `k`. A basic event has `name`, `component`,
`automaton`, `transition`, `target`, and flattened law parameters, for
example `{"law":"exponential","rate":0.001}`. An index references
`tree.basic_events`. Generation warnings retain their static scientific
meaning; an empty warnings array means exact generation.

The reader refuses unknown formats or versions, missing required fields,
unknown event references, duplicate event names, constant trees and
inconsistent omission markers. It does not regenerate or quantify the tree.
An omitted collection is distinct from an empty computed collection.
Structural extraction exceeding its explicit budget fails instead of
publishing partial minimal cut sets.

### Structural provenance

The Python producer records `source.kind="model"`, the source model's
canonical sealed identity (`model_hash`, computed by
`raichu_quantify::model_content_hash`) and its actual generation configuration:
`generation.max_nodes` is the effective budget, including the resolved default;
`generation.profile` is the qualified-name and typed-value pairs applied to
backward chaining. The requested expression or target names are retained in
`top`; extraction configuration is in `settings`.

`generated_at_unix_ms` records structural result production time in UTC as
milliseconds since the Unix epoch. `dependency_lock_hash` is `sha256:` plus
the SHA-256 of the exact Cargo.lock embedded when the producer was compiled;
it identifies the resolved dependency versions, not a guessed runtime environment.

The Rust producer requires an explicit `StructureSource`. Use `Model` only
when the source hash and actual generation configuration are known. Use
`SuppliedTree` for a supplied tree whose source model and generation
configuration are unknown; the serialized `source` is then
`{"kind":"supplied_tree"}` and makes no fabricated model identity claim.
The timestamp in this case dates result packaging, not unknown original tree
generation. The supplied tree itself is carried unchanged in `tree`.

Seeds, simulation mission parameters and trajectory counts are inapplicable:
this is deterministic structural analysis without simulation or quantification.
The existing numerical envelope v1 has not changed.
