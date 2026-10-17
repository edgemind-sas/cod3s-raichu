# Fault-tree envelope format

A fault tree generated from a model (see [Fault trees](../guides/fault-tree.md))
is quantified into one document, the **fault-tree envelope**: what the tree
explains, how it was generated, its probability and that of every gate at
each mission time asked for, and at the last of them, the horizon, its
minimal cut sets and the importance of each basic event. RAICHU produces it and versions it, so a
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
assert envelope["version"] == pyraichu.FAULT_TREE_VERSION == 2
assert envelope["top"]["kind"] == "expression"
assert envelope["horizon"]["probability"] == envelope["instants"][-1]["probability"]
top_gate = envelope["horizon"]["gates"][0]
assert top_gate["id"] == "top" and top_gate["probability"] == envelope["horizon"]["probability"]
p = (1 - math.exp(-1)) * (1 - math.exp(-2))
assert abs(envelope["horizon"]["probability"] - p) < 1e-12
assert pyraichu.read_fault_tree(json.dumps(envelope)) == envelope
```

## Members

Version 2. Every member is present; a member with no value is `null`.
Version 1, which still reads, differs as [Reading and
versions](#reading-and-versions) says.

| Member | Type | Content |
|---|---|---|
| `format` | string | always `"raichu.fault_tree"` |
| `version` | integer | `2` |
| `provenance` | object | `engine_version` (the RAICHU version that generated and quantified the tree) and `tree` (its name, the `define-fault-tree` of its OpenPSA form) |
| `top` | object | what the tree explains: `{"kind": "targets", "targets": [names]}` for the model's declared targets, or `{"kind": "expression", "expression": {...}}` for a top expression, as given |
| `measure` | string | what every probability measures: `"probability_without_repair"`, the probability that the top holds at the mission time when no repair is made (the default), or `"unavailability"`, the probability that it holds then counting the basic events' repairs (see [Measures](#measures)) |
| `generation` | object | see [Generation](#generation) |
| `settings` | object | the quantification settings applied: `max_bdd_nodes`, `cut_set_limit`, `cut_sets`, `engine` (`"auto"`, `"exact"` or `"cut_sets"`), `max_order`, `min_cut_probability`, `max_cut_sets`, `max_expansions` (see the guide's [Quantification](../guides/fault-tree.md#quantification)) |
| `instants` | array | one entry per mission time asked for, in increasing time, the last being the horizon: see [Instants](#instants) |
| `horizon` | object | the quantification at the horizon: see [Horizon](#horizon) |

### Generation

| Member | Type | Content |
|---|---|---|
| `exact` | boolean | whether the numbers are the model's own in the envelope's measure: no warning |
| `warnings` | array of coded warnings | the generation's warnings (one per transition and reason concerned, equal warnings listed once), then the measure's, once per basic event concerned; each `message` is the sentence under the envelope's measure: see [Coded warnings](#coded-warnings) |
| `basic_events` | array | the basic events, each a transition's draw: `name` (unique in the tree), `component`, `automaton` (qualified), `transition`, `target` (the state it leads to), `law`, `{"law": <name>, <parameters>}` with `<name>` one of `exponential` (`rate`), `weibull` (`shape`, `scale`), `lognormal` (`mu`, `sigma`), `gamma` (`shape`, `scale`), `uniform` (`low`, `high`), `delay` (`time`), `empirical` (`points`), `probability` (`probability`), and `repairs`: the transitions from `target` back to the state the event leaves, each `{"transition", "law", "guarded"}` with `law` as above or `null` when no single law stands for it (absent in a document written before repairs were carried, read as none) |

### Instants

At most 20 mission times, finite, non-negative and strictly increasing. Each
entry:

| Member | Type | Content |
|---|---|---|
| `mission_time` | number | the instant |
| `probability` | number | the top's probability at that instant, in the envelope's `measure` |
| `method` | string | `"bdd"`, `"cut_sets"` or `"bdd+cut_sets"` |
| `exact` | boolean | whether the quantification is exact (no cutoff, no bound) |
| `upper_bound` | number or null | a guaranteed upper bound, equal to `probability` when exact; `null` when none can be certified, which includes an unavailability an approximation of unknown direction touches (see [Measures](#measures)) |
| `warnings` | array of coded warnings | the quantifier's warnings at that instant (a module that fell back to its cut sets, a cutoff that bit): see [Coded warnings](#coded-warnings) |
| `gates` | array | the probability of every gate at that instant: see [Gates](#gates) |

The minimal cut sets and the importance measures are computed at the horizon
only.

### Gates

One entry per gate of the tree, the top's first (when the top is a gate),
then the others in preorder, at every instant and at the horizon:

| Member | Type | Content |
|---|---|---|
| `id` | string | the gate's occurrence identifier, as in the structure's `tree.top` (`top`, `G1`, `G2`, ...): the key to its `origin` and its OpenPSA name |
| `probability` | number or null | its exact probability at that instant, in the envelope's `measure`; `null` when it has none, with the `reason` |
| `reason` | string or null | why `probability` is `null`; `null` when it is not |

A gate is read off what the quantifier's normalisation made of it: a node of
its module's decision diagram, whose probability the diagram gives with the
top's (the diagram nodes of a gate merged into its parent are built beside
the top's, within the module's budget); a single basic event, whose
probability it is; or a constant, `0` or `1`. The top gate's `probability` is
the instant's `probability`, to the bit, when the quantification is exact.

| `reason` | Meaning |
|---|---|
| `cut_sets` | the gate lies in a module quantified from its cut sets (no diagram to read it off; the module's own value is an estimate) |
| `reads_cut_sets` | the gate reads such a module: its value would be an estimate too |
| `outgrew_bdd` | the gate's own diagram nodes outgrew its module's budget |
| `not_reached` | the top, once simplified, does not depend on the gate |

A `null` probability is a partial result announced, not a refusal: the top
is still quantified, as `method`, `exact` and `upper_bound` say. A gate
reduced to a single basic event keeps that event's probability even in a
module quantified from its cut sets.

### Coded warnings

Each warning is an object: a `kind`, the parameters of that kind, and
`message`, the English sentence derived from them. A consumer keys on `kind`
and its parameters, never on `message`; a reader ignores `message`.

| `kind` | Parameters | Meaning |
|---|---|---|
| `guard_reads_state` | `transition` | generation: the transition is guarded by a condition on states; its draw is timed from the start of the mission |
| `entered_during_mission` | `transition`, `source` | generation: the transition leaves a state entered during the mission; its draw is timed from the start of the mission |
| `competing_transition` | `transition`, `source`, `competitors` | generation: other transitions leave the same source; the draw is counted as if none could fire first |
| `target_left` | `automaton`, `state`, `transitions` | generation: the state a basic event enters is left for another state than the one the event leaves (not a repair); the tree ignores it |
| `repair_ignored` | `event`, `automaton`, `state`, `transitions` | measure: the event's repairs are not counted (always under the probability without repair; under the unavailability, a repair whose law cannot be read or a failure with no time); the event keeps its probability without repair |
| `asymptotic_unavailability` | `event` | measure: under the unavailability, a failure or repair law is not exponential and the event is its asymptotic `MTTR / (MTTF + MTTR)` |
| `common_cause_repair_approximation` | `event` | measure: under the unavailability, a common-cause combination (`__cc_`) is valued with its own repair, as if independent of the mode's other combinations |
| `guarded_repair_assumed_available` | `event`, `transition` | measure: under the unavailability, a guarded repair is counted as if its guard always held |
| `module_outgrew_bdd` | `module`, `max_bdd_nodes` | quantification: the module's exact diagram outgrew the budget, it was quantified from its cut sets |
| `max_cut_sets_reached` | `module`, `max_cut_sets` | quantification: the count cutoff dropped the least probable cut sets |
| `max_expansions_exhausted` | `module`, `max_expansions` | quantification: the expansion budget ran out, what remained open is only bounded |
| `neglected_mass_exceeds_estimate` | `module`, `neglected`, `estimate` | quantification: the mass the cutoffs neglected exceeds the retained estimate |
| `legacy` | `message` | a warning of a version-1 document, which carried a plain sentence and no code; never written in version 2 |

`module` is the module's index in the horizon's `provenance.modules`, named
`M<module>` there. The structural generation codes say the tree
over-estimates under the probability without repair; under the
unavailability the direction is not known, and their `message` says only
what the tree does.

### Measures

Under `"probability_without_repair"`, a basic event's probability is its
law's distribution at the mission time, its repairs ignored. Under
`"unavailability"`, an event without repair keeps that distribution, and a
repairable one is the probability of being in the state it enters:

- an exponential failure of rate λ with exponential repairs, their rates
  summed into µ, is `λ/(λ+µ) (1 - e^{-(λ+µ)t})`, exactly;
- any other pair is its asymptotic unavailability `MTTR / (MTTF + MTTR)`,
  from the laws' means (an on-demand or immediate repair has a mean time of
  zero; competing repairs combine as rates, `1 / MTTR = Σ 1 / MTTR_i`),
  with `asymptotic_unavailability`.

`asymptotic_unavailability`, `common_cause_repair_approximation` and
`guarded_repair_assumed_available` are approximations of unknown direction:
when one is present, `upper_bound` is `null` at every instant and at the
horizon, and so is the `upper_bound` of every module of
`provenance.modules` that reads such an event.

The OpenPSA document of a generated tree carries what the format has no
element for as attributes: a `measure` attribute of the
`define-fault-tree` (the measure the tree was generated for, which
`FaultTree.quantify` uses by default), and a `repairs` attribute of each
repairable basic event holding its repairs' JSON list.

### Horizon

| Member | Type | Content |
|---|---|---|
| `mission_time`, `probability`, `method`, `exact`, `upper_bound`, `warnings`, `gates` | | as for an instant, at the horizon |
| `coherent` | boolean | whether the top is monotone in every basic event |
| `cut_set_count` | integer or null | the number of minimal cut sets (exact when `cut_sets_complete`); `null` when they were not extracted |
| `cut_sets_complete` | boolean | whether no cutoff removed any cut set |
| `cut_sets_omitted` | string or null | why the cut sets are not listed, when they are not |
| `minimal_cut_sets` | array or null | each `{"events": [names, sorted], "order": n, "probability": p}`, by order then names; `probability` is the product of the events' probabilities at the horizon (they are independent), `null` when one has none |
| `importance` | array | per basic event, in the tree's order: `event` (its name), `probability`, `birnbaum`, `criticality`, `fussell_vesely`, `diagnostic`, `risk_achievement_worth`, `risk_reduction_worth`; a ratio whose denominator is zero is `null` |
| `provenance` | object | the quantifier's own record: variable-ordering heuristic, engine, cutoffs, and per module its method, variables, diagram size and estimators |

## Reading and versions

A reader checks `format` first and `version` second, and refuses a document of
another format, or of a version above the one it knows, by name: a version 2
reader refuses a version 3 document rather than reading it with members
missing. Any change to the members makes a new version.

This engine writes version 2 and reads versions 1 and 2, each by its own
rules. Version 1, written up to RAICHU 0.99.0, differs in two ways:

- every warning (`generation.warnings`, and `warnings` at each instant and at
  the horizon) is a plain English sentence, with no code. It reads as the
  variant `{"kind": "legacy", "message": <the sentence>}`, since a sentence
  cannot be read as a code; `pyraichu.read_fault_tree` returns it in that
  form;
- no instant and no horizon carries `gates`, and the structure's gates carry
  no `id` nor `origin`.

A version 2 document carries no plain sentence and no `legacy` warning: the
reader refuses one, naming it.

## Independent structural output

`FaultTree.structure()` returns `raichu.fault_tree.structure` version 2.
It is separate from the quantified envelope above. No mission time,
probability or importance is computed for this result. `pyraichu.read_fault_tree_structure` accepts JSON text, a file
path or a dictionary and invokes the Rust producer's reader.

The Rust API is `fault_tree_structure(tree, name, top, settings, source)` and
`read_fault_tree_structure(json)` in `raichu-fta`.

| Member | Content |
|---|---|
| `format`, `version` | `"raichu.fault_tree.structure"`, `2` |
| `provenance` | `engine_version`, `tree` name, `source`, `generated_at_unix_ms`, `dependency_lock_hash` |
| `top` | Same declared expression or targets descriptor as the quantified envelope |
| `tree` | `top` nested nodes, `basic_events` and `warnings` (coded, see [Coded warnings](#coded-warnings)), as generated by the core |
| `settings` | `cut_sets` (boolean), `cut_set_limit` (structural expansion budget) |
| `minimal_cut_sets` | Sorted names within each set, sets by size then name; `null` if omitted |
| `cut_sets_omitted` | `"not requested"` when omitted, otherwise `null` |

A `tree.top` node is `{"node":"basic","event":index}`, or
`{"node":"gate","gate":"and"|"or"|"at_least","children":[...]}`;
`at_least` also carries `k`. A generated gate also carries `id`, its
occurrence identifier: unique in the tree, the same from one generation of a
model to the next, `top` for the top gate and `G1`, `G2`, ... for the others
in preorder; it is the gate's name in the OpenPSA document and in the
quantifier. A gate may carry `origin`, what it stands for in the model:
`{"kind": "attribute"|"state"|"transition", "component", "name", "negated"}`,
where `name` is the attribute, `automaton.state` or `automaton.transition`
within the component, and `negated` is true when the gate holds while the
attribute is false (the loss of a flow). An origin may repeat, since an
attribute read in two places is unrolled into two gates; the identifier does
not. A gate with an origin is never merged into a parent of its connective; a
gate with a single child still reduces to it, the child keeping its own origin
or taking the gate's, a basic event dropping it. Both members are optional
when reading, and the reader refuses a repeated identifier. A basic event has `name`, `component`,
`automaton`, `transition`, `target`, and flattened law parameters, for
example `{"law":"exponential","rate":0.001}`. An index references
`tree.basic_events`. Generation warnings retain their static scientific
meaning; an empty warnings array means exact generation.

The reader reads versions 1 and 2 by the rules of the envelope's
[Reading and versions](#reading-and-versions): a version-1 warning is a plain
sentence, read as `{"kind": "legacy", "message"}`, and a version-1 gate has
no `id` nor `origin`; a version-2 structure carrying a sentence or a
`legacy` warning is refused. The reader refuses unknown formats or versions
(version 3 by name), missing required fields,
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
