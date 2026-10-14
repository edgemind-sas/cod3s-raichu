# Raw sequence corpus format

A sequence campaign records, for every trajectory, the ordered monitored
events it went through and the feared event it stopped at. The analysis
reduces that corpus to minimal sequences; the **raw corpus** is what the
reduction starts from. Written as an artefact, it lets anyone audit a
campaign, recompute its reduction, or filter it, without running it again.

`pyraichu.run_sequences(..., raw_path=...)` writes it, straight from the
engine; `pyraichu.analyse_raw_sequences(path)` reads it back and reduces it
with the engine's own pipeline.

## `raichu.sequences`, version 2

[JSON Lines](https://jsonlines.org): one JSON document per line, UTF-8.

**Line 1 is the header:**

| Field | Type | Meaning |
|---|---|---|
| `format` | string | always `"raichu.sequences"` |
| `version` | integer | `2` (`1` before 0.86.0) |
| `engine_version` | string | the engine version that ran the campaign |
| `model` | string | the model's name |
| `seed` | integer | the campaign's master seed |
| `nb_runs` | integer | the number of replicas, hence of trajectory lines |
| `t_max` | number | the horizon |
| `targets` | array of strings | the feared events the campaign stops at, in declaration order |
| `event_fields` | array of strings | the positions of an event array: `["time", "obj", "attr", "cycle_group"]` |
| `cycle_groups` | array of objects, optional | the per-group decisions of the model the campaign ran, which the reduction applies (see [Cycle-group decisions](#cycle-group-decisions)); absent from a version 1 corpus |
| `observations` | array of objects, optional | what each trajectory's `observed` array holds, `{"name", "time"}` each; absent when nothing is observed (see [Observations](#observations)) |

**Then one line per trajectory**, in replica order:

| Field | Type | Meaning |
|---|---|---|
| `run` | integer | the 0-based replica index |
| `end_cause` | string or `null` | the feared event reached, or `null` when the trajectory ran to the horizon |
| `end_time` | number | when it stopped |
| `events` | array of event arrays | the monitored events, in firing order |
| `automata` | array of strings, optional | the automaton of each event inside its component, parallel to `events`; absent when the source names none |
| `kinds` | array of strings or `null`, optional | the declared role of each event (`"failure"`, `"repair"`, `"observation"`, or `null`), parallel to `events`; absent when no event declares one |
| `observed` | array of numbers, optional | one value per header observation, in that order; absent when nothing is observed |

An **event array** is `[time, obj, attr, cycle_group]`: the firing date, the
component, the monitored state entered, and the cycle group of the transition
that fired (`null` when it has none).

```json
{"format":"raichu.sequences","version":2,"engine_version":"0.96.0","model":"redundant_pair","seed":42,"nb_runs":2000,"t_max":100.0,"targets":["system_down"],"event_fields":["time","obj","attr","cycle_group"],"cycle_groups":[{"component":"fm","cycle_group":"fm__cc_1","automata":["fm.fm__cc_1"],"targets":[{"target":"system_down","cancellable":true,"origin":"analysis"}]},{"component":"fm","cycle_group":"fm__cc_1_2","automata":["fm.fm__cc_1_2"],"targets":[{"target":"system_down","cancellable":true,"origin":"analysis"}]},{"component":"fm","cycle_group":"fm__cc_2","automata":["fm.fm__cc_2"],"targets":[{"target":"system_down","cancellable":true,"origin":"analysis"}]},{"component":"system_down","cycle_group":"system_down","automata":["system_down.ev"],"targets":[{"target":"system_down","cancellable":true,"origin":"analysis"}]}]}
{"run":0,"end_cause":"system_down","end_time":14.939367483634282,"events":[[8.75883378255026,"fm","occ__cc_1","fm__cc_1"],[11.696235051084663,"fm","rep__cc_1","fm__cc_1"],[14.9316885842181,"fm","occ__cc_1","fm__cc_1"],[14.939367483634282,"fm","occ__cc_2","fm__cc_2"],[14.939367483634282,"system_down","occ","system_down"]],"automata":["fm__cc_1","fm__cc_1","fm__cc_1","fm__cc_2","ev"],"kinds":["failure","repair","failure","failure","observation"]}
```

The first two lines of the redundant pair of the
[sequence-analysis guide](../guides/sequence-analysis.md), 2000 replicas
(1.2 MB in all): replica 0 saw `A` fail at 8.76 and be repaired at 11.70, a
cycle the reduction removes, then fail again at 14.93 and `B` fail at 14.94,
which reached the feared event.

The `automata` array tells apart two automata of one component that reach
states of the same name (two failure modes grafted on one pump both reach
`occ`): the reduction keys on `(obj, attr)` and does not read it, while the
[importance measures](../guides/importance-measures.md) name a basic event
`component.automaton.state`. It is a field added to version 1 under the
format's extension rule: a reader that predates it reads the corpus as
before.

The `kinds` array gives the reliability role the model declares for the
transition of each event, and only for an entry into that transition's first
declared target, the one the declaration speaks for: `"failure"`, `"repair"`,
`"observation"`, or `null` when it declares none. The reduction's safety net
reads it (see [The three levels](#the-three-levels)). It was added to version
1 in 0.85.0 under the same extension rule: a reader that predates it reduces
without the safety net, as engines before 0.85.0 did.

Every trajectory weighs one, and the file holds exactly `nb_runs` trajectory
lines. The **cycle group** is what the reduction's cycle filter reads (a
failure repaired before the feared event did not cause it): carried in the
file, it lets the reduction be recomputed from it alone.

Dates are written with the shortest decimal that reads back to the same
double, and the engine's reader rounds correctly, so a corpus reads back bit
for bit and re-reduces to exactly the levels its campaign returned.

## Observations

A campaign may also read, on every trajectory, the value of chosen
attributes at chosen instants: `pyraichu.run_sequences(...,
observations=[Observation(name, component, attribute, time)])`. The header
then lists them, and every trajectory line carries their values:

```json
{"format":"raichu.sequences","version":2,...,"observations":[{"name":"A_flow","time":50.0}]}
{"run":0,"end_cause":null,"end_time":100.0,"events":[...],"observed":[1]}
```

The value is the state the trajectory was in at that instant: after any
event at that very date, and, for a trajectory that stopped at a feared
event earlier, the state it stopped in. An instant past the horizon is read
at the horizon, and the header records the instant actually read. A boolean
reads `0` or `1`. Observing does not change the trajectories: the same seed
gives the same corpus, observations added.

What observations are for is a **condition**: `SequenceCondition(observation,
op, value)` keeps the trajectories whose observed value compares to `value`
as `op` says (`==`, `!=`, `<`, `<=`, `>`, `>=`, on the recorded double,
exactly), and the levels are reduced from those alone. Passed to
`run_sequences`, it filters the campaign's own levels; passed to
`analyse_raw_sequences`, it filters a corpus read back. Either way the raw
corpus keeps every trajectory, so the same file answers another condition
later, and the result's `condition` field reports how many trajectories the
condition kept out of how many. A condition naming an observation the
corpus does not carry is refused: keeping every trajectory would read as a
condition that held on all of them.

Both fields are optional within version 1: a corpus that observes nothing
is written exactly as before, and a reader that predates them ignores them
and still reads the trajectories right.

## Cycle-group decisions

The header's `cycle_groups` lists, for each `(component, cycle_group)` of
the model, whether the reduction may cancel the group, per feared event
(`pyraichu.cycle_groups` gives the same list from a model):

| Field | Meaning |
|---|---|
| `component`, `cycle_group` | the group, as events name it (`obj` and the fourth position of an event array) |
| `automata` | the automata whose transitions belong to it, by qualified name |
| `targets` | one decision per feared event: `target`, `cancellable`, `origin` (`analysis` or `declared`), and when the analysis keeps the group, `witness` (the path from the group to the feared event) and `memory` (the quantity on it that remembers) |

A group not cancellable for a trajectory's end cause keeps all its events
in the cleaned level. A trajectory that reached no feared event cancels
every group: it has no cause to keep, and keeping its history would make
nearly every such trajectory a path of its own. The decisions are why this is version 2: a
version 1 reader ignoring them would cancel every group, which is exactly
what they prevent, so it refuses the corpus instead. A version 1 corpus
carries none; it still reads, and reduces with every group cancelled.

## Reading rules

A reader refuses a corpus whose `format` is not `"raichu.sequences"` or whose
`version` is above the one it knows, a trajectory line out of replica order,
a trajectory count that is not the header's, and a trajectory whose
`observed` values are not one per declared observation. Fields may be added
to either kind of line without a new version as long as a reader that
ignores them still reads the corpus right; a reader ignores fields it does
not know.

## The three levels

| Level | What it holds | Where it comes from |
|---|---|---|
| raw | one sequence per trajectory, every event, dates included | this file |
| cleaned | trajectories grouped by their ordered events, the failure/repair cycles of cancellable groups removed, grouped again | `SequenceCampaign.cleaned` |
| minimal | the cleaned sequences, each longer one absorbed into a shorter one it contains | `SequenceCampaign.minimal`, `analyse_sequences` |

The weights of the cleaned level sum to `nb_runs`; the reduction from the raw
corpus to the minimal level is the stated identity
`analyse(raw) = minimal(clean(raw))`.

One trajectory keeps its cycles in the cleaned level: one that reached a
feared event, yet that cancellation would leave with no cause, either nothing
at all or only repairs and observations while a declared failure was
cancelled. A feared event cannot happen without a cause, so such a trajectory
proves that one of its repaired failures had a lasting effect; cancelled, it
would also absorb every other sequence of its feared event. The campaign
result reports these trajectories under `reduction`:

| Field | Meaning |
|---|---|
| `contradicted` | how many trajectories kept their cycles for that reason |
| `contradicted_weight` | their total weight (a trajectory count here) |
| `contradicted_indices` | the replica indices of the first hundred, in replica order |
| `cycle_groups` | the decisions the reduction applied, or `null` when it knew none (a version 1 corpus) |
| `cycle_groups_unknown` | `true` when the decisions were unknown while some trajectory carried a cycle group: every group was then cancelled |

Re-reducing a corpus gives the same report. A corpus written before 0.85.0
carries no `kinds`, so only the first case is recognised on it: a
trajectory left with nothing at all.
