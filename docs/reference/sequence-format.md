# Raw sequence corpus format

A sequence campaign records, for every trajectory, the ordered monitored
events it went through and the feared event it stopped at. The analysis
reduces that corpus to minimal sequences; the **raw corpus** is what the
reduction starts from. Written as an artefact, it lets anyone audit a
campaign, recompute its reduction, or filter it, without running it again.

`pyraichu.run_sequences(..., raw_path=...)` writes it, straight from the
engine; `pyraichu.analyse_raw_sequences(path)` reads it back and reduces it
with the engine's own pipeline.

## `raichu.sequences`, version 1

[JSON Lines](https://jsonlines.org): one JSON document per line, UTF-8.

**Line 1 is the header:**

| Field | Type | Meaning |
|---|---|---|
| `format` | string | always `"raichu.sequences"` |
| `version` | integer | `1` |
| `engine_version` | string | the engine version that ran the campaign |
| `model` | string | the model's name |
| `seed` | integer | the campaign's master seed |
| `nb_runs` | integer | the number of replicas, hence of trajectory lines |
| `t_max` | number | the horizon |
| `targets` | array of strings | the feared events the campaign stops at, in declaration order |
| `event_fields` | array of strings | the positions of an event array: `["time", "obj", "attr", "cycle_group"]` |

**Then one line per trajectory**, in replica order:

| Field | Type | Meaning |
|---|---|---|
| `run` | integer | the 0-based replica index |
| `end_cause` | string or `null` | the feared event reached, or `null` when the trajectory ran to the horizon |
| `end_time` | number | when it stopped |
| `events` | array of event arrays | the monitored events, in firing order |

An **event array** is `[time, obj, attr, cycle_group]`: the firing date, the
component, the monitored state entered, and the cycle group of the transition
that fired (`null` when it has none).

```json
{"format":"raichu.sequences","version":1,"engine_version":"0.44.0","model":"redundant_pair","seed":42,"nb_runs":2000,"t_max":100.0,"targets":["system_down"],"event_fields":["time","obj","attr","cycle_group"]}
{"run":0,"end_cause":"system_down","end_time":14.939367483634282,"events":[[8.75883378255026,"fm","occ__cc_1","fm__cc_1"],[11.696235051084663,"fm","rep__cc_1","fm__cc_1"],[14.9316885842181,"fm","occ__cc_1","fm__cc_1"],[14.939367483634282,"fm","occ__cc_2","fm__cc_2"],[14.939367483634282,"system_down","occ","system_down"]]}
```

The first two lines of the redundant pair of the
[sequence-analysis guide](../guides/sequence-analysis.md), 2000 replicas
(857 kB in all): replica 0 saw `A` fail at 8.76 and be repaired at 11.70, a
cycle the reduction removes, then fail again at 14.93 and `B` fail at 14.94,
which reached the feared event.

Every trajectory weighs one, and the file holds exactly `nb_runs` trajectory
lines. The **cycle group** is what the reduction's cycle filter reads (a
failure repaired before the feared event did not cause it): carried in the
file, it lets the reduction be recomputed from it alone.

Dates are written with the shortest decimal that reads back to the same
double, and the engine's reader rounds correctly, so a corpus reads back bit
for bit and re-reduces to exactly the levels its campaign returned.

## Reading rules

A reader refuses a corpus whose `format` is not `"raichu.sequences"` or whose
`version` is above the one it knows, a trajectory line out of replica order,
and a trajectory count that is not the header's. A later version may add
fields to either kind of line; a version 1 reader ignores fields it does not
know.

## The three levels

| Level | What it holds | Where it comes from |
|---|---|---|
| raw | one sequence per trajectory, every event, dates included | this file |
| cleaned | trajectories grouped by their ordered events, transient failure/repair cycles removed, grouped again | `SequenceCampaign.cleaned` |
| minimal | the cleaned sequences, each longer one absorbed into a shorter one it contains | `SequenceCampaign.minimal`, `analyse_sequences` |

The weights of the cleaned level sum to `nb_runs`; the reduction from the raw
corpus to the minimal level is the stated identity
`analyse(raw) = minimal(clean(raw))`.
