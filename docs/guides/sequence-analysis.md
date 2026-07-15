# Sequence analysis: feared events & minimal cut sequences

Safety studies do not only ask *how often* an undesired event occurs —
they ask **which chains of failures lead to it**. RAICHU answers both
natively: declare a **feared event** as a *target*, run a Monte-Carlo
campaign where every trajectory records its causal event sequence and
stops at the first occurrence, then reduce the corpus to **minimal cut
sequences** — the irreducible, ordered failure combinations that reach
the event, weighted by how many trajectories they explain.

## Declaring a feared event

The simplest way is an `ObjEvent` plugin object flagged `"target": true`:
its `occ` state becomes the trajectory-stopping target and the label of
every recorded sequence.

```json
{
  "type": "ObjEvent", "name": "system_down", "target": true,
  "cond": [[
    {"obj": "A", "attr": "flow", "ope": "==", "value": false},
    {"obj": "B", "attr": "flow", "ope": "==", "value": false}
  ]]
}
```

At the native level this is a model-wide `targets` entry — any automaton
state can be one (see the [model schema](../reference/model-schema.md)):

```json
"targets": [{"name": "system_down", "component": "system_down",
             "automaton": "ev", "state": "occ"}]
```

When a target state activates, the engine finishes the current instant
(everything due at the same date still fires), records the **end cause**,
and stops the trajectory — the *first-occurrence* semantics of safety
campaigns.

## From trajectories to minimal cut sequences

`pyraichu.analyse_sequences` runs the campaign and the whole reduction
pipeline:

```python
import pyraichu

model = pyraichu.load_model({
    "name": "redundant_pair",
    "plugins": {"muscadet": {"objects": [
        # One failure mode with common-cause orders over two targets:
        # order 1 = independent failures, order 2 = both at once.
        {"type": "ObjFM", "name": "fm", "targets": ["A", "B"],
         "failure": [{"law": "exp", "rate": 0.1}, {"law": "exp", "rate": 0.03}],
         "repair":  [{"law": "exp", "rate": 0.5}, {"law": "exp", "rate": 0.5}],
         "failure_effects": {"flow": False}},
        {"type": "ObjEvent", "name": "system_down", "target": True,
         "cond": [[{"obj": "A", "attr": "flow", "ope": "==", "value": False},
                   {"obj": "B", "attr": "flow", "ope": "==", "value": False}]]},
    ]}},
    "components": [
        {"name": n,
         "attributes": [{"name": "flow", "kind": "bool",
                         "init": {"kind": "bool", "value": True}}]}
        for n in ("A", "B")
    ],
    "indicators": [{"name": "system_down_occ", "target": "state",
                    "component": "system_down", "automaton": "ev",
                    "state": "occ"}],
})

cuts = pyraichu.analyse_sequences(model, nb_runs=2000, t_max=100.0, seed=42)
for cut in cuts:
    if cut["end_cause"] == "system_down":
        chain = " → ".join(f"{e['obj']}.{e['attr']}" for e in cut["events"])
        print(f"{cut['weight']:5.0f}  {chain}")
```

```text
 1075  fm.occ__cc_1_2 → system_down.occ
  478  fm.occ__cc_2 → fm.occ__cc_1 → system_down.occ
  441  fm.occ__cc_1 → fm.occ__cc_2 → system_down.occ
```

Three minimal cut sequences: the **common-cause** direct path dominates,
and the two **order-dependent** sequential paths (A-then-B vs B-then-A)
are reported separately — sequences are ordered, unlike classic cut
*sets*. Weights are trajectory counts: divide by `nb_runs` for
probabilities. The result is seed-reproducible bit-for-bit.

The pipeline behind the call:

1. **Record** — each trajectory logs its *monitored* transitions (the
   plugin marks failure/repair and event transitions automatically) and
   stops at the first target.
2. **Group** — trajectories with the same ordered event signature merge;
   weights add up.
3. **Cancel transient cycles** — a failure that was repaired *before*
   the feared event did not cause it: paired failure/repair events of
   the same mode are removed (per component, so distinct modes never
   cancel each other).
4. **Minimal absorption** — a sequence that contains a shorter reaching
   sequence is absorbed into it; only irreducible cuts remain.

## First-occurrence indicators

The Monte-Carlo estimator has the matching measures. By default
`monte_carlo` lets trajectories run and cycle freely — an
*availability* view. With `stop_at_targets=True` it applies the same
early-stop as the sequence analysis and **latches** the state: once the
feared event occurs, it stays occurred through every later sampling
instant — a *reliability / first-occurrence* view.

```python
est = pyraichu.monte_carlo(model, nb_runs=2000, t_max=100.0,
                           samples=[50.0, 100.0], seed=42,
                           stop_at_targets=True)
ind = est.indicators["system_down_occ"]
ind.nb_occurrences_mean   # P(occurred by t): at most one per trajectory
ind.sojourn_mean          # mean time elapsed since the first occurrence
```

`nb_occurrences_mean` / `nb_occurrences_std` (the number of state
entries up to each instant) are also estimated in the free-cycling mode,
where an event can occur repeatedly.

!!! note "Which mode to compare with what"
    When cross-checking against another tool, match the semantics:
    campaigns recorded **with** targets measure first-occurrence
    quantities (latched sojourn ≈ horizon − first hit), campaigns
    **without** measure cumulated exposure. The two differ by orders of
    magnitude on repairable systems.

## Native model, without plugins

Sequence recording works on hand-written models too: set
`"monitored": true` on the transitions you want in the sequences, group
failure/repair pairs with a shared `"cycle_group"`, and declare
`targets`. The plugin layer simply does this annotation for you.
