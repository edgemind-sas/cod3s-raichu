# Importing COD3S-platform studies

Models authored on a COD3S platform instance are persisted as **two
artefacts**: a *model export* (the topology — component instances of KB
templates, connections, per-instance overrides) and a *study* (the
dynamics — failure modes, feared events, indicators, Monte-Carlo
parameters). `pyraichu.importers.cod3s_platform` fuses both into one
runnable RAICHU model.

## Where the artefacts come from

- **Model export** — the platform's model *export* action (or a database
  dump of the model document): a JSON carrying `model.elements`
  (components + connections) and the embedded knowledge base
  (`kb_embedded`, or `kb` in raw dumps). Both versioned exports and raw
  dumps are accepted.
- **Study** — the study description used by the platform's run
  machinery (a `study.yaml`, parsed to a dict): `failure_modes`,
  `events`, `targets`, `indicators`, `simulation`.

The pair recorded alongside an existing platform run is ideal: you can
then compare RAICHU's results against that run's recorded outputs.

<!-- skip -->
```python
import json, yaml
from pyraichu.importers import translate
import pyraichu

export = json.load(open("model_export.json"))
study = yaml.safe_load(open("study.yaml"))

t = translate(export, study)          # → Translation(model, simulation, measures)
model = pyraichu.load_model(t.model)

est = pyraichu.monte_carlo(
    model,
    nb_runs=t.simulation["nb_runs"],
    t_max=t.simulation["samples"][-1],
    samples=t.simulation["samples"],
    seed=t.simulation.get("seed", 0),
    stop_at_targets=True,             # match the study's target semantics
)
cuts = pyraichu.analyse_sequences(
    model, nb_runs=t.simulation["nb_runs"],
    t_max=t.simulation["samples"][-1], seed=t.simulation.get("seed", 0))
```

`translate_export(export)` alone gives the topology-only model;
`Translation.measures` lists the study's requested measures per
indicator (`nb-occurrences`, `sojourn-time`).

## What the translator understands

**Model export** (versioned exports *and* raw platform DB dumps):

- KB component templates → `ObjFlow` objects: `input_logic`
  (`or` / `and` / integer k-of-n) on inputs, `prod_cond` DNF
  (outer-OR of inner-AND groups, references to in- or out-flows of the
  same component) on outputs;
- per-instance attribute overrides, in both the current role vocabulary
  (`logic_in`, `prod_init`) and the legacy one (`logic`, `init`), with
  the platform's decimal-string k-of-n votes (`"2"`) and strict boolean
  coercion (`"false"` is false);
- UUID-keyed connections, resolved to component/port pairs.

**Study**:

- `failure_modes` — `ObjFMExp` / `ObjFMDelay`, including per-order
  common-cause parameter lists; a **zero exponential rate marks an
  inactive order** (dropped, as the platform does), and an active
  failure with an inactive repair yields a non-repairable mode
  (absorbing failure state);
- `events` — `ObjEvent` feared events; the study's `targets` list flags
  them as sequence-analysis targets;
- `indicators` — state indicators on the declared events, with their
  requested measures;
- `simulation` — `nb_runs`, `schedule` (flattened to `samples`), `seed`,
  `time_unit`, passed through in `Translation.simulation`.

## Matching the study's measures

Platform studies that declare `targets` are **first-occurrence
campaigns**: each trajectory stops at the feared event, and the recorded
indicators latch from the hit to the horizon. To reproduce those
numbers, run the Monte-Carlo with `stop_at_targets=True` (as in the
snippet above) — see
[Sequence analysis](sequence-analysis.md#first-occurrence-indicators)
for the two semantics. `Translation.measures` tells you which measure
each indicator carries:

- `nb-occurrences` → `IndicatorEstimate.nb_occurrences_mean` / `_std`
  (with targets: the probability the event occurred by each instant);
- `sojourn-time` → `IndicatorEstimate.sojourn_mean` / `_std`
  (with targets: mean time elapsed since the first occurrence).

## Converting the outputs

RAICHU's results map line-for-line onto the platform's artefacts. The
indicator table (one row per measure × statistic × instant):

<!-- skip -->
```python
rows = []
for name, measures in t.measures.items():
    ind = est.indicators[name]
    series = {"nb-occurrences": (ind.nb_occurrences_mean, ind.nb_occurrences_std),
              "sojourn-time": (ind.sojourn_mean, ind.sojourn_std)}
    for measure in measures:
        means, stds = series[measure]
        for instant, mean, std in zip(ind.instants, means, stds):
            rows.append({"name": f"{name}_{measure}", "measure": measure,
                         "stat": "mean", "instant": instant, "values": mean})
            rows.append({"name": f"{name}_{measure}", "measure": measure,
                         "stat": "stddev", "instant": instant, "values": std})
```

And the minimal sequences, in the platform's sequence-artefact shape
(`weight` is the trajectory count; divide by `nb_runs` for the
probability):

<!-- skip -->
```python
artefact = {
    "schema_version": "1.0.0",
    "target_group_id": "system_down",
    "sequences": [
        {"weight": s["weight"], "probability": s["weight"] / nb_runs,
         "end_time": s["end_time"], "target_name": s["end_cause"],
         "events": [{"obj": e["obj"], "attr": e["attr"], "time": e["time"]}
                    for e in s["events"]]}
        for s in cuts
    ],
}
```

!!! note "Naming drifts when diffing against recorded runs"
    Older platform runs may write common-cause suffixes without index
    separators (`occ__cc_12` for RAICHU's `occ__cc_1_2`) and prefix the
    failure-mode component with the factorized target name — normalise
    both before comparing sequence sets.

## Fail fast, never silently wrong

Anything outside this scope raises a typed `TranslationError` with the
offending artefact in the message — a tempo/on-trigger flow type, a
`negate` flag, an unknown attribute role, a malformed logic override, a
missing required key. The translator refuses to guess: a model that
translates is a model whose semantics are covered.

<!-- skip -->
```python
from pyraichu.importers import TranslationError

try:
    t = translate(export, study)
except TranslationError as error:
    print("unsupported construct:", error)
```
