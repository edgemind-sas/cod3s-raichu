# Running a COD3S-platform study

Models authored on a COD3S platform instance are persisted as **two
artefacts**: a *model export* (the topology, component instances of KB
templates, connections, per-instance overrides) and a *study* (the
dynamics: failure modes, feared events, indicators, Monte-Carlo
parameters).

**RAICHU does not read that pair.** One library does, muscadet, whose
importer owns the platform's KB vocabulary; what RAICHU reads is the
**system declaration** of the system muscadet builds from it. A platform
export therefore reaches this engine the way any muscadet model does, and
[Running a muscadet model on RAICHU](muscadet-engine.md) is the whole of
the story:

<!-- skip -->
```python
import yaml
from cod3s.scripts.study_runner import run_study
from muscadet.builders import PlatformExportBuilder

run_study(
    system_builder=PlatformExportBuilder("model_export.json"),
    study=yaml.safe_load(open("study.yaml")),
    results_dir="./results",
)
```

`run_study` composes the system from the export and the study's failure
modes, events, indicators and targets, then runs it on the engine the
study selects. RAICHU is selected the same way as anywhere else, by name.

## Why there is no importer here

There used to be a second reader of the same pair inside `pyraichu`, so
one document had two readings and nothing said when they parted. They
did part, in ways a run never announced: a continuous port read as a
discrete boolean because that reader knew no flow family; a condition
clause naming no object read without a guard; an on-demand occurrence
routed to an expander that only built the internal behaviour. Each was a
model quietly different from the one the analyst described.

One reader is the fix, and the one kept is the one the platform itself
runs. What RAICHU owns is what it can answer for: the engine, and the
reading of a **declaration**.

## Getting at the model without running it

`pyraichu.muscadet.engine.build_model` is the translation, reachable
without simulating anything: hand it a declaration
(`muscadet.declare.system_spec` of a live system) and it answers the
model RAICHU would run, sequence targets included.

<!-- skip -->
```python
import muscadet.declare
from pyraichu.muscadet.engine import build_model

declaration = muscadet.declare.system_spec(system)
model = build_model(declaration, ["doors_unsecured"])
```

Anything the declaration carries and this reader cannot build raises a
typed `pyraichu.muscadet.declare.ComponentSpecError` or `SystemSpecError` naming
the component and the key: a model that loads is a model whose semantics
are covered.

## Matching the study's measures

Platform studies that declare `targets` are **first-occurrence
campaigns**: each trajectory stops at the feared event, and the recorded
indicators latch from the hit to the horizon. To reproduce those numbers,
run the Monte-Carlo with `stop_at_targets=True`: see
[Sequence analysis](sequence-analysis.md#first-occurrence-indicators) for
the two semantics. Each study indicator carries its own `measure`, and the
declaration carries it too:

- `nb-occurrences` → `IndicatorEstimate.nb_occurrences_mean` / `_std`
  (with targets: the probability the event occurred by each instant);
- `sojourn-time` → `IndicatorEstimate.sojourn_mean` / `_std`
  (with targets: mean time elapsed since the first occurrence);
- `had_value` → `IndicatorEstimate.reached_mean` / `_std`: the probability
  the indicator has been active at least once by each instant. Per
  trajectory it stays at 1 after the indicator falls back, as the
  reference engine's `realized` computation does.

The `min` and `max` statistics of any measure read the matching
`*_extremes` field (`extremes`, `sojourn_extremes`,
`nb_occurrences_extremes`, `reached_extremes`): the smallest and the
largest value that measure took across the replicas at each instant.

## Converting the outputs

RAICHU's results map line-for-line onto the platform's artefacts. The
minimal sequences, in the platform's sequence-artefact shape (`weight` is
the trajectory count; divide by `nb_runs` for the probability):

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
    separators (`occ__cc_12` for RAICHU's `occ__cc_1_2`). A failure-mode
    component is named after its factorized target (`PLC_X__…__fail`)
    while the study names the mode alone: match by suffix. Normalise both
    before comparing sequence sets.
