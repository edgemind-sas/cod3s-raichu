# Importing COD3S-platform studies

Models authored on a COD3S platform instance are persisted as **two
artefacts**: a *model export* (the topology — component instances of KB
templates, connections, per-instance overrides) and a *study* (the
dynamics — failure modes, feared events, indicators, Monte-Carlo
parameters). `pyraichu.importers.cod3s_platform` fuses both into one
runnable RAICHU model:

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
