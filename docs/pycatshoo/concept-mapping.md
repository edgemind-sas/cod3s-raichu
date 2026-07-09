# Concept mapping (for PyCATSHOO users)

!!! note "This appendix is for readers already familiar with PyCATSHOO"
    RAICHU is a stand-alone engine; the rest of this documentation needs
    no PyCATSHOO background. If you *are* coming from PyCATSHOO, this page
    maps its concepts to RAICHU's, and the [migration
    guide](migration-guide.md) walks two models across step by step.

RAICHU and PyCATSHOO implement the same underlying formalism — a
piecewise-deterministic Markov process realised as communicating hybrid
automata (Desgeorges et al. 2021). RAICHU is **iso-functional, not
iso-API**: the same modelling power, expressed as data, with some
deliberate departures.

## Model structure

| PyCATSHOO | RAICHU | note |
|---|---|---|
| `CComponent` subclass | a `component` object (JSON/dict) | pure data, validated at build time |
| `addVariable(name, type, init)` | `attributes: [{name, kind, init}]` | `kind` ∈ `bool`, `int`, `float` |
| `addReference` + message boxes | **in/out ports** (+ interfaces to group them) | ports are the native connection notion |
| `mb.addExport` / `addImport` + `connect` | `ports: [{name, dir, var}]` + `connections: [{from, to}]` | an out-port exports one attribute; an in-port aggregates its sources |
| `sumValue` / `orValue` / `andValue` on a reference | a `port_agg` expression: `sum`, `any`, `all`, `count`, `mean`, `median` | aggregation is part of the expression tree |
| `addAutomaton` / `addState` / `setInitState` | `automata: [{name, states, init, transitions}]` | state names are scoped **per automaton** |
| sensitive methods (`addSensitiveMethod`) | **sensitive functions**: declarative `effects` | triggers are *derived* from what each expression reads — no manual wiring, no Python callback in the hot loop |
| PDMP manager (`addEquationMethod`, `addODEVariable`) | `equations: [{target, kind: ode\|explicit, expr}]` | no manager object; a model has continuous semantics wherever it declares equations |

## Transitions and distributions

| PyCATSHOO | RAICHU |
|---|---|
| `setDistLaw(defer, t)` | `"distrib": "delay", "time": t` |
| `setDistLaw(inst, p)` | `"distrib": "inst", "probs": [...]` |
| `setDistLaw(expo, rate)` | `"distrib": "exp", "rate": λ` |
| `expo` + modifiable rate (discrete / continuous) | `"distrib": "exp", "rate_expr": …` — re-evaluated on change; on continuous attributes the cumulative hazard is integrated exactly |
| `setCondition(fun)` + watched transition | `"distrib": "watched"` + a `guard` with ordering comparisons (the boundary is root-found) |
| `setCondition(fun)` on a timed transition | a `"guard"` on the transition |
| `setInterruptible(True)` | `"on_interruption": "reset"` (**default**) |
| default (not interruptible) | `"on_interruption": "continue"` |
| *(no equivalent)* | `"on_interruption": "resume"` — a RAICHU extension |
| *(exponential / delay / instant)* | plus `weibull`, `lognormal`, `gamma`, `uniform`, `empirical`, each validated against its closed form |

## Simulation and results

| PyCATSHOO | RAICHU | note |
|---|---|---|
| one system per process (singleton) | any number of engines per process | deliberate departure |
| `simulate({nb_runs, seed, schedule})` | `monte_carlo(model, nb_runs, t_max, seed, samples)` | per-replica RNG substreams; estimates byte-identical for 1 or N threads |
| interactive simulation | `simulate(...)` + the causal journal | single trajectories are deterministic and replayable |
| indicators as `"comp.attr"` strings | typed indicator objects (`target: attribute\|state`) | means, std, quantiles (value + sojourn) |
| trace levels | the structured [causal journal](../guides/causal-journal.md) | queryable: `why_not_fired` / `who_changed` / `cascade_after` |
| silent hang on instantaneous loops | typed errors (fixpoint-iteration cap, Zeno guard) | fail loudly, not silently |
| order-dependent simultaneous effects (modeller's job) | optional non-confluence probe (`confluence_check`) | diagnoses order-dependence instead of hiding it |
| `setDtCond` (event-location step) | explicit integrator tolerances (`rtol`, `tol_event`, …) | recorded in the run's provenance |

## Deliberate departures

RAICHU is not a re-implementation of PyCATSHOO's API. The main
intentional differences:

- **Models are data**, validated at build time with precise typed errors
  instead of runtime crashes.
- **No process singleton** — build and run as many models as you like.
- **Reproducibility by construction** — explicit seeds, substreams,
  byte-identical parallel reduction (see
  [Reproducibility](../guides/reproducibility.md)).
- **Diagnostics** for non-confluence and instantaneous loops that
  PyCATSHOO leaves to the modeller.

The [migration guide](migration-guide.md) puts this mapping to work on
two concrete models.
