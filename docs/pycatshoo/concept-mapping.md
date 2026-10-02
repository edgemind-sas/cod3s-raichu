# Concept mapping (for PyCATSHOO users)

!!! note "This appendix is for readers already familiar with PyCATSHOO"
    RAICHU is a stand-alone engine; the rest of this documentation needs
    no PyCATSHOO background. If you *are* coming from PyCATSHOO, this page
    maps its concepts to RAICHU's, and the [migration
    guide](migration-guide.md) walks two models across step by step.

RAICHU and PyCATSHOO implement the same underlying formalism: a
piecewise-deterministic Markov process realised as communicating hybrid
automata (Desgeorges et al. 2021). RAICHU is **iso-functional, not
iso-API**: the same modelling power, expressed as data, with some
deliberate departures.

## Model structure

| PyCATSHOO | RAICHU | note |
|---|---|---|
| `CComponent` subclass | a `component` object (JSON/dict) | pure data, validated at build time |
| `addVariable(name, type, init)` | `attributes: [{name, kind, init}]` | `kind` ∈ `bool`, `int`, `float` (64-bit). No `string` or `complex` kind, no single-precision `float` distinct from `double` |
| delayed, reinitialised and uncertain variables (`addDelayedVariable`, `setReinitialized`, `setUncertainty`) | *(no core equivalent)* | the muscadet layer folds cod3s' reinitialised variables into a single-writer assignment; the core has none of the three |
| `addReference` + message boxes | **in/out ports** (+ interfaces to group them) | ports are the native connection notion; an interface is a validated group of ports, and a connection joins two ports |
| `mb.addExport` / `addImport` + `connect` | `ports: [{name, dir, attr}]` + `connections: [{from, to}]` | an out-port exports one attribute; an in-port aggregates its sources. Connections are fixed when the model is built: no `connectTo` / `disconnectFrom` at run time |
| `sumValue` / `orValue` / `andValue` on a reference | a `port_agg` expression: `sum`, `any`, `all`, `count`, `mean`, `median` | aggregation is part of the expression tree. No `productValue`, no indexed `value(i)` and no default value for an unconnected reference |
| `takeResource` / `restoreResource` | [`allocations`](../reference/model-schema.md#allocation) | a declared distribution operator rather than calls made from a method |
| a method reading another component's variable | an `attr` expression naming `{component, attribute}` | an expression may read, and an effect may write, any component's attribute: ports carry the connection topology, but the model does not force every coupling through them |
| `addAutomaton` / `addState` / `setInitState` | `automata: [{name, states, init, transitions}]` | state names are scoped **per automaton** |
| sensitive methods (`addSensitiveMethod`) | **sensitive functions**: declarative `effects` | triggers are *derived* from what each expression reads, no manual wiring, no Python callback in the hot loop. No method bound to entering or leaving one state: a transition's `effects` cover that case |
| `addStartMethod` | the initialisation step | every sensitive function runs to its fixpoint at `t = 0`, in declaration order; there is no imperative start method |
| steps (`IStep`), check, pre, in and post methods | *(no equivalent)* | the order of discrete evaluation is the fixpoint's, not a user-defined schedule |
| PDMP manager (`addEquationMethod`, `addODEVariable`) | `equations: [{target, kind: ode\|explicit, expr}]` | no manager object and a single continuous system per model; a model has continuous semantics wherever it declares equations |
| equation-method order (alphabetical unless set) | declaration order, or [`evaluation_order`](../reference/model-schema.md#evaluation-order) | a model relying on PyCATSHOO's alphabetical default must state the order explicitly |
| algebraic variables, Jacobian, stiff solvers, sign constraints (`setVariableConstraint`) | linear algebraic cycles only, solved as blocks; Dormand-Prince or fixed-step Euler | a nonlinear algebraic cycle is refused at build time; no stiff integrator |
| `ISLEManager` (linear systems) | [linear algebraic cycles](../reference/model-schema.md#linear-algebraic-cycles), detected automatically | no manager to declare |
| `IMILPManager` | [`programs`](../reference/model-schema.md#program) (HiGHS) | |

## Transitions and distributions

| PyCATSHOO | RAICHU |
|---|---|
| `setDistLaw(defer, t)` | `"distrib": "delay", "time": t` |
| `setDistLaw(inst, p)` | `"distrib": "inst", "probs": [...]`: N−1 **constant** probabilities for N targets, where PyCATSHOO also accepts N values and variable parameters |
| `setDistLaw(expo, rate)` | `"distrib": "exp", "rate": λ` |
| `expo` + modifiable rate (discrete / continuous) | `"distrib": "exp", "rate_expr": …`: re-evaluated on change; on continuous attributes the cumulative hazard is integrated exactly. See [Modifiable parameters](#modifiable-parameters) |
| `setCondition(fun)` + watched transition | `"distrib": "watched"` + a `guard` containing at least one ordering comparison, possibly composed with and/or/not (the boundary is root-found). One target state; no real-valued boundary-checker method |
| `setCondition(fun)` on a timed transition | a `"guard"` on the transition |
| `setInterruptible(True)` | `"on_interruption": "reset"` (**default**) |
| default (not interruptible) | `"on_interruption": "continue"` |
| *(no equivalent)* | `"on_interruption": "resume"`: a RAICHU extension |
| `weib` (scale, shape and a time shift) | `"distrib": "weibull"` with `shape` and `scale`: no time shift |
| `cstd` (firing at a fixed date) | *(no equivalent)*: a `delay` counts from the source state's entry |
| user-defined laws | `"distrib": "empirical"`, a tabulated CDF |
| *(no equivalent)* | `lognormal`, `gamma`, `uniform`, each validated against its closed form |

### Modifiable parameters

PyCATSHOO lets the modeller choose, per transition, how a parameter that
changes after the date is drawn is taken into account
(`TModificationMode`): `not_modifiable`, the **default**, keeps the date
drawn with the value the parameter had at that moment;
`discrete_modification` and `continuous_modification` redraw or integrate.
The parameter may be a variable on every law.

RAICHU makes no such choice. A `rate_expr` is always followed: the engine
integrates the cumulative hazard, piecewise-constant when λ reads only
discretely-updated state and alongside the continuous state when it reads
an integrated attribute. A model translated from a non-modifiable
PyCATSHOO transition therefore gives different numbers on the two engines
as soon as the rate changes while the transition is pending.

Only the exponential law has a state-dependent parameter. `delay.time`,
the Weibull, lognormal, gamma and uniform parameters and `inst.probs` are
constants of the model.

## Simulation and results

| PyCATSHOO | RAICHU | note |
|---|---|---|
| one system per process (singleton) | any number of engines per process | deliberate departure |
| `simulate({nb_runs, seed, schedule})` | `monte_carlo(model, nb_runs, t_max, seed, samples)` | per-replica RNG substreams; estimates byte-identical for 1 or N threads |
| interactive simulation | [`interactive(model, operator_control=True)`](../guides/interactive-simulation.md#operator-controlled-continuous-advancement): `advance_operator_to` / `fireable` / `fire(name, to=)` / `step` / `set_date` / `snapshot` / `restore` | operator mode draws no stochastic date or unsolicited branch; a snapshot is an exact undo |
| indicators as `"comp.attr"` strings | typed indicator objects (`target: attribute\|state\|predicate`) | mean, std, quantiles, extremes and confidence intervals on the value, sojourn, occurrences and first reach. See [Indicator measures](#indicator-measures) |
| `addTarget` (any boolean method) | `targets` naming an automaton state | a condition becomes a target through a transition into a dedicated state; a `predicate` indicator observes it without ending the sequence |
| trace levels | the structured [causal journal](../guides/causal-journal.md) | queryable: `why_not_fired` / `who_changed` / `cascade_after` |
| `setMaxTransitionsDt0`, a cap on transitions fired at one date | typed errors (fixpoint-iteration cap, Zeno guard) | the error names the loop; no cap to tune |
| order-dependent simultaneous effects (modeller's job) | optional non-confluence probe (`confluence_check`) | diagnoses order-dependence instead of hiding it |
| `setDtCond` (event-location step) | explicit integrator tolerances (`rtol`, `tol_event`, …) | recorded in the run's provenance |
| importance sampling (`IImportanceSampling`, user manual V1.3.7.2, section 5.5.3) | [`quantify(method="cross_entropy")`](../guides/quantification.md#rare-feared-events-cross-entropy) | the biasing law is learnt by cross-entropy rather than supplied; numerical parity of the algorithms is not claimed |
| sequential Monte-Carlo (user manual V1.3.7.2, sections 5.5.4 and 9.3.38.4) | [`quantify(method="splitting")`](../guides/quantification.md#rare-feared-events-splitting), with a numeric attribute or minimal-cut-set importance | generalized adaptive multilevel splitting selects at strict score-level crossings; independent batches provide the interval. This departs from the reference engine's fixed time intervals; numerical parity of the algorithms is not claimed. |
| FMU co-simulation import (developer manual, section 4.1.11.2) | `fmu_units` with attribute bindings and an explicit import grant | outputs are held between scheduled communication points; crossing dates on those outputs have the declared step resolution |
| Standalone FMU construction (developer manual V1.3.7.2, sections 4.3, 4.3.1 and 4.3.2) | [`pyraichu.fmi.export`](../guides/fmi-co-simulation.md#export-a-raichu-model) | FMI 3 co-simulation packages the model and native runtime; outputs are held at communication points. The export guide states the platform and model boundaries. |

### Indicator measures

An indicator's computation mode (`TComputationType`) maps to one of
RAICHU's four measures, which are all computed on every indicator. The
equivalences below were **measured** on PyCATSHOO 1.3.8.0 (2026-10-02),
on a float stepping through zero, positive and negative levels, because
the user manual V1.3.7.2 describes `res_time`, `nb_visits` and
`realized` in terms of a positive or non-zero value (sections 5.4.1 and
9.3.55) and the engine does not compute the first that way.

| PyCATSHOO | RAICHU | equivalence |
|---|---|---|
| `simple` | value (`mean`, `std`, …) | identical |
| `res_time` | sojourn (`sojourn_mean`, …) | identical: both integrate the value over time, sign included. A value held at 2 for 10 time units gives 20 on both |
| `nb_visits` | occurrences (`nb_occurrences_mean`, …) | **identical only on a value that starts at 0 and never goes negative.** PyCATSHOO counts the departures from exactly 0 to any non-zero value and does not count a non-zero initial value; RAICHU counts every rise from ≤ 0 to > 0, an active initial value included. A state active from `t = 0` and never re-entered counts 0 on PyCATSHOO and 1 here |
| `realized` | reached (`reached_mean`, …) | identical on a value that never goes negative: PyCATSHOO answers 1 once the value has been non-zero, RAICHU once it has been positive |

PyCATSHOO's `distribution` (bounded histogram) and `pctQuantile` restitutions,
an indicator computed by an arbitrary method and its restriction by
`setCondFct` have no equivalent: an indicator observes an attribute, a
state or a comparison.

### Result files

The reference engine's result files are described in its user manual
V1.3.7.2, sections 7, 7.4-7.6 and 8. RAICHU keeps its own documented
formats; the COD3S platform translates either engine's results into the
same platform artefacts ([platform import guide](../guides/platform-import.md#converting-the-outputs)).

| Reference engine file | RAICHU counterpart | Reader and boundary |
|---|---|---|
| Complete result file, including monitored data (sections 7.4 and 8) | [`raichu.quantification`](../reference/quantification-format.md) for a quantified study, plus the platform's `indicators.csv` and run metadata | The platform reads its own artefacts. RAICHU does not produce a file for the reference engine's result loader. |
| Grouped or raw `sequences.xml` (section 7.5) | [`raichu.sequences`](../reference/sequence-format.md) raw corpus; the platform also writes `sequences_minimal.json` and `sequences_all.json` | The platform's run viewer and exports read the shared JSON envelope. A reload into the reference engine has no counterpart, by decision. |
| Fault-tree export (section 7.6) | [OpenPSA](../guides/fault-tree.md) | Both engines can exchange this open standard. |
| `pyc_param.xml`, the reference run's parameter record (section 7) | The sealed model document's content hash in [`raichu.quantification`](../reference/quantification-format.md) | Provenance is recorded in RAICHU's own result format; no parameter-file clone is written. |

The manual's NetCDF manager (section 9.3.31) reads model inputs. It is not
a result-file export and is outside this mapping.

## Sequence-tree exploration

PyCATSHOO's sequence-tree explorer (user manual V1.3.7.2, sections 5.5.5,
8.3.33 and 9.3.39) maps to `pyraichu.explore`; see the
[sequence-tree exploration guide](../guides/sequence-tree-exploration.md).

| PyCATSHOO | RAICHU | note |
|---|---|---|
| `setUseSeqTreeExplorer(True)`, `seqTreeExplorer()`, `exploreTree()` | `pyraichu.explore(model, target, horizon, ...)` | a driver beside Monte-Carlo, on the same model; no system-wide switch |
| Harrison algorithm (`setAlgoHarrison(True)`) | `algorithm="exact"` (the default) | same domain (instantaneous branchings, exponential laws); probabilities computed by uniformization in nonnegative arithmetic, with a per-sequence error bound, instead of alternating closed-form sums |
| `setHarrisonParameters(e1, e2, e3, precision)` | `rel_precision`, `max_terms` | one relative precision; a sequence that misses it is flagged `imprecise`, never reported silently |
| `setMinProbability` (MIN_P) | `min_probability` | prunes a prefix whose probability of being completed by the horizon falls below the threshold |
| `setMaxNbFailures` (MAX_FL), with transitions typed as faults | `max_failures`, with `"kind": "failure"` on the transition | the muscadet plugin declares the kind; a model declaring none is refused rather than counting nothing |
| `setMaxNbBranches` (MAX_BR) | `max_branches` | counts expanded nodes; split among the root's children so the result does not depend on the thread count |
| *(no equivalent)* | `max_length` | fired transitions per sequence |
| `setTMax` | `horizon` | |
| `sequences()`, every leaf, `MaxTime` ones included | `result.sequences`, the target sequences only | the rest is summarised by the bounds |
| `curProbability()` (explored mass) | `result.lower`, `result.upper`, `result.cutoff_tallies` | a lower and an upper bound on the target probability, and the mass each cut-off discarded |
| sampling algorithm (non-exponential laws) | `algorithm="discretised"`, a different method | RAICHU's own algorithm, not a reproduction of the sampling one: the next-event distribution is cut into equal-mass cells, and the result states its discretisation level and an error estimate by refinement. No numerical parity with PyCATSHOO is sought here; it is validated against closed forms, exact exploration and Monte-Carlo simulation |

On Markov models the two explorers return the same target sequences, and
their probabilities agree within 1e-9 relative once PyCATSHOO's
Harrison parameters are tightened below their 1e-3 defaults (section
9.3.39.5.1). PyCATSHOO also returns a zero-probability sequence for a
branch of rate 0 (a dormant spare); RAICHU does not branch on it.

## Deliberate departures

RAICHU is not a re-implementation of PyCATSHOO's API. The main
intentional differences:

- **Models are data**, validated at build time with precise typed errors
  instead of runtime crashes.
- **No process singleton**: build and run as many models as you like.
- **Reproducibility by construction**: explicit seeds, substreams,
  byte-identical parallel reduction (see
  [Reproducibility](../guides/reproducibility.md)).
- **Diagnostics** for non-confluence and instantaneous loops that
  PyCATSHOO leaves to the modeller.

## Not covered

Capabilities of PyCATSHOO with no RAICHU counterpart, beyond those named
in the tables above:

- strategy optimisation over a Markov decision process (`IStrategy`, PEI
  and MRAS algorithms, user manual V1.3.7.2, section 6);
- inhibiting a target or a transition (`inhibateTarget`,
  `inhibateTrans`);
- several named PDMP managers in one system, and begin or end methods
  around an integration.

The [migration guide](migration-guide.md) puts this mapping to work on
two concrete models.
