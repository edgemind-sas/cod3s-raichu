# Sequence-tree exploration: dominant sequences without sampling

[Sequence analysis](sequence-analysis.md) finds the sequences leading to
a feared event by drawing Monte-Carlo trajectories and reducing what they
recorded. A sequence no trajectory walked is absent, and the probability
of a rare one is known only to the precision the campaign size allows.
**Sequence-tree exploration** answers the same question by enumeration:
starting from the initial state, it follows every possible next
transition, computes the probability of each path, and stops a branch
when it reaches the feared event or a declared cut-off.
The result lists the retained sequences with their probabilities at the
horizon, and states a **lower and an upper bound** on the probability of
the feared event, so what the cut-offs left out is measured, not guessed.

The two modes run on the same model declaration. Two exploration
algorithms are provided. The **exact** one (the default) covers the
Markov family (see [the exact domain](#the-exact-domain)) and computes
every probability in closed form. The **discretised** one (see
[The discretised algorithm](#the-discretised-algorithm)) covers every
law the engine carries and continuous evolution, at the price of a
discretisation error that it estimates. Exploration is the mode to use
when the sequences of interest are rare enough that a campaign would need
many replicas to see them, or when a result must come with bounds rather
than a confidence interval. Monte-Carlo remains the mode for
availability measures on a free-cycling system, for repairable systems
over horizons long enough that the tree grows faster than the cut-offs
can bound it, and for models whose discretised tree is too large (see
[Limits](#limits)).

Readers coming from PyCATSHOO's sequence-tree explorer will find how its
settings map to these in the
[concept mapping](../pycatshoo/concept-mapping.md#sequence-tree-exploration).

## An example: a repairable pair

Two components `A` and `B` fail at rates 0.1 and 0.03 per unit of time
and are each repaired at rate 0.5. The feared event is both down at
once. This is the model of the [sequence analysis](sequence-analysis.md)
guide without the common-cause mode.

```python
import pyraichu

def objfm(name, target, rate):
    return {"type": "ObjFM", "name": name, "targets": [target],
            "failure": [{"law": "exp", "rate": rate}],
            "repair": [{"law": "exp", "rate": 0.5}],
            "failure_effects": {"flow": False}}

model = pyraichu.load_model({
    "name": "repairable_pair",
    "plugins": {"muscadet": {"objects": [
        objfm("fm_A", "A", 0.1),
        objfm("fm_B", "B", 0.03),
        {"type": "ObjEvent", "name": "system_down", "target": True,
         "cond": [[{"obj": "A", "attr": "flow", "ope": "==", "value": False},
                   {"obj": "B", "attr": "flow", "ope": "==", "value": False}]]},
    ]}},
    "components": [
        {"name": n, "attributes": [{"name": "flow", "kind": "bool",
                                    "init": {"kind": "bool", "value": True}}]}
        for n in ("A", "B")
    ],
})

assert pyraichu.exploration_domain(model) == []

result = pyraichu.explore(model, "system_down", horizon=10.0,
                          min_probability=1e-6, max_length=9)

print(f"lower={result.lower:.6e} upper={result.upper:.6e} "
      f"gap={result.relative_gap:.2e} inconclusive={result.inconclusive}")
print(f"{len(result.sequences)} sequences, {result.expanded_nodes} nodes expanded")
for seq in result.sequences[:4]:
    chain = " -> ".join(f"{e['obj']}.{e['attr']}" for e in seq.events)
    print(f"{seq.probability:.4e}  {chain}")
for name, tally in result.cutoff_tallies.items():
    print(f"{name:16s} pruned={tally['pruned_nodes']:4d} mass={tally['mass']:.3e}")
```

```text
lower=7.174868e-02 upper=7.194860e-02 gap=2.78e-03 inconclusive=False
30 sequences, 117 nodes expanded
2.7889e-02  fm_A.occ -> fm_B.occ -> system_down.occ
2.5107e-02  fm_B.occ -> fm_A.occ -> system_down.occ
6.5033e-03  fm_A.occ -> fm_A.rep -> fm_A.occ -> fm_B.occ -> system_down.occ
5.9724e-03  fm_A.occ -> fm_A.rep -> fm_B.occ -> fm_A.occ -> system_down.occ
min_probability  pruned=  30 mass=1.483e-05
max_length       pruned=  28 mass=1.851e-04
max_failures     pruned=   0 mass=0.000e+00
max_branches     pruned=   0 mass=0.000e+00
```

The probability that both components are down at the same time before
`t = 10` lies in `[0.071749, 0.071949]`. As a check, a first-occurrence
Monte-Carlo campaign of 100 000 replicas on the same model
(`monte_carlo(..., stop_at_targets=True)`, seed 42) estimates 0.07154
with a 95 % Wilson interval of `[0.06996, 0.07315]`, which contains both
bounds. The exploration gives the figure to three significant digits
and says which paths make it up.

The same call works without the plugin layer: `target` names any entry
of the model's `targets` (see the
[model schema](../reference/model-schema.md)), and `horizon` is in the
model's time unit.

## Reading the result

`pyraichu.explore` returns an `Exploration`.

**Sequences.** `result.sequences` lists the retained sequences by
decreasing probability, ties broken by their list of fired transitions. A sequence
is an ordered path from the initial state to the target **in which
nothing else happens**: the fourth line above is "A fails, is repaired,
fails again, then B fails", and it is a different sequence from the
first. Each `ExploredSequence` carries:

- `steps`: its fired transitions in firing order, as indices into
  `result.steps`, the table that holds each distinct step (a transition
  fired into one destination) once;
- `transitions`: those steps resolved, each `{transition, from, to}`,
  instantaneous ones included (computed on access from `steps`);
- `events`: the monitored-state entries among them, each
  `{obj, attr, cycle_group}`, in the vocabulary of the Monte-Carlo
  sequence corpus, without dates (computed on access as well);
- `probability`: the probability that a trajectory follows exactly this
  path and reaches the target by the horizon;
- `error_bound` and `imprecise`: the absolute numerical error bound on
  that probability, and whether it exceeds the declared relative
  precision (see [Numerical method](#numerical-method)).

**Bounds.** The retained sequences are disjoint events: a trajectory
follows one path of the jump chain, never two. Their probabilities
therefore add up, and `result.lower`, their sum, is a lower bound on the
probability of reaching the target by the horizon. `result.upper` adds
the mass every cut-off discarded. Everything the exploration did not
retain is either a path that never reaches the target (absorbing state,
or another target reached first, where a Monte-Carlo trajectory stops
too) or a path under a pruned node, whose mass is counted in the upper
bound. The true probability lies between the two, up to the numerical
error bounds of the individual sequences. For a discretised result, the
probabilities and both bounds are those of the discretised model, and the
distance to the true model is the discretisation error, which the result
estimates (see [Error estimate by refinement](#error-estimate-by-refinement)).

**Cut-off tallies.** `result.cutoff_tallies` gives, for each of the four
cut-offs, the number of nodes it pruned and the mass it added to the
upper bound. `upper - lower` is the sum of the four masses. When
several cut-offs would prune the same node, it is tallied once, under
the first of `min_probability`, `max_length`, `max_failures`; the branch
cap is checked separately. The tally says which setting to relax: in
the example, the length cut-off discarded twelve times more mass than
the probability threshold.

**Inconclusive flag.** `result.inconclusive` is `True` when the relative
gap `(upper - lower) / upper` (`result.relative_gap`) exceeds
`gap_tolerance`, 0.01 by default. The result is still returned in full:
the flag says the bounds are too far apart to quote a single figure, not
that the sequences are wrong. On the same pair at `t = 1000` with a
branch cap of 200 nodes, the exploration retains one sequence, the
bounds are `[0.0385, 1.0]`, and the flag is raised.

`result.cutoffs`, `result.gap_tolerance` and `result.precision` record
the settings the run used, `result.expanded_nodes` the work done, and
`result.imprecise_sequences` how many probabilities are flagged.

## The cut-offs

Each cut-off is optional and is passed as a keyword argument of
`explore`. The minimal-probability, length and failure cut-offs are
applied to a child node before it is fired; the branch cap is applied
when a node is about to be expanded. Every pruned mass goes to the upper
bound and to its cut-off's tally.

| Setting | Prunes a node when | Mass added to the upper bound |
|---|---|---|
| `min_probability` | the probability that its prefix is completed by the horizon falls below the threshold, in `(0, 1]` | that probability |
| `max_length` | its sequence would exceed this many fired transitions | the prefix probability |
| `max_failures` | its sequence would exceed this many fired failures | the prefix probability |
| `max_branches` | this many nodes have already been expanded | the prefix probability |

**Minimal probability.** The quantity compared is not the path
probability in the jump chain but the probability that the prefix is
followed **and completed by the horizon**. That quantity bounds the
probability of every sequence extending the prefix, since each extension
adds a sojourn, so adding it to the upper bound is sound. Raising the
threshold retains fewer sequences and widens the gap: on the example, at
`min_probability=1e-4` the exploration retains 12 sequences instead of
30, the probability cut-off discards 1.37e-3, and the relative gap of
1.9 % raises the inconclusive flag.

**Maximal length** counts every fired transition, instantaneous ones and
the final transition into the target included. In the example, the
direct sequences have length 3.

**Maximal failures** counts fired transitions whose declared `kind` is
`failure` (see [Declared kind](../reference/model-schema.md#declared-kind)),
and only when the transition enters its **first** declared target. An
on-demand draw emitted as one instantaneous transition with targets
`["occ", "parked"]` counts as a failure when it enters `occ`, not when it
enters `parked`. The muscadet plugin declares these kinds on the edges it
emits; a hand-written model declares them itself. A model that declares
no `kind` at all is refused when `max_failures` is set, rather than
counting nothing and never pruning.

**Branch cap.** `max_branches` counts expanded nodes, at least 1. It is
divided among the root's children before the exploration starts, in
equal shares with the remainder going to the first children in order,
and each subtree enforces its own share. The explored set therefore does
not depend on the thread count, but a subtree cannot borrow the unused
share of another: a cap is a budget per root branch, not a global
breadth-first limit.

**No cut-off.** An exploration without cut-offs explores until every
branch ends: target reached, absorbing state, or a prefix of probability
exactly zero. On a non-repairable model this terminates. On a repairable
one each repair reopens the tree, and termination is not guaranteed in
practice: declare at least `min_probability` or `max_length`.

## Determinism and parallelism

`threads` sets the worker count. The root's children are split across
workers, each with its own engine and its own share of the branch cap,
and the partial results are reduced in child order. Children are
visited in transition index order, then destination order. The
sequences, their ranking and every number are bit-identical whatever the
thread count. Nothing is drawn: an exploration has no seed.

## The exact domain

The exact algorithm, `algorithm="exact"` (the default), covers the
Markov family:

- instantaneous transitions (`"distrib": "inst"`) and zero delays, which
  fire with no sojourn and branch over their destinations with their
  declared probabilities;
- exponential transitions whose rate is constant between jumps: a
  constant `rate`, or a `rate_expr` over discrete state;
- no continuous evolution.

The domain is checked twice.

**Before the run**, the model is refused if it integrates an ODE, if a
watched transition sits in a state its automaton can reach from its
initial state (guards ignored), or if any guard, effect, rate,
sensitive function or explicit equation reads the simulation time. The
exploration never advances the clock, so none of these has a meaning in
it. `pyraichu.exploration_domain(model)` runs this check alone and
returns the list of violations, each `{kind, ..., message}` with `kind`
one of `ode`, `watched` or `reads_time`; an empty list means none was
found.

**During the run**, the first armed transition whose law is outside the
domain (a non-zero delay, Weibull, lognormal, gamma, uniform, empirical,
or an exponential whose rate varies continuously) stops the exploration
with a `SimulationError` naming the transition and the sequence that
armed it. Replacing `A`'s repair in the example by a fixed delay of 2
gives:

```text
transition `fm_A.fm.repair` (delay 2) is armed after the sequence [fm_A.fm.failure]: its law is outside the exact exploration domain (instantaneous, zero delay, or exponential with a rate constant between jumps)
```

`exploration_domain` returns an empty list on that model: a law is only
found when it becomes armed. Such a model is explored with
`algorithm="discretised"` instead. A law that is never armed along the
explored sequences does not block. The check applies to every node the
exploration reaches, so a cut-off avoids the refusal only if it prunes
the node before the transition is armed.

Within the domain, an armed exponential transition of rate 0 (a cold
spare, dormant) is not a branch, and a node with no instantaneous
transition and a total exit rate of 0 is an absorbing leaf that
contributes nothing.

## Numerical method

A sequence probability is the product of its branch probabilities in the
embedded jump chain (`rate / total exit rate` at each timed node, the
declared probability at each instantaneous one) and of the probability
that the sum of its sojourn times, `Exp(q_1) + ... + Exp(q_k)` with
`q_i` the total exit rates along the path, repeated rates included, ends
before the horizon. Closed forms exist for that second factor,
repeated rates included (Collet and Renault 1997; Harrison 1990 obtains
passage-time distributions by Laplace-transform inversion), but they are
sums of terms of alternating sign, which lose relative precision through
cancellation (Collet and Renault 1997). RAICHU computes it instead as the absorption probability of
an acyclic chain of phases by **uniformization**, where every term is
nonnegative, so a probability of 1e-25 keeps its relative precision. The
series is truncated with a rigorous bound on the discarded mass, and
when the number of terms would exceed `max_terms` (a stiff path, where
`max(q) * horizon` is large) the computation falls back to scaling and
squaring of the same generator, still in nonnegative arithmetic. Each
probability carries an error bound (truncation plus rounding); one whose
bound exceeds `rel_precision` times its value (default 1e-9) is flagged
`imprecise` and counted in `imprecise_sequences`, never reported as
precise. The treatment of truncated trees through a lower bound and a
neglected mass follows the practice established for GSI (Bon and
Bouissou 1992).

## The discretised algorithm

`algorithm="discretised"` lifts the restriction to the Markov family. It
covers every law the engine carries (delay, exponential with a constant
or state-dependent rate, Weibull, lognormal, gamma, uniform, empirical),
the interruption policies, and continuous evolution with watched
transitions. It is RAICHU's own method: it does not reproduce
PyCATSHOO's sampling algorithm (see the
[concept mapping](../pycatshoo/concept-mapping.md#sequence-tree-exploration)).
Its published analogue on piecewise-deterministic processes is the
quadrature-based dynamic event tree (Elhareef and Yim 2026).

**What is discretised.** Not each transition's own firing date, but the
**random next event**: which armed stochastic transition fires first,
and when. At an explored state where nothing fires at the current
instant, every armed stochastic transition `j` is a competitor with a
cumulative hazard `H_j` over a window that ends at the first of the next
deterministic event, the next watched crossing and the horizon:

- a law with a fixed shape contributes its cumulative hazard conditional
  on the age it has already accrued under its interruption policy;
- an exponential whose rate is piecewise constant contributes
  `rate x elapsed time`;
- one whose rate varies with continuous state contributes its hazard
  integrated along the deterministic flow.

With `H = sum of H_j`, the next event falls in the window with
probability `M = 1 - exp(-H(window end))`. That mass is cut into `K`
**cells of equal probability** `M / K`. Within a cell, the mass is split
among the competitors in proportion to their hazard increments, and
each (competitor, cell) pair of nonzero mass becomes a branch, fired at
the instant where the cell's mass reaches its middle. One further branch
carries the survival mass `exp(-H)` to the window end, where the
deterministic event or the watched crossing fires; at the horizon it is
dropped. Between branch points the engine runs deterministically, so the
explorer never re-implements a guard, an effect or a policy.
Instantaneous transitions, due delays and watched transitions whose
guard holds fire first, as in the exact algorithm.

Two numerical details keep the tree from growing on round-off. A window
no wider than the event-location tolerance (`1e-10` time units by
default), such as two delays whose dates differ by one ulp (`0.1 + 0.2`
against `0.3`), is treated as empty: its mass, of order
`rate x 1e-10`, goes to the survival branch, and a deterministic end
event fires in the same branch, so the tree is the one the two dates
would give if they coincided exactly. A branch fired at a cell's middle
instant that turns out to lie past a watched crossing (the crossing is
located to within the same tolerance) is fired just before the crossing
instead. A competitor whose window hazard evaluates to NaN or to a
negative value stops the exploration with an error naming the
transition.

Leaves that reach the target along the same ordered list of fired
transitions, through different cells, are one sequence: their
probabilities are summed before ranking. The probabilities, the lower
and the upper bound are those of the **discretised model**; the cut-offs
act on it as in the exact algorithm, except that the minimal-probability
cut-off compares a node's mass multiplied by `K` to the power of the
cell branchings on its path, an estimate of the whole event sequence's
mass rather than of one cell's fragment of it.

### Error estimate by refinement

With `refine=True` (the default), the exploration runs at level `K`
(`level`, default 8), then again at `2K`; the reported result is the
`2K` one. `result.discretisation` holds `{level, refinement}`: `level`
is `2K`, and `refinement` holds `base_level` (`K`), `base_lower`,
`base_upper`, `error_estimate` and `truncation_dominated`.
`result.error_estimate` is the estimate itself,
`max(|lower_2K - lower_K|, |upper_2K - upper_K|)`.

- It is an **estimate, not a bound**. When the level-`K` and level-`2K`
  values fall on either side of the true value, their difference
  understates the error: on a sequence of two Weibull laws where the
  second has shape 1.5, the estimate at `K = 8` understates the error of
  the reported level-16 value by a factor 5 (measured against the closed
  form). Comparing two levels on a model of interest, or a Monte-Carlo
  campaign, is the check when the figure matters.
- It is flagged `truncation_dominated` when the two runs' gaps
  `upper - lower` add up to half of the estimate or more: the difference
  between the runs may then reflect what the cut-offs truncated as much
  as the discretisation. Relax the cut-offs before reading the estimate.
- With `refine=False`, only level `K` runs, `refinement` is `None`, and
  `result.error_estimate` is `None`: **no estimate was made**, and the
  numbers carry no statement about the discretisation error.

### Cost

A timed node has up to `K x competitors + 1` children, so the tree grows
as about `(K x competitors)^depth` nodes, and the refined pass costs
about `2^depth` times the base one. `max_branches` defaults to 1 000 000
expanded nodes for each pass when it is not given; the branch cap then
bounds the run, and a result that hits it says so through its tally and
its gap. For long sequences, many concurrent competitors, or a
repairable system over a long horizon, Monte-Carlo is the cheaper tool.

### Example: a wear-out pair

Two non-repairable components wear out: `A` fails after a Weibull time
of shape 2 and scale 20, `B` after one of shape 1.5 and scale 30. The
feared event is both down by `t = 10`. The model is written in the core
schema, since the muscadet plugin carries no Weibull law.

```python
import pyraichu

def unit(name, shape, scale):
    return {"name": name, "automata": [{
        "name": "fail", "states": ["ok", "nok"], "init": "ok",
        "transitions": [{"name": "occ", "source": "ok", "targets": ["nok"],
                         "distrib": "weibull", "shape": shape, "scale": scale,
                         "monitored": True}]}]}

def down(name):
    return {"op": "state_active",
            "state": {"component": name, "automaton": "fail", "state": "nok"}}

model = pyraichu.load_model({
    "name": "wearout_pair",
    "components": [
        unit("A", 2.0, 20.0),
        unit("B", 1.5, 30.0),
        {"name": "sys", "automata": [{
            "name": "watch", "states": ["up", "down"], "init": "up",
            "transitions": [{"name": "down", "source": "up", "targets": ["down"],
                             "distrib": "inst", "probs": [],
                             "guard": {"op": "bool", "bool_op": "and",
                                       "args": [down("A"), down("B")]}}]}]},
    ],
    "targets": [{"name": "sys_down", "component": "sys",
                 "automaton": "watch", "state": "down"}],
})

result = pyraichu.explore(model, "sys_down", horizon=10.0,
                          algorithm="discretised")
d = result.discretisation
print(f"level={d['level']} (base {d['refinement']['base_level']}) "
      f"lower={result.lower:.6e} upper={result.upper:.6e} "
      f"estimate={result.error_estimate:.2e}")
print(f"{len(result.sequences)} sequences, {result.expanded_nodes} nodes expanded")
for seq in result.sequences:
    chain = " -> ".join(f"{e['obj']}.{e['attr']}" for e in seq.events)
    print(f"{seq.probability:.6e}  {chain}")
```

```text
level=16 (base 8) lower=3.873273e-02 upper=3.873273e-02 estimate=2.18e-05
2 sequences, 545 nodes expanded
2.201363e-02  B.nok -> A.nok
1.671909e-02  A.nok -> B.nok
```

The two components are independent, so the probability that both have
failed by `t = 10` has the closed form
`(1 - exp(-(10/20)^2)) x (1 - exp(-(10/30)^1.5)) = 3.872413e-02`. The
reported value differs from it by 8.6e-06, within the estimate of
2.18e-05. The two sequences also have closed forms, 2.201636e-02 for
`B` then `A` and 1.670777e-02 for `A` then `B` (by quadrature of the
first failure's density). The exact algorithm refuses the model before
exploring anything:

```text
transition `A.fail.occ` (weibull shape 2 scale 20) is armed after the sequence []: its law is outside the exact exploration domain (instantaneous, zero delay, or exponential with a rate constant between jumps)
```

Doubling the level shrinks the estimate by about a factor 3.5 on this
model while the node count grows about fourfold:

| `level` (reported `2K`) | lower | estimate | nodes expanded |
|---|---|---|---|
| 2 (4) | 3.883131e-02 | 2.73e-04 | 41 |
| 4 (8) | 3.875449e-02 | 7.68e-05 | 145 |
| 8 (16) | 3.873273e-02 | 2.18e-05 | 545 |
| 16 (32) | 3.872655e-02 | 6.18e-06 | 2 113 |

For Rust users: the discretised algorithm runs the engine in a
deferred-draw mode, selected by a new public field of `EngineConfig`,
`stochastic_dates` (default: dates drawn when a transition is armed, the
Monte-Carlo semantics, unchanged). Adding a public field is why this
release is 0.46.0 rather than a patch: a struct literal written without
`..Default::default()` must add it.

## Relation to minimal cut sequences

An explored sequence and a minimal cut sequence are different objects.
The exploration returns every distinct path, repairs and all; a minimal
cut sequence is a reduction over many paths. `result.minimal_sequences()`
applies the engine's own reduction, the one of
[sequence analysis](sequence-analysis.md#from-trajectories-to-minimal-cut-sequences):
group, cancel failure/repair cycles, absorb super-sequences.

```python
for cut in result.minimal_sequences():
    chain = " -> ".join(f"{e['obj']}.{e['attr']}" for e in cut["events"])
    print(f"{cut['weight']:.4e}  {chain}")
```

```text
3.7654e-02  fm_A.occ -> fm_B.occ -> system_down.occ
3.4095e-02  fm_B.occ -> fm_A.occ -> system_down.occ
```

The 30 retained sequences collapse into the two orderings. Each `weight`
is a **probability** here, not a replica count: the retained sequences
are disjoint, so the probabilities that collapse into one minimal
sequence add up, and the weights total `result.lower`.

An exploration result is **not** a valid input to any post-processing
that replays dated trajectories (reading the state of the model at
chosen instants, for example). The exact exploration chooses the order
of the jumps and never their dates, and a discretised sequence merges
cells fired at different instants: in both cases the events carry no
time, and the dates in the reduced sequences are 0.

## The result format

An `Exploration` is saved and read back in its own open format,
`raichu.exploration`, as a single JSON document. An exact result is
written at version 1 and a discretised one at version 2, the version
that introduced the discretised algorithm:

```python
result.to_json("exploration.json")          # also returns the text
again = pyraichu.read_exploration("exploration.json")
assert again == result
```

The top-level fields are `format`, `version`, `engine_version`, `model`,
`algorithm`, `target`, `horizon`, `cutoffs`, `gap_tolerance`,
`precision`, `steps`, `sequences`, `lower`, `upper`, `cutoff_tallies`,
`inconclusive`, `expanded_nodes` and `imprecise_sequences`, with the
meanings given above. A discretised result adds `discretisation`,
`{level, refinement}` (see
[Error estimate by refinement](#error-estimate-by-refinement)); an exact
result omits the field, so its document is unchanged.

`steps` is the step table: one entry per distinct step occurring in a
retained sequence, `{transition, from, to, event}`, where `event` is the
monitored-state entry `{obj, attr, cycle_group}` the step produces, or
`null` when the transition is not monitored. The table is ordered by the
transition's position in the compiled model, then by destination, so it
does not depend on the thread count. Each entry of `sequences` holds
`steps` (indices into the table, in firing order), `end_cause`,
`probability`, `error_bound` and `imprecise`:

```json
"steps": [
  {"transition": "A.fail.occ", "from": "ok", "to": "nok",
   "event": {"obj": "A", "attr": "nok", "cycle_group": null}},
  {"transition": "sys.watch.down", "from": "up", "to": "down", "event": null}
],
"sequences": [
  {"steps": [0, 1], "end_cause": "sys_down", "probability": 0.39,
   "error_bound": 1e-17, "imprecise": false}
]
```

A sequence's transitions are its step indices resolved through the
table, and its events are the non-null `event` of those steps. A reader
refuses another `format`, a `version` above the one it knows (a reader
of RAICHU 0.45.0 knows version 1 only, so it refuses a discretised
document by its version), a document whose `algorithm` its `version`
predates (a `discretised` result declaring version 1, which no writer
produces), and a sequence referring to an index outside the table; a
later version may add fields, which an earlier reader ignores. The
format is distinct from the [raw sequence corpus](../reference/sequence-format.md)
of a Monte-Carlo campaign, whose sequences carry replica counts and
dates.

## Limits

- **The exact algorithm covers the Markov family only.** A model with an
  ODE, a reachable watched transition, time-dependent expressions, or a
  law other than instantaneous or constant-rate exponential that becomes
  armed, is refused by it; `algorithm="discretised"` explores such a
  model.
- **The discretised algorithm reports an estimate, not a bound, of its
  own error**, and its tree grows as `(K x competitors)^depth`: for long
  sequences or many concurrent competitors, Monte-Carlo is the cheaper
  tool (see [Cost](#cost)).
- **Repairable systems over long horizons need cut-offs.** Each repair
  reopens the tree, and the number of sequences grows quickly with the
  horizon. Declare `min_probability` or `max_length`, read the tallies,
  and treat an inconclusive result as a prompt to relax the cut-off that
  discarded the most mass, or to fall back on Monte-Carlo.
- **Termination without any cut-off is not guaranteed on a repairable
  model.**
- **No dates.** Explored sequences are ordered, not dated, so they do not
  feed any date-based post-processing.
- **Memory grows with depth.** The result is proportional to the sum of
  the retained sequence lengths, stored as small integers (the names
  appear once, in the step table): on a pair whose repairs keep the
  target out of reach for a long time, a horizon of 1 000 retains about
  1 150 sequences totalling some 670 000 fired transitions, about
  1.5 MB of JSON (5 300 sequences and 29 MB at a horizon of 5 000). The explorer also keeps, for every level of the
  path it is on, the phase-type accumulator of that prefix, whose size
  grows with the depth and with the number of uniformization terms
  (about `max_rate x horizon`): memory while exploring grows as
  `depth x (depth + max_rate x horizon)` floating-point numbers. Very
  deep explorations are bounded first by `max_length`, which caps the
  depth.

## References

- Bon, J. L. and Bouissou, M. (1992). Fiabilité des grands systèmes
  séquentiels : résultats théoriques et applications dans le cadre du
  logiciel GSI. *Revue de Statistique Appliquée* 40(2), 45-54.
  [Numdam](https://www.numdam.org/item/RSA_1992__40_2_45_0/).
- Collet, J. and Renault, I. (1997). Path probability evaluation with
  repeated rates. *Annual Reliability and Maintainability Symposium*,
  184-187. DOI [10.1109/rams.1997.571703](https://doi.org/10.1109/rams.1997.571703).
- Elhareef, M. H. and Yim, M.-S. (2026). Quadrature based dynamic event
  tree construction using FMU-driven piecewise deterministic Markov
  process for dynamic probabilistic risk assessment. *Annals of Nuclear
  Energy* 239, 112620.
  DOI [10.1016/j.anucene.2026.112620](https://doi.org/10.1016/j.anucene.2026.112620).
- Harrison, P. G. (1990). Laplace transform inversion and passage-time
  distributions in Markov processes. *Journal of Applied Probability*
  27(1), 74-87. DOI [10.2307/3214596](https://doi.org/10.2307/3214596).
