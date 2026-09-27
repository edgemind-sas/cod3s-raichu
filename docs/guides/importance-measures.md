# Importance measures: which component carries the risk

A safety study rarely stops at the probability of the feared event. The
next question is always the same one, and it is the one that turns a
study into a decision: **which component contributes most, and where does
investment pay**. RAICHU answers it natively, from a single Monte-Carlo
campaign, with the classical measures: **Birnbaum**, **Fussell-Vesely**,
**criticality**, and the two pivotal risk levels that risk-achievement and
risk-reduction worth are built from.

The support is the same one as [sequence
analysis](sequence-analysis.md): the **minimal cut sequences**. Dropped to
sets they are the minimal cut sets, hence the structure function of the
system; the same recorded trajectories replay the state of every failure
mode at any instant. Having both, the measures are computed **by their
definitions** rather than by the rare-event sum a fault-tree tool falls
back on.

## An example: one pump in series, two in parallel

The station loses its supply if `pump_A` fails, or if **both** `pump_B`
and `pump_C` do: two minimal cuts of different sizes, the smallest
diagram where the answer is not obvious.

```python
import pyraichu

RATES = {"pump_A": 0.02, "pump_B": 0.05, "pump_C": 0.08}

model = pyraichu.load_model({
    "name": "pumping_station",
    "plugins": {"muscadet": {"objects": [
        {"type": "ObjFM", "name": f"fm_{block}", "targets": [block],
         "failure": [{"law": "exp", "rate": rate}], "repair": None,
         "failure_effects": {"flow": False}}
        for block, rate in RATES.items()
    ] + [
        {"type": "ObjEvent", "name": "no_supply", "target": True,
         "cond": [
             [{"obj": "pump_A", "attr": "flow", "ope": "==", "value": False}],
             [{"obj": "pump_B", "attr": "flow", "ope": "==", "value": False},
              {"obj": "pump_C", "attr": "flow", "ope": "==", "value": False}],
         ]},
    ]}},
    "components": [
        {"name": block, "attributes": [
            {"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}]}
        for block in RATES
    ],
    "indicators": [],
})

analysis = pyraichu.importance(
    model, nb_runs=20000, t_max=40.0, instants=[10.0, 40.0], seed=42
)

print(f"P(no_supply at t=40) = {analysis.q_cuts[-1]:.3f}")
for cut in analysis.cuts:
    print(f"  cut {' + '.join(cut.events):32s} weight {cut.weight:6.0f}")
for name, c in analysis.components.items():
    print(f"{name:12s} q={c.unavailability[-1]:.3f} "
          f"Birnbaum={c.birnbaum[-1]:.3f} FV={c.fussell_vesely[-1]:.3f} "
          f"crit={c.criticality[-1]:.3f}")
```

```text
P(no_supply at t=40) = 0.924
  cut fm_pump_B.occ + fm_pump_C.occ    weight  11916
  cut fm_pump_A.occ                    weight   6570
fm_pump_B    q=0.867 Birnbaum=0.432 FV=0.901 crit=0.405
fm_pump_C    q=0.960 Birnbaum=0.389 FV=0.901 crit=0.404
fm_pump_A    q=0.550 Birnbaum=0.167 FV=0.595 crit=0.100
```

Components come back **ranked**, most important first, so
`list(analysis.components)` reads as the answer to the question that was
asked. The reported name is the object that owns the monitored failure
transition, which in a muscadet model is the `ObjFM`.

Note what the ranking says here, and that no amount of staring at the
diagram would have said it: the single point of failure `pump_A` is
**not** where to invest. Its cut is a singleton, so it looks like the
obvious weak point, but at this horizon the redundant pair is so degraded
that 90 % of the losses of supply pass through it.

## Reading the four numbers

They answer different questions, and a decision usually needs two of them.

| Measure | Question it answers |
|---|---|
| `unavailability` `q_i(t)` | how often is this component down |
| `birnbaum` | how much does the risk move per unit of this component's unavailability: the **sensitivity**, and it does not depend on how likely the component is to fail |
| `fussell_vesely` | what **share** of the risk passes through a cut containing this component |
| `criticality` | the probability that this component is down *and* critical, given the system is down: Birnbaum re-weighted by `q_i` |

Birnbaum is the one to read when asking "how much would a better component
buy"; Fussell-Vesely is the one to read when asking "where does the risk
go today". They disagree exactly when a component is very sensitive but
rarely down, or the reverse: `fm_pump_A` above has the *lowest* Birnbaum
and a middling share, because its partner cut is realized so often that
its own failure changes little.

The two **pivotal risk levels** are reported as levels rather than as
ratios, because both denominators legitimately reach zero:

```python
component = analysis.components["fm_pump_A"]
component.q_system_failed   # the risk with this component certainly failed
component.q_system_intact   # the risk with this component made perfect
component.risk_achievement(analysis)  # Q⁺/Q, risk-achievement worth
component.risk_reduction(analysis)    # Q/Q⁻, risk-reduction worth
```

## What is actually computed

Write `D` for the set of basic events (monitored failure states) active at
an instant, `E_i` for those of component `i`, and `Φ` for the structure
function reconstructed from the minimal cuts, `Φ(D) = 1` iff some cut
`K ⊆ D`. Then

- **Birnbaum** is the *pivotal* difference
  `I^B_i = E[Φ(D ∪ E_i) − Φ(D \ E_i)]`: the probability that the system is
  **critical** for the component, failing if it fails and holding if it
  holds. Written this way it assumes nothing about independence between
  components, which matters, because a model with common-cause failures
  violates independence by construction and the textbook
  `∂Q/∂q_i` reading would then be wrong.
- **Fussell-Vesely** is
  `FV_i = P(some realized cut contains an event of i | system down)`,
  evaluated on the empirical joint state and not on a rare-event sum of
  cut probabilities. The measures do **not** sum to one, and must not: two
  cuts can be realized at once.
- **Criticality** is `I^B_i · q_i / Q`.

## The completeness diagnostic

The cut structure is empirical: a path no replica walked is not in it, and
a component that only fails through such a path would read as
unimportant. The analysis reports the gap rather than leaving it to be
guessed at:

```python
analysis.q_target   # the feared event as the trajectories recorded it
analysis.q_cuts     # the same instants judged by the reconstructed structure
```

They agree when the corpus is complete. A `q_cuts` below `q_target` means
some way of reaching the feared event was never sampled: raise `nb_runs`
before reading the ranking.

## What it costs

One campaign, plus a reduction linear in the replicas. On the project's
hybrid benchmark (room-temperature ODE, watched thermostats, stochastic
failures) the measures cost **the campaign and nothing measurable on top**:
1.04 s against 1.04 s for a bare Monte-Carlo run of the same size. On a
purely discrete model, where a trajectory is almost free and the
post-processing is as visible as it will ever be, 32 000 replicas of a
16-component plant reduce in 0.39 s, and the growth is linear: 4 000,
8 000, 16 000 and 32 000 replicas take 0.047 s, 0.099 s, 0.198 s and
0.39 s.

## Limits worth knowing

- A component is **failed** when any of the basic events the cut structure
  attributes to it is active, and the pivotal `Φ(D ∪ E_i)` sets all of
  them at once. For a component with one failure mode, the usual case, the
  two readings coincide; for one with several, the component-level
  Birnbaum answers "what if this component fails, however it fails".
- Only components that appear in some minimal cut are reported. One that
  never contributed within the horizon has no importance, which is the
  honest answer, but it also means the list is not the model's component
  list.
- Sequence recording must be able to tell a failure from its repair. The
  muscadet plugin does the annotation; a hand-written model needs
  `"monitored": true` and a shared `"cycle_group"` on the occ/rep pair,
  exactly as for [sequence analysis](sequence-analysis.md#native-model-without-plugins).
