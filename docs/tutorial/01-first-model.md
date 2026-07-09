# 1. Your first model

RAICHU simulates systems that mix two kinds of dynamics: **discrete,
stochastic** changes (a pump fails, a valve is repaired, a controller
switches mode) and **continuous** evolution (a temperature drifts, a
tank fills). This first chapter builds the smallest useful model — a
single repairable component — and runs it. No prior tool experience is
assumed.

A model is plain **data**: a Python `dict` (or JSON). You hand it to
`pyraichu.load_model`, which validates it and returns a `Model` you can
simulate. Because a model is data, it is inspectable, serializable and
diffable — there is no hidden state.

## A repairable component

Our component `P` has one **automaton** — a little state machine —
named `health`, with two states, `working` and `failed`. Two
**transitions** move between them, each governed by an **exponential
distribution**: `fail` fires at rate `0.01` (mean time to failure = 100 time
units), `repair` at rate `0.1` (mean time to repair = 10).

```python
import pyraichu

model = pyraichu.load_model({
    "name": "pump",
    "components": [
        {
            "name": "P",
            "automata": [
                {
                    "name": "health",
                    "states": ["working", "failed"],
                    "init": "working",
                    "transitions": [
                        {"name": "fail", "source": "working",
                         "targets": ["failed"], "distrib": "exp", "rate": 0.01},
                        {"name": "repair", "source": "failed",
                         "targets": ["working"], "distrib": "exp", "rate": 0.1},
                    ],
                }
            ],
        }
    ],
    "indicators": [
        {"name": "P_failed", "target": "state",
         "component": "P", "automaton": "health", "state": "failed"},
    ],
})
```

A few things to note:

- A component only needs a `name`; every other section (`attributes`,
  `ports`, `equations`, …) defaults to empty and can be omitted.
- State names are scoped **to their automaton**, so two components can
  both have a `failed` state without clashing.
- The **indicator** `P_failed` observes whether the `health` automaton
  is in state `failed`. Indicators are what the engine measures.

## One trajectory

`simulate` runs a single trajectory up to `t_max`. Stochastic distributions draw
their firing dates from the `seed`; the same seed replays the exact same
trajectory (see [Reproducibility](../guides/reproducibility.md)).

```python
result = pyraichu.simulate(model, t_max=200.0, seed=1)

print(f"{len(result.events)} events")
for event in result.events[:4]:
    print(f"  t={event.time:7.2f}  {event.transition}: "
          f"{event.from_state} -> {event.to_state}")
```

Each `Event` records a fired transition and the states it moved
between — the exact discrete history of the run.

## Many trajectories: Monte-Carlo

One trajectory tells you little about a random system. `monte_carlo`
runs many independent replicas and estimates each indicator over a
schedule of instants. Replica `r` draws from substream `r` of the seed,
so the replicas are independent and the whole estimate is reproducible.

```python
estimates = pyraichu.monte_carlo(
    model,
    nb_runs=2000,
    t_max=200.0,
    samples=[20.0 * k for k in range(11)],   # 0, 20, 40, …, 200
    seed=1,
)

unavailability = estimates.indicators["P_failed"].mean
print("instant :", estimates.indicators["P_failed"].instants)
print("P(failed):", [round(u, 3) for u in unavailability])
```

The mean of a 0/1 state indicator at instant *t* is the **fraction of
replicas in that state** — here the pump's unavailability over time. It
climbs from 0 and settles near the steady-state value
λ/(λ+μ) = 0.01/0.11 ≈ 0.091.

!!! note "Why this matters"
    A constant-rate failure/repair pair is a two-state Markov chain
    with a known steady state — a case where you can check the
    simulator against pen-and-paper. RAICHU's test suite pins exactly
    these analytical solutions; the [numerical guides](../guides/reproducibility.md)
    explain the confidence intervals behind a Monte-Carlo estimate.

## What you built

- a **component** with an **automaton**, two **states** and two
  **transitions** carrying **exponential distributions**;
- a **state indicator**;
- a single reproducible **trajectory** and a **Monte-Carlo** estimate.

Next, we connect several components together so they can influence each
other.

→ [2. Connecting components](02-connecting-components.md)
