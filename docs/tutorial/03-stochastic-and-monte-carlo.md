# 3. Stochastic behaviour and Monte-Carlo

Chapter 1 used exponential distributions. Real components age, get repaired in
roughly-constant times, or follow measured distributions. This chapter
covers the **distribution library**, **state-dependent rates**, and what a
**Monte-Carlo** run actually estimates — means, quantiles and sojourn
times — plus how seeds make every result reproducible.

## Timed vs instantaneous transitions

A transition comes in two flavours, depending on where its randomness
lives:

- a **timed** transition draws its *firing date* — deterministically
  (`delay`) or from a distribution (`exp`, `weibull`, …) — and moves to a
  single destination;
- an **instantaneous** transition fires *the moment its guard holds*, and
  its randomness is instead a **probabilistic choice of destination**
  (`inst`, with `probs`). We meet these in [chapter 4](04-going-hybrid.md);
  a guard on a *continuous* attribute makes an instantaneous transition
  into a *watched* one.

This chapter is about the timed, stochastic kind — the failure and repair
distributions.

## The distribution library

A timed transition's distribution is chosen with the `distrib` key and its parameters.
The stochastic distributions are all built in:

| `distrib` | parameters | models |
|---|---|---|
| `exp` | `rate` (or `rate_expr`) | constant hazard (memoryless) |
| `delay` | `time` | a fixed, deterministic duration |
| `weibull` | `shape`, `scale` | wear-out / infant mortality (aging) |
| `lognormal` | `mu`, `sigma` | repair durations, multiplicative effects |
| `gamma` | `shape`, `scale` | sums of exponential stages |
| `uniform` | `low`, `high` | a bounded, equiprobable duration |
| `empirical` | `points` = `[[t, F(t)], …]` | a measured CDF |

Each distribution is a drop-in replacement. Here we fail one component seven
ways and read the probability it is down by *t = 120*:

```python
import pyraichu

LAWS = {
    "exp":       {"distrib": "exp", "rate": 0.02},
    "delay":     {"distrib": "delay", "time": 50.0},
    "weibull":   {"distrib": "weibull", "shape": 2.0, "scale": 60.0},
    "lognormal": {"distrib": "lognormal", "mu": 3.9, "sigma": 0.4},
    "gamma":     {"distrib": "gamma", "shape": 2.0, "scale": 30.0},
    "uniform":   {"distrib": "uniform", "low": 40.0, "high": 80.0},
    "empirical": {"distrib": "empirical",
                  "points": [[0.0, 0.0], [50.0, 0.5], [100.0, 1.0]]},
}

for name, distribution in LAWS.items():
    transition = {"name": "fail", "source": "up", "targets": ["down"], **distribution}
    model = pyraichu.load_model({
        "name": name,
        "components": [{"name": "C", "automata": [{
            "name": "a", "states": ["up", "down"], "init": "up",
            "transitions": [transition]}]}],
        "indicators": [{"name": "down", "target": "state",
                        "component": "C", "automaton": "a", "state": "down"}],
    })
    est = pyraichu.monte_carlo(model, nb_runs=1000, t_max=120.0,
                               samples=[120.0], seed=1)
    print(f"{name:10} P(down by 120) = {est.indicators['down'].mean[0]:.3f}")
```

Every distribution is validated against its closed-form CDF in the test suite,
so these are not approximations of a distribution — they *are* the
distribution.

## State-dependent rates

An exponential rate can be an **expression** instead of a constant, via
`rate_expr`. The rate is then re-evaluated as the state it reads
changes — a load that rises, a temperature that climbs. Here the hazard
is proportional to a `stress` attribute:

```python
model = pyraichu.load_model({
    "name": "stressed",
    "components": [{
        "name": "C",
        "attributes": [{"name": "stress", "kind": "float",
                       "init": {"kind": "float", "value": 1.0}}],
        "automata": [{"name": "a", "states": ["up", "down"], "init": "up",
            "transitions": [{
                "name": "fail", "source": "up", "targets": ["down"], "distrib": "exp",
                "rate_expr": {"op": "mul", "args": [
                    {"op": "const", "value": {"kind": "float", "value": 0.01}},
                    {"op": "attr", "attr": {"component": "C", "attribute": "stress"}},
                ]},
            }]}],
    }],
    "indicators": [{"name": "down", "target": "state",
                    "component": "C", "automaton": "a", "state": "down"}],
})
```

When the rate depends on a *continuous* attribute (chapter 4), RAICHU
integrates the cumulative hazard exactly rather than freezing the rate —
see [Numerical tuning](../guides/numerical-tuning.md).

## What a Monte-Carlo run estimates

`monte_carlo` returns, per indicator and per schedule instant:

- **`mean`** and **`std`** — the estimate and its dispersion across
  replicas;
- **`quantiles`** — request them with `quantiles=[…]`;
- **`sojourn_mean` / `sojourn_std` / `sojourn_quantiles`** — the
  time-integral of the indicator up to each instant. For a 0/1 state
  indicator, that is the **cumulated time spent in the state** — e.g.
  total downtime.

```python
est = pyraichu.monte_carlo(
    model, nb_runs=5000, t_max=100.0,
    samples=[100.0], seed=1, quantiles=[0.5, 0.9],
)
down = est.indicators["down"]
print("P(down by 100) :", round(down.mean[0], 3))
print("mean downtime  :", round(down.sojourn_mean[0], 1))
print("median downtime:", round(down.sojourn_quantiles[0.5][0], 1))
print("90th pct        :", round(down.sojourn_quantiles[0.9][0], 1))
```

Because these are estimates from a finite sample, treat them as such:
the `std` and the replica count give the standard error. The
[reproducibility guide](../guides/reproducibility.md) covers seeding and
confidence.

## Reproducible by construction

Randomness flows only from the `seed`. Replica *r* uses substream *r*,
so replicas are independent **and** individually replayable: simulating
the same `(seed, rng_stream)` twice yields byte-identical events.

```python
a = pyraichu.simulate(model, t_max=100.0, seed=7, rng_stream=3)
b = pyraichu.simulate(model, t_max=100.0, seed=7, rng_stream=3)
assert [e.time for e in a.events] == [e.time for e in b.events]
print("replay is exact")
```

→ [4. Going hybrid: continuous dynamics](04-going-hybrid.md)
