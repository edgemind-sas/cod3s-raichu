# Reproducibility

RAICHU is a scientific simulator: **the same seed, code and
configuration reproduce the same results — bit-for-bit where feasible.**
This page states the guarantees and how they are implemented.

## No hidden randomness

- All randomness flows from an explicit **master seed** threaded through
  the API (`simulate(model, seed=…)`, `monte_carlo(model, seed=…)`).
  There is no global RNG, no time-derived seeding.
- The generator is **ChaCha8**, chosen for its *substream* capability:
  replica *k* of a Monte-Carlo campaign draws from `(master seed,
  stream k)`. Replicas are independent **and** individually replayable —
  re-simulating trajectory *k* alone reproduces it exactly.
- The random-number dependencies are pinned (lockfile committed) and
  built without platform-specific math, keeping draws identical across
  platforms.

A single trajectory replays exactly from its `(seed, rng_stream)`:

```python
import pyraichu

model = pyraichu.load_model({
    "name": "unit",
    "components": [{"name": "C", "automata": [{
        "name": "a", "states": ["up", "down"], "init": "up",
        "transitions": [
            {"name": "fail", "source": "up", "targets": ["down"],
             "distrib": "exp", "rate": 0.05},
            {"name": "fix", "source": "down", "targets": ["up"],
             "distrib": "exp", "rate": 0.2},
        ]}]}],
})

a = pyraichu.simulate(model, t_max=100.0, seed=7, rng_stream=2)
b = pyraichu.simulate(model, t_max=100.0, seed=7, rng_stream=2)
assert [(e.time, e.transition) for e in a.events] == \
       [(e.time, e.transition) for e in b.events]
print("exact replay")
```

## Deterministic engine, deterministic reduction

- The single-trajectory engine is **single-threaded and
  deterministic**: stochastic draws happen at scheduling time, in
  transition-index order; the sensitive-function fixpoint runs in a
  documented, stable order (the [cross-validation](../benchmarks/cross-validation.md)
  compares converged states, and an optional confluence probe diagnoses
  order-dependence instead of hiding it).
- The Monte-Carlo driver parallelises across replicas but **reduces the
  results serially, in replica-index order**: estimates are
  **byte-identical whether the campaign runs on 1 or N threads** (see
  [Parallelism](parallelism.md)).
- Simultaneous events are ordered by (date, transition index) — a
  documented, stable tie-break.

## Explicit numerics

- The ODE integrator (Dormand–Prince 4(5) with dense output) exposes its
  tolerances — relative/absolute step control, maximum step and the
  event-location tolerance — as *configuration*, not hidden constants.
  See [Numerical tuning](numerical-tuning.md).
- Boundary crossings are located by bracketing on the dense interpolant,
  never stepped over silently.

## Provenance

Every result carries its provenance — engine version, model name,
horizon, seed, integrator tolerances — so a recorded run can be
re-simulated identically:

```python
result = pyraichu.simulate(model, t_max=50.0, seed=1)
print(sorted(result.provenance))     # keys include engine_version, model, seed
```

## What is *not* promised

- Bit-identity **across engine versions**: numerical or scheduling
  changes are allowed between versions; the benchmark corpus, not
  bit-identity, guards correctness.
- Cross-engine bit-identity: agreement with other engines is a
  statistical/tolerance property, documented in the
  [benchmarks](../benchmarks/cross-validation.md).
