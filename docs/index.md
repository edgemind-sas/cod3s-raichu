# RAICHU

**RAICHU** — *Rust Automata Integrating Continuous & Hazard, Unified* —
is a native, open-source **Rust** engine for the **hybrid simulation of
complex systems**: discrete stochastic behaviour — random failures,
repairs, reconfigurations — tightly coupled with continuous,
multiphysics evolution described by ODEs (temperature, level, current,
pressure, …). It ships a typed Python binding, **`pyraichu`**.

(The name reads, less formally, as *Rust Automata InCredibly Hybrid
Unleashed* — the **H** standing for the hazard rate that drives the
stochastic jumps.)

It targets reliability engineers, safety analysts and system modellers
who need to quantify how a system behaves when random events and
continuous dynamics interact — and who want an engine they can read,
extend and trust. RAICHU implements the **piecewise-deterministic
Markov process** (PDMP) formalism (Desgeorges et al. 2021) with an
emphasis on **reproducibility, numerical rigour and inspectability**.

## Why RAICHU

- **One formalism for both worlds.** Component automata (failures,
  modes, reconfigurations) and continuous equations (ODEs) live in the
  same model and influence each other; boundary crossings become
  discrete events, located precisely.
- **Reproducible by construction.** Randomness flows from an explicit
  seed; every trajectory replays bit-for-bit; a parallel Monte-Carlo
  run gives the *same* numbers on 1 or N threads. See
  [Reproducibility](guides/reproducibility.md).
- **Models are data.** A model is JSON (or a Python `dict`): inspectable,
  serializable, diffable, validated at build time with precise typed
  errors instead of crashes.
- **A rich distribution library.** Exponential (with state-dependent rates),
  Weibull, lognormal, gamma, uniform, empirical, deterministic delay —
  each validated against its closed form.
- **Answers "why".** An optional causal journal records why a transition
  did or did not fire, who changed an attribute, and the full consequence
  chain of an event. See [Causal journal](guides/causal-journal.md).

## Install

RAICHU is a Rust workspace with a Python binding built by
[maturin](https://www.maturin.rs). Prerequisites: Rust stable, Python
≥ 3.10.

```bash
git clone https://github.com/edgemind-sas/cod3s-raichu raichu && cd raichu
python -m venv .venv
VIRTUAL_ENV=$PWD/.venv maturin develop --release -m bindings/pyraichu/Cargo.toml
```

Or build a wheel and `pip install` it:

```bash
maturin build --release -m bindings/pyraichu/Cargo.toml
pip install target/wheels/pyraichu-*.whl
```

## Hello, model

A repairable component, one trajectory, and a Monte-Carlo estimate of
its unavailability:

```python
import pyraichu

model = pyraichu.load_model({
    "name": "pump",
    "components": [{
        "name": "P",
        "automata": [{
            "name": "health", "states": ["working", "failed"], "init": "working",
            "transitions": [
                {"name": "fail", "source": "working", "targets": ["failed"],
                 "distrib": "exp", "rate": 0.01},
                {"name": "repair", "source": "failed", "targets": ["working"],
                 "distrib": "exp", "rate": 0.1},
            ],
        }],
    }],
    "indicators": [{"name": "P_failed", "target": "state",
                    "component": "P", "automaton": "health", "state": "failed"}],
})

result = pyraichu.simulate(model, t_max=200.0, seed=1)
print(len(result.events), "events")

estimates = pyraichu.monte_carlo(model, nb_runs=2000, t_max=200.0,
                                 seed=1, samples=[20.0 * k for k in range(11)])
print("unavailability:", round(estimates.indicators["P_failed"].mean[-1], 3))
```

## Where to go next

- **[Tutorial](tutorial/01-first-model.md)** — from your first model to a
  full hybrid system, step by step.
- **[Model schema reference](reference/model-schema.md)** — every field,
  distribution and expression operator.
- **[Advanced guides](guides/reproducibility.md)** — reproducibility,
  numerical tuning, the causal journal, the muscadet authoring layer,
  parallelism.
- **[Benchmarks](benchmarks/cross-validation.md)** — RAICHU measured,
  honestly, against an established C++ engine.

RAICHU is part of the **COD3S** modelling ecosystem and is released
under the MIT licence.
