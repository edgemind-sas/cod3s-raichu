# RAICHU { .raichu-title }

![RAICHU](assets/logo.svg#only-light){ .raichu-hero }
![RAICHU](assets/logo-light.svg#only-dark){ .raichu-hero }

**RAICHU** (*Rust Automata Integrating Continuous & Hazard, Unified*)
is a native, open-source **Rust** engine for the **hybrid simulation of
complex systems**: discrete stochastic behaviour (random failures,
repairs, reconfigurations) tightly coupled with continuous,
multiphysics evolution described by ODEs (temperature, level, current,
pressure, …). It ships a typed Python binding, **`pyraichu`**.

(The name reads, less formally, as *Rust Automata InCredibly Hybrid
Unleashed*: the **H** standing for the hazard rate that drives the
stochastic jumps.)

It targets reliability engineers, safety analysts and system modellers
who need to quantify how a system behaves when random events and
continuous dynamics interact, and who want an engine they can read,
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
  Weibull, lognormal, gamma, uniform, empirical, deterministic delay:
  each validated against its closed form.
- **Estimates that state their precision.** Every Monte-Carlo estimator
  carries a confidence interval, at a level the study declares rather
  than a constant: a reported probability says how well the campaign
  pinned it down. See
  [Confidence intervals](guides/confidence-intervals.md).
- **Answers "why".** An optional causal journal records why a transition
  did or did not fire, who changed an attribute, and the full consequence
  chain of an event. See [Causal journal](guides/causal-journal.md).
- **The safety output, natively.** Declare a feared event, and a
  Monte-Carlo campaign yields the **minimal cut sequences** (the ordered,
  irreducible failure chains leading to it) plus first-occurrence
  indicators. See [Sequence analysis](guides/sequence-analysis.md).
- **Exact where sampling is not enough.** The sequences leading to a
  feared event can be enumerated rather than sampled, each with its
  probability and a bound on what the cut-offs left out; the fault tree
  of a feared event is generated with its minimal cut sets and written
  as OpenPSA; importance measures (Birnbaum, Fussell-Vesely,
  criticality) come from one campaign. See
  [Sequence-tree exploration](guides/sequence-tree-exploration.md),
  [Fault trees](guides/fault-tree.md) and
  [Importance measures](guides/importance-measures.md).
- **One question, three engines.** A study (feared event, horizon, seed)
  goes to Monte-Carlo simulation, exact exploration or discretised
  exploration through one function, and each answers in the same
  envelope. See [Quantifying a study](guides/quantification.md).
- **Step through a trajectory by hand.** An interactive session fires
  the transition you choose, forces the branch it takes, and undoes. See
  [Interactive simulation](guides/interactive-simulation.md).
- **Runs platform studies.** Models and studies exported from a COD3S
  platform instance run here through muscadet, whose importer reads the
  export and whose system declaration is what this engine takes. See
  [Running a COD3S-platform study](guides/platform-import.md); a
  muscadet model runs here by naming the engine, see
  [Running a muscadet model](guides/muscadet-engine.md).

## Install

**From a release wheel** (no Rust toolchain needed): each
[GitHub release](https://github.com/edgemind-sas/cod3s-raichu/releases)
ships prebuilt `pyraichu` wheels for Linux (x86_64, aarch64), macOS
(Apple Silicon) and Windows; one abi3 wheel covers every
CPython ≥ 3.9 on its platform. Download the wheel matching your
platform and:

```bash
pip install pyraichu-<version>-cp39-abi3-<platform>.whl
```

Intel Macs are not built for: use the source build below there.

**From source**: RAICHU is a Rust workspace with a Python binding
built by [maturin](https://www.maturin.rs). Prerequisites: Rust stable,
Python ≥ 3.9.

```bash
git clone https://github.com/edgemind-sas/cod3s-raichu raichu && cd raichu
python -m venv .venv
VIRTUAL_ENV=$PWD/.venv maturin develop --release -m bindings/pyraichu/Cargo.toml
```

Or build the wheel yourself:

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
failed = estimates.indicators["P_failed"]
print("unavailability:", round(failed.mean[-1], 3),
      f"+/- {failed.ci.half_width(-1):.3f} at {failed.ci.level:.0%}")
```

## Where to go next

- **[Tutorial](tutorial/01-first-model.md)**: from your first model to a
  full hybrid system, step by step.
- **[Examples](examples/index.md)**: worked studies with schematics,
  hypotheses and results on charts: the heated tank benchmark, a heated
  room over a winter, a two-out-of-three system solved three ways, a solar
  hydrogen installation, and two published benchmarks compared with their
  reference values: EDF's emergency power supply of a nuclear plant and a
  gas production system with a buffer reservoir; and the APPRODYN
  feedwater case, whose published result it reproduces once the
  turbo-pumps are set aside.
- **[Model schema reference](reference/model-schema.md)**: every field,
  distribution and expression operator.
- **[Advanced guides](guides/reproducibility.md)**: reproducibility,
  confidence intervals, numerical tuning, the causal journal, interactive
  simulation, sequence analysis, sequence-tree exploration, quantifying
  a study with the three engines, importance measures, fault trees, the muscadet authoring layer, running a muscadet
  model, platform import, parallelism.
- **[Raw sequence corpus format](reference/sequence-format.md)**: the
  file a sequence campaign writes, for auditing or another tool.
- **[Benchmarks](benchmarks/cross-validation.md)**: RAICHU measured,
  honestly, against an established C++ engine.

RAICHU is part of the **COD3S** modelling ecosystem and is released
under the MIT licence.
