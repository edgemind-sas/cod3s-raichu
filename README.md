# RAICHU

**R**ust **A**utomata **InC**redibly **H**ybrid **U**nleashed — a native,
open-source **Rust** engine for the **hybrid simulation of complex
systems**: discrete stochastic behaviour (random failures, repairs,
reconfigurations) tightly coupled with continuous multiphysics evolution
(ODEs describing temperature, level, current, pressure, …).

RAICHU implements the **Piecewise-Deterministic Markov Process** (PDMP)
formalism, realised as **Distributed Stochastic Hybrid Automata**
(DSHA): a system moves through discrete modes, each constraining a
continuous state that evolves by ODEs, while mode changes are triggered
either spontaneously (state-dependent hazard rates) or when the
continuous trajectory reaches a boundary. It ships a typed Python
binding, **`pyraichu`**.

📖 **Documentation:** <https://edgemind-sas.github.io/cod3s-raichu> —
start with the [tutorial](docs/tutorial/01-first-model.md).

## What RAICHU provides

- **Coupled discrete/continuous dynamics** — component-level automata
  (delay, instantaneous, watched/boundary and stochastic transitions)
  over a continuous state integrated with a Dormand–Prince 4(5)
  dense-output solver and guaranteed event location.
- **A rich occurrence-law library** — exponential (with an optional
  state-dependent rate), Weibull, lognormal, gamma, uniform,
  deterministic/delay and empirical laws behind a single interface.
- **Reproducibility by construction** — an explicit, splittable RNG; a
  master seed derives independent per-replica substreams; any single
  trajectory replays bit-for-bit and Monte-Carlo results are
  byte-identical regardless of thread count. Every result carries its
  provenance (engine version, seed, model, parameters).
- **Serializable models** — components, typed variables, in/out ports
  and interfaces, automata, and expression-tree guards, effects and ODE
  right-hand sides, all as plain JSON. A `pyraichu.muscadet` authoring
  layer and a JSON plugin system provide higher-level idioms.
- **Explainability** — an optional, queryable causal journal (event →
  triggered functions → variable changes → rescheduling), zero-cost when
  disabled.
- **Build-time diagnostics** — typed model validation, non-confluence
  and instantaneous-loop detection.

## Landscape

RAICHU belongs to the family of formal modelling and simulation tools
for dependability and dynamic reliability. Neighbouring approaches
include **AltaRica 3.0** (guarded transition systems for stochastic,
discrete-event safety modelling), **FIGARO / KB3** (EDF's reliability
modelling language and its knowledge bases) and **PyCATSHOO** (EDF R&D),
whose PDMP-based treatment of *hybrid* discrete-stochastic and
continuous dynamics is the direct inspiration for RAICHU's formalism.
RAICHU is an independent, open-source implementation with no dependency
on any of these tools; PyCATSHOO is used only as an external reference
oracle for cross-validation. Cross-engine benchmarks are published in
the project documentation.

## Layout

- `crates/` — the pure-Rust engine (model, expressions, core cycle, I/O,
  numerics, RNG) under an umbrella crate `raichu`.
- `bindings/pyraichu/` — PyO3/maturin Python binding, published on PyPI
  as **`pyraichu`**.
- `python/tests/validation/` — cross-validation harness (differential
  testing against the reference oracle).
- `docs/` — the documentation site (`mkdocs`).

## Build

```bash
cargo build --workspace                 # engine
cargo test --workspace                  # Rust tests
cargo clippy --workspace --all-targets  # lint

# Python binding (inside a virtualenv):
maturin develop -m bindings/pyraichu/Cargo.toml
python -c "import pyraichu; print(pyraichu.__version__)"
```

## License

MIT — © Roland Donat / EdgeMind.
