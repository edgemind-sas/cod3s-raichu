# PyCATSHOO C++-native benchmark

Strict, reproducible comparison of the **same three models** authored
three ways — (1) in C++ against PyCATSHOO's C++ API, (2) in the normal
PyCATSHOO Python style, (3) as native RAICHU models — with identical
parameters and RNG configuration and a shared seed on the PyCATSHOO
sides. Findings and analysis: the **Benchmarks** section of the
documentation site (cross-validation → performance → accuracy–cost
parity).

The models are in `../models/` (self-contained JSON — no dependency on
any test tree). This directory holds the PyCATSHOO-side code and the
measurement drivers.

## Reproduce

### RAICHU side only (no PyCATSHOO needed)

```bash
../../.venv/bin/python run_bench.py --raichu-only        # → results-raichu-only.json
../../.venv/bin/python parity_experiment.py --raichu-only
```

### Full comparison (needs a PyCATSHOO install)

PyCATSHOO is EDF R&D freeware; download it yourself (it is not
redistributed here) from <https://pycatshoo.org> and set two
environment variables:

```bash
export PYCATSHOO_DIR=/path/to/pycatshoo-1.4.x     # a 1.4.x install (see ABI note)
export RAICHU_BENCH_PYTHON=python3.11             # interpreter matching its module
make                                              # builds ./pyc_bench (C++ models)
../../.venv/bin/python run_bench.py               # → results.json
../../.venv/bin/python parity_experiment.py       # → parity_results.json
```

`make` reads `PYCATSHOO_DIR` too; override per-invocation with
`make PYCATSHOO_DIR=…` if you don't want to export it.

### ABI note (important)

Build against a PyCATSHOO **1.4.x** distribution whose
`Core/include/PyC` headers ship with their binaries. The benchmark
links the **static** `libPycatshoo.a` (the shared object hides symbols
that header inlines reference) and stubs the FMU `getKB` entry point.
Do **not** build against a distribution whose headers have drifted from
its `.so` (subclassing `CComponent`/`CSystem` then mis-reads members
and segfaults). The 1.4.1.0 Linux tarball from pycatshoo.org is a known
self-consistent set; its Python module needs Python 3.11.

## Files

- `../models/*.json` — the three benchmark models (vendored, the source
  of truth for reproduction).
- `bench.cpp` — C++ PyCATSHOO models (`pure_exp`, `heaters_s1`,
  `heated_room_s3`) + Monte-Carlo driver, statically linked; accepts
  optional `dt`/`dtCond`/`lambda` for the parity experiment.
- `bench_py.py` — Python-callback twins on the same PyCATSHOO engine;
  byte-compatible estimates at the same seed (the faithfulness check of
  the C++ port).
- `run_bench.py` — orchestrates the engines/paths, checks the
  consistency gates, writes `results.json` (or `results-raichu-only.json`
  with `--raichu-only`).
- `parity_experiment.py` — tolerance-parity experiment on
  `heated_room_s3`: achieved accuracy (deterministic thermostat cycle
  vs its closed form) and cost per integration-effort setting on both
  engines; writes `parity_results.json`.
