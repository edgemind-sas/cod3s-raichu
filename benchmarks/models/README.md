# Benchmark models

The three RAICHU models the benchmark measures, as self-contained JSON
so `benchmarks/` reproduces without any external tree:

| File | What it exercises |
|---|---|
| `pure_exp.json` | two-state exponential component, zero callbacks — the pure discrete-stochastic reference |
| `heaters_s1.json` | one thermostatic heater: conditional ON/OFF + exponential failure/repair |
| `heated_room_s3.json` | hybrid: room-temperature ODE + watched thermostats + stochastic failures |

Each file is the exact model used in the cross-validation corpus, so
the benchmark and the correctness suite agree by construction. The
`bench.cpp` / `bench_py.py` PyCATSHOO models transcribe the same
parameters (single numeric source). An internal drift-guard test keeps
these copies byte-identical to the corpus; downstream, these JSON files
are the source of truth.
