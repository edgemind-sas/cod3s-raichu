# Performance

With [correctness established](cross-validation.md), we can compare
speed. Every timing below is on the **same model** with the **same
configuration** (RNG, seed, schedule, replica count); only the engine
and the modelling language differ.

!!! info "How to read these numbers"
    Wall-clock covers the **Monte-Carlo run only** — model construction
    and result extraction are excluded on every side. Each figure is the
    best of three runs. PyCATSHOO is a C++ engine; what varies is how
    often its hot loop calls back into Python (the normal PyCATSHOO
    modelling style). RAICHU never calls into Python during a run — its
    behaviour is compiled expression trees. Absolute times depend on the
    machine; **reproduce them locally** with the `benchmarks/` driver.

## Measured

Measured on an AMD Ryzen AI 9 HX 370 (24 hardware threads), Linux 6.17,
PyCATSHOO 1.4.1.0, RAICHU 0.1.0; single process on both sides.

| model | runs | PyCATSHOO C++ | PyCATSHOO Python | RAICHU 1 thread | RAICHU multi-thread |
|---|---:|---:|---:|---:|---:|
| `pure_exp` | 10 000 | 0.076 s | 0.077 s | 0.013 s | 0.003 s |
| `pure_exp` | 100 000 | 0.766 s | 0.775 s | 0.076 s | 0.026 s |
| `heaters_s1` | 10 000 | 0.079 s | 0.104 s | 0.012 s | 0.003 s |
| `heated_room_s3` | 2 000 | 1.459 s | 34.25 s | 3.888 s | 0.327 s |

## What it shows

- **Pure discrete-stochastic (`pure_exp`, `heaters_s1`).** With no
  interpreter crossings on either side, RAICHU's scheduler is ~×6–10
  faster than PyCATSHOO's C++, single thread against single thread. This
  gap is engine-native.
- **The Python modelling style dominates hybrid cost.** On
  `heated_room_s3`, the *same* PyCATSHOO engine runs the same model
  ×23 slower when its ODE right-hand side and boundary checks are Python
  callbacks (34.25 s) rather than compiled C++ (1.459 s). Most of a
  naive "PyCATSHOO is slow" impression is the interpreter boundary, not
  the engine.
- **The hybrid engine-vs-engine line needs care.** At default settings
  PyCATSHOO C++ (1.459 s) looks faster than RAICHU single-thread
  (3.888 s) here — but the two are running at *very different numerical
  accuracy*. Comparing them fairly requires matching accuracy first,
  which is exactly what the next page does.

RAICHU's default multi-threading (rightmost column, byte-identical to the
single-thread result) uses the whole machine; PyCATSHOO's own parallelism
(MPI) is not exercised here.

→ [Accuracy–cost parity](accuracy-cost-parity.md) — the honest reading of
the hybrid line.
