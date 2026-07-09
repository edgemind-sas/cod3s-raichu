# Cross-validation

A performance number means nothing if the two engines are not computing
the same thing. So the benchmark starts here: **the same model, authored
three ways, produces the same results.** Only once that is established do
the [performance](performance.md) and
[accuracy–cost](accuracy-cost-parity.md) pages report timings.

The comparison is against **PyCATSHOO** (EDF R&D), a mature C++ hybrid
simulation engine, on three models of increasing coupling:

| model | what it exercises |
|---|---|
| `pure_exp` | two-state exponential component — pure discrete-stochastic, zero callbacks |
| `heaters_s1` | a thermostatic heater — conditional switching + failures |
| `heated_room_s3` | a room-temperature ODE + watched thermostats + failures — fully hybrid |

## The same model, three ways

Each model is written (1) in C++ against PyCATSHOO's C++ API, (2) in the
normal PyCATSHOO Python style, and (3) as a native RAICHU model. Here is
the simplest one, `pure_exp`, in all three — the code is included
verbatim from the benchmark sources, so it cannot drift from what is
actually run:

=== "RAICHU (native JSON)"

    ```json
    --8<-- "benchmarks/models/pure_exp.json"
    ```

=== "PyCATSHOO (C++)"

    ```cpp
    --8<-- "benchmarks/pycatshoo-cpp/bench.cpp:pureexp"
    ```

=== "PyCATSHOO (Python)"

    ```python
    --8<-- "benchmarks/pycatshoo-cpp/bench_py.py:pureexp"
    ```

The three describe the same two-state automaton with the same rates. The
larger models (`heaters_s1`, `heated_room_s3`) follow the same
three-way structure in the `benchmarks/` directory.

## The agreement

Every model is run on all three renderings with the **same seed** and
the same configuration (RNG, schedule, replica count). The consistency
gates:

- **The two PyCATSHOO renderings must be identical.** They share every
  random draw, so at a fixed seed the C++ and Python models must return
  the same estimates to the byte (up to the engine's float32 indicator
  storage). Measured worst-case difference across the three models:

  | model | max \|Δmean\| C++ vs Python |
  |---|---|
  | `pure_exp` (100 000 runs) | 2.9·10⁻¹¹ |
  | `heaters_s1` (10 000 runs) | 4.8·10⁻⁹ |
  | `heated_room_s3` (2 000 runs) | 4.1·10⁻⁸ |

  This is the *faithfulness check* of the C++ port: identical results
  prove the C++ and Python models are the same model.

- **RAICHU must agree statistically.** RAICHU uses a different RNG, so it
  need not match run-for-run; instead its Monte-Carlo estimators must be
  statistically indistinguishable. The worst standardised deviation
  (z-score) between the PyCATSHOO and RAICHU means, across all
  indicators and schedule instants, stays within the expected band:

  | model | worst z (RAICHU vs PyCATSHOO) |
  |---|---|
  | `pure_exp` | 2.65 |
  | `heaters_s1` | 2.65 |
  | `heated_room_s3` | 1.49 |

  all below the multiple-comparison critical value — the estimators are
  indistinguishable.

## Reproduce it

The models, the PyCATSHOO code and the measurement driver live in
`benchmarks/`. The RAICHU side runs with no PyCATSHOO install:

```bash
cd benchmarks/pycatshoo-cpp
python run_bench.py --raichu-only          # RAICHU timings + estimates
```

The full three-way comparison needs a PyCATSHOO install; see the
`benchmarks/pycatshoo-cpp/README.md` for the (freeware) download and the
exact steps.

With correctness established, we can compare speed —
[Performance](performance.md).
