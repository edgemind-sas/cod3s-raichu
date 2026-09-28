# Accuracy-cost parity

A speed number without an accuracy number is not a comparison. On a
hybrid model, each engine integrates the same differential equation and
locates the same boundary crossings to a precision its settings decide,
and the two engines' defaults sit at very different precisions: timed at
their defaults, they are not solving the same problem to the same
standard. This page measures what each setting actually achieves, and
picks the settings at which the [performance page](performance.md)
compares the two engines on the hybrid model.

## Measuring achieved accuracy

Set the heater's failure rate to zero and `heated_room_s3` becomes a
**deterministic thermostat cycle** whose piecewise-linear ODE has a
**closed-form solution**: heating and cooling are first-order responses,
and the switching dates are logarithms. Each engine's actual accuracy is
then the largest temperature error, over the 11 schedule instants, of
one trajectory against the exact one. Its cost is the wall clock of the
stochastic campaign (2 000 replicas, one worker) at the same setting.

![Wall clock against achieved accuracy on the hybrid model, for a grid of settings on each engine; the circled pair is the aligned one](../assets/figures/benchmark-accuracy-cost-light.svg#only-light){ .figure }
![Wall clock against achieved accuracy on the hybrid model, for a grid of settings on each engine; the circled pair is the aligned one](../assets/figures/benchmark-accuracy-cost-dark.svg#only-dark){ .figure }

| engine | setting | max \|ΔT\| against exact | wall clock |
|---|---|---:|---:|
| PyCATSHOO C++ | `dtCond` 10⁻² | 9.0·10⁻² | 1.49 s |
| PyCATSHOO C++ | `dtCond` 10⁻³ (its documented default) | 1.3·10⁻² | 1.53 s |
| PyCATSHOO C++ | `dtCond` 10⁻⁵ | 1.3·10⁻⁴ | 1.55 s |
| PyCATSHOO C++ | `dtCond` 10⁻⁶ | 7.7·10⁻⁶ | 1.54 s |
| PyCATSHOO C++ | `dtCond` 10⁻¹⁰ | 8.2·10⁻⁷ † | 1.59 s |
| RAICHU | `rtol` 10⁻³ | 1.8·10⁻² | 0.12 s |
| RAICHU | `rtol` 10⁻⁴ | 4.4·10⁻⁴ | 0.15 s |
| RAICHU | `rtol` 10⁻⁵ | 4.1·10⁻⁶ | 0.23 s |
| RAICHU | `rtol` 10⁻⁷ | 1.5·10⁻⁷ | 0.60 s |
| RAICHU | defaults (`rtol` 10⁻⁹, event 10⁻¹⁰) | 5.0·10⁻¹⁰ | 4.09 s |

† PyCATSHOO's accuracy floors at about 8·10⁻⁷ here because its
indicators are stored in 32-bit floats; tightening further changes
nothing. The full grid, each RAICHU setting's `atol`, `tol_event`,
`max_step` and `sub_samples` included, is in
`benchmarks/results/jaquet-2026-09-28.json`. The RAICHU curve is not
monotonic between `rtol` 10⁻⁵ and 10⁻⁶ because those two settings also
differ in maximum step and scan density: accuracy follows the whole
setting, not `rtol` alone.

## Readings

- **PyCATSHOO's cost barely moves with its event tolerance.** Its
  fixed-step integrator spends about 1.5 s whether it locates events at
  10⁻² or 10⁻¹⁰; `dtCond` only refines where a crossing is placed.
- **RAICHU's cost follows the accuracy asked for.** Its adaptive
  integrator spends effort where the tolerance requires it: 35 times more
  between its loosest setting and its defaults.
- **RAICHU's defaults are precise, not slow.** They buy 5·10⁻¹⁰, three
  orders of magnitude beyond what PyCATSHOO can represent here, for 2.7
  times PyCATSHOO's cost at its own default (which is at 10⁻²).

## The aligned pair

The performance comparison on the hybrid model uses PyCATSHOO at
`dtCond` 10⁻⁶ (the setting of RAICHU's cross-validation baseline, error
7.7·10⁻⁶) and the cheapest RAICHU setting at least as accurate: `rtol`
10⁻⁵, error 4.1·10⁻⁶. At that pair RAICHU is 6.7 times faster, one
worker against one, and the ratio holds from 500 to 20 000 replicas (see
the [performance page](performance.md)).

The lesson is methodological: a timing on a hybrid model means something
only with the accuracy it was measured at. RAICHU makes the accuracy an
explicit setting of the run: see [Numerical tuning](../guides/numerical-tuning.md).

## Reproduce it

The grid is the first part of the campaign script:

```bash
cd benchmarks/pycatshoo-cpp
PYCATSHOO_DIR=/path/to/pycatshoo-1.4.1.0 \
    python campaign.py --parts accuracy --out ../results/<machine>-<date>.json
```

`parity_experiment.py`, the smaller four-setting grid this page first
reported, stays in the same directory.
