# Accuracy–cost parity

The [performance page](performance.md) left one line unresolved: on the
hybrid `heated_room_s3` model, PyCATSHOO C++ (1.46 s) appeared faster
than RAICHU single-thread (3.89 s). That comparison is **incomplete**,
because the two engines were integrating the same ODE to *very different
accuracy*. A speed number without an accuracy number is not a fair
comparison. This page supplies the missing axis.

## Measuring achieved accuracy

Set the heater's failure rate to zero and the model becomes a
**deterministic thermostat cycle** whose piecewise-linear ODE has a
**closed-form solution**. We can therefore measure each engine's *actual*
accuracy — the maximum temperature error against the exact trajectory —
for a range of integration-effort settings, alongside the cost of the
stochastic 2 000-replica run at that setting.

Measured on the same machine as the [performance page](performance.md):

| engine | setting | max \|ΔT\| vs exact | Monte-Carlo wall |
|---|---|---:|---:|
| PyCATSHOO C++ | `dtCond` 10⁻³ (its own default) | 1.3·10⁻² | 2.16 s |
| PyCATSHOO C++ | `dtCond` 10⁻⁶ | 7.7·10⁻⁶ | 2.20 s |
| PyCATSHOO C++ | `dtCond` 10⁻¹⁰ | 8.2·10⁻⁷ † | 2.22 s |
| RAICHU 1t | defaults (`rtol` 10⁻⁹, event 10⁻¹⁰) | 5.0·10⁻¹⁰ | 4.78 s |
| RAICHU 1t | fast (`rtol` 10⁻⁶, event 10⁻⁶) | 8.8·10⁻⁶ | 0.41 s |
| RAICHU 1t | loose (`rtol` 10⁻⁴) | 4.4·10⁻⁴ | 0.18 s |

† PyCATSHOO's accuracy floors here at ~8·10⁻⁷ because its indicators are
stored in 32-bit floats; tightening further changes nothing.

## The honest reading

- **At matched accuracy, RAICHU is faster.** At ~10⁻⁵ °C — the accuracy
  of the cross-validation baseline — RAICHU (0.41 s) is **~×5 faster**
  than PyCATSHOO C++ at its comparable 10⁻⁶ setting (2.20 s). The ×2.7
  the [performance page](performance.md) showed in PyCATSHOO's favour was
  **an artefact of RAICHU's very conservative defaults**, which buy 3–4
  orders of magnitude more accuracy than the comparison needed.
- **PyCATSHOO's cost barely moves with accuracy.** Its fixed-step
  integrator spends the same ~2.2 s whether it locates events at 10⁻³ or
  10⁻¹⁰; tightening the *step* instead (not shown) costs ×10 for no
  accuracy gain on this smooth ODE. RAICHU's adaptive integrator, by
  contrast, spends effort where the tolerance asks — ×12 between its fast
  and default profiles.
- **RAICHU's default is not slow — it is precise.** Its 4.78 s buys
  10⁻¹⁰ accuracy, ~3–4 orders of magnitude beyond what PyCATSHOO can even
  represent, for ~×2 the cost of PyCATSHOO's default (which is at 10⁻²).

So the hybrid engine-vs-engine hierarchy matches the discrete one: **at
equal accuracy RAICHU is several times faster**, single thread, before
its [thread pool](../guides/parallelism.md) adds another order of
magnitude. The lesson is methodological — always state the accuracy a
timing was measured at. RAICHU makes that explicit through its
[numerical-tuning knobs](../guides/numerical-tuning.md).

## Reproduce it

```bash
cd benchmarks/pycatshoo-cpp
python parity_experiment.py --raichu-only     # RAICHU accuracy/cost grid
# full grid (both engines) needs a PyCATSHOO install — see the README
```
