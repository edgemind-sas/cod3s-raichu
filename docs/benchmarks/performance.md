# Performance

With [correctness established](cross-validation.md), we can compare
speed. Every timing below is on the **same model** with the **same
configuration** (RNG, seed, schedule, replica count); only the engine,
the modelling language and the number of workers differ.

!!! info "How to read these numbers"
    Wall clock covers the **Monte-Carlo run only**: model construction
    and result extraction are excluded on every side. Each point is the
    best of three runs. On the hybrid model both engines run at the
    **same achieved accuracy** (see [Accuracy-cost parity](accuracy-cost-parity.md)),
    never at their defaults. PyCATSHOO is a C++ engine; what varies is
    how often its hot loop calls back into Python (the normal PyCATSHOO
    modelling style). RAICHU never calls into Python during a run: its
    behaviour is compiled expression trees. Absolute times depend on the
    machine: **reproduce them locally** with the `benchmarks/` driver.

## The campaign

Measured on 2026-10-14 on an AMD Ryzen 9 7950X3D (16 cores, 32 logical
CPUs), Linux 7.0, with PyCATSHOO 1.4.1.0 (models compiled in C++ against
its API), RAICHU 0.96.0, and Open MPI 5.0.8 for PyCATSHOO's parallel runs.
The raw results are committed in
`benchmarks/results/jaquet-2026-10-14.json`, and the charts are drawn from
that file by `docs/figures/benchmark_performance.py`. The previous
campaign, on RAICHU 0.63.4 (2026-09-28, same machine), stays in
`benchmarks/results/jaquet-2026-09-28.json` for comparison.

| model | what it exercises |
|---|---|
| `pure_exp` | a two-state exponential component: discrete-stochastic only |
| `heaters_s1` | a thermostatic heater: conditional switching and failures |
| `heated_room_s3` | a room-temperature ODE, watched thermostats and failures: hybrid |

Every point also carries a consistency check. The RAICHU estimates agree
statistically with PyCATSHOO's: the largest z-score over the 11 schedule
instants is 2.65, under the 2.83 that a 5 % level corrected for 11
comparisons (Šidák) allows. Under MPI, PyCATSHOO's estimates equal its
single-process ones to 4·10⁻⁶ or better.

Against the 0.63.4 campaign, RAICHU's single-thread timings moved by
less than 10 % on `heaters_s1` and `heated_room_s3`. On `pure_exp` they
moved further, in both directions: +10 to +14 % between 10 000 and
100 000 replicas, +32 to +48 % on the shortest runs (1 to 4 ms), but
−19 % at a million replicas. On the accuracy grid they moved by 5 to
11 %. We read these moves as the run-to-run variation of this machine
rather than a regression, since they have no common direction and
PyCATSHOO moved as much: up to 9 % on the same replica sweep and 8 to
21 % on the accuracy grid. A campaign repeated on one date would settle
it. The z-scores are the same to every digit, since the same seed draws
the same trajectories. The one real change is the
thread scaling on `pure_exp` (see [below](#workers-threads-against-mpi-ranks)).

## Replicas, one worker

![Wall clock against the number of replicas, one worker, on the three models](../assets/figures/benchmark-replicas-light.svg#only-light){ .figure }
![Wall clock against the number of replicas, one worker, on the three models](../assets/figures/benchmark-replicas-dark.svg#only-dark){ .figure }

| model | replicas | PyCATSHOO C++ | PyCATSHOO Python | RAICHU | ratio |
|---|---:|---:|---:|---:|---:|
| `pure_exp` | 100 000 | 0.719 s | | 0.124 s | 5.8 |
| `pure_exp` | 1 000 000 | 6.690 s | | 1.043 s | 6.4 |
| `heaters_s1` | 100 000 | 0.697 s | | 0.164 s | 4.2 |
| `heated_room_s3` | 2 000 | 1.571 s | 36.61 s | 0.230 s | 6.8 |
| `heated_room_s3` | 20 000 | 15.79 s | | 2.274 s | 6.9 |

Both engines are linear in the number of replicas, so the ratio hardly
moves along a sweep: 4 to 6.5 on the discrete models, 7 on the hybrid one
at aligned accuracy. The PyCATSHOO model with Python callbacks costs 23
times its C++ twin on the same engine: the interpreter boundary, not the
engine, dominates the normal PyCATSHOO modelling style.

## Workers: threads against MPI ranks

RAICHU parallelises replicas over threads in one process
(`monte_carlo(..., threads=k)`), with estimates byte-identical for any
`k`. PyCATSHOO parallelises them over MPI ranks (`mpirun -n k`); the
timing is its first rank's, which gathers the others' estimates.

![Wall clock against the number of workers: RAICHU threads against PyCATSHOO MPI ranks, with the ideal scaling dotted](../assets/figures/benchmark-workers-light.svg#only-light){ .figure }
![Wall clock against the number of workers: RAICHU threads against PyCATSHOO MPI ranks, with the ideal scaling dotted](../assets/figures/benchmark-workers-dark.svg#only-dark){ .figure }

| model | workers | PyCATSHOO MPI | RAICHU threads |
|---|---:|---:|---:|
| `pure_exp`, 1 000 000 replicas | 1 | 6.68 s | 0.94 s |
| | 4 | 2.05 s | 0.33 s |
| | 16 | 1.06 s | 0.21 s |
| | 32 | 1.11 s | 0.20 s |
| `heated_room_s3`, 20 000 replicas | 1 | 15.13 s | 2.29 s |
| | 4 | 5.05 s | 0.60 s |
| | 16 | 2.06 s | 0.17 s |
| | 32 | 2.10 s | 0.14 s |

What the curves show:

- **On the hybrid model RAICHU's threads scale well**: 14 times faster
  on 16 threads than on one (the 16 physical cores), and 17 times on
  32.
- **On `pure_exp` they scale less.** A replica there costs about a
  microsecond, so the work shared by all the replicas, the reduction of
  their results, weighs more past eight threads: 4.9 times faster at
  best. PyCATSHOO's ranks scale more in relative terms (6.3 times) while
  staying five times slower in absolute terms. On RAICHU 0.63.4 the same
  sweep stopped at 2.2 times (0.58 s on 32 threads, against 0.20 s now).
- **PyCATSHOO's parallel timings fall on whole seconds** plus a few
  milliseconds (1.06, 2.05, 3.05, 4.06, 5.05 s), and stop improving at
  about 1 s on `pure_exp` and 2 s on `heated_room_s3`. The pattern
  suggests that the first rank collects the others' results on a
  one-second cycle; that reading is ours, drawn from the timings, not
  from PyCATSHOO's documentation.

## Reproduce it

```bash
cd benchmarks/pycatshoo-cpp
make PYCATSHOO_DIR=/path/to/pycatshoo-1.4.1.0        # builds ./pyc_bench
PYCATSHOO_DIR=/path/to/pycatshoo-1.4.1.0 RAICHU_BENCH_PYTHON=python3.11 \
    python campaign.py --out ../results/<machine>-<date>.json
python ../../scripts/build_doc_figures.py benchmark   # with RAICHU_BENCH_RESULTS pointing at it
```

`campaign.py --quick` shrinks every campaign by 20 to check the set-up in
a few minutes. The PyCATSHOO side needs its 1.4.1.0 distribution and, for
the MPI sweep, an Open MPI whose `libmpi.so` is on the library path; the
RAICHU side needs only `pyraichu`. See `benchmarks/pycatshoo-cpp/README.md`.

→ [Accuracy-cost parity](accuracy-cost-parity.md): how the aligned
settings of the hybrid model were chosen.
