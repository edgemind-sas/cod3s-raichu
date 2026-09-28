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

Measured on 2026-09-28 on an AMD Ryzen 9 7950X3D (16 cores, 32 logical
CPUs), Linux 7.0, with PyCATSHOO 1.4.1.0 (models compiled in C++ against
its API), RAICHU 0.63.4, and Open MPI 5.0.8 for PyCATSHOO's parallel runs.
The raw results are committed in
`benchmarks/results/jaquet-2026-09-28.json`, and the charts are drawn from
that file by `docs/figures/benchmark_performance.py`.

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

## Replicas, one worker

![Wall clock against the number of replicas, one worker, on the three models](../assets/figures/benchmark-replicas-light.svg#only-light){ .figure }
![Wall clock against the number of replicas, one worker, on the three models](../assets/figures/benchmark-replicas-dark.svg#only-dark){ .figure }

| model | replicas | PyCATSHOO C++ | PyCATSHOO Python | RAICHU | ratio |
|---|---:|---:|---:|---:|---:|
| `pure_exp` | 100 000 | 0.668 s | | 0.109 s | 6.1 |
| `pure_exp` | 1 000 000 | 6.700 s | | 1.285 s | 5.2 |
| `heaters_s1` | 100 000 | 0.697 s | | 0.156 s | 4.5 |
| `heated_room_s3` | 2 000 | 1.537 s | 34.86 s | 0.230 s | 6.7 |
| `heated_room_s3` | 20 000 | 15.90 s | | 2.232 s | 7.1 |

Both engines are linear in the number of replicas, so the ratio hardly
moves along a sweep: 4 to 9 on the discrete models, 7 on the hybrid one
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
| `pure_exp`, 1 000 000 replicas | 1 | 6.69 s | 1.28 s |
| | 4 | 2.05 s | 0.73 s |
| | 16 | 1.06 s | 0.58 s |
| | 32 | 1.10 s | 0.58 s |
| `heated_room_s3`, 20 000 replicas | 1 | 15.32 s | 2.23 s |
| | 4 | 5.05 s | 0.58 s |
| | 16 | 2.06 s | 0.17 s |
| | 32 | 2.11 s | 0.14 s |

What the curves show:

- **On the hybrid model RAICHU's threads scale well**: 13 times faster
  on 16 threads than on one (the 16 physical cores), and 15.5 times on
  32.
- **On `pure_exp` they do not.** A replica there costs about a
  microsecond, so the work shared by all the replicas, the reduction of
  their results, dominates past four threads: 2.2 times faster at best.
  PyCATSHOO's ranks scale better on this model in relative terms (6.3
  times), while staying slower in absolute terms.
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
