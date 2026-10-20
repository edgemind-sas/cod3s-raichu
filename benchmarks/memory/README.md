# Monte-Carlo sample memory

`sample_chunks.py` runs a seeded, pure exponential model with 165 state
indicators at 101 instants, four worker threads and no quantiles. Each invocation
starts a fresh process. It reports two different quantities:

- Calculated retained sample payload for the largest chunk: 40 bytes per sample
  (five `f64` fields), multiplied by the grid and declared chunk replica count.
- Observed peak process RSS up to the return of `monte_carlo`, including the
  interpreter, loaded libraries, compiled model, worker allocations and estimates.
  RSS is read before the extra serialization used to fingerprint the estimates.

The payload calculation is not an allocation measurement. `--chunk-replicas`
describes the policy established by source inspection and the Rust `chunk_tests`;
it does not set an engine option. Version 0.103.5 uses 1 024 for this grid;
0.103.6 uses 100. This explicit input makes historical comparisons possible
without exposing a public engine memory setting. Update it if the policy changes.

Build and install the binding from the checkout under measurement, then run on
Linux or macOS (Python's `resource` module supplies the peak RSS):

```bash
.venv/bin/python benchmarks/memory/sample_chunks.py --chunk-replicas 100 --runs 1024
.venv/bin/python benchmarks/memory/sample_chunks.py --chunk-replicas 100 --runs 4096
```

For the old engine use the same script with `--chunk-replicas 1024`. Compare
`estimates_sha256` between versions at the same run count. This checks all
indicator statistics, excluding engine-version provenance. Larger campaigns
should keep the same payload bound. RSS is observational: allocator retention,
thread stacks and platform differences prevent treating 64 MiB as an RSS limit.
Quantile storage and an individually oversized replica have separate memory
requirements, as described in the [parallelism guide](../../docs/guides/parallelism.md).

## Measurement, 2026-10-20

Linux x86_64, Python 3.11.11, release binding, four threads, seed 42,
no quantiles. Each row is a fresh process using the commands above.

| Version | Replicas | Chunk replicas | Calculated sample payload (MiB) | Observed peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 0.103.5 | 1024 | 1024 | 650.98 | 686.61 |
| 0.103.6 | 1024 | 100 | 63.57 | 100.14 |
| 0.103.5 | 4096 | 1024 | 650.98 | 686.88 |
| 0.103.6 | 4096 | 100 | 63.57 | 106.39 |

All indicator estimates have identical SHA-256 fingerprints between versions
at each campaign size. The 64 MiB payload budget now holds, while process RSS
remains larger for the documented reasons. The host was under concurrent load
during the corrected runs, so elapsed times are not compared. These are memory
observations on this host, not portable RSS thresholds or throughput guarantees.
