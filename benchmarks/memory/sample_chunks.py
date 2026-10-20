"""Measure a wide Monte-Carlo grid in a fresh process (Linux/macOS).

Run this script separately for each campaign size. --chunk-replicas describes
its reviewed engine policy for the payload calculation; it does not configure
the engine. RSS is measured independently and is not a payload-budget assertion.
"""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from dataclasses import asdict

import pyraichu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-replicas", type=int, required=True)
    args = parser.parse_args()
    if min(args.runs, args.threads, args.chunk_replicas) < 1:
        parser.error("counts must be positive")

    model = pyraichu.load_model(
        {
            "name": "wide_sampling",
            "components": [
                {
                    "name": "C",
                    "automata": [
                        {
                            "name": "a",
                            "states": ["up", "down"],
                            "init": "up",
                            "transitions": [
                                {
                                    "name": "fail",
                                    "source": "up",
                                    "targets": ["down"],
                                    "distrib": "exp",
                                    "rate": 0.02,
                                }
                            ],
                        }
                    ],
                }
            ],
            "indicators": [
                {
                    "name": f"down_{i}",
                    "target": "state",
                    "component": "C",
                    "automaton": "a",
                    "state": "down",
                }
                for i in range(165)
            ],
        }
    )
    start = time.perf_counter()
    result = pyraichu.monte_carlo(
        model,
        nb_runs=args.runs,
        t_max=100.0,
        samples=list(range(101)),
        seed=42,
        threads=args.threads,
    )
    elapsed = time.perf_counter() - start
    # Read RSS before Python's serialization adds temporary result copies.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    indicators = {name: asdict(value) for name, value in result.indicators.items()}
    digest = hashlib.sha256(json.dumps(indicators, sort_keys=True).encode()).hexdigest()
    # Sample is five f64 fields. This is a calculated payload upper bound,
    # separate from RSS and from allocation/worker/estimate overhead.
    replica_bytes = 165 * 101 * 5 * 8
    print(
        json.dumps(
            {
                "engine_version": pyraichu.__version__,
                "platform": platform.platform(),
                "nb_runs": args.runs,
                "threads": args.threads,
                "quantiles": [],
                "indicators": 165,
                "instants": 101,
                "sample_bytes_per_replica": replica_bytes,
                "declared_chunk_replicas": args.chunk_replicas,
                "calculated_chunk_payload_bytes": min(args.runs, args.chunk_replicas)
                * replica_bytes,
                "observed_peak_rss_bytes": rss_bytes,
                "elapsed_seconds": elapsed,
                "estimates_sha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
