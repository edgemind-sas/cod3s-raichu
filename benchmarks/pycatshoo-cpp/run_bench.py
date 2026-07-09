"""Side-by-side Monte-Carlo benchmark: PyCATSHOO C++-native vs
PyCATSHOO Python-callback vs RAICHU — the strict apples-to-apples
comparison of models written and compiled in C++ against PyCATSHOO's
own API versus the same models in RAICHU.

Engines and modelling paths measured on the same models:
  - pyc-cpp   : PyCATSHOO 1.4.1.0, model fully in C++ (bench.cpp),
                zero interpreter crossing in the hot loop;
  - pyc-py    : PyCATSHOO 1.4.1.0, same model with Python callbacks
                (bench_py.py) — the normal PyCATSHOO modelling style;
  - raichu 1t : RAICHU single-thread (engine vs engine);
  - raichu Nt : RAICHU with its default thread pool.

Consistency gate: at the shared seed, pyc-cpp and pyc-py must return
identical estimates (they share every random draw); RAICHU estimates
must agree statistically (|Δmean| within z_crit standard errors).

Wall-clock covers the Monte-Carlo run only (model construction and
result extraction excluded) on every side; best of REPEATS runs.

Run:
  python3 run_bench.py                 # all engines (needs PyCATSHOO — see README)
  python3 run_bench.py --raichu-only   # RAICHU side only (no PyCATSHOO required)

Environment (see README.md):
  PYCATSHOO_DIR   root of a PyCATSHOO 1.4.x install (the C++/Python sides).
                  Must be a 1.4.x tree whose headers match its binaries;
                  see the ABI-trap note in the README. Not needed for
                  --raichu-only.
  RAICHU_BENCH_PYTHON  interpreter matching the PyCATSHOO Python module
                  (3.11 for 1.4.1.0); defaults to `python3.11` on PATH.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"


def pycatshoo_dir() -> Path:
    """Root of the PyCATSHOO 1.4.x install used for the PyCATSHOO side.

    Read from ``PYCATSHOO_DIR``; the benchmark never hard-codes a path.
    """
    root = os.environ.get("PYCATSHOO_DIR")
    if not root:
        raise SystemExit(
            "PYCATSHOO_DIR is not set. Point it at a PyCATSHOO 1.4.x install "
            "(see benchmarks/pycatshoo-cpp/README.md), or run with "
            "--raichu-only to skip the PyCATSHOO side."
        )
    return Path(root)


def bench_python() -> str:
    """Interpreter matching the PyCATSHOO Python module (3.11 for 1.4.1.0)."""
    explicit = os.environ.get("RAICHU_BENCH_PYTHON")
    if explicit:
        return explicit
    found = shutil.which("python3.11")
    if not found:
        raise SystemExit(
            "No python3.11 found for the PyCATSHOO Python side. Set "
            "RAICHU_BENCH_PYTHON, or run with --raichu-only."
        )
    return found

# (model, t_max, nb_runs) — scales of perf_bench.py CASES.
CASES = [
    ("pure_exp", 100.0, 10_000),
    ("pure_exp", 100.0, 100_000),
    ("heaters_s1", 100.0, 10_000),
    ("heated_room_s3", 100.0, 2_000),
]
SEED = 56000
REPEATS = 3


def run_json(cmd: list[str], env: dict | None = None) -> dict:
    import os

    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    out = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd}: {out.stderr[-2000:]}")
    return json.loads(out.stdout)


def pyc_cpp(model: str, t_max: float, nb_runs: int) -> dict:
    best = None
    for _ in range(REPEATS):
        res = run_json(
            [str(HERE / "pyc_bench"), model, str(nb_runs), str(t_max), str(SEED)]
        )
        if best is None or res["wall_clock_s"] < best["wall_clock_s"]:
            best = res
    return best


def pyc_py(model: str, t_max: float, nb_runs: int) -> dict:
    lib = str(pycatshoo_dir() / "Core" / "lib")
    env = {"PYTHONPATH": lib, "LD_LIBRARY_PATH": lib}
    best = None
    for _ in range(REPEATS):
        res = run_json(
            [bench_python(), str(HERE / "bench_py.py"), model, str(nb_runs),
             str(t_max), str(SEED)],
            env,
        )
        if best is None or res["wall_clock_s"] < best["wall_clock_s"]:
            best = res
    return best


def raichu(model: str, t_max: float, nb_runs: int, threads: int | None) -> dict:
    import pyraichu

    m = pyraichu.load_model((MODELS / f"{model}.json").read_text())
    samples = [t_max * k / 10.0 for k in range(11)]
    pyraichu.monte_carlo(
        m, nb_runs=100, t_max=t_max, samples=samples, seed=1, threads=threads
    )
    best = None
    result = None
    for _ in range(REPEATS):
        started = time.perf_counter()
        result = pyraichu.monte_carlo(
            m, nb_runs=nb_runs, t_max=t_max, samples=samples, seed=42,
            threads=threads,
        )
        wall = time.perf_counter() - started
        if best is None or wall < best:
            best = wall
    estimates = {
        name: {"mean": list(est.mean), "std": list(est.std)}
        for name, est in result.indicators.items()
    }
    return {"wall_clock_s": best, "estimates": estimates, "instants": samples}


def check_identical(a: dict, b: dict, tol: float = 1e-6) -> float:
    worst = 0.0
    for name, est in a["estimates"].items():
        for x, y in zip(est["mean"], b["estimates"][name]["mean"]):
            worst = max(worst, abs(x - y))
    return worst


def check_statistical(pyc: dict, rai: dict, nb_runs: int) -> float:
    """Worst z-score across indicators/instants (indicator names are
    matched case-insensitively: RAICHU's pure_exp fixture says C_ko,
    the PyCATSHOO side C_KO)."""
    rai_by_key = {name.lower(): est for name, est in rai["estimates"].items()}
    worst = 0.0
    for name, est in pyc["estimates"].items():
        other = rai_by_key.get(name.lower())
        if other is None:
            continue
        for x, sx, y, sy in zip(
            est["mean"], est["std"], other["mean"], other["std"],
        ):
            se = math.sqrt((sx * sx + sy * sy) / nb_runs)
            if se > 0:
                worst = max(worst, abs(x - y) / se)
    return worst


def main() -> None:
    raichu_only = "--raichu-only" in sys.argv[1:]
    rows = []
    for model, t_max, nb_runs in CASES:
        r1 = raichu(model, t_max, nb_runs, threads=1)
        rn = raichu(model, t_max, nb_runs, threads=None)
        if raichu_only:
            rows.append((model, nb_runs, None, None, r1, rn, None, None))
            print(
                f"{model:>16} n={nb_runs:<7} "
                f"raichu1t={r1['wall_clock_s']:8.3f}s  "
                f"raichuNt={rn['wall_clock_s']:8.3f}s  (RAICHU-only)",
                flush=True,
            )
            continue
        cpp = pyc_cpp(model, t_max, nb_runs)
        py = pyc_py(model, t_max, nb_runs)
        d_cpp_py = check_identical(cpp, py)
        z = check_statistical(cpp, r1, nb_runs)
        rows.append((model, nb_runs, cpp, py, r1, rn, d_cpp_py, z))
        print(
            f"{model:>16} n={nb_runs:<7} "
            f"cpp={cpp['wall_clock_s']:8.3f}s  py={py['wall_clock_s']:8.3f}s "
            f"(x{py['wall_clock_s'] / cpp['wall_clock_s']:5.1f})  "
            f"raichu1t={r1['wall_clock_s']:8.3f}s "
            f"(cpp/r1 x{cpp['wall_clock_s'] / r1['wall_clock_s']:5.2f})  "
            f"raichuNt={rn['wall_clock_s']:8.3f}s  "
            f"|d|cpp-py={d_cpp_py:.1e}  zmax(raichu)={z:.2f}",
            flush=True,
        )
    out = HERE / ("results-raichu-only.json" if raichu_only else "results.json")
    out.write_text(
        json.dumps(
            [
                {
                    "model": model,
                    "nb_runs": nb_runs,
                    "pyc_cpp_s": cpp["wall_clock_s"] if cpp else None,
                    "pyc_py_s": py["wall_clock_s"] if py else None,
                    "raichu_1t_s": r1["wall_clock_s"],
                    "raichu_nt_s": rn["wall_clock_s"],
                    "max_abs_diff_cpp_py": d,
                    "max_z_cpp_raichu": z,
                    "pyc_cpp_estimates": cpp["estimates"] if cpp else None,
                    "raichu_estimates": r1["estimates"],
                }
                for (model, nb_runs, cpp, py, r1, rn, d, z) in rows
            ],
            indent=2,
        )
    )
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
