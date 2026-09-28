"""Scaling campaign: RAICHU against PyCATSHOO at aligned accuracy, with more
measurement points than `run_bench.py`, for the charts of the documentation's
Performance page.

Four measurements, each written into one JSON results file:

1. **accuracy-cost**: on `heated_room_s3`, a grid of integration settings on
   each engine. For every setting, the achieved accuracy (max temperature
   error of the deterministic thermostat cycle against its closed form, as
   in `parity_experiment.py`) and the cost of the stochastic campaign.
2. **aligned settings**: from that grid, the pair of settings at which the
   two engines reach the same accuracy. Every hybrid timing below uses it.
3. **replicas**: wall clock against the number of replicas, one worker, on
   the three models (PyCATSHOO in C++, PyCATSHOO with Python callbacks on
   the hybrid model, RAICHU).
4. **workers**: wall clock against the number of workers at a fixed
   campaign: RAICHU threads against PyCATSHOO MPI ranks (`mpirun -n`).

Each timing is the best of REPEATS runs and covers the Monte-Carlo run only
(model construction and result extraction excluded), on both sides; under
MPI the timing is the first rank's, which includes gathering the ranks'
estimates. Every point also carries a consistency check: the MPI estimates
equal the single-process ones (the ranks share the replica split), and the
RAICHU estimates agree statistically with PyCATSHOO's.

Run on a quiet machine:

    PYCATSHOO_DIR=/path/to/pycatshoo-1.4.1.0 RAICHU_BENCH_PYTHON=python3.11 \\
        python campaign.py --out ../results/<machine>-<date>.json

`--quick` shrinks every campaign by 20 to check the plumbing in a minute.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pyraichu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity_experiment as parity
import run_bench

MODELS = HERE.parent / "models"
SEED = 56000
REPEATS = 3
T_MAX = 100.0
INSTANTS = [T_MAX * k / 10.0 for k in range(11)]

# (dt, dtCond): dt <= 0 keeps PyCATSHOO's default RK4 step.
PYC_ACCURACY_GRID = [
    (-1.0, 1e-2), (-1.0, 1e-3), (-1.0, 1e-4), (-1.0, 1e-5), (-1.0, 1e-6),
    (-1.0, 1e-8), (-1.0, 1e-10),
]
# RAICHU settings, from coarse to tight.
RAICHU_ACCURACY_GRID = [
    ("r1e-3", {"rtol": 1e-3, "atol": 1e-6, "tol_event": 1e-4, "max_step": 10.0, "sub_samples": 2}),
    ("r1e-4", {"rtol": 1e-4, "atol": 1e-7, "tol_event": 1e-6, "max_step": 5.0, "sub_samples": 4}),
    ("r1e-5", {"rtol": 1e-5, "atol": 1e-8, "tol_event": 1e-6, "max_step": 2.0, "sub_samples": 8}),
    ("r1e-6", {"rtol": 1e-6, "atol": 1e-9, "tol_event": 1e-6, "max_step": 1.0, "sub_samples": 8}),
    ("r1e-7", {"rtol": 1e-7, "atol": 1e-10, "tol_event": 1e-8, "max_step": 0.5, "sub_samples": 8}),
    ("r1e-8", {"rtol": 1e-8, "atol": 1e-11, "tol_event": 1e-10, "max_step": 0.2, "sub_samples": 16}),
    ("default", {}),
]

REPLICA_SWEEP = {
    "pure_exp": [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
    "heaters_s1": [1_000, 3_000, 10_000, 30_000, 100_000],
    "heated_room_s3": [500, 1_000, 2_000, 5_000, 10_000, 20_000],
}
PYC_PY_SWEEP = [500, 1_000, 2_000]
WORKERS = [1, 2, 4, 8, 12, 16, 24, 32]
WORKER_CAMPAIGNS = {"pure_exp": 1_000_000, "heated_room_s3": 20_000}


def pyc_bench(model: str, nb_runs: int, dt_: float = -1.0, dt_cond: float = -1.0,
              lam: float | None = None, ranks: int = 1, seed: int = SEED) -> dict:
    cmd = [str(HERE / "pyc_bench"), model, str(nb_runs), str(T_MAX), str(seed)]
    if model == "heated_room_s3":
        cmd += [str(dt_), str(dt_cond)] + ([str(lam)] if lam is not None else [])
    if ranks > 1:
        cmd = ["mpirun", "-n", str(ranks), "--bind-to", "none", "--oversubscribe"] + cmd
    started = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    process_s = time.perf_counter() - started
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {out.stderr[-2000:]}")
    result = json.loads(out.stdout)
    # The whole process, launch, model construction and MPI start-up
    # included: what a user waits for, beside the engine's own timing.
    result["process_s"] = process_s
    return result


def best_of(fn, *args, **kwargs) -> tuple[float, dict]:
    best, kept = math.inf, None
    for _ in range(REPEATS):
        result = fn(*args, **kwargs)
        if result["wall_clock_s"] < best:
            best, kept = result["wall_clock_s"], result
    return best, kept


def raichu_model(model: str):
    return pyraichu.load_model((MODELS / f"{model}.json").read_text())


def raichu_run(model, nb_runs: int, threads: int, ode: dict) -> dict:
    started = time.perf_counter()
    result = pyraichu.monte_carlo(model, nb_runs=nb_runs, t_max=T_MAX, samples=INSTANTS,
                                  seed=42, threads=threads, **ode)
    wall = time.perf_counter() - started
    return {"wall_clock_s": wall, "estimates": {
        name: {"mean": list(e.mean), "std": list(e.std)} for name, e in result.indicators.items()}}


def machine() -> dict:
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    commit = os.environ.get("RAICHU_BENCH_COMMIT") or subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
        check=False).stdout.strip()
    try:
        mpi = subprocess.run(["mpirun", "--version"], capture_output=True, text=True,
                             check=False).stdout.splitlines()
    except FileNotFoundError:
        mpi = []
    return {"host": platform.node(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "kernel": platform.release(), "python": platform.python_version(),
            "raichu": pyraichu.__version__, "pycatshoo": "1.4.1.0",
            "mpi": mpi[0] if mpi else None, "commit": commit or None,
            "date": dt.datetime.now(dt.timezone.utc).date().isoformat()}


def accuracy(scale: float, log) -> dict:
    nb = max(100, int(2_000 * scale))
    points = {"pycatshoo_cpp": [], "raichu_1t": []}
    exact = parity.exact_temperatures(INSTANTS)
    for dt_, dt_cond in PYC_ACCURACY_GRID:
        det = pyc_bench("heated_room_s3", 1, dt_, dt_cond, lam=0.0)
        err = max(abs(a - b) for a, b in zip(det["estimates"]["Room_temperature"]["mean"], exact))
        wall, _ = best_of(pyc_bench, "heated_room_s3", nb, dt_, dt_cond, lam=0.01)
        points["pycatshoo_cpp"].append({"dt": dt_, "dt_cond": dt_cond, "max_temp_err": err, "wall_s": wall})
        log(f"  pyc dtCond={dt_cond:.0e}: err {err:.2e}, {wall:.3f} s")
    for label, ode in RAICHU_ACCURACY_GRID:
        det = pyraichu.monte_carlo(parity.raichu_model(0.0), nb_runs=1, t_max=T_MAX, samples=INSTANTS,
                                   seed=42, threads=1, **ode)
        err = max(abs(a - b) for a, b in zip(det.indicators["Room_temperature"].mean, exact))
        model = parity.raichu_model(0.01)
        raichu_run(model, 50, 1, ode)
        wall, _ = best_of(raichu_run, model, nb, 1, ode)
        points["raichu_1t"].append({"label": label, **ode, "max_temp_err": err, "wall_s": wall})
        log(f"  raichu {label}: err {err:.2e}, {wall:.3f} s")
    return {"nb_runs": nb, "points": points}


def aligned(points: dict) -> dict:
    """The PyCATSHOO setting of the baseline (dtCond 1e-6) and the cheapest
    RAICHU setting at least as accurate."""
    pyc = next(p for p in points["pycatshoo_cpp"] if p["dt_cond"] == 1e-6)
    candidates = [p for p in points["raichu_1t"] if p["max_temp_err"] <= pyc["max_temp_err"]]
    rai = min(candidates, key=lambda p: p["wall_s"])
    ode = {k: rai[k] for k in ("rtol", "atol", "tol_event", "max_step", "sub_samples") if k in rai}
    return {"pycatshoo": {"dt": pyc["dt"], "dt_cond": pyc["dt_cond"], "max_temp_err": pyc["max_temp_err"]},
            "raichu": {"label": rai["label"], "ode": ode, "max_temp_err": rai["max_temp_err"]}}


def z_worst(pyc: dict, rai: dict, nb_runs: int) -> float:
    return run_bench.check_statistical(pyc, rai, nb_runs)


def replicas(scale: float, align: dict, log) -> list[dict]:
    rows = []
    ode = align["raichu"]["ode"]
    for model_name, sweep in REPLICA_SWEEP.items():
        model = raichu_model(model_name)
        hybrid = model_name == "heated_room_s3"
        for n in sweep:
            n = max(100, int(n * scale))
            args = (align["pycatshoo"]["dt"], align["pycatshoo"]["dt_cond"]) if hybrid else ()
            pw, pres = best_of(pyc_bench, model_name, n, *args)
            raichu_run(model, 50, 1, ode if hybrid else {})
            rw, rres = best_of(raichu_run, model, n, 1, ode if hybrid else {})
            row = {"model": model_name, "nb_runs": n, "pycatshoo_cpp_s": pw, "raichu_1t_s": rw,
                   "z_worst": z_worst(pres, rres, n)}
            rows.append(row)
            log(f"  {model_name} n={n}: pyc {pw:.3f} s, raichu {rw:.3f} s, z {row['z_worst']:.2f}")
    for n in PYC_PY_SWEEP:
        n = max(50, int(n * scale))
        lib = str(run_bench.pycatshoo_dir() / "Core" / "lib")
        wall, _ = best_of(run_bench.run_json,
            [run_bench.bench_python(), str(HERE / "bench_py.py"), "heated_room_s3", str(n), str(T_MAX),
             str(SEED)], {"PYTHONPATH": lib, "LD_LIBRARY_PATH": lib})
        rows.append({"model": "heated_room_s3", "nb_runs": n, "pycatshoo_python_s": wall})
        log(f"  heated_room_s3 n={n}: pyc python callbacks {wall:.3f} s")
    return rows


def workers(scale: float, align: dict, log) -> list[dict]:
    rows = []
    for model_name, n in WORKER_CAMPAIGNS.items():
        n = max(1000, int(n * scale))
        model = raichu_model(model_name)
        hybrid = model_name == "heated_room_s3"
        ode = align["raichu"]["ode"] if hybrid else {}
        args = (align["pycatshoo"]["dt"], align["pycatshoo"]["dt_cond"]) if hybrid else ()
        serial = None
        for w in WORKERS:
            pw, pres = best_of(pyc_bench, model_name, n, *args, ranks=w)
            if serial is None:
                serial = pres
            same = run_bench.check_identical(serial, pres)
            raichu_run(model, 50, w, ode)
            rw, rres = best_of(raichu_run, model, n, w, ode)
            rows.append({"model": model_name, "nb_runs": n, "workers": w, "pycatshoo_mpi_s": pw,
                         "pycatshoo_mpi_process_s": pres["process_s"],
                         "raichu_threads_s": rw, "mpi_vs_serial_max_diff": same,
                         "z_worst": z_worst(pres, rres, n)})
            log(f"  {model_name} w={w}: pyc mpi {pw:.3f} s (diff {same:.1e}), raichu {rw:.3f} s")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--parts", default="accuracy,replicas,workers",
                        help="comma-separated subset; the others are kept from --out if it exists")
    args = parser.parse_args()
    scale = 0.05 if args.quick else 1.0
    parts = set(args.parts.split(","))

    def log(message: str) -> None:
        print(message, flush=True)

    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    results["protocol"] = {"repeats": REPEATS, "seed": SEED, "t_max": T_MAX, "quick": args.quick}
    results.setdefault("machine", {})
    results["machine"].update({k: v for k, v in machine().items()
                               if v is not None or k not in results["machine"]})
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        out.write_text(json.dumps(results, indent=1) + "\n")

    if "accuracy" in parts:
        log("accuracy-cost")
        results["accuracy"] = accuracy(scale, log)
        results["aligned"] = aligned(results["accuracy"]["points"])
        log(f"aligned: {json.dumps(results['aligned'])}")
        save()
    if "replicas" in parts:
        log("replicas")
        results["replicas"] = replicas(scale, results["aligned"], log)
        save()
    if "workers" in parts:
        log("workers")
        results["workers"] = workers(scale, results["aligned"], log)
        save()
    log(f"written {out}")


if __name__ == "__main__":
    main()
