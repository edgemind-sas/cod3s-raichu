"""Tolerance-parity experiment on `heated_room_s3` (the
"Accuracy–cost parity" benchmark page of the documentation site).

The engine-vs-engine ×2.7 in PyCATSHOO's favour was measured at very
different numerical effort. This script measures, for a grid of effort
settings on both engines:

- **achieved accuracy** — on the deterministic declination (lambda = 0
  the model is a pure thermostat cycle whose piecewise-linear ODE has a
  closed form: heating T(t) = 63 − (63−T0)e^(−t/10), cooling
  T(t) = 13 + (T0−13)e^(−t/10), switch dates by log formulas), the max
  absolute temperature error over the 11 schedule instants of a single
  trajectory vs the exact solution;
- **cost** — wall-clock of the stochastic Monte-Carlo run (2000
  replicas, seed 56000/42, best of 3), same protocol as run_bench.py.

Comparing wall-clocks *at comparable achieved accuracy* is the fair
version of the hybrid engine-vs-engine number.

Run: ../../.venv/bin/python parity_experiment.py   (→ parity_results.json)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
FIXTURE = MODELS / "heated_room_s3.json"

T_MAX = 100.0
NB_RUNS = 2000
SEED = 56000
INSTANTS = [T_MAX * k / 10.0 for k in range(11)]
REPEATS = 3

# Model constants transcribed from the vendored heated_room_s3 model
# (benchmarks/models/): leakage 0.1, T_out 13, T_init 17, power 5.
K = 0.1  # leakage rate
T_OUT = 13.0
T_INIT = 17.0
POWER = 5.0
T_LO, T_HI = 15.0, 20.0


def exact_temperatures(instants: list[float]) -> list[float]:
    """Exact trajectory of the deterministic thermostat cycle.

    Phases alternate heating (equilibrium T_OUT + POWER/K = 63) and
    cooling (equilibrium T_OUT); each phase is a first-order response
    with time constant 1/K, so switch dates are exact log expressions.
    """
    t_eq_heat = T_OUT + POWER / K

    def heat(t0: float, dt: float) -> float:
        return t_eq_heat - (t_eq_heat - t0) * math.exp(-K * dt)

    def cool(t0: float, dt: float) -> float:
        return T_OUT + (t0 - T_OUT) * math.exp(-K * dt)

    values = []
    for instant in sorted(instants):
        # Walk the cycle from t = 0 (heating, T_INIT) to `instant`.
        t, temp, heating = 0.0, T_INIT, True
        while True:
            if heating:
                t_switch = t + math.log((t_eq_heat - temp) / (t_eq_heat - T_HI)) / K
            else:
                t_switch = t + math.log((temp - T_OUT) / (T_LO - T_OUT)) / K
            if t_switch >= instant:
                dt = instant - t
                values.append(heat(temp, dt) if heating else cool(temp, dt))
                break
            temp, t, heating = (T_HI, t_switch, False) if heating else (
                T_LO,
                t_switch,
                True,
            )
    return values


# --- PyCATSHOO side ---------------------------------------------------------


def pyc_run(nb_runs: int, seed: int, dt: float, dt_cond: float,
            lam: float) -> dict:
    out = subprocess.run(
        [str(HERE / "pyc_bench"), "heated_room_s3", str(nb_runs), str(T_MAX),
         str(seed), str(dt), str(dt_cond), str(lam)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def pyc_point(dt: float, dt_cond: float) -> dict:
    det = pyc_run(1, SEED, dt, dt_cond, lam=0.0)
    exact = exact_temperatures(det["instants"])
    err = max(
        abs(a - b)
        for a, b in zip(det["estimates"]["Room_temperature"]["mean"], exact)
    )
    wall = min(
        pyc_run(NB_RUNS, SEED, dt, dt_cond, lam=0.01)["wall_clock_s"]
        for _ in range(REPEATS)
    )
    return {"dt": dt, "dt_cond": dt_cond, "max_temp_err": err, "wall_s": wall}


# --- RAICHU side ------------------------------------------------------------


def raichu_model(lam: float):
    import pyraichu

    spec = json.loads(FIXTURE.read_text())
    if lam == 0.0:
        # Deterministic declination: drop the failure transitions
        # entirely (RAICHU's build-time validation rejects a zero
        # exponential rate — typed error, by design).
        for component in spec["components"]:
            for automaton in component.get("automata", []):
                automaton["transitions"] = [
                    t for t in automaton["transitions"]
                    if t.get("name") != "OK_to_KO"
                ]
    return pyraichu.load_model(json.dumps(spec))


def raichu_point(label: str, ode: dict) -> dict:
    import pyraichu

    det = pyraichu.monte_carlo(
        raichu_model(0.0), nb_runs=1, t_max=T_MAX, samples=INSTANTS, seed=42,
        threads=1, **ode,
    )
    exact = exact_temperatures(INSTANTS)
    err = max(
        abs(a - b)
        for a, b in zip(det.indicators["Room_temperature"].mean, exact)
    )
    model = raichu_model(0.01)
    pyraichu.monte_carlo(
        model, nb_runs=100, t_max=T_MAX, samples=INSTANTS, seed=1, threads=1,
        **ode,
    )
    wall = math.inf
    for _ in range(REPEATS):
        started = time.perf_counter()
        pyraichu.monte_carlo(
            model, nb_runs=NB_RUNS, t_max=T_MAX, samples=INSTANTS, seed=42,
            threads=1, **ode,
        )
        wall = min(wall, time.perf_counter() - started)
    return {"label": label, **ode, "max_temp_err": err, "wall_s": wall}


PYC_GRID = [
    # (dt, dtCond): dt <= 0 keeps PyCATSHOO's default RK4 step (0.01).
    # A dt = 1e-4 point was measured once: identical error (the float32
    # indicator storage floors the probe at ~8e-7) for ×100 the cost
    # (201.8 s) — fixed-step refinement buys nothing on this smooth ODE.
    (-1.0, 1e-3),   # PyCATSHOO's own documented defaults
    (-1.0, 1e-6),   # the bench/oracle baseline
    (-1.0, 1e-10),  # event location at RAICHU's tightness
    (1e-3, 1e-10),
]

RAICHU_GRID = [
    ("default", {}),
    ("relaxed", {"rtol": 1e-6, "atol": 1e-9, "tol_event": 1e-6,
                  "max_step": 1.0, "sub_samples": 8}),
    ("loose", {"rtol": 1e-4, "atol": 1e-7, "tol_event": 1e-6,
                "max_step": 5.0, "sub_samples": 4}),
    ("floor", {"rtol": 1e-3, "atol": 1e-6, "tol_event": 1e-4,
                "max_step": 10.0, "sub_samples": 2}),
]


def main() -> None:
    raichu_only = "--raichu-only" in sys.argv[1:]
    results = {"pycatshoo_cpp": [], "raichu_1t": []}
    print(f"exact reference computed; {NB_RUNS} runs, t_max={T_MAX}\n")
    if not raichu_only:
        print("PyCATSHOO C++ (dt, dtCond) → max |ΔT| (det), wall (stoch):")
        for dt, dt_cond in PYC_GRID:
            point = pyc_point(dt, dt_cond)
            results["pycatshoo_cpp"].append(point)
            print(f"  dt={dt:>7} dtCond={dt_cond:>7.0e}  "
                  f"err={point['max_temp_err']:.3e}  wall={point['wall_s']:.3f}s",
                  flush=True)
    print("RAICHU 1 thread → max |ΔT| (det), wall (stoch):")
    for label, ode in RAICHU_GRID:
        point = raichu_point(label, ode)
        results["raichu_1t"].append(point)
        print(f"  {label:<8} {json.dumps(ode) if ode else '(defaults)':<80}  "
              f"err={point['max_temp_err']:.3e}  wall={point['wall_s']:.3f}s",
              flush=True)
    out = HERE / (
        "parity_results-raichu-only.json" if raichu_only else "parity_results.json"
    )
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
