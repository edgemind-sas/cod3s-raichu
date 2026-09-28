"""Benchmarks: the charts of the Performance and Accuracy-cost pages.

Unlike the other figure scripts, this one runs nothing: it draws a results
file written by `benchmarks/pycatshoo-cpp/campaign.py` on a quiet machine,
committed under `benchmarks/results/`. Timings are facts about one machine
on one day, so the file, not a rerun, is the source.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from _style import Theme, render

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results"
#: The campaign the pages report. Override with RAICHU_BENCH_RESULTS.
CAMPAIGN = "jaquet-2026-09-28.json"

MODELS = {
    "pure_exp": "pure_exp (discrete)",
    "heaters_s1": "heaters_s1 (discrete)",
    "heated_room_s3": "heated_room_s3 (hybrid)",
}


def spaced(n: int) -> str:
    """12 345 rather than 12,345 or 12_345."""
    return f"{n:,}".replace(",", " ")


def load() -> dict:
    path = Path(os.environ.get("RAICHU_BENCH_RESULTS", RESULTS / CAMPAIGN))
    return json.loads(path.read_text())


def caption(data: dict) -> str:
    m = data["machine"]
    return (f"{m['cpu'] or m['host']}, {m['logical_cpus']} logical CPUs, RAICHU {m['raichu']}, "
            f"PyCATSHOO {m['pycatshoo']}, {m['date']}")


def accuracy_cost(data: dict) -> None:
    points = data["accuracy"]["points"]
    aligned = data["aligned"]

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        pyc = sorted(points["pycatshoo_cpp"], key=lambda p: p["max_temp_err"])
        rai = sorted(points["raichu_1t"], key=lambda p: p["max_temp_err"])
        ax.plot([p["max_temp_err"] for p in pyc], [p["wall_s"] for p in pyc], "o-",
                color=theme.series[1], label="PyCATSHOO C++, dtCond from 1e-2 to 1e-10")
        ax.plot([p["max_temp_err"] for p in rai], [p["wall_s"] for p in rai], "s-",
                color=theme.series[0], label="RAICHU 1 thread, rtol from 1e-3 to 1e-9")
        pyc_aligned = next(p for p in pyc if p["dt_cond"] == aligned["pycatshoo"]["dt_cond"])
        rai_aligned = next(p for p in rai if p["label"] == aligned["raichu"]["label"])
        for p in (pyc_aligned, rai_aligned):
            ax.plot(p["max_temp_err"], p["wall_s"], "o", markersize=13, markerfacecolor="none",
                    markeredgecolor=theme.text, markeredgewidth=1.2)
        ax.annotate("aligned pair", (rai_aligned["max_temp_err"], rai_aligned["wall_s"]),
                    textcoords="offset points", xytext=(10, -14), fontsize=9, color=theme.text)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("achieved max temperature error (more accurate to the right)")
        ax.set_ylabel(f"wall clock, {spaced(data['accuracy']['nb_runs'])} replicas (s)")
        ax.legend(loc="upper left", fontsize=9)
        fig.suptitle(caption(data), x=0.01, ha="left", fontsize=7, color=theme.text, alpha=0.8)

    render("benchmark-accuracy-cost", draw, size=(7.0, 4.2))


def replicas(data: dict) -> None:
    rows = data["replicas"]

    def draw(fig, theme: Theme) -> None:
        axes = fig.subplots(1, 3)
        for ax, model in zip(axes, MODELS):
            mine = [r for r in rows if r["model"] == model and "raichu_1t_s" in r]
            n = [r["nb_runs"] for r in mine]
            ax.plot(n, [r["pycatshoo_cpp_s"] for r in mine], "o-", color=theme.series[1],
                    label="PyCATSHOO C++")
            ax.plot(n, [r["raichu_1t_s"] for r in mine], "s-", color=theme.series[0],
                    label="RAICHU")
            python = [r for r in rows if r["model"] == model and "pycatshoo_python_s" in r]
            if python:
                ax.plot([r["nb_runs"] for r in python], [r["pycatshoo_python_s"] for r in python],
                        "^--", color=theme.series[4], label="PyCATSHOO, Python callbacks")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(MODELS[model], fontsize=9)
            ax.set_xlabel("replicas")
        axes[0].set_ylabel("wall clock, one worker (s)")
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
        fig.suptitle(caption(data), x=0.01, ha="left", fontsize=7, color=theme.text, alpha=0.8)

    render("benchmark-replicas", draw, size=(7.6, 3.6))


def workers(data: dict) -> None:
    rows = data["workers"]

    def draw(fig, theme: Theme) -> None:
        axes = fig.subplots(1, 2)
        for ax, model in zip(axes, ("pure_exp", "heated_room_s3")):
            mine = sorted((r for r in rows if r["model"] == model), key=lambda r: r["workers"])
            w = [r["workers"] for r in mine]
            ax.plot(w, [r["pycatshoo_mpi_s"] for r in mine], "o-", color=theme.series[1],
                    label="PyCATSHOO, MPI ranks")
            ax.plot(w, [r["raichu_threads_s"] for r in mine], "s-", color=theme.series[0],
                    label="RAICHU, threads")
            for key, colour in (("pycatshoo_mpi_s", theme.series[1]), ("raichu_threads_s", theme.series[0])):
                base = mine[0][key]
                ax.plot(w, [base / k for k in w], ":", color=colour, linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(w, [str(k) for k in w])
            ax.set_title(f"{MODELS[model]}, {spaced(mine[0]['nb_runs'])} replicas", fontsize=9)
            ax.set_xlabel("workers")
        axes[0].set_ylabel("wall clock (s); dotted: ideal scaling")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
        fig.suptitle(caption(data), x=0.01, ha="left", fontsize=7, color=theme.text, alpha=0.8)

    render("benchmark-workers", draw, size=(7.6, 3.8))


def main() -> None:
    data = load()
    accuracy_cost(data)
    replicas(data)
    workers(data)


if __name__ == "__main__":
    main()
