"""Examples: the hydrogen installation.

Runs the Python blocks of ``docs/examples/h2-installation.md`` (the page is
the single definition of the model), then draws three charts: one
trajectory around a loss of supply, the unavailability over one year, and
the minimal sequences leading to the loss of supply.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pyraichu
from _style import Theme, render

PAGE = Path(__file__).resolve().parents[1] / "examples" / "h2-installation.md"
FENCE = re.compile(
    r"(?P<skip><!--\s*(?P<marker>skip|model)\s*-->\n)?"
    r"```(?P<lang>python|json)\b[^\n]*\n(?P<body>.*?)\n```",
    re.DOTALL,
)

YEAR = 8760.0
NB_RUNS = 1000
SEED = 1


def page_model() -> pyraichu.Model:
    """The model the page builds, by running the page's model blocks."""
    namespace: dict = {}
    for match in FENCE.finditer(PAGE.read_text()):
        if match.group("lang") != "python" or match.group("marker") == "skip":
            continue
        body = match.group("body")
        if "pyraichu.simulate" in body or "pyraichu.monte_carlo" in body:
            continue  # the page's quick looks; the campaigns below replace them
        exec(compile(body, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace["model"]


def wilson(k: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a proportion k / n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def trajectory_chart(model: pyraichu.Model) -> None:
    """A trajectory in which an electrolyser failure ends in a loss of supply."""
    for seed in range(1, 200):
        probe = pyraichu.simulate(model, t_max=YEAR, seed=seed)
        trips = [e for e in probe.events if e.transition == "USER.supply.trip"]
        if not trips:
            continue
        trip = trips[0].time
        fails = [e for e in probe.events
                 if e.transition == "ELEC.health.fail" and e.time < trip]
        others = [e for e in probe.events if e.transition.endswith("health.fail")
                  and not e.transition.startswith("ELEC") and trip - 400 < e.time < trip]
        if fails and trip - fails[-1].time < 12 * 24 and not others:
            break
    start = max(0.0, fails[-1].time - 2 * 24.0)
    start = 24.0 * math.floor(start / 24.0)
    end = trip + 4 * 24.0
    step = 0.25
    samples = [start + step * k for k in range(int((end - start) / step) + 1)]
    run = pyraichu.simulate(model, t_max=end, seed=seed, samples=samples)
    days = [(t - start) / 24.0 for t in samples]

    def series(name):
        return [v for _, v in run.samples[name]]

    marks = [(e.time, e.transition) for e in run.events
             if start <= e.time <= end and (
                 "health" in e.transition or e.transition.startswith("USER."))]
    labels = {
        "ELEC.health.fail": "electrolyser fails", "ELEC.health.repair": "electrolyser repaired",
        "USER.supply.trip": "user unsupplied", "USER.supply.restart": "supplied again",
    }

    def draw(fig, theme: Theme) -> None:
        axes = fig.subplots(3, 1, sharex=True)
        axes[0].plot(days, series("pv"), color=theme.series[1], label="PV power")
        axes[0].plot(days, series("electrolyser"), color=theme.series[0],
                     label="electrolyser draw")
        axes[0].set_ylabel("kW")
        fig.legend(loc="outside lower center", ncols=2)
        axes[1].plot(days, series("battery"), color=theme.series[2])
        axes[1].set_ylabel("battery\nkWh")
        axes[2].plot(days, series("storage"), color=theme.series[0])
        axes[2].axhline(1.0, color=theme.series[1], linestyle=":", linewidth=1)
        axes[2].set_ylabel("storage\nkg")
        axes[2].set_xlabel(f"days (from t = {start:.0f} h)")
        for t, name in marks:
            x = (t - start) / 24.0
            for ax in axes:
                ax.axvline(x, color=theme.text, linestyle="--", linewidth=0.8, alpha=0.6)
            if name in labels:
                axes[0].annotate(labels[name], (x, 1.02), xycoords=("data", "axes fraction"),
                                 rotation=90, va="bottom", ha="center", fontsize=9)
        axes[2].set_xlim(0, days[-1])

    render("example-h2-trajectory", draw, size=(7.0, 6.6))
    print(f"trajectory: seed {seed}, electrolyser fails at {fails[-1].time:.1f} h, "
          f"user unsupplied at {trip:.1f} h")


def unavailability_chart(model: pyraichu.Model) -> None:
    weeks = [7 * 24.0 * k for k in range(53)] + [YEAR]
    est = pyraichu.monte_carlo(model, nb_runs=NB_RUNS, t_max=YEAR, samples=weeks, seed=SEED)
    u = est.indicators["unsupplied"]
    days = [t / 24.0 for t in u.instants]

    def draw(fig, theme: Theme) -> None:
        left, right = fig.subplots(1, 2)
        left.fill_between(days, u.ci.low, u.ci.high, color=theme.band, linewidth=0,
                          label=f"{u.ci.level:.0%} band")
        left.plot(days, u.mean, color=theme.series[0], label="estimate")
        left.set_title("unsupplied at t")
        left.set_xlabel("days")
        left.set_ylabel("probability")
        left.set_ylim(0, None)
        left.legend(loc="upper left")
        right.fill_between(days, u.reached_ci.low, u.reached_ci.high, color=theme.band,
                           linewidth=0)
        right.plot(days, u.reached_mean, color=theme.series[0])
        right.set_title("unsupplied at least once by t")
        right.set_xlabel("days")
        right.set_ylim(0, None)

    render("example-h2-unavailability", draw, size=(7.0, 3.4))
    mean_u = sum(u.mean[1:]) / len(u.mean[1:])
    print(f"unavailability: mean over the weekly instants {mean_u:.4f}; "
          f"P(lost at least once by one year) = {u.reached_mean[-1]:.3f} "
          f"[{u.reached_ci.low[-1]:.3f}, {u.reached_ci.high[-1]:.3f}]")


def sequences_chart(model: pyraichu.Model) -> None:
    sequences = pyraichu.analyse_sequences(model, nb_runs=NB_RUNS, t_max=YEAR, seed=SEED)
    reaching = [q for q in sequences if q["end_cause"] is not None]
    reaching.sort(key=lambda q: q["weight"])
    names = {"ELEC": "electrolyser", "PV": "PV inverter", "BATT": "battery",
             "COMP_A": "compressor A", "COMP_B": "compressor B"}
    labels = [" → ".join(names.get(e["obj"], e["obj"]) for e in q["events"]) for q in reaching]
    shares = [q["weight"] / NB_RUNS for q in reaching]
    bounds = [wilson(q["weight"], NB_RUNS) for q in reaching]

    def draw(fig, theme: Theme) -> None:
        ax = fig.add_subplot()
        y = range(len(labels))
        ax.barh(y, shares, color=theme.series[0], height=0.6)
        ax.errorbar(shares, y, xerr=[[s - lo for s, (lo, _) in zip(shares, bounds)],
                                     [hi - s for s, (_, hi) in zip(shares, bounds)]],
                    fmt="none", ecolor=theme.series[1], capsize=3, linewidth=1)
        ax.set_yticks(list(y), labels)
        ax.set_xlabel(f"share of {NB_RUNS} one-year trajectories (95 % Wilson interval)")
        ax.grid(axis="y", visible=False)
        for yy, s in zip(y, shares):
            ax.annotate(f"{s:.3f}", (s, yy), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, va="bottom")

    render("example-h2-sequences", draw, size=(7.0, 2.6))
    total = sum(q["weight"] for q in reaching)
    print("sequences:", [(label, q["weight"]) for label, q in zip(labels, reaching)],
          f"total reaching {total:.0f} / {NB_RUNS}")


def main() -> None:
    model = page_model()
    trajectory_chart(model)
    unavailability_chart(model)
    sequences_chart(model)


if __name__ == "__main__":
    main()
