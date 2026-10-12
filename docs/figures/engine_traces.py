"""Record the traces replayed by the "How the engine works" pages.

The model is the one ``docs/engine/index.md`` defines in its executed python
blocks; this script runs those blocks itself (no matplotlib, unlike
``_style.py``, because the docs CI job installs only the wheel and pytest),
drives the real engine, and turns its causal journal into the steps the
player shows. ``python/tests/docs/test_engine_traces.py`` regenerates every
scenario and compares it with the committed files, so a change in the
engine's behaviour on this model fails the docs suite.

Trace format (private to the pages, not a RAICHU format)::

    {"scenario", "model", "seed", "horizon", "attributes", "automata", "steps"}

``attributes`` and ``automata`` list the qualified names in display order.
Each step has a ``kind``:

- ``stable``: ``time``, ``state`` (qualified attribute or automaton name ->
  value) and ``schedule`` (``[{transition, date, status}]``, status one of
  ``armed``, ``kept``, ``new``, ``rescheduled``, ``dropped``);
- ``race``: ``time``, ``armed`` (``[{transition, date}]``, earliest first),
  ``next_date`` and ``winner``;
- ``integrate`` (continuous scenario only): ``from``, ``to``, ``next_date``
  (the clock date integration was heading for, ``None`` if none),
  ``stopped_at_threshold`` and ``samples`` (``[[t, level]]``);
- ``branch``: ``time``, ``transition`` and ``options``
  (``[{label, to, scenario}]``), placed where the reader chooses;
- ``fired``: ``time``, ``transition``, ``from_state``, ``to_state`` and
  ``edge_writes``;
- ``iteration``: ``time``, ``index``, ``function`` and ``writes``
  (``[{attribute, old, new}]``).

Floats are rounded to 9 significant digits so that a regeneration is
byte-stable. Run directly (or through ``scripts/build_doc_figures.py``) to
rewrite ``docs/engine/traces/*.json``.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pyraichu

DOCS = Path(__file__).resolve().parents[1]
PAGE = DOCS / "engine" / "index.md"
OUT = DOCS / "engine" / "traces"

SEED = 1
HORIZON_DISCRETE = 40.0
HORIZON_CONTINUOUS = 60.0
SAMPLE_STEP = 0.25
BRANCH = "P2.mode.start"

# The block rule of python/tests/docs/test_doc_examples.py: every ```python
# block of the page runs, unless the line before it is <!-- skip -->.
_FENCE = re.compile(
    r"(?P<skip><!--\s*skip\s*-->\n)?```python\b[^\n]*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


@functools.cache
def page_models() -> dict:
    """Run the executed python blocks of the introduction page."""
    source = PAGE.read_text(encoding="utf-8")
    code = "\n".join(m["body"] for m in _FENCE.finditer(source) if not m["skip"])
    namespace: dict = {"__name__": "engine_page"}
    exec(compile(code, str(PAGE), "exec"), namespace)  # noqa: S102 (our own page)
    return namespace


def _round(value):
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        if set(value) == {"kind", "value"}:
            return _round(value["value"])
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def _initial_state(model) -> tuple[dict, list[str], list[str]]:
    document = json.loads(model.json)
    state, attributes, automata = {}, [], []
    for component in document["components"]:
        for attribute in component.get("attributes", []):
            name = f"{component['name']}.{attribute['name']}"
            state[name] = attribute["init"]["value"]
            attributes.append(name)
        for automaton in component.get("automata", []):
            name = f"{component['name']}.{automaton['name']}"
            state[name] = automaton["init"]
            automata.append(name)
    return state, attributes, automata


class _Replay:
    """Walks one journal and emits the player's steps."""

    def __init__(self, model, journal, samples=None, sampled_attribute=None, branch_at=None):
        self.state, self.attributes, self.automata = _initial_state(model)
        self.journal = journal
        self.samples = samples
        # The attribute the samples follow: an ODE target writes no journal
        # record, so its value at each event is read from the samples.
        self.sampled_attribute = sampled_attribute
        self.branch_at = branch_at
        self.pending: dict[str, float] = {}
        self.steps: list[dict] = []

    def _schedule_record(self, record, statuses):
        name = record["transition"]
        kind = record["record"]
        if kind == "transition_scheduled":
            statuses[name] = "new"
            self.pending[name] = record["firing_at"]
        elif kind == "transition_rescheduled":
            statuses[name] = "rescheduled"
            self.pending[name] = record["firing_at"]
        elif kind == "transition_dropped":
            statuses[name] = "dropped"
            self.pending.pop(name, None)

    def _pending_by_date(self):
        """The pending transitions, earliest first, ties by name."""
        return sorted(self.pending.items(), key=lambda kv: (kv[1], kv[0]))

    def _stable(self, time, statuses, dropped_dates):
        # `statuses` holds only the non-default ones: a pending transition
        # absent from it kept the date it already had.
        schedule = [
            {"transition": name, "date": date, "status": statuses.get(name, "kept")}
            for name, date in self._pending_by_date()
        ]
        schedule += [
            {"transition": name, "date": dropped_dates.get(name), "status": "dropped"}
            for name, status in sorted(statuses.items())
            if status == "dropped"
        ]
        self.steps.append({"kind": "stable", "time": time,
                           "state": dict(self.state), "schedule": schedule})

    def _samples_between(self, start, end):
        return [[t, v] for t, v in self.samples if start <= t <= end]

    def _follow_samples(self, time):
        if self.sampled_attribute is not None:
            self.state[self.sampled_attribute] = next(
                v for t, v in reversed(self.samples) if t <= time)

    def run(self) -> list[dict]:
        records = self.journal
        index = 0
        statuses: dict[str, str] = {}
        # The initial fixpoint at t = 0 settles silently: the pages start
        # from the stable state it reaches.
        while index < len(records) and records[index]["record"] != "transition_fired":
            record = records[index]
            if record["record"] == "attribute_changed":
                self.state[record["attribute"]] = record["new"]
            elif record["record"].startswith("transition_"):
                self._schedule_record(record, statuses)
            index += 1
        self._follow_samples(0.0)
        self._stable(0.0, {name: "armed" for name in self.pending}, {})
        last_time = 0.0
        while index < len(records):
            fired = records[index]
            time, name = fired["time"], fired["transition"]
            armed = [{"transition": t, "date": d} for t, d in self._pending_by_date()]
            next_date = armed[0]["date"] if armed else None
            if self.samples is not None:
                self.steps.append({
                    "kind": "integrate", "from": last_time, "to": time, "next_date": next_date,
                    "stopped_at_threshold": name not in self.pending,
                    "samples": self._samples_between(last_time, time),
                })
            if name == self.branch_at:
                self.steps.append({"kind": "branch", "time": time, "transition": name,
                                   "options": []})
            else:
                self.steps.append({"kind": "race", "time": last_time, "armed": armed,
                                   "next_date": next_date, "winner": name})
            before = dict(self.pending)
            self._follow_samples(time)
            self.pending.pop(name, None)
            self.state[name.rsplit(".", 1)[0]] = fired["to"]
            index += 1
            edge_writes, iterations, opened = [], [], {}
            statuses = {}
            while index < len(records) and records[index]["record"] != "transition_fired":
                record = records[index]
                kind = record["record"]
                if kind in ("function_triggered", "program_solved"):
                    # Each iteration opens with exactly one of these records.
                    opened[record["iteration"]] = {
                        "kind": "iteration", "time": time, "index": record["iteration"],
                        "function": record.get("function", record.get("program")),
                        "writes": []}
                    iterations.append(opened[record["iteration"]])
                elif kind == "attribute_changed":
                    write = {"attribute": record["attribute"],
                             "old": record["old"], "new": record["new"]}
                    if record["iteration"] == 0:
                        edge_writes.append(write)
                    else:
                        opened[record["iteration"]]["writes"].append(write)
                    self.state[record["attribute"]] = record["new"]
                elif kind.startswith("transition_"):
                    self._schedule_record(record, statuses)
                index += 1
            self.steps.append({"kind": "fired", "time": time, "transition": name,
                               "from_state": fired["from"], "to_state": fired["to"],
                               "edge_writes": edge_writes})
            self.steps.extend(iterations)
            for kept, date in before.items():
                if kept in self.pending and self.pending[kept] == date:
                    statuses.pop(kept, None)
            self._stable(time, statuses, before)
            last_time = time
        return self.steps


def _trace(name, model, seed, horizon, steps):
    _, attributes, automata = _initial_state(model)
    return _round({"scenario": name, "model": json.loads(model.json)["name"], "seed": seed,
                   "horizon": horizon, "attributes": attributes, "automata": automata,
                   "steps": steps})


def _discrete_branches(model) -> dict:
    """Both outcomes of P2's start, recorded from one snapshot."""
    session = pyraichu.Interactive(model, t_max=HORIZON_DISCRETE, journal=True, seed=SEED)
    while BRANCH not in [f.transition for f in session.fireable()]:
        if session.step() is None:
            raise RuntimeError(f"`{BRANCH}` never armed before the horizon")
    checkpoint = session.snapshot()
    journals = {}
    for outcome in ("running", "failed"):
        session.restore(checkpoint)
        session.fire(BRANCH, to=outcome)
        while session.step() is not None:
            pass
        journals[outcome] = session.journal()
    return journals


def scenarios() -> dict[str, dict]:
    """Every trace the pages replay, keyed by file stem."""
    namespace = page_models()
    discrete = namespace["discrete_model"]
    names = {"running": "discrete-backup-starts", "failed": "discrete-backup-fails"}
    labels = {"running": "P2 starts", "failed": "P2 fails on demand"}
    options = [{"label": labels[o], "to": o, "scenario": names[o]} for o in names]
    traces = {}
    for outcome, journal in _discrete_branches(discrete).items():
        steps = _Replay(discrete, journal, branch_at=BRANCH).run()
        for step in steps:
            if step["kind"] == "branch":
                step["options"] = options
        traces[names[outcome]] = _trace(names[outcome], discrete, SEED, HORIZON_DISCRETE, steps)

    continuous = namespace["continuous_model"]
    first = pyraichu.simulate(continuous, t_max=HORIZON_CONTINUOUS, seed=SEED, journal=True)
    grid = {round(SAMPLE_STEP * k, 9) for k in range(int(HORIZON_CONTINUOUS / SAMPLE_STEP) + 1)}
    instants = sorted(grid | {e.time for e in first.events})
    run = pyraichu.simulate(continuous, t_max=HORIZON_CONTINUOUS, seed=SEED, journal=True,
                            samples=instants)
    samples = [[t, v] for t, v in run.samples["level"]]
    steps = _Replay(continuous, run.journal, samples=samples, sampled_attribute="Tank.level").run()
    traces["continuous-level"] = _trace("continuous-level", continuous, SEED,
                                        HORIZON_CONTINUOUS, steps)
    return traces


def serialise(trace: dict) -> str:
    return json.dumps(trace, indent=1, sort_keys=False) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, trace in scenarios().items():
        (OUT / f"{stem}.json").write_text(serialise(trace), encoding="utf-8")


if __name__ == "__main__":
    main()
