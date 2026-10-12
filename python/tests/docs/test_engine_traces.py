"""The "How the engine works" pages replay what the engine really does.

Their traces are recorded by ``docs/figures/engine_traces.py`` from the
model the introduction page defines, and committed under
``docs/engine/traces/``. This test records them again and compares: a
change in the engine's behaviour on that model fails here, naming the
scenario and the first step that differs, instead of leaving the pages
to explain an engine that no longer exists.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[3] / "docs"
GENERATOR = DOCS / "figures" / "engine_traces.py"


def _generator():
    spec = importlib.util.spec_from_file_location("engine_traces", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _generator()
RECORDED = GEN.scenarios()


def first_difference(committed: dict, recorded: dict) -> str | None:
    """Where two traces part, in words; ``None`` when they agree."""
    for key in ("scenario", "model", "seed", "horizon", "attributes", "automata"):
        if committed.get(key) != recorded.get(key):
            return f"header `{key}`: committed {committed.get(key)!r}, engine {recorded.get(key)!r}"
    old, new = committed["steps"], recorded["steps"]
    for index, (a, b) in enumerate(zip(old, new)):
        if a != b:
            return f"step {index} ({a.get('kind')}): committed {a!r}, engine {b!r}"
    if len(old) != len(new):
        return f"{len(old)} committed steps, the engine now records {len(new)}"
    return None


@pytest.mark.parametrize("stem", sorted(RECORDED))
def test_the_committed_trace_is_what_the_engine_records(stem):
    path = GEN.OUT / f"{stem}.json"
    assert path.exists(), f"{path} is missing: run docs/figures/engine_traces.py"
    committed = json.loads(path.read_text(encoding="utf-8"))
    difference = first_difference(committed, RECORDED[stem])
    assert difference is None, (
        f"trace `{stem}` no longer matches the engine, {difference}. "
        "If the change is intended, regenerate with docs/figures/engine_traces.py "
        "and check the pages still tell the truth."
    )


def test_an_altered_trace_is_caught_with_its_step():
    committed = json.loads((GEN.OUT / "discrete-backup-starts.json").read_text(encoding="utf-8"))
    altered = json.loads(json.dumps(committed))
    iteration = next(s for s in altered["steps"] if s["kind"] == "iteration")
    iteration["writes"][0]["new"] = not iteration["writes"][0]["new"]
    message = first_difference(altered, RECORDED["discrete-backup-starts"])
    assert message is not None and message.startswith("step 3 (iteration)")


def test_a_regeneration_is_byte_identical():
    again = GEN.scenarios()
    assert {k: GEN.serialise(v) for k, v in again.items()} == {
        k: GEN.serialise(v) for k, v in RECORDED.items()
    }


def test_both_outcomes_share_every_step_before_the_choice():
    starts = RECORDED["discrete-backup-starts"]["steps"]
    fails = RECORDED["discrete-backup-fails"]["steps"]
    choice = next(i for i, s in enumerate(starts) if s["kind"] == "branch")
    assert starts[: choice + 1] == fails[: choice + 1]
    assert starts[choice + 1]["to_state"] == "running"
    assert fails[choice + 1]["to_state"] == "failed"


def test_iterations_are_numbered_from_one_in_each_fixpoint():
    for trace in RECORDED.values():
        expected = 1
        for step in trace["steps"]:
            if step["kind"] == "fired":
                expected = 1
            elif step["kind"] == "iteration":
                assert step["index"] == expected
                expected += 1


def test_the_race_after_the_failure_has_two_runners():
    steps = RECORDED["discrete-backup-starts"]["steps"]
    races = [s for s in steps if s["kind"] == "race"]
    after_start = races[1]
    dates = [runner["date"] for runner in after_start["armed"]]
    assert len(dates) >= 2 and len(set(dates)) == len(dates)
    assert after_start["winner"] == after_start["armed"][0]["transition"]


def test_a_threshold_interrupts_integration_with_a_date_still_pending():
    steps = RECORDED["continuous-level"]["steps"]
    stopped = [s for s in steps if s["kind"] == "integrate" and s["stopped_at_threshold"]]
    assert stopped, "no integration segment ended at a threshold"
    first = stopped[0]
    assert first["next_date"] is not None and first["next_date"] > first["to"]
    following = steps[steps.index(first) + 1]
    assert following["kind"] == "race"
    assert first["next_date"] in [runner["date"] for runner in following["armed"]]


def test_the_level_stays_physical_and_turns_down_after_the_failure():
    trace = RECORDED["continuous-level"]
    levels = [v for s in trace["steps"] if s["kind"] == "integrate" for _, v in s["samples"]]
    namespace = GEN.page_models()
    assert levels and all(namespace["LOW"] - 1e-9 <= v <= namespace["HIGH"] + 1e-9 for v in levels)
    fail = next(s["time"] for s in trace["steps"]
                if s["kind"] == "fired" and s["transition"] == "P1.mode.fail")
    after = [s for s in trace["steps"]
             if s["kind"] == "integrate" and s["from"] >= fail and s["to"] > s["from"]]
    first = after[0]["samples"]
    assert first[-1][1] < first[0][1]
