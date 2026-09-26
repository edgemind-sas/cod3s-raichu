"""Sequence campaigns that observe attributes, and conditions on them.

A campaign can read the value of chosen attributes at chosen instants on
every trajectory; the raw corpus carries them, and a condition on them
selects the trajectories the levels are reduced from. The Rust suite pins
the reading itself against the recorded events; these tests pin the Python
surface.
"""

import json

import pytest

import pyraichu
from pyraichu import Observation, SequenceCondition

from test_raw_sequences import MODEL, NB_RUNS, T_MAX

A_FLOW = Observation(name="A_flow", component="A", attribute="flow", time=50.0)
A_DOWN = SequenceCondition(observation="A_flow", op="==", value=0)
A_UP = SequenceCondition(observation="A_flow", op="==", value=1)


def _campaign(tmp_path, condition=None, observations=(A_FLOW,)):
    return pyraichu.run_sequences(
        pyraichu.load_model(MODEL),
        NB_RUNS,
        T_MAX,
        seed=42,
        raw_path=tmp_path / "raw.jsonl",
        observations=observations,
        condition=condition,
    )


def test_observing_changes_neither_the_trajectories_nor_the_levels(tmp_path):
    observed = _campaign(tmp_path)
    plain = pyraichu.run_sequences(pyraichu.load_model(MODEL), NB_RUNS, T_MAX, seed=42)
    assert observed.minimal == plain.minimal
    assert observed.cleaned == plain.cleaned
    assert observed.condition is None


def test_the_raw_corpus_carries_one_value_per_trajectory(tmp_path):
    lines = _campaign(tmp_path).raw_path.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["observations"] == [{"name": "A_flow", "time": 50.0}]
    values = [json.loads(line)["observed"] for line in lines[1:]]
    assert len(values) == NB_RUNS
    assert {v[0] for v in values} == {0, 1}


def test_a_condition_reduces_only_the_trajectories_it_holds_on(tmp_path):
    down = _campaign(tmp_path, condition=A_DOWN)
    lines = down.raw_path.read_text().splitlines()[1:]
    expected = sum(1 for line in lines if json.loads(line)["observed"][0] == 0)

    assert down.condition == {
        "observation": "A_flow",
        "op": "==",
        "value": 0.0,
        "total_trajectories": NB_RUNS,
        "kept_trajectories": expected,
    }
    assert 0 < expected < NB_RUNS
    # The cleaned level weighs exactly the kept trajectories.
    assert sum(s["weight"] for s in down.cleaned) == expected
    # The raw corpus still holds every trajectory.
    assert len(lines) == NB_RUNS

    up = _campaign(tmp_path, condition=A_UP)
    assert up.condition["kept_trajectories"] + expected == NB_RUNS


def test_the_corpus_re_reduces_under_a_condition(tmp_path):
    down = _campaign(tmp_path, condition=A_DOWN)
    again = pyraichu.analyse_raw_sequences(down.raw_path, condition=A_DOWN)
    assert again.minimal == down.minimal
    assert again.cleaned == down.cleaned
    assert again.condition == down.condition

    whole = pyraichu.analyse_raw_sequences(down.raw_path)
    assert whole.condition is None
    assert sum(s["weight"] for s in whole.cleaned) == NB_RUNS


def test_a_condition_on_what_was_not_observed_is_refused(tmp_path):
    other = SequenceCondition(observation="B_flow", op="==", value=0)
    with pytest.raises(pyraichu.SimulationError, match="B_flow"):
        _campaign(tmp_path, condition=other)
    plain = _campaign(tmp_path, observations=())
    with pytest.raises(pyraichu.SimulationError, match="observed nothing"):
        pyraichu.analyse_raw_sequences(plain.raw_path, condition=A_DOWN)


def test_a_corpus_without_observations_carries_none(tmp_path):
    text = _campaign(tmp_path, observations=()).raw_path.read_text()
    assert "observ" not in text


def test_what_cannot_be_observed_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a comparison"):
        SequenceCondition(observation="A_flow", op="=~", value=0)
    missing = Observation(name="x", component="A", attribute="nope", time=1.0)
    with pytest.raises(pyraichu.ModelError):
        _campaign(tmp_path, observations=(missing,))
    early = Observation(name="x", component="A", attribute="flow", time=-1.0)
    with pytest.raises(pyraichu.SimulationError, match="not an instant"):
        _campaign(tmp_path, observations=(early,))
