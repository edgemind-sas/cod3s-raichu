"""The reduction's safety net and per-group decision, seen from Python.

Cancelling a failure repaired before the feared event assumes the repair
undid its effect. In the model below it does not: the failure of `A` sets a
latch that stays set, and the feared event needs the latch set and `A`
repaired. Every trajectory reaches the event with its only failure repaired.

The per-group decision sees the latch and keeps `A`'s cycles. Declared
`cycle_cancellation: always`, `A` is cancelled anyway: cancellation then
leaves each trajectory with no cause at all, an empty sequence that would
absorb every other one, and the safety net keeps those trajectories whole
and says so.
"""

import copy
import json
import warnings

import pyraichu


def _state(component, automaton, state):
    return {"op": "state_active", "state": {"component": component, "automaton": automaton, "state": state}}


#: `A` fails and is repaired (a monitored cycle pair, kinds declared); its
#: failure sets the unmonitored latch `L` at once; the unmonitored watcher
#: `ER` reaches the feared event once `L` is set and `A` is back.
LATCH = {
    "name": "latch",
    "components": [
        {
            "name": "A",
            "automata": [
                {
                    "name": "life",
                    "states": ["ok", "ko"],
                    "init": "ok",
                    "transitions": [
                        {
                            "name": "fail",
                            "source": "ok",
                            "targets": ["ko"],
                            "distrib": "exp",
                            "rate": 0.5,
                            "monitored": True,
                            "cycle_group": "life",
                            "kind": "failure",
                        },
                        {
                            "name": "fix",
                            "source": "ko",
                            "targets": ["ok"],
                            "distrib": "exp",
                            "rate": 2.0,
                            "monitored": True,
                            "cycle_group": "life",
                            "kind": "repair",
                        },
                    ],
                }
            ],
        },
        {
            "name": "L",
            "automata": [
                {
                    "name": "latch",
                    "states": ["off", "on"],
                    "init": "off",
                    "transitions": [
                        {
                            "name": "set",
                            "source": "off",
                            "targets": ["on"],
                            "distrib": "delay",
                            "time": 0.0,
                            "guard": _state("A", "life", "ko"),
                        },
                    ],
                }
            ],
        },
        {
            "name": "ER",
            "automata": [
                {
                    "name": "ev",
                    "states": ["not_occ", "occ"],
                    "init": "not_occ",
                    "transitions": [
                        {
                            "name": "occ",
                            "source": "not_occ",
                            "targets": ["occ"],
                            "distrib": "delay",
                            "time": 0.0,
                            "guard": {
                                "op": "bool",
                                "bool_op": "and",
                                "args": [_state("L", "latch", "on"), _state("A", "life", "ok")],
                            },
                        },
                    ],
                }
            ],
        },
    ],
    "targets": [{"name": "feared", "component": "ER", "automaton": "ev", "state": "occ"}],
}

NB_RUNS = 200
T_MAX = 50.0


def _forced(spec):
    """`spec` with `A`'s cycle group declared always cancellable."""
    spec = copy.deepcopy(spec)
    for transition in spec["components"][0]["automata"][0]["transitions"]:
        transition["cycle_cancellation"] = "always"
    return spec


def _model():
    return pyraichu.load_model(_forced(LATCH))


def test_a_campaign_reports_the_trajectories_it_kept_whole(tmp_path):
    campaign = pyraichu.run_sequences(_model(), NB_RUNS, T_MAX, seed=3, raw_path=tmp_path / "raw.jsonl")
    reached = sum(s["weight"] for s in campaign.minimal)
    assert reached > 0
    # Every reaching trajectory had its only failure repaired.
    assert campaign.reduction["contradicted"] == reached
    assert campaign.reduction["contradicted_indices"] == sorted(campaign.reduction["contradicted_indices"])
    assert len(campaign.reduction["contradicted_indices"]) == min(reached, 100)
    # No minimal sequence is cause-free.
    assert all(s["events"] for s in campaign.minimal)
    assert campaign.minimal[0]["events"][0]["obj"] == "A"

    reread = pyraichu.analyse_raw_sequences(tmp_path / "raw.jsonl")
    assert reread.reduction == campaign.reduction
    assert reread.minimal == campaign.minimal


def test_analyse_sequences_warns_and_returns_the_same_list():
    campaign = pyraichu.run_sequences(_model(), NB_RUNS, T_MAX, seed=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        minimal = pyraichu.analyse_sequences(_model(), NB_RUNS, T_MAX, seed=3)
    assert minimal == campaign.minimal
    emitted = [w for w in caught if issubclass(w.category, pyraichu.ReductionWarning)]
    assert len(emitted) == 1
    assert f"{campaign.reduction['contradicted']} trajectories" in str(emitted[0].message)


def test_no_warning_when_every_cancellation_holds():
    # The feared event as soon as the latch is set: `A` is still down
    # then, so its failure survives cancellation.
    spec = copy.deepcopy(LATCH)
    spec["components"][2]["automata"][0]["transitions"][0]["guard"] = _state("L", "latch", "on")
    model = pyraichu.load_model(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyraichu.ReductionWarning)
        minimal = pyraichu.analyse_sequences(model, NB_RUNS, T_MAX, seed=3)
    assert minimal and all(s["events"] for s in minimal)


def test_an_exploration_reports_the_mass_it_kept_whole():
    exploration = pyraichu.explore(_model(), "feared", 5.0)
    minimal, reduction = exploration.minimal_sequences_with_report()
    assert reduction["contradicted"] == len(exploration.sequences)
    assert abs(reduction["contradicted_weight"] - exploration.lower) <= 1e-12 * exploration.lower
    assert abs(reduction["contradicted_share"] - 1.0) <= 1e-12
    # Total probability is preserved, and no minimal sequence is empty.
    assert abs(sum(s["weight"] for s in minimal) - exploration.lower) <= 1e-12 * exploration.lower
    assert all(s["events"] for s in minimal)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert exploration.minimal_sequences() == minimal
    assert any(issubclass(w.category, pyraichu.ReductionWarning) for w in caught)


def test_the_raw_corpus_carries_the_declared_kinds(tmp_path):
    pyraichu.run_sequences(_model(), 20, T_MAX, seed=3, raw_path=tmp_path / "raw.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()[1:]]
    kinds = {kind for line in lines for kind in line.get("kinds", [])}
    assert kinds == {"failure", "repair"}


def test_under_a_condition_the_report_names_replicas_of_the_corpus(tmp_path):
    # Skipping replicas must not shift the indices: they name corpus runs.
    spec = _forced(LATCH)
    spec["components"][1]["attributes"] = [{"name": "set", "kind": "float", "init": {"kind": "float", "value": 0.0}}]
    spec["components"][1]["equations"] = [
        {
            "target": "set",
            "kind": "explicit",
            "expr": {
                "op": "if",
                "cond": _state("L", "latch", "on"),
                "then": {"op": "const", "value": {"kind": "float", "value": 1.0}},
                "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.0}},
            },
        }
    ]
    early = pyraichu.Observation(name="set_early", component="L", attribute="set", time=1.0)
    condition = pyraichu.SequenceCondition(observation="set_early", op="==", value=1)
    raw = tmp_path / "raw.jsonl"
    campaign = pyraichu.run_sequences(
        pyraichu.load_model(spec), NB_RUNS, T_MAX, seed=3, raw_path=raw, observations=[early], condition=condition
    )
    kept = campaign.condition["kept_trajectories"]
    assert 0 < kept < NB_RUNS
    lines = {line["run"]: line for line in map(json.loads, raw.read_text().splitlines()[1:])}
    indices = campaign.reduction["contradicted_indices"]
    assert campaign.reduction["contradicted"] == len(indices) > 0
    # Each index names a corpus run the condition kept and that reached the
    # feared event with its only failure repaired.
    assert all(lines[i]["observed"] == [1] and lines[i]["end_cause"] == "feared" for i in indices)
    assert any(i >= kept for i in indices)
    assert pyraichu.analyse_raw_sequences(raw, condition=condition).reduction == campaign.reduction


def test_the_per_group_decision_keeps_the_latched_failure():
    (decision,) = pyraichu.cycle_groups(pyraichu.load_model(LATCH))
    (target,) = decision["targets"]
    assert decision["component"] == "A"
    assert target["cancellable"] is False
    assert target["origin"] == "analysis"
    assert target["memory"] == "L.latch"
    assert decision["messages"] and "L.latch" in decision["messages"][0]

    campaign = pyraichu.run_sequences(pyraichu.load_model(LATCH), NB_RUNS, T_MAX, seed=3)
    assert campaign.reduction["contradicted"] == 0
    assert campaign.reduction["cycle_groups"][0]["targets"][0]["cancellable"] is False
    # Kept, A's cycle stays in every reaching sequence.
    assert all([(e["obj"], e["attr"]) for e in s["events"]] == [("A", "ko"), ("A", "ok")] for s in campaign.minimal)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyraichu.ReductionWarning)
        assert pyraichu.analyse_sequences(pyraichu.load_model(LATCH), NB_RUNS, T_MAX, seed=3)


def test_a_declaration_overrides_the_decision():
    (decision,) = pyraichu.cycle_groups(_model())
    assert decision["targets"][0]["cancellable"] is True
    assert decision["targets"][0]["origin"] == "declared"


def test_an_exploration_written_before_the_decisions_warns_when_reduced():
    current = pyraichu.explore(pyraichu.load_model(LATCH), "feared", 5.0)
    assert current.cycle_groups is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyraichu.ReductionWarning)
        current.minimal_sequences()

    document = json.loads(current.to_json())
    del document["cycle_groups"]
    old = pyraichu.read_exploration(json.dumps(document))
    assert old.cycle_groups is None
    _, reduction = old.minimal_sequences_with_report()
    assert reduction["cycle_groups"] is None
    assert reduction["cycle_groups_unknown"] is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old.minimal_sequences()
    assert any("written before the per-group decisions" in str(w.message) for w in caught)


def test_a_corpus_read_back_applies_the_decisions_it_carries(tmp_path):
    raw = tmp_path / "raw.jsonl"
    campaign = pyraichu.run_sequences(pyraichu.load_model(LATCH), NB_RUNS, T_MAX, seed=3, raw_path=raw)
    reread = pyraichu.analyse_raw_sequences(raw)
    assert reread.minimal == campaign.minimal
    assert reread.reduction == campaign.reduction
    assert reread.reduction["cycle_groups"][0]["targets"][0]["cancellable"] is False
    assert all([(e["obj"], e["attr"]) for e in s["events"]] == [("A", "ko"), ("A", "ok")] for s in reread.minimal)
