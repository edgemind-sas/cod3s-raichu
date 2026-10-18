"""The repair of an ``external`` failure mode cancels its failure.

In the ``external`` behaviour a muscadet failure mode keeps its own occ/rep
automaton (unmonitored) and drives, through a control attribute, a mirror
automaton grafted on each target, which carries the recorded events. The
mode's automaton reads the mirror back (the mutual lock), so the cycle-group
analysis walks from the mirror into it. It is read only by the mirrors,
which copy it at once, so its state is at every instant what their recorded
events show: it is no memory, and with nothing else remembering, every group
stays cancellable.

Before this was settled, the analysis named the mode's own automaton as the
memory of nearly every group of an ``external`` model, and the minimal
sequences kept every repaired failure (edgemind-sas/cod3s-raichu#148).
"""

from __future__ import annotations

import pyraichu

NB_RUNS = 4_000
T_MAX = 200.0


def _leaf(obj: str) -> dict:
    return {"obj": obj, "attr": "flow", "ope": "==", "value": False}


def _mode(name: str, targets: list[str], failure: list[dict]) -> dict:
    """A repairable ``external`` mode cutting the flow of its targets."""
    return {
        "type": "ObjFM",
        "name": name,
        "behaviour": "external",
        "targets": targets,
        "failure": failure,
        "repair": [{"law": "exp", "rate": 0.5}] * len(failure),
        "failure_effects": {"flow": False},
    }


def _model():
    """A in series with (B ∥ C); B and C share a common cause."""
    exp = lambda rate: {"law": "exp", "rate": rate}  # noqa: E731
    objects = [
        _mode("wear_A", ["A"], [exp(0.01)]),
        _mode("wear_BC", ["B", "C"], [exp(0.05), exp(0.01)]),
        {
            "type": "ObjEvent",
            "name": "no_supply",
            "target": True,
            "cond": [[_leaf("A")], [_leaf("B"), _leaf("C")]],
        },
    ]
    return pyraichu.load_model(
        {
            "name": "external_cycles",
            "plugins": {"muscadet": {"objects": objects}},
            "components": [
                {
                    "name": name,
                    "attributes": [
                        {"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}
                    ],
                }
                for name in ("A", "B", "C")
            ],
            "indicators": [],
        }
    )


def test_every_group_of_an_external_model_is_cancellable():
    decisions = pyraichu.cycle_groups(_model())
    components = {d["component"] for d in decisions}
    assert {"A", "B", "C"} <= components, components
    kept = [
        (d["component"], d["cycle_group"], t.get("memory"))
        for d in decisions
        for t in d["targets"]
        if not t["cancellable"]
    ]
    assert kept == []


def test_the_cleaned_sequences_drop_every_repaired_failure():
    # The cleaned level, not the minimal one: minimisation would absorb a
    # path carrying a repaired cycle into the same path without it, so only
    # the cleaned level shows whether the reduction cancelled the cycle.
    campaign = pyraichu.run_sequences(_model(), NB_RUNS, T_MAX, seed=7)
    hits = [s for s in campaign.cleaned if s["end_cause"] == "no_supply"]
    assert hits, "the campaign never reached the feared event"
    for sequence in hits:
        states = [event["attr"] for event in sequence["events"]]
        # The mirrors name their states after the mode (`wear_A__occ`), so a
        # check on the bare `rep` prefix would hold whatever the reduction did.
        assert any(state.endswith("__occ") for state in states), sequence
        assert not any(state.endswith("__rep") for state in states), sequence
