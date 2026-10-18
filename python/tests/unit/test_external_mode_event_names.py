"""The events an external mode writes on its targets name the mode.

An ``external`` (or ``external_rep_indep``) failure mode drives a mirror
automaton grafted on each target, and that mirror carries the mode's
sequence events. cod3s prefixes the mirror's states and transitions with
the MODE's name (``_build_target_automaton``: ``C1.frun__occ``, never
``C1.occ``), so a target hosting several modes says which one failed it,
and the reference engine records ``<target>.<mode>__occ``. This expansion
used to keep the bare ``occ`` / ``rep``: a COD3S Platform sequence then
carried no mode name on those events, so the run viewer and the exported
sequences could not say which mode failed a target (COD3S #354).

What is pinned:

1. the mirror's states and transitions are ``<mode>__<state>``, for both
   external behaviours;
2. the prefix is the mode's OWN name, not the name of the component the
   mode is filed under: a cod3s document files ``PMP_failure`` acting on
   ``SRC`` and ``PMP`` under ``SRC_PMP__PMP_failure``, and the reference
   engine prefixes with ``PMP_failure``;
3. a plugin object that declares no mode name of its own is named by its
   ``name``, which is then both;
4. a sequence campaign records the prefixed names on the targets.
"""

from __future__ import annotations

import pyraichu
import pytest
from pyraichu.mode_objects import failure_mode_object
from pyraichu.plugins import expand_model

BEHAVIOURS = ["external", "external_rep_indep"]


def _target(name):
    return {
        "name": name,
        "attributes": [
            {"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}
        ],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [],
        "equations": [],
    }


def _leaf(obj):
    return {"obj": obj, "attr": "flow", "ope": "==", "value": False}


def _model(objects, targets):
    return {
        "name": "external_names",
        "plugins": {
            "muscadet": {
                "objects": list(objects)
                + [
                    {
                        "type": "ObjEvent",
                        "name": "ER",
                        "target": True,
                        "cond": [[_leaf(t) for t in targets]],
                    }
                ]
            }
        },
        "components": [_target(name) for name in targets],
        "connections": [],
        "indicators": [],
    }


def _cod3s_mode(behaviour, targets=("SRC", "PMP")):
    """``PMP_failure`` in the cod3s wire, filed under the component name the
    cod3s document gives it, which is not the mode's own."""
    obj = failure_mode_object(
        {
            "cls": "ObjFMExp",
            "fm_name": "PMP_failure",
            "targets": list(targets),
            "behaviour": behaviour,
            "failure_param": [1e-2] * len(targets),
            "repair_param": [1e-1] * len(targets),
            "failure_effects": {"flow": False},
        }
    )
    obj["name"] = "SRC_PMP__PMP_failure"
    return obj


def _mirror(expanded, target, automaton):
    component = next(c for c in expanded["components"] if c["name"] == target)
    return next(a for a in component["automata"] if a["name"] == automaton)


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_the_mirror_names_its_states_and_transitions_after_the_mode(behaviour):
    expanded = expand_model(_model([_cod3s_mode(behaviour)], ("SRC", "PMP")))
    for target in ("SRC", "PMP"):
        mirror = _mirror(expanded, target, "SRC_PMP__PMP_failure")
        assert mirror["states"] == ["PMP_failure__rep", "PMP_failure__occ"]
        assert mirror["init"] == "PMP_failure__rep"
        names = sorted(t["name"] for t in mirror["transitions"])
        assert names == ["PMP_failure__occ", "PMP_failure__rep"]
    pyraichu.load_model(expanded)


def test_a_plugin_object_without_a_mode_name_is_named_by_its_name():
    spec = {
        "type": "ObjFM",
        "name": "wear",
        "behaviour": "external",
        "targets": ["A"],
        "failure": [{"law": "exp", "rate": 1e-2}],
        "repair": [{"law": "exp", "rate": 1e-1}],
        "failure_effects": {"flow": False},
    }
    mirror = _mirror(expand_model(_model([spec], ("A",))), "A", "wear")
    assert sorted(t["name"] for t in mirror["transitions"]) == ["wear__occ", "wear__rep"]


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_a_sequence_names_the_mode_on_the_target_side(behaviour):
    model = pyraichu.load_model(_model([_cod3s_mode(behaviour)], ("SRC", "PMP")))
    campaign = pyraichu.run_sequences(model, 2_000, 200.0, seed=3)
    hits = [s for s in campaign.minimal if s["end_cause"] == "ER"]
    assert hits, "the campaign never reached the feared event"
    target_side = {
        event["attr"]
        for sequence in hits
        for event in sequence["events"]
        if event["obj"] in ("SRC", "PMP")
    }
    assert target_side, hits[:3]
    assert target_side <= {"PMP_failure__occ", "PMP_failure__rep"}, target_side
