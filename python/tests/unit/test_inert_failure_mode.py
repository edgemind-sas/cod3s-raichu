"""A failure mode whose orders are ALL inactive: a benign declaration.

Setting every rate to zero is how an analyst neutralises a failure mode
without deleting it from the study, and that is how it arrives here: the
mode is present, it carries targets and effects, and no order of it can
ever fire. The expansion used to refuse the WHOLE model over such a mode
("target is impacted by no active failure order"), which made a study
unrunnable that the reference engine ran happily, the mode simply doing
nothing.

Building a constant-false control is that same nothing, expressed. The
control exists, so every reader of it (the target's mirror, the merge of
the writers of one attribute) finds what it expects, and it is never
true, so nothing downstream of it ever moves.
"""

import pytest
from pyraichu.plugins import expand_model

RATE = 1e-3


def _target(name):
    """A component carrying the one attribute the modes below write."""
    return {
        "name": name,
        "attributes": [{"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [],
        "equations": [],
    }


def _mode(name, targets, *, active=True, behaviour="internal"):
    """One ObjFM on `targets`, with an order-1 exponential law when
    `active`, and no law on any order otherwise.

    The inert cases below are declared `external`, which is the shape the
    refusal was raised on: an external mode grafts its effects onto each
    target, and it was there that a target impacted by no active order
    took the whole model down."""
    order_1 = {"law": "exp", "rate": RATE} if active else None
    return {
        "type": "ObjFM",
        "name": name,
        "targets": list(targets),
        "failure": [order_1] + [None] * (len(targets) - 1),
        "repair": [None] * len(targets),
        "failure_effects": {"flow": False},
        "behaviour": behaviour,
    }


def _inert(name, targets):
    return _mode(name, targets, active=False, behaviour="external")


def _model(objects, targets=("A", "B")):
    return {
        "name": "inert_mode",
        "plugins": {"muscadet": {"objects": list(objects)}},
        "components": [_target(name) for name in targets],
        "connections": [],
        "indicators": [],
    }


def _attribute(expanded, name):
    for component in expanded["components"]:
        for attribute in component.get("attributes", []):
            if attribute["name"] == name:
                return attribute
    return None


def test_a_model_carrying_an_inert_mode_expands():
    expanded = expand_model(_model([_inert("inert", ["A"]), _mode("live", ["B"])]))
    assert expanded["components"]


def test_the_inert_control_is_built_and_constantly_false():
    """It is built, not skipped: every reader of a mode's control finds
    one, and this one never opens."""
    expanded = expand_model(_model([_inert("inert", ["A"]), _mode("live", ["B"])]))
    control = _attribute(expanded, "ctrl_inert_A")
    assert control is not None
    assert control["init"]["value"] is False


def test_an_inert_mode_leaves_the_others_alone():
    """The composition of the live modes must not depend on whether an
    inert one sits beside them."""

    def gates(expanded):
        return sorted(
            function["name"]
            for component in expanded["components"]
            for function in component.get("sensitive_functions", [])
        )

    with_inert = expand_model(_model([_inert("inert", ["A"]), _mode("live", ["A", "B"])]))
    without = expand_model(_model([_mode("live", ["A", "B"])]))
    assert [gate for gate in gates(with_inert) if "inert" not in gate] == gates(without)


def test_an_inert_mode_alone_is_no_different():
    """No live mode to hide behind: a model whose ONLY mode is inert
    still builds, and its control still never opens."""
    expanded = expand_model(_model([_inert("inert", ["A"])], targets=("A",)))
    assert _attribute(expanded, "ctrl_inert_A")["init"]["value"] is False


def test_an_inert_mode_needs_no_repair_law_when_each_target_repairs_itself():
    """`external_rep_indep` gives each target its own repair, driven by the
    mode's order-1 repair law, so an inactive one is a real hole and is
    refused. Not for an inert mode: it never enters the mirror's failure
    state, so that repair transition is unreachable and the law it would
    carry is not required. Demanding it was the same hard stop over a
    declaration that does nothing."""
    expanded = expand_model(
        _model([_mode("inert", ["A"], active=False, behaviour="external_rep_indep")], targets=("A",))
    )
    assert _attribute(expanded, "ctrl_inert_A")["init"]["value"] is False


def test_a_live_mode_that_repairs_itself_still_needs_its_repair_law():
    """The counter-case: the guard is about a repair that can be reached."""
    live = _mode("live", ["A"], behaviour="external_rep_indep")
    with pytest.raises(ValueError, match="order-1 repair"):
        expand_model(_model([live], targets=("A",)))
