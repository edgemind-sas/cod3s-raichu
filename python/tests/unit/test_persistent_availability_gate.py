"""A persistent availability gate: honoured when nothing writes it,
refused when a failure mode does.

`fed_available_reset` is muscadet's ``FlowOut.var_fed_available_out_reset``,
which switches off the per-step reinitialization of
``{flow}_fed_available_out``. `true`, its default, is this engine's own
semantics: the gate is re-derived at every fixpoint pass and falls back to
its declared initial value as soon as no mode holds it down. `false` asks
for a gate that MEMORISES, so that a detection or an alarm latches, and
this engine has no per-variable reset to switch off.

The control travels on **every** output port of a library that declares
it, while only a handful of ports are ever written by a mode. So the
question this suite pins is not "is the persistent gate implemented" but
"is the model that carries the control the same model on both engines":
untouched, it is, and the port must build; written, it is not, and the
model must be refused rather than silently built with a gate that comes
back up on repair.
"""

import pytest
from pyraichu.plugins import expand_model

RATE = 1e-3


def _objflow(reset):
    """One producer whose `power` output carries the reset control, or
    carries nothing when `reset` is None (the declaration of a library
    that does not know the control at all)."""
    flow = {"name": "power", "var_prod_cond": []}
    if reset is not None:
        flow["var_fed_available_out_reset"] = reset
    return {"type": "ObjFlow", "name": "SRC", "flows_out": [flow]}


def _mode(effects):
    """One non-repairable internal mode on SRC, writing `effects`."""
    return {
        "type": "ObjFM",
        "name": "M",
        "targets": ["SRC"],
        "failure": [{"law": "exp", "rate": RATE}],
        "repair": [None],
        "failure_effects": dict(effects),
    }


def _model(objects):
    return {
        "name": "persistent_gate",
        "plugins": {"muscadet": {"objects": list(objects)}},
        "components": [],
        "connections": [],
        "indicators": [],
    }


def _src(model):
    return next(c for c in model["components"] if c["name"] == "SRC")


class TestAPersistentGateNoModeWritesIsBuilt:
    """Nothing restores the gate and nothing writes it, so both engines
    leave it at its declared initial value for the whole sequence. The
    control is then a statement about a variable no one touches, and the
    model must build."""

    def test_it_builds_and_declares_the_gate(self):
        model = expand_model(_model([_objflow(False)]))
        inits = {a["name"]: a["init"]["value"] for a in _src(model)["attributes"]}
        assert inits["power_fed_available_out"] is True

    def test_it_builds_beside_a_mode_that_writes_another_attribute(self):
        """The refusal is keyed on the GATE, not on the presence of a
        mode on the component that carries it: a mode writing the
        delivered flow of that same port leaves the gate alone, and must
        build."""
        model = expand_model(_model([_objflow(False), _mode({"power_fed_out": False})]))
        writer = next(
            f
            for c in model["components"]
            if c["name"] == "M"
            for f in c["sensitive_functions"]
            if f["name"] == "apply_effects"
        )
        assert writer["effects"][0]["target"] == {
            "component": "SRC",
            "attribute": "power_fed_out",
        }

    def test_the_document_is_the_one_the_control_free_declaration_gives(self):
        """A `false` gate no one writes must not perturb the emission:
        the model is byte-identical to the same declaration without the
        control, which is what makes reading the role safe to apply to
        every port of a library."""
        with_control = expand_model(_model([_objflow(False)]))
        without = expand_model(_model([_objflow(None)]))
        assert with_control == without


class TestAPersistentGateAModeWritesIsRefused:
    """A held effect on a gate that is never reinitialized has no
    faithful expansion: the engine restores the rest state the mode
    declares, so the gate would come back up on repair instead of
    latching."""

    def test_it_names_the_gate_the_flow_and_the_mode(self):
        model = _model([_objflow(False), _mode({"power_fed_available_out": False})])
        with pytest.raises(ValueError) as refusal:
            expand_model(model)
        message = str(refusal.value)
        assert "SRC.power_fed_available_out" in message
        assert "fed_available_reset" in message
        assert "'M'" in message

    def test_a_reinitialized_gate_the_same_mode_writes_still_builds(self):
        """The counter-case that says the refusal is the control's and
        not the effect's: the very same mode on a `true` gate is the
        ordinary model, and it must keep building."""
        for reset in (True, None):
            model = expand_model(
                _model([_objflow(reset), _mode({"power_fed_available_out": False})])
            )
            writer = next(
                f
                for c in model["components"]
                if c["name"] == "M"
                for f in c["sensitive_functions"]
                if f["name"] == "apply_effects"
            )
            target = writer["effects"][0]["target"]
            assert target == {"component": "SRC", "attribute": "power_fed_available_out"}
