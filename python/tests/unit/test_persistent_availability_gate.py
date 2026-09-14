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

**Two routes, one question.** The platform's flatter vocabulary reaches
the engine through `pyraichu.plugins`, a muscadet DECLARATION through
`pyraichu.declare`, and the control travels on both -- the reference
corpus carries 27 persistent gates, all of them on a declaration. The
second half of this suite asks the declaration route the same three
questions, and adds the writer only a declaration has: a failure mode
declared INSIDE the component, whose availability this engine DERIVES.
"""

import pytest

import pyraichu
import pyraichu.declare as declare
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


class TestAGateTheComponentsOwnModeDerivesIsRefused:
    """The writer neither whole-model check sees, because it is not an
    effect at all: a failure mode declared INSIDE the `ObjFlow` makes
    this engine DERIVE the gate from the mode states at every pass,
    which is the opposite of memorising."""

    @staticmethod
    def _objflow_with_mode(reset):
        spec = _objflow(reset)
        spec["failure_modes"] = [
            {
                "name": "wear",
                "law": "exp",
                "failure": RATE,
                "repair": RATE,
                "targets": ["power"],
            }
        ]
        return spec

    def test_it_names_the_gate_the_flow_and_the_mode(self):
        with pytest.raises(ValueError) as refusal:
            expand_model(_model([self._objflow_with_mode(False)]))

        message = str(refusal.value)
        assert "power_fed_available_out" in message
        assert "var_fed_available_out_reset" in message
        assert "wear" in message

    def test_a_reinitialized_gate_the_same_mode_derives_still_builds(self):
        for reset in (True, None):
            model = expand_model(_model([self._objflow_with_mode(reset)]))
            source = _src(model)
            assert any(
                f["name"] == "update_power_fed_available_out"
                for f in source["sensitive_functions"]
            )


# --- the same three questions, asked of a muscadet DECLARATION ---------
#
# Written as `muscadet.declare.system_spec` writes it, field for field,
# because that is the document this route actually reads. The platform
# writes `var_fed_available_out_init: false` beside the reset control on
# every output of a standby channel, so the pair is declared together
# here too.

GATE = "power_fed_available_out"


def _declared_flow_out(init, reset, prod_default=True):
    entry = {
        "cls": "FlowOut",
        "name": "power",
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_prod_cond_inner_mode": "or",
        "var_prod_default": prod_default,
        "negate": False,
    }
    if init is not None:
        entry["var_fed_available_out_init"] = init
    if reset is not None:
        entry["var_fed_available_out_reset"] = reset
    return entry


def _declaration(init=False, reset=False, failure_modes=(), modes=()):
    components = {
        "SRC": {
            "name": "SRC",
            "cls": "ObjFlow",
            "flows": [_declared_flow_out(init, reset)],
            "failure_modes": list(failure_modes),
        }
    }
    for mode in modes:
        components[mode["name"]] = mode
    return {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "dormant_service",
        "components": components,
        "connections": [],
        "indicators": [],
    }


def _standalone_mode(effects, name="SRC__outage"):
    """A two-state mode declared as a component of its own, writing
    `effects` on SRC: the writer a declaration has and the component's
    own `failure_modes` section does not.

    The shape muscadet writes for `add_component(cls="ObjFMDelay", ...)`.
    """
    return {
        "name": name,
        "kind": declare.COMPONENT_KIND_TWO_STATE_MODE,
        "cls": "ObjFMDelay",
        "fm_name": name.split("__", 1)[-1],
        "targets": ["SRC"],
        "target_name": "SRC",
        "failure_effects": dict(effects),
        "failure_param_name": ["ttf"],
        "failure_param": [4],
        "repair_param_name": ["ttr"],
        "repair_param": [2],
    }


def _gate_init(document):
    body = pyraichu.model_body(document)
    source = next(c for c in body["components"] if c["name"] == "SRC")
    return {a["name"]: a["init"]["value"] for a in source["attributes"]}[GATE]


class TestTheDormantServiceFunctionIsDeclarable:
    """The two knobs the platform writes together on a standby output:
    a gate seeded UNAVAILABLE, and a gate that does not fall back on
    that seed. Refused, they refused every export of the corpus."""

    def test_the_declaration_is_accepted(self):
        assert declare.check_spec(_declaration()["components"]["SRC"]) == "SRC"

    def test_the_seed_reaches_the_gate(self):
        assert _gate_init(declare.build_document(_declaration())) is False

    def test_the_counter_case_says_the_seed_is_read_and_not_dropped(self):
        assert _gate_init(declare.build_document(_declaration(init=True))) is True

    def test_a_reset_control_that_says_nothing_is_the_same_document(self):
        """`true` is this engine's own semantics and `false` is honoured
        where nothing writes the gate, so on a flow no mode touches the
        three spellings must give one document."""
        documents = [
            declare.build_document(_declaration(init=True, reset=reset))
            for reset in (True, False, None)
        ]
        assert documents[1:] == documents[:-1]


class TestADeclarationWritingAPersistentGateIsRefused:
    """The two writers a declaration has, each refused where it meets a
    gate that is never reinitialized."""

    def test_a_standalone_mode_writing_it_is_refused_by_name(self):
        spec = _declaration(modes=[_standalone_mode({GATE: False})])
        with pytest.raises(declare.ComponentSpecError) as refusal:
            declare.build_document(spec)

        message = str(refusal.value)
        assert f"SRC.{GATE}" in message
        assert "var_fed_available_out_reset" in message
        assert "SRC__outage" in message

    def test_the_same_mode_on_a_reinitialized_gate_still_builds(self):
        """The counter-case that says the refusal is the control's and
        not the effect's."""
        for reset in (True, None):
            spec = _declaration(reset=reset, modes=[_standalone_mode({GATE: False})])
            assert declare.build_document(spec)

    def test_a_mode_writing_another_attribute_of_the_same_flow_builds(self):
        """Keyed on the GATE, not on the presence of a mode on the
        component that carries it."""
        assert declare.build_document(
            _declaration(modes=[_standalone_mode({"power_fed_out": False})])
        )

    def test_a_failure_mode_of_the_component_deriving_it_is_refused_by_name(self):
        """The writer only a declaration has: a mode declared INSIDE the
        component makes this engine DERIVE the gate from the mode states,
        which is the opposite of memorising."""
        modes = [
            {
                "cls": "delay",
                "name": "outage",
                "failure_state": "occ",
                "failure_time": 4.0,
                "failure_effects": [["power_fed_available_out", False]],
                "repair_state": "rep",
                "repair_time": 2.0,
                "repair_effects": [],
            }
        ]
        with pytest.raises(ValueError) as refusal:
            declare.build_document(_declaration(failure_modes=modes))

        message = str(refusal.value)
        assert GATE in message
        assert "var_fed_available_out_reset" in message
        assert "outage" in message

    def test_the_same_failure_mode_on_a_reinitialized_gate_still_builds(self):
        modes = [
            {
                "cls": "delay",
                "name": "outage",
                "failure_state": "occ",
                "failure_time": 4.0,
                "failure_effects": [["power_fed_available_out", False]],
                "repair_state": "rep",
                "repair_time": 2.0,
                "repair_effects": [],
            }
        ]
        assert declare.build_document(_declaration(reset=True, failure_modes=modes))
