"""A persistent availability gate: honoured when nothing writes it,
LATCHED when a single failure mode holds it.

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
untouched, it is, and the port must build. Written by ONE mode's held
effect, it is too since pyraichu 0.43.0: measured on PyCATSHOO (muscadet
5.6.0, 2026-09-26), a held effect on a gate that is never reinitialized
writes its value when the mode ENTERS the state it is held in and nothing
restores it afterwards, which is an edge write. The expansion emits exactly
that (`effects` on the mode's transitions) instead of the level that would
bring the gate back up on repair. Written by two modes, the reference has
no fixpoint (muscadet documents the hang) and the model is refused.

**Two routes, one question.** The platform's flatter vocabulary reaches
the engine through `pyraichu.plugins`, a muscadet DECLARATION through
`pyraichu.declare`, and the control travels on both -- the reference
corpus carries 27 persistent gates, all of them on a declaration. The
second half of this suite asks the declaration route the same three
questions, and adds the writer only a declaration has: a failure mode
declared INSIDE the component, whose availability this engine DERIVES.
"""

import json

import pytest

import pyraichu
import pyraichu.declare as declare
from pyraichu.plugins import expand_model

RATE = 1e-3

#: The persistent gate every test below is about.
GATE = "power_fed_available_out"


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


#: Where the trajectories below are read: clear of the transition dates of
#: `_delay_mode` (failure at 1, repair at 2, failure at 3, repair at 4).
SAMPLES = [0.5, 1.5, 2.5, 3.5, 4.5]


def _delay_mode(failure_effects, repair_effects=None, name="M", targets=("SRC",)):
    """An internal mode on delays: failure at 1, repair at 2, and again."""
    mode = {
        "type": "ObjFM",
        "name": name,
        "targets": list(targets),
        "failure": [{"law": "delay", "time": 1.0}] * len(targets),
        "repair": [{"law": "delay", "time": 1.0}] * len(targets),
        "failure_effects": dict(failure_effects),
    }
    if repair_effects:
        mode["repair_effects"] = dict(repair_effects)
    return mode


def _objflow_seeded(reset, init):
    spec = _objflow(reset)
    spec["flows_out"][0]["var_fed_available_out_init"] = init
    return spec


def _gate_trajectory(body):
    body = dict(body)
    body["indicators"] = [
        {"name": "gate", "target": "attribute", "attr": {"component": "SRC", "attribute": GATE}}
    ]
    trajectory = pyraichu.simulate(pyraichu.load_model(body), t_max=SAMPLES[-1], samples=SAMPLES)
    return [bool(value) for _, value in trajectory.samples["gate"]]


class TestAHeldWriteOnAPersistentGateLatches:
    """A held effect on a gate that is never reinitialized is an EDGE write.

    Every expected trajectory below is the one PyCATSHOO returns for the same
    muscadet system (5.6.0, `ObjFMDelay` failing at 1 and repairing at 2, and
    so on; gate read at 0.5, 1.5, ...), measured 2026-09-26.
    """

    def test_a_failure_effect_latches_at_the_failure_and_survives_the_repair(self):
        body = expand_model(_model([_objflow(False), _delay_mode({GATE: False})]))
        assert _gate_trajectory(body) == [True, False, False, False, False]

    def test_a_repair_effect_writes_the_other_polarity_on_the_repair_edge(self):
        body = expand_model(
            _model([_objflow(False), _delay_mode({GATE: False}, {GATE: True})])
        )
        assert _gate_trajectory(body) == [True, False, True, False, True]

    def test_a_repair_effect_holds_from_the_start_on_a_gate_seeded_down(self):
        """The mode starts in the state its repair effect is held in, so the
        reference writes the repair value from t = 0: a gate seeded False
        reads True before anything fires."""
        body = expand_model(
            _model([_objflow_seeded(False, False), _delay_mode({GATE: False}, {GATE: True})])
        )
        assert _gate_trajectory(body) == [True, False, True, False, True]

    def test_the_seeded_gate_survives_the_continuous_rebuild(self):
        """A model with a continuous construct is rebuilt from its
        declarations after the expansion, carrying over what other objects
        changed. The gate's moved initial value is a MODIFICATION of an
        attribute the declaration also builds: carried over as a graft, it
        came back twice and the engine refused the duplicate."""
        well = {"type": "ObjFlow", "name": "WELL", "flows_continuous_out": [{"name": "water", "var_fed_default": 5.0}]}
        tank = {
            "type": "ObjFlow",
            "name": "TANK",
            "flows_continuous_in": [{"name": "water", "var_in_default": 0.0}],
            "capacities": [
                {"name": "vol", "flow": "water", "capacity": 100.0, "fill_rate": 1.0, "content_init": {"water": 40.0}}
            ],
        }
        document = _model(
            [_objflow_seeded(False, False), well, tank, _delay_mode({GATE: False}, {GATE: True})]
        )
        document["connections"] = [
            {"from": {"component": "WELL", "port": "water_out"}, "to": {"component": "TANK", "port": "water_in"}}
        ]
        body = expand_model(document)
        assert [a["name"] for a in _src(body)["attributes"]].count(GATE) == 1
        assert _gate_trajectory(body) == [True, False, True, False, True]

    def test_the_held_level_is_gone_from_the_expansion(self):
        """Nothing re-derives the gate any more: a level beside the edge
        writes would be refused by the engine, and would restore on repair."""
        body = expand_model(_model([_objflow(False), _delay_mode({GATE: False})]))
        writers = [
            effect
            for component in body["components"]
            for function in component.get("sensitive_functions") or []
            for effect in function.get("effects") or []
            if effect["target"] == {"component": "SRC", "attribute": GATE}
        ]
        assert writers == []

    def test_two_modes_holding_one_gate_are_refused_by_name(self):
        """The reference has no fixpoint for two held writers of one gate
        that is never reinitialized (muscadet's own write-safety note), so
        the model is refused rather than answered."""
        model = _model(
            [
                _objflow(False),
                _delay_mode({GATE: False}, name="M1"),
                _delay_mode({GATE: False}, name="M2"),
            ]
        )
        with pytest.raises(ValueError) as refusal:
            expand_model(model)
        message = str(refusal.value)
        assert f"SRC.{GATE}" in message
        assert "'M1'" in message and "'M2'" in message

    def test_a_held_write_beside_another_modes_pulse_is_refused(self):
        """Two kinds of writer on one memorised gate: the reference resolves
        them through its clamp, which nothing measured, so the model is
        refused rather than answered."""
        pulse = _delay_mode({}, name="P")
        pulse["failure_effects_trans"] = {GATE: True}
        model = _model([_objflow(False), _delay_mode({GATE: False}, name="M"), pulse])
        with pytest.raises(ValueError) as refusal:
            expand_model(model)
        message = str(refusal.value)
        assert f"SRC.{GATE}" in message and "'M'" in message and "once" in message

    def test_an_external_common_cause_latches_on_each_target(self):
        """An `external` mode's control is already the OR of its
        combinations, and each target's mirror automaton carries the edges:
        nothing to refuse there."""
        second = _objflow(False)
        second["name"] = "SRC2"
        mode = _delay_mode({GATE: False}, targets=("SRC", "SRC2"))
        mode["behaviour"] = "external"
        body = expand_model(_model([_objflow(False), second, mode]))
        assert _gate_trajectory(body) == [True, False, False, False, False]

    def test_an_internal_common_cause_mode_holding_it_is_refused(self):
        """An `internal` common-cause mode holds the gate while ANY
        combination is failed: several automata, no single edge to write on."""
        second = {"type": "ObjFlow", "name": "SRC2", "flows_out": [{"name": "power", "var_prod_cond": []}]}
        model = _model(
            [_objflow(False), second, _delay_mode({GATE: False}, targets=("SRC", "SRC2"))]
        )
        with pytest.raises(ValueError) as refusal:
            expand_model(model)
        assert "common cause" in str(refusal.value) or "combination" in str(refusal.value)

    def test_a_reinitialized_gate_the_same_mode_writes_still_holds_a_level(self):
        """The counter-case that says the latch is the control's and not the
        effect's: the very same mode on a `true` gate keeps its held level,
        which comes back up on repair."""
        for reset in (True, None):
            body = expand_model(_model([_objflow(reset), _delay_mode({GATE: False})]))
            assert _gate_trajectory(body) == [True, False, True, False, True]


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


def test_the_rebuild_refuses_an_attribute_changed_on_both_sides():
    """Carrying a changed attribute over a rebuild that changed it too would
    drop one of the two changes without a word."""
    from pyraichu.plugins.muscadet import _carry_grafts

    declared = {"name": "g", "kind": "bool", "init": {"kind": "bool", "value": True}}
    placeholder = {"name": "C", "attributes": [dict(declared, init={"kind": "bool", "value": False})]}
    pristine = {"name": "C", "attributes": [dict(declared)]}
    rebuilt = {"name": "C", "attributes": [dict(declared, kind="float")]}
    with pytest.raises(ValueError, match="changed both"):
        _carry_grafts(placeholder, pristine, rebuilt)

    rebuilt = {"name": "C", "attributes": [dict(declared)]}
    _carry_grafts(placeholder, pristine, rebuilt)
    assert rebuilt["attributes"] == placeholder["attributes"]


# --- the same three questions, asked of a muscadet DECLARATION ---------
#
# Written as `muscadet.declare.system_spec` writes it, field for field,
# because that is the document this route actually reads. The platform
# writes `var_fed_available_out_init: false` beside the reset control on
# every output of a standby channel, so the pair is declared together
# here too.


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


class TestADeclarationWritingAPersistentGate:
    """The two writers a declaration has. The standalone mode LATCHES the
    gate, as on the reference; the mode declared inside the component, whose
    availability this engine derives, is still refused."""

    #: Clear of the standalone mode's dates: failure at 4, repair at 6,
    #: failure at 10, repair at 12.
    DECLARED_SAMPLES = (2.0, 5.0, 7.0, 11.0, 13.0)

    def _trajectory(self, spec):
        document = declare.build_document(spec)
        body = pyraichu.model_body(document)
        body["indicators"] = [
            {"name": "gate", "target": "attribute", "attr": {"component": "SRC", "attribute": GATE}}
        ]
        trajectory = pyraichu.simulate(
            pyraichu.load_model(json.dumps(document)),
            t_max=self.DECLARED_SAMPLES[-1],
            samples=list(self.DECLARED_SAMPLES),
        )
        return [bool(value) for _, value in trajectory.samples["gate"]]

    def test_a_standalone_mode_holding_it_latches(self):
        spec = _declaration(init=True, modes=[_standalone_mode({GATE: False})])
        assert self._trajectory(spec) == [True, False, False, False, False]

    def test_a_standalone_mode_with_both_faces_alternates(self):
        mode = _standalone_mode({GATE: False})
        mode["repair_effects"] = {GATE: True}
        spec = _declaration(init=False, modes=[mode])
        assert self._trajectory(spec) == [True, False, True, False, True]

    def test_two_standalone_modes_holding_it_are_refused_by_name(self):
        spec = _declaration(
            modes=[
                _standalone_mode({GATE: False}, name="SRC__outage"),
                _standalone_mode({GATE: False}, name="SRC__trip"),
            ]
        )
        with pytest.raises((declare.ComponentSpecError, ValueError)) as refusal:
            declare.build_document(spec)
        message = str(refusal.value)
        assert f"SRC.{GATE}" in message
        assert "SRC__outage" in message and "SRC__trip" in message

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
