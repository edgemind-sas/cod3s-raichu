"""The SECOND shape a component declaration takes: a two-state component.

A ``system.comp`` is not only made of components holding flows. A mode
declared with ``system.add_component(cls="ObjFMDelay", fm_name=...,
targets=[...])`` is a component of its own: it carries no flow, it holds an
occurrence law, and it applies its effects to components it NAMES rather than
owns. muscadet writes it as an entry of the same ``components`` section,
telling itself apart by a ``kind`` key.

What this suite pins:

- **a document written before the key existed reads exactly as it did.**
  ``kind`` absent means the shape every stored declaration already has, so
  nothing has to be rewritten and this reader answers the old documents with
  the old answer;
- **all three vocabularies are read.** A declaration is spelled the way its
  own constructor takes it, and there are three: the ``cod3s.ObjFM`` family in
  ``failure_*`` / ``repair_*``, whose occurrence law is carried by its class,
  ``cod3s.ObjMode2S`` in ``occ_*`` / ``not_occ_*``, whose law is declared as
  data, and ``cod3s.ObjEvent``, spelled with a ``cond`` and its comparison,
  which names no target and writes nothing at all;
- **what has no counterpart is refused BY NAME.** Never as an unknown key:
  the whole cost of the shape being unread was a message listing the
  vocabulary of the other one;
- **the order of construction**, which is not a preference. A mode resolves
  its effects against the components it names, so those are built first.

Oracle-free, like the rest of this directory. That the two engines answer one
of these models alike is the validation suite's, under
`test_muscadet_engine_parity.py`.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.declare as declare

# --- the reference document -------------------------------------------
#
# Written out rather than generated, field for field as
# `muscadet.declare.system_spec` produces it: a test that built it from
# muscadet would need muscadet installed to say anything at all.

FLOW = "is_ok"
GATE = f"{FLOW}_fed_available_out"


def a_source(name="S"):
    return {
        "name": name,
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowOut",
                "name": FLOW,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_is_active_default": True,
                "var_fed_available_out_init": True,
                "var_fed_available_out_reset": True,
                "var_prod_cond_inner_mode": "or",
                "var_prod_default": True,
                "negate": False,
            }
        ],
        "capacities": [],
        "failure_modes": [],
    }


def a_block(name="B"):
    return {
        "name": name,
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowIn",
                "name": FLOW,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": False,
                "var_available_in_default": True,
                "logic": "or",
            },
            {
                "cls": "FlowOut",
                "name": FLOW,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_is_active_default": True,
                "var_fed_available_out_init": True,
                "var_fed_available_out_reset": True,
                "var_prod_cond_inner_mode": "or",
                "var_prod_default": False,
                "negate": False,
                "var_prod_cond": [[{"name": FLOW, "port": "in"}]],
            },
        ],
        "capacities": [],
        "failure_modes": [],
    }


def a_delay_mode(name="B__hw", target="B", **overrides):
    """The shape muscadet writes for `add_component(cls="ObjFMDelay", ...)`."""
    spec = {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": name.split("__", 1)[-1],
        "targets": [target],
        "target_name": target,
        "failure_effects": {GATE: False},
        "failure_param_name": ["ttf"],
        "failure_param": [4],
        "repair_param_name": ["ttr"],
        "repair_param": [2],
    }
    spec.update(overrides)
    return spec


def a_mode_2s(name="B__wear", target="B", **overrides):
    """The same mode, in the engine's own vocabulary: laws declared as data."""
    spec = {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjMode2S",
        "mode_name": name.split("__", 1)[-1],
        "targets": [target],
        "target_name": target,
        "occ_law": {"cls": "delay", "time": 4},
        "occ_effects": {GATE: False},
        "occ_param_name": ["occ_time"],
        "occ_param": [4],
        "not_occ_law": {"cls": "delay", "time": 2},
        "not_occ_param_name": ["not_occ_time"],
        "not_occ_param": [2],
    }
    spec.update(overrides)
    return spec


def a_document(*components, connections=None):
    return {
        "version": "1.0.0",
        "name": "standalone",
        "components": {entry["name"]: entry for entry in components},
        "connections": connections
        if connections is not None
        else [
            {
                "source": "S",
                "source_box": f"{FLOW}_out",
                "target": "B",
                "target_box": f"{FLOW}_in",
            }
        ],
    }


def trajectory(document, t_max=12.0):
    """The dates the built document's transitions fire at."""
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    return [(event.time, str(event)) for event in pyraichu.simulate(model, t_max=t_max).events]


# --- a document without the key reads as it always did -----------------


def test_a_document_carrying_no_kind_reads_exactly_as_before():
    """The compatibility claim, and the reason the key defaults rather than
    being required: every declaration written before it exists means the shape
    it has always meant."""
    document = a_document(a_source(), a_block())
    assert declare.build_document(document) == declare.build_system(document).build_dict()


def test_kind_flow_says_what_its_absence_says():
    document = a_document(
        {**a_source(), "kind": "flow"}, {**a_block(), "kind": "flow"}
    )
    assert declare.build_document(document) == declare.build_document(
        a_document(a_source(), a_block())
    )


def test_an_unknown_kind_is_refused_by_its_own_value():
    document = a_document(a_source(), {**a_block(), "kind": "event"})
    with pytest.raises(declare.ComponentSpecError, match="kind='event' is not one of"):
        declare.check_system_spec(document)


# --- the shape is READ, never met with "unknown declaration keys" -------


@pytest.mark.parametrize("mode", [a_delay_mode(), a_mode_2s()], ids=["ObjFM", "ObjMode2S"])
def test_a_standalone_mode_is_read_and_builds(mode):
    """The whole ticket in one assertion: the entry that used to be refused
    for nine unknown keys builds the automaton it declares."""
    document = a_document(a_source(), a_block(), mode)
    fired = trajectory(document)
    assert [round(time, 6) for time, _ in fired] == [4.0, 6.0, 10.0, 12.0], fired


@pytest.mark.parametrize(
    "mode, unknown",
    [
        (a_delay_mode(failure_rate=0.1), "failure_rate"),
        (a_mode_2s(failure_param=[4]), "failure_param"),
    ],
    ids=["ObjFM", "ObjMode2S"],
)
def test_an_unknown_key_names_the_vocabulary_of_its_own_class(mode, unknown):
    """The refusal a modeller acts on: it says which spelling the class takes,
    not which one the OTHER shape takes.

    `failure_param` on an `ObjMode2S` is the trap this is here for: it is a
    real key of the other vocabulary, so a reader with one vocabulary would
    accept it and build a mode with no law.
    """
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.check_mode_spec(mode)
    message = str(refused.value)
    assert f"'{unknown}'" in message
    assert mode["cls"] in message
    assert "unknown declaration key" in message


def test_a_mode_class_carrying_no_law_is_refused_by_name():
    """`ObjFM` and `ObjFailureMode` are bases: neither declares an occurrence
    law, so a mode declared as one has no transition to build."""
    with pytest.raises(declare.ComponentSpecError, match="is not a mode class"):
        declare.check_mode_spec(a_delay_mode(cls="ObjFM"))


def test_a_multi_state_degradation_mode_is_refused_by_name():
    """`ObjDegMode` is not the shape this `kind` names: it has more than two
    states, so it is refused for what it IS."""
    with pytest.raises(declare.ComponentSpecError, match="MULTI-state"):
        declare.check_mode_spec(a_delay_mode(cls="ObjDegMode"))


# --- the two vocabularies, and the law each one carries -----------------


def test_the_objfm_family_carries_its_law_in_its_class():
    exponential = declare.mode_object(
        a_delay_mode(
            cls="ObjFMExp",
            failure_param_name=["lambda"],
            failure_param=[0.25],
            repair_param_name=["mu"],
            repair_param=[0.5],
        )
    )
    assert exponential["failure"] == [{"law": "exp", "rate": 0.25}]
    assert exponential["repair"] == [{"law": "exp", "rate": 0.5}]
    delay = declare.mode_object(a_delay_mode())
    assert delay["failure"] == [{"law": "delay", "time": 4.0}]


def test_objmode2s_carries_its_law_as_data():
    """The one substantive difference between the two vocabularies."""
    obj = declare.mode_object(
        a_mode_2s(
            occ_law={"cls": "exp", "rate": 0.1},
            occ_param_name=["occ_rate"],
            occ_param=[0.1],
            not_occ_law={"cls": "exp", "rate": 0.5},
            not_occ_param_name=["not_occ_rate"],
            not_occ_param=[0.5],
        )
    )
    assert obj["failure"] == [{"law": "exp", "rate": 0.1}]
    assert obj["repair"] == [{"law": "exp", "rate": 0.5}]
    assert obj["failure_effects"] == {GATE: False}


def test_a_parameter_name_disagreeing_with_its_law_is_refused():
    """RAICHU bakes the value and has no parameter variable, so the name says
    nothing -- but a delay bound to `lambda` is a declaration contradicting
    itself, and reading past it builds the law the class says."""
    with pytest.raises(declare.ComponentSpecError, match="failure_param_name"):
        declare.check_mode_spec(a_delay_mode(failure_param_name=["lambda"]))


def test_both_defaults_of_a_parameter_name_are_accepted():
    """Two writers derive two spellings for one law: the facade names it after
    the law, the engine after the direction and the law's field."""
    declare.check_mode_spec(a_mode_2s(occ_param_name=["occ_time"]))
    declare.check_mode_spec(a_mode_2s(occ_param_name=["ttf"]))


# --- the two traps muscadet hit, and this reader would have -------------


def test_a_list_at_a_common_cause_order_means_what_the_tuple_means():
    """A document has no tuples, and the engine reads one. Round-tripped
    through JSON, the per-order vector of a mode declared with one rate per
    order comes back as `[1.0, [0], [0]]`."""
    obj = declare.mode_object(
        a_delay_mode(
            targets=["B1", "B2"],
            target_name="BX",
            name="BX__hw",
            failure_param=[[4], 7],
            repair_param=[2, [3]],
        )
    )
    assert obj["failure"] == [{"law": "delay", "time": 4.0}, {"law": "delay", "time": 7.0}]
    assert obj["repair"] == [{"law": "delay", "time": 2.0}, {"law": "delay", "time": 3.0}]


def test_an_order_declaring_two_parameters_is_refused_rather_than_truncated():
    with pytest.raises(declare.ComponentSpecError, match="order 1"):
        declare.mode_object(a_delay_mode(failure_param=[[4, 5]]))


def test_the_components_are_built_before_the_modes_that_name_them():
    """Not a preference: a mode resolves its effects against every component
    it names, so a document listing the mode FIRST has to build it last."""
    mode = a_delay_mode()
    listed_first = {
        "version": "1.0.0",
        "name": "standalone",
        "components": {
            mode["name"]: mode,
            "S": a_source(),
            "B": a_block(),
        },
        "connections": a_document()["connections"],
    }
    names = [
        component["name"]
        for component in declare.build_document(listed_first)["components"]
    ]
    assert names.index("B") < names.index("B__hw")


# --- what has no counterpart, refused by name --------------------------


def test_an_effect_naming_an_attribute_the_target_has_not_is_refused():
    """An effect on a variable no flow of the target generates: refused, and
    told what the component DOES carry that a mode writes, because that list
    is what a modeller has to choose from."""
    document = a_document(
        a_source(),
        a_block(),
        a_delay_mode(failure_effects={f"{FLOW}_throughput": True}),
    )
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    message = str(refused.value)
    assert f"B.{FLOW}_throughput" in message
    assert GATE in message, "the refusal names what the component does carry"


# A mode that LATCHES a dormant output on is carried, and lives in
# `test_dormant_function.py`: the block above declares a production
# condition, which is the one regime the latch is refused in.


def test_a_self_hosted_mode_is_refused_by_what_it_is():
    with pytest.raises(declare.ComponentSpecError, match="SELF-HOSTED"):
        declare.check_mode_spec(a_mode_2s(targets=None))


def a_latching_block(name="B"):
    """`a_block` whose output gate is PERSISTENT: what a one-shot effect
    latches (muscadet `fed_available_reset: false`)."""
    block = a_block(name)
    block["flows"][1]["var_fed_available_out_reset"] = False
    return block


def _transitions(document):
    built = declare.build_document(document)
    body = built.get("model", built)
    return {
        (component["name"], automaton["name"], transition["name"]): transition
        for component in body["components"]
        for automaton in component.get("automata", [])
        for transition in automaton["transitions"]
    }


def test_a_one_shot_effect_is_written_on_the_firing_edge():
    """A one-shot effect is written ONCE, on the edge that fires: the mode's
    failure transition carries it, and the repair transition, which declares
    none, writes nothing back, so the persistent gate stays down."""
    mode = a_delay_mode(failure_effects={}, failure_effects_trans={GATE: False})
    transitions = _transitions(a_document(a_source(), a_latching_block(), mode))
    failure = next(t for (_, _, name), t in transitions.items() if name == "failure")
    repair = next(t for (_, _, name), t in transitions.items() if name == "repair")
    assert failure["effects"] == [
        {"target": {"component": "B", "attribute": GATE},
         "value": {"op": "const", "value": {"kind": "bool", "value": False}}}
    ]
    assert "effects" not in repair


def test_a_one_shot_effect_on_a_gate_reset_every_step_is_refused():
    """The reference undoes the pulse at the next step; here it would stick."""
    mode = a_delay_mode(failure_effects={}, failure_effects_trans={GATE: False})
    with pytest.raises(Exception, match="reinitializes that gate"):
        declare.build_document(a_document(a_source(), a_block(), mode))


def test_a_variable_driven_both_ways_is_refused():
    """A level and a pulse on one variable: the level overwrites the pulse."""
    mode = a_delay_mode(failure_effects_trans={GATE: False})
    with pytest.raises(Exception, match="driven both as a level"):
        declare.build_document(a_document(a_source(), a_latching_block(), mode))


def test_a_renamed_common_cause_automaton_is_refused():
    """The templates name automata that indicators and sequences reach by
    name, and this layer names them itself."""
    with pytest.raises(declare.ComponentSpecError, match="trans_name_prefix"):
        declare.check_mode_spec(a_delay_mode(trans_name_prefix="__bin_{target_binary}"))


def test_a_naming_function_is_refused_as_a_function():
    with pytest.raises(declare.ComponentSpecError, match="trans_name_prefix_fun"):
        declare.check_mode_spec(a_delay_mode(trans_name_prefix_fun=lambda **_: ""))


def test_a_document_name_disagreeing_with_the_derived_one_is_refused():
    """The engine derives `{target_name}__{fm_name}`, and the document files
    the mode under it: an entry keyed otherwise is one nothing can reach."""
    with pytest.raises(declare.ComponentSpecError, match="names something"):
        declare.check_mode_spec(a_delay_mode(name="elsewhere"))


def test_a_condition_leaf_with_no_object_is_refused_on_a_common_cause_mode():
    """cod3s resolves it per combination and one condition is shared by all of
    them here, so the case is refused rather than resolved to one target."""
    with pytest.raises(declare.ComponentSpecError, match="names no `obj`"):
        declare.mode_object(
            a_delay_mode(
                name="BX__hw",
                target_name="BX",
                targets=["B1", "B2"],
                failure_param=[4, 7],
                repair_param=[2, 3],
                failure_cond=[[{"attr": f"{FLOW}_fed_in", "value": True}]],
            )
        )


def test_drop_inactive_automata_is_read_by_what_it_asks_for():
    """muscadet writes `False` on every common-cause mode whose orders were
    all built, which is the ordinary case and says nothing here. It asks for
    something only where an order has no active law, and there this layer has
    no automaton that can never fire to build."""
    ordinary = a_delay_mode(
        name="BX__hw",
        target_name="BX",
        targets=["B1", "B2"],
        failure_param=[4, 7],
        repair_param=[2, 3],
        drop_inactive_automata=False,
    )
    assert declare.check_mode_spec(ordinary) == "BX__hw"

    with pytest.raises(declare.ComponentSpecError, match="order 2 inactive"):
        declare.check_mode_spec(
            {**ordinary, "cls": "ObjFMExp",
             "failure_param_name": ["lambda"], "repair_param_name": ["mu"],
             "failure_param": [0.1, 0.0], "repair_param": [0.5, 0.5]}
        )


# --- the arbitration: a mode that gates nothing ------------------------


def test_a_standalone_mode_with_no_effect_builds():
    """The ticket's arbitration, and it goes the other way from the
    on-component shape.

    On a component, a mode's `targets` names FLOWS and an empty list means ALL
    of them, so muscadet's "clamps nothing" and this layer's "clamps
    everything" collide and the absence has to be refused. A standalone mode's
    `targets` names COMPONENTS and its grip is written out as the effects it
    applies to named attributes: it writes what it names and nothing else, so
    an empty `failure_effects` means the same thing on both sides. That is
    exactly `mdc_a` and `mdc_phishing` of muscadet's two interactive examples:
    a state-only mode another mode's condition watches.
    """
    document = a_document(
        a_source(), a_block(), a_delay_mode(failure_effects={}, repair_param=[1e9])
    )
    built = declare.build_document(document)
    mode = next(c for c in built["components"] if c["name"] == "B__hw")
    assert [a["name"] for a in mode["automata"]] == ["fm"]
    assert mode["sensitive_functions"] == [], "it writes nothing, and says so"
    # And the block keeps its output: nothing gates it.
    assert [round(t, 6) for t, _ in trajectory(document)] == [4.0]


def test_an_on_component_mode_gating_nothing_is_still_refused():
    """The rule that does NOT extend, and the reason it does not: there,
    `targets` names flows and an empty list means every one of them."""
    block = a_block()
    block["failure_modes"] = [
        {"cls": "delay", "name": "hw", "failure_time": 4, "repair_time": 2}
    ]
    with pytest.raises(declare.SystemSpecError, match="gates none of the discrete"):
        declare.build_document(a_document(a_source(), block))


def test_a_mode_watching_another_modes_state_reads_it_as_a_state():
    """The cascade. cod3s reaches a mode's states through the same `attr` key
    it reaches a variable with; here a state and a variable are two different
    references, and reading the first as the second compares a state name to a
    boolean."""
    obj = declare.mode_object(
        a_delay_mode(name="B__attack", failure_cond=[[{"attr": "occ", "obj": "B__hw", "value": True}]]),
        {"B__hw": a_delay_mode(), "B": a_block()},
    )
    assert obj["failure_cond"] == [
        [{"obj": "B__hw", "automaton": "fm", "state": "occ", "value": True}]
    ]


def test_a_state_a_watched_mode_has_not_is_refused_by_the_states_it_has():
    with pytest.raises(declare.ComponentSpecError, match="only the states"):
        declare.mode_object(
            a_delay_mode(
                name="B__attack",
                failure_cond=[[{"attr": "broken", "obj": "B__hw", "value": True}]],
            ),
            {"B__hw": a_delay_mode(), "B": a_block()},
        )


def test_a_condition_watching_a_variable_the_target_has_not_is_refused():
    """The read side of the same rule: a condition naming an attribute this
    layer does not generate is a guard the engine would refuse without naming
    the mode that wrote it."""
    document = a_document(
        a_source(),
        a_block(),
        a_delay_mode(failure_cond=[[{"attr": f"{FLOW}_throughput", "value": True}]]),
    )
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    assert f"B.{FLOW}_throughput" in str(refused.value)


def test_a_condition_may_watch_the_production_a_mode_latches():
    """The read side of the shape above: `{flow}_prod_available` is a variable
    now, so the next stage of a cascade may arm itself on it."""
    document = a_document(
        a_source(),
        a_block(),
        a_delay_mode(
            failure_cond=[[{"attr": f"{FLOW}_prod_available", "value": True}]]
        ),
    )
    assert declare.build_document(document)["components"]


def test_a_mode_may_watch_a_mode_the_document_lists_after_it():
    """Checked once every mode is expanded, and not one at a time: an
    automaton does not exist until the mode that carries it is expanded."""
    watcher = a_delay_mode(
        name="B__attack",
        failure_cond=[[{"attr": "occ", "obj": "B__hw", "value": True}]],
        failure_param=[1],
        repair_param=[1e9],
        repair_cond=False,
    )
    document = {
        "version": "1.0.0",
        "name": "standalone",
        "components": {
            "B__attack": watcher,
            "B__hw": a_delay_mode(),
            "S": a_source(),
            "B": a_block(),
        },
        "connections": a_document()["connections"],
    }
    # `B__hw` fails at 4 and repairs at 6; `B__attack`, armed by its `occ`
    # state, fires one unit after it and never repairs.
    assert [round(t, 6) for t, _ in trajectory(document, t_max=6.0)] == [4.0, 5.0, 6.0]


def test_a_mode_naming_a_component_the_document_has_not_is_refused():
    with pytest.raises(declare.ComponentSpecError, match="does not declare"):
        declare.build_document(
            a_document(
                a_source(),
                a_block(),
                a_delay_mode(name="ELSEWHERE__hw", target="ELSEWHERE"),
            )
        )


# --- the muscadet facade spells its effects over FLOWS ------------------


def test_the_muscadet_facade_names_flows_where_cod3s_names_variables():
    """Both spellings are load-bearing in models that exist, and the CLASS is
    what tells them apart."""
    obj = declare.mode_object(
        a_delay_mode(cls="ObjFailureModeDelay", failure_effects={FLOW: False}),
        {"B": a_block()},
    )
    assert obj["failure_effects"] == {GATE: False}


def test_a_facade_pattern_naming_no_output_is_refused_by_the_outputs_there_are():
    with pytest.raises(declare.ComponentSpecError, match="names no discrete output"):
        declare.mode_object(
            a_delay_mode(cls="ObjFailureModeDelay", failure_effects={"absent": False}),
            {"B": a_block()},
        )


# --- the system scale answers the system, the document the document -----


def test_build_system_refuses_a_standalone_mode_and_names_where_it_goes():
    """It holds no flow and is wired to nothing: dropping it would lose the
    model, so it is refused, naming the entry point that carries it."""
    document = a_document(a_source(), a_block(), a_delay_mode())
    with pytest.raises(declare.SystemSpecError, match="build_document"):
        declare.build_system(document)


def test_build_component_refuses_a_mode_and_names_where_it_goes():
    with pytest.raises(declare.ComponentSpecError, match="build_document"):
        declare.check_spec.__globals__["_plan"](a_delay_mode(), {})


# --- the THIRD family under the same kind: an event ---------------------
#
# `cod3s.ObjEvent` is a two-state component like the two mode families, and
# shares almost nothing else with them: it is SELF-HOSTED, so it names no
# target, writes no attribute and has no common-cause order, and its whole
# declaration is a condition over arbitrary components with the comparison it
# is tested by. The COD3S Platform synthesises one per study event and one per
# indicator whose formula has more than one clause, so a document carrying one
# is an ordinary document -- and until it was read, one was refused entire.


def an_event(name="DEGRADED", **overrides):
    """The shape muscadet writes for `add_component(cls="ObjEvent", ...)`."""
    spec = {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjEvent",
        "cond": [[{"attr": "occ", "obj": "B__hw", "value": True}]],
    }
    spec.update(overrides)
    return spec


def test_an_event_is_read_and_builds():
    """The whole ticket in one assertion: the entry that used to be refused
    by its class builds the automaton it declares, and its own dates are
    neither the mode's nor the condition's.

    `B__hw` fails at 4 and repairs at 6; the event waits a quarter to occur
    and three quarters to return, so it moves at 4.25 and 6.75.
    """
    document = a_document(
        a_source(),
        a_block(),
        a_delay_mode(),
        an_event(tempo_occ=0.25, tempo_not_occ=0.75),
    )
    fired = trajectory(document, t_max=9.0)
    assert [round(time, 6) for time, _ in fired] == [4.0, 4.25, 6.0, 6.75], fired


def test_the_whole_event_vocabulary_is_carried():
    """Every key `cod3s.ObjEvent.__init__` takes, and each one honoured: the
    condition and its comparison, the two tempos, the three names and the two
    truth functions carried by NAME."""
    obj = declare.mode_object(
        an_event(
            cond=[[{"attr": "occ", "obj": "B__hw", "value": True}]],
            cond_operator="!=",
            cond_value=False,
            tempo_occ=1.5,
            tempo_not_occ=0.5,
            event_aut_name="alarm",
            occ_state_name="ringing",
            not_occ_state_name="silent",
            inner_logic="any",
            outer_logic="all",
        ),
        {"B__hw": a_delay_mode(), "B": a_block()},
    )
    assert obj == {
        "type": "ObjEvent",
        "name": "DEGRADED",
        "cond": [[{"obj": "B__hw", "automaton": "fm", "state": "occ", "value": True}]],
        "cond_operator": "!=",
        "cond_value": False,
        "tempo_occ": 1.5,
        "tempo_not_occ": 0.5,
        "event_aut_name": "alarm",
        "occ_state_name": "ringing",
        "not_occ_state_name": "silent",
        "inner_logic": "any",
        "outer_logic": "all",
    }


def test_the_event_vocabulary_reaches_the_automaton_it_names():
    """The names are not decoration: they are what the automaton and its two
    states are BUILT under, which is what a state indicator and another
    event's condition reach them by."""
    document = a_document(
        a_source(),
        a_block(),
        a_delay_mode(),
        an_event(
            event_aut_name="alarm",
            occ_state_name="ringing",
            not_occ_state_name="silent",
        ),
    )
    built = declare.build_document(document)
    body = pyraichu.model_body(built) if "model" in built else built
    event = next(c for c in body["components"] if c["name"] == "DEGRADED")
    automaton = event["automata"][0]
    assert automaton["name"] == "alarm"
    assert automaton["states"] == ["silent", "ringing"]
    assert automaton["init"] == "silent"


@pytest.mark.parametrize(
    "operator, expected",
    [("==", "eq"), ("!=", "ne"), ("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge")],
)
def test_every_comparison_the_three_layers_share_is_carried(operator, expected):
    """muscadet's `MODE_OPERATORS`, cod3s' compiled operators and the plugin's
    `_OPE` spell the same six, so nothing is translated -- but a seventh is
    refused here rather than reaching the plugin as a bare `KeyError`."""
    document = a_document(
        a_source(),
        a_block(),
        a_delay_mode(),
        # `cond_value=False` keeps the comparison in the tree: `==` against
        # `True` is the shortcut cod3s and the plugin both take, which emits
        # the guard bare.
        an_event(cond_operator=operator, cond_value=False),
    )
    built = declare.build_document(document)
    body = pyraichu.model_body(built) if "model" in built else built
    event = next(c for c in body["components"] if c["name"] == "DEGRADED")
    guard = event["automata"][0]["transitions"][0]["guard"]
    assert guard["op"] == "cmp" and guard["cmp"] == expected


def test_an_unreadable_comparison_is_refused_by_the_six_there_are():
    with pytest.raises(declare.ComponentSpecError, match="cond_operator"):
        declare.check_mode_spec(an_event(cond_operator="=~"))


def test_an_event_that_watches_nothing_is_refused():
    """`cond` is what an event IS: one that watches nothing never fires, and
    is a component declared for no reason."""
    event = an_event()
    del event["cond"]
    with pytest.raises(declare.ComponentSpecError, match="watches nothing"):
        declare.check_mode_spec(event)


def test_a_truth_function_is_carried_by_name_on_the_keys_an_event_spells():
    """The engine calls them `cond_inner_logic` / `cond_outer_logic` and an
    event calls them `inner_logic` / `outer_logic`: a key a constructor does
    not take is a key nothing builds."""
    with pytest.raises(declare.ComponentSpecError, match="not one of"):
        declare.check_mode_spec(an_event(inner_logic="sum"))
    with pytest.raises(declare.ComponentSpecError, match="unknown declaration key"):
        declare.check_mode_spec(an_event(cond_inner_logic="all"))


def test_an_event_names_no_target():
    """`targets` is the mode families' key, and not an event's: an event
    observes and writes nothing, so a target list has nothing to mean."""
    with pytest.raises(declare.ComponentSpecError, match="unknown declaration key"):
        declare.check_mode_spec(an_event(targets=["B"]))


def test_a_common_cause_gate_is_refused_on_an_event():
    """One automaton and no common-cause order, so there is none to drop."""
    with pytest.raises(declare.ComponentSpecError, match="no common-cause order"):
        declare.check_mode_spec(an_event(drop_inactive_automata=False))


def test_an_event_leaf_with_no_object_is_refused_by_naming_it():
    """A leaf with no `obj` reads the mode's own target, and an event names
    none: cod3s compiles an event's tree with a system-wide resolution, so a
    leaf has to say what it watches. Refused rather than falling over the
    empty target list with a bare `IndexError`."""
    with pytest.raises(declare.ComponentSpecError, match="names no `obj`"):
        declare.mode_object(an_event(cond=[[{"attr": f"{FLOW}_fed_in", "value": True}]]))


def test_an_event_condition_on_a_modes_state_is_read_as_a_state():
    """The same rewrite a mode's condition gets, and it has to be the same
    one: cod3s reaches a mode's states through the `attr` key it reaches a
    variable with, and here a state and a variable are two references."""
    obj = declare.mode_object(
        an_event(), {"B__hw": a_delay_mode(), "B": a_block()}
    )
    assert obj["cond"] == [
        [{"obj": "B__hw", "automaton": "fm", "state": "occ", "value": True}]
    ]


def test_a_condition_on_an_EVENTS_state_reads_the_event_own_names():
    """The symmetric hole, and the one nothing else would report. An event
    entering `MODE_CLASSES` puts it in the dictionary a condition leaf
    consults, and the mode vocabulary would answer `fm` for its automaton and
    refuse `not_occ` as a state it has not."""
    watched = an_event("FIRST")
    watcher = an_event(
        "SECOND", cond=[[{"attr": "not_occ", "obj": "FIRST", "value": True}]]
    )
    obj = declare.mode_object(watcher, {"FIRST": watched})
    assert obj["cond"] == [
        [{"obj": "FIRST", "automaton": "ev", "state": "not_occ", "value": True}]
    ]


def test_a_condition_on_a_RENAMED_events_state_follows_the_rename():
    watched = an_event(
        "FIRST",
        event_aut_name="alarm",
        occ_state_name="ringing",
        not_occ_state_name="silent",
    )
    watcher = an_event(
        "SECOND", cond=[[{"attr": "ringing", "obj": "FIRST", "value": True}]]
    )
    obj = declare.mode_object(watcher, {"FIRST": watched})
    assert obj["cond"] == [
        [{"obj": "FIRST", "automaton": "alarm", "state": "ringing", "value": True}]
    ]


def test_a_state_an_event_has_not_is_refused_by_the_two_it_has():
    """`rep` is the mode vocabulary's return state, and an event's is
    `not_occ`: read with the wrong vocabulary this refusal would be the other
    way round."""
    watcher = an_event(
        "SECOND", cond=[[{"attr": "rep", "obj": "FIRST", "value": True}]]
    )
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.mode_object(watcher, {"FIRST": an_event("FIRST")})
    message = str(refused.value)
    assert "an event" in message
    assert "['not_occ', 'occ']" in message


def test_an_event_watching_a_component_the_document_has_not_is_refused():
    """An event observes arbitrary components, so nothing upstream has
    already checked that the one it names exists."""
    document = a_document(
        a_source(),
        a_block(),
        an_event(cond=[[{"attr": f"{FLOW}_fed_in", "obj": "GHOST", "value": True}]]),
    )
    with pytest.raises(declare.ComponentSpecError, match="the model does not hold"):
        declare.build_document(document)


def test_a_refusal_about_an_event_calls_it_an_event():
    """It is not a failure mode, and a message calling it one sends a
    modeller looking for the targets and the effects it has none of."""
    document = a_document(
        a_source(),
        a_block(),
        an_event(cond=[[{"attr": "nowhere", "obj": "B", "value": True}]]),
    )
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    assert str(refused.value).startswith("Event DEGRADED")
