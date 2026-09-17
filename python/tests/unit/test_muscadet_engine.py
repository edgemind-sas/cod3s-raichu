"""RAICHU as an engine of muscadet, proved without muscadet being installed.

muscadet exposes an extension point and imports no engine; `pyraichu`
registers there and becomes the engine a muscadet run may select. This module
pins the RAICHU side of that seam, and it pins it **oracle-free**: everything
here is either the distribution's own metadata, the declaration reader, or a
trajectory RAICHU computes on its own. What needs PyCATSHOO alive -- the same
model answering alike on both engines -- belongs to the cross-validation
suite's `test_muscadet_engine_parity.py`, which runs where a PyCATSHOO
installation is available.

Four things are pinned, and the first two are the ones a modeller feels:

1. **the distribution advertises the engine.** The registration is packaging,
   not code: an installed pyraichu is a registered engine, so a study naming
   `engine="raichu"` imports nothing;
2. **pyraichu never imports muscadet.** The mirror of muscadet's own guard.
   Reached from either side, the seam has to hold with only one of the two
   packages installed, and this is the side that ships in wheels;
3. **the system declaration builds**, wiring included, which is what a
   component-scale declaration could not carry;
4. **what has no counterpart is refused BY NAME.** The two layers spell a
   failure mode's grip on a discrete output differently and name an
   automaton's states differently; neither is guessed.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import types

import pytest

import pyraichu
import pyraichu.declare as declare
import pyraichu.muscadet_engine as engine

# --- the declaration, in muscadet's own vocabulary ---------------------
#
# Written out rather than generated: this is the document `muscadet.declare.
# system_spec` produces, field for field, and a test that built it from
# muscadet would need muscadet installed to say anything at all. Two blocks in
# parallel behind one source, one of them failing on a deterministic law, and
# a target that needs both -- the smallest boolean model whose answer moves.

FLOW = "is_ok"


def _flow_in(logic="or"):
    return {
        "cls": "FlowIn",
        "name": FLOW,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_in_default": False,
        "var_available_in_default": True,
        "logic": logic,
    }


def _flow_out(prod_cond=None, prod_default=False):
    entry = {
        "cls": "FlowOut",
        "name": FLOW,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_is_active_default": True,
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_cond_inner_mode": "or",
        "var_prod_default": prod_default,
        "negate": False,
    }
    if prod_cond is not None:
        entry["var_prod_cond"] = prod_cond
    return entry


def _component(name, source_cls, flows, failure_modes=()):
    return {
        "name": name,
        "cls": "ObjFlow",
        "source_cls": source_cls,
        "flows": list(flows),
        "capacities": [],
        "measurements_in": [],
        "measurements_out": [],
        "rules": [],
        "transfers": [],
        "automata": [],
        "failure_modes": list(failure_modes),
    }


def _delay_mode(name, failure_time, repair_time, effects):
    return {
        "cls": "delay",
        "name": name,
        "failure_state": "occ",
        "failure_cond": f"{FLOW}_fed_out",
        "failure_time": failure_time,
        "failure_effects": effects,
        "failure_param_name": "ttf",
        "repair_state": "rep",
        "repair_cond": True,
        "repair_time": repair_time,
        "repair_effects": [],
        "repair_param_name": "ttr",
    }


def rbd_declaration():
    """The document, fresh each time: a build consumes what it is handed."""
    produced = [[{"name": FLOW, "port": "in"}]]
    return {
        "version": "1.0.0",
        "name": "rbd",
        # The study observes every flow variable, not only the one it names:
        # a declaration that wants the generated set says so.
        "generated_indicators": True,
        "components": {
            "S": _component("S", "Source", [_flow_out(prod_default=True)]),
            "B1": _component(
                "B1",
                "Block",
                [_flow_in(), _flow_out(prod_cond=produced)],
                [
                    _delay_mode(
                        "failure",
                        4,
                        2,
                        [[f"{FLOW}_fed_available_out", False]],
                    )
                ],
            ),
            "B2": _component("B2", "Block", [_flow_in(), _flow_out(prod_cond=produced)]),
            "T": _component("T", "Target", [_flow_in(logic="and")]),
        },
        "connections": [
            {
                "source": source,
                "source_box": f"{FLOW}_out",
                "target": target,
                "target_box": f"{FLOW}_in",
                "flow": FLOW,
            }
            for source, target in (("S", "B1"), ("S", "B2"), ("B1", "T"), ("B2", "T"))
        ],
        "indicators": [
            {
                "name": f"T_{FLOW}_fed_in",
                "label": f"T_{FLOW}_fed_in",
                "measure": "value",
                "stats": ["mean"],
                "component": "T",
                "operator": "==",
                "var": f"{FLOW}_fed_in",
                "kind": "PycVarIndicator",
            }
        ],
    }


#: The dates B1's mode fires at, and therefore the dates the target loses and
#: recovers its feed: failure at 4 then every 6, repair 2 later.
FAILURE_DATES = (4.0, 10.0, 16.0, 22.0)
REPAIR_DATES = (6.0, 12.0, 18.0, 24.0)


# --- 1. the registration is packaging ---------------------------------


def test_the_distribution_advertises_the_engine():
    """Installed is enough: the study that selects RAICHU imports nothing.

    Read off the DISTRIBUTION and not off this checkout, because the claim is
    about what a wheel carries. A source tree on `PYTHONPATH` carries no
    metadata at all and has nothing to say here.
    """
    try:
        distribution = importlib.metadata.distribution("pyraichu")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("pyraichu is on the path but not installed as a distribution")
    advertised = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "muscadet.engines"
    }
    assert advertised == {engine.ENGINE_NAME: "pyraichu.muscadet_engine:register"}


def test_registering_hands_muscadet_both_runners():
    """The hook's shape, against a stand-in for muscadet.

    A stand-in and not the real thing: what is under test is what pyraichu
    SAYS at the extension point, and saying it must not require the package on
    the other side of the seam to be installed.
    """
    registered = {}
    stand_in = types.ModuleType("muscadet")
    stand_in.register_engine = lambda **kwargs: registered.update(kwargs)
    previous = sys.modules.get("muscadet")
    sys.modules["muscadet"] = stand_in
    try:
        engine.register()
    finally:
        if previous is None:
            del sys.modules["muscadet"]
        else:
            sys.modules["muscadet"] = previous

    assert registered["name"] == engine.ENGINE_NAME
    assert registered["simulate"] is engine.simulate
    assert registered["isimu_start"] is engine.isimu_start
    assert registered["description"] == engine.ENGINE_DESCRIPTION
    # Registering twice is this engine taking its own name over -- an explicit
    # import followed by entry-point discovery -- never two packages racing.
    assert registered["replace"] is True


def test_importing_pyraichu_does_not_import_muscadet():
    """The mirror of muscadet's own guard, on the side that ships in wheels.

    In a subprocess because the assertion is about a fresh interpreter: this
    one has imported half the package already.
    """
    probe = (
        "import sys, pyraichu, pyraichu.muscadet_engine, pyraichu.declare;"
        "assert 'muscadet' not in sys.modules, sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


# --- 2. the declaration builds ----------------------------------------


def test_the_system_declaration_builds_its_components_and_its_wiring():
    system = declare.build_system(rbd_declaration())
    assert sorted(system.comp) == ["B1", "B2", "S", "T"]
    assert {
        (link["from"]["component"], link["to"]["component"])
        for link in system._connections
    } == {("S", "B1"), ("S", "B2"), ("B1", "T"), ("B2", "T")}


def test_the_document_runs_and_its_target_follows_the_failing_block():
    result = engine.simulate(
        rbd_declaration(),
        {"nb_runs": 1, "schedule": [0.0, 5.0, 9.0, 17.0, 21.0], "seed": 7},
    )
    fed = result.indicators[f"T_{FLOW}_fed_in"]
    assert list(fed.instants) == [0.0, 5.0, 9.0, 17.0, 21.0]
    # The block is down over [4, 6) and [16, 18): 5 and 17 fall inside one of
    # those windows, 9 and 21 between two.
    assert list(fed.mean) == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_the_declared_indicator_is_the_one_the_model_carries():
    model = engine.build_model(rbd_declaration())
    body = pyraichu.model_body(json.loads(model.json))
    declared = [
        entry for entry in body["indicators"] if entry["name"] == f"T_{FLOW}_fed_in"
    ]
    assert declared == [
        {
            "name": f"T_{FLOW}_fed_in",
            "target": "attribute",
            "attr": {"component": "T", "attribute": f"{FLOW}_fed_in"},
        }
    ]


def test_an_indicator_the_layer_would_not_have_emitted_is_added():
    """A renamed indicator, which the generated naming cannot produce."""
    spec = rbd_declaration()
    spec["indicators"][0]["name"] = "availability"
    model = engine.build_model(spec)
    body = pyraichu.model_body(json.loads(model.json))
    named = {entry["name"] for entry in body["indicators"]}
    assert "availability" in named and f"T_{FLOW}_fed_in" in named


def an_event(name="DEGRADED", **overrides):
    """An event watching `B1`'s input, as muscadet writes one."""
    spec = {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjEvent",
        "cond": [[{"obj": "B1", "attr": f"{FLOW}_fed_in", "value": False}]],
    }
    spec.update(overrides)
    return spec


def test_a_state_indicator_on_anything_but_an_event_is_refused_by_name():
    """Outside an event, the two layers name automata and their states
    differently, and the refusal says which component family carries one."""
    spec = rbd_declaration()
    spec["indicators"] = [
        {
            "name": "B1_down",
            "kind": "PycSTIndicator",
            "component": "B1",
            "state": "failure_occ",
        }
    ]
    with pytest.raises(declare.SystemSpecError, match="does not declare as an EVENT"):
        engine.build_model(spec)


def test_a_state_indicator_on_an_event_names_the_automaton_the_event_declares():
    """The one case where the translation IS declared: the event states its
    automaton and its two states, and both layers build them under those
    names."""
    spec = rbd_declaration()
    spec["components"]["DEGRADED"] = an_event()
    spec["indicators"] = [
        {
            "name": "DEGRADED_occ",
            "kind": "PycSTIndicator",
            "component": "DEGRADED",
            "state": "occ",
            "stats": ["mean"],
        }
    ]
    body = pyraichu.model_body(json.loads(engine.build_model(spec).json))
    declared = [
        entry for entry in body["indicators"] if entry["name"] == "DEGRADED_occ"
    ]
    assert declared == [
        {
            "name": "DEGRADED_occ",
            "target": "state",
            "component": "DEGRADED",
            "automaton": "ev",
            "state": "occ",
        }
    ]


def test_a_state_indicator_follows_the_names_the_event_renamed():
    """Read through the event's own declaration, not through the defaults: an
    event that renamed its automaton is watched under the new name."""
    spec = rbd_declaration()
    spec["components"]["DEGRADED"] = an_event(
        event_aut_name="alarm", occ_state_name="ringing", not_occ_state_name="silent"
    )
    spec["indicators"] = [
        {
            "name": "DEGRADED_ringing",
            "kind": "PycSTIndicator",
            "component": "DEGRADED",
            "state": "ringing",
        }
    ]
    body = pyraichu.model_body(json.loads(engine.build_model(spec).json))
    declared = [
        entry for entry in body["indicators"] if entry["name"] == "DEGRADED_ringing"
    ]
    assert declared == [
        {
            "name": "DEGRADED_ringing",
            "target": "state",
            "component": "DEGRADED",
            "automaton": "alarm",
            "state": "ringing",
        }
    ]


def test_a_state_the_event_has_not_is_refused_by_the_two_it_has():
    spec = rbd_declaration()
    spec["components"]["DEGRADED"] = an_event()
    spec["indicators"] = [
        {
            "name": "DEGRADED_rep",
            "kind": "PycSTIndicator",
            "component": "DEGRADED",
            "state": "rep",
        }
    ]
    with pytest.raises(declare.SystemSpecError, match=r"which holds \['not_occ', 'occ'\]"):
        engine.build_model(spec)


def test_a_state_indicator_is_read_under_the_spelling_cod3s_actually_writes():
    """cod3s has two spellings of one observation and writes the second: a
    `PycAttrIndicator` carrying `attr_type: "ST"`, whose state is in
    `attr_name`. Both resolve to `{component}.{state}` on the reference
    engine, so reading only `PycSTIndicator` made a state indicator a
    variable indicator on a variable no component holds."""
    spec = rbd_declaration()
    spec["components"]["DEGRADED"] = an_event()
    spec["indicators"] = [
        {
            "name": "DEGRADED_occ",
            "kind": "PycAttrIndicator",
            "component": "DEGRADED",
            "attr_name": "occ",
            "attr_type": "ST",
            "stats": ["mean"],
        }
    ]
    body = pyraichu.model_body(json.loads(engine.build_model(spec).json))
    declared = [
        entry for entry in body["indicators"] if entry["name"] == "DEGRADED_occ"
    ]
    assert declared == [
        {
            "name": "DEGRADED_occ",
            "target": "state",
            "component": "DEGRADED",
            "automaton": "ev",
            "state": "occ",
        }
    ]


def test_the_two_spellings_of_a_state_indicator_give_one_observation():
    """Neither is a variant of the other: the one the document happens to
    carry cannot change what is observed."""
    def document(indicator):
        spec = rbd_declaration()
        spec["components"]["DEGRADED"] = an_event()
        spec["indicators"] = [dict(indicator, name="watched", component="DEGRADED")]
        return pyraichu.model_body(json.loads(engine.build_model(spec).json))[
            "indicators"
        ]

    assert document({"kind": "PycSTIndicator", "state": "occ"}) == document(
        {"kind": "PycAttrIndicator", "attr_name": "occ", "attr_type": "ST"}
    )


def test_a_state_the_event_has_not_is_refused_under_the_attribute_spelling_too():
    spec = rbd_declaration()
    spec["components"]["DEGRADED"] = an_event()
    spec["indicators"] = [
        {
            "name": "DEGRADED_rep",
            "kind": "PycAttrIndicator",
            "component": "DEGRADED",
            "attr_name": "rep",
            "attr_type": "ST",
        }
    ]
    with pytest.raises(declare.SystemSpecError, match=r"which holds \['not_occ', 'occ'\]"):
        engine.build_model(spec)


def test_a_third_attribute_type_is_refused_naming_the_two_that_are_carried():
    spec = rbd_declaration()
    spec["indicators"][0].update(
        kind="PycAttrIndicator", attr_name=f"{FLOW}_fed_in", attr_type="AUT"
    )
    with pytest.raises(declare.SystemSpecError) as raised:
        engine.build_model(spec)

    message = str(raised.value)
    assert "'AUT'" in message and "'VAR'" in message and "'ST'" in message


# --- the ok/nok pair muscadet derives on a discrete output -------------


def with_derived_out_automata(spec, component="B1"):
    """`create_default_out_automata`, as muscadet writes it: only when it
    asks for the pair, which is why refusing the key refused every
    platform export."""
    spec["components"][component][declare.DERIVED_OUT_AUTOMATA_KEY] = True
    return spec


def test_a_document_asking_for_the_derived_pair_runs():
    """The key is behaviour muscadet adds and this layer does not, but what
    it adds is a pair of states at a rate of 1e-100: nothing a run of this
    model can tell apart, and nothing the document names."""
    result = engine.simulate(
        with_derived_out_automata(rbd_declaration()),
        {"nb_runs": 1, "schedule": [0.0, 5.0]},
    )
    assert list(result.indicators[f"T_{FLOW}_fed_in"].mean) == [1.0, 0.0]


def test_the_derived_pair_changes_nothing_the_document_does_not_name():
    """Stronger than "it builds": the model is the one the same declaration
    without the key gives, so accepting the key costs the run nothing."""
    asked = engine.build_model(with_derived_out_automata(rbd_declaration())).json
    assert asked == engine.build_model(rbd_declaration()).json


def test_an_indicator_naming_a_derived_state_is_refused_by_name():
    """Where the refusal moved to: the pair is what an indicator could have
    observed, so an indicator naming one of its states is refused rather
    than silently absent."""
    spec = with_derived_out_automata(rbd_declaration())
    spec["indicators"] = [
        {
            "name": "B1_out_nok",
            "kind": "PycAttrIndicator",
            "component": "B1",
            "attr_name": f"{FLOW}_nok",
            "attr_type": "ST",
        }
    ]
    with pytest.raises(declare.SystemSpecError) as raised:
        engine.build_model(spec)

    message = str(raised.value)
    assert declare.DERIVED_OUT_AUTOMATA_KEY in message
    assert f"{FLOW}_nok" in message


def test_a_state_no_derived_pair_holds_keeps_the_refusal_it_had():
    """The refusal stays indexed on what the document declares: a state the
    pair does not hold is still "this component is no EVENT"."""
    spec = with_derived_out_automata(rbd_declaration())
    spec["indicators"] = [
        {
            "name": "B1_down",
            "kind": "PycSTIndicator",
            "component": "B1",
            "state": "failure_occ",
        }
    ]
    with pytest.raises(declare.SystemSpecError, match="does not declare as an EVENT"):
        engine.build_model(spec)


def test_two_indicators_of_one_name_are_refused():
    spec = rbd_declaration()
    spec["indicators"][0]["var"] = f"{FLOW}_fed_out"
    spec["indicators"][0]["component"] = "B1"
    spec["indicators"][0]["name"] = f"B1_{FLOW}_fed_in"
    with pytest.raises(declare.SystemSpecError, match="already observes"):
        engine.build_model(spec)


# --- 3. the failure mode's grip on a discrete output -------------------


@pytest.mark.parametrize(
    "pattern",
    [f"{FLOW}_fed_available_out", FLOW, ".*"],
    ids=["availability-variable", "flow-name", "wildcard"],
)
def test_every_spelling_of_a_gate_reaches_the_output(pattern):
    """muscadet says "this mode kills this output" three ways; RAICHU says it
    with `targets`, and the translation is what makes the three build."""
    spec = rbd_declaration()
    spec["components"]["B1"]["failure_modes"][0]["failure_effects"] = [
        [pattern, False]
    ]
    system = declare.build_system(spec)
    assert [mode.targets for mode in system.comp["B1"].failure_modes] == [[FLOW]]


def test_a_repair_effect_restoring_availability_is_absorbed():
    """The gate returns on its own: leaving the failing state restores the
    output on both sides, so the declaration is honoured by carrying nothing.
    """
    spec = rbd_declaration()
    spec["components"]["B1"]["failure_modes"][0]["repair_effects"] = [
        [f"{FLOW}_fed_available_out", True]
    ]
    system = declare.build_system(spec)
    mode = system.comp["B1"].failure_modes[0]
    assert mode.targets == [FLOW]
    assert mode.repair.caps == {} and mode.repair.taps == {}


def test_a_mode_gating_nothing_is_refused_rather_than_gating_everything():
    """An empty target list means EVERY output here, so the muscadet mode that
    gates none of them has no spelling and is refused instead of built wrong.

    Refused at the SYSTEM scale, where the document is known to be muscadet's.
    A component declaration on its own may as well have been written for this
    layer, where the same absence has always meant "every output", and that
    meaning is left alone.
    """
    spec = rbd_declaration()
    spec["components"]["B1"]["failure_modes"][0]["failure_effects"] = []
    with pytest.raises(declare.SystemSpecError, match="gates none"):
        declare.build_system(spec)

    component = spec["components"]["B1"]
    built = declare.build_component(pyraichu.muscadet.System(name="alone"), component)
    assert built.failure_modes[0].targets == []


def test_a_failure_that_makes_an_output_available_is_refused():
    spec = rbd_declaration()
    spec["components"]["B1"]["failure_modes"][0]["failure_effects"] = [[FLOW, True]]
    with pytest.raises(declare.ComponentSpecError, match="can only be False"):
        declare.build_system(spec)


# --- 4. the document, and the run parameters beside it -----------------


def test_a_declaration_from_an_unreadable_major_is_refused_with_its_number():
    spec = rbd_declaration()
    spec["version"] = "2.0.0"
    with pytest.raises(declare.SystemSpecError, match="'2.0.0'"):
        declare.build_system(spec)


def test_a_connection_naming_an_undeclared_component_is_refused():
    spec = rbd_declaration()
    spec["connections"][0]["target"] = "GHOST"
    with pytest.raises(declare.SystemSpecError, match="GHOST"):
        declare.build_system(spec)


def test_a_schedule_expands_its_ranges_and_sorts_what_it_names():
    assert engine._instants([10.0, {"start": 0.0, "end": 4.0, "nvalues": 3}]) == [
        0.0,
        2.0,
        4.0,
        10.0,
    ]
    assert engine._instants([{"start": 0.0, "end": 4.0, "nvalues": 1}]) == [4.0]


def test_the_horizon_is_the_last_instant_of_the_schedule():
    result = engine.simulate(
        rbd_declaration(), {"nb_runs": 1, "schedule": [0.0, 3.0], "seed": 1}
    )
    assert list(result.indicators[f"T_{FLOW}_fed_in"].instants) == [0.0, 3.0]


def test_a_run_declaring_no_schedule_is_refused():
    with pytest.raises(declare.SystemSpecError, match="'schedule'"):
        engine.simulate(rbd_declaration(), {"nb_runs": 1})


def test_a_trace_this_engine_does_not_produce_is_refused_by_name():
    with pytest.raises(declare.SystemSpecError, match="trace_level"):
        engine.simulate(
            rbd_declaration(),
            {"nb_runs": 1, "schedule": [0.0, 1.0], "trace_level": 2},
        )


def test_the_parameters_a_run_declares_and_does_not_set_say_nothing():
    """A cod3s parameter object writes every field; the run is not refused for
    the ones it left alone."""
    result = engine.simulate(
        rbd_declaration(),
        {
            "nb_runs": 1,
            "schedule": [0.0, 5.0],
            "seed": None,
            "time_unit": None,
            "rng": "yarn5",
            "rng_bloc_size": 1000,
            "trace_level": 0,
            "trace_elements": [],
        },
        postpone_post_proc=True,
    )
    assert list(result.indicators[f"T_{FLOW}_fed_in"].mean) == [1.0, 0.0]


#: The `simulation` section of a cod3s study, field for field, as the
#: platform's own translator writes it for the reference corpus. Written out
#: rather than derived: it is what actually reaches this seam, and a test
#: deriving it from cod3s would need cod3s installed to say anything.
PLATFORM_SIMULATION_PARAMETERS = {
    "nb_runs": 10,
    "schedule": [250.0, 1200.0, 1750.0, 2750.0, 3250.0],
    "seed": 42,
    "time_unit": "h",
    "rng": "yarn5",
    "rng_bloc_size": 1000,
    "trace_level": 0,
    "monitor_patterns": ["#.*"],
    "filter_objfm_in_sequences": True,
    "filter_objevent_in_sequences": True,
    "strict_failure_modes": False,
    "pdmp_dt": 0.02,
}


def test_the_parameters_a_platform_study_writes_do_not_stop_the_run():
    """A cod3s parameter object writes EVERY field, set or not, so a refusal
    on the presence of a key is a refusal of every study the platform writes.
    The whole `simulation` section of the reference corpus crosses the seam,
    and the run answers."""
    result = engine.simulate(rbd_declaration(), dict(PLATFORM_SIMULATION_PARAMETERS))
    assert result.nb_runs == PLATFORM_SIMULATION_PARAMETERS["nb_runs"]
    assert len(list(result.indicators[f"T_{FLOW}_fed_in"].mean)) == len(
        PLATFORM_SIMULATION_PARAMETERS["schedule"]
    )


@pytest.mark.parametrize(
    "key",
    [
        "monitor_patterns",
        "filter_objfm_in_sequences",
        "filter_objevent_in_sequences",
        "strict_failure_modes",
        "pdmp_dt",
    ],
)
def test_each_of_the_five_is_accepted_where_it_says_something(key):
    """Accepted at a value that ASKS for something, not only at the default
    that says nothing: four of the five arrive at their default in the corpus,
    and a study that changed one would be refused by a check on the value."""
    asking = {
        "monitor_patterns": ["#.*\\.occ.*"],
        "filter_objfm_in_sequences": False,
        "filter_objevent_in_sequences": False,
        "strict_failure_modes": True,
        "pdmp_dt": 0.001,
    }[key]
    result = engine.simulate(
        rbd_declaration(), {"nb_runs": 1, "schedule": [0.0, 5.0], key: asking}
    )
    assert list(result.indicators[f"T_{FLOW}_fed_in"].mean) == [1.0, 0.0]


def test_the_integration_step_is_a_divergence_and_says_so_where_a_reader_looks():
    """`pdmp_dt` is not of the same kind as the four beside it: it asks for
    the reference solver's integration step, which RAICHU has no counterpart
    for, so on a CONTINUOUS model it would change a result rather than a draw.
    It is accepted here -- refusing it would refuse every continuous study the
    platform writes -- and named apart from the silent ones, because the place
    that gap is stated is muscadet's conformance registry."""
    assert "pdmp_dt" not in engine._UNREAD_PARAMETERS
    assert "pdmp_dt" in engine._DIVERGENT_PARAMETERS


def test_a_parameter_this_engine_does_not_read_is_refused_rather_than_dropped():
    with pytest.raises(declare.SystemSpecError, match="invented"):
        engine.simulate(
            rbd_declaration(),
            {"nb_runs": 1, "schedule": [0.0, 1.0], "invented": 3},
        )


def test_an_engine_knob_travels_beside_the_parameters():
    """`threads` is RAICHU's and has no muscadet counterpart, so it reaches the
    engine as a keyword of the run rather than inside the parameters."""
    result = engine.simulate(
        rbd_declaration(),
        {"nb_runs": 4, "schedule": [0.0, 5.0], "seed": 3},
        threads=1,
    )
    assert result.nb_runs == 4


def test_an_interactive_session_takes_the_same_declaration():
    """Step by step and Monte Carlo build the same model: a model that
    behaved differently one step at a time is a divergence nobody reports."""
    session = engine.isimu_start(
        rbd_declaration(), {"nb_runs": 1, "schedule": [0.0, 24.0]}
    )
    assert session.time == 0.0
    fired = session.step()
    assert fired is not None
    assert session.time == FAILURE_DATES[0]


# --- 5. the sequence targets a run declares beside the document --------
#
# muscadet spells one run keyword itself, `muscadet.engine.RUN_TARGETS`: the
# feared events a run stops at. It names the EVENT and stops there, because
# what a trajectory stops at -- an automaton and a state -- is spelled
# differently by the two engines; translating it is this seam's half of the
# vocabulary. The model below is the running example of that door, and it is
# the same one muscadet's `tests/test_engine_run_targets_001.py` measures.

#: The feared event: the flow no longer reaches the far end.
FEARED_EVENT = "EVT_LOSS"

FAILURE_RATE = 0.1
REPAIR_RATE = 0.5

#: What a FREE-CYCLING campaign of this model plateaus at, and therefore the
#: ceiling it cannot pass at any instant: the block's stationary
#: unavailability, 0.1 / 0.6. It is the discriminator of this whole section,
#: and it is a closed form rather than a tuned threshold -- a curve under it is
#: a run whose target reached nobody, whatever the seed.
FREE_CYCLING_CEILING = FAILURE_RATE / (FAILURE_RATE + REPAIR_RATE)

CAMPAIGN_RUNS = 2000
CAMPAIGN_SCHEDULE = [0.0, 5.0, 10.0, 25.0, 50.0]
CAMPAIGN_SEED = 4242


def campaign_declaration(occ_state_name=None):
    """The running example, as `muscadet.declare.system_spec` writes it.

    A source, one repairable block, a target that needs it, and an event on
    the loss of the flow at the far end -- watched by a STATE indicator, an
    event holding no variable to watch. Written out field for field, like
    every other document in this module: a test that built it from muscadet
    would need muscadet installed to say anything at all. Dumped from the real
    thing on 2026-09-15 and transcribed, so what is exercised here is what
    actually crosses.
    """
    event = {
        "name": FEARED_EVENT,
        "kind": "two_state_mode",
        "cls": "ObjEvent",
        "cond": [[{"attr": f"{FLOW}_fed_in", "obj": "Tgt", "value": False}]],
    }
    if occ_state_name is not None:
        event["occ_state_name"] = occ_state_name
    return {
        "version": "1.0.1",
        "name": "RunTargetSys",
        "components": {
            "Src": _component("Src", "Source", [_flow_out(prod_default=True)]),
            "Blk": _component(
                "Blk",
                "Block",
                [_flow_in(), _flow_out(prod_cond=[[{"name": FLOW, "port": "in"}]])],
                [
                    {
                        "cls": "exp",
                        "name": "failure",
                        "failure_state": "occ",
                        "failure_cond": True,
                        "failure_rate": FAILURE_RATE,
                        "failure_effects": [[FLOW, False]],
                        "failure_param_name": "lambda",
                        "repair_state": "rep",
                        "repair_cond": True,
                        "repair_rate": REPAIR_RATE,
                        "repair_effects": [],
                        "repair_param_name": "mu",
                    }
                ],
            ),
            "Tgt": _component("Tgt", "Target", [_flow_in(logic="and")]),
            FEARED_EVENT: event,
        },
        "connections": [
            {
                "source": source,
                "source_box": f"{FLOW}_out",
                "target": target,
                "target_box": f"{FLOW}_in",
                "flow": FLOW,
            }
            for source, target in (("Src", "Blk"), ("Blk", "Tgt"))
        ],
        "indicators": [
            {
                "name": f"{FEARED_EVENT}_occ",
                "label": f"{FEARED_EVENT}_occ",
                "measure": "value",
                "stats": ["mean"],
                "component": FEARED_EVENT,
                "operator": "==",
                "value_test": True,
                "state": occ_state_name or "occ",
                "kind": "PycSTIndicator",
            }
        ],
    }


def campaign_targets(spec, **kwargs):
    """The occurrence indicator's mean at each instant, for one campaign."""
    result = engine.simulate(
        spec,
        {
            "nb_runs": CAMPAIGN_RUNS,
            "schedule": CAMPAIGN_SCHEDULE,
            "seed": CAMPAIGN_SEED,
        },
        **kwargs,
    )
    return list(result.indicators[f"{FEARED_EVENT}_occ"].mean)


def model_targets(model):
    return pyraichu.model_body(json.loads(model.json)).get("targets")


# --- the vocabulary, before any document is involved -------------------


def test_the_keyword_is_the_one_muscadet_spells():
    """Restated rather than imported from muscadet, which this package must
    not import. The two cannot drift in silence: a keyword arriving under
    another name falls through to `monte_carlo` and is refused there, by
    name."""
    assert engine.RUN_TARGETS == "targets"


def test_a_run_that_declares_nothing_declares_no_target():
    assert engine._target_names(None) == []
    assert engine._target_names([]) == []
    assert engine._target_names(()) == []


def test_a_bare_name_is_refused_rather_than_read_letter_by_letter():
    """A string is iterable, and that is the whole danger: taken as a
    sequence, `EVT_LOSS` is eight targets named `E`, `V`, `T`..., so the one
    mistake made would come back as eight sentences that never mention it."""
    with pytest.raises(declare.SystemSpecError, match="one string"):
        engine._target_names(FEARED_EVENT)


def test_what_is_not_a_list_of_names_is_refused_by_its_type():
    with pytest.raises(declare.SystemSpecError, match="dict"):
        engine._target_names({FEARED_EVENT: True})
    with pytest.raises(declare.SystemSpecError, match="name of its event"):
        engine._target_names([{"name": FEARED_EVENT}])
    with pytest.raises(declare.SystemSpecError, match="name of its event"):
        engine._target_names([""])


def test_a_name_declared_twice_is_one_target_and_the_order_is_kept():
    assert engine._target_names(["B", "A", "B"]) == ["B", "A"]


# --- the model's `targets` section -------------------------------------


def test_build_model_takes_the_same_vocabulary_and_writes_the_section():
    """The criterion: one entry per name, resolved through the event's own
    declaration."""
    model = engine.build_model(campaign_declaration(), [FEARED_EVENT])
    assert model_targets(model) == [
        {
            "name": FEARED_EVENT,
            "component": FEARED_EVENT,
            "automaton": "ev",
            "state": "occ",
        }
    ]


def test_a_target_is_resolved_through_the_names_the_event_renamed():
    """Read through `pyraichu.declare.event_automaton` and the event's own
    `occ_state_name`, not through the defaults: a modeller who renamed the
    occurrence state would otherwise get a target on a state no automaton
    holds, and a campaign that stops at nothing."""
    spec = campaign_declaration(occ_state_name="reached")
    spec["components"][FEARED_EVENT]["event_aut_name"] = "alarm"
    assert model_targets(engine.build_model(spec, [FEARED_EVENT])) == [
        {
            "name": FEARED_EVENT,
            "component": FEARED_EVENT,
            "automaton": "alarm",
            "state": "reached",
        }
    ]


def test_a_run_declaring_no_target_builds_the_model_it_always_built():
    """The keyword is absent rather than empty when nobody asks, and the
    document is untouched either way: one system, one declaration, two
    campaigns."""
    plain = engine.build_model(campaign_declaration())
    assert model_targets(plain) == []
    with_none = engine.build_model(campaign_declaration(), None)
    assert with_none.json == plain.json


def test_a_target_naming_no_event_is_refused_with_the_events_there_are():
    """`build_model` is a public door of its own -- the platform reaches for
    the MODEL, to hand it to a second engine call the seam has no kind for --
    so a caller who crossed none of muscadet's checks is answered here."""
    with pytest.raises(declare.SystemSpecError) as refused:
        engine.build_model(campaign_declaration(), ["PANNE_OND"])

    message = str(refused.value)
    assert "PANNE_OND" in message and "declares no" in message
    assert FEARED_EVENT in message


def test_a_target_naming_a_component_that_is_no_event_is_refused():
    """A target is reached at an event's occurrence, and a block has none."""
    with pytest.raises(declare.SystemSpecError, match="Blk"):
        engine.build_model(campaign_declaration(), ["Blk"])


def test_the_state_a_target_ends_at_is_read_where_the_event_names_it():
    """One reading, not two. `event_automaton` says which two states an event
    holds, `event_occurrence_state` says which of them it has OCCURRED in, and
    a target needs the second: a trajectory ends at the occurrence, not at
    either of the two. Recomputing it here from `occ_state_name` would be a
    second copy of the convention, and the kind that agrees until the day it
    does not."""
    spec = campaign_declaration(occ_state_name="reached")
    event = spec["components"][FEARED_EVENT]
    assert declare.event_occurrence_state(event) == "reached"
    assert declare.event_occurrence_state(spec["components"]["Blk"]) is None
    assert model_targets(engine.build_model(spec, [FEARED_EVENT]))[0]["state"] == (
        declare.event_occurrence_state(event)
    )


# --- the run: the keyword, and what it decides -------------------------


def test_simulate_accepts_the_keyword_muscadet_hands_it():
    """The criterion: muscadet passes `targets=("EVT_LOSS",)` beside the
    document, and the run neither refuses it nor lets it fall through to
    `monte_carlo`, which has no such argument."""
    result = engine.simulate(
        campaign_declaration(),
        {"nb_runs": 1, "schedule": [0.0, 5.0], "seed": 1},
        targets=(FEARED_EVENT,),
    )
    assert len(list(result.indicators[f"{FEARED_EVENT}_occ"].mean)) == 2


def test_the_targets_reach_the_model_the_run_is_given(monkeypatch):
    """What the run hands the engine, read where it is handed over."""
    seen = {}

    def spy(model, **kwargs):
        seen["targets"] = model_targets(model)
        seen["kwargs"] = kwargs
        raise SystemExit

    monkeypatch.setattr(engine, "monte_carlo", spy)
    with pytest.raises(SystemExit):
        engine.simulate(
            campaign_declaration(),
            {"nb_runs": 1, "schedule": [0.0, 5.0]},
            targets=[FEARED_EVENT],
        )
    assert seen["targets"] == [
        {
            "name": FEARED_EVENT,
            "component": FEARED_EVENT,
            "automaton": "ev",
            "state": "occ",
        }
    ]
    assert engine.RUN_TARGETS not in seen["kwargs"], "not passed on to the engine"


@pytest.mark.parametrize(
    ("targets", "expected"),
    [(None, False), ([], False), ([FEARED_EVENT], True)],
    ids=["silent", "empty", "declared"],
)
def test_stop_at_targets_is_derived_from_the_presence_of_targets(
    monkeypatch, targets, expected
):
    """The decision, asserted: a target that does not stop the trajectory is
    not a target. It is what the reference engine does without being asked --
    PyCATSHOO stops unconditionally on an `addTarget` -- and what the caller
    who declared a feared event means."""
    seen = {}
    monkeypatch.setattr(
        engine,
        "monte_carlo",
        lambda model, **kwargs: seen.update(kwargs) or SystemExit,
    )
    keywords = {} if targets is None else {"targets": targets}
    engine.simulate(
        campaign_declaration(), {"nb_runs": 1, "schedule": [0.0, 5.0]}, **keywords
    )
    assert seen["stop_at_targets"] is expected


def test_an_explicit_stop_at_targets_still_wins():
    """The two RAICHU knobs stay separable underneath -- the model carries the
    targets, the run decides whether it latches -- so a study that really
    wants a free-cycling campaign over a model carrying targets says so and
    gets it."""
    means = campaign_targets(
        campaign_declaration(), targets=[FEARED_EVENT], stop_at_targets=False
    )
    assert max(means) < FREE_CYCLING_CEILING + 0.05, (
        f"asking for no early stop gave {means}, which passes the free-cycling "
        f"ceiling {FREE_CYCLING_CEILING:.3f}: the explicit knob was overridden"
    )


def test_a_target_inside_the_run_parameters_is_still_refused():
    """One door, not two. A target is a keyword of the RUN beside the
    parameters, and the refusal `_run_parameters` already carried says exactly
    that -- opening a second spelling for one thing is how two vocabularies
    for one notion start."""
    with pytest.raises(declare.SystemSpecError) as refused:
        engine.simulate(
            campaign_declaration(),
            {
                "nb_runs": 1,
                "schedule": [0.0, 5.0],
                "targets": [FEARED_EVENT],
            },
        )
    assert "beside the parameters" in str(refused.value)


# --- the campaign: the latch, and the sequences it makes readable ------


def test_a_campaign_with_a_target_latches_at_the_first_occurrence():
    """The criterion, and the discriminator is a closed form rather than a
    tuned threshold.

    The block fails at 0.1 and repairs at 0.5, so a campaign that does NOT
    stop at the event plateaus at its stationary unavailability, 0.167, and
    cannot approach 1 at any instant. A curve climbing past that ceiling is a
    run whose trajectories stopped at the feared event and whose indicator
    latched from there to the horizon; a curve under it is a run whose target
    reached nobody.
    """
    means = campaign_targets(campaign_declaration(), targets=[FEARED_EVENT])
    assert means[0] == 0.0, "nothing has occurred at t = 0"
    assert means == sorted(means), f"a latched indicator never comes back down: {means}"
    assert means[-1] > 0.95, (
        f"the campaign ended at {means[-1]:.3f}: over 50 units at a failure "
        f"rate of 0.1 almost every trajectory meets the event"
    )


def test_the_same_campaign_without_a_target_free_cycles_under_the_ceiling():
    """The other half of the discriminator, on the same document: what the
    keyword is worth is the difference between these two runs."""
    means = campaign_targets(campaign_declaration())
    assert max(means) < FREE_CYCLING_CEILING + 0.03, (
        f"a free-cycling campaign cannot pass {FREE_CYCLING_CEILING:.3f} at "
        f"any instant, and this one reached {max(means):.3f}"
    )


def test_the_sequences_of_the_same_model_are_not_empty():
    """The second engine call the seam has no kind for, and the reason
    `build_model` takes the vocabulary: the platform builds the model once and
    hands it to the campaign and to the sequence analysis. Without a `targets`
    section this returns the trajectories' worth of nothing -- no end cause,
    no events -- on a run that exits cleanly."""
    model = engine.build_model(campaign_declaration(), [FEARED_EVENT])
    cuts = pyraichu.analyse_sequences(model, nb_runs=200, t_max=50.0, seed=CAMPAIGN_SEED)
    assert cuts, "a model declaring a target yields sequences"
    reached = [cut for cut in cuts if cut["end_cause"] == FEARED_EVENT]
    assert reached, f"no sequence ends at the feared event: {cuts}"
    assert all(cut["events"] for cut in reached), "a sequence names what led there"
    # The rest are the trajectories that never met the event before the
    # horizon, which is a physical answer and not a missing target.
    assert sum(cut["weight"] for cut in reached) > 0.9 * 200


def test_the_same_model_without_a_target_ends_no_trajectory_anywhere():
    """What the workaround this replaces was written against: the campaign
    free-cycles, the sequence file comes back with nothing in it, and the run
    exits 0."""
    model = engine.build_model(campaign_declaration())
    cuts = pyraichu.analyse_sequences(model, nb_runs=50, t_max=50.0, seed=CAMPAIGN_SEED)
    assert all(cut["end_cause"] is None for cut in cuts)


# --- the interactive entry point ---------------------------------------


def test_an_interactive_session_does_not_die_on_the_keyword():
    """muscadet passes `targets` to BOTH kinds of run, deliberately: a keyword
    one entry point takes and the other dies on would make a demonstration and
    a campaign diverge on the study they are two views of."""
    session = engine.isimu_start(
        campaign_declaration(),
        {"nb_runs": 1, "schedule": [0.0, 50.0]},
        targets=[FEARED_EVENT],
    )
    assert session.time == 0.0
    assert session.step() is not None


def test_reaching_the_feared_event_does_not_end_an_interactive_session():
    """What RAICHU does with a target step by step, stated rather than left
    open: the model carries it, and nothing stops the stepping. A session has
    no `stop_at_targets` to set, and whoever drives one by hand is who decides
    what reaching a feared event means -- which is the point of driving it by
    hand."""
    session = engine.isimu_start(
        campaign_declaration(),
        {"nb_runs": 1, "schedule": [0.0, 50.0]},
        targets=[FEARED_EVENT],
    )
    automaton = f"{FEARED_EVENT}.ev"
    assert session.state(automaton) == "not_occ"
    while session.state(automaton) != "occ":
        assert session.step() is not None, "the event never occurred"
    reached = session.time
    assert session.step() is not None, "the session steps past its feared event"
    assert session.time > reached
