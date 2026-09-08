"""RAICHU as an engine of muscadet, proved without muscadet being installed.

muscadet exposes an extension point and imports no engine; `pyraichu`
registers there and becomes the engine a muscadet run may select. This module
pins the RAICHU side of that seam, and it pins it **oracle-free**: everything
here is either the distribution's own metadata, the declaration reader, or a
trajectory RAICHU computes on its own. What needs PyCATSHOO alive -- the same
model answering alike on both engines -- is the cross-validation suite's, under
`python/tests/validation/test_muscadet_engine_parity.py`.

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


def test_a_state_indicator_is_refused_by_name():
    """The two layers name automata and their states differently."""
    spec = rbd_declaration()
    spec["indicators"] = [
        {
            "name": "B1_down",
            "kind": "PycSTIndicator",
            "component": "B1",
            "state": "failure_occ",
        }
    ]
    with pytest.raises(declare.SystemSpecError, match="PycSTIndicator"):
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
