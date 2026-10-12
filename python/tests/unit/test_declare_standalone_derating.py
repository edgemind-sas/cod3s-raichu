"""A standalone failure mode derating a CONTINUOUS output (muscadet format 1.0.3).

A mode declared outside the component it hits (`ObjMode2S` with `targets`)
writes the variables it names. On a continuous output that variable is a
derating muscadet allocates through `ObjFlow.add_derating`, one per
(mode, flow), and the declaration carries the pairs in the component's
`deratings` section so a reader can allocate them again. This layer reads
that section: each pair becomes `{mode}_derating_{flow}`, at the nominal 1,
folded by MINIMUM into the output's effective rate beside the shared
`{flow}_out_rate` and the modes declared on the component.

A derated output also stops its rule DRAWING (muscadet R-13, "uptake"): a
fan whose exhaust a mode cuts to zero no longer empties the room it
ventilates, which is what lets the room fill.

The documents are written out field by field in the shape
`muscadet.engine.system_declaration` produces for them (muscadet 5.14.0),
reduced to the keys that carry a decision, so the module needs no muscadet.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.muscadet.declare as declare

# --- the documents ----------------------------------------------------


def continuous_out(name, **extra):
    return {"cls": "FlowContinuousOut", "name": name, "var_type": "float", **extra}


def continuous_in(name, **extra):
    return {"cls": "FlowContinuousIn", "name": name, "var_type": "float", **extra}


def component(name, flows, **sections):
    return {"name": name, "cls": "ObjFlow", "flows": flows, **sections}


def a_mode(mode, target, flow, date, value):
    """`system.add_component(cls="ObjMode2S", ...)`, never repaired."""
    variable = f"{mode}_derating_{flow}"
    return {
        "name": f"{target}__{mode}",
        "kind": "two_state_mode",
        "cls": "ObjMode2S",
        "mode_name": mode,
        "targets": [target],
        "target_name": target,
        "occ_law": {"cls": "delay", "time": [date]},
        "occ_effects": {variable: value},
        "occ_param_name": ["occ_time"],
        "occ_param": [date],
        "not_occ_law": {"cls": "delay", "time": [1e6]},
        "not_occ_effects": {variable: 1.0},
        "not_occ_param_name": ["not_occ_time"],
        "not_occ_param": [1e6],
    }


def link(source, target, flow):
    return {
        "source": source,
        "source_box": f"{flow}_out",
        "target": target,
        "target_box": f"{flow}_in",
        "flow": flow,
    }


def a_document(components, connections):
    return {
        "version": "1.0.3",
        "name": "standalone_derating",
        "components": {entry["name"]: entry for entry in components},
        "connections": connections,
    }


def a_derated_source(modes=(("cut", 2.0, 0.5), ("dip", 3.0, 0.25)), deratings=None):
    """A source of 10 feeding a load asking 8, and standalone modes
    derating the source's output, each allocated in `deratings`."""
    pairs = (
        deratings
        if deratings is not None
        else [{"mode": mode, "flow": "P"} for mode, _date, _value in modes]
    )
    source = component("SRC", [continuous_out("P", var_fed_default=10.0)], deratings=pairs)
    load = component("LOAD", [continuous_in("P", var_demand_default=8.0)])
    return a_document(
        [source, load, *(a_mode(mode, "SRC", "P", date, value) for mode, date, value in modes)],
        [link("SRC", "LOAD", "P")],
    )


def a_ventilated_room(seized_at=1.0):
    """Air at 4 and hydrogen at 1 into a room; a fan drawing both into an
    exhaust the outside takes at 10; a standalone mode seizing the exhaust."""
    room = component(
        "ROOM",
        [continuous_in("AIR"), continuous_out("AIR"), continuous_in("H2"), continuous_out("H2")],
        capacities=[
            {
                "name": "room",
                "flows": [{"name": "AIR", "weight": 1.0}, {"name": "H2", "weight": 1.0}],
                "side": "out",
                "capacity": 1e5,
                "content_init": {"AIR": 90.0, "H2": 0.0},
                "fill_rate": "inf",
                "transmits": True,
            }
        ],
    )
    fan = component(
        "FAN",
        [continuous_in("AIR"), continuous_in("H2"), continuous_out("exhaust")],
        rules=[
            {
                "name": "extraction",
                "rules": [{"cond": [], "cons": {"AIR": 1.0, "H2": 1.0}, "prod": {"exhaust": 2.0}}],
            }
        ],
        deratings=[{"mode": "seized", "flow": "exhaust"}],
    )
    return a_document(
        [
            component("S_AIR", [continuous_out("AIR", var_fed_default=4.0)]),
            component("S_H2", [continuous_out("H2", var_fed_default=1.0)]),
            room,
            fan,
            component("OUTSIDE", [continuous_in("exhaust", var_demand_default=10.0)]),
            a_mode("seized", "FAN", "exhaust", seized_at, 0.0),
        ],
        [
            link("S_AIR", "ROOM", "AIR"),
            link("S_H2", "ROOM", "H2"),
            link("ROOM", "FAN", "AIR"),
            link("ROOM", "FAN", "H2"),
            link("FAN", "OUTSIDE", "exhaust"),
        ],
    )


def observe(document, readings, instants, t_max):
    """`{name: [(t, value)]}` for each `(name, component, attribute)`."""
    body = declare.build_document(document)
    body["model"]["indicators"] = [
        {"name": name, "target": "attribute", "attr": {"component": comp, "attribute": attr}}
        for name, comp, attr in readings
    ]
    result = pyraichu.simulate(pyraichu.load_model(json.dumps(body)), t_max=t_max, samples=instants)
    return {name: dict(result.samples[name]) for name, _comp, _attr in readings}


# --- the derating -----------------------------------------------------


def test_the_section_allocates_one_variable_per_pair_at_nominal():
    body = declare.build_document(a_derated_source())
    source = next(entry for entry in body["model"]["components"] if entry["name"] == "SRC")
    attributes = {entry["name"]: entry for entry in source["attributes"]}
    for variable in ("cut_derating_P", "dip_derating_P"):
        assert variable in attributes, sorted(attributes)
    assert "P_effective_rate" in attributes, "a derated output folds its caps into one rate"


def test_a_standalone_mode_derates_the_output_from_its_firing_date():
    seen = observe(
        a_derated_source(modes=(("cut", 2.0, 0.5),)),
        [("got", "LOAD", "P_fed_in")],
        [1.0, 2.5],
        t_max=3.0,
    )
    assert seen["got"][1.0] == pytest.approx(8.0), "nominal: the load gets what it asks"
    assert seen["got"][2.5] == pytest.approx(5.0), "derated to 0.5: half of the source's 10"


def test_two_standalone_modes_on_one_output_fold_by_minimum():
    """The second mode is deeper, so the minimum changes hands at its date;
    a last-writer-wins on a shared variable would answer the same here only
    by accident of the order, which is why each mode owns its variable."""
    seen = observe(
        a_derated_source(),
        [("got", "LOAD", "P_fed_in"), ("rate", "SRC", "P_effective_rate")],
        [1.0, 2.5, 3.5],
        t_max=4.0,
    )
    assert [seen["rate"][t] for t in (1.0, 2.5, 3.5)] == pytest.approx([1.0, 0.5, 0.25])
    assert [seen["got"][t] for t in (1.0, 2.5, 3.5)] == pytest.approx([8.0, 5.0, 2.5])


def test_a_pair_allocated_twice_is_one_variable():
    """muscadet's read-back can echo a pair; replaying it must not refuse."""
    document = a_derated_source(
        modes=(("cut", 2.0, 0.5),),
        deratings=[{"mode": "cut", "flow": "P"}, {"mode": "cut", "flow": "P"}],
    )
    seen = observe(document, [("got", "LOAD", "P_fed_in")], [2.5], t_max=3.0)
    assert seen["got"][2.5] == pytest.approx(5.0)


def test_a_pair_naming_no_continuous_output_is_refused_by_the_outputs_there_are():
    document = a_derated_source(modes=(), deratings=[{"mode": "cut", "flow": "Q"}])
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    message = str(refused.value)
    assert "Q" in message and "P" in message, message


def test_a_pair_carries_its_two_keys_and_nothing_else():
    document = a_derated_source(
        modes=(), deratings=[{"mode": "cut", "flow": "P", "value": 0.5}]
    )
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    assert "value" in str(refused.value)


def test_a_standalone_mode_with_no_allocation_is_still_refused():
    """Without the section the variable does not exist, and the refusal that
    names what the component carries is unchanged."""
    document = a_derated_source(modes=(("cut", 2.0, 0.5),), deratings=[])
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    assert "cut_derating_P" in str(refused.value)


# --- the uptake -------------------------------------------------------


def test_a_seized_fan_stops_drawing_the_room_it_ventilates():
    """Before the seizure the fan takes the hydrogen as it arrives and the
    room holds none; after it, the exhaust is cut AND so is the draw, so the
    hydrogen accumulates at its supply rate. Drawing on at the nominal rate
    while producing nothing would keep the room empty and destroy what the
    fan took."""
    seen = observe(
        a_ventilated_room(seized_at=1.0),
        [("h2", "ROOM", "room_content_H2"), ("out", "OUTSIDE", "exhaust_fed_in")],
        [0.5, 2.0, 3.0],
        t_max=3.0,
    )
    assert seen["out"][0.5] == pytest.approx(2.0)
    assert seen["h2"][0.5] == pytest.approx(0.0, abs=1e-9)
    assert seen["out"][2.0] == pytest.approx(0.0, abs=1e-9)
    assert [seen["h2"][t] for t in (2.0, 3.0)] == pytest.approx([1.0, 2.0])
