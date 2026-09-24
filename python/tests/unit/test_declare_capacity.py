"""The capacity names the two layers do not share, read off a declaration.

An indicator names the VARIABLE muscadet created. On almost everything the two
layers spell alike; on a volume they part, and they part on the one reading a
continuous study is written to watch: muscadet holds what a capacity contains
in ``{c}_qty`` and ``{c}_qty_{f}``, this layer in ``{c}_content`` and
``{c}_content_{f}``.

Two readings answer that, both pure and both answering a DOCUMENT, so a caller
sorts one before building anything:

- :func:`pyraichu.declare.capacity_content_variables`, the translation;
- :func:`pyraichu.declare.capacity_absent_variables`, the three muscadet
  variables this layer has no attribute of, each with what stands in its place.

The inventory the second one rests on was measured on 2026-09-15 against a live
pair of engines, on a single-constituent volume and on a two-constituent one.
Of muscadet's capacity variables, ``{c}_fill`` and ``{c}_fill_{f}`` are shared
outright, ``{c}_ratio_{f}`` is shared on a volume holding several constituents
and absent on one holding a single, and ``_inflow_`` and ``_outflow_`` have no
counterpart at all.

``{c}_serve_rate_{f}`` was a fourth absence and is shared since the ceiling
became a variable a failure mode can clamp
(:meth:`pyraichu.muscadet._Capacity.ceiling_of`). It is pinned below as a name
that must NOT be refused, because an absence that outlives the code it
describes refuses a reading both engines answer, and says something false while
doing it.
"""

import pytest

import pyraichu.declare as declare


def a_tank(name="TANK", capacity="tank", flows=("q",)):
    """A volume over one or more continuous inputs, as a document holds it."""
    return {
        "name": name,
        "cls": "ObjFlow",
        "source_cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowContinuousIn",
                "name": flow,
                "var_type": "float",
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": 0.0,
                "var_demand_default": 0.0,
            }
            for flow in flows
        ],
        "capacities": [
            {
                "name": capacity,
                "flows": list(flows),
                "capacity": 100.0,
                "content_init": {flow: 0.0 for flow in flows},
            }
        ],
        "measurements_in": [],
        "rules": [],
        "transfers": [],
        "failure_modes": [],
    }


def a_flow_component(name="PIPE"):
    """A component holding no volume at all, whose flow is named so that the
    muscadet spelling of a capacity would match it if the reading went by
    suffix: `q_qty_q` is what a capacity named `q` over a flow `q` would
    publish, and this component declares no capacity."""
    entry = a_tank(name=name, capacity="q", flows=("q",))
    entry["capacities"] = []
    return entry


# --- the translation -----------------------------------------------------


def test_a_volume_answers_the_quantity_it_holds_under_both_spellings():
    """The total and the one constituent, muscadet's name then this one's."""
    assert declare.capacity_content_variables(a_tank()) == {
        "tank_qty": "tank_content",
        "tank_qty_q": "tank_content_q",
    }


def test_a_volume_holding_several_constituents_answers_each_of_them():
    """One entry per constituent beside the total, so an observer watching a
    mixture reaches the term it named and not only the sum."""
    assert declare.capacity_content_variables(
        a_tank(flows=("water", "heat"))
    ) == {
        "tank_qty": "tank_content",
        "tank_qty_water": "tank_content_water",
        "tank_qty_heat": "tank_content_heat",
    }


def test_the_fill_is_absent_from_that_mapping():
    """Both layers call it `{c}_fill` and `{c}_fill_{f}`, so translating it
    would invent a disagreement."""
    translated = declare.capacity_content_variables(a_tank())
    assert not [name for name in translated if "_fill" in name]


def test_the_short_form_of_one_held_flow_is_read_too():
    """`flow=` is muscadet's single-flow short form of `flows=`, and a document
    exported from a tank written that way carries it."""
    entry = a_tank()
    entry["capacities"][0] = {
        "name": "tank",
        "flow": "q",
        "capacity": 100.0,
    }
    assert declare.capacity_content_variables(entry) == {
        "tank_qty": "tank_content",
        "tank_qty_q": "tank_content_q",
    }


@pytest.mark.parametrize("entry", [None, {}, "TANK"], ids=["none", "empty", "string"])
def test_anything_that_is_not_a_declaration_answers_nothing(entry):
    """So a caller sweeps a document without sorting it first."""
    assert declare.capacity_content_variables(entry) == {}
    assert declare.capacity_absent_variables(entry) == {}


def test_a_component_holding_no_volume_answers_nothing():
    """The reading is keyed on the capacities the component DECLARES, never on
    a suffix: a variable ending in `_qty` on a component holding no volume of
    that name is left exactly as the document wrote it."""
    assert declare.capacity_content_variables(a_flow_component()) == {}
    assert declare.capacity_absent_variables(a_flow_component()) == {}


def test_a_controller_is_not_swept_for_a_volume():
    """A controller holds no flow and no volume, and its own disagreement is
    read elsewhere (`controller_signal_variables`)."""
    controller = {
        "name": "CTRL",
        "kind": "controller",
        "cls": "ObjCtrl",
        "controls_in": [{"name": "tank"}],
        "controls_out": [{"name": "high", "kind": "bool", "default": False}],
    }
    assert declare.capacity_content_variables(controller) == {}
    assert declare.capacity_absent_variables(controller) == {}


# --- the three with no counterpart ----------------------------------------


def test_a_single_constituent_volume_names_its_three_absences():
    """Each of the three muscadet creates and this layer has no attribute for,
    by its own name. A refusal that says only "unknown attribute" sends a
    modeller looking for a typo in a name that is spelled right."""
    assert sorted(declare.capacity_absent_variables(a_tank())) == [
        "tank_inflow_q",
        "tank_outflow_q",
        "tank_ratio_q",
    ]


def test_each_absence_says_what_stands_in_its_place():
    """What makes the refusal worth having: the two hooks onto the allocation
    sweeps point at the pair this layer integrates the content over, and the
    ratio of a volume holding one constituent points at what it holds."""
    absent = declare.capacity_absent_variables(a_tank())
    assert "q_fed_in" in absent["tank_inflow_q"]
    assert "q_fed_out" in absent["tank_outflow_q"]
    assert "tank_content" in absent["tank_ratio_q"]


def test_a_volume_holding_several_constituents_publishes_its_ratios():
    """The one conditional absence, and the reason the reading looks at the
    flows: this layer publishes a ratio per constituent only on a volume
    holding more than one, so on a mixture `{c}_ratio_{f}` is a SHARED name and
    refusing it would refuse an observation this layer answers."""
    absent = declare.capacity_absent_variables(a_tank(flows=("water", "heat")))
    assert "tank_ratio_water" not in absent
    assert "tank_ratio_heat" not in absent
    assert sorted(absent) == [
        "tank_inflow_heat",
        "tank_inflow_water",
        "tank_outflow_heat",
        "tank_outflow_water",
    ]


# --- the ceiling, which is NOT one of them --------------------------------


@pytest.mark.parametrize(
    "flows", [("q",), ("water", "heat")], ids=["single", "mixture"]
)
@pytest.mark.parametrize(
    "serve_rate", [None, 5.0], ids=["unbounded", "declared"]
)
def test_the_service_ceiling_is_never_refused(flows, serve_rate):
    """`{c}_serve_rate_{f}` is a shared name, whatever the declaration says
    about the ceiling and however many constituents the volume holds.

    It was refused while the ceiling was a constant folded into the service
    expression. It is a variable now, one per held flow and under muscadet's
    own spelling, because that is what a failure mode clamps to throttle a
    discharge; refusing an observation on it would refuse the very reading
    that mode makes worth watching. A volume that declares NO ceiling is here
    too: it publishes the unbounded sentinel rather than nothing, so the
    attribute exists either way and the answer may not depend on the number.
    """
    entry = a_tank(flows=flows)
    if serve_rate is not None:
        entry["capacities"][0]["serve_rate"] = serve_rate
    absent = declare.capacity_absent_variables(entry)
    assert not [name for name in absent if "_serve_rate_" in name], absent


# --- the two readings together --------------------------------------------


@pytest.mark.parametrize(
    "flows", [("q",), ("water", "heat")], ids=["single", "mixture"]
)
def test_what_is_translated_is_never_also_refused(flows):
    """The two readings answer disjoint sets of names, so an observation is
    either carried or refused and never both, whichever order a caller asks
    them in."""
    entry = a_tank(flows=flows)
    translated = set(declare.capacity_content_variables(entry))
    refused = set(declare.capacity_absent_variables(entry))
    assert translated & refused == set()


# --- the level a CONDITION watches ----------------------------------------
#
# The other consumer of the same two readings, and the one this file was
# missing. An INDICATOR names the level of a tank from outside and is resolved
# where the model is assembled; a CONDITION names it from inside a two-state
# component, and was resolved nowhere -- so a mode armed on a low tank level,
# which is the most ordinary use a safety study makes of a volume, reached the
# check that refuses a reference nothing holds and was turned away.


def a_watching_event(attr, name="WATCH", obj="TANK"):
    """An `ObjEvent` whose one condition watches `obj.attr`."""
    return {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjEvent",
        "cond": [{"obj": obj, "attr": attr, "value": 1.0, "comp": "gt"}],
    }


def a_watching_mode(occ, not_occ, name="TANK__MODE", target="TANK"):
    """An `ObjMode2S` whose two directions each watch a variable of `target`."""
    return {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjMode2S",
        "mode_name": name.split("__", 1)[-1],
        "targets": [target],
        "target_name": target,
        "occ_law": {"cls": "delay", "time": 4},
        "occ_param_name": ["occ_time"],
        "occ_param": [4],
        "not_occ_law": {"cls": "delay", "time": 2},
        "not_occ_param_name": ["not_occ_time"],
        "not_occ_param": [2],
        "occ_cond": [{"obj": target, "attr": occ, "value": 1.0, "comp": "lt"}],
        "not_occ_cond": [{"obj": target, "attr": not_occ, "value": 5.0, "comp": "gt"}],
    }


def a_document(*components):
    return {
        "version": "1.0.0",
        "name": "volume",
        "components": {entry["name"]: entry for entry in components},
        "connections": [],
    }


def watched(obj):
    """Every `(object, attribute)` the conditions of one plugin object name."""
    return sorted(
        (leaf["obj"], leaf["attr"])
        for key in ("cond", "failure_cond", "repair_cond")
        for group in obj.get(key) or []
        for leaf in group
        if "attr" in leaf
    )


@pytest.mark.parametrize(
    "spelling, carried",
    [("tank_qty", "tank_content"), ("tank_qty_q", "tank_content_q")],
    ids=["total", "constituent"],
)
def test_an_event_condition_reads_the_level_under_muscadets_spelling(
    spelling, carried
):
    """A feared event armed when a vessel passes a threshold, written the way
    muscadet writes it, and resolved to the attribute this layer carries."""
    event = a_watching_event(spelling)
    obj = declare.mode_object(event, {"TANK": a_tank(), "WATCH": event})
    assert watched(obj) == [("TANK", carried)]


def test_both_directions_of_a_standalone_mode_read_it_too():
    """`occ_cond` and `not_occ_cond` alike: a mode armed on a low level and
    released on a high one names the same variable twice, and neither
    direction is more of a reading than the other."""
    mode = a_watching_mode("tank_qty_q", "tank_qty")
    obj = declare.mode_object(mode, {"TANK": a_tank(), mode["name"]: mode})
    assert watched(obj) == [("TANK", "tank_content"), ("TANK", "tank_content_q")]


@pytest.mark.parametrize(
    "component",
    [a_watching_event("tank_qty_q"), a_watching_mode("tank_qty_q", "tank_qty")],
    ids=["event", "mode"],
)
def test_the_document_carrying_such_a_condition_builds(component):
    """End to end, which is what the modeller sees: the same document was
    refused for naming a variable `TANK` did not carry."""
    declare.build_document(a_document(a_tank(), component))


def test_the_translation_is_the_one_the_indicators_go_through():
    """Never a second table: what a condition reads is exactly what
    :func:`capacity_content_variables` answers for that component, so the day
    the layer names the attribute otherwise both consumers follow at once."""
    event = a_watching_event("tank_qty_q")
    obj = declare.mode_object(event, {"TANK": a_tank(), "WATCH": event})
    assert watched(obj) == [
        ("TANK", declare.capacity_content_variables(a_tank())["tank_qty_q"])
    ]


def test_a_component_holding_no_volume_keeps_the_spelling_it_was_given():
    """Keyed on the capacities the component DECLARES, as the indicator
    reading is: `q_qty_q` on a component that holds no volume is left exactly
    as the document wrote it, and refused as the unknown attribute it is."""
    event = a_watching_event("q_qty_q", obj="PIPE")
    document = a_document(a_flow_component(), event)
    with pytest.raises(declare.ComponentSpecError, match="`PIPE.q_qty_q`"):
        declare.build_document(document)


def test_a_condition_on_an_absent_capacity_variable_is_refused_by_its_name():
    """The other half of the reading reaches a condition as well: one of the
    three muscadet creates and this layer has no attribute for is refused
    saying WHAT TO WATCH instead, rather than met with "does not carry"."""
    event = a_watching_event("tank_inflow_q")
    document = a_document(a_tank(), event)
    with pytest.raises(declare.ComponentSpecError, match="q_fed_in"):
        declare.build_document(document)
