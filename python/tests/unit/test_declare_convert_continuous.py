"""The muscadet declaration converter on continuous flows.

A continuous network converts into the std fragments each construct's
plugin-versus-library bench proves (`test_library_std_continuous.py`):
sources and sinks with their profiles and published rates, split policies,
pass-throughs, single-flow output-side volumes, and rule sets with one
guarded rule and one idle rule. Every other construct is refused by name.
The converted documents are compared with the declaration route itself: no
transition here is random. A network without a volume agrees exactly; a
volume that leaves a bound agrees within the integrator's measured spread
(the `integrator` level), never beyond it.

Oracle-free: both routes run on RAICHU. Component and flow names are
invented.
"""

from __future__ import annotations

import copy

import pytest

from pyraichu.muscadet.convert import _armed_tolerated, _Spread, compare_routes, convert
from test_declare_convert import component, document, flow_in, link

SUN = {"cls": "SinusoidalProfile", "amplitude": 6.0, "period": 12.0, "phase_shift": 2.0,
       "offset": 5.0, "value_min": 0.0, "value_max": float("inf"), "name": "SinusoidalProfile"}
DAY = {"cls": "SinusoidalProfile", "amplitude": -1.0, "period": 8.0, "phase_shift": 3.0,
       "offset": 1.5, "value_min": 0.0, "value_max": float("inf"), "name": "SinusoidalProfile"}


def c_in(name: str, demand=0.0, **extra) -> dict:
    return {"cls": "FlowContinuousIn", "name": name, "var_type": "float", "var_fed_default": 0.0,
            "component_authorized": [{"class_name_bkd": ".*"}], "var_in_default": 0.0,
            "var_demand_default": demand, **extra}


def c_out(name: str, rate=0.0, **extra) -> dict:
    return {"cls": "FlowContinuousOut", "name": name, "var_type": "float", "var_fed_default": rate,
            "component_authorized": [{"class_name_bkd": ".*"}], "var_demand_in_default": 0.0,
            "allocation": "proportional", "allocation_shares": {}, "allocation_priorities": {},
            "var_prod_cond_inner_mode": "or", **extra}


def volume(name: str, flows: list[str], capacity: float, content: dict, fill_rate, side="out",
           transmits=False) -> dict:
    return {"name": name, "flows": [{"name": f, "weight": 1.0, "side": side, "cls": "CapacityFlow"} for f in flows],
            "capacity": capacity, "side": side, "content_init": content, "transmits": transmits,
            "fill_rate": fill_rate}


def operand(name: str, **extra) -> dict:
    return {"name": name, "negate": False, "port": "in", "op": None, "value": None, "cls": "RuleOperand",
            **extra}


def rule(name: str, cond: list, cons: dict, prod: dict) -> dict:
    return {"name": name, "cond": cond, "cons": cons, "prod": prod, "cls": "Rule"}


def proved(declaration: dict, t_max=23.3, levels=("exact",)):
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    comparison = compare_routes(declaration, conversion, t_max=t_max)
    assert comparison.identical, comparison.detail
    assert comparison.level in levels, comparison.detail
    return conversion


#: Levels a converted volume may compare at: its bound departures fall a few
#: ulps apart on the two routes (`tolerance`), and from there the integrator
#: may step differently (`integrator`).
VOLUME_LEVELS = ("exact", "tolerance", "integrator")


def refusals(declaration: dict) -> list[str]:
    conversion = convert(declaration)
    assert not conversion.converted
    return [r.construct for r in conversion.refusals]


# --- declarations ------------------------------------------------------------


def bus_plant() -> dict:
    """Two sources feed a bus a sink draws from; a rival on the first source
    keeps the bus short of what its suppliers could give."""
    return document(
        [
            component("S1", [c_out("E", 3.0)], class_name="Generator"),
            component("S2", [c_out("E", 1.0, profile=SUN, publish_rate="capability")], class_name="Solar"),
            component("BUS", [c_in("E"), c_out("E")], class_name="Bus"),
            component("K", [c_in("E", 9.0)], class_name="Load"),
            component("R", [c_in("E", 2.0)], class_name="Load"),
        ],
        [link("S1", "BUS", "E"), link("S1", "R", "E"), link("S2", "BUS", "E"), link("BUS", "K", "E")],
    )


def split_plant() -> dict:
    """A profiled source split by shares between a load drawing the same
    flow and a load whose input bears another name."""
    return document(
        [
            component("S", [c_out("H2", 3.0, profile=SUN, allocation="shares",
                                  allocation_shares={"D": 0.25, "F": 0.75})], class_name="Source"),
            component("D", [c_in("H2", 2.0)], class_name="Load"),
            component("F", [c_in("Fuel", 2.0)], class_name="Burner"),
        ],
        [link("S", "D", "H2"),
         {"source": "S", "source_box": "H2_out", "target": "F", "target_box": "Fuel_in"}],
    )


def store_plant(fill_rate) -> dict:
    """A profiled source feeding a store and a rival consumer by shares; the
    store serves a profiled consumer and publishes what it receives."""
    return document(
        [
            component("S", [c_out("H2", 1.0, profile=SUN, allocation="shares",
                                  allocation_shares={"T": 0.5, "D": 0.5})], class_name="Source"),
            component("T", [c_in("H2", 1.0), c_out("H2", 1.0)], class_name="Tank")
            | {"capacities": [volume("tank", ["H2"], 6.0, {"H2": 1.0}, fill_rate)]},
            component("D", [c_in("H2", 2.0)], class_name="Load"),
            component("C", [c_in("H2", 1.0, profile=DAY, publish_rate="delivered")], class_name="Load"),
        ],
        [link("S", "T", "H2"), link("S", "D", "H2"), link("T", "C", "H2")],
    )


def machine_plant() -> dict:
    """A machine running while its command is fed and its electricity band
    holds, drawing from two suppliers (so it hands back their surplus),
    with a capped product."""
    machine = component(
        "M",
        [flow_in("run", logic="and", default=True), c_in("Water"), c_in("Elec"), c_out("Gas", max_rate=2.5)],
        class_name="Machine",
    )
    machine["rules"] = [{"name": "work", "rules": [
        rule("on", [operand("run"), operand("Elec", op=">=", value=6.0, release=3.0)],
             {"Water": 2.0, "Elec": 10.0}, {"Gas": 3.0}),
        rule("off", [], {"Water": 0.0, "Elec": 0.0}, {"Gas": 0.0}),
    ]}]
    return document(
        [
            component("W", [c_out("Water", 5.0)], class_name="Pump"),
            component("S", [c_out("Elec", 1.0, profile=SUN)], class_name="Solar"),
            component("G", [c_out("Elec", 2.0)], class_name="Grid"),
            machine,
            component("K", [c_in("Gas", 100.0)], class_name="Load"),
        ],
        [link("W", "M", "Water"), link("S", "M", "Elec"), link("G", "M", "Elec"), link("M", "K", "Gas")],
    )


# --- what converts -------------------------------------------------------------


def test_a_bus_hands_on_what_arrives():
    conversion = proved(bus_plant())
    bus = next(t for t in conversion.library["library"]["types"] if t["name"] == "Bus")
    assert {"fragment": "muscadet.pass_through", "with": {"name": "E"}} in bus["uses"]


def test_a_split_names_each_consumer_by_its_own_input():
    conversion = proved(split_plant())
    source = next(t for t in conversion.library["library"]["types"] if t["name"] == "Source")
    out = next(u for u in source["uses"] if u["fragment"] == "muscadet.continuous_flow_out")
    assert {(s["to"]["component"], s["to"]["port"]) for s in out["with"]["shares"]} == {
        ("D", "H2_in"),
        ("F", "Fuel_in"),
    }


def test_a_conditioned_rule_set_with_a_band_converts():
    conversion = proved(machine_plant())
    machine = next(t for t in conversion.library["library"]["types"] if t["name"] == "Machine")
    fragments = [u["fragment"] for u in machine["uses"]]
    # Two suppliers on a consumed input: the surplus goes back.
    assert fragments.count("muscadet.surplus_release") == 1


def test_instances_of_one_class_share_a_type_with_their_rates_as_parameters():
    conversion = proved(bus_plant())
    loads = [i for i in conversion.system["system"]["instances"] if i["name"] in ("K", "R")]
    assert {i["type"] for i in loads} == {"Load"}
    assert sorted(next(iter(i["parameters"].values())) for i in loads) == [2.0, 9.0]


def test_a_conversion_imports_muscadet_and_std_for_the_generic_fragments_it_applies():
    """The bus plant's profiled source applies `std.time_profile` beside
    the `muscadet` flow fragments: the library imports both, `muscadet`
    first."""
    assert convert(bus_plant()).library["library"]["imports"] == ["muscadet", "std"]


def test_a_converted_rate_altered_is_reported_divergent():
    """Negative control: the comparison is not vacuous."""
    declaration = bus_plant()
    conversion = convert(declaration)
    altered = copy.deepcopy(conversion)
    generator = next(t for t in altered.library["library"]["types"] if t["name"] == "Generator")
    source = next(u for u in generator["uses"] if u["fragment"] == "muscadet.continuous_source")
    source["with"]["rate"] = 3.5
    assert not compare_routes(declaration, altered, t_max=24.0).identical


def test_a_converted_profile_period_altered_is_reported_divergent():
    """Negative control on a network with no event: halving a profile's
    period keeps its value at the half and the whole of a horizon that is
    a multiple of it, so the comparison must look elsewhere too."""
    declaration = bus_plant()
    conversion = convert(declaration)
    altered = copy.deepcopy(conversion)
    solar = next(t for t in altered.library["library"]["types"] if t["name"] == "Solar")
    profile = next(u for u in solar["uses"] if u["fragment"] == "std.time_profile")
    profile["with"]["period"] = 6.0
    assert not compare_routes(declaration, altered, t_max=24.0).identical


# --- what is refused by name ---------------------------------------------------


def mixed_room_plant() -> dict:
    return document(
        [
            component("S", [c_out("A", 2.0), c_out("B", 1.0)], class_name="Source"),
            component("T", [c_in("A"), c_in("B"), c_out("A"), c_out("B")], class_name="Room")
            | {"capacities": [volume("room", ["A", "B"], 10.0, {"A": 1.0, "B": 1.0}, 0.0)]},
            component("C", [c_in("A", 1.0), c_in("B", 1.0)], class_name="Load"),
        ],
        [link("S", "T", "A"), link("S", "T", "B"), link("T", "C", "A"), link("T", "C", "B")],
    )


def uses_of(conversion, type_name: str) -> list[dict]:
    return next(t for t in conversion.library["library"]["types"] if t["name"] == type_name)["uses"]


def test_a_mixed_volume_converts_to_pooled_serving():
    conversion = proved(mixed_room_plant(), levels=VOLUME_LEVELS)
    fragments = [use["fragment"] for use in uses_of(conversion, "Room")]
    assert "muscadet.volume_pooled" in fragments
    assert fragments.count("muscadet.volume_pooled_flow") == 2


def test_a_mixed_volume_flow_without_an_outlet_is_refused():
    declaration = mixed_room_plant()
    declaration["components"]["T"]["flows"] = [
        f for f in declaration["components"]["T"]["flows"] if not (f["name"] == "B" and f["cls"] == "FlowContinuousOut")
    ]
    declaration["connections"] = [c for c in declaration["connections"] if not (c["source"] == "T" and c["source_box"] == "B_out")]
    # The declaration route's reader refuses it before the converter asks.
    assert refusals(declaration) == ["declaration refused by the reader"]


def test_a_single_flow_volume_converts():
    single = document(
        [
            component("S", [c_out("A", 2.0)], class_name="Source"),
            component("T", [c_in("A", 1.0), c_out("A", 1.0)], class_name="Room")
            | {"capacities": [volume("room", ["A"], 10.0, {"A": 1.0}, 0.0)]},
            component("C", [c_in("A", 1.0)], class_name="Load"),
        ],
        [link("S", "T", "A"), link("T", "C", "A")],
    )
    proved(single, levels=VOLUME_LEVELS)


def test_a_transmitting_volume_converts():
    declaration = store_plant(0.5)
    declaration["components"]["T"]["capacities"][0]["transmits"] = True
    conversion = proved(declaration, levels=VOLUME_LEVELS)
    volume_uses = [
        u for ty in conversion.library["library"]["types"] for u in ty["uses"]
        if u["fragment"] in ("muscadet.volume_flow", "muscadet.volume_outlet")
    ]
    assert volume_uses and all(u["with"]["transmits"] is True for u in volume_uses)


def ring_plant(held=("Leak", "Air"), cascade: bool = False, leak_init: float = 1.0) -> dict:
    """A leak into a room with a through-path, a fan extracting it to a
    purge and renewing the air, blowing the air it drew back into the room.
    With `cascade`, the extraction exhausts into the renewal's exhaust
    instead, ahead of it."""
    local = component("LOCAL", [c_in("Leak"), c_in("Air"), c_out("Leak"), c_out("Air")], class_name="Room") | {
        "capacities": [volume("room", list(held), 100.0, {f: (89.0 if f == "Air" else leak_init) for f in held},
                              float("inf"), transmits=True)]}
    purge = "Exhaust" if cascade else "Purge"
    vent = component("VENT", [c_in("Leak"), c_in("Air"), c_out("Exhaust"), c_out("Air")]
                     + ([] if cascade else [c_out("Purge")]), class_name="Ventilation")
    vent["rules"] = [
        {"name": "extraction", "rules": [rule("go", [], {"Leak": 2.0}, {purge: 2.0})]},
        {"name": "renewal", "rules": [rule("go", [], {"Air": 25.0}, {"Exhaust": 25.0, "Air": 25.0})]},
    ]
    components = [component("LEAK", [c_out("Leak", 0.05)], class_name="Leak"), local, vent,
                  component("ATM", [c_in("Exhaust", 50.0)], class_name="Outside")]
    links = [link("LEAK", "LOCAL", "Leak"), link("LOCAL", "VENT", "Leak"), link("LOCAL", "VENT", "Air"),
             link("VENT", "LOCAL", "Air"), link("VENT", "ATM", "Exhaust")]
    if not cascade:
        components.append(component("STACK", [c_in("Purge", 1.0)], class_name="Outside"))
        links.append(link("VENT", "STACK", "Purge"))
    return document(components, links)


def test_a_recirculation_ring_converts_torn_where_the_declaration_route_tears_it():
    conversion = proved(ring_plant(), t_max=10.0, levels=VOLUME_LEVELS)
    vent = uses_of(conversion, "Ventilation")
    renewal = next(u for u in vent if u["fragment"] == "muscadet.rule_set" and u["with"]["name"] == "renewal")
    assert renewal["with"]["torn"] == ["Air"]
    extraction = next(u for u in vent if u["fragment"] == "muscadet.rule_set" and u["with"]["name"] == "extraction")
    assert "torn" not in extraction["with"]
    # The fan's inputs come out of a pooled room: each hands back what its
    # rule does not consume.
    released = {u["with"]["name"] for u in vent if u["fragment"] == "muscadet.surplus_release"}
    assert released == {"Leak", "Air"}


def test_a_cascade_after_a_set_fed_by_a_mixed_volume_is_refused():
    """With hydrogen stocked in the room, the extraction asks the whole
    exhaust demand but makes only what the pooled draw hands it; the
    declaration route asks the renewal what the extraction is expected to
    make, which no fragment carries yet."""
    assert refusals(ring_plant(cascade=True)) == ["output produced by two rule sets, an earlier one fed by a mixed volume"]


def group_plant(extra_consumer: bool = False, run_cond: list | None = None) -> dict:
    """A room fed without bound, drawn by one extractor at a volumetric
    rate; optionally a per-flow consumer beside it, which the declaration
    route refuses."""
    room = component("ROOM", [c_in("H2"), c_in("AIR"), c_out("H2"), c_out("AIR")], class_name="Room") | {
        "capacities": [volume("room", ["H2", "AIR"], 100.0, {"H2": 1.0, "AIR": 49.0}, float("inf"))]}
    group = {"name": "extraction", "flows": ["H2", "AIR"], "flow_rate": 5.0}
    if run_cond is not None:
        group["run_cond"] = run_cond
    fan = component("FAN", [c_in("H2"), c_in("AIR")] + ([flow_in("vent")] if run_cond else []), class_name="Fan") | {
        "mixtures": [group]}
    components = [component("SH2", [c_out("H2", 0.5)], class_name="Source"),
                  component("SAIR", [c_out("AIR", 4.0)], class_name="Source"), room, fan]
    links = [link("SH2", "ROOM", "H2"), link("SAIR", "ROOM", "AIR"), link("ROOM", "FAN", "H2"),
             link("ROOM", "FAN", "AIR")]
    if extra_consumer:
        components.append(component("K", [c_in("H2", 1.0)], class_name="Load"))
        links.append(link("ROOM", "K", "H2"))
    return document(components, links)


def test_a_mixture_group_converts():
    conversion = proved(group_plant(), t_max=20.0, levels=VOLUME_LEVELS)
    fan = uses_of(conversion, "Fan")
    assert {"fragment": "muscadet.mixture_in",
            "with": {"name": "extraction", "flows": ["H2", "AIR"], "flow_rate": 5.0}} in fan
    room = uses_of(conversion, "Room")
    outs = [u["with"] for u in room if u["fragment"] == "muscadet.continuous_flow_out"]
    assert {o["available"] for o in outs} == {"mixture"}
    assert all(u["with"]["drawn"] for u in room if u["fragment"] == "muscadet.volume_pooled_flow")


def test_a_group_the_declaration_route_refuses_is_refused_with_its_reason():
    conversion = convert(group_plant(extra_consumer=True))
    assert not conversion.converted
    shape = [r for r in conversion.refusals if r.construct == "mixture group shape"]
    assert shape and "also serves ['K']" in shape[0].detail


def test_a_flow_system_the_reader_refuses_converts_nothing():
    """Every component reads, but a connection names a component nobody
    declares: the whole-graph shapes are unknown, so nothing converts."""
    declaration = group_plant()
    declaration["connections"].append(link("ROOM", "GHOST", "H2"))
    conversion = convert(declaration)
    assert not conversion.converted
    assert ("system", "declaration refused by the reader") in [(r.component, r.construct) for r in conversion.refusals]


def test_a_mixture_group_run_command_is_refused():
    assert "mixture group run command" in refusals(group_plant(run_cond=[["vent"]]))


def test_a_contested_output_converts_unless_capped():
    """Two rule sets of one machine, one always on and one guarded,
    produce its gas: served in declaration order they convert, except
    under a maximum rate on the shared output."""
    declaration = machine_plant()
    machine = declaration["components"]["M"]
    machine["rules"] = [
        {"name": "one", "rules": [rule("go", [], {"Water": 1.0}, {"Gas": 1.0})]},
        {"name": "two", "rules": [rule("on", [operand("run")], {"Elec": 1.0}, {"Gas": 1.0}),
                                  rule("off", [], {}, {"Gas": 0.0})]},
    ]
    machine["rules"][0]["apportionment"] = {}
    assert sorted(refusals(declaration)) == [
        "maximum rate on an always-on rule's output",
        "maximum rate on an output produced by two rule sets",
    ]
    _flow(declaration, "M", "FlowContinuousOut", "Gas").pop("max_rate")
    proved(declaration)


def ventilation_plant() -> dict:
    """An extraction and an air renewal sharing one exhaust, as the
    reference plant's ventilation does: the extraction, declared first and
    held by a band on its leak supply, is asked the whole exhaust demand,
    the renewal what it leaves."""
    vent = component(
        "V", [flow_in("vent", logic="or", default=True), c_in("Leak"), c_in("Air"), c_out("Air"),
              c_out("Exhaust")], class_name="Ventilation",
    )
    vent["rules"] = [
        {"name": "extraction", "rules": [
            rule("running", [operand("vent"), operand("Leak", op=">=", value=2.0, release=1.0)],
                 {"Leak": 2.0}, {"Exhaust": 2.0}),
            rule("idle", [], {"Leak": 0.0}, {"Exhaust": 0.0})]},
        {"name": "renewal", "rules": [
            rule("running", [operand("vent")], {"Air": 1.0}, {"Air": 1.0, "Exhaust": 1.0}),
            rule("idle", [], {"Air": 0.0}, {"Air": 0.0, "Exhaust": 0.0})]},
    ]
    return document(
        [
            component("L", [c_out("Leak", 1.0, profile=SUN)], class_name="Leak"),
            component("F", [c_out("Air", 4.0)], class_name="Fresh"),
            vent,
            component("X", [c_in("Exhaust", 1.0, profile=DAY)], class_name="Outside"),
            component("R", [c_in("Air", 9.0)], class_name="Room"),
        ],
        [link("L", "V", "Leak"), link("F", "V", "Air"), link("V", "X", "Exhaust"), link("V", "R", "Air")],
    )


def test_an_output_produced_by_two_rule_sets_converts():
    conversion = proved(ventilation_plant())
    vent = next(t for t in conversion.library["library"]["types"] if t["name"] == "Ventilation")
    assert {"fragment": "muscadet.shared_output", "with": {"flow": "Exhaust", "producers": ["extraction", "renewal"]}} in vent["uses"]
    assert {"fragment": "muscadet.rule_cascade",
            "with": {"rule_set": "renewal", "flow": "Exhaust", "before": ["extraction"]}} in vent["uses"]


def test_a_third_producer_is_asked_what_both_earlier_ones_leave():
    declaration = ventilation_plant()
    vent = declaration["components"]["V"]
    vent["flows"].append(c_in("Draft"))
    vent["rules"].append(
        {"name": "draft", "rules": [
            rule("running", [operand("vent")], {"Draft": 1.0}, {"Exhaust": 1.0}),
            rule("idle", [], {"Draft": 0.0}, {"Exhaust": 0.0})]}
    )
    declaration["components"]["D"] = component("D", [c_out("Draft", 2.0, profile=SUN)], class_name="Draft")
    declaration["connections"].append(link("D", "V", "Draft"))
    # Asked more than the extraction and the renewal can make together.
    declaration["components"]["X"]["flows"][0]["var_demand_default"] = 6.0
    conversion = proved(declaration)
    types = conversion.library["library"]["types"]
    uses = next(t for t in types if t["name"] == "Ventilation")["uses"]
    assert {"fragment": "muscadet.rule_cascade",
            "with": {"rule_set": "draft", "flow": "Exhaust", "before": ["extraction", "renewal"]}} in uses


def test_a_cascade_served_in_the_other_order_is_reported_divergent():
    """Negative control: the renewal asked first is not the declaration."""
    declaration = ventilation_plant()
    conversion = convert(declaration)
    altered = copy.deepcopy(conversion)
    vent = next(t for t in altered.library["library"]["types"] if t["name"] == "Ventilation")
    for use in vent["uses"]:
        if use["fragment"] == "muscadet.rule_cascade":
            use["with"].update(rule_set="extraction", before=["renewal"])
        elif use["fragment"] == "muscadet.rule_set":
            produced = use["with"]["produces"] + use["with"]["cascaded"]
            first = use["with"]["name"] == "renewal"
            use["with"]["produces"] = [f for f in produced if first or f != "Exhaust"]
            use["with"]["cascaded"] = [f for f in produced if not first and f == "Exhaust"]
    assert not compare_routes(declaration, altered, t_max=23.3).identical


def test_an_input_side_volume_no_rule_draws_from_is_refused():
    declaration = store_plant(0.5)
    declaration["components"]["T"]["capacities"] = [volume("tank", ["H2"], 6.0, {"H2": 1.0}, 0.5, side="in")]
    assert refusals(declaration) == ["input-side volume on a flow no rule draws"]


def membrane_plant(fill_rate=float("inf"), content=None) -> dict:
    """Two leaks gathered in a membrane upstream of two rules always on,
    each venting one of them, as the reference plant's stack does."""
    membrane = component(
        "M", [c_in("A"), c_in("B"), c_out("A_vent"), c_out("B_vent")], class_name="Membrane"
    ) | {"capacities": [volume("membrane", ["A", "B"], 4.0, content or {"A": 0.0, "B": 0.0}, fill_rate,
                               side="in", transmits=True)]}
    membrane["rules"] = [
        {"name": "ra", "rules": [rule("venting", [], {"A": 1.0}, {"A_vent": 1.0})]},
        {"name": "rb", "rules": [rule("venting", [], {"B": 1.0}, {"B_vent": 1.0})]},
    ]
    return document(
        [
            component("SA", [c_out("A", 1.0, profile=SUN)], class_name="Stack"),
            component("SB", [c_out("B", 0.6)], class_name="Cell"),
            membrane,
            component("KA", [c_in("A_vent", 1.0, profile=DAY)], class_name="Room"),
            component("KB", [c_in("B_vent", 0.5)], class_name="Air"),
        ],
        [link("SA", "M", "A"), link("SB", "M", "B"), link("M", "KA", "A_vent"), link("M", "KB", "B_vent")],
    )


def always_on_plant() -> dict:
    """A rule declared without a condition, always applying."""
    machine = component("M", [c_in("Water"), c_out("Gas")], class_name="Machine")
    machine["rules"] = [{"name": "work", "rules": [rule("go", [], {"Water": 2.0}, {"Gas": 3.0})]}]
    return document(
        [
            component("W", [c_out("Water", 1.0, profile=SUN)], class_name="Pump"),
            machine,
            component("K", [c_in("Gas", 1.0, profile=DAY)], class_name="Load"),
        ],
        [link("W", "M", "Water"), link("M", "K", "Gas")],
    )


def test_an_always_on_rule_set_converts():
    conversion = proved(always_on_plant())
    machine = next(t for t in conversion.library["library"]["types"] if t["name"] == "Machine")
    rule_set = next(u for u in machine["uses"] if u["fragment"] == "muscadet.rule_set")
    assert rule_set["with"]["always_on"] is True


@pytest.mark.parametrize("fill", [float("inf"), 0.5])
@pytest.mark.parametrize("t_max", [7.0, 23.3])
def test_a_volume_upstream_of_rules_always_on_converts(fill, t_max):
    proved(membrane_plant(fill), t_max=t_max, levels=VOLUME_LEVELS)


def feeder_plant(content: float, transmits: bool) -> dict:
    """A single-flow hopper upstream of a rule always on that also draws a
    second flow directly."""
    feeder = component("F", [c_in("Fuel"), c_in("Air"), c_out("Heat")], class_name="Burner") | {
        "capacities": [volume("hopper", ["Fuel"], 1.5, {"Fuel": content}, 0.5, side="in", transmits=transmits)]
    }
    feeder["rules"] = [{"name": "burn", "rules": [rule("go", [], {"Fuel": 2.0, "Air": 1.0}, {"Heat": 3.0})]}]
    return document(
        [
            component("S", [c_out("Fuel", 1.0, profile=SUN)], class_name="Silo"),
            component("A", [c_out("Air", 2.0)], class_name="Blower"),
            feeder,
            component("H", [c_in("Heat", 1.0, profile=DAY)], class_name="Room"),
        ],
        [link("S", "F", "Fuel"), link("A", "F", "Air"), link("F", "H", "Heat")],
    )


@pytest.mark.parametrize("content", [0.0, 1.0])
@pytest.mark.parametrize("transmits", [True, False])
def test_a_single_flow_hopper_beside_a_direct_input_converts(content, transmits):
    proved(feeder_plant(content, transmits), levels=VOLUME_LEVELS)


def test_a_membrane_starting_stocked_converts():
    proved(membrane_plant(content={"A": 1.0, "B": 0.0}), levels=VOLUME_LEVELS)


def test_a_membrane_altered_is_reported_divergent():
    """Negative control: the volume a rule draws from is observed."""
    declaration = membrane_plant()
    altered = copy.deepcopy(convert(declaration))
    membrane = next(t for t in altered.library["library"]["types"] if t["name"] == "Membrane")
    for use in membrane["uses"]:
        if use["fragment"] in ("muscadet.volume_flow", "muscadet.volume_total", "muscadet.volume_stock"):
            use["with"]["capacity"] *= 1.0 + 1e-3
    assert not compare_routes(declaration, altered, t_max=23.3).identical


def test_two_guarded_rules_are_refused_naming_the_count():
    declaration = machine_plant()
    rules = declaration["components"]["M"]["rules"][0]["rules"]
    rules[1] = rule("half", [operand("run")], {"Water": 1.0, "Elec": 5.0}, {"Gas": 1.0})
    conversion = convert(declaration)
    assert [r.construct for r in conversion.refusals] == ["multi-rule rule set"]
    assert "2 guarded and 0 idle rules" in conversion.refusals[0].detail


@pytest.mark.parametrize("fill", [0.0, 0.5, float("inf")])
@pytest.mark.parametrize("t_max", [7.0, 23.3, 40.0])
def test_a_store_converts_whatever_its_fill(fill, t_max):
    proved(store_plant(fill), t_max=t_max, levels=VOLUME_LEVELS)


def test_a_store_leaving_a_bound_compares_at_the_integrator_level():
    """Measured: the store runs empty and refills; the two routes date that
    departure a few ulps apart and step differently from there."""
    declaration = store_plant(0.0)
    comparison = compare_routes(declaration, convert(declaration), t_max=40.0)
    assert comparison.identical, comparison.detail
    assert comparison.level == "integrator", comparison.detail
    assert "tank_content_H2" in comparison.detail


@pytest.mark.parametrize("fill", [0.0, 0.5])
def test_a_store_capacity_altered_by_a_millionth_is_reported_divergent(fill):
    """Negative control: the integrator's spread on this plant is below
    1e-6, so a capacity one millionth larger is not swallowed."""
    declaration = store_plant(fill)
    altered = copy.deepcopy(convert(declaration))
    tank = next(t for t in altered.library["library"]["types"] if t["name"] == "Tank")
    for use in tank["uses"]:
        if use["fragment"] in ("muscadet.volume_flow", "muscadet.volume_total"):
            use["with"]["capacity"] *= 1.0 + 1e-6
    comparison = compare_routes(declaration, altered, t_max=40.0)
    assert not comparison.identical, comparison.detail


def test_a_network_without_an_ode_gets_no_integrator_spread():
    """Negative control: a rate one part in 1e10 off is beyond float
    reordering, and a network with no ODE has no spread to hide it in."""
    declaration = bus_plant()
    altered = copy.deepcopy(convert(declaration))
    generator = next(t for t in altered.library["library"]["types"] if t["name"] == "Generator")
    source = next(u for u in generator["uses"] if u["fragment"] == "muscadet.continuous_source")
    source["with"]["rate"] *= 1.0 + 1e-10
    comparison = compare_routes(declaration, altered, t_max=24.0)
    assert not comparison.identical, comparison.detail


def test_the_integrator_spread_is_kept_per_transition_and_per_value():
    """A date spread measured on one transition loosens no other; a value
    spread measured on one attribute loosens no other."""
    spread = _Spread(values={"T.content": 1e-7}, dates={"T.bounds.reach_full": 1e-6})
    assert spread.admits_date(10.0, 10.0 + 5e-7, "T.bounds.reach_full")
    assert not spread.admits_date(10.0, 10.0 + 5e-7, "T.cycle.delay_done")
    assert spread.admits_value(1.0, 1.0 + 5e-8, "T.content")
    assert not spread.admits_value(1.0, 1.0 + 5e-8, "T.rate")
    assert _Spread(values={}, dates={}).armed == 0.0


def test_an_armed_age_beyond_float_reordering_is_never_reported_exact():
    """An armed transition whose age alone differs beyond 1e-12 needs the
    integrator's spread: it is named under that level, never passed
    silently."""
    left = {"M.fail": ('{"distrib": "exp"}', 3.0, 1.0, True, 0.5)}
    right = {"M.fail": ('{"distrib": "exp"}', 3.0, 1.0 + 1e-8, True, 0.5)}
    integrated: set[str] = set()
    assert _armed_tolerated(left, right, integrated) == set()
    assert integrated == {"armed M.fail"}
    # An age read at a shifted date, alike within float reordering, is
    # the reading's own rounding: named nowhere.
    right = {"M.fail": ('{"distrib": "exp"}', 3.0, 1.0 + 1e-15, True, 0.5)}
    integrated = set()
    assert _armed_tolerated(left, right, integrated) == set() and integrated == set()


def test_a_store_output_keeps_its_published_rate():
    declaration = store_plant(0.5)
    _flow(declaration, "T", "FlowContinuousOut", "H2").update(publish_rate="capability")
    conversion = proved(declaration, levels=VOLUME_LEVELS)
    tank = next(t for t in conversion.library["library"]["types"] if t["name"] == "Tank")
    assert {"fragment": "muscadet.rate_publisher", "with": {"name": "H2", "of": "capability"}} in tank["uses"]


def _flow(declaration: dict, component: str, cls: str, name: str) -> dict:
    return next(f for f in declaration["components"][component]["flows"] if f["cls"] == cls and f["name"] == name)


def _rule_set(declaration: dict) -> dict:
    return declaration["components"]["M"]["rules"][0]


def _store() -> dict:
    return store_plant(0.5)


#: Bases of `REFUSED_EDITS` holding a volume, which may compare beyond exact.
STORE_BASES = (_store, membrane_plant)


#: One field changed on a converting plant, and the construct it is refused as.
REFUSED_EDITS = {
    "allocation by priority": (
        bus_plant,
        lambda d: _flow(d, "S1", "FlowContinuousOut", "E").update(allocation="priority"),
        "allocation `priority`",
    ),
    "shares missing a consumer": (
        split_plant,
        lambda d: _flow(d, "S", "FlowContinuousOut", "H2").update(allocation_shares={"D": 1.0}),
        "allocation shares not naming each consumer once",
    ),
    "capability published on an input": (
        bus_plant,
        lambda d: _flow(d, "K", "FlowContinuousIn", "E").update(publish_rate="capability"),
        "published input rate `capability`",
    ),
    "profile on a pass-through input": (
        bus_plant,
        lambda d: _flow(d, "BUS", "FlowContinuousIn", "E").update(profile=DAY),
        "profile on an input that is not a sink",
    ),
    "profile on a pass-through output": (
        bus_plant,
        lambda d: _flow(d, "BUS", "FlowContinuousOut", "E").update(profile=SUN),
        "profile on an output that is not a source",
    ),
    "negated guard": (
        machine_plant,
        lambda d: _rule_set(d)["rules"][0]["cond"][0].update(negate=True),
        "rule guard operand",
    ),
    "threshold without a release": (
        machine_plant,
        lambda d: _rule_set(d)["rules"][0]["cond"][1].update(release=None),
        "rule guard operand",
    ),
    "idle rule moving a flow": (
        machine_plant,
        lambda d: _rule_set(d)["rules"][1]["cons"].update(Water=1.0),
        "idle rule moving a flow",
    ),
    "apportionment": (
        machine_plant,
        lambda d: _rule_set(d).update(apportionment={"Gas": 1.0}),
        "rule set apportionment",
    ),
    "store starting full": (
        _store,
        lambda d: d["components"]["T"]["capacities"][0].update(content_init={"H2": 6.0}),
        "volume starting full",
    ),
    "store with a finite serve rate": (
        _store,
        lambda d: d["components"]["T"]["capacities"][0].update(serve_rate=2.0),
        "volume serve rate",
    ),
    "condition on a rule drawing from a volume": (
        membrane_plant,
        lambda d: d["components"]["M"]["rules"][0]["rules"].insert(
            0, rule("on", [operand("A", op=">=", value=1.0, release=0.5)], {"A": 1.0}, {"A_vent": 1.0})
        ),
        "conditioned rule set drawing from a volume",
    ),
    "membrane starting full": (
        membrane_plant,
        lambda d: d["components"]["M"]["capacities"][0].update(content_init={"A": 3.0, "B": 1.0}),
        "volume starting full",
    ),
    "zero coefficient in an always-on rule": (
        always_on_plant,
        lambda d: d["components"]["M"]["rules"][0]["rules"][0]["cons"].update(Water=0.0),
        "zero coefficient in an always-on rule",
    ),
    "always-on rule consuming nothing": (
        always_on_plant,
        lambda d: d["components"]["M"]["rules"][0]["rules"][0].update(cons={}),
        "always-on rule consuming nothing",
    ),
    "maximum rate on an always-on output": (
        always_on_plant,
        lambda d: _flow(d, "M", "FlowContinuousOut", "Gas").update(max_rate=1.0),
        "maximum rate on an always-on rule's output",
    ),
    "apportioned output produced by two rule sets": (
        ventilation_plant,
        lambda d: [r.update(apportionment={"Exhaust": 1.0}) for r in d["components"]["V"]["rules"]],
        "apportioned output produced by two rule sets",
    ),
    "maximum rate on an output produced by two rule sets": (
        ventilation_plant,
        lambda d: _flow(d, "V", "FlowContinuousOut", "Exhaust").update(max_rate=3.0),
        "maximum rate on an output produced by two rule sets",
    ),
    "input consumed by two rule sets": (
        machine_plant,
        lambda d: d["components"]["M"]["rules"].append(
            {"name": "cool", "rules": [rule("on", [operand("run")], {"Water": 1.0}, {}),
                                      rule("off", [], {"Water": 0.0}, {})]}
        ),
        "input consumed by two rule sets",
    ),
}


@pytest.mark.parametrize("edit", sorted(REFUSED_EDITS))
def test_an_unproved_construct_is_refused_by_name(edit):
    base, change, construct = REFUSED_EDITS[edit]
    declaration = base()
    proved(base(), levels=VOLUME_LEVELS if base in STORE_BASES else ("exact",))
    change(declaration)
    assert refusals(declaration) == [construct]


# --- controllers ---------------------------------------------------------------


def reading(name: str, kind="level", flows=(), aggregate=None, default=0.0) -> dict:
    defaults = {"level_default": 0.0, "fill_default": 0.0, "rate_default": 0.0, "ratio_default": 0.0}
    return {"name": name, "kind": kind, "flows": list(flows), **defaults, f"{kind}_default": default,
            "aggregate": aggregate}


def republished(name: str, of: str, gain=1.0) -> dict:
    return {"name": name, "kind": "value", "flows": [], "level_default": 0.0, "fill_default": 0.0,
            "ratio_default": 0.0, "emit": {"op": "republish", "input": of, "gain": gain}}


def banded(name: str, of: str, direction: str, activate: float, release: float) -> dict:
    return {"name": name, "kind": "bool", "default": False, "emit": {
        "op": "band", "input": of, "direction": direction, "activate": activate, "release": release}}


def controller(name: str, controls_in: list, controls_out: list) -> dict:
    return {"name": name, "kind": "controller", "cls": "ObjCtrl", "source_cls": "ObjCtrl",
            "controls_in": controls_in, "controls_out": controls_out}


def box(source: str, source_box: str, target: str, target_box: str) -> dict:
    return {"source": source, "source_box": source_box, "target": target, "target_box": target_box}


def instrumented_store_plant() -> dict:
    """The store plant read by instruments: one republishes the store's
    level, a voter switches a band on the median of what it republishes,
    and a meter republishes the rate the source delivers."""
    declaration = store_plant(0.5)
    source = declaration["components"]["S"]
    source["flows"][0]["publish_rate"] = "delivered"
    declaration["components"] |= {
        "CT": controller("CT", [reading("tank", default=1.0)], [republished("tank_reading", "tank")]),
        "AP": controller("AP", [reading("tank_reading", aggregate="median", default=1.0)],
                         [banded("low", "tank_reading", "below", 1.5, 3.0)]),
        "MT": controller("MT", [reading("H2", kind="rate", aggregate="sum")], [republished("supply", "H2")]),
    }
    declaration["connections"] += [
        box("T", "tank_level_out", "CT", "tank_level_in"),
        box("CT", "tank_reading_level_out", "AP", "tank_reading_level_in"),
        box("S", "H2_rate_out", "MT", "H2_rate_in"),
    ]
    return declaration


def test_instruments_on_a_store_convert():
    conversion = proved(instrumented_store_plant(), levels=VOLUME_LEVELS)
    types = {t["name"]: t for t in conversion.library["library"]["types"]}
    assert {"fragment": "std.aggregate_measure",
            "with": {"name": "tank_reading_level", "agg": "median", "initial": 1.0}} in types["AP"]["uses"]
    assert {"fragment": "muscadet.rate_publisher",
            "with": {"name": "H2", "of": "delivered", "side": "out"}} in types["Source"]["uses"]


def test_a_band_released_at_another_threshold_is_reported_divergent():
    """Negative control: the voter's band moved in the converted documents."""
    declaration = instrumented_store_plant()
    conversion = convert(declaration)
    altered = copy.deepcopy(conversion)
    voter = next(t for t in altered.library["library"]["types"] if t["name"] == "AP")
    band = next(u for u in voter["uses"] if u["fragment"] == "std.threshold_band")
    band["with"]["release"] = 3.5
    assert not compare_routes(declaration, altered, t_max=23.3).identical


def test_a_controller_reads_the_share_a_volume_publishes():
    declaration = membrane_plant()
    declaration["components"]["CT"] = controller(
        "CT", [reading("membrane", kind="ratio", flows=["A"])],
        [banded("rich", "membrane", "above", 0.6, 0.4)])
    declaration["connections"].append(box("M", "membrane_level_out", "CT", "membrane_level_in"))
    conversion = proved(declaration, levels=VOLUME_LEVELS)
    assert {"from": "M.membrane_ratio_A_out", "to": "CT.membrane_ratio_A_in"} in (
        conversion.system["system"]["connections"])


CONTROLLER_REFUSALS = {
    "combine": (
        lambda d: d["components"]["AP"]["controls_out"].__setitem__(0, {
            "name": "low", "kind": "bool", "default": False, "emit": {"op": "combine", "logic": "not", "operands": [
                {"op": "compare", "input": "tank_reading", "operator": "<", "threshold": 2.0}]}}),
        "controller operator `combine`",
    ),
    "default true": (
        lambda d: d["components"]["AP"]["controls_out"][0].update(default=True),
        "boolean controller output defaulting to true",
    ),
    "failure mode": (
        lambda d: d["components"].update(CT__fail={
            "name": "CT__fail", "kind": "two_state_mode", "cls": "ObjFMDelay", "fm_name": "fail",
            "targets": ["CT"], "target_name": "CT", "failure_effects": {"tank_reading_level_gain": 0.0},
            "failure_param_name": ["ttf"], "failure_param": [5.0], "repair_effects": {"tank_reading_level_gain": 1.0},
            "repair_param_name": ["ttr"], "repair_param": [3.0]}),
        "failure mode on a controller",
    ),
    "emptied observer": (
        lambda d: d["components"]["AP"].update(controls_in=[], controls_out=[]),
        "controller declaration",
    ),
    "unknown publisher box": (
        lambda d: next(c for c in d["connections"] if c["source"] == "CT").update(source_box="drift_level_out"),
        "controller connection",
    ),
}


@pytest.mark.parametrize("edit", sorted(CONTROLLER_REFUSALS))
def test_an_unbenched_controller_construct_is_refused_by_name(edit):
    change, construct = CONTROLLER_REFUSALS[edit]
    declaration = instrumented_store_plant()
    change(declaration)
    assert construct in refusals(declaration)
