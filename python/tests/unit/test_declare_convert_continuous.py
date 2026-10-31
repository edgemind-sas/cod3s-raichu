"""The muscadet declaration converter on continuous flows.

A continuous network converts into the std fragments each construct's
plugin-versus-library bench proves (`test_library_std_continuous.py`):
sources and sinks with their profiles and published rates, split policies,
pass-throughs, and rule sets with one guarded rule and one idle rule. Every
other construct, volumes included, is refused by name. The converted
documents are compared with the declaration route itself, exactly: no
transition here is random.

Oracle-free: both routes run on RAICHU. Component and flow names are
invented.
"""

from __future__ import annotations

import copy

import pytest

from pyraichu.muscadet.convert import compare_routes, convert
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


def proved(declaration: dict, t_max=23.3):
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    comparison = compare_routes(declaration, conversion, t_max=t_max)
    assert comparison.identical, comparison.detail
    assert comparison.level == "exact", comparison.detail
    return conversion


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
    assert {"fragment": "std.pass_through", "with": {"name": "E"}} in bus["uses"]


def test_a_split_names_each_consumer_by_its_own_input():
    conversion = proved(split_plant())
    source = next(t for t in conversion.library["library"]["types"] if t["name"] == "Source")
    out = next(u for u in source["uses"] if u["fragment"] == "std.continuous_flow_out")
    assert {(s["to"]["component"], s["to"]["port"]) for s in out["with"]["shares"]} == {
        ("D", "H2_in"),
        ("F", "Fuel_in"),
    }


def test_a_conditioned_rule_set_with_a_band_converts():
    conversion = proved(machine_plant())
    machine = next(t for t in conversion.library["library"]["types"] if t["name"] == "Machine")
    fragments = [u["fragment"] for u in machine["uses"]]
    # Two suppliers on a consumed input: the surplus goes back.
    assert fragments.count("std.surplus_release") == 1


def test_instances_of_one_class_share_a_type_with_their_rates_as_parameters():
    conversion = proved(bus_plant())
    loads = [i for i in conversion.system["system"]["instances"] if i["name"] in ("K", "R")]
    assert {i["type"] for i in loads} == {"Load"}
    assert sorted(next(iter(i["parameters"].values())) for i in loads) == [2.0, 9.0]


def test_a_converted_rate_altered_is_reported_divergent():
    """Negative control: the comparison is not vacuous."""
    declaration = bus_plant()
    conversion = convert(declaration)
    altered = copy.deepcopy(conversion)
    generator = next(t for t in altered.library["library"]["types"] if t["name"] == "Generator")
    source = next(u for u in generator["uses"] if u["fragment"] == "std.continuous_source")
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


def test_a_mixed_volume_is_refused_and_a_single_flow_one_too():
    mixed = document(
        [
            component("S", [c_out("A", 2.0), c_out("B", 1.0)], class_name="Source"),
            component("T", [c_in("A"), c_in("B"), c_out("A"), c_out("B")], class_name="Room")
            | {"capacities": [volume("room", ["A", "B"], 10.0, {"A": 1.0, "B": 1.0}, 0.0)]},
            component("C", [c_in("A", 1.0), c_in("B", 1.0)], class_name="Load"),
        ],
        [link("S", "T", "A"), link("S", "T", "B"), link("T", "C", "A"), link("T", "C", "B")],
    )
    found = refusals(mixed)
    assert "pooled serving (mixed volume)" in found
    # Checked first: a mixed volume is not also refused for its shape.
    assert "multi-flow volume" not in found
    single = document(
        [
            component("S", [c_out("A", 2.0)], class_name="Source"),
            component("T", [c_in("A", 1.0), c_out("A", 1.0)], class_name="Room")
            | {"capacities": [volume("room", ["A"], 10.0, {"A": 1.0}, 0.0)]},
            component("C", [c_in("A", 1.0)], class_name="Load"),
        ],
        [link("S", "T", "A"), link("T", "C", "A")],
    )
    assert refusals(single) == ["output-side volume"]


def test_a_transmitting_volume_is_refused():
    declaration = store_plant(0.5)
    declaration["components"]["T"]["capacities"][0]["transmits"] = True
    assert refusals(declaration) == ["transmitting volume (`transmits`)"]


def test_an_always_on_rule_set_and_a_contested_output_are_refused():
    declaration = machine_plant()
    machine = declaration["components"]["M"]
    machine["rules"] = [
        {"name": "one", "rules": [rule("go", [], {"Water": 1.0}, {"Gas": 1.0})]},
        {"name": "two", "rules": [rule("on", [operand("run")], {"Elec": 1.0}, {"Gas": 1.0}),
                                  rule("off", [], {}, {"Gas": 0.0})]},
    ]
    machine["rules"][0]["apportionment"] = {}
    found = refusals(declaration)
    assert "output produced by two rule sets" in found
    assert "always-on rule set" in found


def test_an_input_side_volume_is_refused():
    declaration = store_plant(0.5)
    declaration["components"]["T"]["capacities"] = [volume("tank", ["H2"], 6.0, {"H2": 1.0}, 0.5, side="in")]
    assert "input-side volume" in refusals(declaration)


def test_two_guarded_rules_are_refused_naming_the_count():
    declaration = machine_plant()
    rules = declaration["components"]["M"]["rules"][0]["rules"]
    rules[1] = rule("half", [operand("run")], {"Water": 1.0, "Elec": 5.0}, {"Gas": 1.0})
    conversion = convert(declaration)
    assert [r.construct for r in conversion.refusals] == ["multi-rule rule set"]
    assert "2 guarded and 0 idle rules" in conversion.refusals[0].detail


def test_a_store_is_refused_whatever_its_fill():
    for fill in (0.0, 0.5, float("inf")):
        assert refusals(store_plant(fill)) == ["output-side volume"]


def _flow(declaration: dict, component: str, cls: str, name: str) -> dict:
    return next(f for f in declaration["components"][component]["flows"] if f["cls"] == cls and f["name"] == name)


def _rule_set(declaration: dict) -> dict:
    return declaration["components"]["M"]["rules"][0]


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
    "delivered rate published on an output": (
        bus_plant,
        lambda d: _flow(d, "S1", "FlowContinuousOut", "E").update(publish_rate="delivered"),
        "published output rate `delivered`",
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
    proved(base())
    change(declaration)
    assert refusals(declaration) == [construct]
