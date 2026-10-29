"""The continuous fragments of ``std``, each characterised against the
muscadet plugin (unit U14 of the component type library plan).

A standard fragment is written in the library format itself and must mean
what the plugin's declaration means, or a plant rebuilt from a library would
quietly differ from the one the plugin builds. Each bench below builds one
small plant twice, from a muscadet declaration expanded by the plugin and
from library documents expanded by the engine, and asks both the same
question: the same events at the same dates, and the same value of every
indicator the plugin generates at every sample instant.

Together the benches apply every continuous fragment: the network
(``continuous_flow_out``, ``continuous_flow_in``, ``surplus_release``),
sources and sinks (``continuous_source``, ``continuous_sink``,
``time_profile``, ``rate_publisher``), volumes (``volume_flow``,
``volume_outlet``, ``volume_stock``, ``volume_total``), rule sets
(``rule_set``, ``rule_consumes``, ``rule_produces``, ``tap``) and controllers
(``aggregate_measure``, ``threshold_compare``, ``threshold_band``).

Oracle-free: both routes run on RAICHU.
"""

from __future__ import annotations

import math

import pytest

import pyraichu

LIBRARY = {"name": "bench", "version": "1.0.0"}

#: A failure mode's transitions are named by the library route after the
#: plan's mode lifecycle (`occ`, `rep`), by the plugin after its own.
MODE_TRANSITIONS = {"occ": "failure", "rep": "repair"}


def use(fragment, **with_):
    return {"fragment": f"std.{fragment}", "with": with_}


def p(name, kind, **extra):
    return {"name": name, "kind": kind, **extra}


def P(name):
    return {"$param": name}


SINUSOID = [p("amplitude", "number"), p("phase_shift", "number"), p("period", "number"),
            p("offset", "number"), p("initial", "number", default=0.0)]


def sinusoid(target):
    return use("time_profile", target=target, amplitude=P("amplitude"), phase_shift=P("phase_shift"),
               period=P("period"), offset=P("offset"), value_min=0.0, initial=P("initial"))


#: The bench types: the shapes of the muscadet declarations below, from std.
TYPES = [
    {"name": "Source", "parameters": [p("flow", "string"), p("rate", "number")],
     "uses": [use("continuous_flow_out", name="${flow}"),
              use("continuous_source", name="${flow}", rate=P("rate"))]},
    {"name": "ProfiledSource",
     "parameters": SINUSOID + [p("flow", "string"), p("shares", "list", items="object")],
     "uses": [sinusoid("${flow}_out_profile"),
              use("rate_publisher", name="${flow}", of="capability"),
              use("continuous_source", name="${flow}", rate=1.0, profiled=True),
              use("continuous_flow_out", name="${flow}", allocation="shares", shares=P("shares"))]},
    {"name": "Sink", "parameters": [p("flow", "string"), p("demand", "number")],
     "uses": [use("continuous_flow_in", name="${flow}", demand=P("demand")),
              use("continuous_sink", name="${flow}", demand=P("demand"))]},
    {"name": "ProfiledSink", "parameters": SINUSOID + [p("flow", "string")],
     "uses": [use("continuous_flow_in", name="${flow}", demand=1.0),
              use("rate_publisher", name="${flow}", of="delivered"),
              sinusoid("${flow}_in_profile"),
              use("continuous_sink", name="${flow}", demand=1.0, profiled=True)]},
    {"name": "Store",
     "parameters": [p("volume", "string"), p("flow", "string"), p("capacity", "number"),
                    p("rate", "number"), p("fill_rate", "number", default=0.0)],
     "uses": [use("continuous_flow_in", name="${flow}", demand=P("rate")),
              use("volume_flow", volume="${volume}", flow="${flow}", capacity=P("capacity"),
                  demand=P("rate"), fill_rate=P("fill_rate")),
              use("volume_outlet", volume="${volume}", flow="${flow}", rate=P("rate"),
                  empty_automaton="${volume}_bounds"),
              use("continuous_flow_out", name="${flow}", held=True, empty_automaton="${volume}_bounds"),
              use("volume_total", volume="${volume}", flows=["${flow}"], capacity=P("capacity"))]},
    {"name": "Machine",
     "parameters": [p("activate", "number"), p("release", "number")],
     "uses": [use("continuous_flow_in", name="Water"),
              use("continuous_flow_in", name="Elec"),
              use("surplus_release", name="Water"),
              use("surplus_release", name="Elec"),
              use("rule_set", name="work", active="on", idle="off", consumes=["Water", "Elec"],
                  produces=["Gas"], threshold="band", threshold_flow="Elec",
                  activate=P("activate"), release=P("release")),
              use("rule_consumes", rule_set="work", active="on", flow="Water", rate=2.0),
              use("rule_consumes", rule_set="work", active="on", flow="Elec", rate=10.0),
              use("rule_produces", rule_set="work", active="on", flow="Gas", rate=3.0, tapped=True),
              use("tap", flow="Gas", outlets=["Gas_leak"]),
              use("continuous_flow_out", name="Gas", available="produced"),
              use("continuous_flow_out", name="Gas_leak", available="produced")],
     "modes": [{"name": "leak", "nature": "failure",
                "effects": [{"attribute": "Gas_leak_pct", "value": 0.25}]}]},
    {"name": "Room",
     "parameters": [p("capacity", "number"), p("rate", "number"), p("o2_init", "number")],
     "uses": [use("continuous_flow_in", name="H2", demand=P("rate")),
              use("continuous_flow_in", name="O2"),
              use("volume_flow", volume="room", flow="H2", capacity=P("capacity"), demand=P("rate"),
                  fill_rate=P("rate")),
              use("volume_flow", volume="room", flow="O2", capacity=P("capacity"),
                  content_init=P("o2_init"), fill_rate=P("rate"), outlet=False),
              use("volume_stock", volume="room", flow="H2", capacity=P("capacity")),
              use("volume_stock", volume="room", flow="O2", capacity=P("capacity"), initial="stocked"),
              use("volume_outlet", volume="room", flow="H2", rate=P("rate"), empty_automaton="room_stock_H2"),
              use("continuous_flow_out", name="H2", held=True, empty_automaton="room_stock_H2"),
              use("volume_total", volume="room", flows=["H2", "O2"], capacity=P("capacity"),
                  initial="partial", ratio_flows=["H2", "O2"])]},
    {"name": "Fan", "parameters": [p("max_rate", "number")],
     "uses": [use("bool_flow_in", name="vent", logic="and"),
              use("continuous_flow_in", name="H2"),
              use("surplus_release", name="H2"),
              use("rule_set", name="duty", active="venting", idle="shut", consumes=["H2"],
                  produces=["H2_vented"], capped=["H2_vented"], conditions=["vent"]),
              use("rule_consumes", rule_set="duty", active="venting", flow="H2", rate=1.0),
              use("rule_produces", rule_set="duty", active="venting", flow="H2_vented", rate=1.0,
                  capped=True, max_rate=P("max_rate")),
              use("continuous_flow_out", name="H2_vented", available="produced")]},
    {"name": "Comparator",
     "parameters": [p("measure", "string"), p("output", "string"),
                    p("operator", "enum", values=["lt", "le", "gt", "ge"]), p("threshold", "number")],
     "uses": [use("aggregate_measure", name="${measure}"),
              use("threshold_compare", name="${output}", input="${measure}", operator=P("operator"),
                  threshold=P("threshold"))]},
    {"name": "Band",
     "parameters": [p("measure", "string"), p("output", "string"),
                    p("direction", "enum", values=["below", "above"]),
                    p("activate", "number"), p("release", "number")],
     "uses": [use("aggregate_measure", name="${measure}"),
              use("threshold_band", name="${output}", input="${measure}", direction=P("direction"),
                  activate=P("activate"), release=P("release"))]},
]


def at_zero(curve):
    angle = (0.0 - curve["phase_shift"]) * (2 * math.pi / curve["period"])
    return {**curve, "initial": max(0.0, curve["amplitude"] * math.sin(angle) + curve["offset"])}


def profile(curve):
    return {"cls": "SinusoidalProfile", **curve, "value_min": 0.0}


def flow_out(name, rate=0.0, **extra):
    return {"name": name, "var_fed_default": rate, **extra}


def flow_in(name, demand=0.0, **extra):
    return {"name": name, "var_demand_default": demand, **extra}


SUN = {"amplitude": 6.0, "phase_shift": 2.0, "period": 12.0, "offset": 5.0}
DAY = {"amplitude": -1.0, "phase_shift": 3.0, "period": 8.0, "offset": 1.5}


def shares(*pairs):
    return [{"to": {"component": c, "port": port}, "value": v} for c, port, v in pairs]


#: name -> (plugin objects, library instances, connections, study modes, t_max)
BENCHES = {
    # Fragments: continuous_flow_out, continuous_source, continuous_flow_in,
    # continuous_sink.
    "source_to_sink": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out("H2O", 5.0)]},
         {"type": "ObjFlow", "name": "K", "flows_continuous_in": [flow_in("H2O", 3.0)]}],
        [{"name": "S", "type": "Source", "parameters": {"flow": "H2O", "rate": 5.0}},
         {"name": "K", "type": "Sink", "parameters": {"flow": "H2O", "demand": 3.0}}],
        [("S.H2O_out", "K.H2O_in")],
        [], 5.0,
    ),
    # Fragments: time_profile, rate_publisher (both kinds), shares.
    "shares_and_profiles": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
            "Elec", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"A": 0.25, "B": 0.75})]},
         {"type": "ObjFlow", "name": "A", "flows_continuous_in": [flow_in("Elec", 2.0)]},
         {"type": "ObjFlow", "name": "B", "flows_continuous_in": [flow_in(
             "Elec", 1.0, profile=profile(DAY), publish_rate="delivered")]}],
        [{"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "Elec",
                         "shares": shares(("A", "Elec_in", 0.25), ("B", "Elec_in", 0.75))}},
         {"name": "A", "type": "Sink", "parameters": {"flow": "Elec", "demand": 2.0}},
         {"name": "B", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "Elec"}}],
        [("S.Elec_out", "A.Elec_in"), ("S.Elec_out", "B.Elec_in")],
        [], 24.0,
    ),
    # Fragments: volume_flow, volume_outlet, volume_total, a held output.
    # The source outpaces the profiled consumer, so the store fills to its
    # bound, then drains while the demand peaks.
    "stock_fills_and_drains": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
            "H2", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"T": 1.0})]},
         {"type": "ObjFlow", "name": "T", "flows_continuous_in": [flow_in("H2", 2.0)],
          "flows_continuous_out": [flow_out("H2", 2.0)],
          "capacities": [{"name": "buffer", "flow": "H2", "capacity": 4.0,
                          "content_init": {"H2": 0.0}, "fill_rate": 2.0}]},
         {"type": "ObjFlow", "name": "C", "flows_continuous_in": [flow_in(
             "H2", 1.0, profile=profile(DAY), publish_rate="delivered")]}],
        [{"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "H2", "shares": shares(("T", "H2_in", 1.0))}},
         {"name": "T", "type": "Store",
          "parameters": {"volume": "buffer", "flow": "H2", "capacity": 4.0, "rate": 2.0, "fill_rate": 2.0}},
         {"name": "C", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "H2"}}],
        [("S.H2_out", "T.H2_in"), ("T.H2_out", "C.H2_in")],
        [], 24.0,
    ),
    # Fragments: rule_set (band), rule_consumes, rule_produces, tap,
    # surplus_release with a supplier volume that runs out (held), and a
    # failure mode opening the tap.
    "rule_set_band_tap_and_release": (
        [{"type": "ObjFlow", "name": "W", "flows_continuous_out": [flow_out("Water", 5.0)]},
         {"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
             "Elec", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
             allocation_shares={"B": 0.5, "M": 0.5})]},
         {"type": "ObjFlow", "name": "B", "flows_continuous_in": [flow_in("Elec", 4.0)],
          "flows_continuous_out": [flow_out("Elec", 4.0)],
          "capacities": [{"name": "store", "flow": "Elec", "capacity": 10.0, "content_init": {"Elec": 0.0}}]},
         {"type": "ObjFlow", "name": "M", "flows_continuous_in": [{"name": "Water"}, {"name": "Elec"}],
          "flows_continuous_out": [{"name": "Gas"}, {"name": "Gas_leak"}],
          "rules": [{"name": "work", "rules": [
              {"name": "on", "cond": [{"name": "Elec", "port": "in", "op": ">=", "value": 6.0, "release": 3.0}],
               "cons": {"Water": 2.0, "Elec": 10.0}, "prod": {"Gas": 3.0}},
              {"name": "off", "prod": {"Gas": 0.0}}]}],
          "failure_modes": [{"name": "leak", "distrib": "delay", "failure": 4.0, "repair": 3.0,
                             "failure_effects": {"Gas": {"tap": 0.25, "to": "Gas_leak"}}}]},
         {"type": "ObjFlow", "name": "G", "flows_continuous_in": [flow_in("Gas", 100.0), flow_in("Gas_leak", 100.0)]}],
        [{"name": "W", "type": "Source", "parameters": {"flow": "Water", "rate": 5.0}},
         {"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "Elec",
                         "shares": shares(("B", "Elec_in", 0.5), ("M", "Elec_in", 0.5))}},
         {"name": "B", "type": "Store",
          "parameters": {"volume": "store", "flow": "Elec", "capacity": 10.0, "rate": 4.0}},
         {"name": "M", "type": "Machine", "parameters": {"activate": 6.0, "release": 3.0}},
         {"name": "G", "type": "TwoSinks"}],
        [("W.Water_out", "M.Water_in"), ("B.Elec_out", "M.Elec_in"), ("S.Elec_out", "B.Elec_in"),
         ("S.Elec_out", "M.Elec_in"), ("M.Gas_out", "G.Gas_in"), ("M.Gas_leak_out", "G.Gas_leak_in")],
        [{"mode": "leak", "instances": ["M"], "occurrence": {"distrib": "delay", "time": 4.0},
          "repair": {"distrib": "delay", "time": 3.0}}],
        24.0,
    ),
    # A rule set whose band sees the profile alone: the capability crosses
    # its thresholds continuously, so the switching values can be read.
    "band_on_a_profile": (
        [{"type": "ObjFlow", "name": "W", "flows_continuous_out": [flow_out("Water", 5.0)]},
         {"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
             "Elec", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
             allocation_shares={"M": 1.0})]},
         {"type": "ObjFlow", "name": "M", "flows_continuous_in": [{"name": "Water"}, {"name": "Elec"}],
          "flows_continuous_out": [{"name": "Gas"}, {"name": "Gas_leak"}],
          "rules": [{"name": "work", "rules": [
              {"name": "on", "cond": [{"name": "Elec", "port": "in", "op": ">=", "value": 6.0, "release": 3.0}],
               "cons": {"Water": 2.0, "Elec": 10.0}, "prod": {"Gas": 3.0}},
              {"name": "off", "prod": {"Gas": 0.0}}]}],
          "failure_modes": [{"name": "leak", "distrib": "delay", "failure": 100.0, "repair": 1.0,
                             "failure_effects": {"Gas": {"tap": 0.25, "to": "Gas_leak"}}}]},
         {"type": "ObjFlow", "name": "G", "flows_continuous_in": [flow_in("Gas", 100.0), flow_in("Gas_leak", 100.0)]}],
        [{"name": "W", "type": "Source", "parameters": {"flow": "Water", "rate": 5.0}},
         {"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "Elec", "shares": shares(("M", "Elec_in", 1.0))}},
         {"name": "M", "type": "Machine", "parameters": {"activate": 6.0, "release": 3.0}},
         {"name": "G", "type": "TwoSinks"}],
        [("W.Water_out", "M.Water_in"), ("S.Elec_out", "M.Elec_in"),
         ("M.Gas_out", "G.Gas_in"), ("M.Gas_leak_out", "G.Gas_leak_in")],
        [{"mode": "leak", "instances": ["M"], "occurrence": {"distrib": "delay", "time": 100.0},
          "repair": {"distrib": "delay", "time": 1.0}}],
        24.0,
    ),
    # Fragments: volume_stock, a multi-flow volume_total with ratios,
    # threshold_band, aggregate_measure, bool_flow_in, a rule set with no
    # threshold and a capped output.
    "room_band_and_fan": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out("H2", 2.0)]},
         {"type": "ObjFlow", "name": "R", "flows_continuous_in": [flow_in("H2", 10.0), {"name": "O2"}],
          "flows_continuous_out": [flow_out("H2", 10.0)],
          "capacities": [{"name": "room", "flows": ["H2", "O2"], "capacity": 100.0,
                          "content_init": {"H2": 0.0, "O2": 80.0}, "fill_rate": 10.0}]},
         {"type": "ObjCtrl", "name": "AP", "controls_in": [{"name": "room", "kind": "ratio", "flows": ["H2"]}],
          "controls_out": [{"name": "vent", "kind": "bool", "emit": {
              "op": "band", "input": "room", "direction": "above", "activate": 0.05, "release": 0.02}}]},
         {"type": "ObjFlow", "name": "F", "flows_in": [{"name": "vent", "logic": "and"}],
          "flows_continuous_in": [{"name": "H2"}],
          "flows_continuous_out": [{"name": "H2_vented", "max_rate": 6.0}],
          "rules": [{"name": "duty", "rules": [
              {"name": "venting", "cond": [{"name": "vent", "port": "in"}],
               "cons": {"H2": 1.0}, "prod": {"H2_vented": 1.0}},
              {"name": "shut", "prod": {"H2_vented": 0.0}}]}]},
         {"type": "ObjFlow", "name": "A", "flows_continuous_in": [flow_in("H2_vented", 1000.0)]}],
        [{"name": "S", "type": "Source", "parameters": {"flow": "H2", "rate": 2.0}},
         {"name": "R", "type": "Room", "parameters": {"capacity": 100.0, "rate": 10.0, "o2_init": 80.0}},
         {"name": "AP", "type": "Band", "parameters": {
             "measure": "room_ratio_H2", "output": "vent", "direction": "above",
             "activate": 0.05, "release": 0.02}},
         {"name": "F", "type": "Fan", "parameters": {"max_rate": 6.0}},
         {"name": "A", "type": "Sink", "parameters": {"flow": "H2_vented", "demand": 1000.0}}],
        [("S.H2_out", "R.H2_in"), ("R.H2_out", "F.H2_in"), ("R.room_ratio_H2_out", "AP.room_ratio_H2_in"),
         ("AP.vent_out", "F.vent_in"), ("F.H2_vented_out", "A.H2_vented_in")],
        [], 24.0,
    ),
    # Fragments: threshold_compare on a published rate.
    "compare_a_rate": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
            "Elec", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"K": 1.0})]},
         {"type": "ObjFlow", "name": "K", "flows_continuous_in": [flow_in("Elec", 20.0)]},
         {"type": "ObjCtrl", "name": "AP", "controls_in": [{"name": "Elec", "kind": "rate"}],
          "controls_out": [{"name": "low", "kind": "bool", "emit": {
              "op": "compare", "input": "Elec", "operator": "<", "threshold": 4.0}}]}],
        [{"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "Elec", "shares": shares(("K", "Elec_in", 1.0))}},
         {"name": "K", "type": "Sink", "parameters": {"flow": "Elec", "demand": 20.0}},
         {"name": "AP", "type": "Comparator", "parameters": {
             "measure": "Elec_rate", "output": "low", "operator": "lt", "threshold": 4.0}}],
        [("S.Elec_out", "K.Elec_in"), ("S.Elec_rate_out", "AP.Elec_rate_in")],
        [], 24.0,
    ),
}

TYPES.append({
    "name": "TwoSinks",
    "uses": [use("continuous_flow_in", name="Gas", demand=100.0),
             use("continuous_flow_in", name="Gas_leak", demand=100.0),
             use("continuous_sink", name="Gas", demand=100.0),
             use("continuous_sink", name="Gas_leak", demand=100.0)],
})


def plugin_model(name, objects, connections):
    def end(written):
        component, port = written.split(".", 1)
        return {"component": component, "port": port}

    document = {
        "name": name,
        "generated_indicators": True,
        "plugins": {"muscadet": {"objects": objects}},
        "connections": [{"from": end(f), "to": end(t)} for f, t in connections],
    }
    return pyraichu.expand_model(document)


def library_model(name, instances, connections, modes, indicators):
    library = {"raichu_library": {"format": 1},
               "library": {**LIBRARY, "imports": ["std"], "types": TYPES}}
    system = {"raichu_system": {"format": 1},
              "system": {"name": name, "library": LIBRARY, "instances": instances,
                         "connections": [{"from": f, "to": t} for f, t in connections],
                         "indicators": indicators}}
    study = None
    if modes:
        study = {"raichu_study": {"format": 1},
                 "study": {"name": "modes", "system": name, "modes": modes}}
    return pyraichu.expand_documents(system, [library], study)["model"]


def run(model, t_max):
    instants = [t_max * i / 48 for i in range(49)]
    result = pyraichu.simulate(pyraichu.load_model(model), t_max=t_max, samples=instants)
    events = []
    for event in result.events:
        component, automaton, transition = event.transition.split(".")
        events.append((event.time, f"{component}.{automaton}.{MODE_TRANSITIONS.get(transition, transition)}"))
    samples = {name: [v for _, v in series] for name, series in result.samples.items()}
    return events, samples


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_the_fragment_agrees_with_its_plugin_counterpart(bench):
    objects, instances, connections, modes, t_max = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    library = library_model(bench, instances, connections, modes, plugin["indicators"])
    # The same guard on both: without it the library-built network would
    # integrate the "unbounded" magnitude where the plugin's refuses it.
    guards = [pyraichu.model_body(m).get("unbounded_rate") for m in (library, plugin)]
    assert guards[0] == guards[1] is not None
    plugin_events, plugin_samples = run(plugin, t_max)
    library_events, library_samples = run(library, t_max)
    assert library_events == plugin_events
    assert library_samples.keys() == plugin_samples.keys()
    for name, expected in plugin_samples.items():
        assert library_samples[name] == expected, name


#: The fragments of `std` that are not continuous, left out of the guard below.
BOOLEAN_FRAGMENTS = {
    "bool_flow_in",
    "bool_flow_out",
    "bool_source_out",
    "bool_flow_out_tempo",
    "bool_flow_out_trigger",
    "bool_not",
    "logic_gate",
    "state_out",
}


def test_the_benches_exercise_every_continuous_fragment():
    """No fragment of the continuous part of std goes uncharacterised."""
    std = pyraichu.standard_library()["library"]["fragments"]
    continuous = {f["name"] for f in std} - BOOLEAN_FRAGMENTS

    def used(type_name):
        ty = next(t for t in TYPES if t["name"] == type_name)
        return {u["fragment"].removeprefix("std.") for u in ty["uses"]}

    applied = set()
    for _, instances, _, _, _ in BENCHES.values():
        for instance in instances:
            applied |= used(instance["type"])
    assert continuous <= applied, continuous - applied


def bench_run(bench):
    objects, instances, connections, modes, t_max = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    library = library_model(bench, instances, connections, modes, plugin["indicators"])
    return library, run(library, t_max)


def test_a_stock_fills_to_its_bound_and_drains():
    """The stock bench is not a still life: the buffer reaches its upper
    bound and later runs dry, and the dates agree with the plugin's (the
    agreement test above)."""
    _, (events, _) = bench_run("stock_fills_and_drains")
    fired = [name for _, name in events]
    assert "T.buffer_bounds.buffer_reach_full" in fired
    assert "T.buffer_bounds.buffer_leave_full" in fired
    assert "T.buffer_bounds.buffer_reach_empty" in fired


def test_a_band_switches_at_activate_and_back_at_release():
    """The rule starts when the electricity capability reaches `activate`
    and stops when it falls under `release`, both crossings located."""
    library, (events, _) = bench_run("band_on_a_profile")
    starts = [t for t, name in events if name == "M.work_mode.work_off_to_on"]
    stops = [t for t, name in events if name == "M.work_mode.work_on_to_off"]
    assert starts and stops
    result = pyraichu.simulate(pyraichu.load_model(library), t_max=24.0, samples=sorted(starts + stops))
    capability = dict(result.samples["M_Elec_capability_in"])
    for t in starts:
        assert capability[t] == pytest.approx(6.0, abs=1e-6), t
    for t in stops:
        assert capability[t] == pytest.approx(3.0, abs=1e-6), t
