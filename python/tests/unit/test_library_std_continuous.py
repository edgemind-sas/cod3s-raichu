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
``volume_outlet``, ``volume_stock``, ``volume_total``), a pass-through
(``pass_through``), rule sets
(``rule_set``, ``rule_consumes``, ``rule_produces``, ``tap``), an output
two rule sets produce (``shared_output``, ``rule_cascade``) and controllers
(``aggregate_measure``, ``threshold_compare``, ``threshold_band``,
``republish``).

Oracle-free: both routes run on RAICHU.
"""

from __future__ import annotations

import json
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
    {"name": "UnboundedStore",
     "parameters": [p("volume", "string"), p("flow", "string"), p("capacity", "number"),
                    p("rate", "number")],
     "uses": [use("continuous_flow_in", name="${flow}", demand=P("rate")),
              use("volume_flow", volume="${volume}", flow="${flow}", capacity=P("capacity"),
                  demand=P("rate"), fill_unbounded=True),
              use("volume_outlet", volume="${volume}", flow="${flow}", rate=P("rate"),
                  empty_automaton="${volume}_bounds"),
              use("continuous_flow_out", name="${flow}", held=True, empty_automaton="${volume}_bounds"),
              use("volume_total", volume="${volume}", flows=["${flow}"], capacity=P("capacity"))]},
    {"name": "Bus", "parameters": [p("flow", "string")],
     "uses": [use("continuous_flow_in", name="${flow}"),
              use("pass_through", name="${flow}"),
              use("continuous_flow_out", name="${flow}", available="produced")]},
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
    # Fragment: pass_through. Two sources feed a bus the sink draws from
    # through it. A rival consumer on the first source takes part of what
    # it could give, so the bus receives less than its suppliers could
    # offer and hands on what arrived, not what could have. A second bus
    # nothing reads asks for nothing.
    "pass_through_bus": (
        [{"type": "ObjFlow", "name": "S1", "flows_continuous_out": [flow_out("E", 3.0)]},
         {"type": "ObjFlow", "name": "S2", "flows_continuous_out": [flow_out(
             "E", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
             allocation_shares={"BUS": 0.5, "IDLE": 0.5})]},
         {"type": "ObjFlow", "name": "BUS", "flows_continuous_in": [{"name": "E"}],
          "flows_continuous_out": [{"name": "E"}]},
         {"type": "ObjFlow", "name": "IDLE", "flows_continuous_in": [{"name": "E"}],
          "flows_continuous_out": [{"name": "E"}]},
         {"type": "ObjFlow", "name": "K", "flows_continuous_in": [flow_in("E", 9.0)]},
         {"type": "ObjFlow", "name": "R", "flows_continuous_in": [flow_in("E", 2.0)]}],
        [{"name": "S1", "type": "Source", "parameters": {"flow": "E", "rate": 3.0}},
         {"name": "S2", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "E",
                         "shares": shares(("BUS", "E_in", 0.5), ("IDLE", "E_in", 0.5))}},
         {"name": "BUS", "type": "Bus", "parameters": {"flow": "E"}},
         {"name": "IDLE", "type": "Bus", "parameters": {"flow": "E"}},
         {"name": "K", "type": "Sink", "parameters": {"flow": "E", "demand": 9.0}},
         {"name": "R", "type": "Sink", "parameters": {"flow": "E", "demand": 2.0}}],
        [("S1.E_out", "BUS.E_in"), ("S1.E_out", "R.E_in"), ("S2.E_out", "BUS.E_in"),
         ("S2.E_out", "IDLE.E_in"), ("BUS.E_out", "K.E_in")],
        [], 24.0,
    ),
    # Option: volume_flow fill_unbounded. The store claims whatever its
    # supplier has while it is not full, so it fills at the supply and the
    # consumer beside it gets what the split leaves.
    "unbounded_fill": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
            "H2", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"T": 0.5, "D": 0.5})]},
         {"type": "ObjFlow", "name": "T", "flows_continuous_in": [flow_in("H2", 1.0)],
          "flows_continuous_out": [flow_out("H2", 1.0)],
          "capacities": [{"name": "tank", "flow": "H2", "capacity": 30.0,
                          "content_init": {"H2": 0.0}, "fill_rate": None}]},
         {"type": "ObjFlow", "name": "D", "flows_continuous_in": [flow_in("H2", 2.0)]},
         {"type": "ObjFlow", "name": "C", "flows_continuous_in": [flow_in("H2", 3.0)]}],
        [{"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "H2",
                         "shares": shares(("T", "H2_in", 0.5), ("D", "H2_in", 0.5))}},
         {"name": "T", "type": "UnboundedStore",
          "parameters": {"volume": "tank", "flow": "H2", "capacity": 30.0, "rate": 1.0}},
         {"name": "D", "type": "Sink", "parameters": {"flow": "H2", "demand": 2.0}},
         {"name": "C", "type": "Sink", "parameters": {"flow": "H2", "demand": 3.0}}],
        [("S.H2_out", "T.H2_in"), ("S.H2_out", "D.H2_in"), ("T.H2_out", "C.H2_in")],
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
    # Option: rule_set always_on (with rule_consumes and rule_produces).
    # A rule declared without a condition always applies: no mode
    # automaton, its scale follows the profiled supply and the demand.
    "always_on_rule_set": (
        [{"type": "ObjFlow", "name": "W", "flows_continuous_out": [flow_out(
            "Water", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"M": 1.0})]},
         {"type": "ObjFlow", "name": "M", "flows_continuous_in": [{"name": "Water"}],
          "flows_continuous_out": [{"name": "Gas"}],
          "rules": [{"name": "work", "rules": [
              {"name": "go", "cond": [], "cons": {"Water": 2.0}, "prod": {"Gas": 3.0}}]}]},
         {"type": "ObjFlow", "name": "G", "flows_continuous_in": [flow_in(
             "Gas", 1.0, profile=profile(DAY), publish_rate="delivered")]}],
        [{"name": "W", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "Water", "shares": shares(("M", "Water_in", 1.0))}},
         {"name": "M", "type": "Steady"},
         {"name": "G", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "Gas"}}],
        [("W.Water_out", "M.Water_in"), ("M.Gas_out", "G.Gas_in")],
        [], 24.0,
    ),
    # Option: volume_flow side `in` (with rule_set from_volume and
    # rule_consumes from_volume), multi-flow. Two rules always on draw one
    # flow each from a volume upstream of them that fills without bound:
    # each draws what it needs while its flow is stocked, what arrives once
    # it has run out, and the volume asks for whatever it can get until it
    # is full.
    "input_side_volume": (
        [{"type": "ObjFlow", "name": "SA", "flows_continuous_out": [flow_out(
            "A", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"M": 1.0})]},
         {"type": "ObjFlow", "name": "SB", "flows_continuous_out": [flow_out("B", 0.6)]},
         {"type": "ObjFlow", "name": "M", "flows_continuous_in": [{"name": "A"}, {"name": "B"}],
          "flows_continuous_out": [{"name": "A_vent"}, {"name": "B_vent"}],
          "capacities": [{"name": "membrane", "flows": ["A", "B"], "capacity": 4.0, "side": "in",
                          "content_init": {"A": 0.0, "B": 0.0}, "transmits": True, "fill_rate": None}],
          "rules": [{"name": "ra", "rules": [{"name": "go", "cond": [], "cons": {"A": 1.0}, "prod": {"A_vent": 1.0}}]},
                    {"name": "rb", "rules": [{"name": "go", "cond": [], "cons": {"B": 1.0}, "prod": {"B_vent": 1.0}}]}]},
         {"type": "ObjFlow", "name": "KA", "flows_continuous_in": [flow_in(
             "A_vent", 1.0, profile=profile(DAY), publish_rate="delivered")]},
         {"type": "ObjFlow", "name": "KB", "flows_continuous_in": [flow_in("B_vent", 0.5)]}],
        [{"name": "SA", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "A", "shares": shares(("M", "A_in", 1.0))}},
         {"name": "SB", "type": "Source", "parameters": {"flow": "B", "rate": 0.6}},
         {"name": "M", "type": "Membrane", "parameters": {"capacity": 4.0}},
         {"name": "KA", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "A_vent"}},
         {"name": "KB", "type": "Sink", "parameters": {"flow": "B_vent", "demand": 0.5}}],
        [("SA.A_out", "M.A_in"), ("SB.B_out", "M.B_in"), ("M.A_vent_out", "KA.A_vent_in"),
         ("M.B_vent_out", "KB.B_vent_in")],
        [], 24.0,
    ),
    # The same option on a single-flow volume with a finite fill, starting
    # part full: its bounds automaton says when the flow has run out.
    "input_side_volume_single": (
        [{"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
            "Fuel", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"F": 1.0})]},
         {"type": "ObjFlow", "name": "F", "flows_continuous_in": [{"name": "Fuel"}],
          "flows_continuous_out": [{"name": "Heat"}],
          "capacities": [{"name": "hopper", "flow": "Fuel", "capacity": 1.5, "side": "in",
                          "content_init": {"Fuel": 1.0}, "transmits": True, "fill_rate": 0.5}],
          "rules": [{"name": "burn", "rules": [{"name": "go", "cond": [], "cons": {"Fuel": 2.0},
                                                "prod": {"Heat": 3.0}}]}]},
         {"type": "ObjFlow", "name": "H", "flows_continuous_in": [flow_in(
             "Heat", 1.0, profile=profile(DAY), publish_rate="delivered")]}],
        [{"name": "S", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "Fuel", "shares": shares(("F", "Fuel_in", 1.0))}},
         {"name": "F", "type": "Feeder", "parameters": {"capacity": 1.5, "content": 1.0, "fill_rate": 0.5}},
         {"name": "H", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "Heat"}}],
        [("S.Fuel_out", "F.Fuel_in"), ("F.Heat_out", "H.Heat_in")],
        [], 24.0,
    ),
    # Fragments: shared_output, rule_cascade, rule_produces shared and
    # rule_set cascaded. Two rule sets produce one exhaust, served in
    # declaration order: the first, held by a band on its input, is asked
    # the whole demand, the second only what the first leaves unserved.
    # The first's supply follows the sun, so it serves the whole demand,
    # part of it, or nothing once its band has released.
    "cascade_two_rule_sets": (
        [{"type": "ObjFlow", "name": "SA", "flows_continuous_out": [flow_out(
            "A", 1.0, profile=profile(SUN), publish_rate="capability", allocation="shares",
            allocation_shares={"V": 1.0})]},
         {"type": "ObjFlow", "name": "SB", "flows_continuous_out": [flow_out("B", 3.0)]},
         {"type": "ObjFlow", "name": "V", "flows_continuous_in": [{"name": "A"}, {"name": "B"}],
          "flows_continuous_out": [{"name": "X"}, {"name": "Y"}],
          "rules": [{"name": "first", "rules": [
              {"name": "on", "cond": [{"name": "A", "port": "in", "op": ">=", "value": 2.0, "release": 1.0}],
               "cons": {"A": 4.0}, "prod": {"X": 1.0}},
              {"name": "off", "prod": {"X": 0.0}}]},
                    {"name": "second", "rules": [
              {"name": "go", "cond": [], "cons": {"B": 1.0}, "prod": {"X": 1.0, "Y": 0.5}}]}]},
         {"type": "ObjFlow", "name": "KX", "flows_continuous_in": [flow_in(
             "X", 1.0, profile=profile(DAY), publish_rate="delivered")]},
         {"type": "ObjFlow", "name": "KY", "flows_continuous_in": [flow_in("Y", 10.0)]}],
        [{"name": "SA", "type": "ProfiledSource",
          "parameters": {**at_zero(SUN), "flow": "A", "shares": shares(("V", "A_in", 1.0))}},
         {"name": "SB", "type": "Source", "parameters": {"flow": "B", "rate": 3.0}},
         {"name": "V", "type": "Ventilator"},
         {"name": "KX", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "X"}},
         {"name": "KY", "type": "Sink", "parameters": {"flow": "Y", "demand": 10.0}}],
        [("SA.A_out", "V.A_in"), ("SB.B_out", "V.B_in"), ("V.X_out", "KX.X_in"), ("V.Y_out", "KY.Y_in")],
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
    "name": "Steady",
    "uses": [use("continuous_flow_in", name="Water"),
             use("rule_set", name="work", active="go", idle="none", consumes=["Water"], produces=["Gas"], always_on=True),
             use("rule_consumes", rule_set="work", active="go", flow="Water", rate=2.0, always_on=True),
             use("rule_produces", rule_set="work", active="go", flow="Gas", rate=3.0, always_on=True),
             use("continuous_flow_out", name="Gas", available="produced")],
})
TYPES.append({
    "name": "Membrane",
    "parameters": [p("capacity", "number")],
    "uses": [use("continuous_flow_in", name="A"),
             use("continuous_flow_in", name="B"),
             use("volume_flow", volume="membrane", flow="A", capacity=P("capacity"), fill_unbounded=True,
                 side="in", empty_automaton="membrane_stock_A"),
             use("volume_flow", volume="membrane", flow="B", capacity=P("capacity"), fill_unbounded=True,
                 side="in", empty_automaton="membrane_stock_B"),
             use("volume_total", volume="membrane", flows=["A", "B"], capacity=P("capacity"),
                 ratio_flows=["A", "B"]),
             use("volume_stock", volume="membrane", flow="A", capacity=P("capacity")),
             use("volume_stock", volume="membrane", flow="B", capacity=P("capacity")),
             use("rule_set", name="ra", active="go", idle="none", consumes=[], from_volume=["A"], produces=["A_vent"],
                 always_on=True),
             use("rule_consumes", rule_set="ra", active="go", flow="A", rate=1.0, always_on=True, from_volume=True),
             use("rule_produces", rule_set="ra", active="go", flow="A_vent", rate=1.0, always_on=True),
             use("rule_set", name="rb", active="go", idle="none", consumes=[], from_volume=["B"], produces=["B_vent"],
                 always_on=True),
             use("rule_consumes", rule_set="rb", active="go", flow="B", rate=1.0, always_on=True, from_volume=True),
             use("rule_produces", rule_set="rb", active="go", flow="B_vent", rate=1.0, always_on=True),
             use("continuous_flow_out", name="A_vent", available="produced"),
             use("continuous_flow_out", name="B_vent", available="produced")],
})
TYPES.append({
    "name": "Feeder",
    "parameters": [p("capacity", "number"), p("content", "number"), p("fill_rate", "number")],
    "uses": [use("continuous_flow_in", name="Fuel"),
             use("volume_flow", volume="hopper", flow="Fuel", capacity=P("capacity"), content_init=P("content"),
                 fill_rate=P("fill_rate"), side="in", empty_automaton="hopper_bounds"),
             use("volume_total", volume="hopper", flows=["Fuel"], capacity=P("capacity"), initial="partial"),
             use("rule_set", name="burn", active="go", idle="none", consumes=[], from_volume=["Fuel"], produces=["Heat"],
                 always_on=True),
             use("rule_consumes", rule_set="burn", active="go", flow="Fuel", rate=2.0, always_on=True,
                 from_volume=True),
             use("rule_produces", rule_set="burn", active="go", flow="Heat", rate=3.0, always_on=True),
             use("continuous_flow_out", name="Heat", available="produced")],
})
def ventilator(name, order):
    """Two rule sets producing the exhaust X, cascaded in `order`: the
    first named is asked the whole demand, the second what it leaves."""
    head, tail = order
    produces = {"first": [], "second": ["Y"]}
    produces[head] = produces[head] + ["X"]
    cascaded = {head: [], tail: ["X"]}
    return {
        "name": name,
        "uses": [use("continuous_flow_in", name="A"),
                 use("continuous_flow_in", name="B"),
                 use("rule_set", name="first", active="on", idle="off", consumes=["A"],
                     produces=produces["first"], cascaded=cascaded["first"], threshold="band",
                     threshold_flow="A", activate=2.0, release=1.0),
                 use("rule_consumes", rule_set="first", active="on", flow="A", rate=4.0),
                 use("rule_produces", rule_set="first", active="on", flow="X", rate=1.0, shared=True),
                 use("rule_set", name="second", active="go", idle="none", consumes=["B"],
                     produces=produces["second"], cascaded=cascaded["second"], always_on=True),
                 use("rule_consumes", rule_set="second", active="go", flow="B", rate=1.0, always_on=True),
                 use("rule_produces", rule_set="second", active="go", flow="X", rate=1.0, always_on=True,
                     shared=True),
                 use("rule_produces", rule_set="second", active="go", flow="Y", rate=0.5, always_on=True),
                 use("rule_cascade", rule_set=tail, flow="X", before=[head]),
                 use("shared_output", flow="X", producers=["first", "second"]),
                 use("continuous_flow_out", name="X", available="produced"),
                 use("continuous_flow_out", name="Y", available="produced")],
    }


TYPES.append(ventilator("Ventilator", ("first", "second")))
TYPES.append(ventilator("VentilatorReversed", ("second", "first")))
TYPES.append({
    "name": "BusByCapability",
    "parameters": [p("flow", "string")],
    "uses": [use("continuous_flow_in", name="${flow}"),
             use("pass_through", name="${flow}"),
             use("continuous_flow_out", name="${flow}")],
})
TYPES.append({
    "name": "TwoSinks",
    "uses": [use("continuous_flow_in", name="Gas", demand=100.0),
             use("continuous_flow_in", name="Gas_leak", demand=100.0),
             use("continuous_sink", name="Gas", demand=100.0),
             use("continuous_sink", name="Gas_leak", demand=100.0)],
})


TYPES.append({
    "name": "Instrument",
    "parameters": [p("measure", "string"), p("output", "string"), p("gain", "number", default=1.0)],
    "uses": [use("aggregate_measure", name="${measure}"),
             use("republish", name="${output}", input="${measure}", gain=P("gain"))],
})
TYPES.append({
    "name": "Voter",
    "parameters": [p("measure", "string"), p("output", "string"),
                   p("direction", "enum", values=["below", "above"]),
                   p("activate", "number"), p("release", "number"), p("initial", "number", default=0.0)],
    "uses": [use("aggregate_measure", name="${measure}", agg="median", initial=P("initial")),
             use("threshold_band", name="${output}", input="${measure}", direction=P("direction"),
                 activate=P("activate"), release=P("release"))],
})
TYPES.append({
    "name": "RateMeter",
    "parameters": [p("measure", "string"), p("output", "string")],
    "uses": [use("aggregate_measure", name="${measure}", agg="mean"),
             use("republish", name="${output}", input="${measure}")],
})
TYPES.append({
    "name": "DeliveringSource",
    "parameters": SINUSOID + [p("flow", "string"), p("shares", "list", items="object")],
    "uses": [sinusoid("${flow}_out_profile"),
             use("rate_publisher", name="${flow}", of="delivered", side="out"),
             use("continuous_source", name="${flow}", rate=1.0, profiled=True),
             use("continuous_flow_out", name="${flow}", allocation="shares", shares=P("shares"))]},
)


def four_sources():
    """Four profiled sources a quarter period apart, each serving a sink of
    its own, and the capability each publishes."""
    plugin, library, links = [], [], []
    for index in range(4):
        curve = {**SUN, "phase_shift": 3.0 * index}
        source, sink = f"S{index}", f"K{index}"
        plugin += [{"type": "ObjFlow", "name": source, "flows_continuous_out": [flow_out(
                       "Elec", 1.0, profile=profile(curve), publish_rate="capability",
                       allocation="shares", allocation_shares={sink: 1.0})]},
                   {"type": "ObjFlow", "name": sink, "flows_continuous_in": [flow_in("Elec", 20.0)]}]
        library += [{"name": source, "type": "ProfiledSource", "parameters": {
                        **at_zero(curve), "flow": "Elec", "shares": shares((sink, "Elec_in", 1.0))}},
                    {"name": sink, "type": "Sink", "parameters": {"flow": "Elec", "demand": 20.0}}]
        links.append((f"{source}.Elec_out", f"{sink}.Elec_in"))
    return plugin, library, links


def median_of_instruments():
    """Four instruments republishing a rate, the last with a gain of zero (a
    failed instrument publishing nothing), and a voter thresholding the
    median of the four with a band."""
    plugin, library, links = four_sources()
    for index in range(4):
        gain = 0.0 if index == 3 else 1.0
        name = f"CT{index}"
        plugin.append({"type": "ObjCtrl", "name": name, "controls_in": [{"name": "Elec", "kind": "rate"}],
                       "controls_out": [{"name": "reading", "kind": "value", "emit": {
                           "op": "republish", "input": "Elec", "gain": gain}}]})
        library.append({"name": name, "type": "Instrument", "parameters": {
            "measure": "Elec_rate", "output": "reading", "gain": gain}})
        links += [(f"S{index}.Elec_rate_out", f"{name}.Elec_rate_in"),
                  (f"{name}.reading_level_out", "AP.reading_level_in")]
    plugin.append({"type": "ObjCtrl", "name": "AP",
                   "controls_in": [{"name": "reading", "kind": "level", "aggregate": "median"}],
                   "controls_out": [{"name": "high", "kind": "bool", "emit": {
                       "op": "band", "input": "reading", "direction": "above",
                       "activate": 4.5, "release": 1.5}}]})
    library.append({"name": "AP", "type": "Voter", "parameters": {
        "measure": "reading_level", "output": "high", "direction": "above",
        "activate": 4.5, "release": 1.5}})
    return plugin, library, links, [], 24.0


def level_sensor_and_delivered_rate():
    """A store fed by a profiled source that publishes what it delivers: an
    instrument republishes the store's level, a voter switches a band on it,
    and a meter republishes the delivered rate it reads."""
    plugin = [
        {"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out(
            "H2", 1.0, profile=profile(SUN), publish_rate="delivered", allocation="shares",
            allocation_shares={"T": 1.0})]},
        {"type": "ObjFlow", "name": "T", "flows_continuous_in": [flow_in("H2", 4.0)],
         "flows_continuous_out": [flow_out("H2", 4.0)],
         "capacities": [{"name": "tank", "flow": "H2", "capacity": 30.0, "content_init": {"H2": 0.0},
                         "fill_rate": 2.0}]},
        {"type": "ObjFlow", "name": "K", "flows_continuous_in": [flow_in(
            "H2", 1.0, profile=profile(DAY), publish_rate="delivered")]},
        {"type": "ObjCtrl", "name": "CT", "controls_in": [{"name": "tank", "kind": "level"}],
         "controls_out": [{"name": "tank_reading", "kind": "value", "emit": {
             "op": "republish", "input": "tank", "gain": 1.0}}]},
        {"type": "ObjCtrl", "name": "AP",
         "controls_in": [{"name": "tank_reading", "kind": "level", "aggregate": "median"}],
         "controls_out": [{"name": "low", "kind": "bool", "emit": {
             "op": "band", "input": "tank_reading", "direction": "below", "activate": 5.0, "release": 12.0}}]},
        {"type": "ObjCtrl", "name": "M", "controls_in": [{"name": "H2", "kind": "rate", "aggregate": "mean"}],
         "controls_out": [{"name": "supply", "kind": "value", "emit": {
             "op": "republish", "input": "H2", "gain": 1.0}}]},
    ]
    library = [
        {"name": "S", "type": "DeliveringSource",
         "parameters": {**at_zero(SUN), "flow": "H2", "shares": shares(("T", "H2_in", 1.0))}},
        {"name": "T", "type": "Store", "parameters": {
            "volume": "tank", "flow": "H2", "capacity": 30.0, "rate": 4.0, "fill_rate": 2.0}},
        {"name": "K", "type": "ProfiledSink", "parameters": {**at_zero(DAY), "flow": "H2"}},
        {"name": "CT", "type": "Instrument", "parameters": {"measure": "tank_level", "output": "tank_reading"}},
        {"name": "AP", "type": "Voter", "parameters": {
            "measure": "tank_reading_level", "output": "low", "direction": "below",
            "activate": 5.0, "release": 12.0}},
        {"name": "M", "type": "RateMeter", "parameters": {"measure": "H2_rate", "output": "supply"}},
    ]
    links = [("S.H2_out", "T.H2_in"), ("T.H2_out", "K.H2_in"), ("T.tank_level_out", "CT.tank_level_in"),
             ("CT.tank_reading_level_out", "AP.tank_reading_level_in"), ("S.H2_rate_out", "M.H2_rate_in")]
    return plugin, library, links, [], 24.0


# Fragments: aggregate_measure (median), republish, threshold_band.
BENCHES["median_of_instruments"] = median_of_instruments()
# Fragments: republish of a volume level, threshold_band `below`,
# aggregate_measure (mean), rate_publisher delivered by an output.
BENCHES["level_sensor_and_delivered_rate"] = level_sensor_and_delivered_rate()


# --- pooled serving out of a mixed volume -------------------------------------


def mu(fragment, **with_):
    return {"fragment": f"muscadet.{fragment}", "with": with_}


def mixed_room_uses(contents, capacity, rate):
    """A volume holding several flows downstream of the transfer, each
    served to its own output out of the pooled draw."""
    total = sum(contents.values())
    uses = [mu("volume_total", volume="room", flows=list(contents), capacity=capacity,
               initial="full" if total >= capacity else "partial", ratio_flows=list(contents))]
    for flow, content in contents.items():
        stock = f"room_stock_{flow}"
        uses += [mu("continuous_flow_in", name=flow),
                 mu("volume_flow", volume="room", flow=flow, capacity=capacity, content_init=content,
                    fill_rate=0.0),
                 mu("volume_stock", volume="room", flow=flow, capacity=capacity,
                    initial="stocked" if content > 0 else "empty"),
                 mu("volume_outlet", volume="room", flow=flow, rate=rate, empty_automaton=stock),
                 mu("volume_pooled_flow", volume="room", flow=flow),
                 mu("continuous_flow_out", name=flow, available="pooled", held=True, empty_automaton=stock)]
    return uses + [mu("volume_pooled", volume="room", flows=list(contents))]


def mixed_room_object(contents, capacity, rate):
    return {"type": "ObjFlow", "name": "ROOM",
            "flows_continuous_in": [{"name": flow} for flow in contents],
            "flows_continuous_out": [flow_out(flow, rate) for flow in contents],
            "capacities": [{"name": "room", "flows": list(contents), "capacity": capacity,
                            "content_init": dict(contents), "fill_rate": 0.0, "side": "out"}]}


TYPES += [
    # A room of 1 hydrogen in 99 of air, full; and one whose hydrogen has
    # run out (its input is connected to nothing).
    {"name": "MixedRoom", "uses": mixed_room_uses({"H2": 1.0, "AIR": 99.0}, 100.0, 50.0)},
    {"name": "MixedStore", "uses": mixed_room_uses({"H2": 0.0, "AIR": 50.0}, 100.0, 50.0)},
    {"name": "MixedRoomLowCeiling", "uses": mixed_room_uses({"H2": 1.0, "AIR": 99.0}, 100.0, 1.0)},
    # One rule consuming 10 of hydrogen and 10 of air into 20 of mixture,
    # handing back what it does not consume.
    {"name": "Mixer",
     "uses": [mu("continuous_flow_in", name="H2"), mu("continuous_flow_in", name="AIR"),
              mu("surplus_release", name="H2"), mu("surplus_release", name="AIR"),
              mu("rule_set", name="r", active="run", idle="none", consumes=["H2", "AIR"],
                 produces=["mix"], always_on=True),
              mu("rule_consumes", rule_set="r", active="run", flow="H2", rate=10.0, always_on=True),
              mu("rule_consumes", rule_set="r", active="run", flow="AIR", rate=10.0, always_on=True),
              mu("rule_produces", rule_set="r", active="run", flow="mix", rate=20.0, always_on=True),
              mu("continuous_flow_out", name="mix", available="produced")]},
]

MIXER = {"type": "ObjFlow", "name": "MIX", "flows_continuous_in": [{"name": "H2"}, {"name": "AIR"}],
         "flows_continuous_out": [{"name": "mix"}],
         "rules": [{"name": "r", "rules": [{"name": "run", "cons": {"H2": 10.0, "AIR": 10.0},
                                            "prod": {"mix": 20.0}}]}]}

BENCHES.update({
    # Fragments: volume_pooled, volume_pooled_flow, continuous_flow_out
    # `available: pooled`, volume_total starting full. Hydrogen asked 2 and
    # air 3 share the excess of 5 in proportion to the raw contents, until
    # both run out together at t = 33.
    "pooled_draw": (
        [mixed_room_object({"H2": 1.0, "AIR": 99.0}, 100.0, 50.0),
         {"type": "ObjFlow", "name": "CH2", "flows_continuous_in": [flow_in("H2", 2.0)]},
         {"type": "ObjFlow", "name": "CAIR", "flows_continuous_in": [flow_in("AIR", 3.0)]}],
        [{"name": "ROOM", "type": "MixedRoom"},
         {"name": "CH2", "type": "Sink", "parameters": {"flow": "H2", "demand": 2.0}},
         {"name": "CAIR", "type": "Sink", "parameters": {"flow": "AIR", "demand": 3.0}}],
        [("ROOM.H2_out", "CH2.H2_in"), ("ROOM.AIR_out", "CAIR.AIR_in")],
        [], 32.0,
    ),
    # A constituent that has run out (hydrogen, whose input nothing feeds)
    # serves only what arrives of it, nothing; the air takes the whole
    # excess.
    "pooled_draw_one_constituent_empty": (
        [mixed_room_object({"H2": 0.0, "AIR": 50.0}, 100.0, 50.0),
         {"type": "ObjFlow", "name": "CH2", "flows_continuous_in": [flow_in("H2", 2.0)]},
         {"type": "ObjFlow", "name": "CAIR", "flows_continuous_in": [flow_in("AIR", 3.0)]}],
        [{"name": "ROOM", "type": "MixedStore"},
         {"name": "CH2", "type": "Sink", "parameters": {"flow": "H2", "demand": 2.0}},
         {"name": "CAIR", "type": "Sink", "parameters": {"flow": "AIR", "demand": 3.0}}],
        [("ROOM.H2_out", "CH2.H2_in"), ("ROOM.AIR_out", "CAIR.AIR_in")],
        [], 24.0,
    ),
    # Fragments: rule_consumes `consumed_in`, surplus_release on a pooled
    # supplier. The room hands the rule 0.2 of hydrogen and 10 of air; it
    # runs at the draw the hydrogen allows and the air it does not consume
    # stays in the room.
    "pooled_draw_feeds_a_rule": (
        [mixed_room_object({"H2": 1.0, "AIR": 99.0}, 100.0, 50.0), MIXER,
         {"type": "ObjFlow", "name": "SINK", "flows_continuous_in": [flow_in("mix", 20.0)]}],
        [{"name": "ROOM", "type": "MixedRoom"}, {"name": "MIX", "type": "Mixer"},
         {"name": "SINK", "type": "Sink", "parameters": {"flow": "mix", "demand": 20.0}}],
        [("ROOM.H2_out", "MIX.H2_in"), ("ROOM.AIR_out", "MIX.AIR_in"), ("MIX.mix_out", "SINK.mix_in")],
        [], 5.0,
    ),
    # A room both of whose constituents arrive and transit it: the excess
    # is what the consumers ask over what transits.
    "pooled_draw_with_transit": (
        [mixed_room_object({"H2": 1.0, "AIR": 99.0}, 100.0, 50.0),
         {"type": "ObjFlow", "name": "SH2", "flows_continuous_out": [flow_out("H2", 0.5)]},
         {"type": "ObjFlow", "name": "SAIR", "flows_continuous_out": [flow_out("AIR", 1.0)]},
         {"type": "ObjFlow", "name": "CH2", "flows_continuous_in": [flow_in("H2", 2.0)]},
         {"type": "ObjFlow", "name": "CAIR", "flows_continuous_in": [flow_in("AIR", 3.0)]}],
        [{"name": "ROOM", "type": "MixedRoom"},
         {"name": "SH2", "type": "Source", "parameters": {"flow": "H2", "rate": 0.5}},
         {"name": "SAIR", "type": "Source", "parameters": {"flow": "AIR", "rate": 1.0}},
         {"name": "CH2", "type": "Sink", "parameters": {"flow": "H2", "demand": 2.0}},
         {"name": "CAIR", "type": "Sink", "parameters": {"flow": "AIR", "demand": 3.0}}],
        [("SH2.H2_out", "ROOM.H2_in"), ("SAIR.AIR_out", "ROOM.AIR_in"),
         ("ROOM.H2_out", "CH2.H2_in"), ("ROOM.AIR_out", "CAIR.AIR_in")],
        [], 40.0,
    ),
    # Outlets of 1 below requests of 2 and 3: the capability caps each share.
    "pooled_draw_under_a_ceiling": (
        [mixed_room_object({"H2": 1.0, "AIR": 99.0}, 100.0, 1.0),
         {"type": "ObjFlow", "name": "CH2", "flows_continuous_in": [flow_in("H2", 2.0)]},
         {"type": "ObjFlow", "name": "CAIR", "flows_continuous_in": [flow_in("AIR", 3.0)]}],
        [{"name": "ROOM", "type": "MixedRoomLowCeiling"},
         {"name": "CH2", "type": "Sink", "parameters": {"flow": "H2", "demand": 2.0}},
         {"name": "CAIR", "type": "Sink", "parameters": {"flow": "AIR", "demand": 3.0}}],
        [("ROOM.H2_out", "CH2.H2_in"), ("ROOM.AIR_out", "CAIR.AIR_in")],
        [], 40.0,
    ),
    # Two consumers on one pooled output share its delivery pro rata.
    "pooled_output_with_two_consumers": (
        [mixed_room_object({"H2": 1.0, "AIR": 99.0}, 100.0, 50.0),
         {"type": "ObjFlow", "name": "CH2", "flows_continuous_in": [flow_in("H2", 2.0)]},
         {"type": "ObjFlow", "name": "CH2B", "flows_continuous_in": [flow_in("H2", 1.0)]},
         {"type": "ObjFlow", "name": "CAIR", "flows_continuous_in": [flow_in("AIR", 3.0)]}],
        [{"name": "ROOM", "type": "MixedRoom"},
         {"name": "CH2", "type": "Sink", "parameters": {"flow": "H2", "demand": 2.0}},
         {"name": "CH2B", "type": "Sink", "parameters": {"flow": "H2", "demand": 1.0}},
         {"name": "CAIR", "type": "Sink", "parameters": {"flow": "AIR", "demand": 3.0}}],
        [("ROOM.H2_out", "CH2.H2_in"), ("ROOM.H2_out", "CH2B.H2_in"), ("ROOM.AIR_out", "CAIR.AIR_in")],
        [], 30.0,
    ),
    # A rule input fed by the room and by a plain source: the release
    # measures both offers against what the rule consumes.
    "pooled_and_plain_suppliers": (
        [mixed_room_object({"H2": 1.0, "AIR": 99.0}, 100.0, 50.0), MIXER,
         {"type": "ObjFlow", "name": "S", "flows_continuous_out": [flow_out("H2", 0.1)]},
         {"type": "ObjFlow", "name": "SINK", "flows_continuous_in": [flow_in("mix", 20.0)]}],
        [{"name": "ROOM", "type": "MixedRoom"}, {"name": "MIX", "type": "Mixer"},
         {"name": "S", "type": "Source", "parameters": {"flow": "H2", "rate": 0.1}},
         {"name": "SINK", "type": "Sink", "parameters": {"flow": "mix", "demand": 20.0}}],
        [("ROOM.H2_out", "MIX.H2_in"), ("S.H2_out", "MIX.H2_in"), ("ROOM.AIR_out", "MIX.AIR_in"),
         ("MIX.mix_out", "SINK.mix_in")],
        [], 5.0,
    ),
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
               "library": {**LIBRARY, "imports": ["std", "muscadet"], "types": TYPES}}
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


#: Option values of continuous fragments no bench applies, each with its
#: reason. A new option value lands with its bench, never here.
UNBENCHED_VALUES = {
    # The benches cover the direction and operator the reference studies
    # use; the others are the same comparison mirrored.
    ("threshold_compare", "operator", "le"): "mirror of `lt`, benched",
    ("threshold_compare", "operator", "gt"): "mirror of `lt`, benched",
    ("threshold_compare", "operator", "ge"): "mirror of `lt`, benched",
}


def applied_uses():
    """Every (fragment, parameter, value) a bench instance applies, with
    each `$param` resolved from the instance or the type's default."""
    types = {t["name"]: t for t in TYPES}
    applied = set()
    for _, instances, _, _, _ in BENCHES.values():
        for instance in instances:
            ty = types[instance["type"]]
            given = instance.get("parameters", {})
            defaults = {q["name"]: q.get("default") for q in ty.get("parameters", [])}
            for u in ty["uses"]:
                fragment = u["fragment"].removeprefix("std.").removeprefix("muscadet.")
                applied.add((fragment, None, None))
                for key, value in u["with"].items():
                    if isinstance(value, dict) and "$param" in value:
                        value = given.get(value["$param"], defaults.get(value["$param"]))
                    applied.add((fragment, key, json.dumps(value)))
    return applied


def test_the_benches_exercise_every_continuous_fragment():
    """No continuous fragment of std or muscadet, and no value of one of its
    options other than the default, goes uncharacterised: an option of an
    already benched fragment needs a bench of its own."""
    # `muscadet` holds the continuous fragments, `std` the generic ones and
    # the names forwarding to `muscadet`, which take the same options.
    embedded = {f["name"]: f for f in pyraichu.standard_library()["library"]["fragments"]}
    embedded.update({f["name"]: f for f in pyraichu.embedded_library("muscadet")["library"]["fragments"]})
    continuous = [f for f in embedded.values() if f["name"] not in BOOLEAN_FRAGMENTS]
    applied = applied_uses()
    missing = {f["name"] for f in continuous} - {f for f, _, _ in applied}
    assert not missing, missing
    unbenched = set()
    for fragment in continuous:
        for parameter in fragment["parameters"]:
            values = {"bool": [True, False], "enum": parameter.get("values", [])}.get(parameter["kind"], [])
            for value in values:
                if value == parameter.get("default"):
                    continue
                if (fragment["name"], parameter["name"], json.dumps(value)) not in applied:
                    unbenched.add((fragment["name"], parameter["name"], value))
    assert unbenched == set(UNBENCHED_VALUES), unbenched ^ set(UNBENCHED_VALUES)


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


def test_a_cascade_hands_the_second_set_what_the_first_leaves():
    """The cascade bench crosses its three regimes: the first set serves
    the whole demand and the second makes none of it, the first serves
    part and the second the rest, and the first is off and the second
    serves it all."""
    library, (events, _) = bench_run("cascade_two_rule_sets")
    off = [t for t, name in events if name == "V.first_mode.first_on_to_off"]
    on = [t for t, name in events if name == "V.first_mode.first_off_to_on"]
    assert off and on
    instants = [i * 0.25 for i in range(97)]
    result = pyraichu.simulate(pyraichu.load_model(library), t_max=24.0, samples=instants)
    s = {name: dict(series) for name, series in result.samples.items()}
    regimes = set()
    for t in instants:
        first_on = any(a < t < b for a, b in zip(on, off))
        demand, first, second = s["V_X_demand_out"][t], s["V_first_draw_scale"][t], s["V_second_scale"][t]
        if not first_on and off[0] < t < on[1]:
            assert second == pytest.approx(demand), t
            regimes.add("first off")
        elif first_on and second == 0.0 and demand > 0.0:
            regimes.add("first serves all")
        elif first_on and first > 0.0 and second > 0.0:
            assert first + second == pytest.approx(demand), t
            regimes.add("both serve")
    assert regimes == {"first off", "first serves all", "both serve"}


@pytest.mark.parametrize(
    "bench, instance, changed",
    [
        # A bus distributing what could arrive instead of what did hands on
        # more than it received while the supply falls short.
        ("pass_through_bus", "BUS", {"type": "BusByCapability"}),
        # A bounded fill at zero is not an unbounded one.
        ("unbounded_fill", "T", {"type": "Store", "parameters": {
            "volume": "tank", "flow": "H2", "capacity": 30.0, "rate": 1.0, "fill_rate": 0.0}}),
        # Producers cascaded in the other order: the second set is asked
        # the whole demand and the first only what it leaves.
        ("cascade_two_rule_sets", "V", {"type": "VentilatorReversed"}),
        # A band released at another threshold switches at other dates.
        ("median_of_instruments", "AP", {"parameters": {
            "measure": "reading_level", "output": "high", "direction": "above",
            "activate": 4.5, "release": 1.6}}),
        # A voter summing the four readings instead of taking their median.
        ("median_of_instruments", "AP", {"type": "Band", "parameters": {
            "measure": "reading_level", "output": "high", "direction": "above",
            "activate": 4.5, "release": 1.5}}),
    ],
)
def test_a_changed_instance_is_reported_different(bench, instance, changed):
    """Negative control: the agreement above is not vacuous."""
    objects, instances, connections, modes, t_max = BENCHES[bench]
    instances = [{**i, **changed} if i["name"] == instance else i for i in instances]
    plugin = plugin_model(bench, objects, connections)
    library = library_model(bench, instances, connections, modes, plugin["indicators"])
    assert run(library, t_max) != run(plugin, t_max)
