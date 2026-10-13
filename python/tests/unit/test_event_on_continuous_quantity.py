"""An event on a continuous quantity fires at the crossing.

A source feeds hydrogen at 0.5 into a room (capacity 100, 90 of air)
drained by a fixed draft of 0.0125, and an ``ObjEvent`` watches
``room_qty_H2_leak >= 3.75``; a ``had_value`` indicator watches the same
predicate. Nothing discrete happens after the room's stock leaves empty,
so the event used to never fire, while the indicator, in the same run,
flipped at the crossing (edgemind-sas/cod3s-raichu#162). The crossing is
the closed form ``3.75 / (0.5 - 0.0125)``.
"""

from __future__ import annotations

import json

import pyraichu
from pyraichu.muscadet import engine

from conftest import CROSSING_TOL, fired_at

THRESHOLD = 3.75
CROSSING = THRESHOLD / (0.5 - 0.0125)
AUTHORIZED = [{"class_name_bkd": ".*"}]


def flow_in(name: str, var_type: str = "float", demand: float = 0.0) -> dict:
    if var_type == "bool":
        return {
            "name": name, "var_type": "bool", "var_fed_default": False,
            "component_authorized": AUTHORIZED, "var_in_default": False,
            "var_available_in_default": True, "logic": "or", "cls": "FlowIn",
        }
    return {
        "name": name, "var_type": "float", "var_fed_default": 0.0,
        "component_authorized": AUTHORIZED, "var_in_default": 0.0,
        "var_demand_default": demand, "cls": "FlowContinuousIn",
    }


def flow_out(name: str, fed: float = 0.0) -> dict:
    return {
        "name": name, "var_type": "float", "var_fed_default": fed,
        "component_authorized": AUTHORIZED, "var_demand_in_default": 0.0,
        "allocation": "proportional", "allocation_shares": {},
        "allocation_priorities": {}, "var_prod_cond_inner_mode": "or",
        "cls": "FlowContinuousOut",
    }


def rule_set(name: str, rules: list[dict]) -> dict:
    return {"name": name, "rules": [{**rule, "cls": "Rule"} for rule in rules]}


def obj(name: str, class_name: str, flows: list[dict], **sections) -> dict:
    return {
        "name": name, "cls": "ObjFlow", "source_cls": "ObjFlow", "flows": flows,
        "capacities": [], "measurements_in": [], "measurements_out": [],
        "rules": [], "transfers": [], "mixtures": [], "automata": [],
        "failure_modes": [],
        "metadata": {
            "class_name": class_name, "attributes_initial": [],
            "instance_overrides": [], "capacity_overrides": [],
        },
        "create_default_out_automata": True,
        **sections,
    }


def connection(source: str, target: str, flow: str, out: str | None = None) -> dict:
    return {
        "source": source, "source_box": f"{out or flow}_out",
        "target": target, "target_box": f"{flow}_in", "flow": out or flow,
    }


def declaration() -> dict:
    """The issue's reproduction, as the platform exports it."""
    vent = [{"name": "vent", "negate": False, "port": "in", "op": None,
             "value": None, "cls": "RuleOperand"}]
    room = {
        "name": "room", "side": "out", "capacity": 100.0, "transmits": True,
        "fill_rate": float("inf"),
        "flows": [
            {"name": flow, "weight": 1.0, "side": "out", "cls": "CapacityFlow"}
            for flow in ("H2_leak", "AIR")
        ],
        "content_init": {"H2_leak": 0.0, "AIR": 90.0},
    }
    components = {
        "LOCAL": obj(
            "LOCAL", "Room",
            [flow_in("H2_leak"), flow_in("AIR"), flow_out("H2_leak"), flow_out("AIR")],
            capacities=[room],
        ),
        "VENTILATION": obj(
            "VENTILATION", "Ventilation",
            [flow_in("H2_leak"), flow_in("AIR"), flow_in("vent", "bool"),
             flow_out("exhaust"), flow_out("draft"), flow_out("AIR")],
            rules=[
                rule_set("extraction", [
                    {"name": "running", "cond": vent, "cons": {"H2_leak": 25.0},
                     "prod": {"exhaust": 25.0}},
                    {"name": "idle", "cond": [], "cons": {"H2_leak": 0.0},
                     "prod": {"exhaust": 0.0}},
                ]),
                rule_set("natural_extraction", [
                    {"name": "draft", "cond": [], "cons": {"H2_leak": 0.00625},
                     "prod": {"draft": 0.00625}},
                ]),
                rule_set("natural_renewal", [
                    {"name": "draft", "cond": [], "cons": {"AIR": 0.00625},
                     "prod": {"draft": 0.00625, "AIR": 0.00625}},
                ]),
            ],
        ),
        "ATMOSPHERE": obj(
            "ATMOSPHERE", "Atmosphere",
            [flow_in("exhaust", demand=50.0), flow_in("draft", demand=0.0125)],
        ),
        "SRC": obj("SRC", "LeakSource", [flow_out("H2_leak", fed=0.5)]),
        "room_over_threshold": {
            "name": "room_over_threshold", "kind": "two_state_mode",
            "cls": "ObjEvent",
            "cond": [[{"obj": "LOCAL", "attr": "room_qty_H2_leak", "ope": ">=",
                       "value": THRESHOLD}]],
        },
    }
    return {
        "version": "1.0.3",
        "name": "threshold_event",
        "components": components,
        "connections": [
            connection("LOCAL", "VENTILATION", "AIR"),
            connection("LOCAL", "VENTILATION", "H2_leak"),
            connection("VENTILATION", "LOCAL", "AIR"),
            connection("VENTILATION", "ATMOSPHERE", "draft"),
            connection("VENTILATION", "ATMOSPHERE", "exhaust"),
            connection("SRC", "LOCAL", "H2_leak"),
        ],
        "indicators": [{
            "name": "room_over_threshold_had_value", "label": "x",
            "description": "x", "unit": "", "measure": "had_value",
            "stats": ["mean"], "component": "LOCAL", "operator": ">=",
            "value_test": THRESHOLD, "attr_name": "room_qty_H2_leak",
            "attr_type": "VAR", "kind": "PycAttrIndicator",
        }],
        "generated_indicators": True,
    }


def run(tempo: float = 0.0):
    document = declaration()
    if tempo:
        document["components"]["room_over_threshold"]["tempo_occ"] = tempo
    # Through JSON, as a platform export reaches the layer.
    model = engine.build_model(json.loads(json.dumps(document)))
    return pyraichu.simulate(model, t_max=20.0, samples=[0.0, 20.0])


def test_the_event_fires_at_the_crossing_with_no_discrete_event_after_it():
    assert abs(fired_at(run(), "occ") - CROSSING) < CROSSING_TOL


def test_the_event_and_the_indicator_on_its_predicate_agree():
    result = run()
    flips = [t for t, value in result.indicators["room_over_threshold_had_value"] if value]
    assert flips, "the indicator never saw the crossing"
    assert abs(fired_at(result, "occ") - flips[0]) < CROSSING_TOL


def test_a_tempo_runs_from_the_crossing():
    # The room keeps filling past the threshold, so the condition holds
    # through the whole tempo: the event fires one hour after the crossing,
    # with nothing discrete happening in between to start the countdown.
    assert abs(fired_at(run(tempo=1.0), "occ") - (CROSSING + 1.0)) < CROSSING_TOL
