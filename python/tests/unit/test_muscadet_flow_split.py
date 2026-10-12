"""How one supply is shared, and what a rule makes of its share (muscadet 5.14.0).

Three statements of muscadet's continuous-flow semantics, each pinned on the
smallest declaration that shows it:

* **an unbounded fill claim weighs as the whole supply.** A volume with
  ``fill_rate = inf`` publishes an infinite demand, and the producer's split
  truncates every claim at the quantity it has to give
  (``regularize_demands``) before it divides. A source of 10 shared by such a
  volume and a consumer asking 3 hands them 100/13 and 30/13;
* **a rule makes what arrived.** A capability is published whole to every
  consumer of a supply (R-20, ``incoming_capability`` is a sum, never a
  share), so a rule cannot size its production on it: it runs at the scale its
  inputs actually deliver. An electrolyser asking 20 beside an unbounded store
  on a bus of 25 receives 500/45 and makes a tenth of it;
* **rule sets on one input share it in declaration order, on what each
  actually drew** (R-17, with the uptake factor of R-13). A fan whose forced
  extraction a failure mode has cut to zero draws nothing through it, so the
  natural draft declared after it keeps its whole demand.

And one statement of the engine's: **a dense sample reports the same settled
state the dynamics integrate**, so a sampled inflow is the rate its stock
rises at and a sampled production keeps its rule's balance, whatever the
supply does between two discrete events.

Every expected number is the muscadet 5.14.0 reference engine's, recorded on
the very documents below. The documents are written field by field in the
shape ``muscadet.engine.system_declaration`` produces, reduced to the keys that
carry a decision, so the module needs no muscadet.
"""

from __future__ import annotations

import json
import math

import pytest

import pyraichu
import pyraichu.muscadet.declare as declare

#: Agreement with the reference: the split and the balances are closed forms,
#: the stocks are integrated.
REL = 1e-6

# --- the documents ----------------------------------------------------


def continuous_out(name, **extra):
    return {"cls": "FlowContinuousOut", "name": name, "var_type": "float", **extra}


def continuous_in(name, **extra):
    return {"cls": "FlowContinuousIn", "name": name, "var_type": "float", **extra}


def component(name, flows, **sections):
    return {"name": name, "cls": "ObjFlow", "flows": flows, **sections}


def link(source, target, flow):
    return {
        "source": source,
        "source_box": f"{flow}_out",
        "target": target,
        "target_box": f"{flow}_in",
        "flow": flow,
    }


def a_document(name, components, connections):
    return {
        "version": "1.0.3",
        "name": name,
        "components": {entry["name"]: entry for entry in components},
        "connections": connections,
    }


def an_unbounded_store(name, flow, volume):
    """A transiting volume claiming without bound (`fill_rate = inf`)."""
    return component(
        name,
        [continuous_in(flow), continuous_out(flow)],
        capacities=[
            {
                "name": "tank",
                "flows": [{"name": flow, "weight": 1.0, "side": "out"}],
                "capacity": volume,
                "side": "out",
                "content_init": {flow: 0.0},
                "transmits": True,
                "fill_rate": math.inf,
            }
        ],
    )


def a_shared_source():
    """A source of 10 shared by an unbounded buffer and a consumer asking
    3; the buffer feeds a sink asking 1."""
    return a_document(
        "flow_split",
        [
            component("SRC", [continuous_out("q", var_fed_default=10.0)]),
            an_unbounded_store("BUF", "q", 100.0),
            component("DIRECT", [continuous_in("q", var_demand_default=3.0)]),
            component("SINK", [continuous_in("q", var_demand_default=1.0)]),
        ],
        [link("SRC", "BUF", "q"), link("SRC", "DIRECT", "q"), link("BUF", "SINK", "q")],
    )


#: The bus of the solar curve: 30 +- 15 over a 24 h period.
SOLAR = {
    "cls": "SinusoidalProfile",
    "amplitude": 15.0,
    "period": 24.0,
    "phase_shift": 0.0,
    "offset": 30.0,
    "value_min": 0.0,
    "value_max": math.inf,
}


def a_shared_bus(profile=None):
    """A bus shared by an unbounded store and an electrolyser turning 20
    Elec into 2 H2 for a tank asking 2: the showcase's battery and
    electrolyser, reduced. A bus of 25, or the solar curve."""
    output = (
        continuous_out("Elec", var_fed_default=25.0)
        if profile is None
        else continuous_out("Elec", var_fed_default=1.0, profile=profile)
    )
    return a_document(
        "flow_bus",
        [
            component("BUS", [output]),
            an_unbounded_store("STORE", "Elec", 1000.0),
            component(
                "EL",
                [continuous_in("Elec"), continuous_out("H2")],
                rules=[
                    {
                        "name": "electrolysis",
                        "rules": [
                            {"name": "run", "cond": [], "cons": {"Elec": 20.0}, "prod": {"H2": 2.0}}
                        ],
                    }
                ],
            ),
            component("TANK", [continuous_in("H2", var_demand_default=2.0)]),
        ],
        [link("BUS", "STORE", "Elec"), link("BUS", "EL", "Elec"), link("EL", "TANK", "H2")],
    )


def a_derated_fan():
    """A room of two constituents fed 0.06 of hydrogen, ventilated by a fan
    carrying four rule sets: the forced `extraction` and `renewal`, then the
    `natural_extraction` and `natural_renewal` of the draft, declared after
    them. The fan is commanded, and a mode at date 0 cuts its exhaust to
    zero: the showcase's ventilation loss, reduced."""
    mode = "Ventilation_lost"
    room = component(
        "LOCAL",
        [
            continuous_in("H2_leak"),
            continuous_in("AIR"),
            continuous_out("H2_leak"),
            continuous_out("AIR"),
        ],
        capacities=[
            {
                "name": "room",
                "flows": [
                    {"name": "H2_leak", "weight": 1.0, "side": "out"},
                    {"name": "AIR", "weight": 1.0, "side": "out"},
                ],
                "capacity": 100.0,
                "side": "out",
                "content_init": {"H2_leak": 1.0, "AIR": 90.0},
                "transmits": True,
                "fill_rate": math.inf,
            }
        ],
    )

    def guarded(name, cons, prod):
        """A rule set running while commanded, idle otherwise."""
        idle = {flow: 0.0 for flow in prod}
        return {
            "name": name,
            "rules": [
                {
                    "name": "running",
                    "cond": [{"name": "vent", "negate": False, "port": "in"}],
                    "cons": cons,
                    "prod": prod,
                },
                {"name": "idle", "cond": [], "cons": {f: 0.0 for f in cons}, "prod": idle},
            ],
        }

    def draft(name, cons, prod):
        return {"name": name, "rules": [{"name": "draft", "cond": [], "cons": cons, "prod": prod}]}

    fan = component(
        "VENTILATION",
        [
            continuous_in("H2_leak"),
            continuous_in("AIR"),
            {
                "cls": "FlowIn",
                "name": "vent",
                "var_type": "bool",
                "var_in_default": True,
                "logic": "or",
            },
            continuous_out("exhaust"),
            continuous_out("draft"),
            continuous_out("AIR"),
        ],
        rules=[
            guarded("extraction", {"H2_leak": 25.0}, {"exhaust": 25.0}),
            guarded("renewal", {"AIR": 25.0}, {"exhaust": 25.0, "AIR": 25.0}),
            draft("natural_extraction", {"H2_leak": 0.00625}, {"draft": 0.00625}),
            draft("natural_renewal", {"AIR": 0.00625}, {"draft": 0.00625, "AIR": 0.00625}),
        ],
        deratings=[{"mode": mode, "flow": "exhaust"}],
    )
    variable = f"{mode}_derating_exhaust"
    lost = {
        "name": f"VENTILATION__{mode}",
        "kind": "two_state_mode",
        "cls": "ObjMode2S",
        "mode_name": mode,
        "targets": ["VENTILATION"],
        "target_name": "VENTILATION",
        "occ_law": {"cls": "delay", "time": [0.0]},
        "occ_effects": {variable: 0.0},
        "occ_param_name": ["ttf"],
        "occ_param": [0.0],
        "not_occ_law": {"cls": "exp", "rate": [0.0]},
        "not_occ_effects": {variable: 1.0},
        "not_occ_param_name": ["mu"],
        "not_occ_param": [0.0],
    }
    outside = component(
        "ATMOSPHERE",
        [
            continuous_in("exhaust", var_demand_default=50.0),
            continuous_in("draft", var_demand_default=0.0125),
        ],
    )
    return a_document(
        "flow_draft",
        [
            component("S_LEAK", [continuous_out("H2_leak", var_fed_default=0.06)]),
            room,
            fan,
            outside,
            lost,
        ],
        [
            link("S_LEAK", "LOCAL", "H2_leak"),
            link("LOCAL", "VENTILATION", "H2_leak"),
            link("LOCAL", "VENTILATION", "AIR"),
            link("VENTILATION", "LOCAL", "AIR"),
            link("VENTILATION", "ATMOSPHERE", "draft"),
            link("VENTILATION", "ATMOSPHERE", "exhaust"),
        ],
    )


def observe(document, readings, instants, t_max):
    """`{name: {t: value}}` for each `(name, component, attribute)`."""
    body = declare.build_document(document)
    body["model"]["indicators"] = [
        {"name": name, "target": "attribute", "attr": {"component": comp, "attribute": attr}}
        for name, comp, attr in readings
    ]
    result = pyraichu.simulate(
        pyraichu.load_model(json.dumps(body)), t_max=t_max, samples=list(instants)
    )
    return {name: dict(result.samples[name]) for name, _comp, _attr in readings}


def close(actual, expected):
    return abs(actual - expected) <= REL * max(1.0, abs(expected))


# --- an unbounded fill claim -------------------------------------------


def test_an_unbounded_fill_claim_weighs_as_the_whole_supply():
    """10 split between a claim truncated at 10 and a demand of 3: muscadet's
    `test_capacity_fill_001::test_the_distributed_total_is_still_the_available_quantity`,
    on this layer. Weighed at what the consumer beside it left, the claim
    took 7 and the consumer its whole 3."""
    seen = observe(
        a_shared_source(),
        [("buffer", "BUF", "q_fed_in"), ("direct", "DIRECT", "q_fed_in")],
        [0.5],
        t_max=1.0,
    )
    assert close(seen["buffer"][0.5], 100.0 / 13.0), seen  # 7.6923
    assert close(seen["direct"][0.5], 30.0 / 13.0), seen  # 2.3077


def test_a_capability_is_published_whole_to_every_consumer():
    """R-20: what a supply could deliver is not apportioned, so each of
    its consumers reads all of it, whatever the other one receives."""
    seen = observe(
        a_shared_source(),
        [("buffer", "BUF", "q_capability_in"), ("direct", "DIRECT", "q_capability_in")],
        [0.5],
        t_max=1.0,
    )
    assert close(seen["buffer"][0.5], 10.0), seen
    assert close(seen["direct"][0.5], 10.0), seen


# --- a rule makes what arrived ------------------------------------------


def test_a_rule_beside_an_unbounded_store_makes_what_it_receives():
    """Bus 25, store claiming without bound, electrolyser asking 20: the
    store receives 25^2/45, the electrolyser 25*20/45 and makes a tenth of
    it in H2, the store rising at what it receives."""
    seen = observe(
        a_shared_bus(),
        [
            ("store", "STORE", "Elec_fed_in"),
            ("electrolyser", "EL", "Elec_fed_in"),
            ("hydrogen", "EL", "H2_fed_out"),
            ("content", "STORE", "tank_content_Elec"),
        ],
        [0.5, 1.0],
        t_max=1.0,
    )
    for instant in (0.5, 1.0):
        assert close(seen["store"][instant], 625.0 / 45.0), seen  # 13.8889
        assert close(seen["electrolyser"][instant], 500.0 / 45.0), seen  # 11.1111
        assert close(seen["hydrogen"][instant], 50.0 / 45.0), seen  # 1.1111
        assert close(seen["content"][instant], 625.0 / 45.0 * instant), seen


#: The reference on the solar bus: (store inflow, electrolyser inflow, H2
#: made, store content) at t = 3, 6 and 9 h.
SOLAR_REFERENCE = {
    3.0: (27.206543, 13.400059, 1.340006, 68.434433),
    6.0: (31.153846, 13.846154, 1.384615, 157.853164),
    9.0: (27.206543, 13.400059, 1.340006, 247.271894),
}


def test_the_solar_bus_follows_the_reference():
    """The same split under a supply that moves continuously."""
    seen = observe(
        a_shared_bus(SOLAR),
        [
            ("store", "STORE", "Elec_fed_in"),
            ("electrolyser", "EL", "Elec_fed_in"),
            ("hydrogen", "EL", "H2_fed_out"),
            ("content", "STORE", "tank_content_Elec"),
        ],
        sorted(SOLAR_REFERENCE),
        t_max=10.0,
    )
    for instant, expected in SOLAR_REFERENCE.items():
        got = tuple(seen[key][instant] for key in ("store", "electrolyser", "hydrogen", "content"))
        assert all(close(a, b) for a, b in zip(got, expected)), (instant, got, expected)


# --- settled samples ----------------------------------------------------


def test_dense_samples_are_the_state_the_dynamics_integrate():
    """Between two discrete events the solar supply moves on, and a sample
    taken there must report the flows of THAT instant: the store's sampled
    inflow is the rate its integrated content rises at (a central
    difference over two neighbouring samples), and the electrolyser's
    sampled production keeps its rule's balance, 2 H2 per 20 Elec received.

    Measured on 0.90.0, where a sample read the allocations settled at the
    segment start: at t = 3 the content rose at 21.51 while the sampled
    inflow read 23.90, and the electrolyser made 2.0 H2 out of 16.7 Elec."""
    step = 1e-3
    centres = (3.0, 6.0, 9.0)
    instants = sorted({t + offset for t in centres for offset in (-step, 0.0, step)})
    seen = observe(
        a_shared_bus(SOLAR),
        [
            ("inflow", "STORE", "Elec_fed_in"),
            ("content", "STORE", "tank_content_Elec"),
            ("electrolyser", "EL", "Elec_fed_in"),
            ("hydrogen", "EL", "H2_fed_out"),
        ],
        instants,
        t_max=10.0,
    )
    for t in centres:
        rate = (seen["content"][t + step] - seen["content"][t - step]) / (2.0 * step)
        # The central difference of a smooth trajectory, step 1e-3.
        assert seen["inflow"][t] == pytest.approx(rate, rel=1e-5), (t, rate, seen["inflow"][t])
        assert seen["hydrogen"][t] == pytest.approx(
            seen["electrolyser"][t] * 2.0 / 20.0, rel=1e-9
        ), t


# --- rule sets sharing one input ----------------------------------------

#: The reference room: hydrogen held, at t = 0.5, 1 and 2 h. It rises at
#: 0.06 - 0.0125: the draft keeps its whole demand.
ROOM_H2 = {0.5: 1.02375, 1.0: 1.04750, 2.0: 1.09500}


def test_a_derated_rule_set_leaves_its_sibling_the_whole_input():
    """The forced extraction, declared first, has its only output cut to
    zero, so its uptake factor is zero and it draws nothing: the natural
    draft, declared after it on the same hydrogen, is offered all of it and
    keeps its 0.0125. Pro-rated by pre-uptake need, the dead set still
    counted for 50 and the draft received 0.00016."""
    seen = observe(
        a_derated_fan(),
        [
            ("draft", "VENTILATION", "draft_fed_out"),
            ("exhaust", "VENTILATION", "exhaust_fed_out"),
            ("room", "LOCAL", "room_content_H2_leak"),
        ],
        sorted(ROOM_H2),
        t_max=2.0,
    )
    for instant, held in ROOM_H2.items():
        assert close(seen["draft"][instant], 0.0125), (instant, seen)
        assert close(seen["exhaust"][instant], 0.0), (instant, seen)
        assert close(seen["room"][instant], held), (instant, seen)


def test_two_sets_on_one_input_are_served_in_declaration_order():
    """Nothing derated: the forced extraction, first declared, asks 50 of
    a room holding a trace and takes all it is offered, so the draft
    declared after it on the hydrogen is left nothing (R-17, first
    declared, first served). The draft output is still made whole, 0.0125
    as the reference makes it: the natural renewal, which draws air, is
    handed the demand the natural extraction cannot serve. What the two
    hydrogen sets draw together never exceeds what arrived."""
    document = a_derated_fan()
    del document["components"]["VENTILATION__Ventilation_lost"]
    seen = observe(
        document,
        [
            ("draft", "VENTILATION", "draft_fed_out"),
            ("exhaust", "VENTILATION", "exhaust_fed_out"),
            ("arrived", "VENTILATION", "H2_leak_fed_in"),
            ("forced", "VENTILATION", "extraction_draw_scale"),
            ("natural", "VENTILATION", "natural_extraction_draw_scale"),
        ],
        [0.5],
        t_max=1.0,
    )
    assert close(seen["draft"][0.5], 0.0125), seen
    assert close(seen["exhaust"][0.5], 50.0), seen
    drawn = 25.0 * seen["forced"][0.5] + 0.00625 * seen["natural"][0.5]
    assert drawn == pytest.approx(seen["arrived"][0.5], rel=1e-9), seen
    assert seen["natural"][0.5] == pytest.approx(0.0, abs=1e-12), seen
