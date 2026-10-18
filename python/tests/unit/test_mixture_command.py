"""A mixture machine's rate is a variable, commanded on and off (muscadet's
R52, 5.17.0).

The rate of a group is published as ``{group}_flow_rate`` on the consuming
component, the spelling muscadet gives it, so a failure mode written against
muscadet clamps it by name; a held effect releases back to the declared
value, which is what a repair means here. ``run_cond`` commands the machine
in the vocabulary of a capacity's ``serve_cond``, compiled to a gate
automaton ``{group}_run`` and composed with the rate by branching. A command
comparing the rate its own group draws closes a loop of rate comparisons and
is refused.
"""

from __future__ import annotations

import json
import math

import pytest

import pyraichu
import pyraichu.muscadet as mu
import pyraichu.muscadet.declare as declare
from conftest import CROSSING_TOL, sampled

V = 90.0
Q = 50.0
QH2 = 2.0
REL = 1e-6


def source(flow: str, rate: float) -> type[mu.ObjFlow]:
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=rate)

    return Source


class Room(mu.ObjFlow):
    def add_flows(self):
        for flow in ("AIR", "H2"):
            self.add_flow_continuous_in(name=flow)
            self.add_flow_continuous_out(name=flow)
        self.add_capacity(
            name="room",
            flows=["AIR", "H2"],
            side="out",
            capacity=1e5,
            content_init={"AIR": V, "H2": 1.0},
            fill_rate=math.inf,
            transmits=True,
        )


def fan(**group) -> type[mu.ObjFlow]:
    """A terminal extractor; a `vent` port is declared before the group."""

    class Fan(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
            self.add_flow_in(name="vent")
            self.add_mixture_in(
                name="extraction", flows=["AIR", "H2"], flow_rate=Q, **group
            )

    return Fan


def switch(held: bool) -> type[mu.ObjFlow]:
    class Switch(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="vent", var_prod_default=held)

    return Switch


def ventilated(held: bool = True, **group) -> mu.System:
    system = mu.System("commanded_fan")
    system.add_component(source("AIR", Q), "S_AIR")
    system.add_component(source("H2", QH2), "S_H2")
    system.add_component(Room, "ROOM")
    system.add_component(fan(**group), "FAN")
    system.add_component(switch(held), "SW")
    for flow in ("AIR", "H2"):
        system.connect(f"S_{flow}", flow, "ROOM", flow)
        system.connect("ROOM", flow, "FAN", flow)
    system.connect("SW", "vent", "FAN", "vent")
    return system


def drawn(result, instant: float) -> float:
    return sampled(result, "FAN_AIR_fed_in", instant) + sampled(
        result, "FAN_H2_fed_in", instant
    )


# --- 1. the rate is a published variable ----------------------------------


def test_the_rate_is_published_under_the_name_muscadet_gives_it():
    """A variable with no equation, at the declared value: what a mode
    clamps and a repair releases back to."""
    model = ventilated().build_dict()["model"]
    fan_entry = next(c for c in model["components"] if c["name"] == "FAN")
    (rate,) = [a for a in fan_entry["attributes"] if a["name"] == "extraction_flow_rate"]
    defined = {eq["target"] for eq in fan_entry["equations"]}

    assert rate["init"]["value"] == pytest.approx(Q)
    assert "extraction_flow_rate" not in defined


def fan_document(fails_at: float, repaired_after: float) -> dict:
    """The fan alone, drawing a large air room, with a mode clamping its rate
    by name. Written as a document, the shape a standalone mode is declared
    in."""

    def continuous(cls, name, **extra):
        entry = {"name": name, "cls": cls, "var_type": "float", "var_fed_default": 0.0}
        entry.update(extra)
        return entry

    def component(name, flows, **sections):
        entry = {
            "name": name,
            "cls": "ObjFlow",
            "flows": flows,
            "capacities": [],
            "rules": [],
            "transfers": [],
            "automata": [],
            "failure_modes": [],
        }
        entry.update(sections)
        return entry

    room = component(
        "ROOM",
        [
            continuous("FlowContinuousIn", "AIR", var_demand_default=0.0),
            {
                "name": "AIR",
                "cls": "FlowContinuousOut",
                "var_fed_default": 0.0,
                "var_demand_in_default": 0.0,
                "allocation": "proportional",
                "var_prod_cond_inner_mode": "or",
            }
        ],
        capacities=[
            {
                "name": "room",
                "capacity": 1e5,
                "content_init": {"AIR": 1e4},
                "fill_rate": 0.0,
                "side": "out",
                "transmits": True,
                "flows": [
                    {"name": "AIR", "weight": 1.0, "side": "out", "cls": "CapacityFlow"}
                ],
            }
        ],
    )
    fan_entry = component(
        "FAN",
        [continuous("FlowContinuousIn", "AIR", var_demand_default=0.0)],
        mixtures=[{"name": "extraction", "flows": ["AIR"], "flow_rate": Q}],
    )
    mode = {
        "name": "FAN__breakdown",
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": "breakdown",
        "targets": ["FAN"],
        "target_name": "FAN",
        "failure_effects": {"extraction_flow_rate": 0.0},
        "failure_param_name": ["ttf"],
        "failure_param": [fails_at],
        "repair_param_name": ["ttr"],
        "repair_param": [repaired_after],
    }
    return {
        "version": "1.0.0",
        "name": "broken_fan",
        "generated_indicators": True,
        "components": {"ROOM": room, "FAN": fan_entry, mode["name"]: mode},
        "connections": [
            {
                "source": "ROOM",
                "source_box": "AIR_out",
                "target": "FAN",
                "target_box": "AIR_in",
            }
        ],
    }


def test_a_mode_clamps_the_rate_by_name_and_its_repair_restores_it():
    document = fan_document(fails_at=2.0, repaired_after=3.0)
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=8.0, samples=[1.0, 3.0, 6.0])
    draw = dict(result.samples["FAN_AIR_fed_in"])

    assert draw[1.0] == pytest.approx(Q, rel=REL)
    assert draw[3.0] == pytest.approx(0.0, abs=1e-12)
    assert draw[6.0] == pytest.approx(Q, rel=REL), "the repair restores the rate"


# --- 2. the command ----------------------------------------------------------


def test_a_fan_draws_while_its_order_holds():
    result = ventilated(held=True, run_cond=[["vent"]]).simulate(
        t_max=1.0, samples=[1.0]
    )
    assert drawn(result, 1.0) == pytest.approx(Q, rel=REL)


def test_a_fan_draws_nothing_while_its_order_is_off():
    result = ventilated(held=False, run_cond=[["vent"]]).simulate(
        t_max=1.0, samples=[1.0]
    )
    assert drawn(result, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_an_uncommanded_fan_runs_whatever_the_port_says():
    result = ventilated(held=False).simulate(t_max=1.0, samples=[1.0])
    assert drawn(result, 1.0) == pytest.approx(Q, rel=REL)


START_AT = 3.0


class Clock(mu.ObjFlow):
    """An integrated level reading the date."""

    def add_flows(self):
        self.add_flow_continuous_in(name="tick")
        self.add_capacity(
            name="clock", flow="tick", side="in", capacity=1e6, fill_rate=math.inf
        )


class ClockedFan(mu.ObjFlow):
    def add_flows(self):
        for flow in ("AIR", "H2"):
            self.add_flow_continuous_in(name=flow)
        self.add_measurement_in(name="clock")
        self.add_mixture_in(
            name="extraction",
            flows=["AIR", "H2"],
            flow_rate=Q,
            run_cond=[[{"name": "clock_level", "op": ">=", "value": START_AT}]],
        )


def test_a_watched_threshold_starts_the_fan_at_its_date():
    system = mu.System("clocked_fan")
    system.add_component(source("AIR", Q), "S_AIR")
    system.add_component(source("H2", QH2), "S_H2")
    system.add_component(source("tick", 1.0), "S_TICK")
    system.add_component(Room, "ROOM")
    system.add_component(Clock, "CLOCK")
    system.add_component(ClockedFan, "FAN")
    for flow in ("AIR", "H2"):
        system.connect(f"S_{flow}", flow, "ROOM", flow)
        system.connect("ROOM", flow, "FAN", flow)
    system.connect("S_TICK", "tick", "CLOCK", "tick")
    system.connect_measurement("CLOCK", "clock", "FAN")

    result = system.simulate(t_max=4.0, samples=[START_AT - 0.05, START_AT + 0.05])

    assert drawn(result, START_AT - 0.05) == pytest.approx(0.0, abs=1e-12)
    assert drawn(result, START_AT + 0.05) == pytest.approx(Q, rel=REL)
    gate = [
        event.time
        for event in result.events
        if event.transition.rsplit(".", 1)[-1] == "extraction_run_opens"
    ]
    assert gate and gate[-1] == pytest.approx(START_AT, abs=CROSSING_TOL)


def test_a_command_comparing_the_rate_its_group_draws_is_refused():
    with pytest.raises(ValueError) as refused:
        ventilated(
            run_cond=[[{"name": "H2", "port": "in", "op": ">", "value": 0.1}]]
        ).build_dict()
    message = str(refused.value)
    assert "mixture group `extraction` of `FAN`" in message, message
    assert "loop of rate comparisons" in message, message


def test_machines_commanded_on_each_others_draw_are_refused():
    """Each decides within one instant the rate the other reads."""

    def room_of(flow):
        class OneRoom(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
                self.add_capacity(
                    name="room", flow=flow, side="out", capacity=1e5,
                    content_init={flow: V}, transmits=True,
                )

        return OneRoom

    class Crossed(mu.ObjFlow):
        def add_flows(self):
            for flow in ("A1", "A2"):
                self.add_flow_continuous_in(name=flow)
            self.add_mixture_in(
                name="g1", flows=["A1"], flow_rate=Q,
                run_cond=[[{"name": "A2", "port": "in", "op": "<", "value": 1.0}]],
            )
            self.add_mixture_in(
                name="g2", flows=["A2"], flow_rate=Q,
                run_cond=[[{"name": "A1", "port": "in", "op": "<", "value": 1.0}]],
            )

    system = mu.System("crossed")
    system.add_component(room_of("A1"), "R1")
    system.add_component(room_of("A2"), "R2")
    system.add_component(Crossed, "F")
    system.connect("R1", "A1", "F", "A1")
    system.connect("R2", "A2", "F", "A2")

    with pytest.raises(ValueError, match="loop of rate comparisons"):
        system.build_dict()


@pytest.mark.parametrize(
    "held, run_cond, inner, runs",
    [
        (True, [["vent", {"name": "vent", "negate": True}]], "and", False),
        (False, [[{"name": "vent", "negate": True}]], "or", True),
        (True, [["vent"], [{"name": "vent", "negate": True}]], "and", True),
    ],
)
def test_negation_and_inner_mode_decide_the_run(held, run_cond, inner, runs):
    """`and` reads a disjunction of conjunctions, as muscadet does."""
    result = ventilated(
        held=held, run_cond=run_cond, run_cond_inner_mode=inner
    ).simulate(t_max=1.0, samples=[1.0])
    assert drawn(result, 1.0) == pytest.approx(Q if runs else 0.0, rel=REL, abs=1e-12)


def test_a_malformed_command_mode_is_refused():
    with pytest.raises(ValueError, match="run_cond_inner_mode"):
        fan(run_cond=[["vent"]], run_cond_inner_mode="xor")("FAN")


def test_a_command_naming_nothing_is_refused():
    with pytest.raises(ValueError, match="nowhere"):
        fan(run_cond=[["nowhere"]])("FAN")
