"""Several mixture machines draw one volume (muscadet's R52, 5.17.0).

Each machine takes its own rate at the volume's composition and the draws
add up. When the volume cannot serve their sum, each is served pro rata of
its composed share: the claims handed to the split are scaled to what the
volume delivers, so the truncation of a claim at the available quantity
cannot bend the pro-rata (machines of 100 and 1 behind a ceiling of 50.5 get
50 and 0.5, not 49.52 and 0.98).

Refused as muscadet refuses them: a group leaving out a constituent of its
volume, a volume drawn by several groups allocating other than
proportionally, a per-flow consumer beside the groups, and a machine
carrying a drawn flow onward.
"""

from __future__ import annotations

import math

import pytest

import pyraichu.muscadet as mu
from conftest import sampled

V = 90.0
Q = 50.0
QH2 = 2.0
FAN = 30.0
DRAFT = 1.0
REL = 1e-6


def source(flow: str, rate: float) -> type[mu.ObjFlow]:
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=rate)

    return Source


def room(flows=("AIR", "H2"), h2_init=1.0, serve_rate=None, **output):
    class Room(mu.ObjFlow):
        def add_flows(self):
            for flow in flows:
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow, **output)
            content = {flow: 1.0 for flow in flows}
            content.update({"AIR": V, "H2": h2_init})
            self.add_capacity(
                name="room",
                flows=list(flows),
                side="out",
                capacity=1e5,
                content_init={k: v for k, v in content.items() if k in flows},
                fill_rate=math.inf,
                transmits=True,
                serve_rate=serve_rate,
            )

    return Room


def machine(rate: float, flows=("AIR", "H2")) -> type[mu.ObjFlow]:
    class Machine(mu.ObjFlow):
        def add_flows(self):
            for flow in flows:
                self.add_flow_continuous_in(name=flow)
            self.add_mixture_in(name="extraction", flows=list(flows), flow_rate=rate)

    return Machine


def two_machines(fan=FAN, draft=DRAFT, supply=True, **room_settings) -> mu.System:
    system = mu.System("two_machines")
    if supply:
        system.add_component(source("AIR", Q), "S_AIR")
        system.add_component(source("H2", QH2), "S_H2")
    system.add_component(room(**room_settings), "ROOM")
    system.add_component(machine(fan), "FAN")
    system.add_component(machine(draft), "DRAFT")
    for flow in ("AIR", "H2"):
        if supply:
            system.connect(f"S_{flow}", flow, "ROOM", flow)
        system.connect("ROOM", flow, "FAN", flow)
        system.connect("ROOM", flow, "DRAFT", flow)
    return system


def drawn(result, consumer, instant):
    return sampled(result, f"{consumer}_AIR_fed_in", instant) + sampled(
        result, f"{consumer}_H2_fed_in", instant
    )


def test_each_machine_draws_its_own_rate_at_the_composition():
    result = two_machines().simulate(t_max=4.0, samples=[4.0])

    assert drawn(result, "FAN", 4.0) == pytest.approx(FAN, rel=REL)
    assert drawn(result, "DRAFT", 4.0) == pytest.approx(DRAFT, rel=REL)
    ratio = sampled(result, "FAN_H2_fed_in", 4.0) / sampled(result, "DRAFT_H2_fed_in", 4.0)
    assert ratio == pytest.approx(FAN / DRAFT, rel=REL)


def test_a_volume_that_cannot_serve_the_sum_serves_each_pro_rata():
    result = two_machines(
        fan=100.0, draft=1.0, supply=False, h2_init=0.0, serve_rate=50.5
    ).simulate(t_max=0.1, samples=[0.1])

    assert drawn(result, "FAN", 0.1) == pytest.approx(50.0, rel=REL)
    assert drawn(result, "DRAFT", 0.1) == pytest.approx(0.5, rel=REL)


def test_a_group_leaving_out_a_constituent_is_refused():
    system = mu.System("partial")
    system.add_component(room(flows=("AIR", "H2", "CO2")), "ROOM")
    system.add_component(machine(FAN), "FAN")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "FAN", flow)

    with pytest.raises(ValueError, match="CO2") as refused:
        system.build_dict()
    assert "does not draw" in str(refused.value)


def test_two_groups_need_a_proportional_allocation():
    system = two_machines(
        allocation="priority", allocation_priorities={"FAN": 1, "DRAFT": 2}
    )
    with pytest.raises(ValueError, match="proportional"):
        system.build_dict()


def test_one_group_keeps_whatever_allocation_it_declared():
    system = mu.System("one_priority")
    system.add_component(
        room(allocation="priority", allocation_priorities={"FAN": 1}), "ROOM"
    )
    system.add_component(machine(FAN), "FAN")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "FAN", flow)
    system.build_dict()


class Tap(mu.ObjFlow):
    def add_flows(self):
        for flow in ("AIR", "H2"):
            self.add_flow_continuous_in(name=flow, var_demand_in_default=1.0)


def test_a_per_flow_consumer_beside_two_groups_is_refused():
    system = two_machines()
    system.add_component(Tap, "TAP")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "TAP", flow)

    with pytest.raises(ValueError, match="also serves") as refused:
        system.build_dict()
    assert "TAP" in str(refused.value)


def test_a_per_flow_consumer_on_another_held_output_is_refused():
    """The gap of 0.102: the group drew AIR and H2 of a three-constituent
    volume, a tap drew CO2, and nothing refused it."""
    system = mu.System("other_output")
    system.add_component(room(flows=("AIR", "H2", "CO2")), "ROOM")
    system.add_component(machine(FAN, flows=("AIR", "H2", "CO2")), "FAN")

    class CO2Tap(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="CO2", var_demand_in_default=1.0)

    system.add_component(CO2Tap, "TAP")
    for flow in ("AIR", "H2", "CO2"):
        system.connect("ROOM", flow, "FAN", flow)
    system.connect("ROOM", "CO2", "TAP", "CO2")

    with pytest.raises(ValueError, match="also serves"):
        system.build_dict()


def test_two_groups_of_one_component_on_one_volume_are_refused():
    class Twin(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
            self.add_mixture_in(name="fan", flows=["AIR", "H2"], flow_rate=FAN)
            self.add_mixture_in(name="draft", flows=["AIR", "H2"], flow_rate=DRAFT)

    with pytest.raises(ValueError, match="already draws"):
        Twin("TWIN")


def test_a_machine_carrying_a_drawn_flow_onward_is_refused():
    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="AIR")
            self.add_mixture_in(name="pump", flows=["AIR"], flow_rate=FAN)

    with pytest.raises(ValueError, match="continuous OUTPUT"):
        Pump("PUMP")


def test_an_ungrouped_input_of_a_drawing_machine_is_refused():
    """A machine drawing the room as a mixture AND sampling one of its held
    outputs per flow, over a renamed port: the sample is a per-flow request,
    whatever component carries it."""

    class SamplingFan(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
            self.add_flow_continuous_in(name="H2_SAMPLE", var_demand_in_default=5.0)
            self.add_mixture_in(name="extraction", flows=["AIR", "H2"], flow_rate=FAN)

    system = mu.System("sampling")
    system.add_component(room(), "ROOM")
    system.add_component(SamplingFan, "FAN")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "FAN", flow)
    system.connect("ROOM", "H2", "FAN", "H2_SAMPLE")

    with pytest.raises(ValueError, match="also serves"):
        system.build_dict()
