"""A recirculation ring closing through a transiting volume.

The H2 plant's ventilated room is the shape: the fan draws air out of the
room and blows it back in, exhausting part of what it moves. As
declarations go, the fan's rule is bounded by the demand on its return
output, which is the room's inbound demand, which carries the fan's own
demand for room air through the transit: a chain of instantaneous values
closing on itself, which the engine refuses as having no solution.

muscadet accepts it because a volume sits on the ring (a ring with no
volume is its R30 refusal) and tears the edge entering the volume, reading
the torn quantities one evaluation late. This layer tears it at the other
end of the same edge: an output feeding a transiting volume on a ring no
longer bounds the rule producing it, the volume absorbing whatever the two
ends momentarily disagree on. That needs no late read, and it gives the
consistent solution of the sweep: the room holds its level exactly where
the reference shifts it once, by the fan's rate times one integration
stage (89.833 against 90 on the showcase), and settles on the same flows.

Every number here is a closed form of the minimal model the reference
was measured on (muscadet 5.6.0): a leak of 0.05 into a room of 100
holding 90 of air, a fan of rate 50 split between extracting the leak and
renewing the air, and an atmosphere asking for 50 of exhaust.
"""

import math

import pyraichu
import pyraichu.muscadet as mu
import pytest
from conftest import TOL, sampled

VENT_RATE = 50.0
LEAK = 0.05


def ventilated_room(
    held: tuple[str, ...] = ("H2_leak", "AIR"), fill_rate=math.inf, hydrogen: float = 0.0
):
    """LEAK -> LOCAL (a transiting volume) <-> VENTILATION -> ATMOSPHERE."""

    class Leak(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2_leak", var_fed_default=LEAK)

    class Local(mu.ObjFlow):
        def add_flows(self):
            for flow in ("H2_leak", "AIR"):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room",
                flows=[{"name": flow, "weight": 1.0} for flow in held],
                capacity=100.0,
                content_init={flow: 90.0 if flow == "AIR" else hydrogen for flow in held},
                fill_rate=fill_rate,
                transmits=True,
                side="out",
            )

    class Ventilation(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2_leak")
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="exhaust")
            self.add_flow_continuous_out(name="AIR")
            self.add_rule_set(
                name="extraction",
                rules=[
                    {
                        "cons": {"H2_leak": VENT_RATE / 2},
                        "prod": {"exhaust": VENT_RATE / 2},
                    }
                ],
            )
            self.add_rule_set(
                name="renewal",
                rules=[
                    {
                        "cons": {"AIR": VENT_RATE / 2},
                        "prod": {"exhaust": VENT_RATE / 2, "AIR": VENT_RATE / 2},
                    }
                ],
            )

    class Atmosphere(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="exhaust", var_demand_in_default=VENT_RATE)

    system = mu.System("capacity_ring")
    system.add_component(Leak, "LEAK")
    system.add_component(Local, "LOCAL")
    system.add_component(Ventilation, "VENTILATION")
    system.add_component(Atmosphere, "ATMOSPHERE")
    system.connect("LEAK", "H2_leak", "LOCAL", "H2_leak")
    system.connect("LOCAL", "H2_leak", "VENTILATION", "H2_leak")
    system.connect("LOCAL", "AIR", "VENTILATION", "AIR")
    system.connect("VENTILATION", "AIR", "LOCAL", "AIR")
    system.connect("VENTILATION", "exhaust", "ATMOSPHERE", "exhaust")
    return system


@pytest.mark.parametrize("held", [("AIR",), ("H2_leak", "AIR")])
def test_a_ventilated_room_recirculates_through_its_volume(held):
    """The fan exhausts the whole leak and tops the exhaust up with air
    (the cascade of its two rule sets), blows the air it drew back into
    the room, and the room's air neither fills nor drains."""
    result = ventilated_room(held).simulate(t_max=10.0, samples=[1.0, 5.0, 10.0])
    for instant in (1.0, 5.0, 10.0):
        assert abs(sampled(result, "VENTILATION_AIR_fed_in", instant) - 49.95) < TOL
        assert abs(sampled(result, "LOCAL_AIR_fed_in", instant) - 49.95) < TOL
        assert abs(sampled(result, "ATMOSPHERE_exhaust_fed_in", instant) - 50.0) < TOL
        assert abs(sampled(result, "VENTILATION_H2_leak_fed_in", instant) - LEAK) < TOL
        assert abs(sampled(result, "LOCAL_room_content_AIR", instant) - 90.0) < TOL


def test_the_leak_transits_the_room_rather_than_accumulating():
    """A held flow with nothing in stock passes through at what arrives:
    emptiness is judged per flow, so the room's 90 of air does not let
    it serve hydrogen it does not hold."""
    result = ventilated_room().simulate(t_max=10.0, samples=[1.0, 5.0, 10.0])
    for instant in (1.0, 5.0, 10.0):
        assert abs(sampled(result, "LOCAL_room_content_H2_leak", instant)) < TOL
        assert abs(sampled(result, "LOCAL_H2_leak_fed_out", instant) - LEAK) < TOL


def test_a_ring_with_no_volume_is_still_refused():
    """With no volume on it, nothing integrates across the ring and there
    is no edge to tear: the engine refuses it, as muscadet does (R30)."""

    class Pipe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="AIR")

    class Fan(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="AIR")
            self.add_flow_continuous_out(name="exhaust")
            self.add_rule_set(
                name="renewal",
                rules=[{"cons": {"AIR": 1.0}, "prod": {"exhaust": 1.0, "AIR": 1.0}}],
            )

    class Atmosphere(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="exhaust", var_demand_in_default=5.0)

    system = mu.System("capacity_ring_without_volume")
    system.add_component(Pipe, "PIPE")
    system.add_component(Fan, "FAN")
    system.add_component(Atmosphere, "ATMOSPHERE")
    system.connect("PIPE", "AIR", "FAN", "AIR")
    system.connect("FAN", "AIR", "PIPE", "AIR")
    system.connect("FAN", "exhaust", "ATMOSPHERE", "exhaust")
    with pytest.raises((ValueError, pyraichu.ModelError)) as raised:
        system.build_model()
    assert "cycle" in str(raised.value)


def fan_ring(prod_air: float, cons_air: float, exhaust: bool, off_ring: bool = False):
    """ROOM (a transiting volume of 100 holding 50) <-> FAN, the fan
    turning `cons_air` of room air (plus 10 of fresh air when it makes
    more than it draws) into `prod_air` of air blown back."""

    class Fresh(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="FRESH", var_fed_default=10.0)

    class Room(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="AIR")
            self.add_capacity(
                name="room", flow="AIR", capacity=100.0,
                content_init={"AIR": 50.0}, fill_rate=math.inf, transmits=True,
            )

    class Fan(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="AIR")
            cons = {"AIR": cons_air}
            prod = {"AIR": prod_air}
            if prod_air > cons_air:
                self.add_flow_continuous_in(name="FRESH")
                cons["FRESH"] = prod_air - cons_air
            if exhaust:
                self.add_flow_continuous_out(name="exhaust")
                prod["exhaust"] = 1.0
            self.add_rule_set(name="mix", rules=[{"cons": cons, "prod": prod}])

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="exhaust", var_demand_in_default=5.0)
            self.add_flow_continuous_in(name="AIR", var_demand_in_default=3.0)

    system = mu.System("capacity_ring_fan")
    system.add_component(Room, "ROOM")
    system.add_component(Fan, "FAN")
    system.connect("ROOM", "AIR", "FAN", "AIR")
    system.connect("FAN", "AIR", "ROOM", "AIR")
    if prod_air > cons_air:
        system.add_component(Fresh, "SRC")
        system.connect("SRC", "FRESH", "FAN", "FRESH")
    if exhaust or off_ring:
        system.add_component(Sink, "SINK")
    if exhaust:
        system.connect("FAN", "exhaust", "SINK", "exhaust")
    if off_ring:
        system.connect("FAN", "AIR", "SINK", "AIR")
    return system


@pytest.mark.parametrize(
    "shape, prod_air, cons_air, exhaust, off_ring",
    [
        # Nothing but the ring bounds the fan: torn, it would run at the
        # unbounded sentinel (1e30).
        ("pure recirculation", 1.0, 1.0, False, False),
        # The fan makes more air than it draws: the room could not take
        # the surplus and it would vanish on the torn edge.
        ("amplifying fan", 2.0, 1.0, True, False),
        # The return also feeds a consumer off the ring, whose demand
        # would stop bounding the fan.
        ("return read off the ring", 1.0, 1.0, True, True),
    ],
)
def test_a_ring_the_tear_cannot_conserve_is_still_refused(
    shape, prod_air, cons_air, exhaust, off_ring
):
    """The tear is taken only where the volume provably absorbs what the
    two ends disagree on; every other ring keeps the engine's refusal."""
    system = fan_ring(prod_air, cons_air, exhaust, off_ring)
    with pytest.raises((ValueError, pyraichu.ModelError)) as raised:
        system.build_model()
    assert "cycle" in str(raised.value), shape


def test_a_ventilated_room_clears_its_hydrogen_as_a_mixture():
    """The room holds 5 of hydrogen in 90 of air. The fan draws the room's
    atmosphere as a MIXTURE: what the leak brings passes first, and the
    rest of the 50 it asks is taken pro rata of the room's content, so the
    hydrogen decays as h - h1 + a ln(h/h1) = -(R - q)(t - 1), a time
    constant of a/(R - q), about 1.8 h, where it used to be drawn alone
    and gone within seconds. The exhaust stays at the atmosphere's 50: the
    air renewal is asked what the hydrogen extraction will actually make,
    not what it asked for (measured on the reference, h0 = 5)."""
    result = ventilated_room(hydrogen=5.0).simulate(
        t_max=5.0, samples=[1.0, 2.0, 3.0, 5.0]
    )
    rate = VENT_RATE - LEAK
    h1 = sampled(result, "LOCAL_room_content_H2_leak", 1.0)
    air = sampled(result, "LOCAL_room_content_AIR", 1.0)
    for instant in (2.0, 3.0, 5.0):
        h = sampled(result, "LOCAL_room_content_H2_leak", instant)
        assert abs(sampled(result, "LOCAL_room_content_AIR", instant) - air) < 1e-6
        # Both sides are of the order of 100: 5e-3 is a relative 5e-5,
        # the integrator's own tolerance on a logarithmic decay.
        assert abs((h - h1 + air * math.log(h / h1)) + rate * (instant - 1.0)) < 5e-3
        assert abs(sampled(result, "ATMOSPHERE_exhaust_fed_in", instant) - VENT_RATE) < 1e-6
    # The hydrogen clears on the hour scale, as a mixture does.
    assert 0.2 < sampled(result, "LOCAL_room_content_H2_leak", 5.0) < 0.5
