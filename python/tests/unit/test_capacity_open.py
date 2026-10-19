"""A volume open to the atmosphere stays full and breathes (muscadet's R53,
5.17.0).

``makeup_flow`` names the held constituent the outside supplies. With
``D = sum_f w_f (in_f - out_f)``, the make-up enters at ``max(-D, 0) / w`` and
every constituent vents at ``max(D, 0) . m_f / sum_g m_g w_g``, so the volume
occupied never moves; both are published on ``{c}_makeup`` and
``{c}_vent_{f}``. What the declared content leaves of the volume is the
make-up at t=0. An open volume asks upstream for whatever arrives, so no
arrival is refused.

The montage is the platform showcase's room: 100 m^3 of air, a fan of
100 m^3/h, a natural draft of 1 m^3/h. The share settles at what arrives
over what leaves.
"""

from __future__ import annotations

import math

import pytest

import pyraichu.muscadet as mu
from conftest import sampled

VOLUME = 100.0
FAN = 100.0
DRAFT = 1.0
SEEP = 0.05
BREACH = 2.0
REL = 1e-3


def source(flow: str, rate: float) -> type[mu.ObjFlow]:
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=rate)

    return Source


def open_room(h2_init: float = 0.0, **extra) -> type[mu.ObjFlow]:
    class Room(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            settings = dict(
                name="room",
                flows=["AIR", "H2"],
                side="out",
                capacity=VOLUME,
                content_init={"H2": h2_init},
                transmits=True,
                makeup_flow="AIR",
                fill_rate=math.inf,
            )
            settings.update(extra)
            self.add_capacity(**settings)

    return Room


def machine(rate: float, ordered: bool = False) -> type[mu.ObjFlow]:
    class Machine(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
            command = None
            if ordered:
                self.add_flow_in(name="vent")
                command = [["vent"]]
            self.add_mixture_in(
                name="extraction", flows=["AIR", "H2"], flow_rate=rate, run_cond=command
            )

    return Machine


def showcase(leak: float, fan: float = FAN, h2_init: float = 0.0) -> mu.System:
    system = mu.System("open_room")
    system.add_component(source("H2", leak), "LEAK")
    system.add_component(open_room(h2_init), "ROOM")
    system.add_component(machine(fan), "FAN")
    system.add_component(machine(DRAFT), "DRAFT")
    system.connect("LEAK", "H2", "ROOM", "H2")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "FAN", flow)
        system.connect("ROOM", flow, "DRAFT", flow)
    return system


def share(result, instant):
    air = sampled(result, "ROOM_room_content_AIR", instant)
    h2 = sampled(result, "ROOM_room_content_H2", instant)
    return h2 / (air + h2), air + h2


def test_ae1_a_seep_settles_at_what_arrives_over_what_leaves():
    result = showcase(SEEP).simulate(t_max=20.0, samples=[20.0])
    x, occupied = share(result, 20.0)

    assert x == pytest.approx(SEEP / (FAN + DRAFT), rel=REL)
    assert occupied == pytest.approx(VOLUME, rel=1e-9)
    assert sampled(result, "ROOM_room_makeup", 20.0) == pytest.approx(
        FAN + DRAFT - SEEP, rel=1e-6
    )
    assert sampled(result, "ROOM_room_vent_H2", 20.0) == 0.0


def test_ae2_a_lost_fan_lets_the_share_rise_with_a_time_constant_of_100_h():
    start = SEEP / (FAN + DRAFT)
    crossing = -VOLUME / DRAFT * math.log((SEEP - 0.02) / (SEEP - start))
    dates = [30.0, crossing]
    result = showcase(SEEP, fan=0.0, h2_init=start * VOLUME).simulate(
        t_max=crossing + 1.0, samples=dates
    )

    x30, occupied = share(result, 30.0)
    assert x30 == pytest.approx(
        SEEP + (start - SEEP) * math.exp(-30.0 * DRAFT / VOLUME), rel=REL
    )
    assert occupied == pytest.approx(VOLUME, rel=1e-9)
    assert crossing == pytest.approx(50.1, abs=0.05)
    assert share(result, crossing)[0] == pytest.approx(0.02, rel=REL)


def test_ae3_a_breach_settles_at_the_trip_threshold():
    result = showcase(BREACH).simulate(t_max=30.0, samples=[30.0])
    assert share(result, 30.0)[0] == pytest.approx(BREACH / (FAN + DRAFT), rel=REL)


def test_ae4_a_breach_beyond_the_extraction_vents_and_is_never_refused():
    result = showcase(BREACH, fan=0.0).simulate(t_max=50.0, samples=[50.0])
    vent = sampled(result, "ROOM_room_vent_AIR", 50.0) + sampled(
        result, "ROOM_room_vent_H2", 50.0
    )

    assert sampled(result, "LEAK_H2_fed_out", 50.0) == pytest.approx(BREACH, rel=1e-6)
    assert vent == pytest.approx(BREACH - DRAFT, rel=1e-6)
    assert sampled(result, "ROOM_room_makeup", 50.0) == 0.0
    assert share(result, 50.0)[0] == pytest.approx(
        1.0 - math.exp(-50.0 * BREACH / VOLUME), rel=REL
    )


def test_ae5_a_fan_ordered_off_draws_nothing_while_the_draft_goes_on():
    class Switch(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="vent", var_prod_default=False)

    system = mu.System("ordered_off")
    system.add_component(source("H2", SEEP), "LEAK")
    system.add_component(open_room(), "ROOM")
    system.add_component(machine(FAN, ordered=True), "FAN")
    system.add_component(machine(DRAFT), "DRAFT")
    system.add_component(Switch, "SW")
    system.connect("LEAK", "H2", "ROOM", "H2")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "FAN", flow)
        system.connect("ROOM", flow, "DRAFT", flow)
    system.connect("SW", "vent", "FAN", "vent")

    result = system.simulate(t_max=1.0, samples=[1.0])
    fan = sampled(result, "FAN_AIR_fed_in", 1.0) + sampled(result, "FAN_H2_fed_in", 1.0)
    draft = sampled(result, "DRAFT_AIR_fed_in", 1.0) + sampled(
        result, "DRAFT_H2_fed_in", 1.0
    )

    assert fan == pytest.approx(0.0, abs=1e-12)
    assert draft == pytest.approx(DRAFT, rel=1e-6)


def test_the_initial_content_is_completed_by_the_makeup():
    (capacity,) = open_room()("ROOM").capacities
    assert capacity.content_init == {"AIR": VOLUME, "H2": 0.0}


def test_an_open_volume_refuses_a_makeup_it_does_not_hold():
    with pytest.raises(ValueError, match="does not hold"):
        open_room(makeup_flow="N2")("ROOM")


class Tank(mu.ObjFlow):
    """A stocked tank of hydrogen, closed, serving whatever it is asked."""

    def add_flows(self):
        self.add_flow_continuous_in(name="H2")
        self.add_flow_continuous_out(name="H2")
        self.add_capacity(
            name="tank", flow="H2", side="out", capacity=1000.0,
            content_init={"H2": 100.0}, transmits=True,
        )


def test_an_open_room_fed_by_a_tank_must_say_what_it_takes():
    """An open volume's claim follows ``fill_rate`` as for any volume, so the
    tank is never drained by a forced unbounded claim (100 to 36.8 in one
    time unit, measured in muscadet). At the default ``fill_rate=0`` a FED
    volume drawn as a mixture would ask only its composed share, none of a
    hydrogen it does not hold, so it is refused, as muscadet 5.18.0 refuses
    it: the model says what the room takes from the tank."""
    system = mu.System("open_tank")
    system.add_component(Tank, "TANK")
    system.add_component(open_room(fill_rate=0.0), "ROOM")
    system.add_component(machine(DRAFT), "DRAFT")
    system.connect("TANK", "H2", "ROOM", "H2")
    for flow in ("AIR", "H2"):
        system.connect("ROOM", flow, "DRAFT", flow)
    with pytest.raises(ValueError, match="fed on H2, with fill_rate=0"):
        system.simulate(t_max=10.0, samples=[10.0])


def test_weights_keep_the_occupied_volume_whatever_breathes():
    """Make-up (fan running) and venting (breach beyond the draft) with
    weights other than 1: the volume is held in VOLUME terms."""

    class WeightedRoom(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room",
                flows=[{"name": "AIR", "weight": 0.5}, {"name": "H2", "weight": 2.0}],
                side="out", capacity=VOLUME, content_init={"H2": 5.0},
                transmits=True, makeup_flow="AIR", fill_rate=math.inf,
            )

    for fan, leak in ((FAN, SEEP), (0.0, BREACH)):
        system = mu.System("weighted")
        system.add_component(source("H2", leak), "LEAK")
        system.add_component(WeightedRoom, "ROOM")
        system.add_component(machine(fan), "FAN")
        system.add_component(machine(DRAFT), "DRAFT")
        system.connect("LEAK", "H2", "ROOM", "H2")
        for flow in ("AIR", "H2"):
            system.connect("ROOM", flow, "FAN", flow)
            system.connect("ROOM", flow, "DRAFT", flow)
        result = system.simulate(t_max=20.0, samples=[20.0])
        air = sampled(result, "ROOM_room_content_AIR", 20.0)
        h2 = sampled(result, "ROOM_room_content_H2", 20.0)
        assert 0.5 * air + 2.0 * h2 == pytest.approx(VOLUME, rel=1e-6), (fan, leak)
