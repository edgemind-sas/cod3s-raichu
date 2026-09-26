"""A machine that moves a mixture at ONE volumetric rate (muscadet's
``add_mixture_in``, R51).

A ventilation extractor moves cubic metres of whatever is in the room, and
what leaves per constituent is not declarable separately: it is fixed by
the composition of the volume drawn from,

    out_f = R . m_f / sum_g (m_g . w_g)

``w_f`` being the volume one unit of ``f`` occupies, so the volume extracted,
``sum_f out_f . w_f``, is exactly ``R``. Two independent per-flow demands give
the model two degrees of freedom where the physics has one, which is why the
group is one declaration.

The closed forms are muscadet's own (``tests/test_mixture_ventilation_001``),
for a room of ``V`` receiving air at ``Q`` and hydrogen at ``q`` while one
extractor moves ``R = Q``::

    supply open   x(t) = q/(Q+q) . [1 - (V/(V+qt))^((Q+q)/q)],  M = V + q t
    supply cut    x(t) = x0 . exp(-R t / M0),                     M = M0
"""

from __future__ import annotations

import math

import pytest

import pyraichu.muscadet as mu
from conftest import sampled

V = 90.0
Q = 50.0
QH2 = 2.0
H2_PLATEAU = QH2 / (Q + QH2) * V
INSTANTS = [0.5, 1.0, 2.0, 4.0]
REL = 1e-6


def source(flow: str, rate: float) -> type[mu.ObjFlow]:
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=rate)

    return Source


def room(h2_init: float = 0.0, w_h2: float = 1.0) -> type[mu.ObjFlow]:
    class Room(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room",
                flows=[
                    {"name": "AIR", "weight": 1.0},
                    {"name": "H2", "weight": w_h2},
                ],
                side="out",
                capacity=1e5,
                content_init={"AIR": V, "H2": h2_init},
                fill_rate=math.inf,
                transmits=True,
            )

    return Room


def fan(rate: float = Q, flows=("AIR", "H2")) -> type[mu.ObjFlow]:
    class Fan(mu.ObjFlow):
        def add_flows(self):
            for flow in ("AIR", "H2"):
                self.add_flow_continuous_in(name=flow)
            self.add_mixture_in(name="extraction", flows=list(flows), flow_rate=rate)

    return Fan


def ventilated(supply=QH2, h2_init=0.0, w_h2=1.0, rate=Q) -> mu.System:
    system = mu.System("ventilated")
    system.add_component(source("AIR", Q), "S_AIR")
    system.add_component(source("H2", supply), "S_H2")
    system.add_component(room(h2_init, w_h2), "ROOM")
    system.add_component(fan(rate), "FAN")
    for flow in ("AIR", "H2"):
        system.connect(f"S_{flow}", flow, "ROOM", flow)
        system.connect("ROOM", flow, "FAN", flow)
    return system


def shares(result, instant):
    air = sampled(result, "ROOM_room_content_AIR", instant)
    h2 = sampled(result, "ROOM_room_content_H2", instant)
    return air, h2


def test_supply_open_the_share_follows_its_closed_form():
    result = ventilated().simulate(t_max=4.0, samples=INSTANTS)
    exponent = (Q + QH2) / QH2
    for t in INSTANTS:
        air, h2 = shares(result, t)
        total = V + QH2 * t
        x = QH2 / (Q + QH2) * (1.0 - (V / (V + QH2 * t)) ** exponent)
        assert math.isclose(air + h2, total, rel_tol=REL), t
        assert math.isclose(h2 / (air + h2), x, rel_tol=1e-5), t


def test_supply_cut_the_share_decays_exponentially():
    result = ventilated(supply=0.0, h2_init=H2_PLATEAU).simulate(
        t_max=4.0, samples=INSTANTS
    )
    m0 = V + H2_PLATEAU
    x0 = H2_PLATEAU / m0
    for t in INSTANTS:
        air, h2 = shares(result, t)
        assert math.isclose(air + h2, m0, rel_tol=REL), t
        assert math.isclose(
            h2 / (air + h2), x0 * math.exp(-Q * t / m0), rel_tol=1e-5
        ), t


def test_the_group_extracts_exactly_its_volumetric_rate_at_any_weight():
    """``sum_f out_f . w_f = R``: with a hydrogen twice as voluminous, the
    extractor takes fewer units of it and the volume moved is still R."""
    result = ventilated(supply=0.0, h2_init=H2_PLATEAU, w_h2=2.0).simulate(
        t_max=2.0, samples=[0.5, 1.0, 2.0]
    )
    for t in (0.5, 1.0, 2.0):
        air, h2 = shares(result, t)
        out_air = sampled(result, "FAN_AIR_fed_in", t)
        out_h2 = sampled(result, "FAN_H2_fed_in", t)
        assert math.isclose(out_air + 2.0 * out_h2, Q, rel_tol=1e-6), t
        assert math.isclose(out_h2, Q * h2 / (air + 2.0 * h2), rel_tol=1e-6), t


def test_the_draw_is_composed_and_not_what_merely_transits():
    """What arrives enters the composition: with the supply open, hydrogen
    leaves at the room's share and not at the rate it arrives."""
    result = ventilated().simulate(t_max=1.0, samples=[1.0])
    air, h2 = shares(result, 1.0)
    out_h2 = sampled(result, "FAN_H2_fed_in", 1.0)
    assert math.isclose(out_h2, Q * h2 / (air + h2), rel_tol=1e-6)
    assert out_h2 < QH2


def test_a_group_whose_flows_arrive_from_two_producers_is_refused():
    system = mu.System("split")
    system.add_component(source("AIR", Q), "S_AIR")
    system.add_component(source("H2", QH2), "S_H2")
    system.add_component(fan(), "FAN")
    system.connect("S_AIR", "AIR", "FAN", "AIR")
    system.connect("S_H2", "H2", "FAN", "H2")
    with pytest.raises(ValueError, match="one capacity"):
        system.build_dict()


def test_a_group_naming_an_undeclared_input_is_refused():
    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="AIR")
            self.add_mixture_in(name="g", flows=["AIR", "H2"], flow_rate=1.0)

    with pytest.raises(ValueError, match="H2"):
        mu.System("bad").add_component(Bad, "B")


def test_a_group_rate_that_is_not_a_finite_non_negative_volume_is_refused():
    with pytest.raises(ValueError, match="flow_rate"):
        mu.System("bad").add_component(fan(rate=-1.0), "F")
