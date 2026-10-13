"""A stocked volume draining into a FULL one settles on its exact flows (#160).

The shape is a hydrogen leak path: a membrane holding a stock releases it, at
most at its ``serve_rate``, into a room that is full, and the room is emptied
by a ventilation drawing a fixed rate. A full volume admits only what leaves
it, and what leaves it is what arrives plus its stock's share of the rest, so
the admission reads back ``1 - s`` of itself, ``s`` being the share of the
admitted constituent in the room. The network has one fixed point, where the
membrane delivers exactly what the ventilation draws, but iterating onto it
contracts at ``1 - s`` per sweep: 0.9 at 10 % of hydrogen, and 0.9999 at
0.01 %. pyraichu 0.92.1 refused the 10 % case at t = 0 after 83 sweeps, under
a message blaming two alternating allocations; the resolution now measures
the rate and jumps to the limit.

The reference (muscadet 5.15.0 + PyCATSHOO 1.3.8.0, measured 2026-10-13 on the
10 % case) reads the room's outflow one evaluation late and so walks the same
geometric sequence over its first step: it delivers 0.0125 from t = 1 on and
its membrane holds 11.9869 at t = 1, 0.0006 below the closed form below, the
quantity its first step under-delivered. Every figure asserted here is a
closed form: the ventilation draws ``2 x 0.00625`` of hydrogen, so the
membrane delivers 0.0125 for good, empties linearly from 12, and the room
holds its volume.
"""

from __future__ import annotations

import math

import pyraichu
import pyraichu.muscadet as mu
import pytest
from conftest import sampled

DRAW = 0.0125
STOCK = 12.0
ROOM = 100.0
INSTANTS = (1.0, 5.0, 30.0)
#: The resolution settles to a relative 1e-9 per edge, and the levels
#: integrate that residual: well inside this.
TOL = 1e-7


def leak_chain(
    hydrogen: float = 10.0, serve_rate: float = 1.0, air: float | None = None
):
    """STACK_MEMBRANE -> LOCAL (full, `hydrogen` of H2 in a room of 100) <->
    VENTILATION -> ATMOSPHERE, as in the issue's four-component reduction.
    The room is full unless `air` leaves it short of its volume."""

    class Membrane(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2_membrane_leak")
            self.add_flow_continuous_out(name="H2_leak")
            self.add_capacity(
                name="membrane",
                flows=[{"name": "H2_membrane_leak", "weight": 1.0}],
                capacity=ROOM,
                content_init={"H2_membrane_leak": STOCK},
                fill_rate=math.inf,
                serve_rate=serve_rate,
                transmits=True,
                side="in",
            )
            self.add_rule_set(
                name="hydrogen_release",
                rules=[{"cons": {"H2_membrane_leak": 1.0}, "prod": {"H2_leak": 1.0}}],
            )

    class Room(mu.ObjFlow):
        def add_flows(self):
            for flow in ("H2_leak", "AIR"):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room",
                flows=[
                    {"name": "H2_leak", "weight": 1.0},
                    {"name": "AIR", "weight": 1.0},
                ],
                capacity=ROOM,
                content_init={
                    "H2_leak": hydrogen,
                    "AIR": ROOM - hydrogen if air is None else air,
                },
                fill_rate=math.inf,
                transmits=True,
                side="out",
            )

    class Ventilation(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2_leak")
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="draft")
            self.add_flow_continuous_out(name="AIR")
            self.add_rule_set(
                name="natural_extraction",
                rules=[{"cons": {"H2_leak": DRAW / 2}, "prod": {"draft": DRAW / 2}}],
            )
            self.add_rule_set(
                name="natural_renewal",
                rules=[
                    {
                        "cons": {"AIR": DRAW / 2},
                        "prod": {"draft": DRAW / 2, "AIR": DRAW / 2},
                    }
                ],
            )

    class Atmosphere(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="draft", var_demand_in_default=DRAW)

    system = mu.System("leak_chain")
    system.add_component(Membrane, "STACK_MEMBRANE")
    system.add_component(Room, "LOCAL")
    system.add_component(Ventilation, "VENTILATION")
    system.add_component(Atmosphere, "ATMOSPHERE")
    system.connect("STACK_MEMBRANE", "H2_leak", "LOCAL", "H2_leak")
    system.connect("LOCAL", "H2_leak", "VENTILATION", "H2_leak")
    system.connect("LOCAL", "AIR", "VENTILATION", "AIR")
    system.connect("VENTILATION", "AIR", "LOCAL", "AIR")
    system.connect("VENTILATION", "draft", "ATMOSPHERE", "draft")
    return system


def assert_drains_at_the_draw(result, hydrogen):
    for t in INSTANTS:
        assert abs(sampled(result, "STACK_MEMBRANE_H2_leak_fed_out", t) - DRAW) < TOL
        assert abs(sampled(result, "LOCAL_H2_leak_fed_out", t) - DRAW) < TOL
        assert abs(sampled(result, "ATMOSPHERE_draft_fed_in", t) - DRAW) < TOL
        membrane = sampled(
            result, "STACK_MEMBRANE_membrane_content_H2_membrane_leak", t
        )
        assert abs(membrane - (STOCK - DRAW * t)) < TOL
        assert abs(sampled(result, "LOCAL_room_content_H2_leak", t) - hydrogen) < TOL
        assert abs(sampled(result, "LOCAL_room_content", t) - ROOM) < TOL


@pytest.mark.parametrize("hydrogen", [10.0, 1.0, 0.01])
def test_a_stock_draining_into_a_full_room_settles_on_the_draw(hydrogen):
    """At 10 %, 1 % and 0.01 % of hydrogen the loop contracts at 0.9, 0.99 and
    0.9999 per sweep: the plain iteration needs about 200, 2 000 and 200 000
    sweeps, so only the extrapolation holds all three."""
    result = leak_chain(hydrogen).simulate(t_max=30.0, samples=list(INSTANTS))
    assert_drains_at_the_draw(result, hydrogen)


def test_a_campaign_settles_the_same_network():
    """The issue's campaign route: one replica failing used to discard them
    all, so a campaign is pinned on its own."""
    result = leak_chain().monte_carlo(nb_runs=2, t_max=30.0, samples=list(INSTANTS))
    assert result is not None


def test_a_room_with_room_left_still_fills_then_holds():
    """The variant both engines already settled: at 9.99 of hydrogen the room
    is short of 0.01, which the unbounded fill claims at once, and the
    membrane then delivers the draw like the full case."""
    result = leak_chain(hydrogen=9.99, air=90.0).simulate(
        t_max=30.0, samples=list(INSTANTS)
    )
    for t in INSTANTS:
        assert abs(sampled(result, "STACK_MEMBRANE_H2_leak_fed_out", t) - DRAW) < TOL
        assert abs(sampled(result, "LOCAL_room_content", t) - ROOM) < 1e-6


def test_an_unbounded_stock_into_a_full_room_drains_at_what_leaves_it():
    """Deliberate divergence from the reference (issue #160): with no
    `serve_rate` the membrane could serve without bound, but a full room
    admits only what leaves it, so the transfer is bounded by the room. The
    reference refuses it only after its lagged read drops the room below
    full; RAICHU keeps the physical answer, the room holding its volume."""
    result = leak_chain(serve_rate=math.inf).simulate(
        t_max=30.0, samples=list(INSTANTS)
    )
    assert_drains_at_the_draw(result, 10.0)


def test_an_unbounded_stock_into_a_room_with_room_left_is_refused():
    """The room's infinite fill claims without bound and the membrane serves
    without bound: an instantaneous transfer, refused by both engines."""
    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        leak_chain(hydrogen=9.99, serve_rate=math.inf, air=90.0).simulate(
            t_max=30.0, samples=list(INSTANTS)
        )
    assert caught.value.variable == "STACK_MEMBRANE.membrane_content_H2_membrane_leak"
