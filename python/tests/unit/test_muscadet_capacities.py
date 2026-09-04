"""Capacities in the `pyraichu.muscadet` authoring layer.

A capacity is a volume a component holds over one or more of its
continuous flows: it integrates what enters minus what leaves, and it
stays inside its bounds.

What the generated model must hold, and what these tests pin:

- one ODE content per held flow plus a total, and the explicit weighted
  fills that report them;
- the two bounds as **automaton locations** entered by watched
  transitions, so a crossing is located rather than stepped over and the
  derivative never has to re-decide saturation for itself;
- the two bound effects: a full capacity asks upstream only for what it
  can still take, an empty one serves downstream only what currently
  transits through it;
- a declared hysteresis width, whose whole job is to bound the number of
  integration segments a capacity riding its bound costs;
- the read-only publication of the level and the fill: a reader receives
  a number and exchanges no quantity.
"""

import math

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, TOL, at_zero, fired_at, sampled


class Source(mu.ObjFlow):
    """A pure producer of 5 units of `water` per unit time."""

    def add_flows(self):
        self.add_flow_continuous_out(name="water", var_fed_default=5.0)


class Tank(mu.ObjFlow):
    """A tank of volume 100, filling as fast as it is served."""

    def add_flows(self):
        self.add_flow_continuous_in(name="water")
        self.add_capacity(name="reservoir", flow="water", capacity=100.0, fill_rate=10.0)


def filling_tank() -> mu.System:
    system = mu.System("capacity_filling")
    system.add_component(Source, "S")
    system.add_component(Tank, "T")
    system.connect("S", "water", "T", "water")
    return system


# --- the bound is reached where the closed form says it is -------------


def test_a_tank_fills_at_the_delivered_rate_and_reaches_its_bound_on_time():
    """Content is `min(rate * t, volume)`: asserted against the closed
    form, and the bound crossing against `volume / rate`."""
    rate, volume = 5.0, 100.0
    expected = volume / rate
    result = filling_tank().simulate(
        t_max=30.0, samples=[0.0, 5.0, 10.0, expected, 25.0]
    )

    for instant in (0.0, 5.0, 10.0):
        # Ordinary flow resolution and ODE integration, well before the
        # bound at t=20: not tied to a located crossing, so TOL, not
        # CROSSING_TOL.
        assert abs(sampled(result, "T_reservoir_content", instant) - rate * instant) < (
            TOL
        ), instant
    # Past the bound the closed form is the volume, not the ramp.
    assert abs(sampled(result, "T_reservoir_content", 25.0) - volume) < CROSSING_TOL
    assert abs(fired_at(result, "reservoir_reach_full") - expected) < CROSSING_TOL


def test_the_fill_is_the_content_over_the_volume():
    """The reported fill is the content scaled by the volume, at every
    sampled instant."""
    result = filling_tank().simulate(t_max=30.0, samples=[0.0, 7.0, 25.0])
    for instant in (0.0, 7.0, 25.0):
        content = sampled(result, "T_reservoir_content", instant)
        assert abs(sampled(result, "T_reservoir_fill", instant) - content / 100.0) < TOL


# --- the full bound throttles the demand it carries upstream ------------


def test_at_the_bound_the_content_stops_and_the_demand_falls():
    """R7: a full capacity asks upstream only for what it can still take,
    which is nothing at all when nothing leaves it."""
    result = filling_tank().simulate(t_max=30.0, samples=[10.0, 25.0])

    # Before the bound the tank asks for its fill rate and the producer
    # serves what it can.
    assert abs(sampled(result, "T_water_demand_in", 10.0) - 10.0) < TOL
    assert abs(sampled(result, "T_water_fed_in", 10.0) - 5.0) < TOL

    # At the bound the demand collapses and the producer delivers nothing.
    assert abs(sampled(result, "T_water_demand_in", 25.0)) < TOL
    assert abs(sampled(result, "T_water_fed_in", 25.0)) < TOL
    assert abs(sampled(result, "S_water_fed_out", 25.0)) < TOL
    assert abs(sampled(result, "T_reservoir_content", 25.0) - 100.0) < CROSSING_TOL


# --- the empty bound throttles what the capacity delivers --------------


def test_a_draining_tank_is_clamped_and_the_delivery_falls_with_it():
    """A capacity cannot serve what it does not hold: once empty its
    output falls to what currently transits through it, which is nothing
    when nothing feeds it, and the content stops at zero."""

    class Drum(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="water", var_fed_default=5.0)
            self.add_capacity(
                name="drum",
                flow="water",
                capacity=100.0,
                content_init={"water": 50.0},
            )

    class Draw(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water", var_demand_in_default=5.0)

    system = mu.System("capacity_draining")
    system.add_component(Drum, "D")
    system.add_component(Draw, "C")
    system.connect("D", "water", "C", "water")
    result = system.simulate(t_max=20.0, samples=[0.0, 5.0, 10.0, 15.0, 19.0])

    # Ordinary ODE integration, well before the bound at t=10: TOL, not
    # CROSSING_TOL.
    assert abs(sampled(result, "D_drum_content", 5.0) - 25.0) < TOL
    assert abs(fired_at(result, "drum_reach_empty") - 10.0) < CROSSING_TOL

    # Clamped: the content stops at zero and stays there. What is left
    # below it is the drain rate times the event-location tolerance
    # (5 x 1e-10 here), the residual of locating the crossing, and not
    # something the capacity keeps draining.
    drained = sampled(result, "D_drum_content", 15.0)
    assert abs(drained) < CROSSING_TOL, drained
    assert drained > -5.0 * 1e-9, "the content kept draining below zero"
    assert drained == sampled(result, "D_drum_content", 19.0), drained
    assert abs(sampled(result, "C_water_fed_in", 15.0)) < TOL
    assert abs(sampled(result, "D_water_capability_out", 15.0)) < TOL


# --- riding the bound costs a bounded number of segments ---------------


#: What the capacity's pass-through demand sits under what it delivers in
#: the bound-riding scenario. An *exactly* matched inflow cancels bit for
#: bit and costs nothing at any hysteresis width, which would make the
#: measurement vacuous; the flow resolution promises only its tolerance
#: (`1e-9` relative), so the honest stress is an inflow matched to that
#: promise and no better.
RESIDUAL = 5e-9


def bound_rider(hysteresis: float | None = None) -> mu.System:
    """A tank that fills, reaches its bound, and is then held there by an
    inflow matched to what it delivers downstream.

    Ten units are available and five are drawn, so the tank gains five per
    unit time until t=20 and, from there on, as near to nothing as the
    flow resolution can promise: the bound is ridden for the remaining
    980 units of simulated time rather than crossed once."""
    extra = {} if hysteresis is None else {"hysteresis": hysteresis}

    class Supply(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="water", var_fed_default=10.0)

    class Buffer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(
                name="water", var_demand_in_default=5.0 - RESIDUAL
            )
            self.add_flow_continuous_out(name="water", var_fed_default=5.0)
            self.add_capacity(
                name="buffer",
                flow="water",
                capacity=100.0,
                fill_rate=5.0,
                **extra,
            )

    class Draw(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water", var_demand_in_default=5.0)

    system = mu.System("capacity_bound_riding")
    system.add_component(Supply, "S")
    system.add_component(Buffer, "B")
    system.add_component(Draw, "C")
    system.connect("S", "water", "B", "water")
    system.connect("B", "water", "C", "water")
    return system


#: What riding the bound for 1000 units of simulated time may cost. The
#: shipped width measures 3 (one segment to the bound, one restart at the
#: crossing, one to the horizon) against about 10 000 solver steps, so the
#: budget is that count with room for a few restarts rather than anything
#: proportional to the horizon. See `pyraichu.muscadet.DEFAULT_HYSTERESIS`
#: for the width-by-width measurement this budget comes from.
SEGMENT_BUDGET = 8

#: What a location held past the physical crossing costs on the held
#: quantity: at most the hysteresis width times the volume.
INDUCED_ERROR = mu.DEFAULT_HYSTERESIS * 100.0


def test_riding_the_bound_costs_a_bounded_number_of_segments():
    """KTD6: every bound entry and exit ends an integration segment, so a
    capacity chattering on its bound would drive the segment count
    towards the step count. The hysteresis width is what stops it, and
    this is the measurement that sizes it."""
    result = bound_rider().simulate(t_max=1000.0, samples=[999.0])
    assert result.work["segments"] <= SEGMENT_BUDGET, result.work
    # Still at its bound, to within the error the band itself induces.
    held = sampled(result, "B_buffer_content", 999.0)
    assert 100.0 - INDUCED_ERROR <= held <= 100.0 + CROSSING_TOL, held
    # And the flow across it is exact inside the location: what is drawn
    # downstream is served in full for the whole horizon.
    assert abs(sampled(result, "C_water_fed_in", 999.0) - 5.0) < TOL


def test_a_narrower_hysteresis_is_what_the_budget_pays_for():
    """The width is a parameter, not an intention: driven below the flow
    resolution's own noise floor it stops separating the two bound
    decisions, and the segment count grows towards the step count.
    Asserting the shipped width beats a degenerate one by orders of
    magnitude is what makes the default a measurement rather than a
    taste."""
    wide = bound_rider().simulate(t_max=1000.0)
    assert wide.work["segments"] <= SEGMENT_BUDGET

    # A band of exactly zero is not a slow model, it is a broken one:
    # the two bound decisions collapse onto the same threshold and the
    # inclusive guards both hold at once. The engine names that rather
    # than grinding through it.
    with pytest.raises(pyraichu.SimulationError, match="boundary loop"):
        bound_rider(hysteresis=0.0).simulate(t_max=1000.0)

    # Just above zero the band separates the decisions again, and the
    # cost is what makes the default a measurement rather than a taste:
    # 21053 segments at 1e-12 against 3 at the shipped width.
    narrow = bound_rider(hysteresis=1e-12).simulate(t_max=1000.0)
    assert narrow.work["segments"] > 100 * wide.work["segments"], (
        wide.work,
        narrow.work,
    )
    # Without a usable band the segment count tracks the step count,
    # which is exactly the cost KTD6 refuses to pay.
    assert narrow.work["segments"] > 0.5 * narrow.work["solver_steps_accepted"]


# --- several held flows share one volume -------------------------------


def test_two_flows_with_different_weights_report_a_consistent_fill():
    """One volume holding two flows: each unit of `additive` occupies two
    units of volume, so the total fill is the weighted sum and the bound
    is reached on the weighted rate."""

    class Vessel(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_flow_continuous_in(name="additive")
            self.add_capacity(
                name="vessel",
                flows=[
                    {"name": "water", "weight": 1.0},
                    {"name": "additive", "weight": 2.0},
                ],
                capacity=100.0,
                content_init={"water": 10.0, "additive": 20.0},
                fill_rate=10.0,
            )

    class Feed(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="water", var_fed_default=1.0)
            self.add_flow_continuous_out(name="additive", var_fed_default=1.0)

    system = mu.System("capacity_two_flows")
    system.add_component(Feed, "F")
    system.add_component(Vessel, "V")
    system.connect("F", "water", "V", "water")
    system.connect("F", "additive", "V", "additive")

    # Fill starts at (10 * 1 + 20 * 2) / 100 and climbs at (1 + 2) / 100.
    expected = (1.0 - 0.5) / 0.03
    result = system.simulate(t_max=30.0, samples=[0.0, expected + 5.0])

    assert abs(at_zero(result, "V_vessel_fill_water") - 0.1) < TOL
    assert abs(at_zero(result, "V_vessel_fill_additive") - 0.4) < TOL
    assert abs(at_zero(result, "V_vessel_fill") - 0.5) < TOL
    assert abs(at_zero(result, "V_vessel_content") - 30.0) < TOL
    assert abs(fired_at(result, "vessel_reach_full") - expected) < CROSSING_TOL

    # At the bound the weighted fill is one, which neither constituent
    # reaches on its own: that is what "consistent with both" means.
    saturated = sampled(result, "V_vessel_fill", expected + 5.0)
    assert abs(saturated - 1.0) < CROSSING_TOL
    water = sampled(result, "V_vessel_fill_water", expected + 5.0)
    additive = sampled(result, "V_vessel_fill_additive", expected + 5.0)
    assert abs(water + additive - saturated) < TOL
    assert water < 1.0 and additive < 1.0


# --- the level is published read-only ----------------------------------


class Gauge(mu.ObjFlow):
    """An instrument reading a published level, and nothing else."""

    def add_flows(self):
        self.add_measurement_in(name="reservoir")


def observed_tank() -> mu.System:
    system = filling_tank()
    system.name = "capacity_observed"
    system.add_component(Gauge, "G")
    system.connect_measurement("T", "reservoir", "G")
    return system


def test_a_reader_receives_the_published_level():
    """R7: the observer reads the level and the fill of the volume it
    watches."""
    result = observed_tank().simulate(t_max=30.0, samples=[10.0, 25.0])
    for instant in (10.0, 25.0):
        content = sampled(result, "T_reservoir_content", instant)
        assert abs(sampled(result, "G_reservoir_level", instant) - content) < TOL
        assert abs(sampled(result, "G_reservoir_fill", instant) - content / 100.0) < TOL


def test_a_reader_exchanges_no_quantity():
    """The measurement link carries no quantity: the observer creates no
    demand, enters no allocation operator, and leaves the flow answer
    exactly as it was without it."""
    document = pyraichu.model_body(observed_tank().build_dict())
    gauge = next(c for c in document["components"] if c["name"] == "G")
    assert gauge["ports"] and all(port["dir"] == "in" for port in gauge["ports"])
    # No channel is declared or read on the observer's side.
    assert not any("channels" in port for port in gauge["ports"])
    assert not gauge.get("allocations")

    for component in document["components"]:
        for allocation in component.get("allocations", []):
            consumers = [
                param["to"]["component"]
                for key in ("shares", "priorities")
                for param in allocation.get(key, [])
            ]
            assert "G" not in consumers, allocation
    # No per-edge channel attribute is materialised towards the observer.
    for connection in document["connections"]:
        if connection["to"]["component"] == "G":
            assert "name" not in connection or "alloc" not in connection["name"]

    watched = observed_tank().simulate(t_max=30.0, samples=[10.0, 25.0])
    alone = filling_tank().simulate(t_max=30.0, samples=[10.0, 25.0])
    for instant in (10.0, 25.0):
        assert sampled(watched, "T_reservoir_content", instant) == sampled(
            alone, "T_reservoir_content", instant
        )
        assert sampled(watched, "S_water_fed_out", instant) == sampled(
            alone, "S_water_fed_out", instant
        )


# --- generated-model shape ---------------------------------------------


def test_the_bounds_are_watched_transitions_not_a_branch_in_the_derivative():
    """KTD6: re-deciding saturation inside the derivative makes it
    discontinuous without telling the solver, so the bounds are automaton
    locations entered by located crossings."""
    document = pyraichu.model_body(filling_tank().build_dict())
    tank = next(c for c in document["components"] if c["name"] == "T")
    bounds = next(a for a in tank["automata"] if a["name"] == "reservoir_bounds")
    assert [state for state in bounds["states"]] == ["empty", "partial", "full"]
    assert bounds["init"] == "empty"
    assert all(t["distrib"] == "watched" for t in bounds["transitions"])

    ode = [e for e in tank["equations"] if e["kind"] == "ode"]
    assert [e["target"] for e in ode] == ["reservoir_content_water"]
    # The derivative is what enters minus what leaves, with no branch.
    assert "if" not in str(ode[0]["expr"])


def test_the_capacity_steps_join_the_evaluation_order_exactly_once():
    """Every explicit equation the capacity adds is swept, before the
    flow bands that report on it, and the order stays a permutation."""
    document = observed_tank().build_dict()
    order = [
        (step["component"], step["attribute"])
        for step in pyraichu.model_body(document)["evaluation_order"]
    ]
    assert len(order) == len(set(order))
    for step in (
        ("T", "reservoir_content"),
        ("T", "reservoir_fill"),
        ("T", "reservoir_fill_water"),
        ("G", "reservoir_level"),
        ("G", "reservoir_fill"),
    ):
        assert step in order, step
    assert order.index(("T", "reservoir_fill_water")) < order.index(
        ("T", "reservoir_fill")
    )
    assert order.index(("T", "reservoir_fill")) < order.index(("G", "reservoir_fill"))
    assert order.index(("T", "reservoir_content")) < order.index(
        ("S", "water_capability_out")
    )
    pyraichu.load_model(document)


def test_capacity_and_measurement_variables_are_indicators():
    """A level generated and never observed otherwise is of no use: the
    capacity's own variables carry indicators like the flow channels."""
    document = observed_tank().build_dict()
    names = {i["name"] for i in pyraichu.model_body(document)["indicators"]}
    assert {
        "T_reservoir_content",
        "T_reservoir_content_water",
        "T_reservoir_fill",
        "T_reservoir_fill_water",
        "G_reservoir_level",
        "G_reservoir_fill",
    } <= names


def test_a_system_without_a_capacity_carries_no_capacity_material():
    """Regression: a continuous model that declares no capacity gains
    neither an automaton, nor an ODE, nor a level variable."""

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water", var_demand_in_default=4.0)

    system = mu.System("capacity_absent")
    system.add_component(Source, "S")
    system.add_component(Consumer, "C")
    system.connect("S", "water", "C", "water")
    document = pyraichu.model_body(system.build_dict())
    for component in document["components"]:
        assert not component["automata"], component["name"]
        assert all(e["kind"] == "explicit" for e in component["equations"])
        assert not any(
            "_content" in a["name"] or "_fill" in a["name"]
            for a in component["attributes"]
        ), component["name"]


# --- refusals -----------------------------------------------------------


def test_a_capacity_must_hold_a_continuous_flow():
    """A capacity over a flow the component does not carry continuously
    is refused, naming both."""

    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="water")
            self.add_capacity(name="tank", flow="water", capacity=10.0)

    with pytest.raises(ValueError, match="continuous flow"):
        Bad("B")


def test_a_capacity_needs_a_strictly_positive_volume():
    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_capacity(name="tank", flow="water", capacity=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        Bad("B")


def test_two_capacities_cannot_hold_the_same_flow():
    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_capacity(name="a", flow="water", capacity=10.0)
            self.add_capacity(name="b", flow="water", capacity=10.0)

    with pytest.raises(ValueError, match="already held"):
        Bad("B")


def test_content_init_naming_an_unheld_flow_is_refused():
    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_capacity(
                name="tank",
                flow="water",
                capacity=10.0,
                content_init={"steam": 1.0},
            )

    with pytest.raises(ValueError, match="content_init"):
        Bad("B")


def test_a_side_the_held_flow_does_not_carry_is_refused():
    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_capacity(name="tank", flow="water", capacity=10.0, side="out")

    with pytest.raises(ValueError, match="side"):
        Bad("B")


def test_a_negative_fill_rate_is_refused():
    class Bad(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_capacity(name="tank", flow="water", capacity=10.0, fill_rate=-1.0)

    with pytest.raises(ValueError, match="fill rate"):
        Bad("B")


def test_a_measurement_reading_an_unheld_constituent_is_refused():
    """The observer names a constituent the volume does not hold: refused
    at connection, naming what it does hold."""

    class Probe(mu.ObjFlow):
        def add_flows(self):
            self.add_measurement_in(name="reservoir", flows=["steam"])

    system = filling_tank()
    system.add_component(Probe, "P")
    with pytest.raises(ValueError, match="does not hold"):
        system.connect_measurement("T", "reservoir", "P")


def test_an_unbounded_fill_rate_asks_for_the_whole_capability():
    """muscadet's `math.inf` fill rate: whatever the producer can
    deliver, which is the published capability rather than a number the
    document cannot carry."""

    class Greedy(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_capacity(
                name="tank", flow="water", capacity=100.0, fill_rate=math.inf
            )

    system = mu.System("capacity_unbounded_claim")
    system.add_component(Source, "S")
    system.add_component(Greedy, "T")
    system.connect("S", "water", "T", "water")
    result = system.simulate(t_max=10.0, samples=[5.0])
    assert abs(sampled(result, "T_water_demand_in", 5.0) - 5.0) < TOL
    # The bound (capacity 100 at rate 5) is not reached within t_max=10:
    # ordinary ODE integration, TOL not CROSSING_TOL.
    assert abs(sampled(result, "T_tank_content", 5.0) - 25.0) < TOL
