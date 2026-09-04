"""Transfers, time profiles and deratings in the `pyraichu.muscadet`
authoring layer.

Three mechanisms a continuous output carries beside its rate, and the
refusals that keep them declarative:

- a **transfer pair**, a signed quantity a component moves between two
  flows because a gradient drives it, in the two documented shapes: the
  two-stream exchange, which adds the signed delta on top of what both
  streams carry, and the metered conduit, which *replaces* what crosses
  the component with the computed quantity;
- a **time profile**, a declared continuous function of the simulation
  clock multiplying what an output produces;
- **deratings**, what the failure modes bearing on a continuous output
  leave of its rate, folded by minimum with the shared
  `{flow}_out_rate`, released implicitly on the opposite state.

What these tests pin, beyond the arithmetic:

- only the conductive law and the sinusoidal profile are declarable, over
  muscadet's own two potential operands (a constant and a measurement
  reading). A bespoke Python function, in either place, is refused by
  name rather than approximated;
- a derating and a profile compose by **product**, two deratings by
  **minimum**: a source at 30 % of its profile that is also derated to
  0.5 produces 15 %, and two modes leaving 0.5 and 0.8 leave 0.5;
- a rate of zero is a total loss of production, there being no separate
  boolean gate on a continuous flow;
- a model declaring none of the three is generated exactly as before.

The counter-flow exchanger here reads its two inlet temperatures over
**measurement channels**, which is the shape the conductive law can
carry. The literature model reads them as an input delivery divided by a
constant, which is neither of muscadet's two operand forms: that shape is
refused, and `test_a_potential_read_off_a_delivered_input_is_refused`
pins the refusal rather than widening the vocabulary.
"""

import math

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, TOL, at_zero, sampled

# --- a thermal probe, so a potential has somewhere to come from --------


def probe(temperature: float) -> type:
    """A component publishing a constant level, the only declarable
    source of a potential.

    A capacity holding a flow declared on the input side only integrates
    nothing, so its content stays where it started: that is a fixed
    temperature an exchanger can read over a measurement channel."""

    class Probe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="celsius")
            self.add_capacity(
                name="probe",
                flow="celsius",
                capacity=1000.0,
                content_init={"celsius": temperature},
            )

    return Probe


class Load(mu.ObjFlow):
    """A consumer asking for far more than it will be given, so what it
    receives is what was produced."""

    def add_flows(self):
        self.add_flow_continuous_in(name="heat_hot", var_demand_in_default=1e4)
        self.add_flow_continuous_in(name="heat_cold", var_demand_in_default=1e4)


# --- the two-stream exchange -------------------------------------------

#: The counter-flow relation `Q = effectiveness x C_min x (T_hot - T_cold)`,
#: split into the conductance the law carries and the two potentials.
EFFECTIVENESS = 0.8
C_MIN = 2.5
HOT_STREAM = 200.0
COLD_STREAM = 50.0


def exchanger_system(t_hot: float, t_cold: float) -> mu.System:
    """A counter-flow exchanger between two streams, driven by the
    inlet temperatures two probes publish.

    Each stream's carried heat is declared as its output rate: this layer
    has no identity transfer, so what a stream carries across the
    component is what its output was declared with, and the pair adds its
    signed delta on top of both."""

    class Exchanger(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="heat_hot", var_fed_default=HOT_STREAM)
            self.add_flow_continuous_out(name="heat_cold", var_fed_default=COLD_STREAM)
            self.add_measurement_in(name="hot")
            self.add_measurement_in(name="cold")
            self.add_transfer(
                name="exchange",
                flows=("heat_hot", "heat_cold"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": EFFECTIVENESS * C_MIN,
                    "potential_a": {"measurement": "hot"},
                    "potential_b": {"measurement": "cold"},
                },
            )

    system = mu.System("counter_flow_exchanger")
    system.add_component(probe(t_hot), "TH")
    system.add_component(probe(t_cold), "TC")
    system.add_component(Exchanger, "X")
    system.add_component(Load, "L")
    system.connect_measurement("TH", "probe", "X", "hot")
    system.connect_measurement("TC", "probe", "X", "cold")
    system.connect("X", "heat_hot", "L", "heat_hot")
    system.connect("X", "heat_cold", "L", "heat_cold")
    return system


def test_a_counter_flow_exchanger_moves_the_effectiveness_quantity():
    """`Q = effectiveness x C_min x (T_hot - T_cold)`: the hot stream is
    relieved of exactly what the cold one receives, and the raw balance
    the pair guarantees is untouched."""
    t_hot, t_cold = 80.0, 20.0
    expected = EFFECTIVENESS * C_MIN * (t_hot - t_cold)
    result = exchanger_system(t_hot, t_cold).simulate(t_max=1.0)

    assert abs(at_zero(result, "X_exchange_requested") - expected) < TOL
    assert abs(at_zero(result, "X_exchange_moved") - expected) < TOL
    assert abs(at_zero(result, "L_heat_hot_fed_in") - (HOT_STREAM - expected)) < TOL
    assert abs(at_zero(result, "L_heat_cold_fed_in") - (COLD_STREAM + expected)) < TOL
    # The raw balance: a pair moves, it neither creates nor destroys.
    assert (
        abs(
            at_zero(result, "L_heat_hot_fed_in")
            + at_zero(result, "L_heat_cold_fed_in")
            - (HOT_STREAM + COLD_STREAM)
        )
        < TOL
    )


def test_the_sign_is_the_direction():
    """The equation returns a signed quantity and the layer routes it: a
    reversed gradient moves the quantity the other way, with no direction
    clamp anywhere in the declaration."""
    t_hot, t_cold = 20.0, 40.0
    expected = EFFECTIVENESS * C_MIN * (t_hot - t_cold)
    assert -COLD_STREAM < expected < 0.0
    result = exchanger_system(t_hot, t_cold).simulate(t_max=1.0)

    assert abs(at_zero(result, "X_exchange_moved") - expected) < TOL
    assert abs(at_zero(result, "L_heat_hot_fed_in") - (HOT_STREAM - expected)) < TOL
    assert abs(at_zero(result, "L_heat_cold_fed_in") - (COLD_STREAM + expected)) < TOL


def test_a_stream_is_not_relieved_of_more_than_it_carries():
    """The moved magnitude is capped by what the origin stream carries,
    so a large gradient empties the hot stream and never drives it
    negative."""
    result = exchanger_system(1000.0, 0.0).simulate(t_max=1.0)

    assert at_zero(result, "X_exchange_requested") > HOT_STREAM
    assert abs(at_zero(result, "X_exchange_moved") - HOT_STREAM) < TOL
    assert abs(at_zero(result, "L_heat_hot_fed_in")) < TOL
    assert abs(at_zero(result, "L_heat_cold_fed_in") - (COLD_STREAM + HOT_STREAM)) < TOL


# --- the metered conduit ------------------------------------------------


def conduit_system(supply: float, t_in: float = 30.0, t_out: float = 5.0) -> mu.System:
    """A wall metering the heat that crosses it, fed by a supply of
    `supply` and read by a load asking for far more.

    The wall's output is declared with a large rate on purpose: a conduit
    REPLACES what the flow would otherwise carry, so that rate must not
    reach the consumer."""

    class Supply(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="heat", var_fed_default=supply)

    class Wall(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat")
            self.add_flow_continuous_out(name="heat", var_fed_default=1000.0)
            self.add_measurement_in(name="inside")
            self.add_measurement_in(name="outside")
            self.add_transfer(
                name="wall",
                flows=("heat", "heat"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 2.0,
                    "potential_a": {"measurement": "inside"},
                    "potential_b": {"measurement": "outside"},
                },
            )

    class Room(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat", var_demand_in_default=1e4)

    system = mu.System("metered_conduit")
    system.add_component(probe(t_in), "TI")
    system.add_component(probe(t_out), "TO")
    system.add_component(Supply, "S")
    system.add_component(Wall, "W")
    system.add_component(Room, "R")
    system.connect_measurement("TI", "probe", "W", "inside")
    system.connect_measurement("TO", "probe", "W", "outside")
    system.connect("S", "heat", "W", "heat")
    system.connect("W", "heat", "R", "heat")
    return system


def test_a_metered_conduit_replaces_what_the_flow_would_carry():
    """`G x (T_in - T_out)` crosses the wall, not the 1000 the output was
    declared with and not the 500 the supply could deliver."""
    expected = 2.0 * (30.0 - 5.0)
    result = conduit_system(supply=500.0).simulate(t_max=1.0)

    assert abs(at_zero(result, "W_wall_requested") - expected) < TOL
    assert abs(at_zero(result, "W_heat_capability_out") - expected) < TOL
    assert abs(at_zero(result, "R_heat_fed_in") - expected) < TOL
    assert abs(at_zero(result, "W_wall_moved") - expected) < TOL


def test_a_conduit_asks_upstream_for_what_it_is_about_to_move():
    """A conduit replaced its flow's transit, so nothing else claims that
    input: the demand it carries upstream is its own computed quantity,
    and the supply delivers exactly that."""
    expected = 2.0 * (30.0 - 5.0)
    result = conduit_system(supply=500.0).simulate(t_max=1.0)

    assert abs(at_zero(result, "W_heat_demand_in") - expected) < TOL
    assert abs(at_zero(result, "S_heat_fed_out") - expected) < TOL


def test_a_saturated_conduit_shows_its_shortfall():
    """The supply cannot cover the computed quantity: what was asked for
    stays readable beside what actually crossed."""
    result = conduit_system(supply=20.0).simulate(t_max=1.0)

    assert abs(at_zero(result, "W_wall_requested") - 50.0) < TOL
    assert abs(at_zero(result, "W_wall_moved") - 20.0) < TOL
    assert abs(at_zero(result, "R_heat_fed_in") - 20.0) < TOL


def test_a_reversed_conduit_crosses_nothing():
    """A conduit's direction is its connection's: a computed reversal
    crosses nothing rather than warming what the law is cooling, and the
    negative ask stays readable."""
    result = conduit_system(supply=500.0, t_in=5.0, t_out=30.0).simulate(t_max=1.0)

    assert abs(at_zero(result, "W_wall_requested") + 50.0) < TOL
    assert abs(at_zero(result, "R_heat_fed_in")) < TOL


# --- what stays out of the declaration vocabulary (R8) ------------------


def test_a_bespoke_transfer_function_is_refused():
    """A Python callable carries no continuity attestation and no
    serializable form: it is refused, naming the component and the
    transfer."""

    class Bespoke(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat")
            self.add_flow_continuous_out(name="heat")
            self.add_transfer(
                name="wall", flows=("heat", "heat"), equation=lambda comp: 1.0
            )

    with pytest.raises(ValueError) as raised:
        Bespoke("W")
    message = str(raised.value)
    assert "`W`" in message and "`wall`" in message
    assert "watched" in message


def test_a_potential_read_off_a_delivered_input_is_refused():
    """The counter-flow model of the literature reads its temperatures as
    an input delivery divided by a constant, which is neither of the two
    operand forms: it is refused here rather than bought with a wider
    vocabulary."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat_hot")
            self.add_flow_continuous_out(name="heat_hot")
            self.add_flow_continuous_out(name="heat_cold")
            self.add_transfer(
                name="exchange",
                flows=("heat_hot", "heat_cold"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 1.0,
                    "potential_a": {"ratio": ["heat_hot", "flow_hot"]},
                    "potential_b": {"const": 20.0},
                },
            )

    with pytest.raises(ValueError) as raised:
        Wrong("X")
    message = str(raised.value)
    assert "`X`" in message and "`exchange`" in message
    assert "const" in message and "measurement" in message


def test_an_unknown_transfer_law_is_refused():
    """Only the conductive law is declarable, and an unknown one names
    what is."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a")
            self.add_flow_continuous_out(name="b")
            self.add_transfer(
                name="pair", flows=("a", "b"), equation={"cls": "RadiativeTransfer"}
            )

    with pytest.raises(ValueError) as raised:
        Wrong("X")
    assert "ConductiveTransfer" in str(raised.value)


def test_a_negative_conductance_is_refused():
    """A negative conductance drives the quantity up its own gradient,
    which is not transport."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a")
            self.add_flow_continuous_out(name="b")
            self.add_transfer(
                name="pair",
                flows=("a", "b"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": -1.0,
                    "potential_a": 10.0,
                    "potential_b": 0.0,
                },
            )

    with pytest.raises(ValueError, match="conductance"):
        Wrong("X")


def test_a_transferred_flow_must_be_a_continuous_output():
    """A pair writes both balances and a balance is written on the output
    side: naming a flow this component only consumes is refused."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="b")
            self.add_transfer(
                name="pair",
                flows=("a", "b"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 1.0,
                    "potential_a": 10.0,
                    "potential_b": 0.0,
                },
            )

    with pytest.raises(ValueError) as raised:
        Wrong("X")
    assert "`X`" in str(raised.value) and "`a`" in str(raised.value)


def test_a_conduit_needs_the_input_side():
    """A conduit meters a transit, and there is no transit to meter
    without an input."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="heat")
            self.add_transfer(
                name="wall",
                flows=("heat", "heat"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 1.0,
                    "potential_a": 10.0,
                    "potential_b": 0.0,
                },
            )

    with pytest.raises(ValueError) as raised:
        Wrong("W")
    assert "`W`" in str(raised.value) and "`wall`" in str(raised.value)


def test_a_conduit_may_not_meter_a_flow_a_rule_set_names():
    """The conduit replaces what crosses the component, so a flow a rule
    already carries would cross twice."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat")
            self.add_flow_continuous_out(name="heat")
            self.add_rule_set(name="warm", rules=[{"prod": {"heat": 1}}])
            self.add_transfer(
                name="wall",
                flows=("heat", "heat"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 1.0,
                    "potential_a": 10.0,
                    "potential_b": 0.0,
                },
            )

    with pytest.raises(ValueError) as raised:
        Wrong("W")
    assert "`W`" in str(raised.value) and "`heat`" in str(raised.value)


def test_a_measurement_a_component_does_not_declare_is_refused():
    """A potential read over a channel the component never declared is a
    dangling read, refused where it was written."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a")
            self.add_flow_continuous_out(name="b")
            self.add_transfer(
                name="pair",
                flows=("a", "b"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 1.0,
                    "potential_a": {"measurement": "hot"},
                    "potential_b": 0.0,
                },
            )

    with pytest.raises(ValueError) as raised:
        Wrong("X")
    assert "`hot`" in str(raised.value)


# --- time profiles ------------------------------------------------------


def panel_system(profile: dict) -> mu.System:
    """A profiled panel charging a battery.

    The battery is what makes the run a *trajectory*: production is
    integrated into its content, so the declared curve is followed at the
    integration points the solver chooses rather than only at the
    discrete instants of a purely algebraic model."""

    class Panel(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="power", var_fed_default=10.0, profile=profile
            )

    class Battery(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="power")
            self.add_capacity(
                name="store", flow="power", capacity=1e4, fill_rate=1e4
            )

    system = mu.System("profiled_source")
    system.add_component(Panel, "P")
    system.add_component(Battery, "B")
    system.connect("P", "power", "B", "power")
    return system


def test_a_sinusoidal_profile_modulates_production():
    """The trajectory follows the declared curve: a clamped sinusoid of
    period 24 scales the panel's 10 units, the negative half-cycle is cut
    at zero rather than driving the flow backwards, and what the battery
    holds is the integral of what the curve let through."""
    system = panel_system(
        {
            "cls": "SinusoidalProfile",
            "amplitude": 1.0,
            "period": 24.0,
            "phase_shift": 0.0,
            "offset": 0.0,
            "value_min": 0.0,
        }
    )
    instants = [0.0, 3.0, 6.0, 12.0, 18.0]
    result = system.simulate(t_max=24.0, samples=instants)

    for instant in instants:
        factor = max(0.0, math.sin(2 * math.pi * instant / 24.0))
        assert abs(sampled(result, "P_power_out_profile", instant) - factor) < (
            CROSSING_TOL
        ), instant
        assert abs(sampled(result, "B_power_fed_in", instant) - 10.0 * factor) < (
            CROSSING_TOL
        ), instant

    # The closed form of the half-cycle: 10 x (24 / 2 pi) x (1 - cos(2 pi t / 24)).
    for instant in (3.0, 6.0, 12.0):
        stored = 10.0 * (24.0 / (2 * math.pi)) * (
            1.0 - math.cos(2 * math.pi * instant / 24.0)
        )
        assert abs(sampled(result, "B_store_content", instant) - stored) < 1e-4, instant
    # Past the half-cycle the curve is clamped at zero, so the content
    # stops where the first half-cycle left it.
    assert abs(
        sampled(result, "B_store_content", 18.0)
        - 10.0 * (24.0 / (2 * math.pi)) * 2.0
    ) < 1e-4


def test_a_bespoke_profile_function_is_refused():
    """A callable carries no continuity attestation: the layer refuses
    it, naming the component, the flow and the mechanism a discontinuous
    profile would need."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="power", var_fed_default=10.0, profile=lambda t: 1.0
            )

    with pytest.raises(ValueError) as raised:
        Wrong("P")
    message = str(raised.value)
    assert "`P`" in message and "`power`" in message
    assert "watched" in message


def test_a_profile_declared_without_a_shape_is_refused():
    """`{"cls": "Profile"}` is muscadet's bare-callable shape: its whole
    content is a Python function, so no mapping can carry it and naming
    it is refused rather than answered with a default curve."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", profile={"cls": "Profile"})

    with pytest.raises(ValueError) as raised:
        Wrong("P")
    assert "SinusoidalProfile" in str(raised.value)


def test_a_profile_may_not_go_negative():
    """A profile scales production, so a negative factor would mean a
    negative quantity: the lower clamp is refused below zero."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="power",
                profile={"cls": "SinusoidalProfile", "value_min": -1.0},
            )

    with pytest.raises(ValueError, match="value_min"):
        Wrong("P")


# --- deratings ----------------------------------------------------------


def derated_system(*modes: dict, profile: dict | None = None) -> mu.System:
    """A source of 10 units of `power`, carrying the declared modes."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="power", var_fed_default=10.0, profile=profile
            )
            for mode in modes:
                self.add_delay_failure_mode(**mode)

    class Grid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="power", var_demand_in_default=1e4)

    system = mu.System("derated_source")
    system.add_component(Plant, "P")
    system.add_component(Grid, "G")
    system.connect("P", "power", "G", "power")
    return system


def test_two_deratings_compose_by_minimum():
    """Two modes leaving 0.5 and 0.8 of one output leave 0.5, not 0.4:
    the composition is order-independent and safe on repair."""
    system = derated_system(
        {
            "name": "half",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.5)],
        },
        {
            "name": "slight",
            "failure_time": 4.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.8)],
        },
    )
    result = system.simulate(t_max=10.0, samples=[1.0, 3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 10.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 3.0) - 5.0) < CROSSING_TOL
    # A product would read 4.0 here, a minimum reads 5.0.
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 5.0) < CROSSING_TOL


def test_a_rate_of_zero_stops_production_entirely():
    """There is no separate boolean gate on a continuous flow: a mode
    that leaves nothing of the output is what stops it."""
    system = derated_system(
        {
            "name": "dead",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.0)],
        }
    )
    result = system.simulate(t_max=10.0, samples=[1.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 10.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0)) < CROSSING_TOL
    assert abs(sampled(result, "P_power_capability_out", 5.0)) < CROSSING_TOL


def test_leaving_the_failing_state_restores_the_nominal_rate():
    """The return to nominal is implicit: the mode declares an effect on
    one state only, and repairing it restores the output with nothing
    declared on the other."""
    system = derated_system(
        {
            "name": "degrade",
            "failure_time": 2.0,
            "repair_time": 2.0,
            "failure_effects": [("power", 0.4)],
        }
    )
    result = system.simulate(t_max=10.0, samples=[1.0, 3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 10.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 3.0) - 4.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 10.0) < CROSSING_TOL


def test_a_mode_may_return_degraded_rather_than_as_new():
    """The implicit release is a default, not a rule: a mode declaring an
    effect on its repair keeps that value instead."""
    system = derated_system(
        {
            "name": "wear",
            "failure_time": 2.0,
            "repair_time": 2.0,
            "failure_effects": [("power", 0.4)],
            "repair_effects": [("power", 0.9)],
        }
    )
    result = system.simulate(t_max=10.0, samples=[3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 3.0) - 4.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 9.0) < CROSSING_TOL


def test_a_profile_and_a_derating_compose_by_product():
    """A source at 30 % of its profile that is also derated to 0.5
    produces 15 %, not 30 %: the two channels multiply, and folding the
    profile into the shared rate would read the minimum instead."""
    system = derated_system(
        {
            "name": "half",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.5)],
        },
        profile={"cls": "SinusoidalProfile", "amplitude": 0.0, "offset": 0.3},
    )
    result = system.simulate(t_max=10.0, samples=[1.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 3.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 1.5) < CROSSING_TOL


def test_a_derating_pattern_is_anchored():
    """`"H2"` names the hydrogen output and not the water one beside it:
    an unanchored match would derate a neighbour a declaration never
    meant."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2", var_fed_default=10.0)
            self.add_flow_continuous_out(name="H2O", var_fed_default=10.0)
            self.add_delay_failure_mode(
                name="stack",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("H2", 0.5)],
            )

    class Grid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2", var_demand_in_default=1e4)
            self.add_flow_continuous_in(name="H2O", var_demand_in_default=1e4)

    system = mu.System("anchored_derating")
    system.add_component(Plant, "P")
    system.add_component(Grid, "G")
    system.connect("P", "H2", "G", "H2")
    system.connect("P", "H2O", "G", "H2O")
    result = system.simulate(t_max=10.0, samples=[5.0])

    assert abs(sampled(result, "G_H2_fed_in", 5.0) - 5.0) < CROSSING_TOL
    assert abs(sampled(result, "G_H2O_fed_in", 5.0) - 10.0) < CROSSING_TOL


def test_the_fed_out_alias_names_the_same_output():
    """The 1.x spelling of an effect on an output keeps working: `"X"`
    and `"X_fed_out"` designate the same continuous output."""
    system = derated_system(
        {
            "name": "half",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power_fed_out", 0.5)],
        }
    )
    result = system.simulate(t_max=10.0, samples=[5.0])
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 5.0) < CROSSING_TOL


def test_an_effect_naming_no_continuous_output_is_refused():
    """An effect that reaches nothing is a silent no-op, which is the
    defect class the refusal exists to close: it names the component, the
    mode and the pattern."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)
            self.add_delay_failure_mode(
                name="half",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("water", 0.5)],
            )

    with pytest.raises(ValueError) as raised:
        Wrong("P")
    message = str(raised.value)
    assert "`P`" in message and "`half`" in message and "`water`" in message


def test_a_derating_outside_the_unit_interval_is_refused():
    """A derating is what a mode LEAVES of an output: a factor above one
    is not a degradation and a negative one is not a quantity."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)
            self.add_delay_failure_mode(
                name="boost",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("power", 1.5)],
            )

    with pytest.raises(ValueError) as raised:
        Wrong("P")
    assert "`boost`" in str(raised.value)


def test_a_boolean_effect_on_a_continuous_output_is_a_full_derating():
    """`False` on a continuous output is the muscadet idiom for a total
    loss, and it reads as a rate of zero rather than as a missing gate."""
    system = derated_system(
        {
            "name": "dead",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", False)],
        }
    )
    result = system.simulate(t_max=10.0, samples=[5.0])
    assert abs(sampled(result, "G_power_fed_in", 5.0)) < CROSSING_TOL


def test_the_shared_out_rate_folds_in_with_the_per_mode_deratings():
    """`{flow}_out_rate` is the endpoint a mode declared outside this
    layer clamps: it enters the same minimum, so the deeper of the two
    is what the output produces at."""
    document = pyraichu.model_body(
        derated_system(
            {
                "name": "half",
                "failure_time": 2.0,
                "repair_time": 1e9,
                "failure_effects": [("power", 0.5)],
            }
        ).build_dict()
    )
    plant = next(c for c in document["components"] if c["name"] == "P")
    equation = next(
        e for e in plant["equations"] if e["target"] == "power_effective_rate"
    )
    assert equation["expr"]["op"] == "min"
    reads = {
        arg["attr"]["attribute"]
        for arg in equation["expr"]["args"]
        if arg["op"] == "attr"
    }
    assert "power_out_rate" in reads


# --- regression ---------------------------------------------------------


def test_a_model_declaring_none_of_these_is_unchanged():
    """A continuous model with no transfer, no profile and no derating
    generates exactly the attributes it did before this unit: none of the
    three mechanisms leaves a trace it does not have to."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="power", var_demand_in_default=4.0)

    system = mu.System("continuous_pair")
    system.add_component(Source, "S")
    system.add_component(Consumer, "C")
    system.connect("S", "power", "C", "power")
    document = pyraichu.model_body(system.build_dict())

    source = next(c for c in document["components"] if c["name"] == "S")
    assert [attribute["name"] for attribute in source["attributes"]] == [
        "power_out_rate",
        "power_capability_out",
        "power_demand_out",
        "power_fed_out",
    ]
    assert [equation["target"] for equation in source["equations"]] == [
        "power_capability_out",
        "power_out__capability__C__power_in",
        "power_out__demand__C__power_in",
        "power_demand_out",
        "power_fed_out",
    ]


def test_two_pairs_on_one_stream_compose_sequentially():
    """Two pairs relieving one stream are capped one after the other, in
    declaration order: between them they cannot take more than the stream
    carries, and the raw total stays put."""

    class Splitter(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="heat_hot", var_fed_default=100.0)
            self.add_flow_continuous_out(name="heat_a", var_fed_default=0.0)
            self.add_flow_continuous_out(name="heat_b", var_fed_default=0.0)
            self.add_measurement_in(name="hot")
            for name, target in (("first", "heat_a"), ("second", "heat_b")):
                self.add_transfer(
                    name=name,
                    flows=("heat_hot", target),
                    equation={
                        "cls": "ConductiveTransfer",
                        "conductance": 1.0,
                        "potential_a": {"measurement": "hot"},
                        "potential_b": {"const": 0.0},
                    },
                )

    class Sink(mu.ObjFlow):
        def add_flows(self):
            for flow in ("heat_hot", "heat_a", "heat_b"):
                self.add_flow_continuous_in(name=flow, var_demand_in_default=1e4)

    system = mu.System("sequential_pairs")
    # Each pair asks for 80, and the stream carries 100.
    system.add_component(probe(80.0), "TH")
    system.add_component(Splitter, "X")
    system.add_component(Sink, "L")
    system.connect_measurement("TH", "probe", "X", "hot")
    for flow in ("heat_hot", "heat_a", "heat_b"):
        system.connect("X", flow, "L", flow)
    result = system.simulate(t_max=1.0)

    assert abs(at_zero(result, "X_first_moved") - 80.0) < TOL
    # Only 20 is left for the second, which asked for 80 as well.
    assert abs(at_zero(result, "X_second_moved") - 20.0) < TOL
    assert abs(at_zero(result, "L_heat_hot_fed_in")) < TOL
    assert abs(at_zero(result, "L_heat_a_fed_in") - 80.0) < TOL
    assert abs(at_zero(result, "L_heat_b_fed_in") - 20.0) < TOL


@pytest.mark.parametrize(
    "name",
    ["exchange", "conduit", "profile", "derating"],
)
def test_every_swept_step_appears_once_in_the_evaluation_order(name):
    """The engine sweeps the explicit equations in the order the model
    carries and refuses an omission rather than completing it: every
    equation and every allocation this unit adds is in that order, and
    none of them twice."""
    system = {
        "exchange": lambda: exchanger_system(80.0, 20.0),
        "conduit": lambda: conduit_system(500.0),
        "profile": lambda: panel_system(
            {"cls": "SinusoidalProfile", "period": 24.0}
        ),
        "derating": lambda: derated_system(
            {
                "name": "half",
                "failure_time": 2.0,
                "repair_time": 1e9,
                "failure_effects": [("power", 0.5)],
            }
        ),
    }[name]()
    document = pyraichu.model_body(system.build_dict())

    order = [
        (step["component"], step["attribute"])
        for step in document["evaluation_order"]
    ]
    declared = {
        (component["name"], target)
        for component in document["components"]
        for target in (
            [
                equation["target"]
                for equation in component["equations"]
                if equation["kind"] == "explicit"
            ]
            + [
                allocation["name"]
                for allocation in component.get("allocations", [])
            ]
        )
    }
    assert len(order) == len(set(order))
    assert set(order) == declared


def test_a_derated_conduit_draws_no_more_than_it_hands_on():
    """A conduit's draw is scaled by what its output's deratings leave of
    it: a dead conduit asks its supplier for nothing, rather than drawing
    a quantity it cannot pass on and destroying the difference."""

    class Supply(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="heat", var_fed_default=500.0)

    class Wall(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat")
            self.add_flow_continuous_out(name="heat", var_fed_default=1000.0)
            self.add_measurement_in(name="inside")
            self.add_transfer(
                name="wall",
                flows=("heat", "heat"),
                equation={
                    "cls": "ConductiveTransfer",
                    "conductance": 2.0,
                    "potential_a": {"measurement": "inside"},
                    "potential_b": {"const": 5.0},
                },
            )
            self.add_delay_failure_mode(
                name="blocked",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("heat", 0.0)],
            )

    class Room(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="heat", var_demand_in_default=1e4)

    system = mu.System("derated_conduit")
    system.add_component(probe(30.0), "TI")
    system.add_component(Supply, "S")
    system.add_component(Wall, "W")
    system.add_component(Room, "R")
    system.connect_measurement("TI", "probe", "W", "inside")
    system.connect("S", "heat", "W", "heat")
    system.connect("W", "heat", "R", "heat")
    result = system.simulate(t_max=10.0, samples=[1.0, 5.0])

    assert abs(sampled(result, "R_heat_fed_in", 1.0) - 50.0) < CROSSING_TOL
    assert abs(sampled(result, "W_heat_demand_in", 1.0) - 50.0) < CROSSING_TOL
    # Dead, it neither hands anything on nor draws anything.
    assert abs(sampled(result, "R_heat_fed_in", 5.0)) < CROSSING_TOL
    assert abs(sampled(result, "W_heat_demand_in", 5.0)) < CROSSING_TOL
    assert abs(sampled(result, "S_heat_fed_out", 5.0)) < CROSSING_TOL
