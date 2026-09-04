"""A time profile on a continuous INPUT: a demand that varies with the
clock, as a production already could.

`profile` existed on an output and had no counterpart on an input, which
was an accident of which side was built first rather than a statement
about either. A consumer whose appetite follows the hour is as ordinary
as a source whose output follows the sun, and the two are now declared in
the same shape, refused under the same rule, and published the same way:
`{flow}_in_profile` beside `{flow}_out_profile`.

The profile scales whatever the demand is derived from, so a rule set
consuming the flow is scaled with it. It is applied BEFORE the volume
holding the flow bounds the result: the profile says how much is wanted,
the volume says how much of that there is room for.

One limitation is pinned here rather than left to be discovered: a
profile only moves while something in the model integrates. With no
continuous state anywhere, the engine has nothing to advance, evaluates
the sweep once and records that value at every sample instant. It is the
same on an output, so this is a property of sampling and not of the
profile.
"""

import math

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

#: The curve both suites use: 0.6 + 0.4 sin(2 pi t / 24), which is 0.6 at
#: midnight, 1.0 at six, 0.6 again at noon and 0.2 at eighteen.
CURVE = {
    "cls": "SinusoidalProfile",
    "amplitude": 0.4,
    "period": 24.0,
    "offset": 0.6,
}
HOURS = [0.0, 6.0, 12.0, 18.0]


def curve_at(instant: float) -> float:
    return 0.6 + 0.4 * math.sin(2 * math.pi * instant / 24.0)


def town(demand: float, profile, integrating: bool = True) -> type:
    class Town(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(
                name="w", var_demand_in_default=demand, profile=profile
            )
            if integrating:
                # Something has to integrate for the clock to advance
                # continuously; a basin that never fills is the smallest
                # thing that does.
                self.add_capacity(
                    name="basin", flow="w", capacity=1.0e9, fill_rate=0.0
                )

    return Town


def supply(rate: float = 100.0) -> type:
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="w", var_fed_default=rate)

    return Source


def network(consumer: type, rate: float = 100.0) -> mu.System:
    system = mu.System("profiled_demand")
    system.add_component(supply(rate), "S")
    system.add_component(consumer, "T")
    system.connect("S", "w", "T", "w")
    return system


def test_a_sinusoidal_demand_follows_its_closed_form():
    """The point of the feature: what the consumer receives is its
    declared demand times the curve, at every sampled hour."""
    result = network(town(1.0, CURVE)).simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        assert abs(sampled(result, "T_w_fed_in", hour) - curve_at(hour)) < (
            CROSSING_TOL
        )


def test_the_profile_scales_the_declared_demand_and_is_published():
    """A demand of 3 under the same curve asks for three times it, and
    the factor is readable on its own rather than only inferable from
    the product."""
    result = network(town(3.0, CURVE)).simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        assert abs(sampled(result, "T_w_in_profile", hour) - curve_at(hour)) < (
            CROSSING_TOL
        )
        assert abs(
            sampled(result, "T_w_fed_in", hour) - 3.0 * curve_at(hour)
        ) < CROSSING_TOL


def test_the_profile_scales_a_demand_a_rule_set_derives():
    """Not only the declared constant: the profile scales whatever the
    demand is derived from, so a component that consumes to produce asks
    for less when the curve is low."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", profile=CURVE)
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="convert", rules=[{"cons": {"w": 1.0}, "prod": {"x": 1.0}}]
            )

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x", var_demand_in_default=2.0)
            # The integrator lives here: a volume on a flow the rule
            # produces would make the flow cross the plant twice, which
            # the layer refuses.
            self.add_capacity(name="basin", flow="x", capacity=1.0e9, fill_rate=0.0)

    system = mu.System("profiled_rule")
    system.add_component(supply(), "S")
    system.add_component(Plant, "P")
    system.add_component(Sink, "K")
    system.connect("S", "w", "P", "w")
    system.connect("P", "x", "K", "x")
    result = system.simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        # The rule would ask for 2; the profile scales that request.
        assert abs(
            sampled(result, "P_w_fed_in", hour) - 2.0 * curve_at(hour)
        ) < CROSSING_TOL


def test_a_supply_short_of_the_profiled_demand_still_binds():
    """The profile is a demand and not a delivery: a supply below it
    delivers what it has."""
    result = network(town(1.0, CURVE), rate=0.8).simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        expected = min(0.8, curve_at(hour))
        assert abs(sampled(result, "T_w_fed_in", hour) - expected) < CROSSING_TOL


def test_a_discontinuous_input_profile_is_refused_as_on_an_output():
    """The same refusal on both sides: continuity is an attestation this
    layer cannot make for the modeller, and it derives no watched
    transition from a callable's breakpoints."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", profile=lambda t: 1.0)

    with pytest.raises(ValueError):
        Wrong("T")


def test_a_negative_input_profile_is_refused_as_on_an_output():
    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(
                name="w", profile={"cls": "SinusoidalProfile", "value_min": -1.0}
            )

    with pytest.raises(ValueError, match="value_min"):
        Wrong("T")


def test_an_input_declaring_no_profile_generates_the_document_it_did():
    """The counterpart of the same guarantee on the output side: the
    factor is a variable and an equation added where it is declared, not
    something every input pays for."""
    without = network(town(1.0, None)).build_dict()
    consumer = next(
        component
        for component in without["model"]["components"]
        if component["name"] == "T"
    )
    assert not [
        name
        for name in (variable["name"] for variable in consumer["attributes"])
        if name.endswith("_in_profile")
    ]


def test_the_plugin_carries_an_input_profile():
    import pyraichu

    spec = {
        "name": "profiled_demand",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "S",
                        "flows_continuous_out": [
                            {"name": "w", "var_fed_default": 100.0}
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "T",
                        "flows_continuous_in": [
                            {
                                "name": "w",
                                "var_demand_default": 1.0,
                                "profile": CURVE,
                            }
                        ],
                        "capacities": [
                            {
                                "name": "basin",
                                "flow": "w",
                                "capacity": 1.0e9,
                                "fill_rate": 0.0,
                            }
                        ],
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {
                "from": {"component": "S", "port": "w_out"},
                "to": {"component": "T", "port": "w_in"},
            }
        ],
        "indicators": [],
    }
    body = pyraichu.expand_model(spec)
    result = pyraichu.simulate(
        pyraichu.load_model(body), t_max=24.0, samples=HOURS
    )
    for hour in HOURS:
        assert abs(sampled(result, "T_w_fed_in", hour) - curve_at(hour)) < (
            CROSSING_TOL
        )


# --- a profile moves whether or not anything else does -------------------


@pytest.mark.parametrize("side", ["in", "out"])
def test_a_profile_moves_with_nothing_to_integrate(side):
    """A model whose ONLY time dependence is the profile, with no volume
    and no ODE anywhere, still follows its curve.

    The engine decides whether continuous evolution runs at all, and used
    to decide it on the ODE attributes and the armed hazards alone. A
    declared profile is an explicit equation over the clock and nothing
    else, so a model carrying one and nothing else found nothing to
    advance: it evaluated the sweep once at the initial instant and
    reported that value at every sample instant for the rest of the run,
    a curve reported as a constant with nothing to signal it.

    Both sides are asserted together, so a change to one that missed the
    other is caught."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="w",
                var_fed_default=1.0,
                profile=CURVE if side == "out" else None,
            )

    system = mu.System("nothing_integrates")
    system.add_component(Source, "S")
    system.add_component(
        town(1.0, CURVE if side == "in" else None, integrating=False), "T"
    )
    system.connect("S", "w", "T", "w")
    result = system.simulate(t_max=24.0, samples=HOURS)

    owner = "S" if side == "out" else "T"
    for hour in HOURS:
        assert abs(
            sampled(result, f"{owner}_w_{side}_profile", hour) - curve_at(hour)
        ) < CROSSING_TOL, hour
        assert abs(sampled(result, "T_w_fed_in", hour) - curve_at(hour)) < (
            CROSSING_TOL
        ), hour


def test_a_watched_guard_on_a_profile_is_located_with_nothing_to_integrate():
    """The worse half of the same defect, and the reason it was worth
    fixing rather than documenting: a guard on a profiled quantity was
    never crossed, because the quantity never moved.

    The curve passes 0.8 on its way up, at the hour whose sine is 0.5,
    which is 2. A mode watching it must fire there and nowhere else."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="w", var_fed_default=1.0, profile=CURVE
            )

    class Alarm(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=100.0)
            self.add_flow_continuous_out(name="alarm")
            self.add_rule_set(
                name="trip",
                rules=[
                    {
                        "cond": [{"name": "w", "op": ">", "value": 0.8}],
                        "cons": {},
                        "prod": {"alarm": 1.0},
                    }
                ],
            )

    class Watcher(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="alarm", var_demand_in_default=100.0)

    system = mu.System("watched_profile")
    system.add_component(Source, "S")
    system.add_component(Alarm, "A")
    system.add_component(Watcher, "W")
    system.connect("S", "w", "A", "w")
    system.connect("A", "alarm", "W", "alarm")
    # sin(2 pi t / 24) = 0.5 at t = 2, and the curve is 0.8 there.
    assert abs(curve_at(2.0) - 0.8) < 1e-12
    result = system.simulate(t_max=12.0, samples=[1.9, 2.1, 6.0, 11.0])
    assert abs(sampled(result, "W_alarm_fed_in", 1.9)) < CROSSING_TOL
    assert abs(sampled(result, "W_alarm_fed_in", 2.1) - 1.0) < CROSSING_TOL
    assert abs(sampled(result, "W_alarm_fed_in", 6.0) - 1.0) < CROSSING_TOL
    # And back below on the way down, at t = 10.
    assert abs(sampled(result, "W_alarm_fed_in", 11.0)) < CROSSING_TOL
