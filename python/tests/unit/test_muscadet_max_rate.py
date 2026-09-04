"""A ceiling on what a continuous output can deliver per unit time.

`max_rate` is a property of the equipment: this component cannot make
more than that, whatever it is fed and whatever is asked of it. It has no
muscadet counterpart, and it is not a failure-mode cap: a cap is a
FRACTION of nominal owned by the mode that declares it, this is an
absolute quantity that stands for the whole run.

Where the distinction bites is the demand. On an output a rule set
produces, the ceiling bounds the scale the rule runs at as well as the
quantity it delivers, so the component asks its own suppliers for what it
will actually make. Bounded only at the output, it would go on drawing
the full quantity upstream and lose the difference, which is matter
created and then destroyed inside one component.
"""

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled


def reactor(max_rate: float | None, demand: float = 100.0) -> mu.System:
    """One unit of `a` and one of `b` make one of `x`, with a supply of
    100 of each so nothing upstream is ever the binding constraint."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a", var_fed_default=100.0)
            self.add_flow_continuous_out(name="b", var_fed_default=100.0)

    class Reactor(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_in(name="b")
            self.add_flow_continuous_out(name="x", max_rate=max_rate)
            self.add_rule_set(
                name="mix",
                rules=[{"cons": {"a": 1.0, "b": 1.0}, "prod": {"x": 1.0}}],
            )

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x", var_demand_in_default=demand)

    system = mu.System("ceiling")
    system.add_component(Source, "S")
    system.add_component(Reactor, "R")
    system.add_component(Sink, "K")
    system.connect("S", "a", "R", "a")
    system.connect("S", "b", "R", "b")
    system.connect("R", "x", "K", "x")
    return system


def test_a_ceiling_bounds_what_a_rule_delivers():
    """Asked for 100 and able to make 100, the reactor makes its
    ceiling."""
    result = reactor(max_rate=4.0).simulate(t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "K_x_fed_in", 0.5) - 4.0) < CROSSING_TOL


def test_a_ceiling_bounds_what_the_component_draws_upstream():
    """The load-bearing half: the reactor draws what it needs to make 4,
    not what it was asked for.

    Drawing 100 of each to deliver 4 would take 96 of each out of the
    supply and record them nowhere, which is the defect this bound
    exists to close."""
    result = reactor(max_rate=4.0).simulate(t_max=1.0, samples=[0.5])
    for flow in ("a", "b"):
        drawn = sampled(result, f"R_{flow}_fed_in", 0.5)
        assert abs(drawn - 4.0) < CROSSING_TOL, flow


def test_a_ceiling_above_the_demand_binds_on_nothing():
    """A ceiling is a maximum and not a target: below it, the demand
    still decides."""
    result = reactor(max_rate=50.0, demand=3.0).simulate(t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "K_x_fed_in", 0.5) - 3.0) < CROSSING_TOL
    assert abs(sampled(result, "R_a_fed_in", 0.5) - 3.0) < CROSSING_TOL


def test_no_ceiling_generates_the_document_it_always_did():
    """An output declaring none carries nothing extra: the bound is an
    expression added where it is declared, not a variable every output
    pays for."""
    assert reactor(max_rate=None).build_dict() == reactor(max_rate=None).build_dict()
    without = reactor(max_rate=None).build_dict()
    with_it = reactor(max_rate=4.0).build_dict()
    assert without != with_it
    names = {
        variable["name"]
        for component in without["model"]["components"]
        for variable in component["attributes"]
    }
    assert names == {
        variable["name"]
        for component in with_it["model"]["components"]
        for variable in component["attributes"]
    }


def test_a_ceiling_scales_correlated_outputs_together():
    """A rule's outputs are correlated by construction, so a ceiling on
    one holds the others down in proportion rather than letting them run
    on and produce a surplus with nowhere to go."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a", var_fed_default=100.0)

    class Splitter(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x", max_rate=2.0)
            self.add_flow_continuous_out(name="y")
            self.add_rule_set(
                name="split",
                rules=[{"cons": {"a": 1.0}, "prod": {"x": 1.0, "y": 3.0}}],
            )

    def sink(flow):
        return type(
            f"Take{flow}",
            (mu.ObjFlow,),
            {
                "add_flows": lambda self, flow=flow: self.add_flow_continuous_in(
                    name=flow, var_demand_in_default=1e4
                )
            },
        )

    system = mu.System("correlated")
    system.add_component(Source, "S")
    system.add_component(Splitter, "R")
    system.add_component(sink("x"), "KX")
    system.add_component(sink("y"), "KY")
    system.connect("S", "a", "R", "a")
    system.connect("R", "x", "KX", "x")
    system.connect("R", "y", "KY", "y")
    result = system.simulate(t_max=1.0, samples=[0.5])

    assert abs(sampled(result, "KX_x_fed_in", 0.5) - 2.0) < CROSSING_TOL
    # The scale is 2, so the correlated output follows at 3 x 2.
    assert abs(sampled(result, "KY_y_fed_in", 0.5) - 6.0) < CROSSING_TOL
    assert abs(sampled(result, "R_a_fed_in", 0.5) - 2.0) < CROSSING_TOL


def test_a_ceiling_bounds_a_declared_rate_too():
    """An output need not come from a rule: the ceiling is a property of
    the equipment and binds wherever the quantity comes from."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0, max_rate=3.0)

    class Grid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="power", var_demand_in_default=1e4)

    system = mu.System("declared")
    system.add_component(Plant, "P")
    system.add_component(Grid, "G")
    system.connect("P", "power", "G", "power")
    result = system.simulate(t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "G_power_fed_in", 0.5) - 3.0) < CROSSING_TOL


def test_a_ceiling_and_a_failure_cap_compose_by_minimum():
    """Two ceilings, one absolute and one a fraction of nominal: the
    binding one wins, which is what two ceilings do.

    A rate of 10 capped at 0.5 by a mode is 5, and a `max_rate` of 3
    holds below it; raise the ceiling above 5 and the mode's cap binds
    instead."""

    def plant(max_rate: float) -> mu.System:
        class Plant(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_out(
                    name="power", var_fed_default=10.0, max_rate=max_rate
                )
                self.add_delay_failure_mode(
                    name="halved",
                    failure_time=1.0,
                    repair_time=1e9,
                    failure_effects=[("power", 0.5)],
                )

        class Grid(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_in(name="power", var_demand_in_default=1e4)

        system = mu.System("two_ceilings")
        system.add_component(Plant, "P")
        system.add_component(Grid, "G")
        system.connect("P", "power", "G", "power")
        return system

    low = plant(max_rate=3.0).simulate(t_max=4.0, samples=[0.5, 2.0])
    assert abs(sampled(low, "G_power_fed_in", 0.5) - 3.0) < CROSSING_TOL
    assert abs(sampled(low, "G_power_fed_in", 2.0) - 3.0) < CROSSING_TOL

    high = plant(max_rate=8.0).simulate(t_max=4.0, samples=[0.5, 2.0])
    assert abs(sampled(high, "G_power_fed_in", 0.5) - 8.0) < CROSSING_TOL
    assert abs(sampled(high, "G_power_fed_in", 2.0) - 5.0) < CROSSING_TOL


def test_a_negative_ceiling_is_refused():
    with pytest.raises(ValueError, match="is not negative"):

        class Wrong(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_out(name="power", max_rate=-1.0)

        Wrong("P")


def test_a_ceiling_of_zero_stops_the_output():
    """Zero is a legitimate ceiling and means what it says, which is why
    it is not folded into the `None` that means "no ceiling"."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0, max_rate=0.0)

    class Grid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="power", var_demand_in_default=1e4)

    system = mu.System("shut")
    system.add_component(Plant, "P")
    system.add_component(Grid, "G")
    system.connect("P", "power", "G", "power")
    result = system.simulate(t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "G_power_fed_in", 0.5)) < CROSSING_TOL


def test_the_plugin_carries_the_ceiling():
    """Declared in data, the ceiling reaches the authoring layer and
    generates the same component."""
    import pyraichu

    spec = {
        "name": "ceiling",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "P",
                        "flows_continuous_out": [
                            {
                                "name": "power",
                                "var_fed_default": 10.0,
                                "max_rate": 3.0,
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "G",
                        "flows_continuous_in": [
                            {"name": "power", "var_demand_default": 1e4}
                        ],
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {
                "from": {"component": "P", "port": "power_out"},
                "to": {"component": "G", "port": "power_in"},
            }
        ],
        "indicators": [],
    }
    body = pyraichu.expand_model(spec)
    result = pyraichu.simulate(pyraichu.load_model(body), t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "G_power_fed_in", 0.5) - 3.0) < CROSSING_TOL
