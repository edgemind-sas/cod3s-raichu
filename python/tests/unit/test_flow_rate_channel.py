"""The rate a flow carries, published as an observable channel.

A volume publishes what it holds, and a controller can threshold it. A
flow published nothing, so a controller could not threshold what a
component produces or consumes: the `rate` input kind was declared and
unusable for want of a publisher, exactly as the `ratio` kind was.

`publish_rate` on a continuous flow materialises `{flow}_rate` on a port
of its own. One name for both directions on purpose: what an observer
wants is the quantity crossing the wire, and which side of it the
publisher sits on is the publisher's business. It carries what actually
crossed and not what could have, because an observer watching a supply
wants what arrived.

Declared and not implied: publishing every rate would put a port and an
equation on every flow of every model for the few an observer reads.
"""

import math

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

CURVE = {
    "cls": "SinusoidalProfile",
    "amplitude": 10.0,
    "period": 24.0,
    "offset": 10.0,
}
HOURS = [3.0, 6.0, 12.0, 18.0]


def curve_at(instant: float) -> float:
    return 10.0 + 10.0 * math.sin(2 * math.pi * instant / 24.0)


def network(publish_out: bool = True, publish_in: bool = True, demand=100.0):
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="E",
                var_fed_default=1.0,
                profile=CURVE,
                publish_rate=publish_out,
            )

    class Load(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(
                name="E", var_demand_in_default=demand, publish_rate=publish_in
            )

    system = mu.System("rates")
    system.add_component(Source, "S")
    system.add_component(Load, "L")
    system.connect("S", "E", "L", "E")
    return system


def test_an_output_publishes_what_it_delivers():
    result = network().simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        assert abs(sampled(result, "S_E_rate", hour) - curve_at(hour)) < CROSSING_TOL


def test_an_input_publishes_what_it_receives():
    result = network().simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        assert abs(sampled(result, "L_E_rate", hour) - curve_at(hour)) < CROSSING_TOL


def test_the_two_ends_of_a_connection_publish_the_same_quantity():
    """A conservation identity across the wire, and the reason one name
    serves both directions."""
    result = network().simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        assert sampled(result, "S_E_rate", hour) == sampled(result, "L_E_rate", hour)


def test_a_rate_is_what_crossed_and_not_what_could_have():
    """A demand of 4 against a supply that rises to 20: the published
    rate follows the demand where the demand binds, and the supply where
    the supply does."""
    result = network(demand=4.0).simulate(t_max=24.0, samples=HOURS)
    for hour in HOURS:
        expected = min(4.0, curve_at(hour))
        assert abs(sampled(result, "S_E_rate", hour) - expected) < CROSSING_TOL, hour


def test_a_flow_declaring_no_channel_carries_none():
    """The bound is a port and an equation added where it is declared,
    not something every flow of every model pays for."""
    document = network(publish_out=False, publish_in=False).build_dict()["model"]
    for component in document["components"]:
        names = {variable["name"] for variable in component["attributes"]}
        # Exactly `E_rate`, not `E_out_rate`: the second is the shared
        # derating endpoint every output has always carried, and a check
        # that caught it too would pass for the wrong reason.
        assert "E_rate" not in names, component["name"]
        assert not [
            port for port in component["ports"] if port["name"] == "E_rate_out"
        ], component["name"]
    with_channel = network().build_dict()["model"]
    assert "E_rate" in {
        variable["name"]
        for component in with_channel["components"]
        for variable in component["attributes"]
    }


def test_a_rate_is_swept_after_the_delivery_it_reports():
    """Asserted on the emitted order: read one sweep early, a rate
    reports the previous instant's delivery and still looks plausible."""
    order = [
        f"{entry['component']}.{entry['attribute']}"
        for entry in network().build_dict()["model"]["evaluation_order"]
    ]
    assert order.index("S.E_fed_out") < order.index("S.E_rate")
    assert order.index("L.E_fed_in") < order.index("L.E_rate")
    assert order.index("S.E_rate") < order.index("L.E_rate")


def test_a_controller_thresholds_a_published_rate():
    """What the publication exists for. The supply crosses 12 on its way
    down at the hour whose sine is 0.2, and the controller must follow
    the curve rather than a stale reading."""
    spec = {
        "name": "rate_control",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "S",
                        "flows_continuous_out": [
                            {
                                "name": "E",
                                "var_fed_default": 1.0,
                                "publish_rate": True,
                                "profile": CURVE,
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "L",
                        "flows_continuous_in": [
                            {"name": "E", "var_demand_default": 100.0}
                        ],
                    },
                    {
                        "type": "ObjCtrl",
                        "name": "CTRL",
                        "controls_in": [{"name": "E", "kind": "rate"}],
                        "controls_out": [
                            {
                                "name": "low",
                                "kind": "bool",
                                "emit": {
                                    "op": "compare",
                                    "input": "E",
                                    "operator": "<",
                                    "threshold": 12.0,
                                },
                            }
                        ],
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {
                "from": {"component": "S", "port": "E_out"},
                "to": {"component": "L", "port": "E_in"},
            },
            {
                "from": {"component": "S", "port": "E_rate_out"},
                "to": {"component": "CTRL", "port": "E_rate_in"},
            },
        ],
        "indicators": [],
    }
    body = pyraichu.expand_model(spec)
    result = pyraichu.simulate(
        pyraichu.load_model(body), t_max=24.0, samples=[3.0, 6.0, 12.0, 18.0]
    )
    for hour in (3.0, 6.0, 12.0, 18.0):
        assert sampled(result, "CTRL_E_rate", hour) == sampled(result, "S_E_rate", hour)
        assert sampled(result, "CTRL_low", hour) is (curve_at(hour) < 12.0), hour
