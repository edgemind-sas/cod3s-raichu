"""A volume and a rule set on one flow of one component.

muscadet lets one component hold a volume AND transform flows, the volume
sitting on one side of the rules: ``side="in"`` puts it upstream, so the
rules draw from it, and ``side="out"`` downstream, so the rules fill it.
The H2 plant's membrane is the shape that needs it: the leak is buffered
in the membrane's void before it is vented.

Neither side carries a flow twice. Upstream, what arrives fills the
volume and the rule's draw is its only outflow; downstream, what the rule
makes is the volume's only inflow and what the consumers take its only
outflow. The rules keep their declared coefficients; the volume changes
only what crosses the component's boundary, which is what the demand it
publishes upstream and the delivery it hands downstream say.

Every number below is a closed form of the minimal model, checked
against the muscadet 5.6.0 reference at the instants it was recorded:
a source of `S`, a unit turning ``k = 2`` of `a` into one `y`, a volume
of ten, and a consumer asking for one.
"""

import math

import pyraichu.muscadet as mu
import pytest
from conftest import CROSSING_TOL, sampled

K = 2.0
VOLUME = 10.0
DEMAND = 1.0
#: The slack on a closed form reached through located crossings and a
#: dense-output integration of piecewise-constant slopes.
REL = 1e-6


def close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= REL * max(1.0, abs(expected))


def unit_system(side: str | None, source: float, fill_rate, init: float = 0.0):
    """Source -> unit (``a:2 -> y:1``, a volume on `a` or on `y`) -> sink.

    `side=None` declares the capacity on `a` without a side, which is how
    an exported document that left it out arrives."""
    held = "y" if side == "out" else "a"

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a", var_fed_default=source)

    class Unit(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="y")
            self.add_rule_set(
                name="conv", rules=[{"cons": {"a": K}, "prod": {"y": 1.0}}]
            )
            settings = {
                "name": "buf",
                "flow": held,
                "capacity": VOLUME,
                "fill_rate": fill_rate,
                "content_init": {held: init},
            }
            if side is not None:
                settings["side"] = side
            self.add_capacity(**settings)

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="y", var_demand_in_default=DEMAND)

    system = mu.System(f"u32_{side}")
    system.add_component(Source, "SRC")
    system.add_component(Unit, "UNIT")
    system.add_component(Sink, "SINK")
    system.connect("SRC", "a", "UNIT", "a")
    system.connect("UNIT", "y", "SINK", "y")
    return system, held


def run(system, t_max, instants):
    return system.simulate(t_max=t_max, samples=list(instants))


# --- the seven scenarios of the minimal model ---------------------------

#: (side, source, fill_rate, init, t_max, {instant: (level, delivered,
#: source output, published upstream demand or None)}). `None` for the
#: fill rate is the declared spelling of an unbounded claim.
SCENARIOS = {
    # Upstream, unbounded claim: the source gives all it has, the rule
    # draws k*d, the volume rises at S - k*d = 3 and is full at 10/3.
    "in_fill_inf": (
        "in", 5.0, None, 0.0, 10.0,
        {1.0: (3.0, 1.0, 5.0, 5.0), 2.0: (6.0, 1.0, 5.0, 5.0),
         3.0: (9.0, 1.0, 5.0, 5.0), 5.0: (10.0, 1.0, 2.0, 2.0),
         9.0: (10.0, 1.0, 2.0, 2.0)},
    ),
    # Downstream, unbounded claim: the rule runs at S/k = 2.5, the
    # consumer takes 1, the volume rises at 1.5 and is full at 20/3.
    "out_fill_inf": (
        "out", 5.0, None, 0.0, 10.0,
        {1.0: (1.5, 1.0, 5.0, None), 2.0: (3.0, 1.0, 5.0, None),
         4.0: (6.0, 1.0, 5.0, None), 6.0: (9.0, 1.0, 5.0, None),
         8.0: (10.0, 1.0, 2.0, 2.0)},
    ),
    # Upstream, a claim of 0.5 on top of k*d: published 2.5, rises at
    # 0.5, full at 20.
    "in_fill_half": (
        "in", 5.0, 0.5, 0.0, 25.0,
        {2.5: (1.25, 1.0, 2.5, 2.5), 10.0: (5.0, 1.0, 2.5, 2.5),
         22.0: (10.0, 1.0, 2.0, 2.0)},
    ),
    # Downstream, the claim is in y units and multiplied by k upstream:
    # k*(d + f) = 3 published, the volume rises at 0.5 (y units).
    "out_fill_half": (
        "out", 5.0, 0.5, 0.0, 25.0,
        {2.5: (1.25, 1.0, 3.0, 3.0), 10.0: (5.0, 1.0, 3.0, 3.0),
         22.0: (10.0, 1.0, 2.0, 2.0)},
    ),
    # A pure buffer: it never stocks up, the source is throttled to k*d.
    "in_fill_zero": (
        "in", 5.0, 0.0, 0.0, 4.0,
        {1.0: (0.0, 1.0, 2.0, 2.0), 3.0: (0.0, 1.0, 2.0, 2.0)},
    ),
    "out_fill_zero": (
        "out", 5.0, 0.0, 0.0, 4.0,
        {1.0: (0.0, 1.0, 2.0, 2.0), 3.0: (0.0, 1.0, 2.0, 2.0)},
    ),
    # Draining upstream: a source of 1 against a need of 2, six held. The
    # volume falls at 1 and is empty at 6, after which the rule runs on
    # what arrives, S/k = 0.5.
    "in_drain": (
        "in", 1.0, 0.0, 6.0, 16.0,
        {1.6: (4.4, 1.0, 1.0, 2.0), 3.2: (2.8, 1.0, 1.0, 2.0),
         4.8: (1.2, 1.0, 1.0, 2.0), 8.0: (0.0, 0.5, 1.0, None),
         14.4: (0.0, 0.5, 1.0, None)},
    ),
    # Draining downstream: the rule makes 0.5, the consumer takes 1, the
    # volume falls at 0.5 and is empty at 12.
    "out_drain": (
        "out", 1.0, 0.0, 6.0, 16.0,
        {1.6: (5.2, 1.0, 1.0, None), 8.0: (2.0, 1.0, 1.0, None),
         14.4: (0.0, 0.5, 1.0, None)},
    ),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_a_volume_beside_the_rules_follows_its_closed_form(scenario):
    """Level, delivery, what the source gave and what the unit asked of
    it, at the instants the reference recorded.

    The published demand is asserted where it is the SAME statement on
    both engines. It is left out where the rule's need is bounded by the
    input's own supply here and not in muscadet (an empty upstream
    volume, a downstream one fed short): a convention of the rule layer
    that the volume does not touch, and that moves no delivery."""
    side, source, fill_rate, init, t_max, expected = SCENARIOS[scenario]
    system, held = unit_system(side, source, fill_rate, init)
    result = run(system, t_max, expected)
    for instant, (level, delivered, given, published) in expected.items():
        got = {
            "level": sampled(result, f"UNIT_buf_content_{held}", instant),
            "delivered": sampled(result, "SINK_y_fed_in", instant),
            "given": sampled(result, "SRC_a_fed_out", instant),
            "published": sampled(result, "UNIT_a_demand_in", instant),
        }
        assert close(got["level"], level), (instant, got)
        assert close(got["delivered"], delivered), (instant, got)
        assert close(got["given"], given), (instant, got)
        if published is not None:
            assert close(got["published"], published), (instant, got)


@pytest.mark.parametrize(
    "scenario, bound, date",
    [
        ("in_fill_inf", "reach_full", 10.0 / 3.0),
        ("out_fill_inf", "reach_full", 20.0 / 3.0),
        ("in_fill_half", "reach_full", 20.0),
        ("out_fill_half", "reach_full", 20.0),
        ("in_drain", "reach_empty", 6.0),
        ("out_drain", "reach_empty", 12.0),
    ],
)
def test_the_bound_is_located_at_its_closed_form_date(scenario, bound, date):
    side, source, fill_rate, init, t_max, _ = SCENARIOS[scenario]
    system, _held = unit_system(side, source, fill_rate, init)
    result = system.simulate(t_max=t_max)
    fired = [
        event.time
        for event in result.events
        if event.transition.rsplit(".", 1)[-1] == f"buf_{bound}"
    ]
    assert fired and abs(fired[0] - date) < CROSSING_TOL, (fired, date)


def test_a_volume_beside_the_rules_conserves():
    """What the source gave is what the rule drew plus what the volume
    holds, and what the rule drew is k times what it delivered: nothing
    made, nothing lost, on either side."""
    for side in ("in", "out"):
        system, held = unit_system(side, 5.0, 0.5)
        result = run(system, 25.0, [10.0])
        given = 10.0 * sampled(result, "SRC_a_fed_out", 10.0)
        delivered = 10.0 * sampled(result, "SINK_y_fed_in", 10.0)
        level = sampled(result, f"UNIT_buf_content_{held}", 10.0)
        stored_in_a = level if side == "in" else K * level
        assert close(given, K * delivered + stored_in_a), (side, given, level)


# --- the side a capacity resolves to -----------------------------------


def test_a_capacity_left_without_a_side_resolves_as_muscadet_does():
    """A rule input carried on the in side only resolves to ``"in"``,
    muscadet's resolution, and runs as the declared upstream volume."""
    declared, _ = unit_system("in", 5.0, 0.5)
    resolved, _ = unit_system(None, 5.0, 0.5)
    assert [c.side for c in resolved.comp["UNIT"].capacities] == ["in"]
    first = run(declared, 25.0, [10.0, 22.0])
    second = run(resolved, 25.0, [10.0, 22.0])
    for instant in (10.0, 22.0):
        for indicator in ("UNIT_buf_content_a", "SINK_y_fed_in", "SRC_a_fed_out"):
            assert sampled(first, indicator, instant) == sampled(
                second, indicator, instant
            )


def ambiguous_reactor(order: str) -> mu.ObjFlow:
    """A flow the rules consume AND that the component also declares as
    an output: a volume on it would sit upstream of the rules and on the
    flow's own way out at once."""
    obj = mu.ObjFlow("R")
    obj.add_flow_continuous_in(name="feed")
    obj.add_flow_continuous_out(name="feed")
    obj.add_flow_continuous_out(name="prod")

    def capacity():
        obj.add_capacity(name="hopper", flow="feed", capacity=50.0, side="in")

    def rules():
        obj.add_rule_set(
            name="convert", rules=[{"cons": {"feed": 1.0}, "prod": {"prod": 1.0}}]
        )

    halves = (capacity, rules) if order == "capacity first" else (rules, capacity)
    for half in halves:
        half()
    return obj


@pytest.mark.parametrize("order", ["capacity first", "rule set first"])
def test_a_volume_on_a_rule_flow_that_also_leaves_the_component_is_refused(order):
    """The one shape that still double counts: the volume would be both
    the rules' supply and the flow's transit, and what leaves it has two
    readings. Refused in either declaration order."""
    with pytest.raises(ValueError) as raised:
        ambiguous_reactor(order)
    message = str(raised.value)
    assert "`R`" in message and "`hopper`" in message and "`feed`" in message
    assert "both" in message


def test_a_volume_downstream_of_the_rules_bounds_them_with_no_consumer_connected():
    """A tank the rules fill, whose output nobody reads yet, still stops
    the rules once full: the volume's claim is a bound whether or not a
    consumer is connected, and without it the level ran past the volume
    (5, 25, 47.5 in a volume of ten)."""
    system, held = unit_system("out", 5.0, None)
    system._connections = [
        c for c in system._connections if c["from"]["component"] != "UNIT"
    ]
    result = run(system, 20.0, [2.0, 10.0, 19.0])
    for instant in (2.0, 10.0, 19.0):
        assert sampled(result, "UNIT_buf_content", instant) <= VOLUME + 1e-9


def test_two_rule_sets_drawing_one_flow_from_a_volume_upstream_are_refused():
    """Two rule sets each bounded by the whole inflow of an empty volume
    draw twice what arrives, and the volume went below zero (1, -1, -4,
    -8). How they share it is not decided, so the shape is refused
    rather than simulated, naming the component, the volume and the flow."""

    class Unit(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="y1")
            self.add_flow_continuous_out(name="y2")
            self.add_capacity(name="buf", flow="a", capacity=VOLUME, side="in")
            self.add_rule_set(name="r1", rules=[{"cons": {"a": 1.0}, "prod": {"y1": 1.0}}])
            self.add_rule_set(name="r2", rules=[{"cons": {"a": 1.0}, "prod": {"y2": 1.0}}])

    system = mu.System("u32_two_consumers")
    with pytest.raises(ValueError) as raised:
        system.add_component(Unit, "UNIT")
        system.build_dict()
    message = str(raised.value)
    assert "`UNIT`" in message and "`buf`" in message and "`a`" in message
    assert "`r1`" in message and "`r2`" in message


def test_a_volume_downstream_holding_a_consumed_flow_is_refused():
    """The mirror of the upstream refusal: a volume on the side the rules
    do not face holds a flow they consume, which it cannot sit beside."""

    class Unit(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="y")
            self.add_rule_set(name="conv", rules=[{"cons": {"a": 1.0}, "prod": {"y": 1.0}}])
            self.add_capacity(name="buf", flow="a", capacity=VOLUME, side="out")

    with pytest.raises(ValueError) as raised:
        mu.System("u32_wrong_side").add_component(Unit, "UNIT")
    message = str(raised.value)
    assert "`buf`" in message and "`a`" in message


# --- the membrane: two leaks, one volume, two venting rules ------------


def membrane_system(h2_sink: float, o2_sink: float):
    """The showcase's MembraneVoid as it declares itself: one volume over
    both leaks, upstream of two independent 1:1 venting rules."""

    class Leak(mu.ObjFlow):
        rate = 0.0
        flow = ""

        def add_flows(self):
            self.add_flow_continuous_out(name=self.flow, var_fed_default=self.rate)

    class H2Leak(Leak):
        rate, flow = 0.7, "H2_membrane_leak"

    class O2Leak(Leak):
        rate, flow = 0.015, "O2_membrane_leak"

    class MembraneVoid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2_membrane_leak")
            self.add_flow_continuous_in(name="O2_membrane_leak")
            self.add_flow_continuous_out(name="H2_leak")
            self.add_flow_continuous_out(name="O2_vent")
            self.add_capacity(
                name="membrane",
                flows=[
                    {"name": "H2_membrane_leak", "weight": 1.0},
                    {"name": "O2_membrane_leak", "weight": 1.0},
                ],
                capacity=100.0,
                side="in",
                content_init={},
                fill_rate=math.inf,
            )
            self.add_rule_set(
                name="hydrogen_release",
                rules=[{"name": "venting", "cons": {"H2_membrane_leak": 1.0},
                        "prod": {"H2_leak": 1.0}}],
            )
            self.add_rule_set(
                name="oxygen_release",
                rules=[{"name": "venting", "cons": {"O2_membrane_leak": 1.0},
                        "prod": {"O2_vent": 1.0}}],
            )

    def sink(flow, demand):
        class Sink(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_in(name=flow, var_demand_in_default=demand)

        return Sink

    system = mu.System("membrane")
    system.add_component(H2Leak, "SRC_H2")
    system.add_component(O2Leak, "SRC_O2")
    system.add_component(MembraneVoid, "MV")
    system.add_component(sink("H2_leak", h2_sink), "SINK_H2")
    system.add_component(sink("O2_vent", o2_sink), "SINK_O2")
    system.connect("SRC_H2", "H2_membrane_leak", "MV", "H2_membrane_leak")
    system.connect("SRC_O2", "O2_membrane_leak", "MV", "O2_membrane_leak")
    system.connect("MV", "H2_leak", "SINK_H2", "H2_leak")
    system.connect("MV", "O2_vent", "SINK_O2", "O2_vent")
    return system


def test_a_bottomless_vent_passes_both_leaks_straight_through():
    result = membrane_system(1000.0, 1000.0).simulate(t_max=50.0, samples=[5.0, 50.0])
    for instant in (5.0, 50.0):
        assert close(sampled(result, "MV_membrane_content_H2_membrane_leak", instant), 0.0)
        assert close(sampled(result, "MV_membrane_content_O2_membrane_leak", instant), 0.0)
        assert close(sampled(result, "SINK_H2_H2_leak_fed_in", instant), 0.7)
        assert close(sampled(result, "SINK_O2_O2_vent_fed_in", instant), 0.015)


def test_a_throttled_vent_buffers_its_leak_until_the_membrane_is_full():
    """H2 vented at 0.5 of the 0.7 leaking: the surplus fills the void at
    0.2 until the shared volume of 100 is full at t=500, after which the
    membrane asks upstream for what it vents. The oxygen, vented freely,
    never stocks and never goes below zero: emptiness is judged per
    constituent."""
    instants = [100.0, 400.0, 550.0, 590.0]
    result = membrane_system(0.5, 1000.0).simulate(t_max=600.0, samples=instants)
    for instant in instants:
        h2 = sampled(result, "MV_membrane_content_H2_membrane_leak", instant)
        o2 = sampled(result, "MV_membrane_content_O2_membrane_leak", instant)
        assert close(h2, 0.2 * min(instant, 500.0)), (instant, h2)
        assert close(o2, 0.0), (instant, o2)
        assert close(sampled(result, "SINK_H2_H2_leak_fed_in", instant), 0.5)
        assert close(sampled(result, "SINK_O2_O2_vent_fed_in", instant), 0.015)
    for instant in (550.0, 590.0):
        assert close(sampled(result, "MV_H2_membrane_leak_demand_in", instant), 0.5)
        assert close(sampled(result, "SRC_H2_H2_membrane_leak_fed_out", instant), 0.5)


def test_a_dead_end_vent_stocks_its_leak():
    """Nobody takes the oxygen: it accumulates at its leak rate while the
    hydrogen passes through."""
    result = membrane_system(1000.0, 0.0).simulate(t_max=600.0, samples=[300.0, 600.0])
    for instant in (300.0, 600.0):
        assert close(
            sampled(result, "MV_membrane_content_O2_membrane_leak", instant),
            0.015 * instant,
        )
        assert close(sampled(result, "MV_membrane_content_H2_membrane_leak", instant), 0.0)
        assert close(sampled(result, "SINK_H2_H2_leak_fed_in", instant), 0.7)
