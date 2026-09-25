"""A component that carries a continuous flow in and out passes it on.

muscadet calls it the identity transfer (`get_identity_transfer_flows`,
its R31): a continuous flow declared on BOTH sides of a component, named
by no rule set, metered by no conduit, is transferred unchanged from the
input to the output of the same name. A plain pipe, a valve, a cable run
need no ceremonial same-in-same-out rule.

This layer used to derive an output's capability only from a rule, a
conduit, a volume or its declared rate, so such a component published its
declared rate (zero) and everything downstream of it read zero: the H2
showcase lost its whole water chain to it. What is pinned here:

- the capability crosses forward and the demand crosses back, so the
  consumer is served what it asks and the source is asked for it;
- what crosses is what ARRIVES, not what is asked: a scarce supply is
  handed on as it is;
- the output's own factors apply, as on any output: a derating halves
  what the pipe hands on;
- a pipe whose output nothing reads asks for nothing, so it takes no share
  of a supply a wired rival needs;
- a pipe whose input nothing feeds hands on nothing;
- the shapes that already say what the component does are untouched: a
  rule set naming the flow, a volume holding it.
"""

import pyraichu
import pyraichu.muscadet as mu
import pytest
from conftest import TOL, at_zero, sampled

PLENTY = 10.0
DEMAND = 1.5


def source_class(rate: float):
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="feed", var_fed_default=rate)

    return Source


def consumer_class(demand: float):
    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="feed", var_demand_in_default=demand)

    return Consumer


class Pipe(mu.ObjFlow):
    """One continuous flow in, the same one out, and nothing else."""

    def add_flows(self):
        self.add_flow_continuous_in(name="feed")
        self.add_flow_continuous_out(name="feed")


def chain(pipe=Pipe, rate: float = PLENTY, demand: float = DEMAND) -> mu.System:
    """Source, pipe, consumer."""
    system = mu.System("passthrough")
    system.add_component(source_class(rate), "SRC")
    system.add_component(pipe, "PIPE")
    system.add_component(consumer_class(demand), "CON")
    system.connect("SRC", "feed", "PIPE", "feed")
    system.connect("PIPE", "feed", "CON", "feed")
    return system


def test_a_pipe_hands_on_what_its_consumer_asks_for():
    """The ten-line repro of the showcase: 0 before, the demand now."""
    result = chain().simulate(t_max=1.0)
    assert abs(at_zero(result, "CON_feed_fed_in") - DEMAND) < TOL
    assert abs(at_zero(result, "PIPE_feed_fed_out") - DEMAND) < TOL
    assert abs(at_zero(result, "SRC_feed_fed_out") - DEMAND) < TOL


def test_the_capability_crosses_forward_and_the_demand_back():
    """Both channels cross, each in its own direction: the consumer sees
    what the source could give, the source sees what the consumer asks."""
    result = chain().simulate(t_max=1.0)
    assert abs(at_zero(result, "PIPE_feed_capability_out") - PLENTY) < TOL
    assert abs(at_zero(result, "CON_feed_capability_in") - PLENTY) < TOL
    assert abs(at_zero(result, "PIPE_feed_demand_in") - DEMAND) < TOL
    assert abs(at_zero(result, "SRC_feed_demand_out") - DEMAND) < TOL


def test_a_scarce_supply_is_handed_on_as_it_arrives():
    """What crosses is what arrived, never what was asked."""
    result = chain(rate=1.0).simulate(t_max=1.0)
    assert abs(at_zero(result, "CON_feed_fed_in") - 1.0) < TOL


def test_a_derated_pipe_hands_on_its_share_of_what_arrives():
    """The output's factors apply to what crosses, as on any output: the
    pipe still asks for the whole demand and hands on half of it."""

    class Leaky(Pipe):
        def add_flows(self):
            super().add_flows()
            self.add_delay_failure_mode(
                name="half",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("feed", 0.5)],
            )

    result = chain(pipe=Leaky).simulate(t_max=10.0, samples=[1.0, 5.0])
    assert abs(sampled(result, "CON_feed_fed_in", 1.0) - DEMAND) < TOL
    assert abs(sampled(result, "CON_feed_fed_in", 5.0) - DEMAND / 2) < TOL
    assert abs(sampled(result, "PIPE_feed_demand_in", 5.0) - DEMAND) < TOL


def test_a_dangling_pipe_takes_no_share_of_a_wired_rivals_supply():
    """An output nothing reads asks for nothing, so the pipe publishes no
    demand and the rival on the same source is served in full."""
    system = mu.System("dangling")
    system.add_component(source_class(6.0), "SRC")
    system.add_component(Pipe, "DANGLING")
    system.add_component(consumer_class(5.0), "RIVAL")
    system.connect("SRC", "feed", "DANGLING", "feed")
    system.connect("SRC", "feed", "RIVAL", "feed")
    result = system.simulate(t_max=1.0)
    assert abs(at_zero(result, "DANGLING_feed_demand_in")) < TOL
    assert abs(at_zero(result, "RIVAL_feed_fed_in") - 5.0) < TOL


def test_an_unfed_pipe_hands_on_nothing():
    """A pipe whose input nothing feeds reads its declared default, zero,
    and passes that on rather than inventing a supply."""
    system = mu.System("unfed")
    system.add_component(Pipe, "PIPE")
    system.add_component(consumer_class(DEMAND), "CON")
    system.connect("PIPE", "feed", "CON", "feed")
    result = system.simulate(t_max=1.0)
    assert abs(at_zero(result, "CON_feed_fed_in")) < TOL


def test_a_flow_a_rule_set_names_is_the_rules_business():
    """A rule consuming and producing the flow says what the component does
    with it, so the transfer stands aside: the output's capability is the
    rule's yield, not the whole of what arrives."""

    class Halver(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="feed")
            self.add_flow_continuous_out(name="feed")
            self.add_rule_set(
                name="halve", rules=[{"cons": {"feed": 1.0}, "prod": {"feed": 0.5}}]
            )

    result = chain(pipe=Halver).simulate(t_max=1.0)
    assert abs(at_zero(result, "PIPE_feed_capability_out") - PLENTY / 2) < TOL


def test_a_flow_a_volume_holds_keeps_its_own_path():
    """A volume on the flow decides what the output delivers, transit or
    not: a volume that does not transit and holds nothing delivers
    nothing, where a transfer would have handed the supply on."""

    class Reservoir(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="feed")
            self.add_flow_continuous_out(name="feed")
            self.add_capacity(name="tank", flow="feed", capacity=100.0, side="out")

    result = chain(pipe=Reservoir).simulate(t_max=1.0)
    assert abs(at_zero(result, "CON_feed_fed_in")) < TOL


def test_a_pipe_publishes_what_it_made_for_the_allocation():
    """The allocation of a pass-through distributes what crossed, as for a
    rule, so the output carries a `produced_out` beside its capability."""
    document = pyraichu.model_body(chain().build_dict())
    pipe = next(c for c in document["components"] if c["name"] == "PIPE")
    attributes = {attribute["name"] for attribute in pipe["attributes"]}
    assert "feed_produced_out" in attributes


def test_a_capped_pipe_asks_for_no_more_than_it_can_hand_on():
    """A ceiling on the pipe's output bounds what the pipe asks upstream
    too: asking for the whole downstream demand would take from a shared
    supply what the pipe can never pass on, and starve a rival."""

    class Capped(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="feed")
            self.add_flow_continuous_out(name="feed", max_rate=1.0)

    system = mu.System("capped")
    system.add_component(source_class(3.0), "SRC")
    system.add_component(Capped, "PIPE")
    system.add_component(consumer_class(2.0), "CON")
    system.add_component(consumer_class(2.0), "RIVAL")
    system.connect("SRC", "feed", "PIPE", "feed")
    system.connect("SRC", "feed", "RIVAL", "feed")
    system.connect("PIPE", "feed", "CON", "feed")
    result = system.simulate(t_max=1.0)
    assert abs(at_zero(result, "PIPE_feed_demand_in") - 1.0) < TOL
    assert abs(at_zero(result, "CON_feed_fed_in") - 1.0) < TOL
    assert abs(at_zero(result, "RIVAL_feed_fed_in") - 2.0) < TOL


def _sinusoid(amplitude, offset):
    return {
        "cls": "SinusoidalProfile", "amplitude": amplitude, "period": 24.0,
        "phase_shift": 6.0, "offset": offset, "value_min": 0.0,
        "value_max": float("inf"), "name": "day",
    }


def two_varying_suppliers(first_varies: bool = True):
    """A pipe fed by two sources whose rates follow the day, and a
    consumer asking more than both: the H2 plant's electrical bus."""

    def supply(rate, profile):
        class Source(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_out(name="feed", var_fed_default=rate, profile=profile)

        return Source

    class Pipe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="feed")
            self.add_flow_continuous_out(name="feed")

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="feed", var_demand_in_default=100.0)

    system = mu.System("passthrough_two_varying")
    system.add_component(supply(3.0, _sinusoid(0.5, 0.5) if first_varies else None), "S1")
    system.add_component(supply(5.0, _sinusoid(0.75, 0.25)), "S2")
    system.add_component(Pipe, "PIPE")
    system.add_component(Consumer, "CON")
    system.connect("S1", "feed", "PIPE", "feed")
    system.connect("S2", "feed", "PIPE", "feed")
    system.connect("PIPE", "feed", "CON", "feed")
    return system


@pytest.mark.parametrize("first_varies", [True, False])
def test_a_pipe_sums_every_supplier_while_they_vary(first_varies):
    """An input sums its suppliers only once every one of them has
    published: summed at the FIRST supplier's visit, the later ones were
    read at the previous evaluation's value, which the discrete fixpoint
    hides and a continuously varying source exposes (the H2 plant's bus
    read the PV alone, 9.0 where the reference reads 9.81). Each source
    publishes its own rate correctly, so the pipe must carry their sum."""
    from test_rule_input_surplus import _forward_reads

    system = two_varying_suppliers(first_varies)
    assert _forward_reads(system.build_dict()) == []
    result = system.simulate(t_max=12.0, samples=[3.0, 6.0])
    for instant in (3.0, 6.0):
        total = sampled(result, "S1_feed_fed_out", instant) + sampled(result, "S2_feed_fed_out", instant)
        assert abs(sampled(result, "PIPE_feed_fed_in", instant) - total) < 1e-9
        assert abs(sampled(result, "PIPE_feed_capability_in", instant) - total) < 1e-9
        assert abs(sampled(result, "CON_feed_fed_in", instant) - total) < 1e-9
