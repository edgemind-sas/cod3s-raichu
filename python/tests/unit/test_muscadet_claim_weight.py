"""A split weighs each consumer at the demand it declares (muscadet 5.24.0).

The proportional split reads each consumer's claim as a weight, and the
claim is the demand as declared: consumers in parallel on a supply that
falls short are all served one fraction of what they ask. A source of 12
shared by claims of 6 and 18 serves 3 and 9. From muscadet 5.11.0 to
5.23.0 a claim was truncated at the supply before the split read it, so
the same claims weighed 6 and 12, served 4 and 8, and the split favoured
whoever asked least.

Two consequences are pinned here beside the rule:

* **an unbounded claim takes everything.** It is the limit of a growing
  one, so a finite claim beside it is served nothing, through a
  pass-through pipe as well as directly;
* **a rule claims its input at the need behind it**, bounded by what its
  outputs are asked and by what its OTHER inputs could sustain, never by
  its own supplier. Otherwise one need would weigh differently according
  to whether a rule or a plain consumer carries it.
"""

import math

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

SUPPLY = 5.0
NEED = 2.0


def shared_source(asked: float, through_pipe: bool, supply: float = SUPPLY, **policy):
    """One source, a modest consumer, and a greedy one, directly or through
    a pass-through pipe."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="w", var_fed_default=supply, **policy)

    class Pipe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_flow_continuous_out(name="w")

    class Greedy(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=asked)

    class Modest(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=NEED)

    system = mu.System("claim_weight")
    system.add_component(Source, "S")
    system.add_component(Greedy, "G")
    system.add_component(Modest, "M")
    system.connect("S", "w", "M", "w")
    if through_pipe:
        system.add_component(Pipe, "P")
        system.connect("S", "w", "P", "w")
        system.connect("P", "w", "G", "w")
    else:
        system.connect("S", "w", "G", "w")
    return system.simulate(t_max=1.0, samples=[0.5])


@pytest.mark.parametrize("through_pipe", [False, True])
@pytest.mark.parametrize("asked", [10.0, 1000.0, 1e6])
def test_a_claim_above_the_supply_weighs_as_declared(asked, through_pipe):
    """5 between claims of 2 and `asked`: one fraction of each, 5 / (asked + 2).
    The modest consumer receives 0.83, 0.00998 and 1e-5."""
    result = shared_source(asked, through_pipe)

    served = SUPPLY / (asked + NEED)
    assert abs(sampled(result, "G_w_fed_in", 0.5) - asked * served) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5) - NEED * served) < CROSSING_TOL


@pytest.mark.parametrize("through_pipe", [False, True])
@pytest.mark.parametrize("asked", [1e30, math.inf])
def test_an_unbounded_claim_takes_the_whole_supply(asked, through_pipe):
    """Declared infinite or at the engine's unbounded magnitude, the claim
    is handed everything and the finite one beside it nothing."""
    result = shared_source(asked, through_pipe)

    assert abs(sampled(result, "G_w_fed_in", 0.5) - SUPPLY) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5)) < CROSSING_TOL


def test_the_published_demand_is_the_weight():
    """Nothing stands between what a consumer asks and what the split
    weighs: the producer's total demand is the sum of the claims."""
    result = shared_source(1000.0, through_pipe=False)

    assert sampled(result, "G_w_demand_in", 0.5) == pytest.approx(1000.0)
    assert sampled(result, "S_w_demand_out", 0.5) == pytest.approx(1000.0 + NEED)


def test_claims_within_the_supply_are_served_whole():
    """Nothing to share out: 2 and 3 of a supply of 6 are both served."""
    result = shared_source(3.0, through_pipe=False, supply=6.0)

    assert abs(sampled(result, "G_w_fed_in", 0.5) - 3.0) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5) - 2.0) < CROSSING_TOL


def test_an_ordered_priority_does_not_move():
    """Priority serves in order and never weighs a claim."""
    result = shared_source(
        1e6,
        through_pipe=False,
        allocation="priority",
        allocation_priorities={"G": 1, "M": 2},
    )

    assert abs(sampled(result, "G_w_fed_in", 0.5) - SUPPLY) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5)) < CROSSING_TOL


def one_need_two_carriers(carried_by_a_rule: bool):
    """A source of 12, a consumer asking 6, and a need of 18 carried by a
    plain consumer or by a one-to-one rule whose output is asked 18."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="w", var_fed_default=12.0)

    class Small(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=6.0)

    class Large(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=18.0)

    class Transformer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_flow_continuous_out(name="p")
            self.add_rule_set(name="make", rules=[{"cons": {"w": 1.0}, "prod": {"p": 1.0}}])

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="p", var_demand_in_default=18.0)

    system = mu.System("one_need_two_carriers")
    system.add_component(Source, "S")
    system.add_component(Small, "C")
    system.connect("S", "w", "C", "w")
    if carried_by_a_rule:
        system.add_component(Transformer, "T")
        system.add_component(Sink, "K")
        system.connect("S", "w", "T", "w")
        system.connect("T", "p", "K", "p")
    else:
        system.add_component(Large, "T")
        system.connect("S", "w", "T", "w")
    return system.simulate(t_max=1.0, samples=[0.5])


@pytest.mark.parametrize("carried_by_a_rule", [False, True])
def test_a_need_weighs_the_same_behind_a_rule(carried_by_a_rule):
    """12 between 6 and 18 is 3 and 9, whether the 18 is a plain demand or
    the input of a rule asked 18 downstream. Bounded by its own supplier,
    the rule claimed the 12 the source could give and was served 8."""
    result = one_need_two_carriers(carried_by_a_rule)

    assert sampled(result, "T_w_demand_in", 0.5) == pytest.approx(18.0)
    assert abs(sampled(result, "C_w_fed_in", 0.5) - 3.0) < CROSSING_TOL
    assert abs(sampled(result, "T_w_fed_in", 0.5) - 9.0) < CROSSING_TOL


def test_a_rule_claim_is_bounded_by_its_other_inputs():
    """A rule drawing water and power one to one, asked 18, with 4 of
    power to run on: it claims 4 of water, what the power could sustain,
    and the 12 of water are split 6 : 4 between the two claims, which fit."""

    class Water(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="w", var_fed_default=12.0)

    class Power(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="e", var_fed_default=4.0)

    class Small(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=6.0)

    class Machine(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="p")
            self.add_rule_set(
                name="make", rules=[{"cons": {"w": 1.0, "e": 1.0}, "prod": {"p": 1.0}}]
            )

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="p", var_demand_in_default=18.0)

    system = mu.System("claim_bounded_by_the_other_input")
    for cls, name in ((Water, "W"), (Power, "E"), (Small, "C"), (Machine, "T"), (Sink, "K")):
        system.add_component(cls, name)
    system.connect("W", "w", "C", "w")
    system.connect("W", "w", "T", "w")
    system.connect("E", "e", "T", "e")
    system.connect("T", "p", "K", "p")
    result = system.simulate(t_max=1.0, samples=[0.5])

    assert sampled(result, "T_w_demand_in", 0.5) == pytest.approx(4.0)
    # The power is the rule's only other input, and nothing bounds the
    # claim on it but what the output is asked and the water could give.
    assert sampled(result, "T_e_demand_in", 0.5) == pytest.approx(12.0)
    assert abs(sampled(result, "C_w_fed_in", 0.5) - 6.0) < CROSSING_TOL
    assert abs(sampled(result, "T_w_fed_in", 0.5) - 4.0) < CROSSING_TOL
    assert abs(sampled(result, "K_p_fed_in", 0.5) - 4.0) < CROSSING_TOL
