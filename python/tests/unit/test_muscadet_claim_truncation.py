"""A split weighs each consumer at its demand truncated at the supply.

The proportional split reads each consumer's claim as a weight. Weighed at
its raw demand, a claim far above the supply outweighed everyone else: a
source of 5 shared by a consumer asking 2 and one asking `D` served the
first 0.83, 0.01 and 1e-5 for `D` = 10, 1000 and 1e6. Truncated at the
quantity available, the claim of `D` weighs 5 whenever `D` reaches the
supply, and the split is 5 against 2 whatever `D` is, through a
pass-through pipe as well as directly. The demand the consumer publishes
is not touched: the truncation belongs to the split.
"""

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

SUPPLY = 5.0
NEED = 2.0
GREEDY_SHARE = SUPPLY * SUPPLY / (SUPPLY + NEED)
MODEST_SHARE = SUPPLY * NEED / (SUPPLY + NEED)


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

    system = mu.System("claim_truncation")
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
@pytest.mark.parametrize("asked", [10.0, 1000.0, 1e6, 1e30])
def test_a_claim_above_the_supply_weighs_as_the_supply(asked, through_pipe):
    result = shared_source(asked, through_pipe)

    assert abs(sampled(result, "G_w_fed_in", 0.5) - GREEDY_SHARE) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5) - MODEST_SHARE) < CROSSING_TOL


def test_the_published_demand_stays_as_declared():
    """The truncation belongs to the split: the need is still what is read."""
    result = shared_source(1000.0, through_pipe=False)

    assert sampled(result, "G_w_demand_in", 0.5) == pytest.approx(1000.0)
    assert sampled(result, "S_w_demand_out", 0.5) == pytest.approx(1000.0 + NEED)


def test_claims_below_the_supply_split_as_declared():
    """Nothing to truncate: 2 and 3 of a supply of 4 weigh 2 : 3."""
    result = shared_source(3.0, through_pipe=False, supply=4.0)

    assert abs(sampled(result, "G_w_fed_in", 0.5) - 2.4) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5) - 1.6) < CROSSING_TOL


def test_an_ordered_priority_does_not_move():
    """Priority never hands a consumer more than the supply anyway."""
    result = shared_source(
        1e6,
        through_pipe=False,
        allocation="priority",
        allocation_priorities={"G": 1, "M": 2},
    )

    assert abs(sampled(result, "G_w_fed_in", 0.5) - SUPPLY) < CROSSING_TOL
    assert abs(sampled(result, "M_w_fed_in", 0.5)) < CROSSING_TOL
