"""The second limit-cycle budget, and the fact that Python can reach it.

0.21.0 gave the engine two budgets against a limit cycle, a model that
advances time by a little every turn and never gets anywhere:
`max_transition_firings` for a mode that flips, and `max_flow_restarts`
for a network whose routing does. Only the first reached the Python
binding, which is where nearly every model is authored: the second was
settable from Rust alone, so the guard against the harder of the two
cycles was, in practice, unreachable and unliftable.

The two cover different things, and one does not stand in for the other.
The under-declared buffer of `test_capacity_bound_chatter` fires a
million transitions without ever restarting the flow active set; the
model below restarts the active set on a saturation crossing while its
transitions behave. A budget that only caught the first would leave the
second class of cycle exactly as invisible as before.

What is pinned here is reach and effect, not a contrived cycle: a
legitimate model that crosses one flow boundary, a cap set below what it
needs, and the diagnosis naming the edge that crossed.
"""

import pytest

import pyraichu
import pyraichu.muscadet as mu

#: A supply that swings between 0 and 8 over a period of 6.
WIND = {
    "cls": "SinusoidalProfile",
    "amplitude": 4.0,
    "offset": 4.0,
    "period": 6.0,
    "phase_shift": 4.0,
}

#: Where the supply meets the demand and the network re-routes. Read off
#: the diagnosis, not predicted: it is the instant the two tanks stop
#: being served in full.
CROSSES_AT = 1.7238756097076116


def two_tanks() -> mu.System:
    """A swinging supply split in equal shares between two tanks that
    each ask for 5. While the supply exceeds 10 both are served in full;
    below it the shares bind, and that crossing is one restart of the
    flow active set."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(
                name="E",
                var_fed_default=8.0,
                profile=WIND,
                allocation="shares",
                allocation_shares={"A": 0.5, "B": 0.5},
            )

    class Tank(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E", var_demand_in_default=5.0)
            self.add_capacity(
                name="tank", flow="E", capacity=10.0, content_init={"E": 0.0}
            )

    system = mu.System("two_tanks")
    system.add_component(Source, "S")
    system.add_component(Tank, "A")
    system.add_component(Tank, "B")
    system.connect("S", "E", "A", "E")
    system.connect("S", "E", "B", "E")
    return system


def test_the_model_runs_under_the_default_budget():
    """The premise: this is a legitimate model, not a cycle. Whatever the
    cap proves below, it proves about a run that is otherwise fine."""
    result = two_tanks().simulate(t_max=12.0)
    assert len(result.events) == 4, result.events


def test_the_flow_budget_is_reachable_from_python():
    """The fix itself. Before it, this keyword did not exist on
    `simulate` and the engine default was the only value obtainable from
    Python."""
    with pytest.raises(pyraichu.SimulationError) as raised:
        two_tanks().simulate(t_max=12.0, max_flow_restarts=1)
    message = str(raised.value)
    assert "active set of the continuous flow network restarted" in message
    assert "chattering, not evolving" in message


def test_the_diagnosis_names_the_edge_that_crossed():
    """What makes the error actionable: not that something re-routed, but
    which connection did, and when."""
    with pytest.raises(pyraichu.SimulationError) as raised:
        two_tanks().simulate(t_max=12.0, max_flow_restarts=1)
    message = str(raised.value)
    assert "S.E_alloc[S.E_out__alloc__A__E_in]" in message, message
    assert f"t={CROSSES_AT}" in message, message


def test_one_restart_more_is_enough():
    """The cap counts restarts and nothing else: this model needs exactly
    one, so two is the smallest value under which it runs. That is what
    makes the number in the diagnosis a measurement rather than a
    threshold effect."""
    result = two_tanks().simulate(t_max=12.0, max_flow_restarts=2)
    assert len(result.events) == 4, result.events


def test_the_guard_can_be_lifted():
    """Zero disables it, as for the transition budget: a genuinely
    fast-switching network is a legitimate thing to model, and a cap that
    could not be raised would be a limit on what may be modelled rather
    than a diagnostic."""
    result = two_tanks().simulate(t_max=12.0, max_flow_restarts=0)
    assert len(result.events) == 4, result.events


def test_the_two_budgets_are_independent():
    """Neither stands in for the other, which is why both must be
    reachable. Here the transition budget is set as low as this model
    allows and changes nothing, because what runs away in this class of
    cycle is the routing, not a mode."""
    result = two_tanks().simulate(
        t_max=12.0, max_transition_firings=2, max_flow_restarts=2
    )
    assert len(result.events) == 4, result.events
