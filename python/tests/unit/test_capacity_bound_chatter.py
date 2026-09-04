"""A volume that asks for nothing when full does not drain: it chatters,
and the engine now says so.

`add_capacity` documents that a full volume carries upstream "what it can
still take, which is what currently leaves it, capped by the demand
already passing through it", and warns that a volume declaring no
pass-through demand "asks for nothing once full, and therefore drains".

**It does not drain.** It asks for nothing, falls below its bound by the
width of the hysteresis band, leaves the bound, claims its fill rate
again, refills at once, and re-enters. The cycle turns at the scale of
the hysteresis, `1e-6` of the volume: for a volume of 4 drained at 1 that
is four microseconds a turn, a quarter of a million turns over five units
of time, and tens of millions over the horizon an industrial study asks
for.

Nothing saw it before. `WatchedLoop` counts watched firings **at one
instant** and these are at different instants; `FlowChattering` counts
active-set restarts **within one segment** and each turn is its own
segment. Time does advance, by a little, every turn. So the run did not
fail: it advanced, produced a trajectory that reads correctly at every
sample instant, and never finished.

`max_transition_firings` closes that. It is not a fix for the model, it
is the guarantee that a model like this fails loudly rather than
silently, which is the whole difference between a wrong answer and no
answer.

The cure for the model is one declaration, and it is asserted here beside
the diagnosis so the error message has something true to point at.
"""

import pytest

import pyraichu
import pyraichu.muscadet as mu

#: Where the buffer fills: capacity 4, net inflow 1 from an empty start.
FILLS_AT = 4.0


def buffer_system(pass_through: float, fill_rate: float = 2.0) -> mu.System:
    """A supply of 3, a buffer of 4 claiming `fill_rate` on top of the
    demand passing through it, and a load taking 1."""

    class Supply(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2", var_fed_default=3.0)

    class Buffer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(
                name="H2", var_demand_in_default=pass_through
            )
            self.add_flow_continuous_out(name="H2", var_fed_default=2.0)
            self.add_capacity(
                name="buffer",
                flow="H2",
                capacity=4.0,
                content_init={"H2": 0.0},
                fill_rate=fill_rate,
            )

    class Load(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2", var_demand_in_default=1.0)

    system = mu.System("buffer_claim")
    system.add_component(Supply, "S")
    system.add_component(Buffer, "T")
    system.add_component(Load, "L")
    system.connect("S", "H2", "T", "H2")
    system.connect("T", "H2", "L", "H2")
    return system


# --- 1. the model that is right ------------------------------------------


def test_a_buffer_that_declares_its_pass_through_demand_holds():
    """The cure, and it is one declaration.

    Full, the volume asks for what leaves it, so what enters equals what
    leaves and the content sits on the bound. Two events for the whole
    run: leaving empty, and reaching full."""
    result = buffer_system(pass_through=1.0).simulate(
        t_max=8.0, samples=[3.0, 5.0, 8.0]
    )
    assert len(result.events) == 2, result.events
    content = dict(result.samples["T_buffer_content"])
    entering = dict(result.samples["T_H2_fed_in"])
    leaving = dict(result.samples["T_H2_fed_out"])
    for instant in (5.0, 8.0):
        assert abs(content[instant] - 4.0) < 1e-9, instant
        assert abs(entering[instant] - leaving[instant]) < 1e-9, instant


def test_a_buffer_that_claims_nothing_never_fills_and_never_chatters():
    """The other stable model: with no claim the volume only asks for
    what passes through it, so it never fills and nothing crosses a
    bound.

    Between the two, the failure below is a property of the claim WITHOUT
    the demand, and not of the capacity, the outflow or the supply."""
    result = buffer_system(pass_through=0.0, fill_rate=0.0).simulate(t_max=5.0)
    assert len(result.events) == 0, result.events[:5]


# --- 2. the model that is under-declared, and the diagnosis --------------


def test_the_trajectory_is_exact_right_up_to_the_bound():
    """The chatter is at the bound and nowhere before it: up to t = 4 the
    buffer takes its claim of 2, delivers 1, and rises at 1 per unit
    time. A model that is wrong from the start would be a different
    finding."""
    result = buffer_system(pass_through=0.0).simulate(
        t_max=1.0, samples=[0.25, 0.75]
    )
    content = dict(result.samples["T_buffer_content"])
    assert abs(content[0.25] - 0.25) < 1e-6
    assert abs(content[0.75] - 0.75) < 1e-6


def test_an_under_declared_buffer_fails_loudly_at_its_bound():
    """What this guard exists for: the run stops, names the transition
    that ran away, and reports the mean step between its firings.

    That step is the number that makes the diagnosis land. Microseconds
    between firings of a bound automaton is a numerical scale, not a
    physical one, and no model of a plant means it."""
    with pytest.raises(pyraichu.SimulationError) as raised:
        buffer_system(pass_through=0.0).simulate(t_max=8.0)
    message = str(raised.value)
    assert "buffer_bounds" in message, message
    assert "chattering, not evolving" in message, message
    # It ran away at the bound, not before it.
    assert f"t={FILLS_AT}" in message.replace("t=4.000000000000002", "t=4.0"), message


def test_the_guard_can_be_lifted_for_a_genuinely_fast_model():
    """A cap that could not be raised would be a limit on what may be
    modelled rather than a diagnostic. Zero disables it, and the
    under-declared buffer then runs to its horizon as it used to: slowly,
    and wrongly, but by the modeller's choice."""
    result = buffer_system(pass_through=0.0).simulate(
        t_max=4.05, max_transition_firings=0
    )
    assert len(result.events) > 1000, len(result.events)


def test_the_budget_is_per_transition_and_not_per_run():
    """A model with many transitions firing normally is not caught by a
    cap on one of them, which is what makes the cap usable at a low
    value."""
    result = buffer_system(pass_through=1.0).simulate(
        t_max=8.0, max_transition_firings=3
    )
    assert len(result.events) == 2, result.events
