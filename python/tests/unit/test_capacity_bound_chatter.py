"""A buffer that claims a fill rate and also delivers chatters at its
full bound, forever, and nothing says so.

This is a **known defect**, pinned here rather than left to be found
again. It was found while authoring an industrial plant whose buffer does
exactly this, and it is what stops that plant from running at all.

What happens
------------

A volume declaring `fill_rate` claims that rate for itself while it has
room. Give it an outflow as well and, the instant it fills, the two
compete:

1. the volume reaches its bound and stops taking;
2. its outflow keeps running, so the content falls below the bound;
3. the bound is left, the claim resumes, and the volume refills at once;
4. the bound is reached again.

The cycle repeats every **four microseconds**, which is the
event-location tolerance: at t = 4 exactly, where the buffer below fills,
and 62 500 times per half unit of time thereafter. The content sits at
3.999997 and never moves. The hysteresis band that exists to prevent
exactly this does not, because the claim of 2 against a drain of 1 carries
the content back across the band inside a single step.

Why nothing catches it
----------------------

The engine has two guards and neither can see this. `WatchedLoop` counts
watched firings **at one instant**, and these are at different instants,
four microseconds apart. `FlowChattering` counts active-set restarts
within one integration segment, and each of these cycles is its own
segment. So the run does not fail: it advances, emitting a quarter of a
million events over five units of time, and would emit tens of millions
over the twenty units the plant's own study asks for.

That is the serious half. A simulator that produces a non-physical
trajectory must fail loudly; this one runs, and the only symptom is that
it never finishes.

What this test asserts
----------------------

The **correct** behaviour, marked `xfail(strict=True)`: a buffer filling
against a steady supply reaches its bound once and stays there. The day
the defect is fixed, this test passes, strict mode turns that into a
failure, and whoever fixed it is told to delete the marker.
"""

import pytest

import pyraichu.muscadet as mu

#: Where the buffer fills: capacity 4, net inflow 1 from an empty start.
FILLS_AT = 4.0


def buffer_with_claim(fill_rate: float) -> mu.System:
    """A supply of 3, a buffer of 4 claiming `fill_rate` on top of the
    demand passing through it, and a load taking 1."""

    class Supply(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2", var_fed_default=3.0)

    class Buffer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
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


def test_a_buffer_that_claims_nothing_is_stable():
    """The control: with no claim the buffer never fills, because it
    only asks for what passes through it, and nothing chatters.

    This is what makes the failure below a property of the claim and not
    of the capacity, the outflow or the supply."""
    result = buffer_with_claim(fill_rate=0.0).simulate(t_max=5.0)
    assert len(result.events) == 0, result.events[:5]


def test_a_buffer_reaches_its_bound_before_it_fills():
    """Up to the bound the trajectory is exact: the buffer takes its
    claim of 2, delivers 1, and rises at 1 per unit time."""
    result = buffer_with_claim(fill_rate=2.0).simulate(
        t_max=1.0, samples=[0.25, 0.75]
    )
    content = dict(result.samples["T_buffer_content"])
    assert abs(content[0.25] - 0.25) < 1e-6
    assert abs(content[0.75] - 0.75) < 1e-6


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known defect: a buffer claiming a fill rate chatters at its full "
        "bound, every 4 us, without terminating and without either Zeno "
        "guard seeing it. Delete this marker when it is fixed."
    ),
)
def test_a_full_buffer_settles_on_its_bound():
    """What should happen: the buffer reaches its bound once, at t = 4,
    and stays there.

    A handful of events is the whole of a correct run: entering the
    bound, and whatever the flow resolution needs to settle around it.
    What happens instead is 125 000."""
    result = buffer_with_claim(fill_rate=2.0).simulate(t_max=4.5)
    assert len(result.events) < 10, (
        f"{len(result.events)} events; the bound is being crossed "
        f"repeatedly from t={result.events[1].time if len(result.events) > 1 else '?'}"
    )


def test_the_chatter_is_at_the_bound_and_not_before_it():
    """The defect is localised, so a fix can be judged: every event past
    the first is at or after the instant the buffer fills, and they are
    spaced by the event-location tolerance rather than by anything
    physical."""
    result = buffer_with_claim(fill_rate=2.0).simulate(t_max=4.2)
    events = result.events
    assert len(events) > 1000, "the defect has changed shape; re-read it"
    assert all(event.time >= FILLS_AT - 1e-6 for event in events[1:])
    gaps = [b.time - a.time for a, b in zip(events[2:20], events[3:21])]
    assert all(abs(gap - 4e-6) < 1e-7 for gap in gaps), gaps
