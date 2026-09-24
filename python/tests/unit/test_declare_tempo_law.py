"""A temporised output's law, read under the name muscadet writes it with.

muscadet's read-back serialises a `FlowOutTempo` law as the cod3s object it
holds, so the declaration carries `{"cls": "DelayOccDistribution", "time": t}`
and never the short `{"cls": "delay"}` this layer was written against. Reading
only the short name refused every temporised output a platform study declares,
under a message ("only 'delay' and 'inst' are carried") that blamed the law
when the law was a delay all along.

What is pinned:

- the cod3s name of a delay builds the same flow as the short name;
- the cod3s name of an instantaneous law is a zero delay, as `inst` is;
- a random law is still refused by name, under either spelling, since a
  temporised output here waits a fixed time and reducing a draw to its mean
  would compute something else.
"""

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu


def a_relay(enable: dict, disable: dict | None = None) -> dict:
    """One component: a discrete input and a temporised output fed by it."""
    flow = {"cls": "FlowOutTempo", "name": "out", "var_prod_cond": [["cmd"]], "occ_enable_flow": enable}
    if disable is not None:
        flow["occ_disable_flow"] = disable
    return {"name": "RELAY", "flows": [{"cls": "FlowIn", "name": "cmd"}, flow]}


def tempo_of(spec: dict) -> dict:
    system = mu.System(name="S")
    relay = declare.build_component(system, spec)
    (flow,) = relay.flows_out
    return flow.tempo


def test_the_cod3s_name_of_a_delay_builds_the_flow_the_short_name_builds():
    written = tempo_of(
        a_relay({"cls": "DelayOccDistribution", "time": 400.0}, {"cls": "DelayOccDistribution", "time": 300.0})
    )
    short = tempo_of(a_relay({"cls": "delay", "time": 400.0}, {"cls": "delay", "time": 300.0}))

    assert written == short
    assert (written["enable_time"], written["disable_time"]) == (400.0, 300.0)


def test_the_cod3s_name_of_an_instantaneous_law_is_a_zero_delay():
    assert tempo_of(a_relay({"cls": "InstOccDistribution"}))["enable_time"] == 0.0


@pytest.mark.parametrize("probs", [[], [1], [1.0]])
def test_an_instantaneous_law_that_surely_fires_is_a_zero_delay(probs):
    """muscadet writes `probs` on every instantaneous law, empty by default."""
    assert tempo_of(a_relay({"cls": "InstOccDistribution", "probs": probs}))["enable_time"] == 0.0


def test_an_instantaneous_law_that_may_not_fire_is_refused_naming_its_probabilities():
    """A firing probability below one is a draw, not a delay: reading it as a
    zero delay would build a sure, immediate enable in its place."""
    with pytest.raises(declare.ComponentSpecError, match="probs"):
        tempo_of(a_relay({"cls": "InstOccDistribution", "probs": [0.3, 0.7]}))


@pytest.mark.parametrize("law", ["exp", "ExpOccDistribution"])
def test_a_random_law_is_still_refused_naming_it(law):
    with pytest.raises(declare.ComponentSpecError, match=f"occurrence law '{law}'"):
        tempo_of(a_relay({"cls": law, "rate": 1e-3}))


def test_the_check_accepts_what_the_build_accepts():
    """The data-only check and the build read the law through one path."""
    assert declare.check_spec(a_relay({"cls": "DelayOccDistribution", "time": 2.0})) == "RELAY"
