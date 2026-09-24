"""A temporised output's law: read under the name muscadet writes, and drawn
when it is a draw.

muscadet's read-back serialises a `FlowOutTempo` law as the cod3s object it
holds, so a declaration carries `{"cls": "DelayOccDistribution", "time": t}`
rather than the short `{"cls": "delay"}`; both spellings are one law.

A temporised output may also wait on an EXPONENTIAL law (`exp`, or the cod3s
`ExpOccDistribution`), which the platform offers beside the fixed delay. Up to
0.33.2 this layer refused it, a draw reduced to its mean being another model;
it now builds the tempo automaton's transition on that law, so the switch-on
(or switch-off) instant is drawn. The reference engine's behaviour was measured
before building (2026-09-24, 4000 replicas, switch-on at rate 1/400 h): the
probability of being switched on t hours after the production condition
starts to hold is 1 - exp(-t/400) at every sample, within one standard error.
That closed form is what the simulation test below holds the engine to.

What is pinned:

- the cod3s name of a delay builds the same law as the short name, and an
  instantaneous law is a zero delay unless it may not fire;
- an exponential law builds an exponential transition, under either
  spelling, and a malformed one is refused naming what is wrong;
- simulated, the switch-on probability follows the closed form.
"""

import math

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu
import pyraichu.muscadet_engine as engine


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


# --- the fixed delay -------------------------------------------------------


def test_the_cod3s_name_of_a_delay_builds_the_law_the_short_name_builds():
    written = tempo_of(
        a_relay({"cls": "DelayOccDistribution", "time": 400.0}, {"cls": "DelayOccDistribution", "time": 300.0})
    )
    short = tempo_of(a_relay({"cls": "delay", "time": 400.0}, {"cls": "delay", "time": 300.0}))

    assert written == short
    assert (written["enable"], written["disable"]) == (
        {"distrib": "delay", "time": 400.0},
        {"distrib": "delay", "time": 300.0},
    )


def test_the_cod3s_name_of_an_instantaneous_law_is_a_zero_delay():
    assert tempo_of(a_relay({"cls": "InstOccDistribution"}))["enable"] == {"distrib": "delay", "time": 0.0}


@pytest.mark.parametrize("probs", [[], [1], [1.0]])
def test_an_instantaneous_law_that_surely_fires_is_a_zero_delay(probs):
    """muscadet writes `probs` on every instantaneous law, empty by default."""
    enable = tempo_of(a_relay({"cls": "InstOccDistribution", "probs": probs}))["enable"]
    assert enable == {"distrib": "delay", "time": 0.0}


def test_an_instantaneous_law_that_may_not_fire_is_refused_naming_its_probabilities():
    """A firing probability below one is a draw, not a delay: reading it as a
    zero delay would build a sure, immediate enable in its place."""
    with pytest.raises(declare.ComponentSpecError, match="probs"):
        tempo_of(a_relay({"cls": "InstOccDistribution", "probs": [0.3, 0.7]}))


def authored_tempo(**tempo) -> dict:
    """The tempo a subclass declares through `add_flow_out_tempo`."""

    class Relay(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="cmd")
            self.add_flow_out_tempo(name="out", var_prod_cond=[["cmd"]], **tempo)

    system = mu.System(name="S")
    system.add_component(Relay, "RELAY")
    (flow,) = system.comp["RELAY"].flows_out
    return flow.tempo


def test_the_authoring_call_keeps_its_fixed_delay_keywords():
    """`enable_time` / `disable_time` are the short form of two delay laws."""
    tempo = authored_tempo(enable_time=2.0, disable_time=3.0)
    assert (tempo["enable"], tempo["disable"]) == (
        {"distrib": "delay", "time": 2.0},
        {"distrib": "delay", "time": 3.0},
    )


# --- the exponential law ---------------------------------------------------


@pytest.mark.parametrize("law", ["exp", "ExpOccDistribution"])
def test_an_exponential_law_builds_an_exponential_transition(law):
    tempo = tempo_of(a_relay({"cls": law, "rate": 0.0025}, {"cls": "delay", "time": 300.0}))
    assert tempo["enable"] == {"distrib": "exp", "rate": 0.0025}
    assert tempo["disable"] == {"distrib": "delay", "time": 300.0}


@pytest.mark.parametrize(
    ("declared", "named"),
    [
        ({"cls": "exp"}, "rate"),
        ({"cls": "exp", "rate": -1.0}, "rate"),
        ({"cls": "exp", "rate": "fast"}, "rate"),
        ({"cls": "exp", "rate": 0.1, "time": 3.0}, "time"),
    ],
    ids=["no-rate", "negative-rate", "rate-not-a-number", "unknown-key"],
)
def test_a_malformed_exponential_law_is_refused_naming_what_is_wrong(declared, named):
    with pytest.raises(declare.ComponentSpecError, match=named):
        tempo_of(a_relay(declared))


@pytest.mark.parametrize("law", ["weibull", "WeibullOccDistribution"])
def test_a_law_a_temporised_output_does_not_carry_is_still_refused_naming_it(law):
    with pytest.raises(declare.ComponentSpecError, match=f"occurrence law '{law}'"):
        tempo_of(a_relay({"cls": law, "scale": 1.0, "shape": 2.0}))


def test_the_authoring_call_takes_a_law_and_refuses_it_beside_a_time():
    assert authored_tempo(enable_law={"distrib": "exp", "rate": 0.1})["enable"] == {"distrib": "exp", "rate": 0.1}
    with pytest.raises(ValueError, match="enable_time"):
        authored_tempo(enable_time=2.0, enable_law={"distrib": "exp", "rate": 0.1})


def test_the_check_accepts_what_the_build_accepts():
    """The data-only check and the build read the law through one path."""
    assert declare.check_spec(a_relay({"cls": "DelayOccDistribution", "time": 2.0})) == "RELAY"
    assert declare.check_spec(a_relay({"cls": "ExpOccDistribution", "rate": 0.1})) == "RELAY"


# --- simulated against the closed form ------------------------------------

RATE = 1 / 400.0
SAMPLES = (250.0, 1000.0)
NB_RUNS = 4000


def _relay_document() -> dict:
    """A source always feeding a relay whose output switches on at `RATE`."""
    flow_common = {"var_type": "bool", "var_fed_default": False, "component_authorized": [{"class_name_bkd": ".*"}]}

    def component(name, flows):
        return {
            "name": name,
            "cls": "ObjFlow",
            "flows": flows,
            "capacities": [],
            "measurements_in": [],
            "measurements_out": [],
            "rules": [],
            "transfers": [],
            "automata": [],
            "failure_modes": [],
        }

    return {
        "version": "1.0.0",
        "name": "tempo_exp",
        "generated_indicators": True,
        "components": {
            "SRC": component("SRC", [{**flow_common, "cls": "FlowOut", "name": "cmd", "var_prod_default": True}]),
            "RELAY": component(
                "RELAY",
                [
                    {**flow_common, "cls": "FlowIn", "name": "cmd"},
                    {
                        **flow_common,
                        "cls": "FlowOutTempo",
                        "name": "out",
                        "var_prod_cond": [["cmd"]],
                        "occ_enable_flow": {"cls": "ExpOccDistribution", "rate": RATE},
                        "occ_disable_flow": {"cls": "DelayOccDistribution", "time": 0.0},
                    },
                ],
            ),
        },
        "connections": [
            {"source": "SRC", "source_box": "cmd_out", "target": "RELAY", "target_box": "cmd_in", "flow": "cmd"}
        ],
        "indicators": [],
    }


def test_the_switch_on_instant_is_exponential():
    """P(switched on by t) = 1 - exp(-rate * t), the reference engine's reading.

    Tolerance is four standard errors of a proportion over `NB_RUNS` replicas
    at a fixed seed, so the decision is reproducible and a correct engine
    fails it with negligible probability.
    """
    result = engine.simulate(_relay_document(), {"nb_runs": NB_RUNS, "schedule": list(SAMPLES), "seed": 11})
    fed = result.indicators["RELAY_out_fed_out"]
    for instant, mean in zip(fed.instants, fed.mean):
        expected = 1 - math.exp(-RATE * instant)
        standard_error = math.sqrt(expected * (1 - expected) / NB_RUNS)
        assert abs(mean - expected) < 4 * standard_error, (
            f"at {instant} h the output is on in {mean:.4f} of the replicas, where an exponential "
            f"switch-on at rate {RATE} gives {expected:.4f} (4 standard errors: {4 * standard_error:.4f})"
        )


# --- the edges of the law ---------------------------------------------------


def test_a_zero_rate_is_a_transition_that_never_fires():
    """cod3s writes `rate: 0` by default, and the reference engine reads it as
    a transition that never fires. Built, it must run, with the output never
    switched on, rather than pass the check and be refused by the engine."""
    document = _relay_document()
    document["components"]["RELAY"]["flows"][1]["occ_enable_flow"] = {"cls": "ExpOccDistribution", "rate": 0.0}
    assert declare.check_spec(document["components"]["RELAY"]) == "RELAY"
    result = engine.simulate(document, {"nb_runs": 20, "schedule": list(SAMPLES), "seed": 3})
    assert list(result.indicators["RELAY_out_fed_out"].mean) == [0.0, 0.0]


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_parameter_that_is_no_number_is_refused_on_both_routes(value):
    with pytest.raises(declare.ComponentSpecError, match="rate"):
        tempo_of(a_relay({"cls": "exp", "rate": value}))
    with pytest.raises(declare.ComponentSpecError, match="time"):
        tempo_of(a_relay({"cls": "delay", "time": value}))
    with pytest.raises(ValueError, match="rate"):
        authored_tempo(enable_law={"distrib": "exp", "rate": value})


def test_the_plugin_section_carries_the_law_too():
    """A plugin writing a law under `tempo` gets it, rather than a zero delay."""
    import pyraichu

    document = {
        "name": "relay",
        "plugins": {
            "muscadet": {
                "objects": [
                    {"type": "ObjFlow", "name": "SRC", "flows_out": [{"name": "cmd", "var_prod_default": True}]},
                    {
                        "type": "ObjFlow",
                        "name": "RELAY",
                        "flows_in": [{"name": "cmd"}],
                        "flows_out": [
                            {
                                "name": "out",
                                "var_prod_cond": [["cmd"]],
                                "tempo": {"enable_law": {"distrib": "exp", "rate": RATE}},
                            }
                        ],
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {"from": {"component": "SRC", "port": "cmd_out"}, "to": {"component": "RELAY", "port": "cmd_in"}}
        ],
        "indicators": [],
    }
    body = pyraichu.model_body(pyraichu.expand_model(document))
    relay = next(c for c in body["components"] if c["name"] == "RELAY")
    (automaton,) = [a for a in relay["automata"] if a["name"] == "out_tempo"]
    enable = next(t for t in automaton["transitions"] if t["name"] == "out_enable")
    assert (enable["distrib"], enable["rate"]) == ("exp", RATE)
