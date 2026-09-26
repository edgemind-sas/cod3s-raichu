"""A discrete output that publishes the NEGATION of what it would deliver
(muscadet's ``FlowOut.negate``).

muscadet defines it on the output itself, not on the condition: the output
publishes ``not (production and active and available)``. Its demonstrator is
an inverter chain, source -> INV1 -> INV2 -> target, each inverter re-emitting
the negation of its input, so the target sees the source's value and INV1
between them sees its opposite (``examples/isimu/inverter_chain.py``).
"""

from __future__ import annotations

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import sampled

#: The source's delay mode: down on [4, 8), up again at 8.
INSTANTS = [2.0, 6.0, 10.0]
SOURCE = [True, False, True]


class Source(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_out(name="signal", var_prod_default=True)
        self.add_delay_failure_mode(name="fail_S", failure_time=4.0, repair_time=4.0)


class Inverter(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name="signal", logic="and")
        self.add_flow_out(name="signal", var_prod_cond=["signal"], negate=True)


class Target(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name="signal", logic="and")


def inverter_chain() -> mu.System:
    system = mu.System("inverter_chain")
    system.add_component(Source, "S")
    system.add_component(Inverter, "INV1")
    system.add_component(Inverter, "INV2")
    system.add_component(Target, "T")
    system.connect("S", "signal", "INV1", "signal")
    system.connect("INV1", "signal", "INV2", "signal")
    system.connect("INV2", "signal", "T", "signal")
    return system


def test_each_inverter_publishes_the_negation_of_its_input():
    result = inverter_chain().simulate(t_max=12.0, samples=INSTANTS)
    for instant, source in zip(INSTANTS, SOURCE):
        assert sampled(result, "S_signal_fed_out", instant) is source, instant
        assert sampled(result, "INV1_signal_fed_out", instant) is (not source), instant
        assert sampled(result, "INV2_signal_fed_out", instant) is source, instant
        assert sampled(result, "T_signal_fed_in", instant) is source, instant


def test_an_unavailable_inverter_feeds_nothing_downstream():
    """What a negated output breaks, and what muscadet reads instead. An
    inverter whose own gate a mode takes down publishes ``not (prod and
    active and False)``, which is True, while its availability is False.
    muscadet's consumer reads the feed AND the availability, so it reads
    False; reading the feed alone would read True."""

    class Unreliable(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="signal", logic="and")
            self.add_flow_out(name="signal", var_prod_cond=["signal"], negate=True)
            self.add_delay_failure_mode(
                name="fail", failure_time=1.0, repair_time=100.0
            )

    system = mu.System("unavailable_inverter")
    system.add_component(Source, "S")
    system.add_component(Unreliable, "INV")
    system.add_component(Target, "T")
    system.connect("S", "signal", "INV", "signal")
    system.connect("INV", "signal", "T", "signal")
    result = system.simulate(t_max=3.0, samples=[0.5, 2.0])
    # Healthy: the source feeds, the inverter publishes False, T reads False.
    assert sampled(result, "INV_signal_fed_out", 0.5) is False
    assert sampled(result, "T_signal_fed_in", 0.5) is False
    # Down: the inverter publishes True, its gate is False, T reads False.
    assert sampled(result, "INV_signal_fed_out", 2.0) is True
    assert sampled(result, "T_signal_fed_in", 2.0) is False


def test_an_output_that_is_not_negated_is_unchanged():
    """The default builds the expression it always built: a plain `and`
    over the production, the gate and nothing else."""

    class Plain(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="signal", logic="and")
            self.add_flow_out(name="signal", var_prod_cond=["signal"])

    system = mu.System("plain")
    system.add_component(Plain, "P")
    body = pyraichu.model_body(system.build_dict())
    component = next(c for c in body["components"] if c["name"] == "P")
    function = next(
        f
        for f in component["sensitive_functions"]
        if f["name"] == "update_signal_fed_out"
    )
    assert function["effects"][0]["value"]["bool_op"] == "and"


def test_the_plugin_section_carries_the_negation():
    document = {
        "name": "plugin_inverter",
        "generated_indicators": True,
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "INV",
                        "flows_in": [{"name": "signal", "logic": "and"}],
                        "flows_out": [
                            {
                                "name": "signal",
                                "var_prod_cond": ["signal"],
                                "negate": True,
                            }
                        ],
                    }
                ]
            }
        },
    }
    model = pyraichu.load_model(pyraichu.expand_model(document))
    # An unfed input reads False, so the negated output reads True.
    result = pyraichu.simulate(model, t_max=1.0, samples=[1.0])
    assert result.samples["INV_signal_fed_out"][-1][1] is True


@pytest.mark.parametrize("dynamic", ["tempo", "trigger"])
def test_a_dynamic_output_refuses_the_negation_on_the_plugin_section(dynamic):
    entry = {"name": "signal", "var_prod_cond": ["signal"], "negate": True, dynamic: {}}
    document = {
        "name": "dynamic_inverter",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "INV",
                        "flows_in": [{"name": "signal", "logic": "and"}],
                        "flows_out": [entry],
                    }
                ]
            }
        },
    }
    with pytest.raises(ValueError, match="negate"):
        pyraichu.expand_model(document)
