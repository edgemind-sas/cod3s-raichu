"""A pure sink takes its demand and hands the rest back (muscadet 6.1.0).

A continuous input nothing on its component consumes, transfers or groups
ends there: a sink. Fed by several suppliers, each serving it the demand it
published in full, it used to keep the sum of the offers, so two sources of
9 and 3 behind a demand of 6 delivered 12. It now keeps its demand and
returns the surplus pro rata of what each supplier offered, exactly as a
rule consuming the flow one for one does (`surplus_release`, R-12).

Two things do not move: a single supplier already serves no more than the
demand, and an unbounded demand still takes everything offered.

Oracle-free: both routes run on RAICHU. Component and flow names are
invented.
"""

from __future__ import annotations

import math

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled
from test_declare_convert import component, document, link
from test_declare_convert_continuous import c_in, c_out, proved, uses_of

DEMAND = 6.0


def sink_plant(offers, demand: float = DEMAND, machine: bool = False):
    """One source per offer, all feeding `R`: a plain sink, or a machine
    turning `q` into `p` one for one in front of a consumer."""

    def source(rate):
        class Source(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_out(name="q", var_fed_default=rate)

        return Source

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="q", var_demand_in_default=demand)

    class Machine(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="q")
            self.add_flow_continuous_out(name="p")
            self.add_rule_set(name="r", rules=[{"cons": {"q": 1.0}, "prod": {"p": 1.0}}])

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="p", var_demand_in_default=demand)

    system = mu.System("sink_release")
    for i, rate in enumerate(offers):
        system.add_component(source(rate), f"S{i}")
    if machine:
        system.add_component(Machine, "R")
        system.add_component(Consumer, "C")
        system.connect("R", "p", "C", "p")
    else:
        system.add_component(Sink, "R")
    for i in range(len(offers)):
        system.connect(f"S{i}", "q", "R", "q")
    return system.simulate(t_max=1.0, samples=[0.5])


CASES = [
    # offers, what each source delivers, total received
    ((10.0,), (6.0,), 6.0),
    ((9.0, 3.0), (4.0, 2.0), 6.0),
    ((4.0, 3.0), (24.0 / 7.0, 18.0 / 7.0), 6.0),
    ((5.0, 5.0, 5.0), (2.0, 2.0, 2.0), 6.0),
    ((2.0, 3.0), (2.0, 3.0), 5.0),
]


@pytest.mark.parametrize("offers, delivered, received", CASES)
def test_a_sink_keeps_its_demand_and_releases_the_rest(offers, delivered, received):
    result = sink_plant(offers)

    assert abs(sampled(result, "R_q_fed_in", 0.5) - received) < CROSSING_TOL
    for i, expected in enumerate(delivered):
        assert abs(sampled(result, f"S{i}_q_fed_out", 0.5) - expected) < CROSSING_TOL


@pytest.mark.parametrize("offers", [(9.0, 3.0), (4.0, 3.0), (5.0, 5.0, 5.0)])
def test_a_sink_draws_what_a_one_for_one_machine_draws(offers):
    """One need, one draw, whether a rule or a plain consumer carries it."""
    sink = sink_plant(offers)
    machine = sink_plant(offers, machine=True)

    for i in range(len(offers)):
        name = f"S{i}_q_fed_out"
        assert abs(sampled(sink, name, 0.5) - sampled(machine, name, 0.5)) < CROSSING_TOL


@pytest.mark.parametrize("demand", [1e30, math.inf])
def test_an_unbounded_sink_still_takes_everything(demand):
    result = sink_plant((9.0, 3.0), demand=demand)

    assert abs(sampled(result, "R_q_fed_in", 0.5) - 12.0) < CROSSING_TOL


def converted_sink_plant() -> dict:
    return document(
        [
            component("S1", [c_out("q", 9.0)], class_name="Source"),
            component("S2", [c_out("q", 3.0)], class_name="Source"),
            component("R", [c_in("q", DEMAND)], class_name="Load"),
        ],
        [link("S1", "R", "q"), link("S2", "R", "q")],
    )


def test_the_library_route_releases_the_sink_as_the_plugin_does():
    """The converted sink applies `surplus_release` against its demand and
    agrees exactly with the declaration route."""
    conversion = proved(converted_sink_plant())

    releases = [u for u in uses_of(conversion, "Load") if u["fragment"] == "muscadet.surplus_release"]
    assert releases == [{"fragment": "muscadet.surplus_release", "with": {"name": "q"}}]
