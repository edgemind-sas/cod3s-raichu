"""A rule set none of whose outputs is read runs at its nominal scale.

muscadet sizes a rule from what its outputs are asked for, over the outputs
something can ask: a connection, or a volume downstream of the rule. With
none, the rule is not unbounded: it runs and claims at its nominal scale of
1, its declared coefficients, where its inputs allow it. A machine nothing
is connected behind therefore does not draw its supplier's whole capability,
and takes no more than its declared need of a supply it shares.

Every expected number is the muscadet reference engine's, recorded on the
same models. Component and flow names are invented.
"""

from __future__ import annotations

import pytest

import pyraichu.muscadet as mu
from pyraichu.muscadet.convert import compare_routes, convert
from test_declare_convert import component, document, link
from test_declare_convert_continuous import c_in, c_out, rule


def plant(cons, prod, *, rate=10.0, rival=None, others=(), wired=None) -> mu.System:
    """A machine `T` making `prod` out of `cons`, fed `w` by a source of
    `rate` it may share with a `rival` load. `others` are further sources,
    `wired` the outputs a load asks for, by demand."""

    def source(flow: str, supply: float) -> type:
        class Source(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_out(name=flow, var_fed_default=supply)

        return Source

    def load(flow: str, demand: float) -> type:
        class Load(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_in(name=flow, var_demand_in_default=demand)

        return Load

    class Machine(mu.ObjFlow):
        def add_flows(self):
            for flow in cons:
                self.add_flow_continuous_in(name=flow)
            for flow in prod:
                self.add_flow_continuous_out(name=flow)
            self.add_rule_set(name="make", rules=[{"cons": cons, "prod": prod}])

    system = mu.System("unread")
    system.add_component(source("w", rate), "S")
    system.add_component(Machine, "T")
    system.connect("S", "w", "T", "w")
    if rival is not None:
        system.add_component(load("w", rival), "C")
        system.connect("S", "w", "C", "w")
    for flow, supply in others:
        system.add_component(source(flow, supply), f"S_{flow}")
        system.connect(f"S_{flow}", flow, "T", flow)
    for flow, demand in (wired or {}).items():
        system.add_component(load(flow, demand), f"C_{flow}")
        system.connect("T", flow, f"C_{flow}", flow)
    return system


def read(system: mu.System, *names: str) -> list[float]:
    result = system.simulate(t_max=1.0, samples=[1.0])
    return [result.samples[name][-1][1] for name in names]


def test_an_unread_rule_claims_its_declared_coefficient():
    """Behind a source of 10, `cons w:2` asks 2 and receives 2. It used to
    ask the 10 the source could give."""
    assert read(plant({"w": 2.0}, {"p": 3.0}), "T_w_demand_in", "T_w_fed_in") == [2.0, 2.0]


def test_an_unread_rule_leaves_a_rival_its_share():
    """The reason it matters: the source also feeds a load asking 4. Asked
    10 by the machine, it served the load 2.857; asked 2, it serves both in
    full."""
    values = read(plant({"w": 2.0}, {"p": 3.0}, rival=4.0), "T_w_fed_in", "C_w_fed_in")
    assert values == [2.0, 4.0]


def test_the_nominal_claim_is_bounded_by_the_other_inputs_only():
    """Two inputs, the second able to sustain half the nominal scale: each
    is claimed at the scale the OTHER could sustain, bounded by the nominal
    1. `w` asks 2 x 0.5, `e` asks its nominal 1 from a source of 0.5."""
    system = plant({"w": 2.0, "e": 1.0}, {"p": 3.0}, rival=4.0, others=[("e", 0.5)])
    values = read(system, "T_w_demand_in", "T_e_demand_in", "T_w_fed_in", "C_w_fed_in")
    assert values == [1.0, 1.0, 1.0, 4.0]


def test_a_short_supply_is_still_asked_the_nominal_need():
    """A source of 1 for a nominal need of 2: the claim is the need, never
    bounded by its own supplier, and the rule draws what arrives."""
    values = read(plant({"w": 2.0}, {"p": 3.0}, rate=1.0), "T_w_demand_in", "T_w_fed_in")
    assert values == [2.0, 1.0]


def test_a_rule_producing_nothing_runs_at_its_nominal_scale_too():
    assert read(plant({"w": 2.0}, {}), "T_w_demand_in", "T_w_fed_in") == [2.0, 2.0]


def test_one_read_output_still_sizes_the_rule():
    """Unread is not zero: with `p` asked 1.5 at a coefficient of 3 and `q`
    read by nothing, the scale is the 0.5 the read output sets, and the
    unread one bounds nothing."""
    system = plant({"w": 2.0}, {"p": 3.0, "q": 1.0}, wired={"p": 1.5})
    assert read(system, "T_w_demand_in", "T_w_fed_in") == [1.0, 1.0]


# --- the library route ----------------------------------------------------


def declared(cons, prod, *, rate=10.0, rival=4.0, others=(), wired=None) -> dict:
    machine = component(
        "T", [c_in(flow) for flow in cons] + [c_out(flow) for flow in prod], class_name="Machine"
    )
    machine["rules"] = [{"name": "make", "rules": [rule("go", [], cons, prod)]}]
    components = [component("S", [c_out("w", rate)], class_name="Source"), machine]
    links = [link("S", "T", "w")]
    if rival:
        components.append(component("C", [c_in("w", rival)], class_name="Load"))
        links.append(link("S", "C", "w"))
    for flow, supply in others:
        components.append(component(f"S_{flow}", [c_out(flow, supply)], class_name="Source"))
        links.append(link(f"S_{flow}", "T", flow))
    for flow, demand in (wired or {}).items():
        components.append(component(f"C_{flow}", [c_in(flow, demand)], class_name="Load"))
        links.append(link("T", f"C_{flow}", flow))
    return document(components, links)


@pytest.mark.parametrize(
    "declaration",
    [
        declared({"w": 2.0}, {"p": 3.0}),
        declared({"w": 2.0, "e": 1.0}, {"p": 3.0}, others=[("e", 0.5)]),
        declared({"w": 2.0}, {}),
        declared({"w": 2.0}, {"p": 3.0, "q": 1.0}, wired={"p": 1.5}),
        declared({"w": 2.0}, {"p": 3.0}, rate=1.0, rival=0.0),
    ],
    ids=["unread", "two-inputs", "no-product", "one-read", "short-supply"],
)
def test_the_library_route_agrees_on_an_unread_rule(declaration):
    """The conversion leaves an unread output out of the lists whose demand
    sizes the scale, and a set naming none runs at its nominal scale. The
    library route used to read the unread output's demand, an empty sum,
    and stop the rule."""
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    comparison = compare_routes(declaration, conversion, t_max=5.0)
    assert comparison.identical, comparison.detail
    assert comparison.level == "exact", comparison.detail
