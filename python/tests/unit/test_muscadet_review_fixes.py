"""Review fixes on the `pyraichu.muscadet` authoring layer.

Eight findings of one review pass, each pinned by the property it broke
rather than by the diff that closed it. Three of them were conservation
holes, and those three are asserted **in the quantity**: what entered a
component equals what left it plus what it stored. A test that only
watched an exception be raised would keep passing if the refusal were
later relaxed for the wrong reason.

- a capacity and a rule set on one flow of one component made matter,
  the flow crossing once by the rule and once by the volume;
- a two-stream transfer on a rule-produced stream made matter, the
  origin's shortfall being clamped away while the target still gained
  the whole moved quantity;
- a derating pattern carrying a top-level alternation reached flows it
  never named, `^H2|O2$` matching `H2O`;
- the rule-cycle search stopped on its budget and read as "no cycle
  found", so a conservation guard quietly stopped guarding;
- three malformed or degenerate declarations raised the wrong error, or
  none;
- two clash refusals held in one declaration order only;
- three docstrings cited requirement identifiers no plan carries.

A later pass added two more, both resting on one cause: a guard operand
naming a measurement channel was read as neither a rate nor a level,
although what it reads is a published capacity level.

- a rule guarded on a level read over a measurement link was emitted as
  an instantaneous transition, so the crossing was never located and the
  rule switched at whatever event came next, or never;
- an equality on that same level was accepted, where an equality on a
  level the component holds is refused.
"""

import inspect
import re

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, TOL, fired_at, sampled
from test_declare_build import the_plant_by_class, the_plant_by_declaration

# --- shared shapes -----------------------------------------------------


class Sink(mu.ObjFlow):
    """A pure consumer, asking for 2 of `prod` per unit time."""

    def add_flows(self):
        self.add_flow_continuous_in(name="prod", var_demand_in_default=2.0)


class Reactor(mu.ObjFlow):
    """One unit of `feed` becomes one unit of `prod`, and nothing is
    held: the transformation is the whole of what this component does."""

    def add_flows(self):
        self.add_flow_continuous_in(name="feed")
        self.add_flow_continuous_out(name="prod")
        self.add_rule_set(
            name="convert", rules=[{"cons": {"feed": 1.0}, "prod": {"prod": 1.0}}]
        )


class Buffer(mu.ObjFlow):
    """Ten units of `feed` in a volume of its own, served as fast as they
    are asked for. A buffer on its own component, which is where a buffer
    conserves."""

    def add_flows(self):
        self.add_flow_continuous_out(name="feed", var_fed_default=100.0)
        self.add_capacity(
            name="store", flow="feed", capacity=10.0, content_init={"feed": 10.0}
        )


def a_reactor_declaring(order: str) -> mu.ObjFlow:
    """A reactor whose capacity and rule set name the same flow, declared
    in the given order. Both orders describe the same component, so both
    must reach the same verdict."""
    obj = mu.ObjFlow("R")
    obj.add_flow_continuous_in(name="feed")
    obj.add_flow_continuous_out(name="prod")

    def capacity():
        obj.add_capacity(name="hopper", flow="feed", capacity=50.0, side="in")

    def rules():
        obj.add_rule_set(
            name="convert", rules=[{"cons": {"feed": 1.0}, "prod": {"prod": 1.0}}]
        )

    halves = (capacity, rules) if order == "capacity first" else (rules, capacity)
    for half in halves:
        half()
    return obj


# --- finding 1: a volume and a rule may not both carry one flow --------


@pytest.mark.parametrize("order", ["capacity first", "rule set first"])
def test_a_capacity_over_a_rule_carried_flow_is_refused_in_either_order(order):
    """A rule TRANSFORMS what crosses the component and a volume STORES
    it: on one flow the two double-count, and the component makes matter.

    Refused whichever half was written first, which is the point: the
    check reads the component, not the declaration it was called from."""
    with pytest.raises(ValueError) as raised:
        a_reactor_declaring(order)
    message = str(raised.value)
    assert "`R`" in message and "`hopper`" in message and "`feed`" in message


def test_a_refused_declaration_leaves_the_component_as_it_was():
    """The refusal takes the offending declaration back out, so a
    component that raised is not left half-declared."""
    obj = mu.ObjFlow("R")
    obj.add_flow_continuous_in(name="feed")
    obj.add_flow_continuous_out(name="prod")
    obj.add_capacity(name="hopper", flow="feed", capacity=50.0, side="in")
    with pytest.raises(ValueError):
        obj.add_rule_set(
            name="convert", rules=[{"cons": {"feed": 1.0}, "prod": {"prod": 1.0}}]
        )
    assert obj.rule_sets == []
    assert [capacity.name for capacity in obj.capacities] == ["hopper"]


def test_a_buffer_upstream_of_a_reactor_conserves_what_it_held():
    """The shape the refusal leaves standing, asserted in the quantity:
    the volume is on its own component, upstream of the rules.

    Ten units are held, two leave per unit time, and the store is empty
    at t=5 to the crossing tolerance. What the sink received over the run
    is what the store held at the start and no more: nothing was made,
    nothing was lost."""
    system = mu.System("review_buffer_upstream")
    system.add_component(Buffer, "B")
    system.add_component(Reactor, "R")
    system.add_component(Sink, "K")
    system.connect("B", "feed", "R", "feed")
    system.connect("R", "prod", "K", "prod")
    result = system.simulate(t_max=10.0, samples=[0.0, 2.5, 4.9, 6.0, 9.0])

    assert sampled(result, "B_store_content_feed", 0.0) == pytest.approx(10.0, abs=TOL)
    assert sampled(result, "B_store_content_feed", 2.5) == pytest.approx(
        5.0, abs=CROSSING_TOL
    )
    assert sampled(result, "B_store_content_feed", 4.9) == pytest.approx(
        0.2, abs=CROSSING_TOL
    )
    # Empty, and staying empty: what is served afterwards is what arrives,
    # and nothing arrives.
    for instant in (6.0, 9.0):
        assert sampled(result, "B_store_content_feed", instant) == pytest.approx(
            0.0, abs=CROSSING_TOL
        )
        assert sampled(result, "K_prod_fed_in", instant) == pytest.approx(0.0, abs=TOL)

    # While it lasts, every unit that leaves the store crosses the reactor
    # and reaches the sink, one for one.
    for instant in (0.0, 2.5, 4.9):
        served = sampled(result, "B_feed_fed_out", instant)
        assert served == pytest.approx(2.0, abs=TOL)
        assert sampled(result, "R_feed_fed_in", instant) == pytest.approx(
            served, abs=TOL
        )
        assert sampled(result, "R_prod_fed_out", instant) == pytest.approx(
            served, abs=TOL
        )
        assert sampled(result, "K_prod_fed_in", instant) == pytest.approx(served, abs=TOL)

    # Conservation, written as the equality it is: what the store lost
    # over an interval is what the sink received over it. The defect this
    # pins made the store keep its content while the reactor delivered
    # all the same, so this equality was the one that broke.
    dropped = sampled(result, "B_store_content_feed", 0.0) - sampled(
        result, "B_store_content_feed", 2.5
    )
    assert dropped == pytest.approx(2.0 * 2.5, abs=CROSSING_TOL)


def test_a_capacity_and_a_rule_on_different_components_still_build():
    """The electrolysis plant, muscadet's own four-component reference:
    two capacities, on the battery and on the local store, and a rule set
    on the electrolyser between them.

    That is the ordinary shape, and the refusal must leave it alone. Both
    authorings still generate one and the same model, and the battery
    still drains as the reaction draws on it."""
    by_class = the_plant_by_class()
    assert by_class.build_dict() == the_plant_by_declaration().build_dict()

    result = by_class.simulate(t_max=5.0, samples=[0.0, 1.0])
    # The battery holds a capacity over `Elec`; the electrolyser consumes
    # `Elec` through a rule. Different components, so both carry their own
    # half and the store empties at the rate the reaction draws.
    drawn = sampled(result, "Electro_Elec_fed_in", 1.0)
    assert drawn > 0.0
    assert sampled(result, "B1_Elec_fed_out", 1.0) == pytest.approx(drawn, abs=TOL)


# --- finding 2: a pair moves what the stream actually made -------------


class Splitter(mu.ObjFlow):
    """A rule makes `hot` out of `feed`, and a gradient diverts part of
    `hot` into `cold`. The gradient asks for 5, well above what the rule
    is asked to make: the pair can only move what exists."""

    def add_flows(self):
        self.add_flow_continuous_in(name="feed")
        self.add_flow_continuous_out(name="hot", var_fed_default=0.0)
        self.add_flow_continuous_out(name="cold", var_fed_default=0.0)
        self.add_rule_set(
            name="convert", rules=[{"cons": {"feed": 1.0}, "prod": {"hot": 1.0}}]
        )
        self.add_transfer(
            name="leak",
            flows=["hot", "cold"],
            equation={
                "cls": "ConductiveTransfer",
                "conductance": 1.0,
                "potential_a": {"const": 5.0},
                "potential_b": {"const": 0.0},
            },
        )


class Source(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="feed", var_fed_default=10.0)


class HotSink(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_in(name="hot", var_demand_in_default=1.0)


class ColdSink(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_in(name="cold", var_demand_in_default=100.0)


def a_splitting_system() -> mu.System:
    system = mu.System("review_pair_on_production")
    system.add_component(Source, "S")
    system.add_component(Splitter, "R")
    system.add_component(HotSink, "H")
    system.add_component(ColdSink, "C")
    system.connect("S", "feed", "R", "feed")
    system.connect("R", "hot", "H", "hot")
    system.connect("R", "cold", "C", "cold")
    return system


def test_a_pair_on_a_rule_produced_stream_conserves_the_component_total():
    """What leaves the splitter equals what entered it.

    The gradient asks for 5 while the rule is asked to make 1, because
    the hot sink asks for 1. Capping the pair on the **capability** let
    it move 5: the origin lost 1 and was clamped at zero, while the
    target gained 5, and four units appeared out of nothing. Capping it
    on what was actually made moves one number on both sides."""
    result = a_splitting_system().simulate(t_max=1.0, samples=[0.5])
    entering = sampled(result, "R_feed_fed_in", 0.5)
    leaving = sampled(result, "R_hot_fed_out", 0.5) + sampled(
        result, "R_cold_fed_out", 0.5
    )
    assert leaving == pytest.approx(entering, abs=TOL)
    # No volume on this component, so the balance is exact rather than
    # up to a stored quantity: one unit in, one unit out.
    assert entering == pytest.approx(1.0, abs=TOL)


def test_a_pair_moves_no_more_than_the_stream_was_made_of():
    """The cap itself, read off the two published bases: the gradient
    asked for 5, the capability base stood at 10, and the pair moved the
    1 the production base held."""
    result = a_splitting_system().simulate(t_max=1.0, samples=[0.5])
    assert sampled(result, "R_leak_requested", 0.5) == pytest.approx(5.0, abs=TOL)
    assert sampled(result, "R_hot_transfer_base", 0.5) == pytest.approx(10.0, abs=TOL)
    made = sampled(result, "R_hot_produced_base", 0.5)
    assert made == pytest.approx(1.0, abs=TOL)
    assert sampled(result, "R_leak_moved", 0.5) == pytest.approx(made, abs=TOL)


def test_a_stream_no_rule_produces_keeps_the_capability_base_alone():
    """The production base is emitted only where the two bases differ: a
    pair on a stream no rule makes reads the capability base, as it
    always has."""
    body = pyraichu.model_body(a_splitting_system().build_dict())
    splitter = next(
        component for component in body["components"] if component["name"] == "R"
    )
    named = {attribute["name"] for attribute in splitter["attributes"]}
    assert "hot_produced_base" in named
    assert "cold_produced_base" not in named


# --- finding 3: an alternation is grouped before it is anchored --------


class Plant(mu.ObjFlow):
    """Three outputs whose names overlap, and one mode meant for two of
    them. `H2O` contains `H2` and ends in `O2`'s last two characters,
    which is exactly what an unparenthesised alternation trips on."""

    def add_flows(self):
        for flow in ("H2", "O2", "H2O"):
            self.add_flow_continuous_out(name=flow, var_fed_default=4.0)
        self.add_delay_failure_mode(
            name="df",
            failure_time=2.0,
            repair_time=1e9,
            failure_effects=[("H2|O2", 0.0)],
        )


@pytest.mark.parametrize(
    "name, named",
    [("H2", True), ("O2", True), ("H2O", False), ("XO2", False), ("H2X", False)],
)
def test_an_alternation_names_only_what_it_spells_out(name, named):
    """`"H2|O2"` names two flows. Anchored without grouping it read as
    "starts with H2, or ends with O2", which names three more."""
    assert mu._matches_flow("H2|O2", name) is named


def test_an_alternating_derating_reaches_exactly_the_two_flows_it_names():
    """The resolution, and then the quantity: the mode zeroes `H2` and
    `O2` when it stands, and `H2O` keeps delivering its nominal 4."""
    plant = Plant("P")
    assert sorted(plant.failure_modes[0].failure.caps) == ["H2", "O2"]

    system = mu.System("review_alternation")
    system.add_component(Plant, "P")
    for flow, consumer in (("H2", "A"), ("O2", "B"), ("H2O", "C")):
        consumer_cls = type(
            f"Take{flow}",
            (mu.ObjFlow,),
            {
                "add_flows": (
                    lambda self, flow=flow: self.add_flow_continuous_in(
                        name=flow, var_demand_in_default=4.0
                    )
                )
            },
        )
        system.add_component(consumer_cls, consumer)
        system.connect("P", flow, consumer, flow)
    result = system.simulate(t_max=4.0, samples=[1.0, 3.0])

    for flow in ("H2", "O2", "H2O"):
        assert sampled(result, f"P_{flow}_fed_out", 1.0) == pytest.approx(4.0, abs=TOL)
    assert sampled(result, "P_H2_fed_out", 3.0) == pytest.approx(0.0, abs=TOL)
    assert sampled(result, "P_O2_fed_out", 3.0) == pytest.approx(0.0, abs=TOL)
    # The flow the declaration never named, still delivering: this is the
    # number the ungrouped pattern silently took to zero.
    assert sampled(result, "P_H2O_fed_out", 3.0) == pytest.approx(4.0, abs=TOL)


# --- finding 4: an exhausted search refuses, it does not report nothing


def a_two_rule_system() -> mu.System:
    """A rule graph with arcs on it, and no cycle: what the search should
    walk to the end and clear."""
    system = mu.System("review_cycle_budget")
    system.add_component(Source, "S")
    system.add_component(Reactor, "R")
    system.add_component(Sink, "K")
    system.connect("S", "feed", "R", "feed")
    system.connect("R", "prod", "K", "prod")
    return system


def test_a_rule_graph_the_search_can_walk_is_accepted():
    """The reference the next test is read against: with its real budget
    the search completes and the model builds."""
    assert a_two_rule_system().build_dict()


def test_an_exhausted_cycle_search_refuses_the_model(monkeypatch):
    """A budget spent before the walk ends means part of the graph was
    never visited, and a cycle that creates matter may sit in it.

    Refusing is the right direction for a build-time conservation guard:
    the alternative is a model that builds because the guard gave up,
    which reads exactly like a model with nothing to refuse."""
    monkeypatch.setattr(mu, "_CYCLE_SEARCH_BUDGET", 0)
    with pytest.raises(ValueError) as raised:
        a_two_rule_system().build_dict()
    message = str(raised.value)
    assert "rule-cycle search" in message
    assert "budget of 0 arcs" in message
    assert "flow endpoints" in message


# --- findings 5 and 7: a capacity entry names a flow, at a real weight -


def test_a_capacity_entry_naming_no_flow_is_refused_by_name():
    """A malformed entry raised a bare `KeyError` naming nothing. Every
    other malformed declaration in this layer names the component and
    what is wrong, and so does this one now."""
    obj = mu.ObjFlow("T")
    obj.add_flow_continuous_in(name="water")
    with pytest.raises(ValueError) as raised:
        obj.add_capacity(name="tank", flows=[{"weight": 2.0}], capacity=10.0)
    message = str(raised.value)
    assert "`T`" in message and "`tank`" in message and "name" in message


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_a_capacity_weight_must_be_strictly_positive(weight):
    """A weight is how much volume one unit of the flow occupies. At
    zero the volume holds an unbounded quantity and never reaches its
    bound; negative, it empties as it fills."""
    obj = mu.ObjFlow("T")
    obj.add_flow_continuous_in(name="water")
    with pytest.raises(ValueError) as raised:
        obj.add_capacity(
            name="tank", flows=[{"name": "water", "weight": weight}], capacity=10.0
        )
    message = str(raised.value)
    assert "`T`" in message and "`tank`" in message and "weight" in message


def test_a_positive_capacity_weight_is_still_accepted():
    """The guard refuses the degenerate values and nothing else: a
    declared weight of 2 halves the volume the flow may occupy."""
    obj = mu.ObjFlow("T")
    obj.add_flow_continuous_in(name="water")
    obj.add_capacity(
        name="tank", flows=[{"name": "water", "weight": 2.0}], capacity=10.0
    )
    assert obj.capacities[0].flows[0].weight == 2.0


# --- finding 6: the transfer clashes hold in both orders ---------------


def a_conduit_component(order: str) -> mu.ObjFlow:
    """A conduit metering `heat` and a two-stream pair on the same flow,
    declared in the given order."""
    obj = mu.ObjFlow("W")
    obj.add_flow_continuous_in(name="heat")
    obj.add_flow_continuous_out(name="heat")
    obj.add_flow_continuous_out(name="cold")
    equation = {
        "cls": "ConductiveTransfer",
        "conductance": 1.0,
        "potential_a": 10.0,
        "potential_b": 0.0,
    }
    def conduit():
        obj.add_transfer(name="wall", flows=("heat", "heat"), equation=equation)

    def exchange():
        obj.add_transfer(name="swap", flows=("heat", "cold"), equation=equation)

    halves = (conduit, exchange) if order == "conduit first" else (exchange, conduit)
    for half in halves:
        half()
    return obj


@pytest.mark.parametrize("order", ["conduit first", "pair first"])
def test_a_conduit_and_a_pair_on_one_flow_are_refused_in_either_order(order):
    """A conduit REPLACED that flow's transit, so a delta on top of it
    has no stream to sit on. True of the pair whichever was written
    first, and now refused whichever was written first."""
    with pytest.raises(ValueError) as raised:
        a_conduit_component(order)
    message = str(raised.value)
    assert "`W`" in message and "heat" in message


def a_metered_reactor(order: str) -> mu.ObjFlow:
    """A conduit and a rule set both claiming `heat`."""
    obj = mu.ObjFlow("W")
    obj.add_flow_continuous_in(name="heat")
    obj.add_flow_continuous_out(name="heat")
    def conduit():
        obj.add_transfer(
            name="wall",
            flows=("heat", "heat"),
            equation={
                "cls": "ConductiveTransfer",
                "conductance": 1.0,
                "potential_a": 10.0,
                "potential_b": 0.0,
            },
        )

    def rules():
        obj.add_rule_set(name="warm", rules=[{"prod": {"heat": 1}}])

    halves = (conduit, rules) if order == "conduit first" else (rules, conduit)
    for half in halves:
        half()
    return obj


@pytest.mark.parametrize("order", ["conduit first", "rule set first"])
def test_a_conduit_and_a_rule_on_one_flow_are_refused_in_either_order(order):
    """The refusal that already existed, now independent of the order it
    was written in."""
    with pytest.raises(ValueError) as raised:
        a_metered_reactor(order)
    message = str(raised.value)
    assert "`W`" in message and "heat" in message


def test_the_clash_checks_run_again_when_the_document_is_written():
    """The declaration methods are the early diagnostic, not the guard:
    a component whose lists were filled by other means still meets the
    refusal at the point the document is written."""
    obj = mu.ObjFlow("R")
    obj.add_flow_continuous_in(name="feed")
    obj.add_flow_continuous_out(name="prod")
    obj.add_rule_set(
        name="convert", rules=[{"cons": {"feed": 1.0}, "prod": {"prod": 1.0}}]
    )
    # Around the declaration method, as an unwary subclass or a future
    # loader could.
    obj.capacities.append(
        mu._Capacity(
            name="hopper",
            flows=[mu._CapacityFlow(name="feed")],
            volume=50.0,
            side="in",
            content_init={},
            fill_rate=0.0,
            hysteresis=mu.DEFAULT_HYSTERESIS,
        )
    )
    with pytest.raises(ValueError) as raised:
        obj._build()
    assert "`hopper`" in str(raised.value)


# --- finding 8: a citation names a real identifier or none -------------


def test_no_docstring_cites_an_identifier_no_plan_carries():
    """The governing plan numbers its requirements `R<n>` and its key
    technical decisions `KTD<n>`. Three passages cited a family that has
    never existed, so a reader chasing them found nothing and a reviewer
    could not check the claim against anything."""
    source = inspect.getsource(mu)
    invented = re.findall(r"\bKD\d+\b", source)
    assert invented == []


def test_the_corrected_citations_name_what_the_passages_describe():
    """Each of the three now cites the requirement it actually states:
    the fold of deratings and profile on an output, twice, and the
    signed transfer quantity whose direction this layer routes."""
    source = inspect.getsource(mu)
    assert "The two are separate channels on purpose (R6)" in source
    assert "derating this output (R6)" in source
    assert "The sign is the direction (R5)" in source


# --- a later pass: a level read over a measurement link ----------------


class Rain(mu.ObjFlow):
    """A steady unit of `w` a unit of time, the only thing filling the
    cistern."""

    def add_flows(self):
        self.add_flow_continuous_out(name="w", var_fed_default=1.0)


class Cistern(mu.ObjFlow):
    """A volume filling at one a unit of time from empty, publishing its
    level and its fill."""

    def add_flows(self):
        self.add_flow_continuous_in(name="w")
        self.add_capacity(name="vol", flow="w", capacity=100.0, fill_rate=1.0)


def a_reader_guarded_on(channel: str, op: str = ">=", value: float = 5.0):
    """A reactor switched by the level of a volume it does not hold: it
    reads the volume over a measurement link and holds no capacity of
    its own, so nothing on it makes the guard continuous except the link
    itself."""

    class Reader(mu.ObjFlow):
        def add_flows(self):
            self.add_measurement_in(name="vol", flows=["w"])
            self.add_flow_continuous_in(name="feed")
            self.add_flow_continuous_out(name="prod")
            self.add_rule_set(
                name="duty",
                rules=[
                    {
                        "cond": [{"name": channel, "op": op, "value": value}],
                        "cons": {"feed": 1.0},
                        "prod": {"prod": 1.0},
                    }
                ],
            )

    return Reader


def a_measured_switch(reader=None) -> mu.System:
    """The cistern, the reader, and nothing that could stop the
    integration at the threshold: the volume's own bounds sit at 0 and at
    100, it fills uniformly, and no other component carries a guard. What
    happens at t = 5 is therefore the measured guard being located, and
    nothing else."""
    system = mu.System("review_measured_guard")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.add_component(Source, "S")
    system.add_component(reader or a_reader_guarded_on("vol_level"), "R")
    system.add_component(Sink, "K")
    system.connect("P", "w", "T", "w")
    system.connect("S", "feed", "R", "feed")
    system.connect("R", "prod", "K", "prod")
    system.connect_measurement("T", "vol", "R")
    return system


def test_a_guard_read_over_a_measurement_link_is_located_at_the_crossing():
    """The cistern fills at one a unit of time from empty, so it holds 5
    at t = 5, and the rule reading that level over the link is selected
    there.

    The date is the assertion. Read as neither a rate nor a level, the
    guard was emitted as an instantaneous transition, and since nothing
    else in this system stops the integration at t = 5 the rule then
    never switched at all: no error, no warning, a trajectory that looks
    like a plant which simply never started."""
    result = a_measured_switch().simulate(t_max=20.0, samples=[4.9, 5.1])
    assert abs(fired_at(result, "duty_none_to_rule_0") - 5.0) < CROSSING_TOL
    assert sampled(result, "K_prod_fed_in", 4.9) == 0.0
    assert abs(sampled(result, "K_prod_fed_in", 5.1) - 2.0) < TOL


@pytest.mark.parametrize(
    "channel, value",
    [
        ("vol_level", 5.0),
        ("vol_fill", 0.05),
        ("vol_level_w", 5.0),
        ("vol_fill_w", 0.05),
    ],
)
def test_every_measurement_channel_makes_the_mode_transition_watched(channel, value):
    """The cheap regression beside the located one, over the whole
    vocabulary of a link: each of the four channels is a published
    capacity level or a fill computed from one, so each is continuous and
    a guard on any of them is a located crossing."""
    document = pyraichu.model_body(
        a_measured_switch(a_reader_guarded_on(channel, value=value)).build_dict()
    )
    reader = next(c for c in document["components"] if c["name"] == "R")
    mode = next(a for a in reader["automata"] if a["name"] == "duty_mode")
    assert {t["distrib"] for t in mode["transitions"]} == {"watched"}


def a_component_guarded_on(kind: str, op: str) -> mu.ObjFlow:
    """A rule guarded by a level, read either on the component holding
    the volume or on one observing it over a link. Both read the same
    quantity, so both must reach the same verdict on the comparison."""
    obj = mu.ObjFlow("R")
    obj.add_flow_continuous_in(name="feed")
    obj.add_flow_continuous_out(name="prod")
    if kind == "held":
        obj.add_flow_continuous_in(name="w")
        obj.add_capacity(name="vol", flow="w", capacity=100.0, fill_rate=1.0)
        channel = "vol_content"
    else:
        obj.add_measurement_in(name="vol", flows=["w"])
        channel = "vol_level"
    obj.add_rule_set(
        name="duty",
        rules=[
            {
                "cond": [{"name": channel, "op": op, "value": 5.0}],
                "cons": {"feed": 1.0},
                "prod": {"prod": 1.0},
            }
        ],
    )
    return obj


@pytest.mark.parametrize(
    "kind, channel", [("held", "vol_content"), ("measured", "vol_level")]
)
def test_an_equality_on_a_level_is_refused_however_it_is_read(kind, channel):
    """A crossing cannot be located on an equality, whether the level is
    held or observed: the guard would hold only if a float landed exactly
    on the threshold. The observed side used to be accepted, which
    promised a switch that no run would deliver."""
    with pytest.raises(ValueError) as raised:
        a_component_guarded_on(kind, "==")
    message = str(raised.value)
    assert "`R`" in message and "`duty`" in message and f"`{channel}`" in message
    assert "ordering comparison" in message


@pytest.mark.parametrize("kind", ["held", "measured"])
def test_an_ordering_comparison_on_a_level_is_accepted_however_it_is_read(kind):
    """The refusal is on the comparison, not on the reading: written with
    an ordering comparison, the same guard is the located crossing this
    layer is for."""
    assert a_component_guarded_on(kind, ">=").rule_sets[0].rules[0].cond
