"""A rule's input fed by several suppliers returns its surplus.

Every supplier of one input is asked the input's whole demand, which is
muscadet's reading and this layer's, so together they may offer more
than the rule needs. A rule consumes what it needs and no more: the
surplus goes back to the suppliers **pro rata of what each offered**,
which is muscadet's `release_unused_supply`. Without it the input took
everything offered and the difference vanished, and a battery feeding an
electrolyser beside a bus emptied at twice the rate the rule consumed.

Every number is the closed form ``delivered_i = need * offer_i / sum
offers`` whenever the offers exceed the need, measured on the muscadet
5.6.0 reference first (17.63 and 20 offered to a need of 20 are
delivered as 9.370 and 10.630).

A consumer declaring no rule keeps the whole of what it is offered on
both engines alike, so that shape is pinned as it is.
"""

import pytest

import pyraichu.muscadet as mu
from conftest import TOL, sampled

NEED = 20.0


def source(flow: str, quantity: float):
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=quantity)

    return Source


class Electrolyser(mu.ObjFlow):
    """Twenty units of `e` make one of `h`, asked for one."""

    def add_flows(self):
        self.add_flow_continuous_in(name="e")
        self.add_flow_continuous_out(name="h")
        self.add_rule_set(name="conv", rules=[{"cons": {"e": NEED}, "prod": {"h": 1.0}}])


class Sink(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_in(name="h", var_demand_in_default=1.0)


def two_suppliers(bus: float, second, consumer=Electrolyser) -> mu.System:
    system = mu.System("rule_input_surplus")
    system.add_component(source("e", bus), "BUS")
    system.add_component(second, "SECOND")
    system.add_component(consumer, "EL")
    system.connect("BUS", "e", "EL", "e")
    system.connect("SECOND", "e", "EL", "e")
    if consumer is Electrolyser:
        system.add_component(Sink, "SINK")
        system.connect("EL", "h", "SINK", "h")
    return system


@pytest.mark.parametrize(
    "bus, second, bus_gets, second_gets",
    [
        # The reference's own numbers: 20 * 17.63 / 37.63 and 20 * 20 / 37.63.
        (17.63, 20.0, 9.370183364337, 10.629816635663),
        # A second supplier offering more than the demand offers the demand.
        (15.0, 100.0, 20.0 * 15.0 / 35.0, 20.0 * 20.0 / 35.0),
        # Offers below the need: nothing is released, each gives all it has.
        (5.0, 8.0, 5.0, 8.0),
    ],
)
def test_a_rule_input_returns_its_surplus_pro_rata_of_the_offers(
    bus, second, bus_gets, second_gets
):
    result = two_suppliers(bus, source("e", second)).simulate(t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "BUS_e_fed_out", 0.5) - bus_gets) < 1e-9
    assert abs(sampled(result, "SECOND_e_fed_out", 0.5) - second_gets) < 1e-9
    assert (
        abs(sampled(result, "EL_e_fed_in", 0.5) - min(NEED, bus + min(second, NEED)))
        < 1e-9
    )


def test_a_supplying_volume_is_drawn_only_by_what_the_rule_took():
    """The showcase's battery: a transiting volume beside a bus, both
    feeding the electrolyser. The volume's content falls by its share of
    the need, 20 * 20 / 37.63 per unit time, not by the 20 it offered."""

    class Battery(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="e")
            self.add_capacity(
                name="battery", flow="e", capacity=1000.0,
                content_init={"e": 500.0}, transmits=True,
            )

    result = two_suppliers(17.63, Battery).simulate(t_max=2.0, samples=[1.0, 2.0])
    share = NEED * NEED / (NEED + 17.63)
    for instant in (1.0, 2.0):
        assert abs(sampled(result, "SECOND_battery_content", instant) - (500.0 - share * instant)) < 1e-6
        assert abs(sampled(result, "EL_e_fed_in", instant) - NEED) < 1e-9


def test_a_single_supplier_is_unchanged():
    system = mu.System("rule_input_single")
    system.add_component(source("e", 50.0), "BUS")
    system.add_component(Electrolyser, "EL")
    system.add_component(Sink, "SINK")
    system.connect("BUS", "e", "EL", "e")
    system.connect("EL", "h", "SINK", "h")
    result = system.simulate(t_max=1.0, samples=[0.5])
    assert abs(sampled(result, "BUS_e_fed_out", 0.5) - NEED) < TOL
    assert abs(sampled(result, "EL_e_fed_in", 0.5) - NEED) < TOL


def test_a_consumer_with_no_rule_keeps_what_it_is_offered_as_the_reference_does():
    """No rule consumes the input, so nothing sizes a need to release
    against: both engines deliver 15 + 20 to a consumer asking 20."""

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e", var_demand_in_default=NEED)

    result = two_suppliers(15.0, source("e", 100.0), Consumer).simulate(
        t_max=1.0, samples=[0.5]
    )
    assert abs(sampled(result, "EL_e_fed_in", 0.5) - 35.0) < TOL


def test_a_supplier_also_feeding_a_volume_keeps_the_balance():
    """The showcase's electrical bus: it feeds the electrolyser directly
    AND the battery beside it, which feeds the electrolyser too. The
    battery's own arrival must be settled in the same evaluation as the
    bus's allocation, or the battery passes on what it has not been
    given: what the electrolyser takes equals what the bus delivers less
    what the battery keeps."""

    class Battery(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="e")
            self.add_capacity(
                name="battery", flow="e", capacity=1000.0,
                content_init={"e": 500.0}, transmits=True,
            )

    system = mu.System("rule_input_surplus_bus")
    system.add_component(source("e", 30.0), "BUS")
    system.add_component(Battery, "B1")
    system.add_component(Electrolyser, "EL")
    system.add_component(Sink, "SINK")
    system.connect("BUS", "e", "B1", "e")
    system.connect("BUS", "e", "EL", "e")
    system.connect("B1", "e", "EL", "e")
    system.connect("EL", "h", "SINK", "h")
    result = system.simulate(t_max=3.0, samples=[1.0, 2.0, 3.0])
    kept = sampled(result, "B1_battery_content", 3.0) - sampled(
        result, "B1_battery_content", 2.0
    )
    for instant in (2.0, 3.0):
        assert abs(sampled(result, "EL_e_fed_in", instant) - NEED) < 1e-9
        assert (
            abs(
                sampled(result, "B1_e_fed_in", instant)
                - sampled(result, "B1_e_fed_out", instant)
                - kept
            )
            < 1e-6
        )
        assert abs(sampled(result, "BUS_e_fed_out", instant) - NEED - kept) < 1e-6


def test_a_pipe_fed_by_a_supplier_of_a_releasing_input_passes_on_what_it_gets():
    """The bus feeds the electrolyser (whose input releases its surplus)
    and a plain pipe to another consumer. The pipe's arrival is settled
    with the bus's allocation, before the pipe passes it on: a pipe read
    later than its own visit passed on nothing, and the matter vanished
    between its two ends."""

    class Pipe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="e")

    class Lamp(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e", var_demand_in_default=5.0)

    system = mu.System("rule_input_surplus_pipe")
    system.add_component(source("e", 30.0), "BUS")
    system.add_component(source("e", 20.0), "SECOND")
    system.add_component(Electrolyser, "EL")
    system.add_component(Sink, "SINK")
    system.add_component(Pipe, "T")
    system.add_component(Lamp, "LAMP")
    system.connect("BUS", "e", "EL", "e")
    system.connect("SECOND", "e", "EL", "e")
    system.connect("EL", "h", "SINK", "h")
    system.connect("BUS", "e", "T", "e")
    system.connect("T", "e", "LAMP", "e")
    result = system.simulate(t_max=1.0, samples=[0.5])
    assert sampled(result, "T_e_fed_in", 0.5) > 0.0
    assert abs(sampled(result, "T_e_fed_out", 0.5) - sampled(result, "T_e_fed_in", 0.5)) < 1e-9
    assert abs(sampled(result, "LAMP_e_fed_in", 0.5) - sampled(result, "T_e_fed_in", 0.5)) < 1e-9


def _forward_reads(document: dict) -> list:
    """Every explicit equation reading an attribute the evaluation order
    sweeps AFTER it: a read of the previous evaluation's value where the
    sweep promises the current one. Reads through a port count: an input
    summing its suppliers reads each one's channel."""
    model = document["model"]
    position = {
        (step["component"], step["attribute"]): index
        for index, step in enumerate(model["evaluation_order"])
    }

    # A port aggregation reads, through each connection into the port,
    # the channel the supplier materialises for that connection, or the
    # allocation writing it, or the supplier port's own attribute.
    components = {component["name"]: component for component in model["components"]}
    feeding: dict = {}
    for connection in model["connections"]:
        destination = connection["to"]
        edge = connection.get("name") or f"{destination['component']}__{destination['port']}"
        feeding.setdefault((destination["component"], destination["port"]), []).append(
            (connection["from"]["component"], connection["from"]["port"], edge)
        )

    def aggregated(port, channel):
        for supplier, supplier_port, edge in feeding.get(
            (port["component"], port["port"]), []
        ):
            if channel is None:
                for declared in components[supplier]["ports"]:
                    if declared["name"] == supplier_port and declared.get("attr"):
                        yield supplier, declared["attr"]
                continue
            materialised = (supplier, f"{supplier_port}__{channel}__{edge}")
            if materialised in position:
                yield materialised
            for allocation in components[supplier].get("allocations", []):
                if allocation["port"] == supplier_port and allocation["allocated"] == channel:
                    yield supplier, allocation["name"]

    def reads(expr, found):
        if isinstance(expr, dict):
            if expr.get("op") == "attr":
                found.add((expr["attr"]["component"], expr["attr"]["attribute"]))
            if expr.get("op") == "port_agg":
                found.update(aggregated(expr["port"], expr.get("channel")))
            for value in expr.values():
                reads(value, found)
        elif isinstance(expr, list):
            for value in expr:
                reads(value, found)

    late = []
    for component in model["components"]:
        for equation in component.get("equations", []):
            if equation["kind"] != "explicit":
                continue
            target = (component["name"], equation["target"])
            found: set = set()
            reads(equation["expr"], found)
            late += [
                (target, read)
                for read in found
                if read in position and position[read] > position[target]
            ]
    return late


def test_the_release_leaves_no_equation_reading_a_later_sweep():
    """Deferring the release to the end of the production band must not
    defer anything a consumer reads during its own visit: a pipe or a
    volume served by the same bus as a releasing input read what arrived
    one evaluation late, and passed on nothing."""

    class Pipe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="e")

    class Lamp(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e", var_demand_in_default=5.0)

    class Hopper(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="k")
            self.add_capacity(name="hopper", flow="e", capacity=50.0, side="in")
            self.add_rule_set(name="r", rules=[{"cons": {"e": 1.0}, "prod": {"k": 1.0}}])

    class Bin(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="k", var_demand_in_default=1.0)

    system = mu.System("rule_input_surplus_order")
    system.add_component(source("e", 30.0), "BUS")
    system.add_component(source("e", 20.0), "SECOND")
    system.add_component(Electrolyser, "EL")
    system.add_component(Sink, "SINK")
    system.add_component(Pipe, "T")
    system.add_component(Lamp, "LAMP")
    system.add_component(Hopper, "V")
    system.add_component(Bin, "BIN")
    system.connect("BUS", "e", "EL", "e")
    system.connect("SECOND", "e", "EL", "e")
    system.connect("EL", "h", "SINK", "h")
    system.connect("BUS", "e", "T", "e")
    system.connect("T", "e", "LAMP", "e")
    system.connect("BUS", "e", "V", "e")
    system.connect("V", "k", "BIN", "k")
    assert _forward_reads(system.build_dict()) == []


def test_an_empty_supplying_volume_passes_on_what_arrives_without_chattering():
    """A battery with nothing in it, beside the source that feeds it,
    both feeding the electrolyser. Scaled with the others it would hand
    on less than it receives, refill, leave its empty bound, offer its
    whole request, drain, and chatter on the bound (13 000 crossings in
    two hours on the H2 plant). Empty, it passes on what arrives first
    and the source's own edge takes the rest: the content stays at zero,
    the bound is never crossed, and the electrolyser gets its need."""

    class Battery(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="e")
            self.add_flow_continuous_out(name="e")
            self.add_capacity(
                name="battery", flow="e", capacity=1.0, fill_rate=0.0, transmits=True
            )

    # A trickle into the empty battery, and a source that alone covers
    # the need: the plant's shape at dawn.
    system = mu.System("rule_input_surplus_empty")
    system.add_component(source("e", 2.0), "TRICKLE")
    system.add_component(source("e", 30.0), "SOLAR")
    system.add_component(Battery, "B1")
    system.add_component(Electrolyser, "EL")
    system.add_component(Sink, "SINK")
    system.connect("TRICKLE", "e", "B1", "e")
    system.connect("SOLAR", "e", "EL", "e")
    system.connect("B1", "e", "EL", "e")
    system.connect("EL", "h", "SINK", "h")
    result = system.simulate(t_max=5.0, samples=[1.0, 3.0, 5.0])
    assert len(result.events) < 10, len(result.events)
    for instant in (1.0, 3.0, 5.0):
        assert abs(sampled(result, "B1_battery_content", instant)) < 1e-6
        assert abs(sampled(result, "EL_e_fed_in", instant) - NEED) < 1e-9
        assert (
            abs(sampled(result, "B1_e_fed_out", instant) - sampled(result, "B1_e_fed_in", instant))
            < 1e-9
        )
