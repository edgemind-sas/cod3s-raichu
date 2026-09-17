"""What a boolean in-flow reads while NOTHING is connected to it.

muscadet answers that question with a declared value, `var_in_default`
(`FlowDiscreteIn`, default `False`), and the answer is not the aggregate's
own convention: `all` over an empty port is the vacuous TRUTH, so an `and`
input nobody wired used to read fed here and unfed there. The output it
conditions then delivered from the initial instant to the end of the run,
which is the expensive direction -- a campaign that comes back too sure of
itself, on a model whose only fault is a forgotten wire.

Three of the twelve cells diverged, not one, and the second cause is worth
naming because it was silent: `add_flow_in` dropped `var_in_default` on the
k-out-of-n branch, so a caller who HAD declared an always-fed boundary
input got the opposite answer, with no error and no warning. That one is
pessimistic rather than optimistic, but two engines answering opposite
availabilities on one model is the fault either way round.

The cure is written where the trigger port next door writes it: INTO the
expression, as `count(p) >= 1`, never by indexing on the connection list.
A rule written on the port means the same thing whatever is wired to it, so
a component generates the same document wherever it is instantiated and a
model rebuilt with one connection more has nothing to regenerate.

muscadet's own dissymmetry is why a cure that treats only `and` is not
enough, and this file pins both halves of it: the default is an ARGUMENT of
the aggregate for `and` and `or` (`andValue`, `orValue`) and an early
return guarded by `nbCnx() == 0` for an integer k.

The parity itself is pronounced against PyCATSHOO, next door in the
validation suite, on a fixture holding these same twelve cells. What this
file holds is RAICHU's side of it, plus the two things the oracle cannot
say: that the two AUTHORING SURFACES agree, and what the generated document
looks like.
"""

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu
from conftest import held_at, settled

#: The four aggregations a boolean input may declare. `1` and `2` are
#: k-out-of-n, spelled as muscadet spells them: a bare integer.
LOGICS = ("and", "or", 1, 2)

#: The three ways a declaration can leave `var_in_default`: unsaid, and
#: said at either value. Unsaid is not a fourth semantics -- it IS
#: muscadet's `False` -- and the three are kept apart all the same,
#: because the layer stores "unsaid" as `None` and a cure that only
#: looked at the declared values would leave the common case behind.
DEFAULTS = (None, True, False)

#: What each cell reads out of connection: muscadet's answer, which is
#: the declared default and nothing else. Measured against PyCATSHOO /
#: muscadet 5.3.1 on 2026-09-15, all four logics, both instants: this
#: table IS the oracle's twelve answers, transcribed.
UNCONNECTED = {default: bool(default) for default in DEFAULTS}


def lonely_authored(logic, default) -> mu.System:
    """One component, one in-flow nothing feeds, one output it conditions.

    The smallest model that shows the fault: no connection at all, so
    every cell of the table is decided by the declaration alone."""

    class Lonely(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic=logic, var_in_default=default)
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])

    system = mu.System(name="lonely")
    system.add_component(Lonely, "L")
    return system


def lonely_declared(logic, default) -> mu.System:
    """The same component, through the declaration reader.

    The route a real muscadet run takes: a system declaration crosses the
    engine seam and is built here, so a cure that only reached the
    authoring surface would leave every platform model behind."""
    flow = {"cls": "FlowIn", "name": "power", "logic": logic}
    if default is not None:
        flow["var_in_default"] = default
    system = mu.System(name="lonely")
    declare.build_component(
        system,
        {
            "name": "L",
            "flows": [
                flow,
                {"cls": "FlowOut", "name": "cooling", "var_prod_cond": ["power"]},
            ],
        },
    )
    return system


SURFACES = {"authored": lonely_authored, "declared": lonely_declared}


# --- the twelve cells, out of connection ---------------------------------


@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize("default", DEFAULTS)
@pytest.mark.parametrize("logic", LOGICS)
def test_an_unconnected_input_reads_what_it_declares(surface, default, logic):
    """The twelve cells, on both authoring surfaces.

    Held from the initial instant to the end of the run, which is not a
    detail of the assertion but the whole shape of the fault: nothing can
    ever reach an input nothing feeds, so whatever it reads at t = 0 it
    reads for ever, and a wrong answer there is a wrong answer everywhere.
    """
    system = SURFACES[surface](logic, default)
    result = system.simulate(t_max=20.0, samples=[0.0, 15.0])

    expected = UNCONNECTED[default]
    fed_in = settled(result.indicators["L_power_fed_in"])
    assert held_at(fed_in, 0.0) is expected
    assert held_at(fed_in, 15.0) is expected


@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize("default", DEFAULTS)
@pytest.mark.parametrize("logic", LOGICS)
def test_the_output_it_conditions_follows_it(surface, default, logic):
    """And the cost of the cell, which is what a reader of the results
    sees: a component that delivers, or one that does not."""
    system = SURFACES[surface](logic, default)
    result = system.simulate(t_max=20.0, samples=[0.0, 15.0])

    expected = UNCONNECTED[default]
    fed_out = settled(result.indicators["L_cooling_fed_out"])
    assert held_at(fed_out, 0.0) is expected
    assert held_at(fed_out, 15.0) is expected


def test_the_boundary_input_is_the_one_cell_that_delivers():
    """The same table read the other way round, so the point survives a
    parametrisation that could silently stop covering anything.

    Four of the twelve cells deliver, and they are exactly the four that
    DECLARE `var_in_default=True`. An always-fed boundary input is how a
    model grounds a chain at the physical edge -- an external supply, a
    utility nobody modelled -- and it is the one thing an unconnected
    input may legitimately be."""
    delivering = {
        (default, logic)
        for default in DEFAULTS
        for logic in LOGICS
        if held_at(
            settled(
                lonely_authored(logic, default)
                .simulate(t_max=20.0, samples=[0.0])
                .indicators["L_cooling_fed_out"]
            ),
            0.0,
        )
    }

    assert delivering == {(True, logic) for logic in LOGICS}


# --- and what a wire changes, which is everything -------------------------


def wired(logic, default=None, sources=2, failure_at=8.0) -> mu.System:
    """The same input, this time fed by `sources` producers, the first of
    which fails at `failure_at`.

    The counterpart the cure must not disturb: once a port carries
    connections, `count(p) >= 1` is true and the aggregate decides alone,
    so every wired model answers exactly what it answered before."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="power", var_prod_default=True)

    class Failing(Source):
        def add_flows(self):
            super().add_flows()
            self.add_delay_failure_mode(
                name="hw",
                failure_time=failure_at,
                repair_time=1000.0,
                targets=["power"],
            )

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic=logic, var_in_default=default)
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])

    system = mu.System(name="wired")
    system.add_component(Failing, "S1")
    for index in range(2, sources + 1):
        system.add_component(Source, f"S{index}")
    system.add_component(Consumer, "C")
    for index in range(1, sources + 1):
        system.connect(f"S{index}", "power", "C", "power")
    return system


#: What the consumer reads before and after the single failure, per logic,
#: on two producers. `and` and k = 2 need both, so they drop; `or` and
#: k = 1 need one, so they hold.
WIRED_ANSWERS = {
    "and": (True, False),
    "or": (True, True),
    1: (True, True),
    2: (True, False),
}


@pytest.mark.parametrize("logic", LOGICS)
def test_a_wired_input_still_reads_its_aggregate(logic):
    """Two producers, one failing at t = 8, over the four logics."""
    result = wired(logic).simulate(t_max=20.0, samples=[0.0, 15.0])
    fed_in = settled(result.indicators["C_power_fed_in"])

    before, after = WIRED_ANSWERS[logic]
    assert held_at(fed_in, 0.0) is before
    assert held_at(fed_in, 15.0) is after


@pytest.mark.parametrize("logic", LOGICS)
def test_a_declared_default_says_nothing_once_a_wire_is_there(logic):
    """And `var_in_default` declared `True` changes none of it.

    This is the assertion that makes the cure a rule about EMPTINESS
    rather than a second opinion on the aggregate: a boundary input that
    someone later wires is decided by what feeds it, exactly as if it had
    declared nothing."""
    result = wired(logic, default=True).simulate(t_max=20.0, samples=[0.0, 15.0])
    fed_in = settled(result.indicators["C_power_fed_in"])

    before, after = WIRED_ANSWERS[logic]
    assert held_at(fed_in, 0.0) is before
    assert held_at(fed_in, 15.0) is after


# --- the generated document ----------------------------------------------


def aggregation_of(system: mu.System, component: str, flow: str) -> dict:
    """The expression the sensitive function of `{flow}_fed_in` assigns."""
    body = system.build_dict()
    entry = next(c for c in body["components"] if c["name"] == component)
    function = next(
        f
        for f in entry["sensitive_functions"]
        if f["name"] == f"update_{flow}_fed_in"
    )
    return function["effects"][0]["value"]


def counts_ports(expression) -> bool:
    """Whether `count` appears anywhere in an expression."""
    if isinstance(expression, dict):
        if expression.get("agg") == "count":
            return True
        return any(counts_ports(value) for value in expression.values())
    if isinstance(expression, list):
        return any(counts_ports(item) for item in expression)
    return False


@pytest.mark.parametrize("logic", ("and", 1, 2))
def test_the_emptiness_is_written_into_the_expression(logic):
    """`and` and k-out-of-n carry the port's own count.

    Not the connection list: the document a component generates is the
    same whatever is wired to it, which is what lets a system add a
    connection without regenerating the components it joins."""
    assert counts_ports(aggregation_of(lonely_authored(logic, None), "L", "power"))


def test_an_or_needs_no_count_of_its_own():
    """`any` that answers true already names a connection, so the
    emptiness needs no term there, and a redundant one would be dead
    weight on every `or` input of every model."""
    assert not counts_ports(aggregation_of(lonely_authored("or", None), "L", "power"))


def test_the_document_never_reads_the_connection_list():
    """The same component, alone and wired, generates the same in-flow
    expression. This is the structural form of the two tests above, and
    the one that would catch a cure quietly reindexed on `connected_in`."""
    alone = aggregation_of(lonely_authored("and", None), "L", "power")
    joined = aggregation_of(wired("and"), "C", "power")

    def renamed(expression, component):
        if isinstance(expression, dict):
            return {
                key: (component if key == "component" else renamed(value, component))
                for key, value in expression.items()
            }
        if isinstance(expression, list):
            return [renamed(item, component) for item in expression]
        return expression

    assert renamed(alone, "X") == renamed(joined, "X")


# --- the availability channel, which does not diverge ---------------------


def test_the_availability_default_is_accepted_only_where_it_says_nothing():
    """muscadet's counterpart on the OTHER channel,
    `var_available_in_default`, defaults to `True`, and this layer carries
    it at that value alone.

    That is not an omission left to be found later, and the reason is
    worth keeping beside the feed channel it belongs to:

    - **out of connection** `True` is the neutral element. muscadet reads
      `fed = agg_in(var_in, var_in_default) AND agg_avail(avail, True)`,
      so on a port nothing feeds the second factor is true and the FEED
      channel decides alone -- which is exactly the twelve cells above;
    - **wired**, it is implied point by point by the feed channel. A
      producer publishes `fed = prod AND is_active AND fed_available`, so
      `fed` entails `available` per producer, and `all`, `any` and
      `sum >= k` are monotone: the second factor cannot change a verdict
      the first has given. The one construction that would break the
      implication is a NEGATED output, which this layer refuses.

    So `False` would say something this layer cannot honour, and it is
    refused by name rather than dropped."""
    spec = {
        "name": "L",
        "flows": [
            {
                "cls": "FlowIn",
                "name": "power",
                "logic": "and",
                "var_available_in_default": False,
            }
        ],
    }

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(mu.System(name="probe"), spec)

    assert "var_available_in_default" in str(raised.value)

    spec["flows"][0]["var_available_in_default"] = True
    declare.build_component(mu.System(name="probe"), spec)


def test_a_negated_output_is_what_would_break_the_implication():
    """The exception named just above, pinned rather than argued.

    `negate` inverts the whole conjunction on a muscadet output, so a
    negated output can publish `fed` while its availability is false --
    and then a consumer's two channels no longer agree. Neither surface
    can declare one: the authoring layer has no such keyword, and the
    declaration reader refuses the key at anything but `False`."""
    assert "negate" not in mu.ObjFlow.add_flow_out.__code__.co_varnames

    spec = {
        "name": "L",
        "flows": [{"cls": "FlowOut", "name": "cooling", "negate": True}],
    }
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(mu.System(name="probe"), spec)

    assert "negate" in str(raised.value)
