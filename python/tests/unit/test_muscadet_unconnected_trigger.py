"""A trigger input nothing feeds: the backup runs, and from t = 0.

A `FlowOutOnTrigger` is a cold standby: it delivers while its trigger input
is ABSENT, which is how the loss of the main equipment starts the backup.
The modeller has to wire that input, and the question this file settles is
what the flow is worth when they forget.

muscadet's answer, read first-hand off muscadet 5.3.1 and PyCATSHOO on
2026-09-13, and the same for the three trigger logics
-----------------------------------------------------------------------
An unconnected input reads the **no-connection value** the aggregation is
given, and that value is `False` for the three logics of a trigger
(`muscadet/flow.py`: `andValue(False)`, `orValue(False)`,
`sumValue(0) >= k`). muscadet states it in that file, over the same
aggregation on a flow input: "the value pass in andValue and orValue is the
returned value in the case of no connection". Measured on the same
reference with zero connections, `andValue(False)` answers `False` while
`andValue(True)` answers `True`, so the argument IS the answer and the
conjunction over nothing is never taken as vacuously true.

A trigger nothing feeds therefore inhibits nothing: its up condition
`not(aggregate)` holds, and the standby delivers. `examples/isimu/
power_plant` says the same of its own backup pump: "With
`trigger_logic="and"`, the backup activates when the trigger input is False
(main pump no longer feeding)."

**It is not a particularity of `"and"`.** `any` over nothing and `sum >= k`
over nothing are already false, so all three logics arm. What differed on
RAICHU was `all`, whose aggregate over an empty port is the vacuous truth:
the backup never started, and two engines answered opposite availabilities
on one model with nothing to say so.

What is pinned here
-------------------
- the standby delivers from the **initial instant**, for `and`, for `or`
  and for an integer k. The reference engine reaches `up` through a
  zero-delay transition inside the initial instant, so the generated
  automaton starts there rather than arriving a moment later;
- a trigger that IS wired still inhibits: the fix cannot be "always up";
- a declared `time_up` is still a wait, unconnected or not;
- the declaration reader gets the same answers as the class surface, which
  is the path a muscadet run takes through the engine seam.

The two engines are compared on this over muscadet's own `power_plant` by
the cross-validation suite's `test_muscadet_unconnected_trigger.py`, which
runs where a PyCATSHOO installation is available. Here there is no oracle:
what RAICHU answers is a property of RAICHU, and it goes on being checked
on a machine that has no PyCATSHOO.
"""

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu
from conftest import sampled

#: The three trigger logics, `and` and `or` and an integer k, over which
#: every claim below is made: the aggregate over an empty port is false in
#: all three, and an alignment covering `and` alone would leave two thirds
#: of the divergence open. Two values of k rather than one: `sum >= 1` and
#: `sum >= 2` are the same expression and a different threshold, and the
#: emptiness has to answer for both.
LOGICS = ("and", "or", 1, 2)


def standby_system(
    logic: str | int, *, connected: bool, time_up: float = 0.0
) -> mu.System:
    """A grid, a main pump failing for good at t = 8, and a backup pump
    whose output is triggered by the main pump's -- wired or not.

    The shape of `examples/isimu/power_plant` reduced to what the trigger
    needs: `var_prod_cond=[["power"]]` on the backup, so a standby armed
    for want of a trigger still has to be powered to deliver.
    """

    class Grid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="power", var_prod_default=True)

    class MainPump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and")
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])
            self.add_delay_failure_mode(
                name="hw", failure_time=8.0, repair_time=1e9, targets=["cooling"]
            )

    class BackupPump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and")
            self.add_flow_out_on_trigger(
                name="cooling",
                trigger_logic=logic,
                trigger_time_up=time_up,
                trigger_time_down=0.0,
                var_prod_default=True,
                var_prod_cond=[["power"]],
            )

    system = mu.System(name="standby")
    system.add_component(Grid, "Grid")
    system.add_component(MainPump, "PumpA")
    system.add_component(BackupPump, "PumpB")
    system.connect("Grid", "power", "PumpA", "power")
    system.connect("Grid", "power", "PumpB", "power")
    if connected:
        system.connect_trigger("PumpA", "PumpB", "cooling")
    return system


def trigger_automaton(system: mu.System, component: str, flow: str) -> dict:
    """The trigger automaton of one flow, out of the generated model."""
    built = next(
        entry
        for entry in system.build_dict()["components"]
        if entry["name"] == component
    )
    return next(
        automaton
        for automaton in built["automata"]
        if automaton["name"] == f"{flow}_trigger"
    )


# --- the semantics, over the three logics -------------------------------


@pytest.mark.parametrize("logic", LOGICS)
def test_an_unconnected_trigger_delivers_from_the_initial_instant(logic):
    """The alignment itself: nothing feeds the trigger, so nothing inhibits
    the standby, and it delivers at t = 0 like the reference engine's.

    Before the alignment, `and` answered 0 throughout (a conjunction over
    an empty port being vacuously true, the up condition never held) while
    `or` and k answered 0 at t = 0 alone (the automaton reaching `up`
    through a transition the initial sample is taken before)."""
    system = standby_system(logic, connected=False)
    result = system.simulate(t_max=20.0, samples=[0.0, 4.0, 9.0, 15.0])

    delivered = [
        sampled(result, "PumpB_cooling_fed_out", instant)
        for instant in (0.0, 4.0, 9.0, 15.0)
    ]
    assert delivered == [1.0, 1.0, 1.0, 1.0], (
        f"trigger_logic={logic!r}: a standby whose trigger nothing feeds "
        f"delivers throughout, the initial instant included"
    )


@pytest.mark.parametrize("logic", LOGICS)
def test_a_wired_trigger_still_inhibits_the_standby(logic):
    """The other half, and the reason the fix is not "always up": wired to
    a main pump that feeds, the standby stays idle until that pump falls.

    Read at t = 9 rather than t = 8: the failure date is where the two
    engines legitimately differ on the observation convention, and this
    claim is about the standby taking over, not about that instant.

    k = 2 is left out, and deliberately: `sum >= 2` over ONE connection is
    never satisfied, so a 2-of-1 trigger is armed from the start whatever
    is wired to it. That is muscadet's arithmetic, not this alignment's,
    and the model it describes is degenerate."""
    if logic == 2:
        pytest.skip("`sum >= 2` over one connection is never satisfied")
    system = standby_system(logic, connected=True)
    result = system.simulate(t_max=20.0, samples=[0.0, 4.0, 9.0, 15.0])

    assert sampled(result, "PumpB_cooling_fed_out", 0.0) == 0.0
    assert sampled(result, "PumpB_cooling_fed_out", 4.0) == 0.0
    assert sampled(result, "PumpB_cooling_fed_out", 9.0) == 1.0
    assert sampled(result, "PumpB_cooling_fed_out", 15.0) == 1.0


@pytest.mark.parametrize("logic", LOGICS)
def test_an_unconnected_trigger_still_waits_the_time_it_declares(logic):
    """`trigger_time_up` is a real wait, and an unconnected trigger does
    not shorten it: the standby is armed at t = 0 and delivers at t = 3.

    This is what keeps the initial state from being read as "unconnected
    means up": it means "armed", and the declared delay decides when that
    arming delivers."""
    system = standby_system(logic, connected=False, time_up=3.0)
    result = system.simulate(t_max=20.0, samples=[0.0, 2.0, 4.0, 9.0])

    assert sampled(result, "PumpB_cooling_fed_out", 0.0) == 0.0
    assert sampled(result, "PumpB_cooling_fed_out", 2.0) == 0.0
    assert sampled(result, "PumpB_cooling_fed_out", 4.0) == 1.0
    assert sampled(result, "PumpB_cooling_fed_out", 9.0) == 1.0


# --- and what the generated model says of it -----------------------------


@pytest.mark.parametrize("logic", LOGICS)
def test_the_generated_automaton_starts_where_the_reference_reads_it(logic):
    """The initial instant, read off the document rather than off a run.

    With nothing connected the aggregate is false for ever, so the up
    transition is enabled at t = 0 and the down one never is: `down` is a
    state of zero duration, and the automaton is declared in the state the
    reference engine already reads at t = 0. Wired, or waiting on a
    declared `time_up`, it starts `down` and the transition does the work.
    """
    unconnected = trigger_automaton(
        standby_system(logic, connected=False), "PumpB", "cooling"
    )
    assert unconnected["init"] == "up"

    wired = trigger_automaton(
        standby_system(logic, connected=True), "PumpB", "cooling"
    )
    assert wired["init"] == "down"

    waiting = trigger_automaton(
        standby_system(logic, connected=False, time_up=3.0), "PumpB", "cooling"
    )
    assert waiting["init"] == "down"


def test_the_and_aggregate_carries_the_emptiness_of_its_port():
    """`all` over an empty port is the vacuous truth, so the emptiness is
    written into the expression: `all(port) AND count(port) >= 1`.

    Asserted on the document because it is a claim about what the model
    SAYS: a condition on the port and not on the connection list, so the
    guard means the same thing whatever is wired to it, and a model
    rebuilt with a connection added needs no regeneration to stay right.
    """
    automaton = trigger_automaton(
        standby_system("and", connected=True), "PumpB", "cooling"
    )
    up = next(
        transition
        for transition in automaton["transitions"]
        if transition["name"] == "cooling_trigger_up"
    )
    # The up condition is `not(aggregate)`: the aggregate is its argument.
    aggregate = up["guard"]["args"][0]
    assert aggregate["bool_op"] == "and"
    assert [
        (term.get("agg") or term["lhs"]["agg"]) for term in aggregate["args"]
    ] == ["all", "count"]


# --- the same, through the declaration reader ----------------------------
#
# The path a muscadet run takes: what crosses the engine seam is
# `muscadet.declare.system_spec`, a document, and the reader builds the
# authoring layer from it. A claim held by the class surface alone would say
# nothing about the run the parity bench measures.

TRIGGERED_DOCUMENT = {
    "version": 1,
    "name": "standby",
    "components": {
        "Grid": {
            "name": "Grid",
            "cls": "ObjFlow",
            "flows": [
                {
                    "cls": "FlowOut",
                    "name": "power",
                    "var_prod_default": True,
                }
            ],
        },
        "PumpB": {
            "name": "PumpB",
            "cls": "ObjFlow",
            "flows": [
                {"cls": "FlowIn", "name": "power", "logic": "and"},
                {
                    "cls": "FlowOutOnTrigger",
                    "name": "cooling",
                    "var_prod_default": True,
                    "var_prod_cond": [["power"]],
                    "trigger_time_up": 0.0,
                    "trigger_time_down": 0.0,
                    "trigger_logic": "and",
                },
            ],
        },
    },
    "connections": [
        {
            "source": "Grid",
            "source_box": "power_out",
            "target": "PumpB",
            "target_box": "power_in",
        }
    ],
    "indicators": [],
}


def test_the_declaration_reader_arms_an_unconnected_trigger_too():
    """The document declares a trigger and wires nothing to it: the standby
    delivers from the initial instant, as it does on the class surface."""
    system = declare.build_system(TRIGGERED_DOCUMENT)
    result = system.simulate(t_max=10.0, samples=[0.0, 5.0])

    assert sampled(result, "PumpB_cooling_fed_out", 0.0) == 1.0
    assert sampled(result, "PumpB_cooling_fed_out", 5.0) == 1.0


def test_the_declaration_reader_leaves_a_wired_trigger_inhibiting():
    """The same document with the trigger wired to a source that feeds:
    the standby stays down, which is what the reader must not lose."""
    document = {
        **TRIGGERED_DOCUMENT,
        "connections": TRIGGERED_DOCUMENT["connections"]
        + [
            {
                "source": "Grid",
                "source_box": "cooling_out",
                "target": "PumpB",
                "target_box": "cooling_trigger_in",
            }
        ],
    }
    document["components"] = {
        **TRIGGERED_DOCUMENT["components"],
        "Grid": {
            "name": "Grid",
            "cls": "ObjFlow",
            "flows": [
                {"cls": "FlowOut", "name": "power", "var_prod_default": True},
                {"cls": "FlowOut", "name": "cooling", "var_prod_default": True},
            ],
        },
    }

    system = declare.build_system(document)
    result = system.simulate(t_max=10.0, samples=[0.0, 5.0])

    assert sampled(result, "PumpB_cooling_fed_out", 0.0) == 0.0
    assert sampled(result, "PumpB_cooling_fed_out", 5.0) == 0.0
