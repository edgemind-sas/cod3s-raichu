"""A trigger input nothing feeds, named before the campaign runs.

`test_muscadet_unconnected_trigger.py` next door settles what such an
input is *worth*: absent, so the standby delivers, from the initial
instant and for ever, on both engines. That is the right semantics and
it is exactly what makes the oversight dangerous: the two engines now
agree on a silently optimistic answer, the run terminates normally, and
the campaign returns an availability that is too good.

This file is the other half. `pyraichu.unfed_triggers(model)` finds the
missing wire from the model alone, before anything is simulated, and
says what to do about it.

It **warns and never refuses**, which is not caution but the design: a
trigger nothing feeds is a *valid* model under the semantics that was
just settled, and refusing it would reject models muscadet runs today.
What is reported is therefore narrow. A guard is pinned only when the
emptiness folds it to a constant exactly, so a condition that also reads
something moving is silent; and the way to declare an always-on output
on purpose is to declare no trigger at all, which is silent too. Same
compromise as `switching_loops`, where a loop whose every switch has a
band is silent.

A trigger is not the only shape of the fault, and the diagnostic no
longer stops at the ones that name a port inside a guard. A rule set
thresholding an input nobody wired -- "if the grid is low, start the
backup" -- is armed from the initial instant and for ever, which is the
same optimism with none of the spelling. The guard there reads
`P.E_capability_in` and not `P.E_in`, so reaching it means **crossing
the definition** the authoring layer put in between, and the two halves
of that crossing are measured in this file:

- `test_what_a_guard_reaches_through_its_definitions` says which
  constructors put a port within a guard's reach at all;
- the rule-set tests below say which of those are worth a line, because
  crossing the definition also reaches modes that are sealed the
  pessimistic way round and must stay silent.
"""

import pyraichu
import pyraichu.declare as declare
import pyraichu.muscadet as mu
from conftest import sampled
from test_muscadet_unconnected_trigger import (
    LOGICS,
    TRIGGERED_DOCUMENT,
    standby_system,
)


def triggers_of(system: mu.System) -> list[dict]:
    """The diagnostic, taken the way a modeller would take it: from the
    system they authored, through the model the engine compiles."""
    return pyraichu.unfed_triggers(system.build_model())


# --- what it finds, over the three trigger logics ------------------------


def test_a_trigger_nothing_feeds_is_found_without_simulating():
    found = triggers_of(standby_system("and", connected=False))
    assert len(found) == 1, found
    assert found[0]["ports"] == ["PumpB.cooling_trigger_in"]
    assert found[0]["automaton"] == "PumpB.cooling_trigger"


def test_the_three_logics_are_all_caught():
    """`and`, `or` and k-out-of-n all answer false over an empty port,
    each for its own reason. A diagnostic covering `and` alone would
    leave two thirds of the fault silent."""
    for logic in LOGICS:
        found = triggers_of(standby_system(logic, connected=False))
        assert len(found) == 1, f"{logic}: {found}"
        assert found[0]["ports"] == ["PumpB.cooling_trigger_in"], logic


def test_a_declared_wait_before_the_rise_is_caught_too():
    """With a `time_up`, the automaton starts `down` and the rise is a
    real delay: the sealed state is not the initial one, and the
    diagnostic has to reach it rather than read it off the declaration."""
    found = triggers_of(standby_system("and", connected=False, time_up=3.0))
    assert len(found) == 1, found
    assert found[0]["state"] == "up"


# --- what it says --------------------------------------------------------


def test_each_entry_locates_the_port_and_says_the_cure():
    found = triggers_of(standby_system("and", connected=False))[0]
    assert set(found) == {"ports", "automaton", "state", "message"}
    message = found["message"]
    assert message.startswith(
        "unfed trigger: nothing is connected to PumpB.cooling_trigger_in"
    ), message
    # The mode, so the finding can be checked rather than trusted.
    assert "PumpB.cooling_trigger" in message, message
    # What it costs, in one clause: this warning exists because the fault
    # does not look like one, so a reader told only that a mode is sealed
    # has no reason to act on it.
    assert "availability the model does not have" in message, message
    # And the remedy, which is what separates a diagnostic that is read
    # from one that is scrolled past.
    assert "Connect that input" in message, message
    # Three things and no more. The length is held against the switching
    # loop message in `the_diagnostic_stays_in_the_register_of_its_sibling`
    # (Rust); here it is enough to say that it fits a terminal.
    assert len(message) < 400, f"{len(message)} characters: {message}"


# --- what it stays silent about ------------------------------------------


def test_a_model_whose_triggers_are_all_fed_is_silent():
    """One `connect_trigger` apart, the same model. The aggregate now
    reads something that moves, so nothing is pinned and there is
    nothing to say."""
    for logic in LOGICS:
        assert triggers_of(standby_system(logic, connected=True)) == [], logic


def test_a_standby_declared_without_a_trigger_is_silent():
    """The deliberate spelling of an output that is always on:
    `add_flow_out` rather than `add_flow_out_on_trigger`. No trigger
    port, nothing to wire, and no warning at every compilation."""

    class Grid(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="power", var_prod_default=True)

    class AlwaysOn(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and")
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])

    system = mu.System(name="always_on")
    system.add_component(Grid, "Grid")
    system.add_component(AlwaysOn, "PumpB")
    system.connect("Grid", "power", "PumpB", "power")
    assert triggers_of(system) == []


def test_an_unconnected_boolean_flow_input_is_out_of_reach():
    """A `power` input nobody wired, read by an output and by no guard.

    Silent, and it is worth being exact about why, because the reason
    changed and only half of it did. It is **still** out of reach by the
    route: `power_fed_in` is written by a *sensitive function*, and an
    attribute a sensitive function assigns is not one the fold crosses.
    Two writers of one attribute are ordinary there, so the fold abstains.
    That half is unmoved, and it is a frontier of reach rather than of
    intent.

    What did move is the VALUE, and with it the whole cost of the silence.
    `all` over an empty port used to be the vacuous truth here, so this
    input read fed from the initial instant and the output delivered for
    ever -- the optimistic direction, which is the one this diagnostic
    exists for. It now reads what muscadet reads, its declared
    `var_in_default`, `False` unless a declaration says otherwise. The
    input is unfed for ever, the output never delivers, and the model
    lands the PESSIMISTIC way round: a component that produces nothing,
    which is what a reader sees in the results.

    So this is no longer a gap waiting to be closed, and that distinction
    is the point of the test. Nothing reaches further than before; what is
    out of reach has stopped flattering the campaign.

    `test_flow_in_default.py` holds that semantics and its twelve cells,
    and the parity against PyCATSHOO is pronounced on a fixture of the
    same twelve in the validation suite.
    """

    class Orphan(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and")
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])

    system = mu.System(name="orphan")
    system.add_component(Orphan, "PumpB")
    assert triggers_of(system) == []

    # The value, measured: the unwired input reads UNFED, and the output
    # with it, from t = 0 to the end of the run.
    result = system.simulate(t_max=20.0, samples=[0.0, 15.0])
    assert sampled(result, "PumpB_cooling_fed_out", 0.0) is False
    assert sampled(result, "PumpB_cooling_fed_out", 15.0) is False

    # And a declared boundary input is the other way round, which is the
    # only shape of unconnected input that still delivers. The diagnostic
    # is silent on it too, and rightly: it is a declaration, not an
    # oversight.
    class Boundary(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and", var_in_default=True)
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])

    declared = mu.System(name="boundary")
    declared.add_component(Boundary, "PumpC")
    assert triggers_of(declared) == []

    result = declared.simulate(t_max=20.0, samples=[0.0, 15.0])
    assert sampled(result, "PumpC_cooling_fed_out", 0.0) is True
    assert sampled(result, "PumpC_cooling_fed_out", 15.0) is True


# --- the rule the emptiness arms, and the one it starves -----------------


#: The rule set this ticket is about, in one factory: "if what the input
#: carries is below `floor`, run". `draws` is the `cons` that makes the
#: mode consume the very flow it thresholds, `wired` the one line a
#: modeller forgets, and `below` the side of the threshold the mode is
#: on.
def rule_system(
    *, wired: bool, draws: bool, below: bool = True, floor: float = 4.0
) -> mu.System:
    condition = {
        "name": "E",
        "port": "in",
        "op": "<" if below else ">=",
        "value": floor,
    }
    running: dict = {"name": "on", "cond": [condition], "prod": {"W": 1.0}}
    if draws:
        running["cons"] = {"E": 10.0}

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="E", var_fed_default=10.0)

    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E")
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[running, {"name": "off", "prod": {"W": 0.0}}],
            )

    system = mu.System(name="rules")
    system.add_component(Source, "S")
    system.add_component(Pump, "P")
    if wired:
        system.connect("S", "E", "P", "E")
    return system


def test_a_rule_armed_by_a_forgotten_wire_is_found():
    """The shape this widening exists for, and the one that flatters.

    "If the grid is low, start the backup", with the wire that carries
    the grid left out. An empty sum is zero, zero is below the floor, so
    the mode is entered at the initial instant and has no way out of it:
    the pump delivers for ever and the campaign reads an availability the
    model does not have.

    The guard names no port -- it thresholds `P.E_capability_in`, the
    attribute the authoring layer derives from the input -- so what the
    report names comes from crossing that definition. It names the
    **port**, because the port is the thing the reader can wire.
    """
    found = triggers_of(rule_system(wired=False, draws=False))
    assert len(found) == 1, found
    assert found[0]["ports"] == ["P.E_in"]
    assert found[0]["automaton"] == "P.duty_mode"
    assert found[0]["state"] == "on"
    assert found[0]["message"].startswith(
        "unfed trigger: nothing is connected to P.E_in"
    ), found[0]["message"]


def test_the_same_rule_with_the_wire_in_place_is_silent():
    """One `connect` apart. The equation now aggregates a port that
    carries something, so nothing is pinned and there is nothing to
    say."""
    assert triggers_of(rule_system(wired=True, draws=False)) == []


def test_a_rule_that_draws_the_flow_nothing_feeds_is_silent():
    """The narrowing, and the reason the widening does not cry.

    Same missing wire, same threshold, same sealed mode -- and a `cons`
    on the flow the condition reads. The rule is entered and cannot
    draw, so it produces zero: the PESSIMISTIC direction, already in
    front of the reader in the results as a pump that makes nothing.

    This is the case the previous test in this place asserted, and it
    stays silent. What has changed is the reason: it was silent because
    nothing crossed the definition, it is silent now because the seal
    raises nothing.
    """
    assert triggers_of(rule_system(wired=False, draws=True)) == []
    assert triggers_of(rule_system(wired=True, draws=True)) == []


def test_a_threshold_the_emptiness_keeps_shut_is_silent():
    """The same rule stated the safe way round, `>=`: the empty input
    holds the mode OFF for ever. Sealed just as thoroughly, in a state
    nothing reads, so nothing is flattered and nothing is said."""
    assert triggers_of(rule_system(wired=False, draws=False, below=False)) == []


def test_the_armed_rule_still_compiles_and_still_runs():
    """A warning and never a refusal, on the shape this ticket adds:
    the model with the forgotten wire loads, simulates, and its output
    is found risen. The rise from 0 to 1 is the whole fault -- it is
    what the missing wire buys, and nothing in the run says so."""
    system = rule_system(wired=False, draws=False)
    assert len(triggers_of(system)) == 1

    result = system.simulate(t_max=20.0, samples=[0.0, 15.0])
    assert sampled(result, "P_W_capability_out", 0.0) == 0.0
    assert sampled(result, "P_W_capability_out", 15.0) == 1.0

    # And the same model drawing the flow: entered, unable to draw, flat
    # at zero. Two models one `cons` apart, and the diagnostic speaks
    # about exactly the one whose number moved.
    drawing = rule_system(wired=False, draws=True)
    flat = drawing.simulate(t_max=20.0, samples=[0.0, 15.0])
    assert sampled(flat, "P_W_capability_out", 15.0) == 0.0


def test_the_input_keeps_its_port_when_nothing_was_declared():
    """What makes the crossing possible at all, and it is a property of
    the generated document rather than of the detector.

    An unconnected continuous input reads `var_in_default`. Declared, it
    is a boundary value and the equation states it as the constant it is.
    Left at zero, it is what the sum over the empty port already answers,
    so the equation states the **aggregation**: same number, and the only
    spelling of the two that says where the number comes from. Write the
    constant in both cases and the port is gone from the document, taking
    with it everything that tells a forgotten wire from a declared
    boundary.
    """

    def capability_expr(default: float) -> dict:
        class Pump(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_continuous_in(name="E", var_in_default=default)

        system = mu.System(name="input")
        system.add_component(Pump, "P")
        document = system.build_dict()
        component = pyraichu.model_body(document)["components"][0]
        return next(
            equation["expr"]
            for equation in component["equations"]
            if equation["target"] == "E_capability_in"
        )

    assert capability_expr(0.0) == {
        "op": "port_agg",
        "port": {"component": "P", "port": "E_in"},
        "agg": "sum",
        "channel": "capability",
    }
    assert capability_expr(7.5) == {
        "op": "const",
        "value": {"kind": "float", "value": 7.5},
    }


# --- how far it reaches, and what decides that ---------------------------


def reachable_port_aggregates(system: mu.System) -> list[str]:
    """The in ports some automaton guard of `system` can reach, read off
    the generated document rather than off the authoring source.

    "Reach" is what the detector's fold does, restated here over the
    document: a guard reaches a port aggregate it contains, and one held
    by the **single explicit definition** of an attribute it reads.
    Restated rather than imported on purpose -- this is the measurement
    that says how far the detector gets on muscadet, and a measurement
    taken with the instrument it measures says nothing.
    """
    document = system.build_dict()
    body = pyraichu.model_body(document)

    writers: dict[tuple[str, str], int] = {}
    solved: dict[tuple[str, str], dict] = {}
    for component in body["components"]:
        for attribute in component.get("attributes", []):
            writers[(component["name"], attribute["name"])] = 0
    for component in body["components"]:
        for equation in component.get("equations", []):
            key = (component["name"], equation["target"])
            if key in writers:
                writers[key] += 1
                if equation.get("kind", "explicit") == "explicit":
                    solved[key] = equation["expr"]
        for function in component.get("sensitive_functions", []):
            for effect in function["effects"]:
                key = (effect["target"]["component"], effect["target"]["attribute"])
                if key in writers:
                    writers[key] += 1

    found: set[str] = set()

    def walk(node, crossed: frozenset) -> None:
        if isinstance(node, dict):
            if node.get("op") == "port_agg":
                found.add(node["port"]["port"])
            if node.get("op") == "attr":
                key = (node["attr"]["component"], node["attr"]["attribute"])
                if writers.get(key) == 1 and key in solved and key not in crossed:
                    walk(solved[key], crossed | {key})
            for value in node.values():
                walk(value, crossed)
        elif isinstance(node, list):
            for value in node:
                walk(value, crossed)

    for component in body["components"]:
        for automaton in component.get("automata", []):
            for transition in automaton["transitions"]:
                walk(transition.get("guard"), frozenset())
    return sorted(found)


def test_what_a_guard_reaches_through_its_definitions():
    """How far the diagnostic can reach, measured on generated documents.

    The definition in the core is structural and knows nothing of
    muscadet's spellings: an in port whose emptiness seals a mode, read
    through whatever single-definition equations lie between. Of
    muscadet's constructors, **two** put a port within that reach, and
    this test is what makes it a fact rather than a hope:

    - `add_flow_out_on_trigger` writes the aggregate inside the guard;
    - `add_rule_set` reaches one through the explicit equation of a
      **continuous** input it thresholds.

    Everything else stays out of reach, and one of them deserves to be
    named, though no longer for the reason it once was: `add_rule_set` on
    a **boolean** condition. Its guard reads `{flow}_fed_in`, which a
    *sensitive function* assigns, and the fold does not cross a sensitive
    function -- two writers of one attribute are ordinary there. That is
    where the frontier sits, and it will move the day the fold crosses
    that route.

    What it is NOT, any more, is the same fault in the other half of the
    layer. A boolean input nothing feeds now reads its declared
    `var_in_default` -- `False` unless stated -- so the mode it guards
    stays SHUT for ever instead of being armed for ever. Pinned, not
    pessimism by accident: the emptiness is written into the aggregating
    expression (`test_flow_in_default.py`). A mode sealed shut starves the
    component that reads it, which shows in the results as a component
    producing nothing, and that is the direction this diagnostic
    deliberately stays silent on.

    A constructor that one day put a port within reach would widen the
    diagnostic without one line changing in `triggers.rs`. This test
    fails on that day, which is the point: the reach is decided by the
    authoring layer, so it has to be measured there.
    """

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="power", var_prod_default=True)

    def system_of(component_class) -> mu.System:
        system = mu.System(name="probe")
        system.add_component(Source, "S")
        system.add_component(component_class, "P")
        return system

    class Plain(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and")
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])

    class Tempo(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power")
            self.add_flow_out_tempo(
                name="cooling", enable_time=2.0, var_prod_cond=[["power"]]
            )

    class Failing(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power")
            self.add_flow_out(name="cooling", var_prod_cond=[["power"]])
            self.add_exp_failure_mode(
                name="hw", failure_rate=0.01, repair_rate=0.1, targets=["cooling"]
            )

    class DiscreteRules(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="power", logic="and")
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[
                    {"name": "on", "cond": ["power"], "prod": {"W": 1.0}},
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    class Store(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E", var_demand_in_default=5.0)
            self.add_flow_continuous_out(name="E", var_fed_default=5.0)
            self.add_capacity(
                name="store", flow="E", capacity=100.0, content_init={"E": 50.0}
            )

    class ContinuousRules(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E")
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[
                    {
                        "name": "on",
                        "cond": [
                            {"name": "E", "port": "in", "op": "<", "value": 4.0}
                        ],
                        "prod": {"W": 1.0},
                    },
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    class Triggered(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out_on_trigger(
                name="cooling", trigger_logic="and", var_prod_default=True
            )

    for component_class in (Plain, Tempo, Failing, DiscreteRules):
        system = system_of(component_class)
        assert reachable_port_aggregates(system) == [], component_class.__name__

    store = mu.System(name="store")
    store.add_component(Store, "P")
    assert reachable_port_aggregates(store) == []

    rules = system_of(ContinuousRules)
    assert reachable_port_aggregates(rules) == ["E_in"]

    triggered = mu.System(name="triggered")
    triggered.add_component(Triggered, "P")
    assert reachable_port_aggregates(triggered) == ["cooling_trigger_in"]


# --- a warning, not a refusal --------------------------------------------


def test_the_model_still_compiles_and_still_runs():
    """The point that separates a warning from a refusal, and it is
    checked by running rather than by reading: the model with the
    forgotten wire loads, simulates, and returns its series. The backup
    is found delivering at t = 0 and still delivering at t = 15, which
    is the optimism the diagnostic exists to announce."""
    system = standby_system("and", connected=False)
    assert len(triggers_of(system)) == 1

    result = system.simulate(t_max=20.0, samples=[0.0, 15.0])
    assert sampled(result, "PumpB_cooling_fed_out", 0.0) == 1.0
    assert sampled(result, "PumpB_cooling_fed_out", 15.0) == 1.0


def test_the_declaration_reader_answers_the_same():
    """The path a muscadet run actually takes through the engine seam:
    the declaration reader rather than the class surface. It builds the
    same ports and the same guards, so it gets the same diagnostic."""
    found = triggers_of(declare.build_system(TRIGGERED_DOCUMENT))
    assert len(found) == 1, found
    assert found[0]["ports"] == ["PumpB.cooling_trigger_in"]
    assert found[0]["automaton"] == "PumpB.cooling_trigger"


# --- and the loop detector is left alone ---------------------------------


def test_the_two_diagnostics_answer_different_questions():
    """A trigger nothing feeds is not a cycle of the dependency graph,
    so the loop detector says nothing about it, and it is right not to.
    Keeping them apart is what lets each stay readable."""
    model = standby_system("and", connected=False).build_model()
    assert pyraichu.switching_loops(model) == []
    assert len(pyraichu.unfed_triggers(model)) == 1
