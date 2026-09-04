"""Transformation rules in the `pyraichu.muscadet` authoring layer.

A rule set is an ordered list of rules, each carrying a guard, a map of
consumed input coefficients and a map of produced output coefficients.
The set runs at the scale its scarcest input and its least demanded
output allow.

What the generated model must hold, and what these tests pin:

- the scale as a **continuous minimum** over the inputs divided by their
  coefficients and over the demanded outputs divided by theirs, with no
  crossing transition per input pair: when the limiting input changes
  identity the output keeps a kink and never jumps, which is the whole
  claim the encoding rests on;
- correlated outputs held in their declared proportion, because one
  scale serves the whole rule;
- ordered selection: the guarded rules are tried in order and the
  unguarded one is the fallback, through a mode automaton whose
  transitions are **watched** when a guard reads a continuous quantity,
  so a threshold is crossed at its instant rather than at the following
  step;
- the three build-time refusals the rule vocabulary makes sayable: a
  self-feeding cycle that creates matter, a loop of rate comparisons,
  and a contested output with no declared apportionment.
"""

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, TOL, fired_at, sampled


def settled(result, indicator: str) -> float:
    """The value `indicator` settled on at the initial instant.

    The last record at t = 0, not the first: a rule set starts in its
    default (or in "no rule applies") and the transitions fireable at
    t = 0 fire there, so the first record predates the selection."""
    series = result.indicators[indicator]
    assert series, f"indicator `{indicator}` recorded nothing"
    initial = [value for time, value in series if time == 0.0]
    assert initial, f"indicator `{indicator}` recorded nothing at t=0"
    return initial[-1]


class Sink(mu.ObjFlow):
    """A pure consumer of `x`, asking for more than any rule here can
    make, so the output side never sets the scale."""

    def add_flows(self):
        self.add_flow_continuous_in(name="x", var_demand_in_default=1000.0)


def source(flow: str, quantity: float):
    """A pure producer of `quantity` units of `flow`."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=quantity)

    return Source


# --- the scarcest input sets the scale ---------------------------------


class Reactor(mu.ObjFlow):
    """Four units of `a` and one of `b` make one of `x`."""

    def add_flows(self):
        self.add_flow_continuous_in(name="a")
        self.add_flow_continuous_in(name="b")
        self.add_flow_continuous_out(name="x")
        self.add_rule_set(
            name="make_x", rules=[{"cons": {"a": 4, "b": 1}, "prod": {"x": 1}}]
        )


def reaction(offered_a: float, offered_b: float) -> mu.System:
    system = mu.System("rule_reaction")
    system.add_component(source("a", offered_a), "A")
    system.add_component(source("b", offered_b), "B")
    system.add_component(Reactor, "R")
    system.add_component(Sink, "K")
    system.connect("A", "a", "R", "a")
    system.connect("B", "b", "R", "b")
    system.connect("R", "x", "K", "x")
    return system


def test_the_scarcest_input_sets_the_scale():
    """`4.a + 1.b -> 1.x` fed 8 of `a` and 1 of `b` runs at scale 1:
    `b` is the limiting reagent, and the surplus of `a` raises
    nothing."""
    result = reaction(8.0, 1.0).simulate(t_max=1.0)
    assert abs(settled(result, "K_x_fed_in") - 1.0) < TOL
    # Consumption is in the declared ratio, and the abundant input is
    # drawn at its coefficient, not at what it could deliver.
    assert abs(settled(result, "R_a_fed_in") - 4.0) < TOL
    assert abs(settled(result, "R_b_fed_in") - 1.0) < TOL


def test_halving_the_scarcer_input_halves_the_output():
    """The scale is a ratio, so halving the limiting reagent halves what
    the rule makes and what it draws from every input."""
    result = reaction(8.0, 0.5).simulate(t_max=1.0)
    assert abs(settled(result, "K_x_fed_in") - 0.5) < TOL
    assert abs(settled(result, "R_a_fed_in") - 2.0) < TOL
    assert abs(settled(result, "R_b_fed_in") - 0.5) < TOL


def test_correlated_outputs_keep_their_declared_proportion():
    """One scale serves the whole rule, so a reaction making two things
    at once makes them in the declared ratio even when only one of them
    is asked for less than it could deliver."""

    class TwoProducts(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_flow_continuous_out(name="y")
            self.add_rule_set(
                name="split",
                rules=[{"cons": {"a": 1}, "prod": {"x": 1, "y": 2}}],
            )

    class Small(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x", var_demand_in_default=3.0)

    system = mu.System("rule_correlated")
    system.add_component(source("a", 100.0), "A")
    system.add_component(TwoProducts, "R")
    system.add_component(Small, "KX")
    system.add_component(Sink, "KY")
    system.connect("A", "a", "R", "a")
    system.connect("R", "x", "KX", "x")
    system.connect("R", "y", "KY", "x")
    result = system.simulate(t_max=1.0)

    # `x` is wanted at 3, so the rule runs at scale 3 and `y` follows at
    # twice that: 6, not the 200 an unconstrained `y` could take.
    assert abs(settled(result, "KX_x_fed_in") - 3.0) < TOL
    assert abs(settled(result, "KY_x_fed_in") - 6.0) < TOL
    assert abs(settled(result, "R_a_fed_in") - 3.0) < TOL


def test_a_rule_whose_output_is_unwanted_scales_to_nothing():
    """A consumer asking for nothing is a real bound: the rule makes
    nothing and therefore draws nothing from its suppliers."""

    class Refuser(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x", var_demand_in_default=0.0)

    system = mu.System("rule_unwanted")
    system.add_component(source("a", 100.0), "A")
    system.add_component(source("b", 100.0), "B")
    system.add_component(Reactor, "R")
    system.add_component(Refuser, "K")
    system.connect("A", "a", "R", "a")
    system.connect("B", "b", "R", "b")
    system.connect("R", "x", "K", "x")
    result = system.simulate(t_max=1.0)

    assert settled(result, "K_x_fed_in") == 0.0
    assert settled(result, "R_a_fed_in") == 0.0
    assert settled(result, "R_b_fed_in") == 0.0


# --- the limiting input changes identity, and nothing jumps ------------


class RampSource(mu.ObjFlow):
    """A producer of `a` whose capability falls linearly, from 10 at
    t = 0.

    U10 declares a falling output as a time profile; until it does, the
    test drives `capability_expr` directly, which is the seam a profile
    will write. What matters here is only that the quantity offered
    varies **continuously**, so the limiting reagent can change identity
    without any discrete event."""

    def add_flows(self):
        self.add_flow_continuous_out(name="a", var_fed_default=0.0)
        self.flows_continuous_out[-1].capability_expr = {
            "op": "sub",
            "lhs": {"op": "const", "value": {"kind": "float", "value": 10.0}},
            "rhs": {"op": "time"},
        }


class Mixer(mu.ObjFlow):
    """One of `a` and one of `b` make one of `x`."""

    def add_flows(self):
        self.add_flow_continuous_in(name="a")
        self.add_flow_continuous_in(name="b")
        self.add_flow_continuous_out(name="x")
        self.add_rule_set(
            name="mix", rules=[{"cons": {"a": 1, "b": 1}, "prod": {"x": 1}}]
        )


class Basin(mu.ObjFlow):
    """A consumer that takes everything and integrates it, so the run
    has a continuous trajectory to be sampled along."""

    def add_flows(self):
        self.add_flow_continuous_in(name="x")
        self.add_capacity(name="basin", flow="x", capacity=1.0e6, fill_rate=1000.0)


def crossing_reaction() -> mu.System:
    system = mu.System("rule_crossing")
    system.add_component(RampSource, "A")
    system.add_component(source("b", 5.0), "B")
    system.add_component(Mixer, "R")
    system.add_component(Basin, "K")
    system.connect("A", "a", "R", "a")
    system.connect("B", "b", "R", "b")
    system.connect("R", "x", "K", "x")
    return system


def test_the_limiting_input_changes_identity_without_a_jump():
    """`min(10 - t, 5)` switches limiting reagent at t = 5: the output
    is continuous there, and only its slope changes."""
    instants = [0.0, 4.0, 4.999, 5.0, 5.001, 6.0, 9.0]
    result = crossing_reaction().simulate(t_max=9.0, samples=instants)
    values = {t: sampled(result, "K_x_fed_in", t) for t in instants}

    # Before the switch `b` limits at 5; after it, `a` does, at 10 - t.
    assert abs(values[0.0] - 5.0) < TOL
    assert abs(values[4.0] - 5.0) < TOL
    assert abs(values[6.0] - 4.0) < TOL
    assert abs(values[9.0] - 1.0) < TOL

    # Continuity at the switch. The gap across a bracket of 2e-3 is
    # exactly the 1e-3 the ramp itself travels on its half of it, so the
    # jump component is zero to a part in 1e9: what a jump would leave
    # is a gap that does not shrink with the bracket.
    assert abs(values[5.0] - 5.0) < TOL
    assert abs((values[5.001] - values[4.999]) + 1.0e-3) < TOL

    # And it is a kink, not a straight line: the slope changes from 0
    # to -1 across the crossing.
    left = (values[5.0] - values[4.0]) / 1.0
    right = (values[6.0] - values[5.0]) / 1.0
    assert abs(left) < TOL
    assert abs(right + 1.0) < TOL


def test_the_limiting_input_changes_identity_with_no_transition():
    """KTD14: the minimum keeps the trajectory continuous, so the switch
    needs no crossing transition. None is declared, and none fires."""
    result = crossing_reaction().simulate(t_max=9.0)
    switching = [event for event in result.events if 4.0 < event.time < 6.0]
    assert switching == [], f"a transition fired at the switch: {switching}"

    document = pyraichu.model_body(crossing_reaction().build_dict())
    mixer = next(c for c in document["components"] if c["name"] == "R")
    assert mixer["automata"] == []


# --- ordered selection through a mode automaton ------------------------


class Switchable(mu.ObjFlow):
    """One of `a` makes two of `x` while `boost` is fed, one otherwise."""

    def add_flows(self):
        self.add_flow_in(name="boost")
        self.add_flow_continuous_in(name="a")
        self.add_flow_continuous_out(name="x")
        self.add_rule_set(
            name="make_x",
            rules=[
                {"cond": ["boost"], "cons": {"a": 1}, "prod": {"x": 2}},
                {"cons": {"a": 1}, "prod": {"x": 1}},
            ],
        )


def switchable_system(boosting: bool) -> mu.System:
    class Trigger(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="boost", var_prod_default=boosting)

    system = mu.System("rule_guarded")
    system.add_component(Trigger, "T")
    system.add_component(source("a", 3.0), "A")
    system.add_component(Switchable, "R")
    system.add_component(Sink, "K")
    system.connect("T", "boost", "R", "boost")
    system.connect("A", "a", "R", "a")
    system.connect("R", "x", "K", "x")
    return system


def test_the_guarded_rule_runs_when_its_guard_holds():
    """`boost` fed: the guarded rule is selected and three units of `a`
    make six of `x`."""
    result = switchable_system(True).simulate(t_max=1.0)
    assert abs(settled(result, "K_x_fed_in") - 6.0) < TOL


def test_the_unguarded_rule_is_the_default_of_its_set():
    """`boost` absent: no guard holds, and the unguarded rule applies."""
    result = switchable_system(False).simulate(t_max=1.0)
    assert abs(settled(result, "K_x_fed_in") - 3.0) < TOL


def test_a_set_with_no_default_produces_nothing_while_no_guard_holds():
    """A rule set carrying no unguarded rule selects nothing when no
    guard holds, and a set that selects nothing produces nothing."""

    class GuardedOnly(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="boost")
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="make_x",
                rules=[{"cond": ["boost"], "cons": {"a": 1}, "prod": {"x": 1}}],
            )

    class Trigger(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name="boost", var_prod_default=False)

    system = mu.System("rule_no_default")
    system.add_component(Trigger, "T")
    system.add_component(source("a", 3.0), "A")
    system.add_component(GuardedOnly, "R")
    system.add_component(Sink, "K")
    system.connect("T", "boost", "R", "boost")
    system.connect("A", "a", "R", "a")
    system.connect("R", "x", "K", "x")
    result = system.simulate(t_max=1.0)
    assert settled(result, "K_x_fed_in") == 0.0
    assert settled(result, "R_a_fed_in") == 0.0


# --- a guard on a continuous quantity is a located crossing ------------


class Batcher(mu.ObjFlow):
    """Transforms `a` into `x`, but only once its own buffer of `water`
    holds at least 50 units."""

    def add_flows(self):
        self.add_flow_continuous_in(name="water")
        self.add_flow_continuous_in(name="a")
        self.add_flow_continuous_out(name="x")
        self.add_capacity(name="buffer", flow="water", capacity=100.0, fill_rate=10.0)
        self.add_rule_set(
            name="batch",
            rules=[
                {
                    "cond": [{"name": "buffer_content", "op": ">=", "value": 50.0}],
                    "cons": {"a": 1},
                    "prod": {"x": 1},
                }
            ],
        )


def batching_system(component=Batcher) -> mu.System:
    system = mu.System("rule_located_guard")
    system.add_component(source("water", 5.0), "W")
    system.add_component(source("a", 10.0), "A")
    system.add_component(component, "R")
    system.add_component(Sink, "K")
    system.connect("W", "water", "R", "water")
    system.connect("A", "a", "R", "a")
    system.connect("R", "x", "K", "x")
    return system


def test_a_guard_on_a_capacity_level_fires_at_the_crossing_instant():
    """The buffer fills at 5 a unit of time, so it reaches 50 at t = 10:
    the rule is selected there, located, not at the following step."""
    result = batching_system().simulate(t_max=20.0, samples=[9.9, 10.1])
    assert abs(fired_at(result, "batch_none_to_rule_0") - 10.0) < CROSSING_TOL
    assert sampled(result, "K_x_fed_in", 9.9) == 0.0
    assert abs(sampled(result, "K_x_fed_in", 10.1) - 10.0) < TOL


def test_a_guard_on_a_continuous_quantity_is_a_watched_transition():
    """The mode transition is declared `watched`: the intent cannot be
    inferred from the guard, so the layer declares it."""
    document = pyraichu.model_body(batching_system().build_dict())
    batcher = next(c for c in document["components"] if c["name"] == "R")
    mode = next(a for a in batcher["automata"] if a["name"] == "batch_mode")
    assert {t["distrib"] for t in mode["transitions"]} == {"watched"}


def test_a_continuous_guard_and_a_default_cross_once():
    """A set carrying both a continuously-guarded rule and a default
    crosses between them exactly once, at the threshold.

    The two selection guards are complementary comparisons, not a guard
    and its `not`: the denial of `>= 50` is the **strict** `< 50`, which
    is tightened away from the boundary. Written as a negation, both
    guards hold exactly at 50 and the mode chatters there instead of
    crossing, which the engine reports as a boundary loop."""

    class Doubling(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_capacity(
                name="buffer", flow="water", capacity=100.0, fill_rate=10.0
            )
            self.add_rule_set(
                name="batch",
                rules=[
                    {
                        "cond": [{"name": "buffer_content", "op": ">=", "value": 50.0}],
                        "cons": {"a": 1},
                        "prod": {"x": 2},
                    },
                    {"cons": {"a": 1}, "prod": {"x": 1}},
                ],
            )

    result = batching_system(Doubling).simulate(
        t_max=20.0, samples=[5.0, 9.9, 10.1, 19.0]
    )
    selections = [
        event for event in result.events if "batch_mode" in event.transition
    ]
    assert len(selections) == 1, selections
    assert abs(selections[0].time - 10.0) < CROSSING_TOL
    # The default runs below the threshold and the guarded rule above it.
    assert abs(sampled(result, "K_x_fed_in", 5.0) - 10.0) < TOL
    assert abs(sampled(result, "K_x_fed_in", 9.9) - 10.0) < TOL
    assert abs(sampled(result, "K_x_fed_in", 10.1) - 20.0) < TOL
    assert abs(sampled(result, "K_x_fed_in", 19.0) - 20.0) < TOL


def test_a_guard_mixing_a_state_and_a_level_stays_located():
    """A discrete gate composed with a continuous threshold: the rule is
    selected at the located crossing while the gate holds, and dropped
    when the gate falls."""

    class GatedBatcher(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="water")
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_capacity(
                name="buffer", flow="water", capacity=100.0, fill_rate=10.0
            )
            self.add_delay_failure_mode(name="fm", failure_time=15.0, repair_time=1.0e9)
            self.add_rule_set(
                name="batch",
                rules=[
                    {
                        "cond": [
                            {"automaton": "fm", "state": "ok"},
                            {"name": "buffer_content", "op": ">=", "value": 50.0},
                        ],
                        "cons": {"a": 1},
                        "prod": {"x": 1},
                    }
                ],
            )

    result = batching_system(GatedBatcher).simulate(
        t_max=20.0, samples=[9.9, 10.1, 14.9, 15.1]
    )
    assert abs(fired_at(result, "batch_none_to_rule_0") - 10.0) < CROSSING_TOL
    assert sampled(result, "K_x_fed_in", 9.9) == 0.0
    assert abs(sampled(result, "K_x_fed_in", 10.1) - 10.0) < TOL
    # The gate falls at t = 15 and the rule is dropped, with the level
    # still above its threshold.
    assert abs(sampled(result, "K_x_fed_in", 14.9) - 10.0) < TOL
    assert sampled(result, "K_x_fed_in", 15.1) == 0.0


# --- a contested output declares its apportionment ---------------------


class Contested(mu.ObjFlow):
    """Two rule sets making the same product, sharing its demand three
    to one."""

    def add_flows(self):
        self.add_flow_continuous_in(name="a")
        self.add_flow_continuous_in(name="b")
        self.add_flow_continuous_out(name="x")
        self.add_rule_set(
            name="from_a",
            rules=[{"cons": {"a": 1}, "prod": {"x": 1}}],
            apportionment={"x": 3.0},
        )
        self.add_rule_set(
            name="from_b",
            rules=[{"cons": {"b": 1}, "prod": {"x": 1}}],
            apportionment={"x": 1.0},
        )


def test_two_rule_sets_split_a_contested_output_in_the_declared_ratio():
    """A consumer asking 8 is served 6 by one set and 2 by the other,
    and each set draws exactly what it made."""

    class Modest(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x", var_demand_in_default=8.0)

    system = mu.System("rule_contested")
    system.add_component(source("a", 100.0), "A")
    system.add_component(source("b", 100.0), "B")
    system.add_component(Contested, "R")
    system.add_component(Modest, "K")
    system.connect("A", "a", "R", "a")
    system.connect("B", "b", "R", "b")
    system.connect("R", "x", "K", "x")
    result = system.simulate(t_max=1.0)

    assert abs(settled(result, "K_x_fed_in") - 8.0) < TOL
    assert abs(settled(result, "R_a_fed_in") - 6.0) < TOL
    assert abs(settled(result, "R_b_fed_in") - 2.0) < TOL


def test_a_contested_output_without_apportionment_is_refused():
    """muscadet has no field for it, so the layer adds one and refuses
    its absence, naming the component and the flow."""

    class Undeclared(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_in(name="b")
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="from_a", rules=[{"cons": {"a": 1}, "prod": {"x": 1}}]
            )
            self.add_rule_set(
                name="from_b", rules=[{"cons": {"b": 1}, "prod": {"x": 1}}]
            )

    system = mu.System("rule_contested_undeclared")
    system.add_component(Undeclared, "R")
    with pytest.raises(ValueError) as raised:
        system.build_dict()
    message = str(raised.value)
    assert "`R`" in message and "`x`" in message
    assert "apportionment" in message


# --- a cycle that creates matter from nothing --------------------------


def rule_cycle(external: bool) -> mu.System:
    """Two components each turning the other's product into its own,
    with a coefficient product of 2: matter out of nothing, unless a
    finite external input bounds it."""

    class Doubler(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="b")
            if external:
                self.add_flow_continuous_in(name="feed")
                cons = {"a": 1, "feed": 1}
            else:
                cons = {"a": 1}
            self.add_rule_set(name="double", rules=[{"cons": cons, "prod": {"b": 2}}])

    class Returner(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="b")
            self.add_flow_continuous_out(name="a")
            self.add_rule_set(name="back", rules=[{"cons": {"b": 1}, "prod": {"a": 1}}])

    system = mu.System("rule_cycle")
    system.add_component(Doubler, "C1")
    system.add_component(Returner, "C2")
    system.connect("C1", "b", "C2", "b")
    system.connect("C2", "a", "C1", "a")
    if external:
        system.add_component(source("feed", 4.0), "S")
        system.connect("S", "feed", "C1", "feed")
    return system


def test_a_self_feeding_rule_cycle_creating_matter_is_refused():
    """Every input of the cycle comes from inside it and the coefficient
    product exceeds 1: the loop has no bound, and is refused by name."""
    system = rule_cycle(external=False)
    with pytest.raises(ValueError) as raised:
        system.build_dict()
    message = str(raised.value)
    assert "`C1`" in message and "`C2`" in message
    assert "`b`" in message and "`a`" in message
    assert "coefficient product" in message


def test_the_same_cycle_fed_from_outside_is_accepted():
    """A finite external input bounds the same cycle, so the rule
    diagnostic lets it through: the coefficient product is unchanged and
    only the feed differs.

    What this pins is the rule diagnostic's verdict, which is what this
    layer owns. The generated document is a **continuous cycle** on top
    of that, and the engine has its own word to say about those: see the
    note in :meth:`System._refuse_unbounded_rule_cycles`."""
    document = rule_cycle(external=True).build_dict()
    assert pyraichu.model_body(document)["name"] == "rule_cycle"
    # The unfed cycle differs from it by the feed alone, and is refused.
    with pytest.raises(ValueError, match="coefficient product"):
        rule_cycle(external=False).build_dict()


# --- a loop of rate comparisons ----------------------------------------


def test_a_loop_of_rate_comparisons_is_refused():
    """Each component's rule is guarded on a rate the other's rule
    drives: the selection has no fixpoint, and the layer says so."""

    class First(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x")
            self.add_flow_continuous_out(name="b")
            self.add_rule_set(
                name="s1",
                rules=[
                    {
                        "cond": [{"name": "x", "op": ">=", "value": 5.0}],
                        "cons": {"x": 1},
                        "prod": {"b": 1},
                    }
                ],
            )

    class Second(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="b")
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="s2",
                rules=[
                    {
                        "cond": [{"name": "b", "op": ">=", "value": 3.0}],
                        "cons": {"b": 1},
                        "prod": {"x": 1},
                    }
                ],
            )

    system = mu.System("rule_rate_loop")
    system.add_component(First, "C1")
    system.add_component(Second, "C2")
    system.connect("C1", "b", "C2", "b")
    system.connect("C2", "x", "C1", "x")
    with pytest.raises(ValueError) as raised:
        system.build_dict()
    message = str(raised.value)
    assert "`C1`" in message and "`C2`" in message
    assert "`x`" in message or "`b`" in message
    assert "rate" in message


def test_a_guard_on_a_level_is_not_a_rate_comparison():
    """A threshold on an integrated level breaks the loop a threshold on
    a rate would close: the same shape, guarded on a capacity, builds."""
    document = batching_system().build_dict()
    assert pyraichu.model_body(document)["name"] == "rule_located_guard"


# --- declaration-time refusals -----------------------------------------


def test_two_unguarded_rules_are_refused():
    """A set with two defaults leaves the selected rule undefined."""

    class TwoDefaults(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="make_x",
                rules=[
                    {"cons": {"a": 1}, "prod": {"x": 1}},
                    {"cons": {"a": 2}, "prod": {"x": 1}},
                ],
            )

    with pytest.raises(ValueError, match="one unguarded rule"):
        TwoDefaults("R")


def test_a_consumed_flow_must_be_a_continuous_input():
    """A rule consumes inputs and produces outputs: naming an output on
    the consumed side is refused, naming the component and the flow."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="make_x", rules=[{"cons": {"x": 1}, "prod": {"x": 1}}]
            )

    with pytest.raises(ValueError) as raised:
        Wrong("R")
    assert "`R`" in str(raised.value) and "`x`" in str(raised.value)


def test_a_guard_on_an_unknown_name_is_refused():
    """A guard operand naming nothing the component carries is refused
    rather than compiled into a dangling read."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="make_x",
                rules=[
                    {
                        "cond": [{"name": "nowhere", "op": ">=", "value": 1.0}],
                        "cons": {"a": 1},
                        "prod": {"x": 1},
                    }
                ],
            )

    with pytest.raises(ValueError) as raised:
        Wrong("R")
    assert "`R`" in str(raised.value) and "`nowhere`" in str(raised.value)


def test_an_equality_on_a_continuous_quantity_is_refused():
    """A crossing cannot be located on an equality: the guard would fire
    only if a float landed exactly on the threshold."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_rule_set(
                name="make_x",
                rules=[
                    {
                        "cond": [{"name": "a", "op": "==", "value": 1.0}],
                        "cons": {"a": 1},
                        "prod": {"x": 1},
                    }
                ],
            )

    with pytest.raises(ValueError, match="ordering comparison"):
        Wrong("R")


# --- the generated document --------------------------------------------


def test_rule_steps_join_the_evaluation_order_exactly_once():
    """The two scales and the produced quantity are swept in their band:
    the capability scale before the outputs publish, the demand scale
    after the outputs are asked and before the inputs ask."""
    document = reaction(8.0, 1.0).build_dict()
    order = [
        (step["component"], step["attribute"])
        for step in pyraichu.model_body(document)["evaluation_order"]
    ]
    assert len(order) == len(set(order))
    assert order.index(("R", "make_x_capability_scale")) < order.index(
        ("R", "x_capability_out")
    )
    assert order.index(("R", "x_demand_out")) < order.index(("R", "make_x_scale"))
    assert order.index(("R", "make_x_scale")) < order.index(("R", "a_demand_in"))
    assert order.index(("R", "make_x_scale")) < order.index(("R", "x_produced_out"))
    assert order.index(("R", "x_produced_out")) < order.index(("R", "x_alloc"))
    pyraichu.load_model(document)


def test_rule_scales_are_indicators():
    """A scale is the quantity a rule set runs at: generated and never
    observable otherwise."""
    document = reaction(8.0, 1.0).build_dict()
    names = {i["name"] for i in pyraichu.model_body(document)["indicators"]}
    assert "R_make_x_scale" in names
    assert "R_make_x_capability_scale" in names
    assert "R_x_produced_out" in names


def test_a_system_without_a_rule_set_carries_no_rule_material():
    """Regression: a model declaring no rule is exactly what it was, so
    the whole continuous suite is untouched by this unit."""

    class Relay(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="power", var_demand_in_default=5.0)
            self.add_flow_continuous_out(name="power", var_fed_default=5.0)

    system = mu.System("no_rules")
    system.add_component(source("power", 10.0), "S")
    system.add_component(Relay, "M")
    system.connect("S", "power", "M", "power")
    document = pyraichu.model_body(system.build_dict())
    for component in document["components"]:
        assert component["automata"] == []
        targets = {equation["target"] for equation in component["equations"]}
        assert not any(
            target.endswith(("_scale", "_produced_out")) for target in targets
        )
