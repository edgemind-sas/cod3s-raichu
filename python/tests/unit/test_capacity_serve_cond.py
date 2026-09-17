"""A volume whose DISCHARGE is commanded, and the ceiling a mode can clamp.

muscadet's `CapacityContinuous` takes two keys beyond the transit: a
``control`` port and a ``serve_cond`` naming it, which together command what
the volume releases. Neither had a counterpart here, so the reader refused
`serve_cond` by name and a commanded tank never reached the engine at all.

What this module pins, and the order is the order of the work:

- **the command is carried**, with the two parallel matrices muscadet lifts a
  negation and a comparison out of its operands into. One condition, two
  spellings, and a contradiction between them refused rather than resolved;
- **the gate bites on BOTH branches**. muscadet's `serve_ceiling` answers zero
  outright when the command does not hold, so `serve_limit` is zero whether
  the volume serves from stock or merely transits. A gate on the stocked
  branch alone leaves a commanded buffer standing down passing its inlet
  straight through to its outlet -- measured, on the empty charging volume
  below: 0.5 crossed instead of nothing;
- **what it asks upstream is capped by what it can release**, which is
  muscadet's `demand_claim` and the same arithmetic. Without it a volume told
  to stop serving goes on drawing its consumer's demand and accumulates it;
- **the ceiling is a VARIABLE**, one per held flow, under the name muscadet
  gives it. That is what a failure mode clamps to throttle a discharge, and a
  constant folded into the capability leaves such a mode with nothing to
  reach;
- **the pre-run loop diagnostic sees the command** exactly as it sees a rule
  guard, because the command is compiled into an automaton and not inlined in
  the served quantity. A band silences it, and a band is declarable here where
  muscadet has no field for one.

Oracle-free, like the rest of this directory. That the two engines answer a
commanded volume alike, instant by instant, is the validation suite's
(`test_muscadet_declaration_vocabulary.py`, the `commanded_tank` model).
"""

import json

import pytest

import pyraichu
import pyraichu.declare as declare
import pyraichu.muscadet as mu
import pyraichu.muscadet_engine as engine
from conftest import CROSSING_TOL, TOL, sampled

#: The running example, in the units the feature states it in: a source of
#: two, a volume of a hundred, a consumer asking for one.
SOURCE_RATE = 2.0
VOLUME = 100.0
HELD = 10.0
DEMAND = 1.0

#: The port of command, and therefore the name the condition's operand
#: carries: muscadet declares the two together and resolves the second
#: against the first.
COMMAND = "discharge"


def source_class(rate: float = SOURCE_RATE):
    """A producer of `rate`, which is more than the consumer asks for
    unless a case says otherwise."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="q", var_fed_default=rate)

    return Source


class Sink(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_in(name="q", var_demand_in_default=DEMAND)


def command_class(held: bool):
    """A discrete output holding the command at one value."""

    class Command(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_out(name=COMMAND, var_prod_default=held)

    return Command


def tank_class(**capacity):
    """A both-sided volume over `q`, with a port of command beside it.

    The port is declared BEFORE the capacity, as `CapacityContinuous` does
    and for the reason it states: a `serve_cond` operand naming it is
    resolved as the capacity is declared.
    """
    settings = dict(
        name="tank",
        flow="q",
        capacity=VOLUME,
        side="out",
        transmits=True,
        content_init={"q": HELD},
    )
    settings.update(capacity)

    class Tank(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="q")
            self.add_flow_continuous_out(name="q")
            self.add_flow_in(name=COMMAND)
            self.add_capacity(**settings)

    return Tank


def tank_system(
    held: bool = True, source_rate: float = SOURCE_RATE, **capacity
) -> mu.System:
    """Source, volume, consumer, and a command held at one value."""
    system = mu.System("commanded_tank")
    system.add_component(source_class(source_rate), "SRC")
    system.add_component(tank_class(**capacity), "CAP")
    system.add_component(Sink, "SINK")
    system.add_component(command_class(held), "CMD")
    system.connect("SRC", "q", "CAP", "q")
    system.connect("CAP", "q", "SINK", "q")
    system.connect("CMD", COMMAND, "CAP", COMMAND)
    return system


#: Where the trajectories are read. Never t = 0: the gate is an automaton
#: entered by the initial fixpoint, as a rule set's mode is, so at the
#: initial instant it is still in the location it was declared in. The
#: parity bench's own schedule starts at 1 for the same reason.
INSTANTS = (1.0, 3.0, 5.0)


def served(result) -> list[float]:
    return [sampled(result, "SINK_q_fed_in", t) for t in INSTANTS]


def drawn(result) -> list[float]:
    return [sampled(result, "CAP_q_fed_in", t) for t in INSTANTS]


# --- 1. the command decides what the volume releases ---------------------


def test_a_commanded_volume_serves_while_the_command_holds():
    result = tank_system(held=True, serve_cond=[[COMMAND]]).simulate(
        t_max=6.0, samples=list(INSTANTS)
    )
    assert served(result) == pytest.approx([DEMAND] * len(INSTANTS))


def test_the_same_volume_serves_nothing_once_the_command_falls():
    """The other half, and the one that says the condition is read: the
    model differs by the value the command holds and by nothing else."""
    result = tank_system(held=False, serve_cond=[[COMMAND]]).simulate(
        t_max=6.0, samples=list(INSTANTS)
    )
    assert served(result) == pytest.approx([0.0] * len(INSTANTS))


def test_a_volume_no_command_gates_serves_as_it_always_did():
    """The control: an empty `serve_cond` is an UNCOMMANDED volume, which
    is every volume declared before the key existed."""
    for declared in (None, []):
        result = tank_system(held=False, serve_cond=declared).simulate(
            t_max=6.0, samples=list(INSTANTS)
        )
        assert served(result) == pytest.approx([DEMAND] * len(INSTANTS)), declared


# --- 2. the gate bites on the branch that TRANSITS too -------------------


def test_an_empty_volume_still_charging_delivers_nothing_while_commanded_down():
    """The measurement the branch question turns on.

    An empty volume claiming a `fill_rate` goes on being fed while its
    discharge stands down, so something IS crossing it and its stock is nil.
    A gate on the stocked branch alone would let that transit through: it did,
    and the consumer read 0.5 where PyCATSHOO read nothing.
    """
    result = tank_system(
        held=False,
        serve_cond=[[COMMAND]],
        content_init={"q": 0.0},
        fill_rate=0.5,
    ).simulate(t_max=6.0, samples=list(INSTANTS))

    assert drawn(result) == pytest.approx([0.5] * len(INSTANTS)), (
        "the volume goes on charging: its claim is not what the command gates"
    )
    assert served(result) == pytest.approx([0.0] * len(INSTANTS))


def test_the_gate_wraps_the_ceiling_above_the_bound_and_not_inside_it():
    """The same claim read off the emitted expression, so it survives a
    model whose numbers happen to agree.

    The bound branches on the volume's own automaton; the command wraps the
    ceiling that branch is written over, so both arms of it read the gate.
    """
    built = tank_system(serve_cond=[[COMMAND]]).build_dict()["model"]
    capability = next(
        equation["expr"]
        for component in built["components"]
        if component["name"] == "CAP"
        for equation in component["equations"]
        if equation["target"] == "q_capability_out"
    )
    # `q_capability_out` is the bound itself here: the component declares no
    # production of its own, so the volume REPLACES the rate.
    assert capability["cond"]["state"]["automaton"] == "tank_bounds"
    for arm in ("then", "otherwise"):
        rendered = json.dumps(capability[arm])
        assert "tank_serve" in rendered, arm
        assert "tank_serve_rate_q" in rendered, arm


# --- 3. and what it asks upstream is capped by what it can release -------


def test_a_volume_told_to_stop_serving_stops_drawing_for_the_transit():
    """muscadet's `demand_claim`, and the rule it states: a volume does not
    fill out of a demand it cannot honour.

    Without the cap the volume goes on asking its consumer's demand and
    accumulates every unit of it, which is the divergence this model was
    measured on: the two engines agreed on what reached the consumer and
    parted on what the volume drew behind it.
    """
    result = tank_system(held=False, serve_cond=[[COMMAND]]).simulate(
        t_max=6.0, samples=list(INSTANTS)
    )
    assert drawn(result) == pytest.approx([0.0] * len(INSTANTS))


def test_a_declared_ceiling_caps_the_transit_a_volume_asks_for():
    """The same cap, with no command in sight: it is the ceiling's, and the
    command only takes it to zero.

    A pure buffer -- `fill_rate` at its default 0, "it never stocks up" --
    becomes an accumulator the moment a ceiling is declared, unless what it
    asks for the transit is capped by what it can release.
    """
    result = tank_system(serve_rate=0.25).simulate(t_max=6.0, samples=list(INSTANTS))
    assert served(result) == pytest.approx([0.25] * len(INSTANTS))
    assert drawn(result) == pytest.approx([0.25] * len(INSTANTS)), (
        "what enters equals what leaves: the buffer stocks nothing"
    )


def test_the_claim_a_volume_makes_for_itself_is_not_capped_by_the_ceiling():
    """The other side of it, and muscadet's arithmetic term for term: the
    ceiling bounds the TRANSIT, the `fill_rate` is added on top of it."""
    result = tank_system(serve_rate=0.25, fill_rate=0.5).simulate(
        t_max=6.0, samples=list(INSTANTS)
    )
    assert served(result) == pytest.approx([0.25] * len(INSTANTS))
    assert drawn(result) == pytest.approx([0.75] * len(INSTANTS))


# --- 4. the ceiling is a variable a mode can clamp -----------------------
#
# Written as a DOCUMENT rather than through the authoring layer: a failure
# mode reaching an attribute by name is a standalone mode, the second shape a
# component declaration takes, and the authoring layer declares modes on the
# components that own them. `test_declare_standalone_mode` is where that shape
# is pinned; here it is only the vehicle.

THROTTLED = 0.5


def commanded_document(serve_rate: float, throttle_at: float) -> dict:
    """A source, a volume with a ceiling, a consumer, and a mode clamping
    that ceiling by name at t = 5."""

    def component(name, flows, capacities=()):
        return {
            "name": name,
            "cls": "ObjFlow",
            "flows": list(flows),
            "capacities": list(capacities),
            "rules": [],
            "transfers": [],
            "automata": [],
            "failure_modes": [],
        }

    def continuous_in(demand=0.0):
        return {
            "name": "q",
            "cls": "FlowContinuousIn",
            "var_type": "float",
            "var_fed_default": 0.0,
            "var_demand_default": demand,
        }

    def continuous_out(rate=0.0):
        return {
            "name": "q",
            "cls": "FlowContinuousOut",
            "var_fed_default": rate,
            "var_demand_in_default": 0.0,
            "allocation": "proportional",
            "var_prod_cond_inner_mode": "or",
        }

    tank = component(
        "CAP",
        [continuous_in(), continuous_out()],
        [
            {
                "name": "tank",
                "capacity": VOLUME,
                "content_init": {"q": VOLUME / 2},
                "fill_rate": 0.0,
                "side": "out",
                "transmits": True,
                "serve_rate": serve_rate,
                "flows": [
                    {"name": "q", "weight": 1.0, "side": "out", "cls": "CapacityFlow"}
                ],
            }
        ],
    )
    mode = {
        "name": "CAP__throttle",
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": "throttle",
        "targets": ["CAP"],
        "target_name": "CAP",
        "failure_effects": {"tank_serve_rate_q": throttle_at},
        "failure_param_name": ["ttf"],
        "failure_param": [5],
        "repair_param_name": ["ttr"],
        "repair_param": [1000],
    }
    components = [
        component("SRC", [continuous_out(rate=SOURCE_RATE)]),
        tank,
        component("SINK", [continuous_in(demand=2 * SOURCE_RATE)]),
    ]
    return {
        "version": "1.0.0",
        "name": "throttled",
        "generated_indicators": True,
        "components": {entry["name"]: entry for entry in components}
        | {mode["name"]: mode},
        "connections": [
            {
                "source": "SRC",
                "source_box": "q_out",
                "target": "CAP",
                "target_box": "q_in",
            },
            {
                "source": "CAP",
                "source_box": "q_out",
                "target": "SINK",
                "target_box": "q_in",
            },
        ],
    }


def test_a_failure_mode_clamps_the_ceiling_by_the_name_muscadet_gives_it():
    """The criterion: the ceiling is a public variable, not a constant
    frozen at generation.

    The mode reaches `tank_serve_rate_q` and nothing else; the consumer asks
    for more than the volume may release throughout, so what it receives IS
    the ceiling, before and after.
    """
    document = commanded_document(serve_rate=2.0, throttle_at=THROTTLED)
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=12.0, samples=[1.0, 4.0, 8.0])
    trajectory = dict(result.samples["SINK_q_fed_in"])

    assert trajectory[1.0] == pytest.approx(2.0)
    assert trajectory[4.0] == pytest.approx(2.0)
    assert trajectory[8.0] == pytest.approx(THROTTLED), (
        "the mode fired at 5 and the discharge follows the variable it wrote"
    )


def test_the_mode_would_be_refused_if_the_ceiling_were_not_published():
    """The premise, stated rather than assumed: the reader refuses an effect
    on an attribute the target does not carry, so the test above would fail
    at declaration and not on its numbers if the variable went away.

    Checked on a name that really is absent, so it says what the refusal
    looks like rather than trusting that it exists.
    """
    document = commanded_document(serve_rate=2.0, throttle_at=THROTTLED)
    document["components"]["CAP__throttle"]["failure_effects"] = {
        "tank_serve_rate_absent": THROTTLED
    }
    with pytest.raises(declare.ComponentSpecError, match="tank_serve_rate_absent"):
        declare.build_document(document)


def ceiling_indicator(component: str = "CAP") -> dict:
    """An indicator on the ceiling itself, named as muscadet names the
    variable, which is how a study declares it."""
    return {
        "name": f"{component}_tank_serve_rate_q",
        "label": f"{component}_tank_serve_rate_q",
        "measure": "value",
        "stats": ["mean"],
        "component": component,
        "operator": "==",
        "var": "tank_serve_rate_q",
        "kind": "PycVarIndicator",
    }


def test_the_ceiling_is_observable_and_the_clamp_shows_in_its_trajectory():
    """The other half of a public ceiling: what a mode may clamp, a study may
    WATCH.

    The seam refused this while the ceiling was a constant, and said so in as
    many words -- "no variable carries it" -- which was true of the code it
    was written against and false of this one. Measured here on the same model
    as the clamp above, so the two answers are read side by side: the ceiling
    steps from 2 to 0.5 at the mode's date, and the consumer steps with it.
    """
    document = commanded_document(serve_rate=2.0, throttle_at=THROTTLED)
    document["indicators"] = [ceiling_indicator()]
    result = pyraichu.simulate(
        engine.build_model(document), t_max=12.0, samples=[1.0, 4.0, 8.0]
    )
    ceiling = dict(result.samples["CAP_tank_serve_rate_q"])

    assert ceiling[1.0] == pytest.approx(2.0)
    assert ceiling[4.0] == pytest.approx(2.0)
    assert ceiling[8.0] == pytest.approx(THROTTLED)
    assert ceiling == pytest.approx(dict(result.samples["SINK_q_fed_in"])), (
        "the consumer asks for more than the volume may release throughout, so "
        "what it receives IS the ceiling, before the clamp and after"
    )


def test_a_volume_declaring_no_ceiling_is_observed_at_the_unbounded_sentinel():
    """The untouched case, and the reason the observation answers on every
    volume rather than on the ones that named a number.

    A document is JSON and carries no literal for an infinity, so an unbounded
    volume publishes :data:`pyraichu.muscadet.UNBOUNDED_SERVICE` rather than
    nothing at all. An observer reading it gets that sentinel, not a missing
    attribute, and the volume serves whatever is asked of it meanwhile.
    """
    document = commanded_document(serve_rate=2.0, throttle_at=THROTTLED)
    del document["components"]["CAP"]["capacities"][0]["serve_rate"]
    del document["components"]["CAP__throttle"]
    document["indicators"] = [ceiling_indicator()]
    result = pyraichu.simulate(
        engine.build_model(document), t_max=12.0, samples=[1.0, 8.0]
    )
    ceiling = dict(result.samples["CAP_tank_serve_rate_q"])

    assert ceiling[1.0] == pytest.approx(mu.UNBOUNDED_SERVICE)
    assert ceiling[8.0] == pytest.approx(mu.UNBOUNDED_SERVICE)
    assert dict(result.samples["SINK_q_fed_in"])[8.0] == pytest.approx(
        2 * SOURCE_RATE
    ), (
        "unbounded and still stocked, the volume serves the consumer's whole "
        "demand: what the source delivers, and the rest off its own stock"
    )


# --- 5. the pre-run loop diagnostic sees it --------------------------


def reserve_floor(
    fill_rate: float = 0.0, source_rate: float = SOURCE_RATE, **operand
) -> mu.System:
    """The tank commanded on its OWN level: a reserve floor, which is the
    montage a switching loop is made of.

    The command decides the discharge, the discharge moves the level, and the
    level is what the command reads. muscadet prescribes thresholding a level
    exactly here -- an integrated state breaks a loop where a rate does not --
    so this is the shape a modeller writes, not a pathological one.
    """
    condition = {"name": "tank_content", "op": ">=", "value": 20.0}
    condition.update(operand)
    return tank_system(
        serve_cond=[[condition]],
        content_init={"q": 30.0},
        fill_rate=fill_rate,
        source_rate=source_rate,
    )


def loops_of(system: mu.System) -> list[dict]:
    return pyraichu.switching_loops(
        pyraichu.load_model(pyraichu.expand_model(system.build_dict()["model"]))
    )


def test_a_command_switching_on_a_single_threshold_is_reported():
    """No `simulate` call: the finding comes out of the compiled tables.

    It costs the diagnostic not one line: the command is compiled into an
    automaton, so the cycle it closes runs automaton to variable to automaton
    like any other. Inlined in the served quantity it would close a cycle
    among VARIABLES alone, which is an algebraic loop and is reported by
    nobody.
    """
    found = loops_of(reserve_floor())
    assert len(found) == 1, found
    assert found[0]["bandless"] == ["CAP.tank_serve"], found[0]


def test_the_same_command_with_a_band_is_silent():
    """The control, and the RAICHU extension the band is: muscadet has no
    field for one, a condition of its own being entered and left at one
    threshold. The dependency is unchanged and still a cycle; what changed is
    that crossing the band costs physical time."""
    assert loops_of(reserve_floor(release=5.0)) == []


def test_the_band_is_what_the_volume_actually_holds_between():
    """And it is a band at run time, not only in the tables: the volume
    serves down to the release edge, stops, refills to the entry edge and
    starts again, so the cycle has the period the two levels give it and not
    the one numerical hysteresis would.

    Supplied half a unit where it is asked for one, so the stock actually
    moves, and claiming that half for itself so that a volume standing down
    goes on charging -- shut, it asks for nothing on account of the transit,
    which is exactly the cap the criterion above is about.
    """
    system = reserve_floor(release=5.0, fill_rate=0.5, source_rate=0.5)
    result = system.simulate(
        t_max=140.0,
        samples=[1.0, 45.0, 55.0, 65.0, 90.0],
        max_transition_firings=200,
    )
    at = {t: sampled(result, "SINK_q_fed_in", t) for t in (1.0, 45.0, 55.0, 65.0, 90.0)}
    level = {t: sampled(result, "CAP_tank_content", t) for t in (45.0, 55.0, 65.0, 90.0)}

    # Draining from 30 at half a unit: the release edge is reached at 50.
    assert at[1.0] == pytest.approx(DEMAND) and at[45.0] == pytest.approx(DEMAND)
    assert level[45.0] == pytest.approx(7.5, abs=CROSSING_TOL)
    assert at[55.0] == pytest.approx(0.0), "shut at the release edge, not at 20"
    # Charging from 5 at half a unit: the entry edge is reached at 80.
    assert level[55.0] == pytest.approx(7.5, abs=CROSSING_TOL)
    assert at[65.0] == pytest.approx(0.0), "still shut inside the band"
    assert level[65.0] == pytest.approx(12.5, abs=CROSSING_TOL)
    assert at[90.0] == pytest.approx(DEMAND), "and open again above the entry edge"
    assert level[90.0] == pytest.approx(15.0, abs=CROSSING_TOL)


def test_a_command_comparing_a_rate_it_drives_is_refused():
    """The diagnostic that costs code: the pre-run loop detector reads the
    compiled tables and is free, but the RATE comparison refusal enumerates
    declarations and had to be taught about capacities.

    A command comparing a rate its own discharge moves has no more of a
    fixpoint than a rule selected on one, and muscadet names this exact case
    as a blind spot of its own detectors.
    """
    with pytest.raises(ValueError) as refused:
        tank_system(
            serve_cond=[[{"name": "q", "port": "out", "op": ">", "value": 0.5}]]
        ).build_dict()
    message = str(refused.value)
    assert "capacity `tank` of `CAP`" in message, message
    assert "loop of rate comparisons" in message, message


def test_a_command_comparing_a_level_closes_no_such_loop():
    """The control: an integrated level is carried across the sweep, which
    is why the two kinds of comparison are told apart rather than counted
    together. The same model with the same topology builds."""
    reserve_floor(release=5.0).build_dict()


# --- 6. the reader carries the key, and its three matrices ---------------


def capacity_spec(**extra) -> dict:
    """A muscadet-shaped capacity document, as `system_spec` writes one for
    a `CapacityContinuous` carrying a `control` port."""
    capacity = {
        "name": "tank",
        "capacity": VOLUME,
        "content_init": {"q": HELD},
        "fill_rate": 0.0,
        "side": "out",
        "transmits": True,
        "flows": [{"name": "q", "weight": 1.0, "side": "out", "cls": "CapacityFlow"}],
    }
    capacity.update(extra)
    return {
        "name": "CAP",
        "cls": "ObjFlow",
        "flows": [
            {
                "name": COMMAND,
                "cls": "FlowIn",
                "var_type": "bool",
                "var_fed_default": False,
                "var_in_default": False,
                "var_available_in_default": True,
                "logic": "and",
            },
            {
                "name": "q",
                "cls": "FlowContinuousIn",
                "var_type": "float",
                "var_fed_default": 0.0,
                "var_in_default": 0.0,
            },
            {
                "name": "q",
                "cls": "FlowContinuousOut",
                "var_fed_default": 0.0,
                "var_demand_in_default": 0.0,
                "allocation": "proportional",
                "var_prod_cond_inner_mode": "or",
            },
        ],
        "capacities": [capacity],
        "rules": [],
        "transfers": [],
        "automata": [],
        "failure_modes": [],
    }


def read(**extra):
    """The capacity the reader builds out of that document."""
    spec = capacity_spec(**extra)
    declare.check_spec(spec)
    return declare.build_component(mu.System("read"), spec).capacities[0]


def test_the_reader_carries_the_command_muscadet_exports():
    """The shape a read-back writes: the operands alone, muscadet having
    rebuilt them from its own matrices, plus the inner mode."""
    capacity = read(
        serve_cond=[[{"name": COMMAND, "port": "in"}]], serve_cond_inner_mode="or"
    )
    assert capacity.commanded
    assert [[operand.name for operand in group] for group in capacity.serve_cond] == [
        [COMMAND]
    ]
    assert capacity.serve_cond_inner_mode == "or"


def test_the_reader_carries_the_two_parallel_matrices():
    """The other spelling: a document written against muscadet's `Capacity`
    model carries the negation and the comparison beside the operands rather
    than inside them, and the two declare one condition."""
    capacity = read(
        serve_cond=[[{"name": "tank_content"}, {"name": COMMAND, "port": "in"}]],
        serve_cond_negate=[[False, True]],
        serve_cond_compare=[[{"op": ">=", "value": 20.0}, None]],
    )
    level, command = capacity.serve_cond[0]
    assert (level.op, level.value) == (">=", 20.0)
    assert command.negate is True


def test_a_matrix_contradicting_its_own_operand_is_refused():
    """One condition has one comparison. Resolved either way, which of the
    two holds would depend on the order they were read in."""
    with pytest.raises(ValueError, match="one condition has one comparison"):
        read(
            serve_cond=[[{"name": "tank_content", "op": ">=", "value": 20.0}]],
            serve_cond_compare=[[{"op": ">=", "value": 40.0}]],
        )


def test_the_inner_mode_swaps_both_levels_as_muscadet_does():
    """`"or"` is a conjunction of disjunctions and `"and"` the other way
    round, exactly as a production condition's own inner mode. Read off the
    emitted guard, since that is what the two spellings decide."""
    guards = {}
    for mode in ("or", "and"):
        system = tank_system(
            serve_cond=[[COMMAND], [{"name": "tank_content", "op": ">=", "value": 5.0}]],
            serve_cond_inner_mode=mode,
        )
        automaton = next(
            entry
            for component in system.build_dict()["model"]["components"]
            if component["name"] == "CAP"
            for entry in component["automata"]
            if entry["name"] == "tank_serve"
        )
        opens = next(t for t in automaton["transitions"] if t["source"] == "withheld")
        guards[mode] = opens["guard"]["bool_op"]
    assert guards == {"or": "and", "and": "or"}


def test_an_unknown_inner_mode_is_refused():
    with pytest.raises(ValueError, match="serve_cond_inner_mode"):
        read(serve_cond=[[COMMAND]], serve_cond_inner_mode="xor")


def test_the_capacity_vocabulary_refuses_none_of_the_four_keys():
    """The entry the work was opened on: four keys `uncarried`, so a
    `CapacityContinuous` whose discharge is commanded was refused by name
    before it reached the engine."""
    _, vocabulary = declare.PLAIN_SECTIONS["capacities"]
    four = {
        "serve_cond",
        "serve_cond_negate",
        "serve_cond_compare",
        "serve_cond_inner_mode",
    }
    assert not four & set(vocabulary.uncarried)
    assert four <= set(vocabulary.carried)


# --- 7. and the declaration is checked -----------------------------------


def test_a_command_naming_nothing_is_refused_at_declaration():
    """A misspelling is refused here rather than reaching the engine as a
    dangling read, which is what a rule guard gets too."""
    with pytest.raises(ValueError, match="dischrage"):
        tank_class(serve_cond=[["dischrage"]])("CAP").add_flows()


def test_a_refused_command_leaves_the_component_as_it_was():
    """The declaration is taken back out whole: a capacity half-recorded
    would carry the volume without the command that qualifies it, and the
    command is resolved AFTER the volume is recorded so that a reserve floor
    can name it."""
    component = tank_class()("CAP")
    component.add_flow_continuous_in(name="spare")
    component.add_flow_continuous_out(name="spare")
    with pytest.raises(ValueError):
        component.add_capacity(
            name="other", flow="spare", capacity=VOLUME, serve_cond=[["nowhere"]]
        )
    assert [capacity.name for capacity in component.capacities] == ["tank"]


def test_an_equality_on_a_level_is_refused_as_it_is_in_a_rule_guard():
    """The gate transition is watched, and a crossing is located on an
    ordering comparison: an equality has no side to be located from."""
    with pytest.raises(ValueError, match="ordering comparison"):
        tank_class(
            serve_cond=[[{"name": "tank_content", "op": "==", "value": 20.0}]]
        )("CAP").add_flows()


def test_a_command_may_read_the_volume_it_commands():
    """A reserve floor names the capacity being declared, so the command is
    resolved once the volume is recorded. muscadet cannot state this at all
    on `CapacityContinuous`: reading a level needs a measurement channel
    there, and that class declares none."""
    component = tank_class(
        serve_cond=[[{"name": "tank_content", "op": ">=", "value": 20.0}]]
    )("CAP")
    assert component.capacities[0].serve_cond[0][0].name == "tank_content"


def test_the_gate_is_a_watched_crossing_when_it_reads_a_level():
    """A level moves inside an integration step and announces nothing, so
    the transition carrying its threshold is watched: a reserve floor
    nothing watched would be crossed late by up to one step.

    A command on a boolean port moves only at an event, and its transition
    is instantaneous. The distinction is the rule mode's own."""
    kinds = {}
    for name, system in (
        ("level", reserve_floor()),
        ("port", tank_system(serve_cond=[[COMMAND]])),
    ):
        automaton = next(
            entry
            for component in system.build_dict()["model"]["components"]
            if component["name"] == "CAP"
            for entry in component["automata"]
            if entry["name"] == "tank_serve"
        )
        kinds[name] = {t["distrib"] for t in automaton["transitions"]}
    assert kinds == {"level": {"watched"}, "port": {"inst"}}


def test_the_gate_is_published_as_an_automaton_a_rule_may_read():
    """The gate is a location like any other, so a rule set of the same
    component may be selected on it. That falls out of declaring it rather
    than inlining it, and is worth pinning as a property."""
    component = tank_class(serve_cond=[[COMMAND]])("CAP")
    assert component._declared_automata()["tank_serve"] == ["serving", "withheld"]


def test_an_empty_group_in_a_command_is_refused():
    with pytest.raises(ValueError, match="empty group"):
        tank_class(serve_cond=[[]])("CAP").add_flows()


def test_a_command_that_is_neither_a_name_nor_a_list_is_refused():
    with pytest.raises(ValueError, match="neither a name"):
        tank_class(serve_cond=7)("CAP").add_flows()


def test_the_short_forms_read_as_they_do_in_a_rule_guard():
    """A bare name is one operand in one group, and a flat list is one group
    per element -- two clauses of one operand, which is what the layer
    underneath reads a flat production condition as too."""
    single = tank_class(serve_cond=COMMAND)("CAP").capacities[0]
    assert [[o.name for o in g] for g in single.serve_cond] == [[COMMAND]]

    flat = tank_class(
        serve_cond=[COMMAND, {"name": "tank_content", "op": ">=", "value": 5.0}]
    )("CAP").capacities[0]
    assert [[o.name for o in g] for g in flat.serve_cond] == [
        [COMMAND],
        ["tank_content"],
    ]


def reservoir_system(held: bool) -> mu.System:
    """A volume with no way IN, commanded, and a consumer behind it.

    A reservoir: no through-path whatever it declares, so what it delivers is
    its own declared out-flow rate bounded by what it holds. The shape exists
    here to say that the command is on the CEILING and therefore reaches a
    volume with no transit at all.
    """

    class Reservoir(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="q", var_fed_default=SOURCE_RATE)
            self.add_flow_in(name=COMMAND)
            self.add_capacity(
                name="tank",
                flow="q",
                capacity=VOLUME,
                side="out",
                content_init={"q": HELD},
                serve_cond=[[COMMAND]],
            )

    system = mu.System("commanded_reservoir")
    system.add_component(Reservoir, "CAP")
    system.add_component(Sink, "SINK")
    system.add_component(command_class(held), "CMD")
    system.connect("CAP", "q", "SINK", "q")
    system.connect("CMD", COMMAND, "CAP", COMMAND)
    return system


def test_a_volume_with_no_through_path_is_commanded_too():
    """The gate sits on the ceiling, which is what a store serves out of as
    much as what a buffer passes on. `transmits` says whether the empty
    branch exists; the command says whether either branch delivers."""
    assert served(
        reservoir_system(held=True).simulate(t_max=6.0, samples=list(INSTANTS))
    ) == pytest.approx([DEMAND] * len(INSTANTS))
    assert served(
        reservoir_system(held=False).simulate(t_max=6.0, samples=list(INSTANTS))
    ) == pytest.approx([0.0] * len(INSTANTS))


def test_an_unbounded_ceiling_is_the_magnitude_and_not_an_infinity():
    """A document has no literal for an infinity, so a volume that declares
    no ceiling publishes `UNBOUNDED_SERVICE`. Worth pinning now that the
    quantity is a variable somebody may read off a run."""
    built = tank_class()("CAP")._build(set(), set())
    ceiling = next(
        entry for entry in built["attributes"] if entry["name"] == "tank_serve_rate_q"
    )
    assert ceiling["init"]["value"] == mu.UNBOUNDED_SERVICE
    assert TOL < mu.UNBOUNDED_SERVICE  # a magnitude, never compared for equality
