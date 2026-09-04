"""ObjCtrl: the controller ported from muscadet into the plugin layer.

What the port buys, and why it is a lot of machinery for one idea: every
automaton the muscadet authoring layer derives comes from a declaration that
needs one, and a mode transition's guard reads only the target rule, never
the source state, so a mode holds no memory. A guard therefore switches back
the instant its condition stops holding. Measured consequence: the
heated-tank regulation (pumps below 6, valve above 8) does not port, the
level freezing at 6 and never cycling.

:class:`~pyraichu.plugins.controller.CtrlBand` is the missing capability, a
declarable automaton with continuous guards and two-threshold memory, and
the heated-tank test below is the measurement that it is: the benchmark's
two-hour cycle, reproduced to the last digit the event location carries.

Every date asserted here is an **analytic crossing instant**, never the
declared nature of a transition. A structural assertion (`distrib` is
`watched`) passes on a model that still never fires, which is exactly the
0.10.1 defect: a guard reading a level through a measurement link was
emitted `inst`, and the crossing was never located.
"""

import json
import math

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, TOL
from pyraichu.plugins.muscadet import MuscadetPlugin

# --- shared shapes -----------------------------------------------------

#: A rate no failure law will reach inside any horizon here.
NEVER = 1e6


class Rain(mu.ObjFlow):
    """A steady unit of `w` per unit of time."""

    def add_flows(self):
        self.add_flow_continuous_out(name="w", var_fed_default=1.0)


class Cistern(mu.ObjFlow):
    """A volume filling at whatever reaches it, publishing its level."""

    def add_flows(self):
        self.add_flow_continuous_in(name="w")
        self.add_flow_continuous_out(name="w", var_fed_default=2.0)
        self.add_capacity(
            name="vol", flow="w", capacity=100.0, fill_rate=math.inf
        )


class Drain(mu.ObjFlow):
    """A gated outlet: it draws two per unit of time while its control
    port is on, and nothing otherwise."""

    def add_flows(self):
        self.add_flow_in(name="open", logic="and")
        self.add_flow_continuous_in(name="w")
        self.add_flow_continuous_out(name="waste")
        self.add_rule_set(
            name="duty",
            rules=[
                {
                    "name": "draining",
                    "cond": [{"name": "open", "port": "in"}],
                    "cons": {"w": 2.0},
                    "prod": {"waste": 2.0},
                },
                {
                    "name": "shut",
                    "cond": [{"name": "open", "port": "in", "negate": True}],
                    "cons": {"w": 0.0},
                    "prod": {"waste": 0.0},
                },
            ],
        )


def filling(content: float) -> type[mu.ObjFlow]:
    """A volume starting at `content` and rising with whatever reaches
    it, publishing its level like any other capacity."""

    class Filling(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_capacity(
                name="vol",
                flow="w",
                capacity=100.0,
                content_init={"w": content},
                fill_rate=math.inf,
            )

    return Filling


def probe(content: float) -> type[mu.ObjFlow]:
    """A volume holding `content` and nothing moving it: a fixed reading,
    published on `vol_level_out` like any other capacity.

    Its continuous input is left unconnected and it delivers nothing, so
    the integrated content has a zero derivative and the published level
    is exactly what it was initialised with."""

    class Probe(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_capacity(
                name="vol",
                flow="w",
                capacity=100.0,
                content_init={"w": content},
            )

    return Probe


def with_controllers(system: mu.System, objects, connections) -> dict:
    """Attach controller objects and their wiring to a built system, and
    answer the document.

    The plant is authored through `pyraichu.muscadet` and the controllers
    are a plugin section beside it, which is the shape a model takes when
    only part of it is a controller."""
    document = system.build_dict()
    body = pyraichu.model_body(document)
    body["plugins"] = {"muscadet": {"objects": list(objects)}}
    for source, source_port, target, target_port in connections:
        body["connections"].append(
            {
                "from": {"component": source, "port": source_port},
                "to": {"component": target, "port": target_port},
            }
        )
    return document


def banding(name: str, control: str, direction: str, activate, release=None) -> dict:
    """One controller: a level in, a banded boolean out."""
    emit = {
        "op": "band",
        "input": "vol",
        "direction": direction,
        "activate": activate,
    }
    if release is not None:
        emit["release"] = release
    return {
        "type": "ObjCtrl",
        "name": name,
        "controls_in": [{"name": "vol", "kind": "level"}],
        "controls_out": [{"name": control, "kind": "bool", "emit": emit}],
    }


def firings(result, automaton: str) -> list[tuple[float, str]]:
    """Every firing of one automaton, as `(date, destination state)`."""
    return [
        (event.time, event.to_state)
        for event in result.events
        if event.transition.split(".")[1] == automaton
    ]


def switches(series) -> list[tuple[float, object]]:
    """An indicator series with its repeated values collapsed: what
    actually changed, and when.

    Several entries may share one instant, the fixpoint propagation
    recording each pass over an attribute; only the value the instant
    settled on is a change, so the last entry of each instant is kept."""
    settled: list[tuple[float, object]] = []
    for time, value in series:
        if settled and settled[-1][0] == time:
            settled[-1] = (time, value)
        else:
            settled.append((time, value))
    kept: list[tuple[float, object]] = []
    for time, value in settled:
        if not kept or kept[-1][1] != value:
            kept.append((time, value))
    return kept


# --- 1. a band on a capacity level -------------------------------------


def a_banded_cistern(activate=5.0, release=3.0) -> dict:
    """A cistern filling at one and drained at two while the band holds:
    a self-regulating montage whose only decision is the band's.

    The level rises at +1 to the activation edge, then falls at -1 (one in,
    two out) to the release edge, then rises again. Every crossing date is
    therefore an exact ratio, and the deadband width is the whole period."""
    system = mu.System("banded_cistern")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.add_component(Drain, "D")
    system.connect("P", "w", "T", "w")
    system.connect("T", "w", "D", "w")
    return with_controllers(
        system,
        [banding("CTRL", "open", "above", activate, release)],
        [
            ("T", "vol_level_out", "CTRL", "vol_level_in"),
            ("CTRL", "open_out", "D", "open_in"),
        ],
    )


def test_a_band_activates_and_releases_at_its_two_levels():
    """The two edges are located, and they are located at the two levels
    the declaration names: 5 reached at t=5 on a unit ramp, 3 reached two
    units later at -1, and 5 again two units after that.

    The assertion is the **date**. Asserting that the transitions were
    emitted `watched` would pass on a montage that never fires, which is
    precisely how a crossing goes missing."""
    result = pyraichu.simulate(pyraichu.load_model(a_banded_cistern()), t_max=10.0)
    fired = firings(result, "open_band")
    assert [state for _, state in fired] == [
        "open_band_activated",
        "open_band_released",
        "open_band_activated",
    ]
    for (time, _), expected in zip(fired, (5.0, 7.0, 9.0)):
        assert abs(time - expected) < CROSSING_TOL, fired


def test_a_band_holds_between_its_edges_instead_of_chattering():
    """What the band buys over a comparison: between the two levels the
    output keeps what the last crossing left it.

    A single-threshold comparison switches back the instant its condition
    stops holding, so a montage gated on one changes state at every
    integration step around its level. The band changes state exactly
    twice per cycle, and the whole of each half-cycle is spent held."""
    result = pyraichu.simulate(pyraichu.load_model(a_banded_cistern()), t_max=10.0)
    fired = firings(result, "open_band")
    dates = [time for time, _ in fired]
    assert len(dates) == len(set(dates)), f"two firings at one instant: {fired}"
    assert switches(result.indicators["CTRL_open"])[1:] == [
        (time, state.endswith("activated")) for time, state in fired
    ]
    # Held, not merely toggled: nothing happens strictly between two
    # crossings, so the whole of each half-cycle is spent on one side of a
    # band a bare comparison would have switched back at once.
    assert not [t for t in dates if 5.0 + CROSSING_TOL < t < 7.0 - CROSSING_TOL]


# --- 2. the heated tank, which is why this lot exists -------------------

NOMINAL = 1.5
LEVEL_MIN = 6.0
LEVEL_MAX = 8.0
#: Above the regulation band on purpose: a capacity bound is a hard
#: throttle, and putting the volume at the band's top would make the wall
#: and the regulation the same thing.
TANK_VOLUME = 12.0
#: Below `LEVEL_MIN`, so the plant enters the benchmark's cycle on its
#: first crossing rather than sitting inside the band with every control
#: port released.
LEVEL_INIT = 5.0


class HTPump(mu.ObjFlow):
    """A pump: the benchmark's nominal rate, gated by a control port."""

    def add_flows(self):
        self.add_flow_in(name="run", logic="and")
        self.add_flow_continuous_out(name="water")
        self.add_rule_set(
            name="duty",
            rules=[
                {
                    "name": "running",
                    "cond": [{"name": "run", "port": "in"}],
                    "prod": {"water": NOMINAL},
                },
                {
                    "name": "idle",
                    "cond": [{"name": "run", "port": "in", "negate": True}],
                    "prod": {"water": 0.0},
                },
            ],
        )


class HTTank(mu.ObjFlow):
    """The regulated volume, publishing its level to whoever watches it."""

    def add_flows(self):
        self.add_flow_continuous_in(name="water")
        self.add_flow_continuous_out(name="water", var_fed_default=NOMINAL)
        self.add_capacity(
            name="tank",
            flow="water",
            capacity=TANK_VOLUME,
            content_init={"water": LEVEL_INIT},
            fill_rate=math.inf,
        )


class HTValve(mu.ObjFlow):
    """The drain valve: the pump's mirror, consuming instead of producing.

    `drain` is deliberately wired to nothing. An unconnected continuous
    output constrains no demand, so the rule runs at its nominal scale and
    the valve claims exactly the coefficient it declares."""

    def add_flows(self):
        self.add_flow_in(name="open", logic="and")
        self.add_flow_continuous_in(name="water")
        self.add_flow_continuous_out(name="drain")
        self.add_rule_set(
            name="duty",
            rules=[
                {
                    "name": "draining",
                    "cond": [{"name": "open", "port": "in"}],
                    "cons": {"water": NOMINAL},
                    "prod": {"drain": NOMINAL},
                },
                {
                    "name": "shut",
                    "cond": [{"name": "open", "port": "in", "negate": True}],
                    "cons": {"water": 0.0},
                    "prod": {"drain": 0.0},
                },
            ],
        )


def regulator(name: str, control: str, direction: str, activate, release) -> dict:
    return {
        "type": "ObjCtrl",
        "name": name,
        "controls_in": [{"name": "tank", "kind": "level"}],
        "controls_out": [
            {
                "name": control,
                "kind": "bool",
                "emit": {
                    "op": "band",
                    "input": "tank",
                    "direction": direction,
                    "activate": activate,
                    "release": release,
                },
            }
        ],
    }


def the_heated_tank() -> dict:
    """The benchmark's regulation: two pumps below 6, one valve above 8.

    The classical dynamic reliability benchmark of T. Aldemir,
    *Computer-Assisted Markov Failure Modeling of Process Control
    Systems*, IEEE Transactions on Reliability (1987), DOI
    `10.1109/tr.1987.5222318`.

    Below 6 both pumps run and the valve is shut, so the level rises at
    3.0; above 8 both pumps stop and the valve drains, so it falls at 1.5.
    Two units up at 3.0 and two units down at 1.5 is a period of exactly
    two hours, and that period is a property of the deadband rather than of
    the integrator."""
    system = mu.System("heated_tank")
    system.add_component(HTPump, "P1")
    system.add_component(HTPump, "P2")
    system.add_component(HTTank, "T")
    system.add_component(HTValve, "V3")
    system.connect("P1", "water", "T", "water")
    system.connect("P2", "water", "T", "water")
    system.connect("T", "water", "V3", "water")
    return with_controllers(
        system,
        [
            regulator("LOW", "run", "below", LEVEL_MIN, LEVEL_MAX),
            regulator("HIGH", "open", "above", LEVEL_MAX, LEVEL_MIN),
        ],
        [
            ("T", "tank_level_out", "LOW", "tank_level_in"),
            ("T", "tank_level_out", "HIGH", "tank_level_in"),
            ("LOW", "run_out", "P1", "run_in"),
            ("LOW", "run_out", "P2", "run_in"),
            ("HIGH", "open_out", "V3", "open_in"),
        ],
    )


@pytest.fixture(scope="module")
def heated_tank_run():
    """One run, checked for confluence: two controllers write at the same
    instants (one releases where the other activates), so the answer must
    not depend on the order the engine happens to visit them in."""
    return pyraichu.simulate(
        pyraichu.load_model(the_heated_tank()), t_max=9.0, confluence_check=True
    )


def test_the_heated_tank_regulation_cycles_at_the_benchmark_s_two_hours(
    heated_tank_run,
):
    """The measurement this whole lot exists for.

    Re-encoded on the muscadet layer's rule guards the level freezes at 6
    and never cycles, a mode holding no memory of which side it came from.
    Banded, the plant reproduces the reference cycle: the pumps restart
    every two hours, and the period is exact to the event-location
    tolerance rather than approximate."""
    starts = [
        time
        for time, state in firings(heated_tank_run, "run_band")
        if state.endswith("activated")
    ]
    # The first start is the initial condition (the plant is filled to 5,
    # below the band), not a cycle: the period is measured from the second.
    assert len(starts) >= 4, f"expected several cycles, got {starts}"
    periods = [b - a for a, b in zip(starts[1:], starts[2:])]
    for period in periods:
        assert abs(period - 2.0) < CROSSING_TOL, periods


def test_the_heated_tank_switches_at_the_two_thresholds_it_declares(
    heated_tank_run,
):
    """Where each half-cycle turns: the pumps stop and the valve opens at
    8, and the pumps restart and the valve shuts at 6.

    Starting at 5, the first rise at 3.0 reaches 8 at t = 1; the fall at
    1.5 reaches 6 at t = 1 + 4/3; the rise reaches 8 again at t = 3."""
    assert [
        (round(time, 6), state) for time, state in firings(heated_tank_run, "run_band")
    ][:3] == [
        (0.0, "run_band_activated"),
        (1.0, "run_band_released"),
        (round(1.0 + 4.0 / 3.0, 6), "run_band_activated"),
    ]
    assert [
        (round(time, 6), state) for time, state in firings(heated_tank_run, "open_band")
    ][:2] == [
        (1.0, "open_band_activated"),
        (round(1.0 + 4.0 / 3.0, 6), "open_band_released"),
    ]


def test_the_heated_tank_holds_its_level_inside_the_regulation_band(
    heated_tank_run,
):
    """The regulation does what it is for: past the first rise from the
    initial 5, the level never leaves [6, 8]."""
    level = [
        value
        for time, value in heated_tank_run.indicators["T_tank_content"]
        if time > 1.0
    ]
    assert level, "the plant never reached its operating band"
    assert min(level) >= LEVEL_MIN - CROSSING_TOL
    assert max(level) <= LEVEL_MAX + CROSSING_TOL


def test_the_pumps_and_the_valve_never_run_together_in_band(heated_tank_run):
    """The deadband is what keeps the two regulators from fighting: the
    plant is either filling at 3.0 or draining at 1.5, never both."""
    run = dict(heated_tank_run.indicators["LOW_run"])
    open_ = dict(heated_tank_run.indicators["HIGH_open"])
    for instant in sorted(set(run) | set(open_)):
        if instant <= 1.0:
            continue
        filling = run.get(instant)
        draining = open_.get(instant)
        if filling is not None and draining is not None:
            assert not (filling and draining), instant


# --- 3. aggregating several readings -----------------------------------


def a_voter(aggregate: str, contents: list[float]) -> dict:
    """One controller reading `len(contents)` publishers on one input,
    reduced by `aggregate`."""
    system = mu.System(f"voter_{aggregate}_{len(contents)}")
    for index, content in enumerate(contents):
        system.add_component(probe(content), f"S{index}")
    return with_controllers(
        system,
        [
            {
                "type": "ObjCtrl",
                "name": "V",
                "controls_in": [
                    {"name": "vol", "kind": "level", "aggregate": aggregate}
                ],
                "controls_out": [],
            }
        ],
        [
            (f"S{index}", "vol_level_out", "V", "vol_level_in")
            for index in range(len(contents))
        ],
    )


@pytest.mark.parametrize(
    "contents, expected",
    [
        ([2.0, 7.0, 5.0], 5.0),
        ([9.0, 1.0, 4.0, 6.0, 3.0], 4.0),
        ([2.0, 7.0, 5.0, 11.0], 6.0),
        ([1.0, 4.0], 2.5),
    ],
    ids=["odd-3", "odd-5", "even-4", "even-2"],
)
def test_a_median_input_reduces_an_odd_and_an_even_count(contents, expected):
    """The median of an odd count is the central reading; the median of an
    **even** count is the mean of the two central ones, and not a
    tie-break.

    muscadet settles that in one place and this engine's `AggOp::Median`
    answers the same way, so an even count needs no caution beyond knowing
    that it interpolates: `[2, 5, 7, 11]` reads 6, which is a value none of
    the four publishers carries."""
    result = pyraichu.simulate(
        pyraichu.load_model(a_voter("median", contents)), t_max=1.0
    )
    assert abs(result.indicators["V_vol_level"][-1][1] - expected) < TOL


def test_a_mean_and_a_sum_input_reduce_the_same_readings_differently():
    """The two linear reductions, on the readings the median test uses:
    they are what a rank-sensitive one is chosen against."""
    for aggregate, expected in (("mean", 6.25), ("sum", 25.0)):
        result = pyraichu.simulate(
            pyraichu.load_model(a_voter(aggregate, [2.0, 7.0, 5.0, 11.0])),
            t_max=1.0,
        )
        assert abs(result.indicators["V_vol_level"][-1][1] - expected) < TOL


def test_a_minimum_or_a_maximum_is_refused_by_name():
    """`min` and `max` are declarable and refused, with the reason named.

    The only variable-arity reader of an in port is `port_agg`, whose
    aggregations carry no minimum and no maximum, and a measurement link
    declares no per-connection channel a fixed-arity expression could read
    one by one. Answering with a sum or a mean would be a different
    question asked in silence."""
    for aggregate in ("min", "max"):
        with pytest.raises(ValueError, match="port_agg"):
            pyraichu.expand_model(a_voter(aggregate, [2.0, 7.0, 5.0]))


# --- 4. k-of-n ----------------------------------------------------------


def a_vote(k: int) -> dict:
    """Three instruments, one k-of-n vote over three comparisons.

    `A` reads a fixed 6 and holds from the start; `B` fills at one from 4
    and crosses 5 at t = 1; `C` reads a fixed 0 and never holds. So one
    operand holds before t = 1 and two hold after it, and `k` is what
    decides whether the vote turns there."""
    system = mu.System(f"vote_{k}_of_3")
    system.add_component(probe(6.0), "A")
    system.add_component(Rain, "P")
    system.add_component(filling(4.0), "B")
    system.add_component(probe(0.0), "C")
    system.connect("P", "w", "B", "w")
    compare = [
        {"op": "compare", "input": name, "operator": ">=", "threshold": 5.0}
        for name in ("a", "b", "c")
    ]
    return with_controllers(
        system,
        [
            {
                "type": "ObjCtrl",
                "name": "VOTE",
                "controls_in": [{"name": name, "kind": "level"} for name in "abc"],
                "controls_out": [
                    {
                        "name": "trip",
                        "kind": "bool",
                        "emit": {"op": "combine", "logic": "k", "k": k, "operands": compare},
                    }
                ],
            }
        ],
        [
            ("A", "vol_level_out", "VOTE", "a_level_in"),
            ("B", "vol_level_out", "VOTE", "b_level_in"),
            ("C", "vol_level_out", "VOTE", "c_level_in"),
        ],
    )


def test_a_two_of_three_vote_turns_when_the_second_operand_crosses():
    """`B` starts at 4 and fills at one, so it reaches 5 at t = 1. One
    operand holds before that instant and two after it, so a 2-of-3 vote
    turns exactly there: the located crossing of the operand, not the
    date some other event happened to produce."""
    result = pyraichu.simulate(pyraichu.load_model(a_vote(2)), t_max=3.0)
    trip = switches(result.indicators["VOTE_trip"])
    assert [value for _, value in trip] == [False, True]
    assert abs(trip[1][0] - 1.0) < CROSSING_TOL, trip


def test_a_three_of_three_vote_never_holds_while_one_operand_is_below():
    """`C` reads zero for the whole run, so a unanimity vote never turns:
    the count is a number the expression carries, not a shape it takes."""
    result = pyraichu.simulate(pyraichu.load_model(a_vote(3)), t_max=3.0)
    assert switches(result.indicators["VOTE_trip"]) == [(0.0, False)]


def test_a_one_of_three_vote_holds_from_the_start():
    """`A` reads six for the whole run, so a 1-of-3 vote holds at t = 0
    and never falls back."""
    result = pyraichu.simulate(pyraichu.load_model(a_vote(1)), t_max=3.0)
    assert switches(result.indicators["VOTE_trip"]) == [(0.0, True)]


def test_the_logical_combinations_reduce_the_same_operands():
    """`and`, `or` and `not` over the vote's own operands: one holds and
    one does not before t = 1, which separates the three."""

    def combined(logic, operands):
        system = mu.System(f"combine_{logic}")
        system.add_component(probe(6.0), "A")
        system.add_component(probe(0.0), "C")
        return with_controllers(
            system,
            [
                {
                    "type": "ObjCtrl",
                    "name": "K",
                    "controls_in": [{"name": name, "kind": "level"} for name in "ac"],
                    "controls_out": [
                        {
                            "name": "out",
                            "kind": "bool",
                            "emit": {
                                "op": "combine",
                                "logic": logic,
                                "operands": [
                                    {
                                        "op": "compare",
                                        "input": name,
                                        "operator": ">=",
                                        "threshold": 5.0,
                                    }
                                    for name in operands
                                ],
                            },
                        }
                    ],
                }
            ],
            [
                ("A", "vol_level_out", "K", "a_level_in"),
                ("C", "vol_level_out", "K", "c_level_in"),
            ],
        )

    def emitted(logic, operands):
        result = pyraichu.simulate(
            pyraichu.load_model(combined(logic, operands)), t_max=1.0
        )
        return result.indicators["K_out"][-1][1]

    assert emitted("and", "ac") is False
    assert emitted("or", "ac") is True
    assert emitted("not", "c") is True
    assert emitted("not", "a") is False


# --- 5. an instrument that lies when it fails ---------------------------


def a_lying_instrument() -> dict:
    """A cistern filling at one, read by an instrument that republishes
    its level, watched in turn by a control that trips above 9.

    Left alone the trip would come at t = 9, the instant the true level
    reaches 9. A failure mode forces the instrument's publication to 20 at
    t = 3, and what the control reacts to is that number."""
    system = mu.System("lying_instrument")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.connect("P", "w", "T", "w")
    return with_controllers(
        system,
        [
            {
                "type": "ObjCtrl",
                "name": "INSTR",
                "controls_in": [{"name": "vol", "kind": "level"}],
                "controls_out": [
                    {
                        "name": "reading",
                        "kind": "value",
                        "emit": {"op": "republish", "input": "vol", "gain": 1.0},
                    }
                ],
            },
            {
                "type": "ObjCtrl",
                "name": "CTRL",
                "controls_in": [{"name": "reading", "kind": "level"}],
                "controls_out": [
                    {
                        "name": "trip",
                        "kind": "bool",
                        "emit": {
                            "op": "compare",
                            "input": "reading",
                            "operator": ">=",
                            "threshold": 9.0,
                        },
                    }
                ],
            },
            {
                "type": "ObjFM",
                "name": "blind",
                "targets": ["INSTR"],
                "failure": {"distrib": "delay", "time": 3.0},
                "repair": {"distrib": "delay", "time": NEVER},
                "failure_effects": {"reading_forced": True, "reading_forced_value": 20.0},
            },
        ],
        [
            ("T", "vol_level_out", "INSTR", "vol_level_in"),
            ("INSTR", "reading_level_out", "CTRL", "reading_level_in"),
        ],
    )


def test_a_forced_instrument_makes_the_control_react_to_the_wrong_reading():
    """The instrument lies from t = 3 and the control believes it.

    The true level is 3 there and would not reach the trip threshold of 9
    before t = 9. The trip nevertheless fires at t = 3, on the forced
    publication: what a control reacts to is what its instrument tells it,
    which is the whole point of making the publication an endpoint a mode
    can clamp."""
    result = pyraichu.simulate(pyraichu.load_model(a_lying_instrument()), t_max=8.0)
    trip = switches(result.indicators["CTRL_trip"])
    assert [value for _, value in trip] == [False, True]
    assert abs(trip[1][0] - 3.0) < CROSSING_TOL, trip
    # The volume itself goes on being right underneath the lie: a dead
    # instrument is not an empty tank.
    truth = dict(result.indicators["T_vol_content"])
    assert abs(truth[trip[1][0]] - 3.0) < CROSSING_TOL


def test_a_forced_publication_is_scaled_by_the_gain_like_any_other():
    """One publication path and one gain. Routing a forced value around
    the gain would make a mode that kills the gain of a forced instrument
    a silent no-op, so the forced number is published times the gain: at
    the declared gain of 1 it is exactly the number the mode named."""
    result = pyraichu.simulate(pyraichu.load_model(a_lying_instrument()), t_max=8.0)
    published = result.indicators["INSTR_reading_level"]
    # An indicator records what changed, so the forced publication is the
    # last thing it recorded and it was recorded at the failure date.
    assert abs(published[-1][1] - 20.0) < TOL, published
    assert abs(published[-1][0] - 3.0) < CROSSING_TOL, published


def a_blinded_signal() -> dict:
    """A control whose boolean output is blinded by a mode, and un-blinded
    by its repair."""
    system = mu.System("blinded_signal")
    system.add_component(probe(6.0), "A")
    return with_controllers(
        system,
        [
            {
                "type": "ObjCtrl",
                "name": "CTRL",
                "controls_in": [{"name": "vol", "kind": "level"}],
                "controls_out": [
                    {
                        "name": "trip",
                        "kind": "bool",
                        "default": False,
                        "emit": {
                            "op": "compare",
                            "input": "vol",
                            "operator": ">=",
                            "threshold": 5.0,
                        },
                    }
                ],
            },
            {
                "type": "ObjFM",
                "name": "mute",
                "targets": ["CTRL"],
                "failure": {"distrib": "delay", "time": 2.0},
                "repair": {"distrib": "delay", "time": 3.0},
                "failure_effects": {"trip_signal_available": False},
            },
        ],
        [("A", "vol_level_out", "CTRL", "vol_level_in")],
    )


def test_a_blinded_boolean_output_falls_back_and_comes_back():
    """Both edges, and the returning one is the load-bearing half.

    The reading holds the condition for the whole run, so the signal is
    true except while the mode blinds it: false from t = 2, true again at
    t = 5. muscadet needs an automaton of its own for that return, a
    PyCATSHOO sensitive method having to hang off something that
    announces; here the emission reads the availability directly and the
    engine derives the trigger from that read, on both edges."""
    result = pyraichu.simulate(pyraichu.load_model(a_blinded_signal()), t_max=6.0)
    trip = switches(result.indicators["CTRL_trip"])
    assert [value for _, value in trip] == [True, False, True]
    assert abs(trip[1][0] - 2.0) < CROSSING_TOL, trip
    assert abs(trip[2][0] - 5.0) < CROSSING_TOL, trip


# --- 6. the degenerate band ---------------------------------------------


def a_degenerate_pair() -> dict:
    """Two degenerate bands, one each way, on a level rising at one.

    Neither declares a release edge, so each releases at its own activation
    level. The release comparison being strict is what keeps the two edges
    mutually exclusive at that single level."""
    system = mu.System("degenerate_band")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.connect("P", "w", "T", "w")
    return with_controllers(
        system,
        [
            banding("UP", "high", "above", 5.0),
            banding("DOWN", "low", "below", 5.0),
        ],
        [
            ("T", "vol_level_out", "UP", "vol_level_in"),
            ("T", "vol_level_out", "DOWN", "vol_level_in"),
        ],
    )


def test_a_degenerate_band_switches_at_its_single_level():
    """Release equal to activate is no hysteresis at all, and muscadet
    documents what that leaves: the two edges stay mutually exclusive
    because the release comparison is strict, so the output switches at
    that one level instead of holding both edges at once.

    On a unit ramp through 5, the `above` band activates at t = 5 and the
    `below` band releases at t = 5, and at no instant are both held."""
    result = pyraichu.simulate(pyraichu.load_model(a_degenerate_pair()), t_max=8.0)
    high = switches(result.indicators["UP_high"])
    low = switches(result.indicators["DOWN_low"])
    assert [value for _, value in high] == [False, True]
    assert [value for _, value in low] == [True, False]
    assert abs(high[1][0] - 5.0) < CROSSING_TOL, high
    assert abs(low[1][0] - 5.0) < CROSSING_TOL, low
    held_high = dict(result.indicators["UP_high"])
    held_low = dict(result.indicators["DOWN_low"])
    for instant in sorted(set(held_high) & set(held_low)):
        assert not (held_high[instant] and held_low[instant]), instant


def test_an_inverted_band_is_refused_where_it_is_written():
    """A band detecting below 3 and releasing at 1 can never release: the
    reading has to fall to 1 while the band is what stops it falling. The
    montage would latch on its first activation and never speak again,
    with nothing raised anywhere, so the declaration is refused."""
    with pytest.raises(ValueError, match="releases at or above it"):
        pyraichu.expand_model(
            {
                "name": "inverted",
                "plugins": {
                    "muscadet": {"objects": [banding("X", "y", "below", 3.0, 1.0)]}
                },
            }
        )


# --- the evaluation order and the feature envelope ----------------------


def test_every_step_a_controller_emits_joins_the_evaluation_order_once():
    """The order must cover the declared steps exactly, one entry each, so
    a controller landing beside a continuous plant has to add its own or
    the whole model is refused."""
    body = pyraichu.expand_model(the_heated_tank())
    order = [(step["component"], step["attribute"]) for step in body["evaluation_order"]]
    assert len(order) == len(set(order)), "a step is listed twice"
    declared = [
        (component["name"], equation["target"])
        for component in body["components"]
        for equation in component["equations"]
        if equation["kind"] == "explicit"
    ] + [
        (component["name"], allocation["name"])
        for component in body["components"]
        for allocation in component.get("allocations", [])
    ]
    assert sorted(order) == sorted(declared)
    # And each controller's reading is swept downstream of the level it
    # mirrors, which is what makes the band read the current content.
    assert order.index(("T", "tank_content")) < order.index(("LOW", "tank_level"))
    assert order.index(("T", "tank_content")) < order.index(("HIGH", "tank_level"))


def test_a_controller_on_a_bare_body_needs_no_evaluation_order():
    """A model carrying no order keeps the positional sweep, and a
    controller's component is appended after everything already in the
    model, so its reading is already downstream of what it reads. The
    field is not written to say what the default already says."""
    body = pyraichu.expand_model(a_voter("median", [2.0, 7.0, 5.0]))
    assert "evaluation_order" in body  # the plant itself carries capacities
    bare = pyraichu.expand_model(
        {
            "name": "bare",
            "components": [],
            "plugins": {
                "muscadet": {
                    "objects": [
                        {
                            "type": "ObjCtrl",
                            "name": "C",
                            "controls_in": [{"name": "vol", "kind": "level"}],
                            "controls_out": [],
                        }
                    ]
                }
            },
        }
    )
    assert "evaluation_order" not in bare


def test_the_band_transitions_are_declared_watched():
    """The cheap regression beside the located ones: a band's two edges
    turn on a continuously-evolving reading, so both are watched and the
    engine root-finds their dates.

    This assertion is worth exactly what it says and no more, which is why
    every behavioural test above asserts a date instead."""
    body = pyraichu.expand_model(the_heated_tank())
    low = next(c for c in body["components"] if c["name"] == "LOW")
    band = next(a for a in low["automata"] if a["name"] == "run_band")
    assert {t["distrib"] for t in band["transitions"]} == {"watched"}
    assert [t["guard"]["cmp"] for t in band["transitions"]] == ["le", "gt"]


# --- refusals -----------------------------------------------------------


def controller(**over) -> dict:
    spec = {
        "type": "ObjCtrl",
        "name": "C",
        "controls_in": [{"name": "vol", "kind": "level"}],
        "controls_out": [],
    }
    spec.update(over)
    return {"name": "refusal", "plugins": {"muscadet": {"objects": [spec]}}}


@pytest.mark.parametrize(
    "over, message",
    [
        (
            {"controls_out": [{"name": "o", "emit": lambda level: level > 3}]},
            "closed operators",
        ),
        (
            {"controls_out": [{"name": "o", "emit": {"op": "divide", "input": "vol"}}]},
            "unknown operator",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "emit": {
                            "op": "compare",
                            "input": "vol",
                            "operator": "==",
                            "threshold": 1.0,
                        },
                    }
                ]
            },
            "comparison operator is one of",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "emit": {"op": "combine", "logic": "or", "operands": []},
                    }
                ]
            },
            "no operand is a constant",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "emit": {
                            "op": "combine",
                            "logic": "k",
                            "k": 3,
                            "operands": [
                                {
                                    "op": "band",
                                    "input": "vol",
                                    "direction": "above",
                                    "activate": 1.0,
                                }
                            ],
                        },
                    }
                ]
            },
            "can never hold",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "emit": {
                            "op": "combine",
                            "logic": "or",
                            "k": 2,
                            "operands": [
                                {
                                    "op": "band",
                                    "input": "vol",
                                    "direction": "above",
                                    "activate": 1.0,
                                }
                            ],
                        },
                    }
                ]
            },
            "has no meaning beside",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "kind": "bool",
                        "emit": {"op": "republish", "input": "vol"},
                    }
                ]
            },
            "carries a condition",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "kind": "value",
                        "emit": {
                            "op": "band",
                            "input": "vol",
                            "direction": "above",
                            "activate": 1.0,
                        },
                    }
                ]
            },
            "carries a number",
        ),
        (
            {
                "controls_out": [
                    {
                        "name": "o",
                        "emit": {
                            "op": "band",
                            "input": "missing",
                            "direction": "above",
                            "activate": 1.0,
                        },
                    }
                ]
            },
            "does not declare",
        ),
        (
            {"controls_in": [{"name": "vol", "kind": "temperature"}]},
            "a measurement reads one of",
        ),
        (
            {"controls_in": [{"name": "vol", "kind": "ratio"}]},
            "share of ONE constituent",
        ),
        (
            {"controls_in": [{"name": "vol", "kind": "rate", "flows": ["w"]}]},
            "has no constituent",
        ),
        (
            {"controls_in": [{"name": "vol", "kind": "level", "combine": "min"}]},
            "does not accept declaration key",
        ),
        ({"controls_in": [], "controls_out": []}, "at least one observation input"),
    ],
)
def test_a_malformed_controller_is_refused_where_it_is_written(over, message):
    """Every refusal names what is wrong and where. A misspelt key is
    otherwise swallowed whole, and a controller silently missing a
    threshold is indistinguishable from one that never declared any."""
    with pytest.raises(ValueError, match=message):
        pyraichu.expand_model(controller(**over))


# --- regression ---------------------------------------------------------


def test_a_model_declaring_no_controller_expands_exactly_as_before():
    """The port is additive: one entry in the expander registry and a
    module of its own, so a model naming no `ObjCtrl` must expand
    byte-identically to what it expanded to before.

    Measured rather than argued: the same model is expanded with the entry
    present and with it removed, which is exactly the state of the registry
    before this lot, and the two documents are compared as text."""
    system = mu.System("no_controller")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.connect("P", "w", "T", "w")
    document = system.build_dict()
    body = pyraichu.model_body(document)
    body["plugins"] = {
        "muscadet": {
            "objects": [
                {
                    "type": "ObjFlow",
                    "name": "S",
                    "flows_out": [{"name": "is_ok", "var_prod_default": True}],
                }
            ]
        }
    }
    with_entry = json.dumps(pyraichu.expand_model(document), sort_keys=True)
    removed = MuscadetPlugin.EXPANDERS.pop("ObjCtrl")
    try:
        without_entry = json.dumps(pyraichu.expand_model(document), sort_keys=True)
    finally:
        MuscadetPlugin.EXPANDERS["ObjCtrl"] = removed
    assert with_entry == without_entry


def test_the_registry_gained_exactly_one_entry():
    """The footprint on the shared plugin module, asserted rather than
    remembered: `ObjCtrl` beside the five object types that were there."""
    assert sorted(MuscadetPlugin.EXPANDERS) == [
        "ObjCtrl",
        "ObjEvent",
        "ObjFM",
        "ObjFMInst",
        "ObjFlow",
        "ObjLogicGate",
    ]


@pytest.mark.parametrize(
    "declaration, attribute",
    [
        ({"name": "vol", "kind": "level"}, "vol_level"),
        ({"name": "vol", "kind": "level", "flows": ["w"]}, "vol_level_w"),
        ({"name": "line", "kind": "rate"}, "line_rate"),
        ({"name": "vol", "kind": "ratio", "flows": ["heat"]}, "vol_ratio_heat"),
    ],
    ids=["level", "constituent-level", "rate", "ratio"],
)
def test_an_observation_input_reads_the_name_its_publisher_writes(
    declaration, attribute
):
    """The three natures of measurement, each landing on the name the
    publisher of that quantity uses: a capacity publishes `{name}_level`
    and `{name}_level_{flow}`, a continuous output `{name}_rate`, a volume
    `{name}_ratio_{flow}`.

    One port per attribute, unlike the PyCATSHOO message box that carries
    several: RAICHU connects attribute to attribute, so a ratio takes a
    port of its own where muscadet shared the level's box. Only the level
    has a publisher in this layer today; the other two are the wire shape a
    publisher of theirs will meet."""
    body = pyraichu.expand_model(
        controller(controls_in=[declaration], controls_out=[])
    )
    component = next(c for c in body["components"] if c["name"] == "C")
    assert [a["name"] for a in component["attributes"]] == [attribute]
    assert component["ports"] == [{"name": f"{attribute}_in", "dir": "in"}]
    assert component["equations"] == [
        {
            "target": attribute,
            "kind": "explicit",
            "expr": {
                "op": "port_agg",
                "port": {"component": "C", "port": f"{attribute}_in"},
                "agg": "sum",
            },
        }
    ]


def test_a_threshold_is_an_attribute_a_failure_mode_can_move():
    """Every number the grammar carries is an attribute of the model, named
    from the node's position in the output's tree.

    That is one change with three consequences, and each needs a value
    somewhere for anything to reach: two instances tuned to different
    thresholds, an indicator naming a threshold as its target, and a
    failure mode moving one. While a threshold was a constant inlined into
    an expression there was no value for any of the three."""
    body = pyraichu.expand_model(the_heated_tank())
    low = next(c for c in body["components"] if c["name"] == "LOW")
    declared = {a["name"]: a["init"]["value"] for a in low["attributes"]}
    assert list(declared) == [
        "tank_level",
        "run",
        "run_signal_available",
        "run_activate",
        "run_release",
    ]
    assert declared["run_activate"] == LEVEL_MIN
    assert declared["run_release"] == LEVEL_MAX
    # And the guards read those attributes rather than the numbers, so a
    # mode that moves one moves the located crossing with it.
    band = next(a for a in low["automata"] if a["name"] == "run_band")
    assert [t["guard"]["rhs"]["attr"]["attribute"] for t in band["transitions"]] == [
        "run_activate",
        "run_release",
    ]


def test_two_nodes_on_one_input_carry_two_thresholds():
    """A path in the tree is what tells two operators apart: two
    comparisons on the same reading against two levels are two automata and
    two threshold attributes, named from their position."""
    body = pyraichu.expand_model(
        controller(
            controls_out=[
                {
                    "name": "o",
                    "emit": {
                        "op": "combine",
                        "logic": "and",
                        "operands": [
                            {"op": "compare", "input": "vol", "operator": ">=", "threshold": 1.0},
                            {"op": "compare", "input": "vol", "operator": "<=", "threshold": 9.0},
                        ],
                    },
                }
            ]
        )
    )
    component = next(c for c in body["components"] if c["name"] == "C")
    assert [a["name"] for a in component["automata"]] == [
        "o_operand_0_compare",
        "o_operand_1_compare",
    ]
    thresholds = [
        a["name"]
        for a in component["attributes"]
        if a["name"].endswith("threshold")
    ]
    assert thresholds == ["o_operand_0_threshold", "o_operand_1_threshold"]


def test_a_chained_controller_is_swept_after_the_one_it_reads():
    """A controller reading another controller's published number is swept
    after it, so the reading is the publication of this evaluation point
    and not of the previous one.

    Declaration order is what settles it: a plugin expands one object at a
    time and has no pre-run step, where muscadet derives the order from a
    topological sort of the whole signal graph. The rule that costs is
    written in the module's docstring, and this is its measurement."""
    body = pyraichu.expand_model(a_lying_instrument())
    order = [(step["component"], step["attribute"]) for step in body["evaluation_order"]]
    assert order.index(("T", "vol_content")) < order.index(("INSTR", "vol_level"))
    assert order.index(("INSTR", "vol_level")) < order.index(("INSTR", "reading_level"))
    assert order.index(("INSTR", "reading_level")) < order.index(("CTRL", "reading_level"))
    assert len(order) == len(set(order))
