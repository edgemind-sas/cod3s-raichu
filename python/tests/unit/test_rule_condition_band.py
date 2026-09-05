"""A rule condition can carry a band, and that is what breaks a mode
whose own guard it moves.

A rule guard comparing a quantity is right when nothing in the model
pushes back on that quantity: the mode switches where the read crosses
the threshold, and stays switched. It is wrong when the rule ITSELF moves
what it reads. Then the two states each produce the condition that
justifies the other, there is no fixpoint, and the mode alternates at the
scale of the numerical hysteresis of the crossing rather than at any
physical scale.

The model below is that loop in four components, and it is the loop of a
real plant reduced to its bones: a pump starts when the supply reaching
it clears a floor, and starting is exactly what drains the reserve that
announces the supply. On one threshold it fires thousands of times per
unit of time; on a band it starts once.

`release` declares the far edge. `value` is where the rule is entered,
`release` where it is left, and between the two the mode holds whatever
it already was. The loop is still there in the dependency graph: what
changes is that crossing the band costs physical time, so the cycle has a
physical period instead of a numerical one. That is why the cure is a
band and not the removal of the threshold, which would have thrown away
a real constraint.
"""

import pytest

import pyraichu
import pyraichu.muscadet as mu

#: The floor the pump starts on, and the reserve that announces it.
START_AT = 4.0
RESERVE = 5.0


def pump_system(release: float | None = None, floor: float = START_AT) -> mu.System:
    """A supply of 3 into a reserve, a pump drawing 10 when it runs, and
    a load. The reserve announces 10 while it holds something and only
    what arrives once empty, so the pump's own draw destroys the reading
    its start condition rests on."""
    cond: dict[str, object] = {
        "name": "E",
        "port": "in",
        "op": ">=",
        "value": floor,
    }
    if release is not None:
        cond["release"] = release

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="E", var_fed_default=3.0)

    class Reserve(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E", var_demand_in_default=10.0)
            self.add_flow_continuous_out(name="E", var_fed_default=10.0)
            self.add_capacity(
                name="store",
                flow="E",
                capacity=100.0,
                content_init={"E": RESERVE},
            )

    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E")
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[
                    {
                        "name": "on",
                        "cond": [cond],
                        "cons": {"E": 10.0},
                        "prod": {"W": 1.0},
                    },
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    class Load(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="W", var_demand_in_default=10.0)

    system = mu.System("pump")
    system.add_component(Source, "S")
    system.add_component(Reserve, "T")
    system.add_component(Pump, "P")
    system.add_component(Load, "L")
    system.connect("S", "E", "T", "E")
    system.connect("T", "E", "P", "E")
    system.connect("P", "W", "L", "W")
    return system


# --- 1. the loop, and the cure -------------------------------------------


def test_a_single_threshold_on_a_quantity_the_rule_moves_does_not_settle():
    """The premise. Without this failing, the band below would be a
    feature with no problem to solve."""
    with pytest.raises(pyraichu.SimulationError) as raised:
        pump_system().simulate(t_max=20.0, max_transition_firings=5000)
    message = str(raised.value)
    assert "duty_off_to_on" in message, message
    assert "chattering, not evolving" in message, message


def test_a_band_settles_the_same_model():
    """The cure, and it is one key. The pump starts once and the reserve
    then empties once: two events for the whole run, where the single
    threshold could not finish one unit of time in five thousand."""
    result = pump_system(release=1.0).simulate(t_max=20.0)
    assert len(result.events) == 2, result.events
    assert result.events[0].transition.endswith("duty_off_to_on")
    assert result.events[0].time == 0.0
    assert result.events[1].transition.endswith("store_reach_empty")


def test_the_band_holds_the_mode_between_its_edges():
    """What the band means, asserted rather than assumed: below the entry
    threshold and above the release one, the mode keeps the state it had.

    The reserve is drained past 4 and down towards 1 while the pump keeps
    running, which a single threshold would have forbidden."""
    result = pump_system(release=1.0).simulate(
        t_max=20.0, samples=[0.5, 0.72, 2.0]
    )
    running = dict(result.samples["P_W_fed_out"])
    # Still delivering after the supply fell through the entry threshold.
    assert running[0.72] > 0.0
    assert running[2.0] > 0.0


# --- 2. what a band is not -----------------------------------------------


def one_pump(cond: dict[str, object]) -> mu.System:
    """A pump carrying one guarded rule, for the declarations that must be
    refused and for reading the automaton the layer generates."""

    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E")
            self.add_flow_continuous_in(name="stop")
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[
                    {"name": "on", "cond": [cond], "prod": {"W": 1.0}},
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    system = mu.System("one_pump")
    system.add_component(Pump, "P")
    return system


def test_a_band_needs_a_comparison():
    """A band widens a threshold, so there has to be one."""
    with pytest.raises(ValueError, match="without comparing it"):
        one_pump({"name": "E", "port": "in", "release": 1.0})


def test_a_band_needs_an_ordering_comparison():
    """`==` has no side to be left on."""
    with pytest.raises(ValueError, match="needs an ordering comparison"):
        one_pump(
            {
                "name": "E",
                "port": "in",
                "op": "==",
                "value": 4.0,
                "release": 1.0,
            }
        )


def test_a_band_of_zero_width_is_refused():
    """The one outcome worth ruling out: a band that looks declared and
    behaves as the single threshold it was meant to replace."""
    with pytest.raises(ValueError, match="band of zero width"):
        pump_system(release=START_AT)


def test_a_band_on_the_wrong_side_is_refused():
    """A rule entered on the way up is left on the way down. Released
    above its entry threshold, the mode would switch twice on one
    crossing instead of holding."""
    with pytest.raises(ValueError, match="sits BELOW"):
        pump_system(release=START_AT + 1.0)


def test_a_falling_band_is_refused_below_its_entry():
    """The mirror rule, on a rule entered on the way down."""
    with pytest.raises(ValueError, match="sits ABOVE"):
        one_pump(
            {
                "name": "E",
                "port": "in",
                "op": "<=",
                "value": 4.0,
                "release": 3.0,
            }
        )


# --- 3. one key, two thresholds, each on its own transition --------------


def mode_guards(system: mu.System) -> dict[str, str]:
    """The generated mode automaton's transitions, as text."""
    automaton = next(
        a
        for component in system.build_dict()["model"]["components"]
        for a in component.get("automata", [])
        if a["name"] == "duty_mode"
    )
    return {t["name"]: str(t["guard"]) for t in automaton["transitions"]}


def test_the_release_edge_only_governs_the_mode_being_left():
    """Read off the generated automaton: entering `on` compares with the
    entry edge, leaving it compares with the release edge."""
    guards = mode_guards(
        one_pump(
            {
                "name": "E",
                "port": "in",
                "op": ">=",
                "value": 4.0,
                "release": 1.0,
            }
        )
    )
    assert "4.0" in guards["duty_off_to_on"], guards["duty_off_to_on"]
    assert "1.0" not in guards["duty_off_to_on"], guards["duty_off_to_on"]
    assert "1.0" in guards["duty_on_to_off"], guards["duty_on_to_off"]
    assert "4.0" not in guards["duty_on_to_off"], guards["duty_on_to_off"]


def test_without_a_band_both_transitions_read_the_same_threshold():
    """The behaviour before this feature, kept as the control: one
    threshold means the mode is entered and left at the same point, which
    is right whenever nothing in the model pushes back on the read."""
    guards = mode_guards(
        one_pump({"name": "E", "port": "in", "op": ">=", "value": 4.0})
    )
    assert "4.0" in guards["duty_off_to_on"]
    assert "4.0" in guards["duty_on_to_off"]


def test_a_band_does_not_hold_against_a_higher_priority_rule():
    """Rules are ordered, and a band says where its own rule is left, not
    that the mode stops looking at the earlier ones.

    Without this, declaring a band anywhere would quietly disable the
    priority order the rest of the rule set rests on."""

    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E")
            self.add_flow_continuous_in(name="stop")
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[
                    {
                        "name": "halt",
                        "cond": [
                            {"name": "stop", "port": "in", "op": ">=", "value": 7.0}
                        ],
                        "prod": {"W": 0.0},
                    },
                    {
                        "name": "on",
                        "cond": [
                            {
                                "name": "E",
                                "port": "in",
                                "op": ">=",
                                "value": 4.0,
                                "release": 1.0,
                            }
                        ],
                        "prod": {"W": 1.0},
                    },
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    system = mu.System("priority")
    system.add_component(Pump, "P")
    guards = mode_guards(system)
    # Entering the higher-priority rule reads ITS threshold and never the
    # band of the rule being left.
    assert "7.0" in guards["duty_on_to_halt"], guards["duty_on_to_halt"]
    assert "1.0" not in guards["duty_on_to_halt"], guards["duty_on_to_halt"]
    # Falling through to the default still releases at the band's edge.
    assert "1.0" in guards["duty_on_to_off"], guards["duty_on_to_off"]


def test_a_falling_band_holds_the_mode_the_same_way():
    """The mirror case, simulated and not only refused: a rule entered
    when a level falls THROUGH a threshold is left when it rises back
    past the far edge.

    Without this, the whole band would rest on `>=` alone, and the
    complement of `<=` could be wrong in the same direction as the
    original bug with no test noticing."""

    class Tank(mu.ObjFlow):
        """Drains at 1 per unit time from 10, and publishes its level."""

        def add_flows(self):
            # It has to ASK for what the refill offers: a volume that
            # declares no pass-through demand takes nothing, which is
            # the failure the capacity guide warns about.
            self.add_flow_continuous_in(name="W", var_demand_in_default=3.0)
            self.add_flow_continuous_out(name="W", var_fed_default=1.0)
            self.add_capacity(
                name="tank", flow="W", capacity=10.0, content_init={"W": 10.0}
            )

    class Refill(mu.ObjFlow):
        """Starts refilling when the level falls to 4 and does not stop
        until it is back above 8."""

        def add_flows(self):
            self.add_measurement_in(name="tank")
            self.add_flow_continuous_out(name="W", var_fed_default=3.0)
            self.add_rule_set(
                name="duty",
                rules=[
                    {
                        "name": "on",
                        "cond": [
                            {
                                "name": "tank_level",
                                "op": "<=",
                                "value": 4.0,
                                "release": 8.0,
                            }
                        ],
                        "prod": {"W": 3.0},
                    },
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    class Drain(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="W", var_demand_in_default=1.0)

    system = mu.System("refill")
    system.add_component(Tank, "T")
    system.add_component(Refill, "R")
    system.add_component(Drain, "D")
    system.connect_measurement("T", "tank", "R")
    system.connect("R", "W", "T", "W")
    system.connect("T", "W", "D", "W")

    guards = mode_guards(system)
    assert "4.0" in guards["duty_off_to_on"], guards["duty_off_to_on"]
    assert "8.0" in guards["duty_on_to_off"], guards["duty_on_to_off"]

    result = system.simulate(t_max=20.0, samples=[3.0, 7.0, 7.5, 9.0])
    level = dict(result.samples["T_tank_content"])
    delivering = dict(result.samples["R_W_fed_out"])
    # Draining while the level is above the entry threshold.
    assert level[3.0] > 4.0 and delivering[3.0] == 0.0, (level, delivering)
    # Still refilling at 7 and 7.5, where the level has climbed well back
    # past the entry threshold of 4 and has not yet reached the release
    # edge of 8. A single threshold would have stopped the instant the
    # level rose above 4, which is the whole point.
    for instant in (7.0, 7.5):
        assert 4.0 < level[instant] < 8.0, (instant, level)
        assert delivering[instant] > 0.0, (instant, delivering)
    # Past the release edge it stops, and stays stopped while the level
    # falls back through the band.
    assert 4.0 < level[9.0] < 8.0, level
    assert delivering[9.0] == 0.0, (level, delivering)
