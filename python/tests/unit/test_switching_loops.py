"""A guard that reads what its own decision moves, found before any run.

The two budgets of 0.21.x name such a loop after the fact and after the
wait: the run does not fail, it grinds, and the trajectory reads
correctly at every sample instant while it does. This is the same finding
taken from the compiled tables alone, in the time it takes to build the
model.

It **warns and never refuses**, and the distinction is the whole design.
The loop is not the fault: a thermostat is a switching loop, and so is
every controlled tank. What makes one pathological is a switch with no
band, so a loop is reported only when some automaton on it is entered and
left at the same threshold. A loop whose every switch has a band is
silent, because crossing the band costs physical time and the cycle has a
physical period.
"""

import pyraichu
import pyraichu.muscadet as mu


def pump_system(release: float | None = None) -> mu.System:
    """The loop in four components: a pump starts when the supply
    reaching it clears a floor, and starting is exactly what drains the
    reserve that announces the supply."""
    cond: dict[str, object] = {"name": "E", "port": "in", "op": ">=", "value": 4.0}
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
                name="store", flow="E", capacity=100.0, content_init={"E": 5.0}
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


def loops_of(system: mu.System) -> list[dict]:
    return pyraichu.switching_loops(
        pyraichu.load_model(pyraichu.expand_model(system.build_dict()["model"]))
    )


def test_a_single_threshold_loop_is_found_without_simulating():
    """The finding, and the point of the whole module: no `simulate`
    call anywhere in this test."""
    found = loops_of(pump_system())
    assert len(found) == 1, found
    assert found[0]["bandless"] == ["P.duty_mode"], found[0]


def test_the_same_model_with_a_band_is_silent():
    """The control. The dependency is unchanged and still a cycle; what
    changed is that crossing the band costs physical time. A diagnostic
    that fired here would be noise, and noise is what makes a warning
    ignored."""
    assert loops_of(pump_system(release=1.0)) == []


def test_the_message_leads_with_what_to_change():
    """A loop through a flow network passes through dozens of
    attributes. The message sizes the cycle and names the automaton to
    act on; the cycle itself stays available as a field."""
    message = loops_of(pump_system())[0]["message"]
    assert message.startswith("switching loop: P.duty_mode switches"), message
    assert "band" in message, message


def test_the_cycle_is_reported_in_full_beside_the_message():
    """What a reader needs to check the finding rather than trust it."""
    found = loops_of(pump_system())[0]
    assert "P.duty_mode" in found["automata"]
    assert any("E_capability_in" in name for name in found["through"]), found["through"]


def test_a_model_with_no_loop_is_silent():
    """A threshold on a quantity nothing in the model pushes back on is
    exactly the case a single threshold is right for."""

    class Sensor(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="E", var_demand_in_default=1.0)
            self.add_flow_continuous_out(name="W")
            self.add_rule_set(
                name="duty",
                rules=[
                    {
                        "name": "on",
                        "cond": [{"name": "E", "port": "in", "op": ">=", "value": 1.0}],
                        "prod": {"W": 1.0},
                    },
                    {"name": "off", "prod": {"W": 0.0}},
                ],
            )

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="E", var_fed_default=3.0)

    system = mu.System("sensor")
    system.add_component(Source, "S")
    system.add_component(Sensor, "P")
    system.connect("S", "E", "P", "E")
    assert loops_of(system) == []
