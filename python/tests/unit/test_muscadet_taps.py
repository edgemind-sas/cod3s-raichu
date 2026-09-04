"""Caps and taps: the two things a failure mode can do to a continuous
output, and why they are not one mechanism.

A failure-mode effect on a continuous output is declared in one of two
conventions, and the arithmetic follows from which:

- a **cap** says what the mode LEAVES of the output. Simultaneous caps
  fold by **minimum**, the binding constraint winning: a valve stuck at
  0.8 and a pump degraded to 0.9 leave 0.8 between them, not 0.72. This
  is the 1.x reading of a bare number and is unchanged;
- a **tap** says what fraction the mode TAKES OFF the output.
  Simultaneous taps fold by **sum**, being parallel draws on one stream:
  what leaves by one does not pass the other, so leaks of 0.1 and 0.2
  leave 0.7. Folded by minimum they would leave 0.8, and the second leak
  would cost nothing at all.

Neither rule is universally right, which is the whole reason the
declaration carries the kind: read as a cap, a tap of 0.1 leaves a tenth
of the output instead of taking a tenth off it, and no inspection of the
number could tell the two apart.

A tap may also be **routed**, delivering what it takes to another
continuous out-flow of the component. That is what closes the mass
balance: the fraction is moved rather than extinguished, and the two
sides divide one published quantity, what the stream carried before its
taps drew on it.

What these tests pin beyond the arithmetic:

- the demand channel is NOT divided by the tap rate. A plant that does
  not know about its leak produces what it was asked for and delivers
  less; the shortfall is the consumer's, which is what an uncompensated
  leak is. Compensating it is a controller's job, declared as one;
- a routed tap reads its source on what that source actually MADE where
  it publishes one, and on what it could deliver otherwise;
- the refusals that keep a tap conserving: a route that names nothing, a
  route that loops, taps that could together take more than the stream
  carries, and a receiving output that also produces.
"""

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

# --- scaffolding --------------------------------------------------------


def plant(*modes: dict, outputs: tuple[str, ...] = ()) -> type:
    """A source of 10 units of `power`, carrying the declared modes and
    whatever extra outputs a routed tap delivers into."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)
            for name in outputs:
                self.add_flow_continuous_out(name=name)
            for mode in modes:
                self.add_delay_failure_mode(**mode)

    return Plant


def sink(flow: str) -> type:
    """A consumer asking for far more than it will be given."""

    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name=flow, var_demand_in_default=1e4)

    return Sink


def plant_system(*modes: dict, outputs: tuple[str, ...] = ()) -> mu.System:
    system = mu.System("tapped_source")
    system.add_component(plant(*modes, outputs=outputs), "P")
    system.add_component(sink("power"), "G")
    system.connect("P", "power", "G", "power")
    for index, name in enumerate(outputs):
        system.add_component(sink(name), f"S{index}")
        system.connect("P", name, f"S{index}", name)
    return system


def after(name: str, at: float, tap: float, to: str | None = None) -> dict:
    """A mode tapping `tap` off `power` from date `at`, never repaired."""
    effect: dict = {"tap": tap}
    if to is not None:
        effect["to"] = to
    return {
        "name": name,
        "failure_time": at,
        "repair_time": 1e9,
        "failure_effects": [("power", effect)],
    }


# --- the composition that separates the two kinds -----------------------


def test_two_taps_compose_by_sum():
    """Leaks of 0.1 and 0.2 leave 0.7 of the output. A minimum would
    leave 0.8, which makes the second leak free: that is the arithmetic
    this channel exists to avoid."""
    system = plant_system(after("small", 2.0, 0.10), after("large", 4.0, 0.20))
    result = system.simulate(t_max=10.0, samples=[1.0, 3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 10.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 3.0) - 9.0) < CROSSING_TOL
    # A minimum would read 8.0 here, a sum reads 7.0.
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 7.0) < CROSSING_TOL


def test_two_caps_still_compose_by_minimum():
    """The other convention, unchanged: two modes leaving 0.9 and 0.8 of
    one output leave 0.8, the binding constraint winning."""
    system = plant_system(
        {
            "name": "slight",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.9)],
        },
        {
            "name": "worse",
            "failure_time": 4.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.8)],
        },
    )
    result = system.simulate(t_max=10.0, samples=[3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 3.0) - 9.0) < CROSSING_TOL
    # A sum would read 7.0 here, a minimum reads 8.0.
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 8.0) < CROSSING_TOL


def test_a_cap_and_a_tap_compose_by_product():
    """They are separate channels: the cap bounds what the output can
    deliver, the tap takes a fraction off what it does deliver. An
    output capped at 0.5 and tapped by 0.2 delivers 0.4 of nominal."""
    system = plant_system(
        {
            "name": "halved",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.5)],
        },
        after("leaking", 4.0, 0.20),
    )
    result = system.simulate(t_max=10.0, samples=[3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 3.0) - 5.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 4.0) < CROSSING_TOL


def test_an_explicit_cap_is_the_bare_number():
    """`{"cap": 0.4}` and `0.4` are one declaration in two spellings, so
    they generate the same document."""
    bare = plant_system(
        {
            "name": "degrade",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", 0.4)],
        }
    )
    spelled = plant_system(
        {
            "name": "degrade",
            "failure_time": 2.0,
            "repair_time": 1e9,
            "failure_effects": [("power", {"cap": 0.4})],
        }
    )
    assert bare.build_dict() == spelled.build_dict()


def test_leaving_the_failing_state_closes_the_tap():
    """The release is implicit on this channel too: the mode declares a
    tap on one state only, and repairing it restores the whole output
    with nothing declared on the other."""
    system = plant_system(
        {
            "name": "leak",
            "failure_time": 2.0,
            "repair_time": 2.0,
            "failure_effects": [("power", {"tap": 0.3})],
        }
    )
    result = system.simulate(t_max=10.0, samples=[1.0, 3.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 10.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 3.0) - 7.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 10.0) < CROSSING_TOL


def test_a_repair_effect_leaves_the_tap_partly_open():
    """A mode returning degraded rather than as-new is a legitimate
    model: the repair side keeps a tap of its own."""
    system = plant_system(
        {
            "name": "leak",
            "failure_time": 2.0,
            "repair_time": 2.0,
            "failure_effects": [("power", {"tap": 0.3})],
            "repair_effects": [("power", {"tap": 0.1})],
        }
    )
    result = system.simulate(t_max=10.0, samples=[1.0, 3.0, 5.0])

    # Before the first failure the repair value already stands: the mode
    # starts in its nominal state, and that state is what it declares.
    assert abs(sampled(result, "G_power_fed_in", 1.0) - 9.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 3.0) - 7.0) < CROSSING_TOL
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 9.0) < CROSSING_TOL


# --- routing: what a tap takes has to arrive somewhere ------------------


def test_a_routed_tap_delivers_exactly_what_it_takes():
    """The fraction is moved, not extinguished: what the leak port
    carries and what the main port still carries add back up to what the
    stream carried before the tap drew."""
    system = plant_system(after("leak", 2.0, 0.25, to="vent"), outputs=("vent",))
    result = system.simulate(t_max=10.0, samples=[1.0, 5.0])

    assert abs(sampled(result, "G_power_fed_in", 1.0) - 10.0) < CROSSING_TOL
    assert abs(sampled(result, "S0_vent_fed_in", 1.0)) < CROSSING_TOL

    delivered = sampled(result, "G_power_fed_in", 5.0)
    vented = sampled(result, "S0_vent_fed_in", 5.0)
    carried = sampled(result, "P_power_pre_tap_capability_out", 5.0)
    assert abs(delivered - 7.5) < CROSSING_TOL
    assert abs(vented - 2.5) < CROSSING_TOL
    assert abs(delivered + vented - carried) < CROSSING_TOL


def test_two_routed_taps_split_one_stream_three_ways():
    """Parallel draws on one stream, each delivering into its own port:
    0.1 and 0.2 leave 0.7, and the three quantities sum to the whole."""
    system = plant_system(
        after("membrane", 2.0, 0.10, to="vent_a"),
        after("local", 2.0, 0.20, to="vent_b"),
        outputs=("vent_a", "vent_b"),
    )
    result = system.simulate(t_max=10.0, samples=[5.0])

    delivered = sampled(result, "G_power_fed_in", 5.0)
    first = sampled(result, "S0_vent_a_fed_in", 5.0)
    second = sampled(result, "S1_vent_b_fed_in", 5.0)
    assert abs(delivered - 7.0) < CROSSING_TOL
    assert abs(first - 1.0) < CROSSING_TOL
    assert abs(second - 2.0) < CROSSING_TOL
    assert abs(delivered + first + second - 10.0) < CROSSING_TOL


def test_an_unrouted_tap_takes_the_fraction_out_of_the_system():
    """Without `to` the fraction leaves: a loss to the environment is a
    legitimate model, and it is what a tap does when nothing says where
    what it takes goes."""
    system = plant_system(after("leak", 2.0, 0.25))
    result = system.simulate(t_max=10.0, samples=[5.0])

    assert abs(sampled(result, "G_power_fed_in", 5.0) - 7.5) < CROSSING_TOL
    assert abs(sampled(result, "P_power_pre_tap_capability_out", 5.0) - 10.0) < (
        CROSSING_TOL
    )


# --- routing off a stream that publishes what it made -------------------


def tapped_reaction(demand: float) -> mu.System:
    """A reactor turning `a` into `x` one for one, leaking a quarter of
    what it makes into `drain`, feeding a consumer that asks for
    `demand`."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="a", var_fed_default=100.0)

    class Reactor(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="a")
            self.add_flow_continuous_out(name="x")
            self.add_flow_continuous_out(name="drain")
            self.add_rule_set(
                name="convert", rules=[{"cons": {"a": 1}, "prod": {"x": 1}}]
            )
            self.add_delay_failure_mode(
                name="leak",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("x", {"tap": 0.25, "to": "drain"})],
            )

    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="x", var_demand_in_default=demand)

    system = mu.System("tapped_reaction")
    system.add_component(Source, "A")
    system.add_component(Reactor, "R")
    system.add_component(Consumer, "K")
    system.add_component(sink("drain"), "D")
    system.connect("A", "a", "R", "a")
    system.connect("R", "x", "K", "x")
    system.connect("R", "drain", "D", "drain")
    return system


def test_a_tap_on_a_produced_stream_scales_with_what_was_made():
    """The reactor is held to a demand of 4 and makes 4, of which the
    leak takes a quarter. The leak follows the production, not the 100
    the reactor could have made had it been asked."""
    result = tapped_reaction(demand=4.0).simulate(t_max=10.0, samples=[1.0, 5.0])

    assert abs(sampled(result, "K_x_fed_in", 1.0) - 4.0) < CROSSING_TOL
    assert abs(sampled(result, "D_drain_fed_in", 1.0)) < CROSSING_TOL

    delivered = sampled(result, "K_x_fed_in", 5.0)
    drained = sampled(result, "D_drain_fed_in", 5.0)
    assert abs(delivered - 3.0) < CROSSING_TOL
    assert abs(drained - 1.0) < CROSSING_TOL
    assert abs(delivered + drained - 4.0) < CROSSING_TOL


def test_an_uncompensated_leak_is_a_shortfall_at_the_consumer():
    """The demand channel is not divided by the tap rate: the reactor was
    asked for 4, made 4, and the consumer receives 3. A plant that made
    5.33 to cover its own leak would be a regulated one, and regulation
    is declared as a controller, not inferred from a leak."""
    result = tapped_reaction(demand=4.0).simulate(t_max=10.0, samples=[5.0])

    assert abs(sampled(result, "R_x_produced_out", 5.0) - 3.0) < CROSSING_TOL
    assert abs(sampled(result, "R_x_pre_tap_produced_out", 5.0) - 4.0) < CROSSING_TOL
    assert abs(sampled(result, "K_x_fed_in", 5.0) - 3.0) < CROSSING_TOL


def test_a_chain_of_taps_settles_whatever_the_declaration_order():
    """A tap into an output that is itself tapped, with the receivers
    declared BEFORE the stream they are fed from.

    The sweep order follows the routing and not the declaration: read in
    declaration order, the middle output would settle on a base its
    source had not written yet, and the chain would carry a stale
    quantity with nothing to signal it."""

    class Plant(mu.ObjFlow):
        def add_flows(self):
            # Deliberately the reverse of the routing: end, middle, source.
            self.add_flow_continuous_out(name="vent_b")
            self.add_flow_continuous_out(name="vent_a")
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)
            self.add_delay_failure_mode(
                name="first",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("power", {"tap": 0.5, "to": "vent_a"})],
            )
            self.add_delay_failure_mode(
                name="second",
                failure_time=2.0,
                repair_time=1e9,
                failure_effects=[("vent_a", {"tap": 0.5, "to": "vent_b"})],
            )

    system = mu.System("chained")
    system.add_component(Plant, "P")
    for flow, name in (("power", "G"), ("vent_a", "A"), ("vent_b", "B")):
        system.add_component(sink(flow), name)
        system.connect("P", flow, name, flow)
    result = system.simulate(t_max=10.0, samples=[5.0])

    delivered = sampled(result, "G_power_fed_in", 5.0)
    middle = sampled(result, "A_vent_a_fed_in", 5.0)
    end = sampled(result, "B_vent_b_fed_in", 5.0)
    assert abs(delivered - 5.0) < CROSSING_TOL
    assert abs(middle - 2.5) < CROSSING_TOL
    assert abs(end - 2.5) < CROSSING_TOL
    assert abs(delivered + middle + end - 10.0) < CROSSING_TOL

    # The quantities above do NOT pin the order: sampled at a steady
    # state, a base read one sweep late carries the same number. What
    # pins it is the emitted sweep itself, so this is asserted on the
    # order and not inferred from a value that cannot see it.
    order = [
        entry["attribute"]
        for entry in system.build_dict()["model"]["evaluation_order"]
        if entry["component"] == "P" and entry["attribute"].endswith(
            "_pre_tap_capability_out"
        )
    ]
    assert order == [
        "power_pre_tap_capability_out",
        "vent_a_pre_tap_capability_out",
    ]


# --- the refusals that keep a tap conserving ----------------------------


def build(*modes: dict, outputs: tuple[str, ...] = ()) -> None:
    plant_system(*modes, outputs=outputs).build_dict()


def test_an_effect_declaring_both_kinds_is_refused():
    with pytest.raises(ValueError, match="both `cap` and `tap`"):
        plant(
            {
                "name": "m",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_effects": [("power", {"cap": 0.5, "tap": 0.5})],
            }
        )("P")


def test_an_effect_declaring_neither_kind_is_refused():
    with pytest.raises(ValueError, match="neither `cap` and `tap`"):
        plant(
            {
                "name": "m",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_effects": [("power", {"to": "vent"})],
            }
        )("P")


def test_a_cap_may_not_carry_a_route():
    with pytest.raises(ValueError, match="cap bounds what the output delivers"):
        plant(
            {
                "name": "m",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_effects": [("power", {"cap": 0.5, "to": "vent"})],
            },
            outputs=("vent",),
        )("P")


def test_an_unknown_effect_key_is_refused():
    with pytest.raises(ValueError, match="unknown key"):
        plant(
            {
                "name": "m",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_effects": [("power", {"tap": 0.5, "into": "vent"})],
            }
        )("P")


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_a_tap_outside_the_unit_interval_is_refused(fraction):
    with pytest.raises(ValueError, match="TAKES OFF"):
        plant(
            {
                "name": "m",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_effects": [("power", {"tap": fraction})],
            }
        )("P")


def test_a_route_naming_no_output_of_the_component_is_refused():
    with pytest.raises(ValueError, match="not a continuous out-flow"):
        build(after("leak", 1.0, 0.5, to="elsewhere"))


def test_a_route_naming_the_tapped_flow_itself_is_refused():
    with pytest.raises(ValueError, match="taps `power` to itself"):
        build(after("leak", 1.0, 0.5, to="power"))


def test_two_sides_routing_to_different_outputs_are_refused():
    with pytest.raises(ValueError, match="one destination on each side"):
        build(
            {
                "name": "leak",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_effects": [("power", {"tap": 0.5, "to": "vent_a"})],
                "repair_effects": [("power", {"tap": 0.1, "to": "vent_b"})],
            },
            outputs=("vent_a", "vent_b"),
        )


def test_taps_that_could_take_more_than_the_stream_carries_are_refused():
    """Bounded at build time and over all the modes at once, which is
    what lets the tap rate be written without a clamp."""
    with pytest.raises(ValueError, match="can together take 1.1 of it"):
        build(after("first", 1.0, 0.6), after("second", 2.0, 0.5))


def test_a_receiving_output_that_also_produces_is_refused():
    """It would carry what it was handed on top of what it makes, and the
    fraction would be counted twice."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)
            self.add_flow_continuous_out(name="vent", var_fed_default=1.0)
            self.add_delay_failure_mode(
                name="leak",
                failure_time=1.0,
                repair_time=1e9,
                failure_effects=[("power", {"tap": 0.5, "to": "vent"})],
            )

    system = mu.System("wrong")
    system.add_component(Wrong, "P")
    with pytest.raises(ValueError, match="declared at a rate of 1"):
        system.build_dict()


def test_a_routing_cycle_is_refused():
    """Two receiving outputs feeding each other: neither produces, so the
    previous refusal lets this through and only the cycle catches it.
    There is no sweep order that settles both."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="power", var_fed_default=10.0)
            self.add_flow_continuous_out(name="vent_a")
            self.add_flow_continuous_out(name="vent_b")
            self.add_delay_failure_mode(
                name="seed",
                failure_time=1.0,
                repair_time=1e9,
                failure_effects=[("power", {"tap": 0.5, "to": "vent_a"})],
            )
            self.add_delay_failure_mode(
                name="out",
                failure_time=1.0,
                repair_time=1e9,
                failure_effects=[("vent_a", {"tap": 0.5, "to": "vent_b"})],
            )
            self.add_delay_failure_mode(
                name="back",
                failure_time=1.0,
                repair_time=1e9,
                failure_effects=[("vent_b", {"tap": 0.5, "to": "vent_a"})],
            )

    system = mu.System("wrong")
    system.add_component(Wrong, "P")
    with pytest.raises(ValueError, match="route in a cycle"):
        system.build_dict()


def test_a_routed_tap_naming_two_outputs_is_refused():
    """A routed tap moves matter out of one stream, so its pattern names
    a single output; an alternation would merge two streams into one
    port without saying so."""

    class Wrong(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2", var_fed_default=10.0)
            self.add_flow_continuous_out(name="O2", var_fed_default=10.0)
            self.add_flow_continuous_out(name="vent")
            self.add_delay_failure_mode(
                name="leak",
                failure_time=1.0,
                repair_time=1e9,
                failure_effects=[("(H2|O2)", {"tap": 0.5, "to": "vent"})],
            )

    with pytest.raises(ValueError, match="names a single output"):
        Wrong("P")


# --- the same declaration as plugin data --------------------------------


def test_the_plugin_carries_a_tap_to_the_authoring_layer():
    """A mode declared in plugin data taps exactly as the same mode
    declared in Python: same component document, same split.

    The plugin's own `failure_modes` vocabulary carried no effects at
    all before this, so a leak declared in data built, ran to completion,
    and reported the figures of a plant whose modelled leak never
    happened."""
    import pyraichu

    spec = {
        "name": "tapped_source",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "P",
                        "flows_continuous_out": [
                            {"name": "power", "var_fed_default": 10.0},
                            {"name": "vent"},
                        ],
                        "failure_modes": [
                            {
                                "name": "leak",
                                "distrib": "delay",
                                "failure": 2.0,
                                "repair": 1e9,
                                "failure_effects": {
                                    "power": {"tap": 0.25, "to": "vent"}
                                },
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "G",
                        "flows_continuous_in": [
                            {"name": "power", "var_demand_default": 1e4}
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "S0",
                        "flows_continuous_in": [
                            {"name": "vent", "var_demand_default": 1e4}
                        ],
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {
                "from": {"component": "P", "port": "power_out"},
                "to": {"component": "G", "port": "power_in"},
            },
            {
                "from": {"component": "P", "port": "vent_out"},
                "to": {"component": "S0", "port": "vent_in"},
            },
        ],
        "indicators": [],
    }
    body = pyraichu.expand_model(spec)
    assert "plugins" not in body

    authored = plant_system(
        after("leak", 2.0, 0.25, to="vent"), outputs=("vent",)
    ).build_dict()
    from_plugin = next(c for c in body["components"] if c["name"] == "P")
    from_python = next(
        c for c in authored["model"]["components"] if c["name"] == "P"
    )
    assert from_plugin == from_python

    result = pyraichu.simulate(
        pyraichu.load_model(body), t_max=10.0, samples=[5.0]
    )
    assert abs(sampled(result, "G_power_fed_in", 5.0) - 7.5) < CROSSING_TOL
    assert abs(sampled(result, "S0_vent_fed_in", 5.0) - 2.5) < CROSSING_TOL
