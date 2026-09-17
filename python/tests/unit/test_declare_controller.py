"""The THIRD shape a component declaration takes: a controller.

``muscadet.ObjCtrl`` is a PEER of ``ObjFlow`` and not a subclass of it: a flow
transports a conserved quantity, a controller transports a reading or a signal,
and nothing is allocated. It is therefore neither of the two shapes this reader
already knew, and a document carrying one was refused at the door -- which is
not a loss of decoration but a loss of MODEL: a controller is what commands a
changeover, so the study that reached the engine was one where nothing ever
switches, and its reliability figure was wrong and looked right.

What this suite pins:

- **a document carrying a controller reads and RUNS**, the emission grammar of
  every output compiled to something that fires at the level the declaration
  names. Every date asserted is an analytic crossing instant, never the declared
  nature of a transition: a structural assertion passes on a model that never
  fires at all;
- **the three numbers muscadet and this layer spell differently** are carried,
  and each is a number a modeller tuned: an observation input's seed (one field
  per nature there, one ``default`` here), a value output's seed, and a
  republication's gain;
- **the threshold variables are named as R44 names them**
  (``run_threshold``, ``alarm_operand_1_activate``), because that is what an
  indicator observes and what a failure mode moves;
- **a measurement link is wired by the RAW route**, which is the whole of the
  second half of this ticket: muscadet stopped writing a ``flow`` key on a pair
  that merely follows the flow convention, so a level link now arrives with
  nothing but its two box names and must not be read as a flow connection;
- **what has no counterpart is refused BY NAME**, never as an unknown key and
  never as a value silently dropped;
- **a document carrying no controller reads exactly as it did.**

Oracle-free, like the rest of this directory. That muscadet's own controller
montage reads here and lands on the same dates is measured against the real
export, outside the suite; what is written out below is that export's shape,
field for field.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.declare as declare
from conftest import CROSSING_TOL, settled

# --- the reference document -------------------------------------------
#
# Written out rather than generated, field for field as
# `muscadet.declare.system_spec` produces it: a test that built it from
# muscadet would need muscadet installed to say anything at all.

#: The tank starts empty and is filled at one per unit time, so its level reads
#: `t` and every threshold below is also the date it is crossed at.
FILL_RATE = 1.0

#: What the instance below starts at. muscadet's own montage tunes three
#: instances of one class to three different levels; here the point is that the
#: number the DOCUMENT carries is the number the engine runs, so one is enough.
START = 2.0

#: The nested band's two edges, at a date of their own so a crossing cannot be
#: confused with the comparison's.
ALARM_ACTIVATE = 3.0
ALARM_RELEASE = 0.5

#: Past the horizon on purpose: the operand that never fires, so what fires in
#: the disjunction below is the band and nothing else.
UNREACHABLE = 100.0


def a_filler(name="FILL"):
    return {
        "name": name,
        "cls": "ObjFlow",
        "source_cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowContinuousOut",
                "name": "q",
                "var_type": "float",
                "var_fed_default": FILL_RATE,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_demand_in_default": 0.0,
                "allocation": "proportional",
                "allocation_shares": {},
                "allocation_priorities": {},
            }
        ],
        "capacities": [],
        "measurements_in": [],
        "rules": [],
        "transfers": [],
        "failure_modes": [],
    }


def a_tank(name="TANK", flows=("q",)):
    """A volume filled by whatever reaches it, publishing its level.

    Holding more than one constituent is what makes it publish a ratio per
    constituent, which is the only way a closed grammar can threshold a
    fraction: there is no division among the four operators.
    """
    return {
        "name": name,
        "cls": "ObjFlow",
        "source_cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowContinuousIn",
                "name": flow,
                "var_type": "float",
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": 0.0,
                "var_demand_default": 0.0,
            }
            for flow in flows
        ],
        "capacities": [
            {
                "name": "level",
                "flows": list(flows),
                "capacity": 1000.0,
                "content_init": {flow: 0.0 for flow in flows},
                "fill_rate": "inf",
            }
        ],
        "measurements_in": [],
        "rules": [],
        "transfers": [],
        "failure_modes": [],
    }


def a_gate(name="GATE"):
    """Something a controller commands: one discrete control port."""
    return {
        "name": name,
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowIn",
                "name": "run",
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": False,
                "var_available_in_default": True,
                "logic": "and",
            }
        ],
        "capacities": [],
        "failure_modes": [],
    }


def a_channel(name, kind="level", flows=(), aggregate=None):
    """One observation input, field for field as muscadet writes one.

    The four seeds are always written, whatever the nature: the one that says
    something for the declared nature is the reading's, and the other three are
    muscadet's own defaults.
    """
    return {
        "name": name,
        "kind": kind,
        "flows": list(flows),
        "level_default": 0.0,
        "fill_default": 0.0,
        "rate_default": 0.0,
        "ratio_default": 0.0,
        "aggregate": aggregate,
    }


def a_signal(name, emit=None, default=False):
    """One BOOLEAN output: muscadet's vocabulary is this layer's, key for key."""
    entry = {"name": name, "kind": "bool", "default": default}
    if emit is not None:
        entry["emit"] = emit
    return entry


def a_publication(name, emit=None, **overrides):
    """One VALUE output, as muscadet writes one: a measurement publication,
    so it carries the publication's own fields and NO ``default``.

    ``gain_default`` is written only when nothing computes the output: a
    republication's gain rides in the operator that publishes through it, and
    muscadet refuses a declaration carrying both.
    """
    entry = {
        "name": name,
        "kind": "value",
        "flows": [],
        "level_default": 0.0,
        "fill_default": 0.0,
        "ratio_default": 0.0,
    }
    if emit is not None:
        entry["emit"] = emit
    else:
        entry["gain_default"] = 1.0
    entry.update(overrides)
    return entry


def a_pump(name="PUMP", start=START, controls_in=None, controls_out=None):
    """The controller muscadet's own threshold-override montage declares: a
    level in, a comparison out, and an alarm combining an unreachable
    comparison with a band.

    The thresholds are the INSTANCE's, not the class's: the platform importer
    folds an instance's tuning into the emission grammar at the parse layer, so
    what the document carries is already the tuned value. A document carrying
    the class value would be a reliability figure that is wrong and looks right.
    """
    return {
        "name": name,
        "kind": "controller",
        "cls": "ObjCtrl",
        "source_cls": "ObjCtrl",
        "controls_in": controls_in
        if controls_in is not None
        else [a_channel("level")],
        "controls_out": controls_out
        if controls_out is not None
        else [
            a_signal(
                "run",
                {
                    "op": "compare",
                    "input": "level",
                    "operator": ">=",
                    "threshold": start,
                },
            ),
            a_signal(
                "alarm",
                {
                    "op": "combine",
                    "logic": "or",
                    "operands": [
                        {
                            "op": "compare",
                            "input": "level",
                            "operator": ">=",
                            "threshold": UNREACHABLE,
                        },
                        {
                            "op": "band",
                            "input": "level",
                            "direction": "above",
                            "activate": ALARM_ACTIVATE,
                            "release": ALARM_RELEASE,
                        },
                    ],
                },
            ),
        ],
        "metadata": {"controller": True},
    }


def a_measurement(source, capacity, target, channel, flow=None):
    """A measurement link, as muscadet writes one SINCE e9145d67: the two
    message boxes and no ``flow`` key.

    The key is what tells a pair that really names a flow on both ends from one
    that merely follows the convention, and a level link is the second: a
    channel called ``tank`` used to come out as ``flow: "tank_level"``, a flow
    nothing declares. `flow` is a parameter here only so the two spellings can
    be compared, which is what pins that this reader is indifferent to it.
    """
    entry = {
        "source": source,
        "source_box": f"{capacity}_level_out",
        "target": target,
        "target_box": f"{channel}_level_in",
    }
    if flow is not None:
        entry["flow"] = flow
    return entry


def a_document(*components, connections):
    return {
        "version": "1.0.0",
        "name": "controlled",
        # Every trajectory below reads an observation nobody declared, so the
        # document asks for the generated set: what a declaration does not
        # ask for, no route emits.
        "generated_indicators": True,
        "components": {entry["name"]: entry for entry in components},
        "connections": list(connections),
    }


def a_filled_tank(pump=None, extra=()):
    """The montage every trajectory below runs on: a tank filled at one per
    unit time, observed by one controller commanding one gate."""
    pump = pump if pump is not None else a_pump()
    return a_document(
        a_filler(),
        a_tank(),
        a_gate(),
        pump,
        *extra,
        connections=[
            {"source": "FILL", "source_box": "q_out", "target": "TANK",
             "target_box": "q_in", "flow": "q"},
            a_measurement("TANK", "level", pump["name"], "level"),
            {"source": pump["name"], "source_box": "run_out",
             "target": "GATE", "target_box": "run_in"},
        ],
    )


def body_of(document):
    return pyraichu.model_body(declare.build_document(document))


def component_of(document, name):
    body = body_of(document)
    return next(entry for entry in body["components"] if entry["name"] == name)


def trajectory(document, t_max=12.0):
    """The dates the built document's transitions fire at."""
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    return [
        (event.time, event.transition.rsplit(".", 1)[-1])
        for event in pyraichu.simulate(model, t_max=t_max).events
    ]


def crossings(document, transition, t_max=12.0):
    return [time for time, fired in trajectory(document, t_max) if fired == transition]


# --- the whole ticket, in one trajectory -------------------------------


def test_a_controller_is_read_and_its_grammar_fires_at_the_declared_levels():
    """The entry that used to be refused for its `kind` alone now compiles to
    automata that fire, and they fire at the levels the declaration names.

    On a unit ramp the level reads `t`, so the comparison's 2 is crossed at
    t = 2 and the nested band's 3 at t = 3. The dates are the assertion:
    asserting that the transitions came out `watched` would pass on a montage
    where the crossing is never located.
    """
    fired = trajectory(a_filled_tank())
    dated = {name: time for time, name in fired}
    assert "run_compare_up" in dated, fired
    assert abs(dated["run_compare_up"] - START) < CROSSING_TOL, fired
    assert abs(dated["alarm_operand_1_band_up"] - ALARM_ACTIVATE) < CROSSING_TOL, fired


def test_the_command_reaches_the_component_it_commands():
    """A boolean output exports the very box a discrete in-flow imports, so the
    order arrives with no adapter. What is asserted is the GATE's own reading:
    a controller whose signal went nowhere would switch on time and command
    nothing."""
    document = a_filled_tank()
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=12.0)
    switched = settled(result.indicators["GATE_run_fed_in"])
    assert switched[0][1] is False, switched
    turned_on = next(time for time, value in switched if value is True)
    assert abs(turned_on - START) < CROSSING_TOL, switched


def test_the_threshold_variables_are_named_as_a_mode_and_an_indicator_name_them():
    """R44: every number the grammar carries is an attribute of the model, named
    `{output}{path}_{edge}` from the node's position in the output's tree. It is
    what an indicator observes and what a failure mode MOVES, so the spelling is
    a correspondence to honour and not an internal detail."""
    pump = component_of(a_filled_tank(), "PUMP")
    seeded = {
        entry["name"]: entry["init"]["value"] for entry in pump["attributes"]
    }
    assert seeded["run_threshold"] == START
    assert seeded["alarm_operand_0_threshold"] == UNREACHABLE
    assert seeded["alarm_operand_1_activate"] == ALARM_ACTIVATE
    assert seeded["alarm_operand_1_release"] == ALARM_RELEASE
    # The endpoints a mode reaches, each an ordinary attribute it names exactly.
    assert "run_signal_available" in seeded
    assert "alarm_signal_available" in seeded


# --- the two sections, key by key --------------------------------------


def test_the_translation_is_reachable_without_building_anything():
    """A caller comparing what the two engines were handed reads the object,
    not a trajectory."""
    obj = declare.controller_object(a_pump())
    assert obj["type"] == "ObjCtrl"
    assert obj["name"] == "PUMP"
    assert obj["controls_in"] == [
        {"name": "level", "kind": "level", "flows": [], "aggregate": None,
         "default": 0.0}
    ]
    assert [entry["name"] for entry in obj["controls_out"]] == ["run", "alarm"]
    # The grammar passes VERBATIM: it is muscadet's own, operator for operator.
    assert obj["controls_out"][0]["emit"] == a_pump()["controls_out"][0]["emit"]


@pytest.mark.parametrize(
    "kind, seed, flows",
    [("level", "level_default", ()), ("rate", "rate_default", ()),
     ("ratio", "ratio_default", ("q",))],
)
def test_an_input_seed_is_read_from_the_field_of_its_own_nature(kind, seed, flows):
    """muscadet writes one seed field per nature and this layer reads one
    `default`, the nature being already declared by `kind`. Read from the wrong
    field, a reading starts somewhere the document did not put it."""
    entry = a_channel("obs", kind=kind, flows=flows)
    entry[seed] = 7.5
    obj = declare.controller_object(a_pump(controls_in=[entry]))
    assert obj["controls_in"][0]["default"] == 7.5


@pytest.mark.parametrize(
    "seed", ["level_default", "fill_default", "rate_default", "ratio_default"]
)
def test_a_seed_of_another_nature_is_refused_rather_than_dropped(seed):
    """The other three say nothing at muscadet's own default and are refused
    above it: a seed silently dropped is the same defect as one read from the
    wrong field, without even a name to look for."""
    entry = a_channel("obs", kind="rate")
    if seed == "rate_default":
        pytest.skip("the seed of the declared nature, which is carried")
    entry[seed] = 7.5
    with pytest.raises(declare.ComponentSpecError, match=seed):
        declare.check_controller_spec(a_pump(controls_in=[entry]))


def test_a_value_output_seeds_from_level_default_and_a_boolean_one_from_default():
    """muscadet declares a value output as a measurement PUBLICATION, so it
    writes the publication's own `level_default` and no `default` at all."""
    obj = declare.controller_object(
        a_pump(
            controls_out=[
                a_publication(
                    "mirror",
                    {"op": "republish", "input": "level", "gain": 2.0},
                    level_default=4.0,
                ),
                a_signal("cmd", default=True),
            ]
        )
    )
    published, signal = obj["controls_out"]
    assert published == {
        "name": "mirror", "kind": "value", "default": 4.0,
        "emit": {"op": "republish", "input": "level", "gain": 2.0},
    }
    assert signal == {"name": "cmd", "kind": "bool", "default": True}


def test_a_republication_gain_is_read_from_the_operator_that_publishes_through_it():
    """One number, one spelling. A gain is the initial value of
    `{name}_level_gain`, so a document declaring it twice would leave a modeller
    with two places to change it and one of them ignored -- which is why
    muscadet leaves `gain_default` out of a republishing output, and why this
    reads the operator's own `gain`."""
    document = a_filled_tank(
        a_pump(
            controls_out=[
                a_publication(
                    "mirror", {"op": "republish", "input": "level", "gain": 2.0}
                ),
                a_signal(
                    "run",
                    {"op": "compare", "input": "level", "operator": ">=",
                     "threshold": START},
                ),
            ]
        )
    )
    pump = component_of(document, "PUMP")
    seeded = {entry["name"]: entry["init"]["value"] for entry in pump["attributes"]}
    assert seeded["mirror_level_gain"] == 2.0
    # And the endpoints a mode reaches on a value output, which a boolean one
    # has no need of: a number has no rest value one flag could stand for.
    assert seeded["mirror_forced"] is False
    assert seeded["mirror_forced_value"] == 0.0


def test_a_gain_on_an_output_nothing_computes_is_refused_by_name():
    """The one place the two layers really differ: an uncomputed output keeps
    the number it was created with and carries no gain at all, so a declared one
    would be dropped. Accepted at muscadet's own default, where it says
    nothing."""
    declare.check_controller_spec(
        a_pump(controls_out=[a_publication("free")])
    )
    with pytest.raises(declare.ComponentSpecError, match="gain_default"):
        declare.check_controller_spec(
            a_pump(controls_out=[a_publication("free", gain_default=5.0)])
        )


def test_a_per_constituent_publication_is_refused_by_name():
    declare.check_controller_spec(a_pump(controls_out=[a_publication("free")]))
    with pytest.raises(declare.ComponentSpecError, match="constituents"):
        declare.check_controller_spec(
            a_pump(controls_out=[a_publication("free", flows=["q"])])
        )


# --- the four operators, each compiled to something that fires ---------


def test_a_band_holds_between_its_two_edges_where_a_comparison_does_not():
    """What the band buys, and the capability the whole controller port exists
    for: a mode transition's guard reads only the target rule, so a mode holds
    no memory and switches back the instant its condition stops holding. The
    band's two edges are two levels, and between them the output keeps what the
    last crossing left it."""
    document = a_filled_tank(
        a_pump(
            controls_out=[
                a_signal(
                    "run",
                    {"op": "band", "input": "level", "direction": "above",
                     "activate": 4.0, "release": 1.0},
                )
            ]
        )
    )
    fired = crossings(document, "run_band_up")
    assert len(fired) == 1 and abs(fired[0] - 4.0) < CROSSING_TOL, fired
    # Nothing happens strictly after the activation: the level goes on rising,
    # and a band does not chatter around its edge.
    assert not [
        time
        for time, name in trajectory(document)
        if name.startswith("run_band") and time > 4.0 + CROSSING_TOL
    ]


def test_an_inverted_band_is_refused_before_anything_is_built():
    """A band detecting above 3 and releasing at 5 can never release, because
    the reading has to fall to 5 while the band is what stops it falling: the
    montage latches on its first activation and never speaks again."""
    with pytest.raises(declare.ComponentSpecError, match="releases"):
        declare.check_controller_spec(
            a_pump(
                controls_out=[
                    a_signal(
                        "run",
                        {"op": "band", "input": "level", "direction": "above",
                         "activate": 3.0, "release": 5.0},
                    )
                ]
            )
        )


def test_a_combination_votes_over_its_operands():
    """k-of-n, the shape a redundant instrument set exists for: two of the three
    comparisons have to hold, so the output switches on the SECOND crossing and
    not the first."""
    document = a_filled_tank(
        a_pump(
            controls_out=[
                a_signal(
                    "run",
                    {
                        "op": "combine",
                        "logic": "k",
                        "k": 2,
                        "operands": [
                            {"op": "compare", "input": "level",
                             "operator": ">=", "threshold": level}
                            for level in (2.0, 5.0, 9.0)
                        ],
                    },
                )
            ]
        )
    )
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=12.0)
    switched = settled(result.indicators["PUMP_run"])
    turned_on = next(time for time, value in switched if value is True)
    assert abs(turned_on - 5.0) < CROSSING_TOL, switched


def test_an_unknown_operator_is_refused_naming_the_closed_list():
    with pytest.raises(declare.ComponentSpecError, match="unknown operator"):
        declare.check_controller_spec(
            a_pump(
                controls_out=[
                    a_signal("run", {"op": "integrate", "input": "level"})
                ]
            )
        )


def test_a_python_callable_is_refused_whatever_continuity_it_attests():
    """The property the closed grammar exists for: nothing can read a threshold
    out of arbitrary Python, so nothing could compile one to a watched
    transition, and the fallback would be a sensitive function on a reading,
    which never re-evaluates between discrete events."""
    with pytest.raises(declare.ComponentSpecError):
        declare.check_controller_spec(
            a_pump(controls_out=[a_signal("run", {"op": "compare", "input": "level",
                                                  "operator": ">=",
                                                  "threshold": lambda t: t})])
        )


def test_an_equality_has_no_counterpart_and_is_refused_by_name():
    """A controller reads a continuously-evolving quantity, and an equality on
    one brackets no crossing for the engine to locate."""
    with pytest.raises(declare.ComponentSpecError, match="equality"):
        declare.check_controller_spec(
            a_pump(
                controls_out=[
                    a_signal("run", {"op": "compare", "input": "level",
                                     "operator": "==", "threshold": 2.0})
                ]
            )
        )


# --- a value output read by a second controller ------------------------


def test_one_controller_reads_what_another_publishes():
    """What makes a chain possible: a value output publishes on
    `{name}_level_out`, indistinguishable from a capacity's own publication by
    whoever observes it. The doubled reading crosses the second controller's
    threshold at half the level, which is the measurement that the gain and the
    link both reached the engine."""
    mirror = a_pump(
        name="METER",
        controls_in=[a_channel("level")],
        controls_out=[
            a_publication("twice", {"op": "republish", "input": "level", "gain": 2.0})
        ],
    )
    vote = a_pump(
        name="VOTE",
        controls_in=[a_channel("reading")],
        controls_out=[
            a_signal(
                "run",
                {"op": "compare", "input": "reading", "operator": ">=",
                 "threshold": 6.0},
            )
        ],
    )
    document = a_document(
        a_filler(),
        a_tank(),
        a_gate(),
        mirror,
        vote,
        connections=[
            {"source": "FILL", "source_box": "q_out", "target": "TANK",
             "target_box": "q_in", "flow": "q"},
            a_measurement("TANK", "level", "METER", "level"),
            {"source": "METER", "source_box": "twice_level_out",
             "target": "VOTE", "target_box": "reading_level_in"},
            {"source": "VOTE", "source_box": "run_out",
             "target": "GATE", "target_box": "run_in"},
        ],
    )
    fired = crossings(document, "run_compare_up")
    assert len(fired) == 1 and abs(fired[0] - 3.0) < CROSSING_TOL, fired


# --- the wiring: the RAW route, and nothing read from `flow` -----------


def test_a_measurement_link_reaches_the_port_the_controller_holds():
    """muscadet's message box carries several variables and RAICHU connects
    attribute to attribute, so the anchor the document writes has to resolve to
    the ONE variable the input reads."""
    body = body_of(a_filled_tank())
    reaching = [
        (entry["from"]["port"], entry["to"]["port"])
        for entry in body["connections"]
        if entry["to"]["component"] == "PUMP"
    ]
    assert reaching == [("level_level_out", "level_level_in")]
    pump = component_of(a_filled_tank(), "PUMP")
    assert [port["name"] for port in pump["ports"] if port["dir"] == "in"] == [
        "level_level_in"
    ]


def test_the_flow_key_changes_nothing_whether_it_is_there_or_not():
    """The second half of this ticket. muscadet stopped writing `flow` on a pair
    that merely FOLLOWS the flow convention, a measurement link included, and
    this reader has never read the key: it tells the three families apart from
    the box names, which is the same distinction. So a stored document carrying
    it and the same document without it build the same model, and neither is
    routed down the flow path for a name it happens to carry."""
    without = a_filled_tank()
    with_key = a_filled_tank()
    with_key["connections"][1] = a_measurement(
        "TANK", "level", "PUMP", "level", flow="level_level"
    )
    assert declare.build_document(with_key) == declare.build_document(without)


def test_a_measurement_link_between_two_flow_components_is_not_read_as_a_flow():
    """The case that is new for models that already existed: a level link
    between two `ObjFlow` also comes out without the key now. Read as a flow
    connection it would wire `level_level`, a flow nothing declares; read as
    what it is, it emits the whole family the anchor stands for."""
    observer = a_gate("OBS")
    observer["measurements_in"] = [
        {"name": "level", "flows": [], "level_default": 0.0, "fill_default": 0.0}
    ]
    document = a_document(
        a_filler(),
        a_tank(),
        observer,
        connections=[
            {"source": "FILL", "source_box": "q_out", "target": "TANK",
             "target_box": "q_in", "flow": "q"},
            a_measurement("TANK", "level", "OBS", "level"),
        ],
    )
    body = body_of(document)
    reaching = sorted(
        (entry["from"]["port"], entry["to"]["port"])
        for entry in body["connections"]
        if entry["to"]["component"] == "OBS"
    )
    # The family the anchor stands for, and no `level_level_in` port anywhere.
    assert reaching == [("level_fill_out", "level_fill_in"),
                        ("level_level_out", "level_level_in")]
    observed = next(entry for entry in body["components"] if entry["name"] == "OBS")
    assert "level_level_level_in" not in [port["name"] for port in observed["ports"]]


def test_a_constituent_and_a_share_resolve_to_the_alias_the_volume_publishes():
    """A ratio imports on the very box its level comes on, muscadet's box
    carrying both. Here the two are two ports, so the anchor resolves to the
    alias the volume actually publishes -- and a link joining a total to a share
    would read four times the fraction it means."""
    pump = a_pump(
        controls_in=[
            a_channel("share", kind="ratio", flows=["h2"]),
            a_channel("held", kind="level", flows=["h2"]),
        ],
        controls_out=[
            a_signal(
                "run",
                {"op": "band", "input": "share", "direction": "above",
                 "activate": 0.2, "release": 0.1},
            )
        ],
    )
    document = a_document(
        a_filler("H2"),
        a_tank(flows=("h2", "air")),
        a_gate(),
        pump,
        connections=[
            {"source": "H2", "source_box": "q_out", "target": "TANK",
             "target_box": "h2_in"},
            a_measurement("TANK", "level", "PUMP", "share"),
            a_measurement("TANK", "level", "PUMP", "held"),
            {"source": "PUMP", "source_box": "run_out", "target": "GATE",
             "target_box": "run_in"},
        ],
    )
    body = body_of(document)
    reaching = sorted(
        (entry["from"]["port"], entry["to"]["port"])
        for entry in body["connections"]
        if entry["to"]["component"] == "PUMP"
    )
    assert reaching == [
        ("level_level_h2_out", "held_level_h2_in"),
        ("level_ratio_h2_out", "share_ratio_h2_in"),
    ]
    # And the volume really publishes both of them.
    tank = next(entry for entry in body["components"] if entry["name"] == "TANK")
    published = {port["name"] for port in tank["ports"] if port["dir"] == "out"}
    assert {"level_level_h2_out", "level_ratio_h2_out"} <= published


def test_a_rate_reaches_the_input_that_reads_one():
    """R38: what a continuous output DELIVERS is observable, on a box of its own
    -- the one controller input that does not import on `{name}_level_in`."""
    source = a_filler()
    source["flows"][0]["publish_rate"] = True
    pump = a_pump(
        controls_in=[a_channel("flux", kind="rate")],
        controls_out=[
            a_signal(
                "run",
                {"op": "compare", "input": "flux", "operator": ">=",
                 "threshold": 0.5},
            )
        ],
    )
    document = a_document(
        source,
        a_tank(),
        a_gate(),
        pump,
        connections=[
            {"source": "FILL", "source_box": "q_out", "target": "TANK",
             "target_box": "q_in", "flow": "q"},
            {"source": "FILL", "source_box": "q_rate_out", "target": "PUMP",
             "target_box": "flux_rate_in"},
            {"source": "PUMP", "source_box": "run_out", "target": "GATE",
             "target_box": "run_in"},
        ],
    )
    body = body_of(document)
    assert [
        (entry["from"]["port"], entry["to"]["port"])
        for entry in body["connections"]
        if entry["to"]["component"] == "PUMP"
    ] == [("q_rate_out", "flux_rate_in")]
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=2.0)
    assert settled(result.indicators["PUMP_flux_rate"])[-1][1] == pytest.approx(
        FILL_RATE
    )


def test_a_connection_naming_a_box_no_interface_holds_is_refused_by_name():
    document = a_filled_tank()
    document["connections"][1] = a_measurement("TANK", "level", "PUMP", "reading")
    with pytest.raises(declare.SystemSpecError, match="observes on"):
        declare.build_document(document)

    document = a_filled_tank()
    document["connections"][2] = {
        "source": "PUMP", "source_box": "stop_out",
        "target": "GATE", "target_box": "run_in",
    }
    with pytest.raises(declare.SystemSpecError, match="publishes on"):
        declare.build_document(document)


#: What a value output publishes in the montage below, and what muscadet leaves
#: BOTH references of the reading channel at. Measured on PyCATSHOO, on an
#: `ObjCtrl` value output wired to an `ObjFlow`'s `add_measurement_in`:
#: `get_level() -> 7.5` and `get_fill() -> 7.5`, `cnctCount() == 1` on each.
#: muscadet's `MeasurementOut.publish` writes `var_fill = level * gain` when no
#: fill is given, and its single box exports both aliases.
PUBLISHED = 7.5


def test_a_flow_component_reading_a_controller_reads_it_on_both_references():
    """A measurement channel reads a VOLUME, so it materialises a level and a
    weighted fill; a controller publishes a number and holds no volume.

    The two are reconciled by the one published number reaching BOTH references,
    which is exactly what muscadet's single message box does. Wired to the level
    alone, the channel's fill would read 0 where muscadet reads the level, and
    nothing would say so: an indicator or a guard naming `{channel}_fill` is a
    reachable way to observe it.
    """
    meter = a_pump(
        name="METER",
        controls_in=[a_channel("seed")],
        # An output nothing computes, seeded to the reading: a republication of
        # an input nothing feeds would publish zero and measure nothing.
        controls_out=[a_publication("twice", level_default=PUBLISHED)],
    )
    observer = a_gate("OBS")
    observer["measurements_in"] = [
        {"name": "twice", "flows": [], "level_default": 0.0, "fill_default": 0.0}
    ]
    document = a_document(
        meter,
        observer,
        connections=[
            {"source": "METER", "source_box": "twice_level_out",
             "target": "OBS", "target_box": "twice_level_in"},
        ],
    )
    body = body_of(document)
    assert sorted(
        entry["to"]["port"]
        for entry in body["connections"]
        if entry["to"]["component"] == "OBS"
    ) == ["twice_fill_in", "twice_level_in"]

    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=1.0)
    assert settled(result.indicators["OBS_twice_level"])[-1][1] == pytest.approx(
        PUBLISHED
    )
    assert settled(result.indicators["OBS_twice_fill"])[-1][1] == pytest.approx(
        PUBLISHED
    )


def test_a_channel_reading_a_controller_it_declares_no_link_for_is_refused():
    meter = a_pump(
        name="METER",
        controls_in=[a_channel("seed")],
        controls_out=[a_publication("twice", level_default=PUBLISHED)],
    )
    document = a_document(
        meter,
        a_gate("OBS"),
        connections=[
            {"source": "METER", "source_box": "twice_level_out",
             "target": "OBS", "target_box": "twice_level_in"},
        ],
    )
    with pytest.raises(declare.SystemSpecError, match="declares no measurement link"):
        declare.build_document(document)


def test_a_channel_naming_constituents_of_a_controller_is_refused_by_name():
    """Refused where muscadet under-reads in silence: there the constituent
    references are left unconnected and answer their default, so a channel
    naming a substance reads zero of it from an instrument that holds none."""
    meter = a_pump(
        name="METER",
        controls_in=[a_channel("seed")],
        controls_out=[a_publication("twice", level_default=PUBLISHED)],
    )
    observer = a_gate("OBS")
    observer["measurements_in"] = [
        {"name": "twice", "flows": ["q"], "level_default": 0.0, "fill_default": 0.0}
    ]
    document = a_document(
        meter,
        observer,
        connections=[
            {"source": "METER", "source_box": "twice_level_out",
             "target": "OBS", "target_box": "twice_level_in"},
        ],
    )
    with pytest.raises(declare.SystemSpecError, match="holds none"):
        declare.build_document(document)


def test_a_constituent_read_off_a_controller_is_refused_by_name():
    """A controller publishes ONE number, which holds no constituent, so an
    input reading a share of it reads a share of nothing."""
    mirror = a_pump(
        name="METER",
        controls_in=[a_channel("level")],
        controls_out=[
            a_publication("twice", {"op": "republish", "input": "level", "gain": 2.0})
        ],
    )
    vote = a_pump(
        name="VOTE",
        controls_in=[a_channel("twice", kind="ratio", flows=["q"])],
        controls_out=[
            a_signal(
                "run",
                {"op": "band", "input": "twice", "direction": "above",
                 "activate": 0.5, "release": 0.1},
            )
        ],
    )
    document = a_document(
        a_filler(),
        a_tank(),
        mirror,
        vote,
        connections=[
            {"source": "FILL", "source_box": "q_out", "target": "TANK",
             "target_box": "q_in", "flow": "q"},
            a_measurement("TANK", "level", "METER", "level"),
            {"source": "METER", "source_box": "twice_level_out",
             "target": "VOTE", "target_box": "twice_level_in"},
        ],
    )
    with pytest.raises(declare.SystemSpecError, match="holds no constituent"):
        declare.build_document(document)


# --- what has no counterpart, and the shapes this layer refuses --------


def test_min_and_max_are_refused_rather_than_answered_by_a_sum():
    """The one aggregation muscadet declares that this engine cannot compute:
    the only variable-arity reader of an in port has no minimum and no maximum,
    and a measurement link declares no per-connection channel a fixed-arity
    expression could read one by one. Refused rather than approximated, which
    would answer a different question in silence."""
    for aggregate in ("min", "max"):
        with pytest.raises(declare.ComponentSpecError, match="cannot compute"):
            declare.check_controller_spec(
                a_pump(controls_in=[a_channel("level", aggregate=aggregate)])
            )
    for aggregate in ("sum", "mean", "median"):
        declare.check_controller_spec(
            a_pump(controls_in=[a_channel("level", aggregate=aggregate)])
        )


def test_an_unknown_declaration_key_names_the_controller_vocabulary():
    """Never met with the vocabulary of one of the OTHER two shapes: the whole
    cost of a shape being unread was a message listing a vocabulary a modeller
    could do nothing with."""
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.check_controller_spec({**a_pump(), "flows": []})
    message = str(refused.value)
    assert "'flows'" in message
    assert "unknown declaration key" in message
    assert "controls_in" in message and "controls_out" in message


def test_a_subclass_is_expanded_onto_the_peer_class_and_says_so():
    """`cls` is the peer class and the class actually read travels under
    `source_cls`: a subclass declares its interfaces in its own constructor, so
    rebuilding as one AND handing it the sections read back would declare every
    interface twice."""
    declare.check_controller_spec({**a_pump(), "source_cls": "TankPump"})
    with pytest.raises(declare.ComponentSpecError, match="source_cls"):
        declare.check_controller_spec({**a_pump(), "cls": "TankPump"})


def test_two_interfaces_claiming_one_box_are_refused_before_the_wiring():
    """Not a duplicate NAME, which each section already refuses: a value output
    called `x` and a boolean output called `x_level` publish on the same box
    under two different names, so a connection reaching it names neither."""
    with pytest.raises(declare.ComponentSpecError, match="message box"):
        declare.check_controller_spec(
            a_pump(
                controls_out=[
                    a_publication("mirror"),
                    a_signal("mirror_level"),
                ]
            )
        )


def test_an_entry_filed_under_one_name_and_declaring_another_is_refused():
    """The component is built under the name it DECLARES, so a connection naming
    the document's key for it reaches nothing. Refused as an undeclared
    component, which is what the same disagreement on a flow entry already
    gets."""
    document = a_filled_tank()
    document["components"]["PUMP"]["name"] = "PLC"
    with pytest.raises(declare.SystemSpecError, match="not a declared component"):
        declare.build_document(document)


def test_a_controller_declaring_nothing_is_refused():
    with pytest.raises(declare.ComponentSpecError, match="empty component"):
        declare.check_controller_spec(
            a_pump(controls_in=[], controls_out=[])
        )


def test_build_system_refuses_a_controller_rather_than_dropping_it():
    """A controller is no part of the flow system: it holds no flow, so that
    system has no place to put one, and its wiring joins ports a flow component
    does not declare. A caller reaching a system for its wiring gets the system;
    one reaching a model to run gets the document."""
    with pytest.raises(declare.SystemSpecError, match="PEERS of a flow component"):
        declare.build_system(a_filled_tank())


# --- the three kinds, and the document that carries none of the new ----


def test_an_unknown_kind_is_refused_by_its_own_value_and_the_three_it_reads():
    document = a_filled_tank()
    document["components"]["PUMP"]["kind"] = "logic_gate"
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.check_system_spec(document)
    message = str(refused.value)
    assert "kind='logic_gate' is not one of" in message
    assert "'flow'" in message
    assert "'two_state_mode'" in message
    assert "'controller'" in message


def test_a_document_carrying_no_controller_reads_exactly_as_before():
    """The compatibility claim: a document of the two shapes this reader already
    knew is answered by the flow system alone, with no pass over the plugin."""
    document = a_document(
        a_filler(),
        a_tank(),
        connections=[
            {"source": "FILL", "source_box": "q_out", "target": "TANK",
             "target_box": "q_in", "flow": "q"},
        ],
    )
    assert declare.build_document(document) == declare.build_system(
        document
    ).build_dict()


# --- a mode that REACHES a controller ----------------------------------
#
# R44 makes three endpoints of a controller ordinary attributes a `cod3s.ObjFM`
# writes by their exact name, so a mode reaching one is not an exotic corner:
# it is the cyber scenario the whole port exists for -- an instrument that is
# not destroyed but made to stop speaking, the reading still right, the band
# underneath still activated, and the order never arriving.
#
# What stood in the way was the document build. Its mode pass resolved a
# target against the flow components and the modes alone, so a mode naming a
# controller was refused as affecting a component "which the document does not
# declare" -- the scenario turned away at the door, before either engine saw
# anything.

#: Where the blinding mode fires, and how long it holds. Both past `START`, so
#: the comparison underneath is HOLDING throughout: that is what tells a
#: blinded output from one whose condition simply stopped.
BLIND_DATE = 5.0
UNBLIND_DELAY = 3.0


def a_blinding_mode(target="PUMP", output="run"):
    """`add_component(cls="ObjFMDelay", targets=[<a controller>], ...)`.

    Both polarities are written, because neither the availability endpoint nor
    the signal it gates is reinitialised: what does not fall back to rest on
    its own has to be handed back.
    """
    return {
        "name": f"{target}__blind",
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": "blind",
        "targets": [target],
        "target_name": target,
        "failure_effects": {f"{output}_signal_available": False},
        "failure_param_name": ["ttf"],
        "failure_param": [BLIND_DATE],
        "repair_effects": {f"{output}_signal_available": True},
        "repair_param_name": ["ttr"],
        "repair_param": [UNBLIND_DELAY],
    }


def test_a_mode_may_name_a_controller_among_the_components_it_reaches():
    """The refusal that used to arrive first, and the component it now builds.

    Structural on purpose, and paired with the trajectory below: this one says
    the mode was BUILT, the next says it was built onto something that answers.
    """
    body = body_of(a_filled_tank(extra=[a_blinding_mode()]))
    assert "PUMP__blind" in {entry["name"] for entry in body["components"]}


def test_a_blinded_output_stops_carrying_its_comparison_and_is_handed_back():
    """The measurement, and the only thing that says the effect reached the
    engine rather than the document.

    The level reads `t`, so the comparison holds from `START` on and never
    stops. What the gate downstream reads is therefore the availability
    endpoint and nothing else: True from 2, False from 5, True again from 8.
    An engine that derived only the falling edge -- the endpoint is not
    reinitialised, so nothing puts it back on its own -- would answer a gate
    that never closes again.
    """
    document = a_filled_tank(extra=[a_blinding_mode()])
    model = pyraichu.load_model(json.dumps(declare.build_document(document)))
    result = pyraichu.simulate(model, t_max=12.0)
    switched = settled(result.indicators["GATE_run_fed_in"])

    assert switched[0][1] is False, switched
    dates = [time for time, _ in switched[1:]]
    values = [value for _, value in switched[1:]]
    assert values == [True, False, True], switched
    for reached, expected in zip(dates, (START, BLIND_DATE, BLIND_DATE + UNBLIND_DELAY)):
        assert abs(reached - expected) < CROSSING_TOL, switched


def test_a_mode_naming_a_controller_that_is_not_there_is_still_refused():
    """The refusal the fix must not have taken away: what was wrong before was
    the LIST a target is looked up in, never that it was looked up."""
    document = a_filled_tank(extra=[a_blinding_mode(target="NOBODY")])
    with pytest.raises(declare.ComponentSpecError, match="does not declare"):
        declare.build_document(document)


# --- the one attribute the two layers spell apart -----------------------


def test_a_controller_answers_the_boolean_outputs_the_two_layers_name_apart():
    """muscadet holds a boolean output's signal in `{output}_signal_out`, so
    that a mode's unanchored regular expression has a name of its own to anchor
    on; this layer holds it in `{output}` and exports it on `{output}_out`.
    Every other name a controller exposes is shared, which is why this one is
    written down rather than left to be discovered by a refusal.

    Pure: it answers a declaration, so a caller sorts a document with it before
    building anything.
    """
    assert declare.controller_signal_variables(a_pump()) == {
        "run_signal_out": "run",
        "alarm_signal_out": "alarm",
    }


def test_a_value_output_is_absent_from_that_mapping():
    """Both layers call it `{output}_level`, so translating it would invent a
    disagreement."""
    published = a_pump(
        controls_out=[
            a_publication(
                "mirror", {"op": "republish", "input": "level", "gain": 2.0}
            )
        ]
    )
    assert declare.controller_signal_variables(published) == {}


@pytest.mark.parametrize("entry", [None, {}, "PUMP"], ids=["none", "empty", "string"])
def test_anything_that_is_not_a_controller_answers_nothing(entry):
    """So a caller sweeps a document without sorting it first."""
    assert declare.controller_signal_variables(entry) == {}


def test_a_flow_component_is_not_swept_for_a_signal():
    """The mapping is keyed on the component KIND, not on a suffix: a flow
    component holding a variable that happens to end in `_signal_out` is left
    exactly as the document names it."""
    assert declare.controller_signal_variables(a_tank()) == {}
