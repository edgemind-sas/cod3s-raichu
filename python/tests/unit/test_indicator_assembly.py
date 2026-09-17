"""A declared indicator, assembled by either route, is one indicator.

Two routes build a muscadet model into a RAICHU document, and both of them
write indicators into it:

* the **plugin** route (`pyraichu.expand_model` over a `plugins.muscadet`
  section), where a continuous construct makes the whole flow network be
  rebuilt at the end of the expansion, and the rebuild re-emits one
  indicator per observable variable it generated;
* the **declaration** route (`pyraichu.muscadet_engine.build_model` over a
  `muscadet.declare.system_spec`), which builds the document first and
  merges the declared indicators into it afterwards.

They agree on the names -- `{component}_{variable}` is muscadet's own
convention, on both sides -- which is precisely why they must agree on the
assembly too. Measured on 2026-09-15 they did not: the plugin route
APPENDED, so a document declaring an indicator the rebuild was going to
emit anyway got it twice and the engine refused the whole model on
`duplicate indicator name`, naming the modeller's indicator rather than the
rebuild that recopied it. Removing the two continuous components from the
same document made it load.

The engine's refusal is not the defect and is pinned here as it stands: two
observations under one name is a genuine ambiguity, and the floor belongs
where it is. What was wrong is the layer ABOVE producing the duplicate, so
the tests below are written on the two routes and, first of all, ACROSS
them: a test of one route alone would not have seen the drift.

**What each route generated used to be a divergence, and is now a model
key.** On a model with no continuous construct the plugin expansion never
reached the rebuild, so it emitted none of the generated set while the
declaration route emitted it always: the presence of a tank in one corner of
a model decided what every component of it was observed by, and removing
that tank to compare two variants silently took ten observations away.
Settled 2026-09-17 by `pyraichu.indicators.GENERATED_INDICATORS`, the
model-level key both routes read, whose absence means the declared
indicators and nothing else. The test that recorded the divergence
(`test_where_the_agreement_stops_a_model_with_no_continuous_construct`) is
the acceptance of that settlement, rewritten below as
`test_a_boolean_model_agrees_on_both_values_of_the_key`.
"""

import json

import pytest

import pyraichu
import pyraichu.declare as declare
import pyraichu.muscadet as mu
import pyraichu.muscadet_engine as engine
import test_declare_controller as controllers
from pyraichu.indicators import GENERATED_INDICATORS

#: The indicator the modeller declares: the one the running example names on
#: an inverter's input to follow what feeds it. Its name is the one the
#: authoring layer generates for that same observation, which is what makes
#: it the interesting case rather than a special one.
DECLARED = "SNK_power_fed_in"


# --- the same system, written for either route ---------------------------
#
# The smallest model that shows it: a boolean component carrying the
# declared indicator, plus a continuous source and a capacity, which are
# what engages the rebuild. Nothing in the declared indicator has anything
# to do with the continuous part -- that is the point.


def plugin_document(indicators, generated=True):
    """The model as plugin data, indicators declared on the body.

    `generated` is what the model answers on
    :data:`~pyraichu.indicators.GENERATED_INDICATORS`, spelled here because
    every test below that reads a generated observation has to ask for it:
    the key's absence is the declared indicators and nothing else.
    """
    return {
        "name": "probe",
        GENERATED_INDICATORS: generated,
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "SNK",
                        "flows_in": [{"name": "power", "var_in_default": True}],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "WELL",
                        "flows_continuous_out": [
                            {"name": "water", "var_fed_default": 5.0}
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "TANK",
                        "flows_continuous_in": [
                            {"name": "water", "var_in_default": 0.0}
                        ],
                        "capacities": [
                            {
                                "name": "vol",
                                "flow": "water",
                                "capacity": 100.0,
                                "fill_rate": 1.0,
                                "content_init": {"water": 40.0},
                            }
                        ],
                    },
                ]
            }
        },
        "connections": [
            {
                "from": {"component": "WELL", "port": "water_out"},
                "to": {"component": "TANK", "port": "water_in"},
            }
        ],
        "indicators": [
            {
                "name": name,
                "target": "attribute",
                "attr": {"component": component, "attribute": attribute},
            }
            for name, component, attribute in indicators
        ],
    }


def _flow_component(name, flows, **extra):
    """One flow component, as a system declaration spells one."""
    spec = {
        "name": name,
        "cls": "ObjFlow",
        "flows": list(flows),
        "capacities": [],
        "measurements_in": [],
        "measurements_out": [],
        "rules": [],
        "transfers": [],
        "automata": [],
        "failure_modes": [],
    }
    spec.update(extra)
    return spec


def declaration(indicators, generated=True):
    """The same model as a `muscadet.declare.system_spec`.

    The key is spelled at the same level and with the same name as on the
    other route: it is one key of the model, not one per writer.
    """
    return {
        "version": "1.0.0",
        "name": "probe",
        GENERATED_INDICATORS: generated,
        "components": {
            "SNK": _flow_component(
                "SNK",
                [
                    {
                        "cls": "FlowIn",
                        "name": "power",
                        "var_in_default": True,
                        "var_fed_default": False,
                        "logic": "or",
                    }
                ],
            ),
            "WELL": _flow_component(
                "WELL",
                [
                    {
                        "cls": "FlowContinuousOut",
                        "name": "water",
                        "var_fed_default": 5.0,
                    }
                ],
            ),
            "TANK": _flow_component(
                "TANK",
                [
                    {
                        "cls": "FlowContinuousIn",
                        "name": "water",
                        "var_in_default": 0.0,
                    }
                ],
                capacities=[
                    {
                        "name": "vol",
                        "flow": "water",
                        "capacity": 100.0,
                        "fill_rate": 1.0,
                        "content_init": {"water": 40.0},
                    }
                ],
            ),
        },
        "connections": [
            {
                "source": "WELL",
                "source_box": "water_out",
                "target": "TANK",
                "target_box": "water_in",
                "flow": "water",
            }
        ],
        "indicators": [
            {
                "name": name,
                "label": name,
                "measure": "value",
                "stats": ["mean"],
                "component": component,
                "operator": "==",
                "var": attribute,
                "kind": "PycVarIndicator",
            }
            for name, component, attribute in indicators
        ],
    }


#: The declaration of the running example: one indicator, named the way the
#: authoring layer names that same observation.
ONE_DECLARED = [(DECLARED, "SNK", "power_fed_in")]


def observed(body):
    """What a document's indicators observe, keyed by name.

    A mapping and not a list, because the two routes order their
    indicators differently -- the plugin route carries the declared ones
    first, the declaration route emits them where the generation puts them
    -- and the order of indicators is not a property either of them
    promises. A duplicate would be lost by this reduction, so every test
    that builds one asserts on the LIST.
    """
    return {
        entry["name"]: {k: v for k, v in entry.items() if k != "name"}
        for entry in body["indicators"]
    }


def named(body, name):
    """Every entry of `name` in a document, duplicates included."""
    return [entry for entry in body["indicators"] if entry["name"] == name]


# --- 1. the plugin route carries a declared indicator through ------------


def test_a_declared_indicator_and_a_continuous_construct_load_and_run():
    """The defect, at the scale a modeller met it: the model did not load.

    Run and not only loaded: an indicator that survives the assembly and
    records nothing would pass a test written on the document alone.
    """
    model = pyraichu.load_model(plugin_document(ONE_DECLARED))
    result = pyraichu.simulate(model, t_max=10.0, samples=[0.0, 5.0, 10.0])

    assert result.samples[DECLARED] == [(0.0, True), (5.0, True), (10.0, True)]
    # The continuous part is live too, so the rebuild the defect came from
    # really did run: the volume fills at its `fill_rate` of 1.0 from 40.
    filled = dict(result.samples["TANK_vol_content_water"])
    assert filled[0.0] == pytest.approx(40.0)
    assert filled[10.0] == pytest.approx(50.0)


def test_the_declared_indicator_appears_once_and_observes_what_it_named():
    body = pyraichu.expand_model(plugin_document(ONE_DECLARED))

    assert named(body, DECLARED) == [
        {
            "name": DECLARED,
            "target": "attribute",
            "attr": {"component": "SNK", "attribute": "power_fed_in"},
        }
    ]


def test_without_the_continuous_constructs_the_same_declaration_held():
    """What the defect was measured against, kept as the control.

    A boolean-only document used to escape the duplicate for the wrong
    reason: it never engaged the rebuild, so nothing was there to recopy the
    declared indicator. It now asks for the generated set like any other
    model and gets the same answer through the merge, which is what makes it
    a control rather than a coincidence.
    """
    document = plugin_document(ONE_DECLARED)
    document["plugins"]["muscadet"]["objects"] = [
        document["plugins"]["muscadet"]["objects"][0]
    ]
    document["connections"] = []

    assert named(pyraichu.expand_model(document), DECLARED) == [
        {
            "name": DECLARED,
            "target": "attribute",
            "attr": {"component": "SNK", "attribute": "power_fed_in"},
        }
    ]


# --- 2. what the rebuild cannot re-emit is not lost ----------------------


def test_an_indicator_the_rebuild_cannot_re_emit_is_kept():
    """The risk of "replace rather than append", taken naively.

    Two shapes of it, and neither has a counterpart among what the rebuild
    emits: an indicator the modeller RENAMED, whose observation the layer
    re-emits under its own name, and one on a variable the layer generates
    without ever observing (`water_out_rate`, published only when the flow
    asks for a rate channel). Dropping what the rebuild does not re-emit
    would lose both without a word.
    """
    document = plugin_document(
        ONE_DECLARED
        + [
            ("stock", "TANK", "vol_content_water"),
            ("WELL_water_out_rate", "WELL", "water_out_rate"),
        ]
    )
    body = pyraichu.expand_model(document)
    observations = observed(body)

    assert observations["stock"] == {
        "target": "attribute",
        "attr": {"component": "TANK", "attribute": "vol_content_water"},
    }
    assert observations["WELL_water_out_rate"] == {
        "target": "attribute",
        "attr": {"component": "WELL", "attribute": "water_out_rate"},
    }
    # The renamed one does NOT take the place of the generated one: they are
    # two names for one observation, and both are asked for.
    assert "TANK_vol_content_water" in observations
    # And the whole document still loads: no name is carried twice.
    pyraichu.load_model(document)


# --- 3. the same defect on the other writer of the expansion -------------


def a_gauged_cistern():
    """A volume authored class-side, watched by a plugin controller.

    The plugin section holds the CONTROLLER alone, so nothing engages the
    continuous rebuild: this is the expansion's FIRST pass, where a plugin
    hands back its own indicators object by object. A controller emits one
    per attribute it generates, `{controller}_{attribute}`, so a document
    declaring `CTRL_open` to follow the command names an observation the
    expansion was going to emit anyway -- the same encounter as the
    rebuild's, one pass earlier.
    """

    class Cistern(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_capacity(
                name="vol", flow="w", capacity=100.0, content_init={"w": 4.0}
            )

    system = mu.System("gauge")
    system.add_component(Cistern, "T")
    document = system.build_dict()
    body = pyraichu.model_body(document)
    body["plugins"] = {
        "muscadet": {
            "objects": [
                {
                    "type": "ObjCtrl",
                    "name": "CTRL",
                    "controls_in": [{"name": "vol", "kind": "level"}],
                    "controls_out": [
                        {
                            "name": "open",
                            "kind": "bool",
                            "emit": {
                                "op": "band",
                                "input": "vol",
                                "direction": "above",
                                "activate": 5.0,
                                "release": 3.0,
                            },
                        }
                    ],
                }
            ]
        }
    }
    body["connections"].append(
        {
            "from": {"component": "T", "port": "vol_level_out"},
            "to": {"component": "CTRL", "port": "vol_level_in"},
        }
    )
    return document


def test_an_indicator_declared_on_what_a_plugin_object_emits_is_one_indicator():
    document = a_gauged_cistern()
    pyraichu.model_body(document)["indicators"].append(
        {
            "name": "CTRL_open",
            "target": "attribute",
            "attr": {"component": "CTRL", "attribute": "open"},
        }
    )

    assert named(pyraichu.expand_model(document), "CTRL_open") == [
        {
            "name": "CTRL_open",
            "target": "attribute",
            "attr": {"component": "CTRL", "attribute": "open"},
        }
    ]
    pyraichu.load_model(document)


def test_a_plugin_object_emitting_over_a_different_observation_is_refused():
    """The refusal names the PLUGIN that emitted the second one."""
    document = a_gauged_cistern()
    pyraichu.model_body(document)["indicators"].append(
        {
            "name": "CTRL_open",
            "target": "attribute",
            "attr": {"component": "T", "attribute": "vol_content"},
        }
    )

    with pytest.raises(ValueError, match="plugin `muscadet` emits the indicator"):
        pyraichu.expand_model(document)


# --- 4. the two routes, crossed ------------------------------------------


@pytest.mark.parametrize("generated", [True, False], ids=["asked", "not-asked"])
@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(ONE_DECLARED, id="named-as-the-layer-names-it"),
        pytest.param(
            ONE_DECLARED
            + [
                ("stock", "TANK", "vol_content_water"),
                ("WELL_water_out_rate", "WELL", "water_out_rate"),
            ],
            id="with-what-the-layer-cannot-re-emit",
        ),
    ],
)
def test_both_routes_render_the_same_indicators(declared, generated):
    """The test the repository did not have, and the drift it would have
    caught: one model, written for either route, and the same observations
    out of both.

    Written across the two rather than on each, because each route was
    self-consistent while they disagreed: the declaration route merged by
    declared name and the plugin route appended, and only a comparison
    could tell.

    Run on BOTH values of the key, since the key is now what each route
    reads to decide: a route honouring it and a route ignoring it would
    agree on one value and part on the other, which is exactly the shape of
    the divergence it replaces.
    """
    through_plugin = pyraichu.expand_model(plugin_document(declared, generated))
    through_declaration = pyraichu.model_body(
        json.loads(engine.build_model(declaration(declared, generated)).json)
    )

    assert observed(through_plugin) == observed(through_declaration)
    # Set equality would hold over a duplicate; the lists are what say each
    # name is carried once.
    for body in (through_plugin, through_declaration):
        names = [entry["name"] for entry in body["indicators"]]
        assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "route",
    ["plugin", "declaration"],
)
def test_both_routes_refuse_one_name_over_two_observations(route):
    """The collision the merge does NOT resolve, refused on either side.

    `TANK_vol_content` is a name the rebuild emits, declared here on
    another variable of the same component. Skipping the second writer
    would silently drop one of the two observations, so it is refused, and
    refused where both are still in hand: the refusal names the rebuild,
    which the engine's `duplicate indicator name` cannot.
    """
    collision = [("TANK_vol_content", "TANK", "vol_fill")]
    if route == "plugin":
        with pytest.raises(ValueError, match="already observes"):
            pyraichu.expand_model(plugin_document(collision))
    else:
        with pytest.raises(declare.SystemSpecError, match="already observes"):
            engine.build_model(declaration(collision))


def test_the_plugin_refusal_names_the_rebuild_that_emitted_the_other_one():
    """What the modeller was missing: the engine named his indicator, and
    nothing said the rebuild had emitted one of its own."""
    with pytest.raises(ValueError) as refusal:
        pyraichu.expand_model(
            plugin_document([("TANK_vol_content", "TANK", "vol_fill")])
        )

    assert "rebuilding the continuous network" in str(refusal.value)


def a_boolean_model(generated):
    """The running example's comparison, at its smallest: the model of
    `test_both_routes_render_the_same_indicators` with the tank and the well
    taken out, written for either route and carrying one renamed indicator.

    Taking a buffer out of a model to compare two variants is an ordinary
    thing to do, and it is what used to move what the REST of the model was
    observed by.
    """
    renamed = [("availability", "SNK", "power_fed_in")]

    boolean_plugin = plugin_document(renamed, generated)
    boolean_plugin["plugins"]["muscadet"]["objects"] = [
        boolean_plugin["plugins"]["muscadet"]["objects"][0]
    ]
    boolean_plugin["connections"] = []
    boolean_declaration = declaration(renamed, generated)
    boolean_declaration["components"] = {
        "SNK": boolean_declaration["components"]["SNK"]
    }
    boolean_declaration["connections"] = []
    return boolean_plugin, boolean_declaration


@pytest.mark.parametrize(
    "generated, observations",
    [
        pytest.param(True, ["SNK_power_fed_in", "availability"], id="asked"),
        pytest.param(False, ["availability"], id="not-asked"),
    ],
)
def test_a_boolean_model_agrees_on_both_values_of_the_key(generated, observations):
    """Where the agreement used to stop, and no longer does.

    This is `test_where_the_agreement_stops_a_model_with_no_continuous
    _construct`, rewritten. It was written to FAIL the day somebody settled
    the arbitration it recorded: on a model carrying no continuous construct
    the plugin expansion never reached the rebuild and emitted none of the
    generated set, while the declaration route emitted it in every case, so
    what a model observed depended on whether a tank was declared somewhere
    in it. Settled 2026-09-17 -- the model says, through
    `GENERATED_INDICATORS`, and both routes read it -- so the test now
    measures the settlement rather than the gap.

    Both values, because one value proves nothing here: a route that
    ignored the key entirely would still agree with the other on whichever
    value happens to match what it does unconditionally. The pair is what
    says the key is READ.

    The set the key asks for is the muscadet one, `{component}_{variable}`
    per observable variable; `availability` is the declared indicator, under
    the name its author gave it, and it is carried by both routes either way
    -- what a document declares is never what this key governs.
    """
    boolean_plugin, boolean_declaration = a_boolean_model(generated)

    through_plugin = pyraichu.expand_model(boolean_plugin)
    through_declaration = pyraichu.model_body(
        json.loads(engine.build_model(boolean_declaration).json)
    )

    assert sorted(observed(through_plugin)) == sorted(observations)
    assert observed(through_plugin) == observed(through_declaration)
    # Carried once and observing what it named, on both, whatever the key
    # says: the declared indicator is not part of the bargain.
    for body in (through_plugin, through_declaration):
        assert named(body, "availability") == [
            {
                "name": "availability",
                "target": "attribute",
                "attr": {"component": "SNK", "attribute": "power_fed_in"},
            }
        ]


def test_the_key_absent_is_the_declared_indicators_and_nothing_else():
    """The default, stated on its own rather than left to a parameter.

    It is the reading that changes nothing for the corpus: every COD3S
    Platform study is boolean and reaches the engine through the plugin
    expansion, which emitted nothing of its own for them. A model that wants
    the generated set asks for it.
    """
    boolean_plugin, boolean_declaration = a_boolean_model(False)
    del boolean_plugin[GENERATED_INDICATORS]
    del boolean_declaration[GENERATED_INDICATORS]

    through_plugin = pyraichu.expand_model(boolean_plugin)
    through_declaration = pyraichu.model_body(
        json.loads(engine.build_model(boolean_declaration).json)
    )

    assert sorted(observed(through_plugin)) == ["availability"]
    assert observed(through_plugin) == observed(through_declaration)


def test_the_key_governs_a_continuous_model_too():
    """The other half of "the construct no longer decides".

    A model carrying a tank and NOT asking for the set gets the declared
    indicator alone -- the rebuild still runs, since it is what resolves the
    network, and it simply emits no observation of its own. Together with
    the boolean case above, this is what makes the key the only authority:
    neither value of it is reachable from the shape of the model.
    """
    body = pyraichu.expand_model(plugin_document(ONE_DECLARED, generated=False))

    assert sorted(observed(body)) == [DECLARED]
    # The rebuild did run: the netting and the sweep order are there, which
    # is the material only the connection list can supply.
    assert body["evaluation_order"]
    tank = next(c for c in body["components"] if c["name"] == "TANK")
    assert any(eq["target"] == "vol_content_water" for eq in tank["equations"])


@pytest.mark.parametrize(
    "route, refusal",
    [
        pytest.param("plugin", ValueError, id="plugin"),
        pytest.param("declaration", declare.SystemSpecError, id="declaration"),
    ],
)
def test_a_key_that_is_not_a_boolean_is_refused_on_either_route(route, refusal):
    """`"false"` is a true string, and a model read by truthiness alone
    would observe everything while saying it wanted nothing."""
    if route == "plugin":
        document = plugin_document(ONE_DECLARED)
        document[GENERATED_INDICATORS] = "false"
        act = lambda: pyraichu.expand_model(document)  # noqa: E731
    else:
        document = declaration(ONE_DECLARED)
        document[GENERATED_INDICATORS] = "false"
        act = lambda: engine.build_model(document)  # noqa: E731

    with pytest.raises(refusal, match=GENERATED_INDICATORS):
        act()


def test_a_model_level_key_nothing_reads_is_IGNORED_and_that_is_the_trap():
    """Measured 2026-09-17, before the key was written, and pinned because
    it is what makes a misspelling silent.

    The component level is a CLOSED vocabulary: an unknown key there is
    refused by name against `COMPONENT_KEYS`. The model level is open, so
    `generated_indicator` -- the same key, one letter short -- is read by
    nobody and the model is observed by what it declared, with no message.
    Recorded here rather than closed: refusing an unknown model-level key
    would refuse documents this layer has never read, and that is a
    decision of its own.
    """
    document = declaration(ONE_DECLARED, generated=False)
    del document[GENERATED_INDICATORS]
    document["generated_indicator"] = True  # one letter short

    declare.check_system_spec(document)  # accepted, not refused
    body = pyraichu.model_body(json.loads(engine.build_model(document).json))

    assert sorted(observed(body)) == [DECLARED]
    assert "generated_indicator" not in body


# --- 5. the declaration route's own two writers --------------------------
#
# `build_document` writes indicators twice on its own account: once per
# object it expands (`_expand_object`), and once when it puts the rebuilt
# flow components' own set in front of them. Both used to concatenate, and
# both are reachable, because an indicator's name is FLATTENED --
# `{component}_{variable}` -- so two different (component, variable) pairs
# meet as soon as one component's name ends where the other's variable
# begins. Nothing forbids that naming, and it is not exotic: a controller
# named after the level it watches is the ordinary way to name one.
#
# Measured on 2026-09-15 against the branch point: both documents below
# came out carrying a name twice, and `load_model` refused them on
# `duplicate indicator name` -- the modeller's defect exactly, on the route
# that was supposed to be the sound one.


def test_a_controller_name_meeting_a_capacity_variable_is_refused():
    """`build_document`'s prepend of the rebuilt flow components' set.

    The tank's capacity `level` publishes `level_content`, so the flow
    components observe `TANK_level_content`. A controller named `TANK_level`
    carrying a signal `content` emits that same name over its own
    observation. Two observations, one name.

    Before the merge: both entries were kept and the engine refused the
    document on `duplicate indicator name`, naming the indicator and not
    either writer. Refused here instead, where both are still in hand, so
    the message can put the two observations side by side.
    """
    default = controllers.a_pump(name="TANK_level")["controls_out"]
    pump = controllers.a_pump(
        name="TANK_level",
        controls_out=list(default)
        + [
            controllers.a_signal(
                "content",
                {
                    "op": "compare",
                    "input": "level",
                    "operator": ">=",
                    "threshold": controllers.START,
                },
            )
        ],
    )

    with pytest.raises(declare.SystemSpecError) as refusal:
        declare.build_document(controllers.a_filled_tank(pump=pump))

    message = str(refusal.value)
    assert "'TANK_level_content'" in message
    assert "{'component': 'TANK_level', 'attribute': 'content'}" in message
    assert "{'component': 'TANK', 'attribute': 'level_content'}" in message


def test_two_objects_whose_flattened_names_meet_are_refused():
    """`_expand_object`, the same encounter between two expanded objects.

    Controller `P` emits `P_q_r` for its signal `q_r`; controller `P_q`
    emits `P_q_r` for its signal `r`. Neither knows about the other, and
    the second one to be expanded meets what the first wrote.

    Before the merge this document carried THREE duplicated names, the
    signal and the two variables a boolean output drags with it
    (`_signal_available`, `_threshold`), and the engine named only the
    first of them.
    """

    def pump(name, signal):
        default = controllers.a_pump(name=name)["controls_out"]
        return controllers.a_pump(
            name=name,
            controls_out=list(default)
            + [
                controllers.a_signal(
                    signal,
                    {
                        "op": "compare",
                        "input": "level",
                        "operator": ">=",
                        "threshold": controllers.START,
                    },
                )
            ],
        )

    document = controllers.a_document(
        controllers.a_filler(),
        controllers.a_tank(),
        controllers.a_gate(),
        pump("P", "q_r"),
        pump("P_q", "r"),
        connections=[
            {
                "source": "FILL",
                "source_box": "q_out",
                "target": "TANK",
                "target_box": "q_in",
                "flow": "q",
            },
            controllers.a_measurement("TANK", "level", "P", "level"),
            controllers.a_measurement("TANK", "level", "P_q", "level"),
            {
                "source": "P",
                "source_box": "run_out",
                "target": "GATE",
                "target_box": "run_in",
            },
        ],
    )

    with pytest.raises(declare.ComponentSpecError) as refusal:
        declare.build_document(document)

    message = str(refusal.value)
    # The refusal names the OBJECT that emitted the second one, which is
    # what the engine's own message cannot do.
    assert message.startswith("Controller P_q:")
    assert "'P_q_r'" in message


# --- 6. the engine's own refusal, unchanged ------------------------------


@pytest.mark.parametrize(
    "second",
    [
        pytest.param({"component": "SNK", "attribute": "power_fed_in"}, id="identical"),
        pytest.param({"component": "TANK", "attribute": "vol_content"}, id="differing"),
    ],
)
def test_the_engine_still_refuses_two_indicators_of_one_name(second):
    """The floor under the merge, pinned as it stands.

    A CORE document, the plugins section already expanded away, carrying
    one indicator twice: refused by the engine whether the two entries
    observe the same thing or not. This is the document the plugin route
    used to hand it, and that refusal is what makes the merge above
    necessary rather than cosmetic. Relaxing it was never the fix -- the
    duplicate was produced by the writing layer, and that is where it was
    removed.
    """
    body = pyraichu.expand_model(plugin_document(ONE_DECLARED))
    assert "plugins" not in body
    body["indicators"].append(
        {"name": DECLARED, "target": "attribute", "attr": second}
    )

    with pytest.raises(pyraichu.ModelError, match="duplicate indicator name"):
        pyraichu.load_model(body)
