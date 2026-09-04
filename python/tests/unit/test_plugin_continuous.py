"""The continuous constructs, authorable as plugin data.

`ObjFlow` used to refuse six declaration keys by name: the continuous
flows, the capacities, the measurement channels, the rule sets and the
transfer pairs. The refusal was structural rather than cautious. A plugin
object is expanded on its own, before the objects after it exist, so it
cannot see the connection list, and every one of those constructs emits
material that reads it: what a producer publishes to one consumer is what
remains once the OTHERS are accounted for, an allocation operator splits
over the connections it serves, and the sweep order runs along the flow
graph.

The model-level finalisation hook is what lifts it, and the strongest test
of the lift is not that the plugin builds something: it is that the plugin
and the class-based builder write **one document**, byte for byte, for a
model exercising all five families at once. Two implementations of one
semantics is the defect this repository keeps paying for; one document is
the evidence there is only one.
"""

import json
import math

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL


# --- the parity model, authored twice ------------------------------------


class Source(mu.ObjFlow):
    """A supply modulated by a declared time profile."""

    def add_flows(self):
        self.add_flow_continuous_out(
            name="feed",
            var_fed_default=6.0,
            profile={
                "cls": "SinusoidalProfile",
                "amplitude": 0.5,
                "period": 24.0,
                "offset": 0.5,
            },
        )


class Reactor(mu.ObjFlow):
    """A transformation gated by a boolean input: two of feed make one of
    product, and nothing while the gate is down."""

    def add_flows(self):
        self.add_flow_in(name="run")
        self.add_flow_continuous_in(name="feed")
        self.add_flow_continuous_out(name="product")
        self.add_rule_set(
            name="duty",
            rules=[
                {
                    "name": "on",
                    "cond": [{"name": "run", "port": "in"}],
                    "cons": {"feed": 2.0},
                    "prod": {"product": 1.0},
                },
                {"name": "off", "cons": {"feed": 0.0}, "prod": {"product": 0.0}},
            ],
        )


class Tank(mu.ObjFlow):
    """A volume between the reactor and two consumers, splitting a
    shortage in a declared ratio."""

    def add_flows(self):
        self.add_flow_continuous_in(name="product")
        self.add_flow_continuous_out(
            name="product",
            var_fed_default=4.0,
            allocation="shares",
            allocation_shares={"D1": 0.75, "D2": 0.25},
        )
        self.add_capacity(
            name="vol",
            flow="product",
            capacity=50.0,
            content_init={"product": 20.0},
            fill_rate=1.0,
        )


class Drain(mu.ObjFlow):
    """A pure consumer asking for a declared rate."""

    def add_flows(self):
        self.add_flow_continuous_in(name="product", var_demand_in_default=3.0)


class Wall(mu.ObjFlow):
    """A metered conduit: what crosses it is a conductive law written over
    the level it reads through a measurement channel."""

    def add_flows(self):
        self.add_flow_continuous_in(name="heat")
        self.add_flow_continuous_out(name="heat", var_fed_default=9.0)
        self.add_measurement_in(name="vol", flows=["product"])
        self.add_transfer(
            name="cross",
            flows=["heat", "heat"],
            equation={
                "cls": "ConductiveTransfer",
                "conductance": 0.25,
                "potential_a": {"measurement": "vol"},
                "potential_b": {"const": 5.0},
            },
        )


MODEL_NAME = "continuous_plugin_parity"


def authored() -> mu.System:
    """The parity model, written class by class through the builder."""
    system = mu.System(MODEL_NAME)
    system.add_component(Source, "S")
    system.add_component(Reactor, "R")
    system.add_component(Tank, "T")
    system.add_component(Drain, "D1")
    system.add_component(Drain, "D2")
    system.add_component(Wall, "W")
    system.connect("S", "feed", "R", "feed")
    system.connect("R", "product", "T", "product")
    system.connect("T", "product", "D1", "product")
    system.connect("T", "product", "D2", "product")
    system.connect_measurement("T", "vol", "W")
    return system


def link(source: str, port_out: str, target: str, port_in: str) -> dict:
    return {
        "from": {"component": source, "port": port_out},
        "to": {"component": target, "port": port_in},
    }


def declared() -> dict:
    """The same model, written as plugin data.

    The measurement link is four ordinary connections here, which is what
    `System.connect_measurement` writes on the other side: the reading
    carries no quantity, so it is wired like anything else.
    """
    return {
        "name": MODEL_NAME,
        "components": [],
        "indicators": [],
        "connections": [
            link("S", "feed_out", "R", "feed_in"),
            link("R", "product_out", "T", "product_in"),
            link("T", "product_out", "D1", "product_in"),
            link("T", "product_out", "D2", "product_in"),
            link("T", "vol_level_out", "W", "vol_level_in"),
            link("T", "vol_fill_out", "W", "vol_fill_in"),
            link("T", "vol_level_product_out", "W", "vol_level_product_in"),
            link("T", "vol_fill_product_out", "W", "vol_fill_product_in"),
        ],
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "S",
                        "flows_continuous_out": [
                            {
                                "name": "feed",
                                "var_fed_default": 6.0,
                                "profile": {
                                    "cls": "SinusoidalProfile",
                                    "amplitude": 0.5,
                                    "period": 24.0,
                                    "offset": 0.5,
                                },
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "R",
                        "flows_in": [{"name": "run"}],
                        "flows_continuous_in": [{"name": "feed"}],
                        "flows_continuous_out": [{"name": "product"}],
                        "rules": [
                            {
                                "name": "duty",
                                "rules": [
                                    {
                                        "name": "on",
                                        "cond": [{"name": "run", "port": "in"}],
                                        "cons": {"feed": 2.0},
                                        "prod": {"product": 1.0},
                                    },
                                    {
                                        "name": "off",
                                        "cons": {"feed": 0.0},
                                        "prod": {"product": 0.0},
                                    },
                                ],
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "T",
                        "flows_continuous_in": [{"name": "product"}],
                        "flows_continuous_out": [
                            {
                                "name": "product",
                                "var_fed_default": 4.0,
                                "allocation": "shares",
                                "allocation_shares": {"D1": 0.75, "D2": 0.25},
                            }
                        ],
                        "capacities": [
                            {
                                "name": "vol",
                                "flow": "product",
                                "capacity": 50.0,
                                "content_init": {"product": 20.0},
                                "fill_rate": 1.0,
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "D1",
                        "flows_continuous_in": [
                            {"name": "product", "var_demand_default": 3.0}
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "D2",
                        "flows_continuous_in": [
                            {"name": "product", "var_demand_default": 3.0}
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "W",
                        "measurements_in": [{"name": "vol", "flows": ["product"]}],
                        "flows_continuous_in": [{"name": "heat"}],
                        "flows_continuous_out": [
                            {"name": "heat", "var_fed_default": 9.0}
                        ],
                        "transfers": [
                            {
                                "name": "cross",
                                "flows": ["heat", "heat"],
                                "equation": {
                                    "cls": "ConductiveTransfer",
                                    "conductance": 0.25,
                                    "potential_a": {"measurement": "vol"},
                                    "potential_b": {"const": 5.0},
                                },
                            }
                        ],
                    },
                ]
            }
        },
    }


def expanded(document: dict) -> dict:
    """The plugin document, expanded and stripped of the empty core list
    `expand_model` seeds and the builder never writes."""
    body = pyraichu.expand_model(document)
    assert body.pop("targets") == []
    return body


def test_the_plugin_and_the_builder_write_one_document():
    """The whole point of the lift, and the evidence there is one
    implementation and not two.

    Continuous flows, a capacity, a rule set, a transfer pair and a
    measurement channel, in one model, authored both ways: the two
    documents are equal, component for component, equation for equation,
    indicator for indicator, and step for step of the evaluation order.

    A second implementation of any of those five would show up here as a
    difference, because none of them is a detail the engine forgives: the
    netting between two consumers, the allocation operator's parameters
    and the three-band sweep order are all part of the answer.
    """
    plugin = expanded(declared())
    builder = pyraichu.model_body(authored().build_dict())
    assert plugin == builder, json.dumps(
        {
            key: [plugin.get(key), builder.get(key)]
            for key in sorted(set(plugin) | set(builder))
            if plugin.get(key) != builder.get(key)
        },
        indent=1,
    )[:4000]


def test_the_plugin_document_is_not_trivially_equal():
    """The guard on the test above: it would pass just as well on two
    empty documents. The model really does carry the five families."""
    body = expanded(declared())
    tank = next(c for c in body["components"] if c["name"] == "T")
    assert [a["name"] for a in tank.get("allocations", [])] == ["product_alloc"]
    assert tank["allocations"][0]["policy"] == "shares"
    assert len(tank["allocations"][0]["shares"]) == 2
    targets = {
        (component["name"], equation["target"])
        for component in body["components"]
        for equation in component["equations"]
    }
    assert ("T", "vol_content_product") in targets  # the capacity
    assert ("R", "duty_scale") in targets  # the rule set
    assert ("W", "cross_moved") in targets  # the transfer pair
    assert ("W", "vol_level") in targets  # the measurement channel
    # And the netting an isolated object could not write: what T publishes
    # to D1 is its capability less what D2 was allocated.
    netted = next(
        equation
        for equation in tank["equations"]
        if equation["target"] == "product_out__capability__D1__product_in"
    )
    assert json.dumps(netted).count("product_out__alloc__D2__product_in") == 1
    assert len(body["evaluation_order"]) > 40


def test_the_plugin_document_simulates():
    """It loads and runs, and the two answers only the connection list
    can give are the declared ones.

    The tank delivers four to two consumers asking three each, and the
    shortage splits three to one as `allocation_shares` says; the wall
    reads the volume's level through the measurement channel. Neither
    quantity exists in a per-object expansion.
    """
    model = pyraichu.load_model(expanded(declared()))
    result = pyraichu.simulate(model, t_max=4.0, seed=1)
    read = {name: series[0][1] for name, series in result.indicators.items()}
    assert abs(read["T_product_fed_out"] - 4.0) < CROSSING_TOL
    assert abs(read["D1_product_fed_in"] - 3.0) < CROSSING_TOL
    assert abs(read["D2_product_fed_in"] - 1.0) < CROSSING_TOL
    assert abs(read["W_vol_level"] - 20.0) < CROSSING_TOL


# --- controllers and continuous flows in one model ------------------------


def a_regulated_tank() -> dict:
    """A plugin-only model carrying both families: a continuous plant and
    the controller that gates it."""
    return {
        "name": "regulated",
        "components": [],
        "indicators": [],
        "connections": [
            link("S", "w_out", "T", "w_in"),
            link("T", "vol_level_out", "LOW", "vol_level_in"),
            link("LOW", "run_out", "S", "run_in"),
        ],
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "S",
                        "flows_in": [{"name": "run"}],
                        "flows_continuous_out": [{"name": "w"}],
                        "rules": [
                            {
                                "name": "pump",
                                "rules": [
                                    {
                                        "name": "on",
                                        "cond": [{"name": "run", "port": "in"}],
                                        "prod": {"w": 2.0},
                                    },
                                    {"name": "off", "prod": {"w": 0.0}},
                                ],
                            }
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "T",
                        "flows_continuous_in": [{"name": "w"}],
                        "capacities": [
                            {
                                "name": "vol",
                                "flow": "w",
                                "capacity": 100.0,
                                "content_init": {"w": 4.0},
                                "fill_rate": math.inf,
                            }
                        ],
                    },
                    {
                        "type": "ObjCtrl",
                        "name": "LOW",
                        "controls_in": [{"name": "vol", "kind": "level"}],
                        "controls_out": [
                            {
                                "name": "run",
                                "kind": "bool",
                                "emit": {
                                    "op": "band",
                                    "input": "vol",
                                    "direction": "below",
                                    "activate": 6.0,
                                    "release": 8.0,
                                },
                            }
                        ],
                    },
                ]
            }
        },
    }


def test_controllers_and_continuous_flows_share_one_evaluation_order():
    """The single-writer conflict, resolved.

    `ObjCtrl` derives an evaluation order and so does the continuous
    network, and a model-wide property has one writer. The resolution is
    that the network is that writer: it derives the three bands over the
    flow graph, then CLOSES the order over every explicit equation and
    every allocation the model declares, controllers included, which lands
    each controller's readings after the level it mirrors.

    The order must cover the declared steps exactly, one entry each, or
    the engine refuses the model: that is what this asserts.
    """
    body = pyraichu.expand_model(a_regulated_tank())
    order = [
        (step["component"], step["attribute"]) for step in body["evaluation_order"]
    ]
    assert len(order) == len(set(order)), "a step is listed twice"
    declared_steps = [
        (component["name"], equation["target"])
        for component in body["components"]
        for equation in component["equations"]
        if equation["kind"] == "explicit"
    ] + [
        (component["name"], allocation["name"])
        for component in body["components"]
        for allocation in component.get("allocations", [])
    ]
    assert sorted(order) == sorted(declared_steps)
    assert order.index(("T", "vol_content")) < order.index(("LOW", "vol_level"))
    pyraichu.load_model(body)


def test_the_regulated_tank_holds_between_its_two_edges():
    """The order is not just complete, it is right.

    The band activates below 6 and releases at 8, and the pump fills at 2
    from a content of 4. If the controller read a level the sweep had left
    behind, or if the plant's own bands ran the wrong way round, the
    release date would not be the one the crossing gives.

    It releases at t = 2, where the level reaches 8: located by
    root-finding, not noticed at the next discrete event, and eight is the
    release edge rather than the six a bare comparison would stop at.
    """
    result = pyraichu.simulate(
        pyraichu.load_model(pyraichu.expand_model(a_regulated_tank())),
        t_max=12.0,
        seed=1,
    )
    series = result.indicators["LOW_run"]
    changes = [
        (time, value)
        for index, (time, value) in enumerate(series)
        if index and value != series[index - 1][1]
    ]
    assert [value for _, value in changes] == [True, False]
    assert abs(changes[1][0] - 2.0) < CROSSING_TOL
    level = dict(result.indicators["T_vol_content"])
    assert abs(level[changes[1][0]] - 8.0) < CROSSING_TOL


def test_an_asserted_evaluation_order_is_refused_beside_a_continuous_network():
    """The genuine contradiction stays loud.

    A model that both asserts an order and declares continuous flows has
    two authorities on one sweep. The network derives its order from the
    flow graph, so honouring the asserted one would sweep a consumer
    before its producer without a word: refused, naming both.
    """
    document = a_regulated_tank()
    document["evaluation_order"] = [{"component": "T", "attribute": "vol_content"}]
    with pytest.raises(ValueError, match="one sweep is a contradiction"):
        pyraichu.expand_model(document)


# --- the contract of the hook itself --------------------------------------


def test_a_plugin_without_a_finalisation_hook_is_unaffected():
    """The hook is optional, and a plugin defining only `expand_object`
    keeps working: `expand_model` calls what a plugin has."""

    class Minimal:
        def expand_object(self, spec, model):
            return [
                {
                    "name": spec["name"],
                    "attributes": [
                        {
                            "name": "x",
                            "kind": "bool",
                            "init": {"kind": "bool", "value": False},
                        }
                    ],
                    "ports": [],
                    "interfaces": [],
                    "automata": [],
                    "sensitive_functions": [],
                    "equations": [],
                }
            ], [], []

    pyraichu.plugins.PLUGINS["minimal_probe"] = Minimal()
    try:
        body = pyraichu.expand_model(
            {
                "name": "probe",
                "components": [],
                "connections": [],
                "indicators": [],
                "plugins": {"minimal_probe": {"objects": [{"name": "M"}]}},
            }
        )
    finally:
        del pyraichu.plugins.PLUGINS["minimal_probe"]
    assert [component["name"] for component in body["components"]] == ["M"]
    assert "evaluation_order" not in body


def test_a_boolean_model_reaches_the_document_it_always_did():
    """The finalisation engages on the continuous keys and on nothing
    else, so a purely boolean plugin model is byte for byte what it was:
    no evaluation order, and no indicator the plugin never wrote."""
    body = pyraichu.expand_model(
        {
            "name": "rbd",
            "components": [],
            "connections": [],
            "indicators": [],
            "plugins": {
                "muscadet": {
                    "objects": [
                        {
                            "type": "ObjFlow",
                            "name": "S",
                            "flows_out": [{"name": "ok", "var_prod_default": True}],
                        }
                    ]
                }
            },
        }
    )
    assert "evaluation_order" not in body
    assert body["indicators"] == []


def test_a_continuous_flow_leaving_the_plugin_is_refused():
    """A quantity crossing into a component the network does not resolve
    is accounted for nowhere. Refused, naming the component."""
    document = {
        "name": "stranded",
        "components": [
            {
                "name": "X",
                "attributes": [],
                "ports": [{"name": "w_out", "dir": "out"}],
                "interfaces": [],
                "automata": [],
                "sensitive_functions": [],
                "equations": [],
            }
        ],
        "connections": [link("X", "w_out", "T", "w_in")],
        "indicators": [],
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "T",
                        "flows_continuous_in": [{"name": "w"}],
                    }
                ]
            }
        },
    }
    with pytest.raises(ValueError, match="which this system did not build"):
        pyraichu.expand_model(document)


# --- the vocabulary is the declaration layer's own ------------------------


@pytest.mark.parametrize(
    "key,entry,message",
    [
        (
            "capacities",
            {"name": "vol", "flow": "w", "capcity": 10.0},
            "unknown declaration key",
        ),
        (
            "flows_continuous_out",
            {"name": "w", "combine": "sum"},
            "how several producers of one flow combine",
        ),
        (
            "transfers",
            {"name": "t", "flows": ["w", "w"], "equation": {"cls": "Transfer"}},
            "the one declarable shape is",
        ),
    ],
)
def test_a_continuous_section_reads_the_declaration_vocabulary(key, entry, message):
    """The six sections read muscadet's own declaration vocabulary through
    `pyraichu.declare.entry_call`, which is the ONE place a section entry
    is classified. A key that entry point refuses is refused here with the
    same reason, and the message names the object and the entry."""
    document = {
        "name": "probe",
        "components": [],
        "connections": [],
        "indicators": [],
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "P",
                        "flows_continuous_in": [{"name": "w"}],
                        "flows_continuous_out": [{"name": "w"}],
                        key: [entry],
                    }
                ]
            }
        },
    }
    with pytest.raises(ValueError, match=message):
        pyraichu.expand_model(document)


def test_the_sections_are_expanded_in_the_order_they_must_be_declared():
    """A capacity names a flow and a transfer pair refuses a flow a rule
    already carries, so the sections resolve against the ones before them.
    Written in any other order, the refusal that catches a misspelling is
    unreachable: here the spec lists them backwards and the transfer is
    still refused for the flow the rule set carries."""
    document = {
        "name": "probe",
        "components": [],
        "connections": [],
        "indicators": [],
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "P",
                        "transfers": [
                            {
                                "name": "cross",
                                "flows": ["w", "w"],
                                "equation": {
                                    "cls": "ConductiveTransfer",
                                    "conductance": 1.0,
                                    "potential_a": 1.0,
                                    "potential_b": 0.0,
                                },
                            }
                        ],
                        "rules": [
                            {
                                "name": "r",
                                "rules": [{"name": "one", "prod": {"w": 1.0}}],
                            }
                        ],
                        "flows_continuous_in": [{"name": "w"}],
                        "flows_continuous_out": [{"name": "w"}],
                    }
                ]
            }
        },
    }
    with pytest.raises(ValueError, match="cross"):
        pyraichu.expand_model(document)


def an_externally_failed_pair() -> dict:
    """Two continuous producers a common-cause `ObjFM` fails from outside.

    `external` is the behaviour that GRAFTS: it writes a mirror automaton
    and its effect function into each target component, in place, while the
    model is being expanded. The continuous rebuild answers the declaration
    alone, so the replacement has to carry the graft or the ObjFM would
    drive a control attribute nothing reads.
    """
    return {
        "name": "ccf",
        "components": [],
        "indicators": [],
        "connections": [
            link("P1", "w_out", "L", "w_in"),
            link("P2", "w_out", "L", "w_in"),
        ],
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "P1",
                        "flows_continuous_out": [{"name": "w", "var_fed_default": 3.0}],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "P2",
                        "flows_continuous_out": [{"name": "w", "var_fed_default": 3.0}],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "L",
                        "flows_continuous_in": [
                            {"name": "w", "var_demand_default": 6.0}
                        ],
                    },
                    {
                        "type": "ObjFM",
                        "name": "ccf",
                        "behaviour": "external",
                        "targets": ["P1", "P2"],
                        "failure": [
                            {"distrib": "delay", "time": 2.0},
                            {"distrib": "delay", "time": 5.0},
                        ],
                        "repair": [{"distrib": "delay", "time": 100.0}, None],
                        "failure_effects": {"w_out_rate": 0.0},
                    },
                ]
            }
        },
    }


def test_a_graft_survives_the_continuous_rebuild():
    """The hazard the two-pass expansion creates, and its guard.

    An `external` ObjFM writes into its target's component during the first
    pass, and the finalisation replaces that component with the one the
    connection list calls for. The graft has to be carried across, or it
    would be lost with nothing raised: the model would load, run, and never
    fail its pumps.
    """
    body = pyraichu.expand_model(an_externally_failed_pair())
    for name in ("P1", "P2"):
        pump = next(c for c in body["components"] if c["name"] == name)
        assert "ccf" in [automaton["name"] for automaton in pump["automata"]], (
            f"the ObjFM mirror automaton was dropped from `{name}`"
        )
        assert "apply_ccf" in [
            function["name"] for function in pump["sensitive_functions"]
        ]
        # And the component's own continuous material is there too: the
        # carry-over adds, it does not replace.
        assert "w_out__demand__L__w_in" in [
            equation["target"] for equation in pump["equations"]
        ]
        assert [allocation["name"] for allocation in pump["allocations"]] == ["w_alloc"]
    result = pyraichu.simulate(pyraichu.load_model(body), t_max=8.0, seed=1)
    delivered = dict(result.indicators["L_w_fed_in"])
    assert abs(delivered[0.0] - 6.0) < CROSSING_TOL
    # Order 1 fires at t = 2 on each pump, order 2 at t = 5: the load ends
    # up fed by nothing.
    assert abs(delivered[max(delivered)]) < CROSSING_TOL
