"""The DORMANT function: an output a failure mode turns on.

A service function (a maintenance channel, a remote-diagnosis link) is
*exploitable but unused* in nominal operation: it produces nothing until
something activates it, and what activates it is a failure mode. muscadet
spells it with a production DEFAULT of ``False`` and no production
CONDITION, which leaves ``{flow}_prod_available`` with no writer at all; a
mode's effects write it, and the output starts feeding.

What makes the shape declarable is that production is held in a **variable**
rather than derived at the point of use. This suite pins the variable, its
writer, and the absence of that writer, because the three together are the
whole mechanism:

- every discrete output carries ``{flow}_prod_available``, initialised to
  ``var_prod_default``, whether or not it is dormant;
- ``update_{flow}_prod_available`` exists **only** where a production
  condition is declared. Its absence is the dormancy;
- the delivery reads the variable, never the condition again. Re-deriving the
  production where it is consumed would step over the mode's latch at the
  next evaluation.

Oracle-free, like the rest of this directory. That muscadet's own
``examples/isimu/cyber_3comp`` answers the same on both engines is the
validation suite's, under `test_muscadet_engine_parity.py`.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.declare as declare
import pyraichu.muscadet as mu
from pyraichu import plugins
from pyraichu.declare import PRODUCTION_SUFFIX

FLOW = "is_ok"
SERVICE = "svc"


# --- the generated form, read off the authoring layer -------------------


class Source(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_out(name=FLOW, var_prod_default=True)


class Server(mu.ObjFlow):
    """Two outputs, one of each regime: a mission output whose production
    follows its input, and a dormant service output that follows nothing."""

    def add_flows(self):
        self.add_flow_in(name=FLOW)
        self.add_flow_out(name=FLOW, var_prod_cond=[FLOW])
        self.add_flow_out(name=SERVICE, var_prod_default=False)


class Target(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name=SERVICE, logic="or")


def a_built_server() -> dict:
    """The ``Srv`` component of the generated model."""
    system = mu.System("dormant")
    system.add_component(Source, "S")
    system.add_component(Server, "Srv")
    system.add_component(Target, "T")
    system.auto_connect("S", "Srv")
    system.auto_connect("Srv", "T")
    built, _ = system.generate()
    return next(component for component in built if component["name"] == "Srv")


def attribute(component: dict, name: str) -> dict | None:
    return next(
        (entry for entry in component["attributes"] if entry["name"] == name), None
    )


def function(component: dict, name: str) -> dict | None:
    return next(
        (entry for entry in component["sensitive_functions"] if entry["name"] == name),
        None,
    )


def test_every_discrete_output_carries_a_production_variable():
    """Emitted on both regimes, and not only on the dormant one: what tells
    the two apart is the writer, and a variable that appeared only where it
    was latched would make the dormancy a property of the declaration reader
    rather than of the model."""
    server = a_built_server()
    for flow, default in ((FLOW, False), (SERVICE, False)):
        entry = attribute(server, f"{flow}_prod_available")
        assert entry is not None, f"`{flow}` carries no production variable"
        assert entry["kind"] == "bool"
        assert entry["init"] == {"kind": "bool", "value": default}


def test_a_production_default_is_the_variables_initial_value():
    """`var_prod_default=True` with no condition is a permanent producer, and
    it stays one: the variable starts true and nothing ever writes it."""
    system = mu.System("always_on")
    system.add_component(Source, "S")
    built, _ = system.generate()
    source = built[0]
    assert attribute(source, f"{FLOW}_prod_available")["init"] == {
        "kind": "bool",
        "value": True,
    }
    assert function(source, f"update_{FLOW}_prod_available") is None


def test_the_writer_exists_only_where_a_production_condition_is_declared():
    """The dormancy itself: no condition, no writer, so the variable holds
    whatever was last written to it -- which is muscadet's own wiring, where
    the production method is subscribed to the operands of `var_prod_cond`
    and an empty condition leaves it subscribed to nothing."""
    server = a_built_server()
    assert function(server, f"update_{FLOW}_prod_available") is not None
    assert function(server, f"update_{SERVICE}_prod_available") is None


def test_the_delivery_reads_the_variable_and_not_the_condition():
    """Read at the point of use, the condition would be re-derived at every
    evaluation and would step over a mode's latch one step after it was set."""
    server = a_built_server()
    for flow in (FLOW, SERVICE):
        delivery = function(server, f"update_{flow}_fed_out")
        terms = delivery["effects"][0]["value"]["args"]
        assert {
            "op": "attr",
            "attr": {"component": "Srv", "attribute": f"{flow}_prod_available"},
        } in terms, f"`{flow}`'s delivery does not read its production variable"


def test_the_production_writer_is_the_condition_it_replaces():
    """What moved is WHERE the condition is evaluated, not what it says."""
    server = a_built_server()
    written = function(server, f"update_{FLOW}_prod_available")["effects"][0]
    assert written["target"] == {"component": "Srv", "attribute": f"{FLOW}_prod_available"}
    assert written["value"] == {
        "op": "attr",
        "attr": {"component": "Srv", "attribute": f"{FLOW}_fed_in"},
    }


# --- the shape end to end: a mode that turns the output on --------------
#
# Declared the way `muscadet.declare.system_spec` writes it, field for
# field, because that is the document this layer actually reads.


def a_source(name="S", flow=FLOW):
    return {
        "name": name,
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowOut",
                "name": flow,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_fed_available_out_init": True,
                "var_fed_available_out_reset": True,
                "var_prod_cond_inner_mode": "or",
                "var_prod_default": True,
                "negate": False,
            }
        ],
        "capacities": [],
        "failure_modes": [],
    }


def a_dormant_server(name="Srv", prod_cond=None):
    """A component whose service output produces nothing on its own.

    ``prod_cond`` left out is the dormant regime; passed, it is the ordinary
    one, and the SAME mode then writes a variable a function also writes.
    """
    service = {
        "cls": "FlowOut",
        "name": SERVICE,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_cond_inner_mode": "or",
        "var_prod_default": False,
        "negate": False,
    }
    if prod_cond is not None:
        service["var_prod_cond"] = prod_cond
    return {
        "name": name,
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowIn",
                "name": FLOW,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": False,
                "var_available_in_default": True,
                "logic": "or",
            },
            service,
        ],
        "capacities": [],
        "failure_modes": [],
    }


def a_target(name="T"):
    return {
        "name": name,
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowIn",
                "name": SERVICE,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": False,
                "var_available_in_default": True,
                "logic": "or",
            }
        ],
        "capacities": [],
        "failure_modes": [],
    }


def an_activating_mode(name="Srv__mdc", target="Srv", **overrides):
    """The mode of the workshop's slide 51: it starts the service function
    and never repairs, which is what a compromise is."""
    spec = {
        "name": name,
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": name.split("__", 1)[-1],
        "targets": [target],
        "target_name": target,
        "failure_effects": {f"{SERVICE}_prod_available": True},
        "failure_param_name": ["ttf"],
        "failure_param": [5],
        "repair_cond": False,
        "repair_param_name": ["ttr"],
        "repair_param": [1e9],
    }
    spec.update(overrides)
    return spec


def a_document(*components):
    return {
        "version": "1.0.0",
        "name": "dormant",
        "components": {entry["name"]: entry for entry in components},
        "connections": [
            {
                "source": "S",
                "source_box": f"{FLOW}_out",
                "target": "Srv",
                "target_box": f"{FLOW}_in",
            },
            {
                "source": "Srv",
                "source_box": f"{SERVICE}_out",
                "target": "T",
                "target_box": f"{SERVICE}_in",
            },
        ],
    }


def a_session(document, t_max=20.0):
    return pyraichu.interactive(
        pyraichu.load_model(json.dumps(declare.build_document(document))), t_max=t_max
    )


def test_a_mode_that_starts_a_dormant_output_is_declarable():
    """The refusal this replaced: a `failure_effects` on
    `{flow}_prod_available` used to be met with "this layer's component does
    not carry it"."""
    document = a_document(
        a_source(), a_dormant_server(), a_target(), an_activating_mode()
    )
    built = declare.build_document(document)
    latched = next(
        component for component in built["components"] if component["name"] == "Srv"
    )
    assert f"{SERVICE}_prod_available" in {
        entry["name"] for entry in latched["attributes"]
    }, "the mode has nothing to latch"


def test_the_dormant_output_feeds_nothing_until_the_mode_fires():
    """The trajectory the shape exists for: off, then on, at the mode's own
    date and not before."""
    session = a_session(
        a_document(a_source(), a_dormant_server(), a_target(), an_activating_mode())
    )
    assert session.attribute(f"Srv.{SERVICE}_prod_available") is False
    assert session.attribute(f"T.{SERVICE}_fed_in") is False

    event = session.step()
    assert event is not None and event.time == 5.0
    assert session.attribute(f"Srv.{SERVICE}_prod_available") is True
    assert session.attribute(f"Srv.{SERVICE}_fed_out") is True
    assert session.attribute(f"T.{SERVICE}_fed_in") is True


def test_the_latch_holds_at_the_step_after_the_one_that_set_it():
    """muscadet does NOT reinitialise `var_prod_available` between steps, and
    says so where it declines to (`flow.py`: "driven by tempo mecanisms").
    Here nothing has to be exempted from a per-step reset because there is no
    such reset: an effect is HELD while the mode's state lasts. The latch is
    read once more, after an unrelated event has moved the model on, so a
    value that only survived the fixpoint that set it would show."""
    document = a_document(
        a_source(),
        a_dormant_server(),
        a_target(),
        an_activating_mode(),
        # A second mode, later and elsewhere, to force a full re-evaluation
        # after the latch -- and one that cuts the server's own input, so a
        # production that quietly depended on anything upstream would show.
        an_activating_mode(
            name="S__cut",
            target="S",
            failure_effects={f"{FLOW}_fed_available_out": False},
            failure_param=[9],
        ),
    )
    session = a_session(document)
    assert session.step().time == 5.0
    assert session.attribute(f"Srv.{SERVICE}_prod_available") is True
    assert session.step().time == 9.0
    assert session.attribute(f"S.{FLOW}_fed_out") is False, "the cut happened"
    assert session.attribute(f"Srv.{SERVICE}_prod_available") is True, (
        "the latch did not survive the next evaluation"
    )
    assert session.attribute(f"T.{SERVICE}_fed_in") is True, (
        "a dormant output a mode started does not depend on what feeds the "
        "component: it produces nothing and is switched on"
    )


def test_a_latch_a_production_condition_would_fight_over_is_refused():
    """The other regime, and why it is refused rather than answered.

    muscadet lets both writers exist and resolves them by the order events
    happen in: the production method fires when an operand changes, the mode's
    effect when the mode does. Here an effect is HELD and re-evaluated to a
    fixpoint, so it wins at every evaluation and the condition never shows.
    Measured against PyCATSHOO on this very model, the reference produces from
    t = 0 (the source feeds, so the condition holds) while this engine would
    produce only from the mode's date. Refused rather than run: a divergence a
    study cannot see is worse than a model it cannot run.
    """
    document = a_document(
        a_source(),
        a_dormant_server(prod_cond=[[{"name": FLOW, "port": "in"}]]),
        a_target(),
        an_activating_mode(),
    )
    with pytest.raises(declare.ComponentSpecError) as refused:
        declare.build_document(document)
    message = str(refused.value)
    assert f"Srv.{SERVICE}_prod_available" in message
    assert "production condition" in message
    assert "DORMANT" in message, "the refusal says what the shape is FOR"


def test_the_same_flow_without_the_condition_is_not_refused():
    """The guard is on the pair, not on the effect: dropping the condition is
    the rewrite the refusal names, and it has to be the one that works."""
    document = a_document(
        a_source(), a_dormant_server(), a_target(), an_activating_mode()
    )
    assert declare.build_document(document)["components"]


# --- the same arbitration, on the plugin route --------------------------
#
# `pyraichu.plugins.muscadet` is the OTHER way a model reaches this engine
# (the platform export), and it carries flow specs rather than mode objects
# into `finalize_model`. The question is the same and has to be asked where
# each route can see both halves of it.


def a_plugin_model(flows_out, mode_effects=None):
    objects = [{"type": "ObjFlow", "name": "SRC", "flows_out": list(flows_out)}]
    if mode_effects is not None:
        objects.append(
            {
                "type": "ObjFM",
                "name": "M",
                "targets": ["SRC"],
                "failure": [{"law": "exp", "rate": 1e-3}],
                "repair": [None],
                "failure_effects": dict(mode_effects),
            }
        )
    return {
        "name": "dormant_plugin",
        "plugins": {"muscadet": {"objects": objects}},
        "components": [],
        "connections": [],
        "indicators": [],
    }


def test_the_plugin_route_carries_a_latch_on_an_unconditioned_output():
    """No production condition, so the mode's effect is the variable's only
    writer: the ordinary dormant function, and it must build."""
    model = plugins.expand_model(
        a_plugin_model(
            [{"name": "power", "var_prod_cond": [], "var_prod_default": False}],
            mode_effects={f"power{PRODUCTION_SUFFIX}": True},
        )
    )
    source = next(c for c in model["components"] if c["name"] == "SRC")
    inits = {entry["name"]: entry["init"]["value"] for entry in source["attributes"]}
    assert inits[f"power{PRODUCTION_SUFFIX}"] is False


def test_the_plugin_route_refuses_the_latch_a_condition_would_fight_over():
    """And refuses the pair, naming the variable, the flow and the mode, the
    way the persistent availability gate is refused beside it."""
    model = a_plugin_model(
        [
            {"name": "cmd", "var_prod_cond": []},
            {"name": "power", "var_prod_cond": ["cmd"]},
        ],
        mode_effects={f"power{PRODUCTION_SUFFIX}": True},
    )
    with pytest.raises(ValueError) as refused:
        plugins.expand_model(model)
    message = str(refused.value)
    assert f"SRC.power{PRODUCTION_SUFFIX}" in message
    assert "production condition" in message
    assert "'M'" in message


def test_the_plugin_route_leaves_an_untouched_production_alone():
    """The refusal is keyed on the PAIR: a conditioned output nothing writes
    is every ordinary model, and must be emitted unchanged."""
    flows = [
        {"name": "cmd", "var_prod_cond": []},
        {"name": "power", "var_prod_cond": ["cmd"]},
    ]
    plain = next(
        c
        for c in plugins.expand_model(a_plugin_model(flows))["components"]
        if c["name"] == "SRC"
    )
    beside_a_mode = next(
        c
        for c in plugins.expand_model(
            a_plugin_model(flows, mode_effects={"power_fed_available_out": False})
        )["components"]
        if c["name"] == "SRC"
    )
    assert plain == beside_a_mode
