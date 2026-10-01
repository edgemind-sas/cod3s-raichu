"""Portable combinational gates use the same expansion as plugin models."""

import itertools
import json

import pyraichu
import pytest
from conftest import settled
from pyraichu.muscadet import declare


def gate_spec(kind="or", k=None, cond=None):
    return {
        "name": "Gate",
        "kind": "logic_gate",
        "cls": "ObjLogicGate",
        "logic_kind": kind,
        "k": k,
        "cond": cond or [],
        "out_elements": ["g"],
    }


def document(flags, kind="or", k=None):
    sources = {
        f"S{i}": {
            "name": f"S{i}",
            "flows": [{"cls": "FlowOut", "name": "feed", "var_prod_default": flag}],
        }
        for i, flag in enumerate(flags)
    }
    cond = [[{"obj": name, "attr": "feed_fed_out", "value": True}] for name in sources]
    return {
        "version": 1,
        "name": "gates",
        "generated_indicators": True,
        "components": {
            "Gate": gate_spec(kind, k, cond),
            "Sink": {"name": "Sink", "flows": [{"cls": "FlowIn", "name": "g"}]},
            **sources,
        },
        "connections": [
            {
                "source": "Gate",
                "source_box": "g_out",
                "target": "Sink",
                "target_box": "g_in",
            }
        ],
        "indicators": [],
    }


@pytest.mark.parametrize(
    "flags", list(itertools.product([False, True], repeat=3)) + [()]
)
@pytest.mark.parametrize("kind,k", [("or", None), ("and", None), ("k", 2)])
def test_gate_truth_table_reaches_actual_downstream_flow(flags, kind, k):
    model = pyraichu.load_model(
        json.dumps(declare.build_document(document(flags, kind, k)))
    )
    result = pyraichu.simulate(model, t_max=1)
    expected = (
        any(flags) if kind == "or" else all(flags) if kind == "and" else sum(flags) >= k
    )
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is expected


@pytest.mark.parametrize(
    "changes",
    [
        {"logic_kind": "xor"},
        {"logic_kind": "k", "k": 0},
        {"logic_kind": "k", "k": True},
        {"logic_kind": "k", "k": 1.5},
        {"cond": [[{"obj": "S", "attr": "feed", "ope": "!=", "value": True}]]},
        {"cond": [[{"attr": "feed", "value": True}]]},
        {"cond": "invalid"},
        {"out_elements": ["g", "g"]},
        {"cond": [[{"obj": "S", "attr": "feed", "value": float("inf")}]]},
    ],
)
def test_invalid_portable_gate_is_named_before_build(changes):
    with pytest.raises(declare.ComponentSpecError, match="Gate"):
        declare.check_spec({**gate_spec(), **changes})


@pytest.mark.parametrize("compared", [True, 1, 1.0])
def test_gate_chain_reads_named_result_independently_of_declaration_order(compared):
    spec = document([True, False], "or")
    first = spec["components"]["Gate"]
    first["name"] = "First"
    spec["components"] = {
        "Gate": gate_spec(
            "and", cond=[[{"obj": "First", "attr": "result", "value": compared}]]
        ),
        "First": first,
        **{name: comp for name, comp in spec["components"].items() if name != "Gate"},
    }
    model = pyraichu.load_model(json.dumps(declare.build_document(spec)))
    result = pyraichu.simulate(model, t_max=1)
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is True


def test_gate_reads_controller_boolean_variable_in_muscadet_spelling():
    spec = document([])
    spec["components"]["Gate"]["cond"] = [
        [{"obj": "Control", "attr": "run_signal_out", "value": True}]
    ]
    spec["components"]["Control"] = {
        "name": "Control",
        "kind": "controller",
        "cls": "ObjCtrl",
        "controls_in": [],
        "controls_out": [{"name": "run", "kind": "bool", "default": True}],
    }
    model = pyraichu.load_model(json.dumps(declare.build_document(spec)))
    result = pyraichu.simulate(model, t_max=1)
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is True


@pytest.mark.parametrize("source_value", [False, True])
@pytest.mark.parametrize("compared", [True, False, 0, 1, 1.0, 2])
def test_gate_boolean_numeric_equality(source_value, compared):
    spec = document([source_value])
    spec["components"]["Gate"]["cond"][0][0]["value"] = compared
    result = pyraichu.simulate(
        pyraichu.load_model(declare.build_document(spec)), t_max=1
    )
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is (
        source_value == compared
    )


@pytest.mark.parametrize("source_value", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("compared", [False, True])
def test_gate_numeric_boolean_equality(source_value, compared):
    spec = document([])
    spec["components"]["S0"] = {
        "name": "S0",
        "kind": "controller",
        "cls": "ObjCtrl",
        "controls_in": [],
        "controls_out": [
            {
                "name": "reading",
                "kind": "value",
                "flows": [],
                "level_default": source_value,
                "gain_default": 1.0,
            }
        ],
    }
    spec["components"]["Gate"]["cond"] = {
        "obj": "S0",
        "attr": "reading_level",
        "value": compared,
    }
    result = pyraichu.simulate(
        pyraichu.load_model(declare.build_document(spec)), t_max=1
    )
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is (
        source_value == compared
    )


def test_gate_controller_effective_name():
    spec = document([])
    spec["components"]["Gate"]["cond"] = {
        "obj": "Control",
        "attr": "run_signal_out",
        "value": True,
    }
    spec["components"]["ControllerSlot"] = {
        "name": "Control",
        "kind": "controller",
        "cls": "ObjCtrl",
        "controls_in": [],
        "controls_out": [{"name": "run", "kind": "bool", "default": True}],
    }
    result = pyraichu.simulate(
        pyraichu.load_model(declare.build_document(spec)), t_max=1
    )
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is True


def test_gate_reads_standalone_failure_and_repair_state():
    spec = document([True])
    spec["components"]["ModeSlot"] = {
        "name": "S0__fault",
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": "fault",
        "targets": ["S0"],
        "failure_param": [4],
        "repair_param": [2],
    }
    spec["components"]["Gate"]["cond"] = {
        "obj": "S0__fault",
        "attr": "occ",
        "value": True,
    }
    result = pyraichu.simulate(
        pyraichu.load_model(declare.build_document(spec)), t_max=7
    )
    assert settled(result.indicators["Sink_g_fed_in"]) == [
        (0.0, False),
        (4.0, True),
        (6.0, False),
    ]


@pytest.mark.parametrize("form", ["mapping", "flat", "compound"])
def test_gate_normalized_clauses_count_groups(form):
    spec = document([True, False], "k", 2 if form == "compound" else 1)
    yes = {"obj": "S0", "attr": "feed_fed_out", "value": True}
    no = {"obj": "S1", "attr": "feed_fed_out", "value": False}
    spec["components"]["Gate"]["cond"] = (
        yes
        if form == "mapping"
        else [yes, no]
        if form == "flat"
        else [[yes, no], [{"obj": "S1", "attr": "feed_fed_out", "value": True}]]
    )
    result = pyraichu.simulate(
        pyraichu.load_model(declare.build_document(spec)), t_max=1
    )
    assert settled(result.indicators["Sink_g_fed_in"])[-1][1] is (form != "compound")
