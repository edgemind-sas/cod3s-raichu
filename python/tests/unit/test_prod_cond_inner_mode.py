"""``var_prod_cond_inner_mode``: which of the two readings a declared
production condition asks for.

muscadet writes a production condition as a list of groups and says under
``var_prod_cond_inner_mode`` how the two levels combine
(``muscadet/flow.py``, ``prod_cond_holds``). The key swaps BOTH at once:

- ``"or"``, its default, is ``all(any(...))`` -- a CONJUNCTION of
  disjunctions, conjunctive normal form;
- ``"and"`` is ``any(all(...))`` -- a DISJUNCTION of conjunctions, which is
  already the form the layer underneath reads.

So the key does not decorate the condition, it **is** the condition's shape,
and reading one for the other inverts it. This suite opposes the two readings
on the same groups, which is the only way to see it: with ONE operand per
group AND one group, ``[["a"]]`` means ``a`` either way, and a test built on
such a condition is green whichever reading it got.

The condition under measure, three groups of one operand each, is the shape a
platform export writes on a voter -- three detections feeding one alarm -- and
it is where the difference costs a result rather than a refusal: read as
conjunctive normal form the alarm stops as soon as ONE detection falls, and
the campaign is green on an unavailability the model does not state.

Oracle-free, like the rest of this directory: what is pinned is the reading,
not PyCATSHOO's agreement with it.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.declare as declare

#: The three detections the alarm votes over, and the alarm itself.
DETECTIONS = ("det_a", "det_b", "det_c")
ALARM = "alerte"


def _flow_in(name):
    return {
        "cls": "FlowIn",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_in_default": False,
        "var_available_in_default": True,
        "logic": "or",
    }


def _flow_out(name, prod_cond=None, inner_mode=None, prod_default=False):
    entry = {
        "cls": "FlowOut",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_default": prod_default,
        "negate": False,
    }
    if prod_cond is not None:
        entry["var_prod_cond"] = prod_cond
    if inner_mode is not None:
        entry["var_prod_cond_inner_mode"] = inner_mode
    return entry


def _component(name, flows):
    return {"name": name, "cls": "ObjFlow", "flows": list(flows), "failure_modes": []}


def a_voter(groups, inner_mode, fed=()):
    """A declaration whose alarm votes over three detections.

    ``fed`` names the detections a source actually feeds, so the same document
    answers "what does this condition mean" by running rather than by reading
    a keyword back.
    """
    components = {
        "PCC": _component(
            "PCC",
            [_flow_in(name) for name in DETECTIONS]
            + [_flow_out(ALARM, prod_cond=groups, inner_mode=inner_mode)],
        )
    }
    connections = []
    for name in fed:
        components[f"SRC_{name}"] = _component(
            f"SRC_{name}", [_flow_out(name, prod_default=True)]
        )
        connections.append(
            {
                "source": f"SRC_{name}",
                "source_box": f"{name}_out",
                "target": "PCC",
                "target_box": f"{name}_in",
                "flow": name,
            }
        )
    return {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "voter",
        "components": components,
        "connections": connections,
        "indicators": [],
    }


def alarm_condition(spec):
    """The groups the alarm's production condition reaches the layer
    underneath as, where a list of lists is the OR of conjunctions."""
    system = declare.build_system(spec)
    flow = next(f for f in system.comp["PCC"].flows_out if f.name == ALARM)
    return flow.var_prod_cond


def alarm_holds(spec):
    """Whether the alarm is fed at the initial instant of a run of `spec`."""
    document = declare.build_document(spec)
    body = pyraichu.model_body(document)
    body["indicators"] = [
        {
            "name": "alarm",
            "target": "attribute",
            "attr": {"component": "PCC", "attribute": f"{ALARM}_fed_out"},
        }
    ]
    if pyraichu.MODEL_ENVELOPE_KEY in document:
        document = pyraichu.seal(document)
    result = pyraichu.simulate(
        pyraichu.load_model(json.dumps(document)), t_max=1.0, samples=[0.0]
    )
    return bool(result.indicators["alarm"][0][1])


# --- three clauses of one operand: the voter --------------------------

#: Three groups of one operand, the shape a platform export writes on a
#: component whose output votes over its inputs.
THREE_GROUPS = [[{"name": name, "port": "in"}] for name in DETECTIONS]


def test_and_builds_the_disjunction_the_groups_already_are():
    """``"and"`` is ``any(all(...))``: the groups ARE the disjunction of
    conjunctions this layer reads, so they cross untouched."""
    assert alarm_condition(a_voter(THREE_GROUPS, "and")) == [
        [name] for name in DETECTIONS
    ]


def test_or_builds_the_conjunction_the_same_groups_mean_there():
    """``"or"`` is ``all(any(...))``: three clauses of one operand conjoined,
    which this layer spells as ONE group of three."""
    assert alarm_condition(a_voter(THREE_GROUPS, "or")) == [list(DETECTIONS)]


def test_the_default_is_the_conjunctive_reading():
    """A declaration that states no mode means muscadet's own default, and
    that default is the CNF reading, not the one the platform writes."""
    assert alarm_condition(a_voter(THREE_GROUPS, None)) == alarm_condition(
        a_voter(THREE_GROUPS, "or")
    )


@pytest.mark.parametrize(
    "inner_mode, expected",
    [("and", True), ("or", False)],
    ids=["disjunction-holds", "conjunction-falls"],
)
def test_one_detection_out_of_three_decides_the_alarm_differently(
    inner_mode, expected
):
    """The number the two readings disagree on, measured on a run rather than
    on a keyword: with ONE detection fed, the disjunction alerts and the
    conjunction does not. This is the pair the reference corpus turns on --
    lost, the alarm reads unavailable while two of its three detections hold,
    and the campaign is green on an unavailability the model never stated."""
    assert alarm_holds(a_voter(THREE_GROUPS, inner_mode, fed=DETECTIONS[:1])) is expected


def test_all_three_detections_agree_whichever_reading_was_asked_for():
    """The counter-case that says the pair above measures the READING and not
    the wiring: fed all three, both readings hold."""
    for inner_mode in ("and", "or"):
        assert alarm_holds(a_voter(THREE_GROUPS, inner_mode, fed=DETECTIONS)) is True


# --- a nested condition: both levels swap -----------------------------

#: Two groups, the first carrying two operands: the shape where BOTH levels
#: of the key are visible at once.
NESTED_GROUPS = [
    [{"name": DETECTIONS[0], "port": "in"}, {"name": DETECTIONS[1], "port": "in"}],
    [{"name": DETECTIONS[2], "port": "in"}],
]


def test_a_nested_condition_read_as_a_disjunction_keeps_its_groups():
    """``(a and b) or c``: already the form underneath, one group of two and
    one of one."""
    assert alarm_condition(a_voter(NESTED_GROUPS, "and")) == [
        [DETECTIONS[0], DETECTIONS[1]],
        [DETECTIONS[2]],
    ]


def test_a_nested_condition_read_as_a_conjunction_is_expanded():
    """``(a or b) and c``: one conjunction per choice of a single operand from
    each clause, which is the same proposition written the other way up."""
    assert alarm_condition(a_voter(NESTED_GROUPS, "or")) == [
        [DETECTIONS[0], DETECTIONS[2]],
        [DETECTIONS[1], DETECTIONS[2]],
    ]


@pytest.mark.parametrize(
    "inner_mode, expected",
    [("and", True), ("or", False)],
    ids=["disjunction-holds", "conjunction-falls"],
)
def test_the_nested_condition_answers_differently_on_its_first_group(
    inner_mode, expected
):
    """Fed `a` and `b` and not `c`: `(a and b) or c` holds, `(a or b) and c`
    does not."""
    spec = a_voter(NESTED_GROUPS, inner_mode, fed=DETECTIONS[:2])
    assert alarm_holds(spec) is expected


# --- the shapes the key is refused on ---------------------------------


def test_a_third_spelling_is_refused_naming_the_two_that_exist():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            _component("PCC", [_flow_out(ALARM, THREE_GROUPS, inner_mode="xor")])
        )

    message = str(raised.value)
    assert "var_prod_cond_inner_mode" in message
    assert "'or'" in message and "'and'" in message


def test_a_third_spelling_is_refused_on_a_flow_carrying_no_condition():
    """Refused on the KEY and not on the condition it shapes: a flow declaring
    a mode and no condition is a declaration that says something nobody can
    honour, and letting it through would leave the misspelling to be found on
    the next flow that does carry one."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(_component("PCC", [_flow_out(ALARM, inner_mode="xor")]))

    assert "var_prod_cond_inner_mode" in str(raised.value)


def test_the_expansion_ceiling_does_not_apply_to_a_disjunctive_condition():
    """The ceiling exists for the CNF expansion, whose width is the product of
    the clause sizes. A condition already disjunctive multiplies nothing out,
    so it is carried whole where the same groups read as CNF are refused."""
    wide = [
        [{"name": name, "port": "in"} for name in DETECTIONS] for _ in range(12)
    ]
    flows = [_flow_in(name) for name in DETECTIONS]

    assert (
        declare.check_spec(
            _component("PCC", flows + [_flow_out(ALARM, wide, inner_mode="and")])
        )
        == "PCC"
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            _component("PCC", flows + [_flow_out(ALARM, wide, inner_mode="or")])
        )

    assert "disjunctions" in str(raised.value)


@pytest.mark.parametrize("inner_mode", ["and", "or"])
def test_an_empty_group_is_refused_on_both_readings(inner_mode):
    """The one malformed shape that used to run to completion without a word,
    and it did so DIFFERENTLY on the two readings.

    Under ``"and"`` an empty group reaches the layer underneath as an empty
    conjunction, which holds. Under ``"or"`` the expansion multiplies it out
    and a product over an empty clause is EMPTY, so the condition disappears
    entirely: no writer at all, the production variable left on its declared
    default. That is the DORMANT function, and the exact opposite of the empty
    disjunction the group states, which is false.

    **muscadet refuses the same group, for the same reason, on the family it
    left for later.** ``FlowContinuousOut.check_prod_cond_shape`` says "under
    inner mode 'or' the output would never produce, under 'and' the condition
    would never bind: neither is a declaration", and scopes itself to the
    continuous classes because "the discrete classes are 1.x surface with the
    same laxity, and tightening them belongs to its own change". So this pins
    the discrete half of ONE refusal, not a stricture of this reader's own.
    """
    flows = [_flow_in(name) for name in DETECTIONS]
    groups = [[{"name": DETECTIONS[0], "port": "in"}], []]

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            _component("PCC", flows + [_flow_out(ALARM, groups, inner_mode=inner_mode)])
        )

    assert "empty group" in str(raised.value)
