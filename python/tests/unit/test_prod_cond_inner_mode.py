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
import pyraichu.muscadet.declare as declare

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


# --- empty groups ----------------------------------------------------
#
# ``[[]]`` is what the platform's port editor seeded on every fresh output
# for a long time, and libraries built before still carry it: on sources (an
# output with no input at all) and on outputs whose author never wrote the
# condition. muscadet's discrete family reads it as its evaluation says
# (``muscadet/flow.py``, ``prod_cond_holds``), measured on the reference
# engine 2026-10-15:
#
# - under ``"and"``, ``any(all([]))`` HOLDS: the output produces from the
#   start, whatever its inputs and whatever its declared default;
# - under ``"or"``, ``all(any([]))`` never holds: the output never produces.


def a_source_declaring(groups, inner_mode, prod_default):
    """A component whose output declares `groups` and reads no input at all,
    the shape a library's source class carries."""
    return {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "source",
        "components": {
            "PCC": _component(
                "PCC",
                [
                    _flow_out(
                        ALARM,
                        prod_cond=groups,
                        inner_mode=inner_mode,
                        prod_default=prod_default,
                    )
                ],
            )
        },
        "connections": [],
        "indicators": [],
    }


ALL_EMPTY = {"one-empty-group": [[]], "two-empty-groups": [[], []]}


@pytest.mark.parametrize("groups", list(ALL_EMPTY.values()), ids=list(ALL_EMPTY))
def test_all_empty_groups_under_and_compile_what_an_unconditional_output_compiles(
    groups,
):
    """Under ``"and"`` a condition made of empty groups only is an empty
    conjunction, which holds: the output is unconditional. It compiles to the
    model of the output that declares no condition and is produced from the
    start, ``[]`` with ``var_prod_default=True``."""
    unconditional = declare.build_document(a_source_declaring([], "and", True))

    assert declare.build_document(a_source_declaring(groups, "and", True)) == unconditional


@pytest.mark.parametrize("groups", list(ALL_EMPTY.values()), ids=list(ALL_EMPTY))
def test_all_empty_groups_under_and_produce_whatever_the_declared_default(groups):
    """The one place the empty conjunction is NOT ``[]``: a condition that
    holds writes the production, where no condition leaves it on its declared
    default. The reference engine produces the output with the default at
    False too (measured 2026-10-15), so this does."""
    assert declare.build_document(
        a_source_declaring(groups, "and", False)
    ) == declare.build_document(a_source_declaring([], "and", True))
    assert alarm_holds(a_source_declaring(groups, "and", False)) is True


@pytest.mark.parametrize("groups", list(ALL_EMPTY.values()), ids=list(ALL_EMPTY))
def test_all_empty_groups_on_a_relay_ignore_its_unfed_input(groups):
    """The relay that a library left with ``[[]]``: its output does not wait
    for its input, exactly as the reference engine runs it."""
    flows = [_flow_in(name) for name in DETECTIONS] + [
        _flow_out(ALARM, prod_cond=groups, inner_mode="and")
    ]
    spec = a_voter([], "and")
    spec["components"]["PCC"] = _component("PCC", flows)

    assert alarm_holds(spec) is True


@pytest.mark.parametrize("groups", list(ALL_EMPTY.values()), ids=list(ALL_EMPTY))
def test_all_empty_groups_under_or_are_refused(groups):
    """Under ``"or"`` an empty clause is false and the output would never
    produce: a dead port nothing announces, which no author writes on
    purpose. The platform refuses the mode on an empty condition for the
    same reason, so this refuses it too, naming the way out."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            _component("PCC", [_flow_out(ALARM, groups, inner_mode="or")])
        )

    message = str(raised.value)
    assert "empty group" in message
    assert "'or'" in message and "never produce" in message


@pytest.mark.parametrize("inner_mode", ["and", "or"])
def test_an_empty_group_beside_a_named_operand_is_refused_on_both_readings(
    inner_mode,
):
    """A MIXED condition is refused on both readings. Under ``"and"`` the
    empty group would make the whole condition hold and silence the group
    beside it; under ``"or"`` it would make it never hold. Either way the
    operand the author did name would say nothing, so the shape is an editing
    slip rather than a declaration, and a run would hide it."""
    flows = [_flow_in(name) for name in DETECTIONS]
    groups = [[{"name": DETECTIONS[0], "port": "in"}], []]

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            _component("PCC", flows + [_flow_out(ALARM, groups, inner_mode=inner_mode)])
        )

    message = str(raised.value)
    assert "empty group" in message and "beside" in message
