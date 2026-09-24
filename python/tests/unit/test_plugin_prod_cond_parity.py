"""Two routes declare a production condition, and only one carries the key
that decides how it is read.

The serialized plugin's ``flows_out`` section is this layer's OWN vocabulary,
and it reads a production condition in disjunctive form: the groups OR-ed, each
group's operands AND-ed, exactly as
:meth:`~pyraichu.muscadet.ObjFlow.add_flow_out` documents. The muscadet-facing
route (``pyraichu.declare``, a component's ``flows`` section) reads the same
list under ``var_prod_cond_inner_mode``, whose muscadet default ``"or"`` means
the OPPOSITE, ``all(any(...))``, and converts it.

So ``[["a"], ["b"]]`` is ``a or b`` on the plugin route and ``a and b`` on the
muscadet route at its default, and **both are right**: the two sections carry
two conventions on purpose. A recorded industrial model settles which one the
plugin section has:
its six multi-clause conditions carry no mode key and are shapes like
``[["grid_a"], ["grid_b"]]``, a supply fed by either grid, and its goldens were
cross-validated against PyCATSHOO in that reading.

What was wrong is narrower and is what this module pins. The plugin section
accepted ``var_prod_cond_inner_mode`` and **ignored it** -- any value, including
the ``"or"`` that asks for the other reading, and including a value that is no
mode at all. A model author writing the muscadet key there got their condition
read as very nearly its own negation, with nothing refused and nothing logged.
It is now refused above the value that states what the section already does, per
the seam's refuse-by-value rule.

What the two routes DO share is the operand vocabulary, and that is pinned here
too: a negated, compared or ported operand means the same thing whichever
section declared it. Those assertions are the control. They passed before this
change and must keep passing, so a red among them is a regression of the operand
reading rather than of the refusal added beside it.

Oracle-free, like the rest of this directory.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.declare as declare

#: The board's two inputs and the signal whose production they condition.
A = "a"
B = "b"
SIGNAL = "ctrl"


# --- the two declarations of one board --------------------------------


def _flow_in(name: str) -> dict:
    return {
        "cls": "FlowIn",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "var_in_default": False,
        "var_available_in_default": True,
        "logic": "or",
    }


def _flow_out(
    name: str,
    prod_cond=None,
    prod_default: bool = False,
    inner: str | None = None,
) -> dict:
    entry = {
        "cls": "FlowOut",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_default": prod_default,
        "negate": False,
    }
    if inner is not None:
        entry["var_prod_cond_inner_mode"] = inner
    if prod_cond is not None:
        entry["var_prod_cond"] = prod_cond
    return entry


def declared_spec(cond, inner, *, fed: dict, republishes: bool = False) -> dict:
    """The board as a whole-component declaration: the ``flows`` section."""
    flows = [_flow_in(name) for name in sorted(fed)]
    flows.append(_flow_out(SIGNAL, prod_cond=cond, inner=inner))
    if republishes:
        flows.append(_flow_out(A, prod_cond=[[A]], inner="and"))
    components = {
        f"SRC_{name}": {
            "name": f"SRC_{name}",
            "cls": "ObjFlow",
            "failure_modes": [],
            "flows": [_flow_out(name, prod_default=value)],
        }
        for name, value in fed.items()
    }
    components["BRD"] = {
        "name": "BRD",
        "cls": "ObjFlow",
        "failure_modes": [],
        "flows": flows,
    }
    return {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "board",
        "components": components,
        "connections": [
            {
                "source": f"SRC_{name}",
                "source_box": f"{name}_out",
                "target": "BRD",
                "target_box": f"{name}_in",
                "flow": name,
            }
            for name in sorted(fed)
        ],
        "indicators": [],
    }


#: The three shapes a ``flows_out`` entry may take, each reaching a DIFFERENT
#: authoring method. The guard sits above that dispatch, and only a per-shape
#: assertion pins it there rather than inside one arm: moving the call into the
#: plain branch leaves a suite that tests the plain shape alone fully green
#: while a triggered output goes back to the misreading.
SHAPES = {
    "plain": {},
    "tempo": {"tempo": {"enable_time": 1.0, "disable_time": 1.0}},
    "trigger": {"trigger": {"time_up": 1.0, "time_down": 1.0}},
}


def plugin_spec(
    cond,
    inner,
    *,
    fed: dict,
    republishes: bool = False,
    shape: str = "plain",
    extra: dict | None = None,
) -> dict:
    """The same board as plugin data: the ``flows_out`` section."""
    signal = {"name": SIGNAL, "var_prod_cond": cond, **SHAPES[shape]}
    if inner is not None:
        signal["var_prod_cond_inner_mode"] = inner
    if extra:
        signal.update(extra)
    flows_out = [signal]
    if republishes:
        flows_out.append({"name": A, "var_prod_cond": [[A]]})
    objects = [
        {
            "type": "ObjFlow",
            "name": f"SRC_{name}",
            "flows_out": [{"name": name, "var_prod_default": value}],
        }
        for name, value in fed.items()
    ]
    objects.append(
        {
            "type": "ObjFlow",
            "name": "BRD",
            "flows_in": [{"name": name} for name in sorted(fed)],
            "flows_out": flows_out,
        }
    )
    return {
        "name": "board",
        "plugins": {"muscadet": {"objects": objects}},
        "components": [],
        "connections": [
            {
                "from": {"component": f"SRC_{name}", "port": f"{name}_out"},
                "to": {"component": "BRD", "port": f"{name}_in"},
            }
            for name in sorted(fed)
        ],
        "indicators": [],
    }


def _signal_holds(document: dict) -> bool:
    """Whether the board's signal is out at the initial instant of a run."""
    body = pyraichu.model_body(document)
    body["indicators"] = [
        {
            "name": SIGNAL,
            "target": "attribute",
            "attr": {"component": "BRD", "attribute": f"{SIGNAL}_fed_out"},
        }
    ]
    if pyraichu.MODEL_ENVELOPE_KEY in document:
        document = pyraichu.seal(document)
    result = pyraichu.simulate(
        pyraichu.load_model(json.dumps(document)), t_max=1.0, samples=[0.0]
    )
    return bool(result.indicators[SIGNAL][0][1])


def via_declaration(cond, inner, **kwargs) -> bool:
    return _signal_holds(declare.build_document(declared_spec(cond, inner, **kwargs)))


def via_plugin(cond, inner, **kwargs) -> bool:
    return _signal_holds(pyraichu.expand_model(plugin_spec(cond, inner, **kwargs)))


#: The two-clause condition the two conventions read differently.
TWO_CLAUSES = [[A], [B]]

#: The four feed states, as (a, b).
FEEDS = [(True, True), (True, False), (False, True), (False, False)]


# --- 1. the two conventions, each pinned where it applies -------------


@pytest.mark.parametrize("a,b", FEEDS, ids=str)
def test_the_plugin_section_reads_a_condition_disjunctively(a, b):
    """This layer's own convention: the groups are OR-ed.

    ``[["a"], ["b"]]`` is two one-operand groups, so the board produces on
    either input. This is the reading that recorded model was cross-validated in,
    and
    the physical one for its shape: a supply on two grids works on either.
    """
    assert via_plugin(TWO_CLAUSES, None, fed={A: a, B: b}) is (a or b)


@pytest.mark.parametrize("a,b", FEEDS, ids=str)
def test_the_muscadet_route_conjoins_the_clauses_at_its_own_default(a, b):
    """muscadet's convention, converted: the clauses are AND-ed.

    Same list, opposite reading, and the difference is not a bug on either
    side: the muscadet-facing route carries the key that says so and honours
    it, which is why the same bytes may mean two things.
    """
    assert via_declaration(TWO_CLAUSES, None, fed={A: a, B: b}) is (a and b)


@pytest.mark.parametrize("a,b", FEEDS, ids=str)
def test_the_two_routes_agree_once_the_conventions_do(a, b):
    """``"and"`` on the muscadet route states this layer's own reading, so
    there is nothing left to convert and both routes answer alike.

    This is what a platform export carries on every discrete output, which is
    why the two conventions coexist without a corpus noticing.
    """
    fed = {A: a, B: b}

    assert via_declaration(TWO_CLAUSES, "and", fed=fed) is (a or b)
    assert via_plugin(TWO_CLAUSES, "and", fed=fed) is (a or b)


# --- 2. the defect: a key accepted and ignored ------------------------


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=str)
@pytest.mark.parametrize("declared", ["or", "nonsense", ""], ids=repr)
def test_the_plugin_section_refuses_a_mode_it_will_not_honour(declared, shape):
    """The regression under repair: any value at all was accepted and
    dropped, so a condition written in muscadet's convention ran as very
    nearly its own negation.

    ``"or"`` is the one that costs a wrong answer, and it is muscadet's own
    default, so it is what a model author transcribing a library writes. The
    others are here because the silent drop was indiscriminate: a value that
    is no mode was accepted just as quietly, and a refusal that caught only
    the meaningful spelling would still pass a typo through.

    Run over all three entry shapes because each reaches a different
    authoring method. A guard that sat inside the plain branch instead of
    above the dispatch would leave a plain-only suite green while a
    temporised or triggered output returned to the misreading, so the shape
    axis is what pins the placement rather than the behaviour.

    The message is matched on the guidance and not only on the key, because
    a refusal that names the key without naming what to write instead is the
    half of this fix that a reader cannot act on.
    """
    with pytest.raises(ValueError, match=r"var_prod_cond_inner_mode.*DIFFERENT"):
        via_plugin(TWO_CLAUSES, declared, fed={A: True, B: False}, shape=shape)


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=str)
def test_the_carried_mode_is_accepted_on_every_entry_shape(shape):
    """The other half of the placement pin: the guard must not refuse the
    value it carries on a shape it was never exercised against.

    Accepted means the entry builds and runs, which is all the guard decides.
    What the signal then reads is the shape's own business and not this
    module's: a temporised output waits out its enable delay and a triggered
    one is inhibited while its trigger holds, so neither feeds at t = 0 and
    asserting that it did would pin the wrong thing.
    """
    document = pyraichu.expand_model(
        plugin_spec(TWO_CLAUSES, "and", fed={A: True, B: False}, shape=shape)
    )

    assert isinstance(_signal_holds(document), bool)


# --- 2b. the spelling decides which value is the carried one ----------


#: The FLAT spelling of a two-operand condition. This section reads it as ONE
#: group, its operands AND-ed, while the muscadet-facing route splits the same
#: list into one clause per element and so reaches that same answer only under
#: ``"or"``. The carried value therefore FLIPS with the spelling, which a single
#: hard-coded value gets backwards on exactly this shape.
TWO_OPERANDS_FLAT = [A, B]


@pytest.mark.parametrize("a,b", FEEDS, ids=str)
def test_a_flat_condition_conjoins_its_operands_on_this_route(a, b):
    """One group, operands AND-ed, which is what this layer documents for a
    flat list and the opposite of what the nested two-group spelling means.
    """
    assert via_plugin(TWO_OPERANDS_FLAT, None, fed={A: a, B: b}) is (a and b)


def test_a_flat_condition_carries_the_other_mode():
    """The measured flip, and the case a single carried value inverts.

    On a flat condition the muscadet-facing route agrees with this section
    under ``"or"`` and disagrees under ``"and"``, so ``"or"`` is the value
    that states this reading and ``"and"`` is the one that would run the
    condition as a disjunction. A guard keyed on the nested spelling alone
    would bless the inverting value here and refuse the agreeing one.
    """
    fed = {A: True, B: False}

    assert via_declaration(TWO_OPERANDS_FLAT, "or", fed=fed) is via_plugin(
        TWO_OPERANDS_FLAT, None, fed=fed
    )
    assert via_plugin(TWO_OPERANDS_FLAT, "or", fed=fed) is False

    with pytest.raises(ValueError, match=r"var_prod_cond_inner_mode"):
        via_plugin(TWO_OPERANDS_FLAT, "and", fed=fed)


@pytest.mark.parametrize("mode", ["or", "and"], ids=str)
def test_a_single_operand_carries_either_mode(mode):
    """With one operand there is nothing to distribute, so the two readings
    cannot differ and neither value is the wrong one to write.
    """
    assert via_plugin([A], mode, fed={A: True}) is True


@pytest.mark.parametrize("mode", ["or", "and"], ids=str)
def test_a_condition_that_declares_nothing_carries_either_mode(mode):
    """The seam refuses by value, never by key.

    An entry with an empty condition combines nothing, so no mode written
    beside it can change what the entry computes, and refusing one would
    refuse a model that is right. This matters beyond tidiness: recorded
    industrial models carry `var_prod_cond: []` entries, and an exporter that writes gate
    keys unconditionally is the shape that has already refused a whole
    corpus once in this layer's history.
    """
    spec = plugin_spec([], mode, fed={A: True})
    signal = spec["plugins"]["muscadet"]["objects"][-1]["flows_out"][0]
    signal["var_prod_default"] = True

    assert _signal_holds(pyraichu.expand_model(spec)) is True


# --- 2c. the rest of the family, refused the same way -----------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("var_prod_cond_negate", [[True]]),
        ("var_prod_cond_compare", [[{"op": ">", "value": 0}]]),
        ("negate", True),
    ],
    ids=["negate-matrix", "compare-matrix", "flow-negate"],
)
def test_the_rest_of_the_family_is_refused_when_it_declares_something(key, value):
    """Three more keys this section builds nothing from.

    Two of them are the operand negation and comparison written as a matrix
    beside the condition, where the operand carries the same thing inline.
    Dropped rather than refused, they cost the same wrong answer the inner
    mode does: a condition running unnegated or uncompared. The
    muscadet-facing route already treats all three as inert at the value that
    declares nothing and refuses them above it, so this is that treatment
    reaching the other route rather than a new rule.
    """
    with pytest.raises(ValueError, match=key):
        via_plugin(TWO_CLAUSES, None, fed={A: True, B: False}, extra={key: value})


def test_a_negated_output_is_refused_without_being_sent_to_the_operand():
    """``negate`` is not of the production-condition family, and the refusal
    must not pretend it is.

    muscadet defines it as negating the flow OUTPUT, not anything inside the
    condition, and this layer carries that on no route. An author told to
    move it onto an operand would be sent to a key that means something
    else, so the message offers no substitute for this one.
    """
    with pytest.raises(ValueError, match=r"negates the flow OUTPUT") as raised:
        via_plugin(TWO_CLAUSES, None, fed={A: True, B: False}, extra={"negate": True})

    assert "write it on the operand" not in str(raised.value)


@pytest.mark.parametrize(
    "key,inert",
    [
        ("var_prod_cond_negate", []),
        ("var_prod_cond_compare", []),
        ("negate", False),
    ],
    ids=["negate-matrix", "compare-matrix", "flow-negate"],
)
def test_the_rest_of_the_family_is_accepted_at_its_inert_value(key, inert):
    """The read-back writes every key on every entry, so refusing one that
    declares nothing would refuse the corpus at its first component. These
    are the values muscadet's own read-back writes when there is nothing to
    say.
    """
    assert (
        via_plugin(TWO_CLAUSES, None, fed={A: True, B: False}, extra={key: inert})
        is True
    )


def test_the_carried_mode_and_its_absence_are_the_same_model():
    """Stating this section's own reading is accepted, and says nothing.

    Byte identity rather than an equal verdict: two different models can
    agree at one instant, and what is pinned is that the refusal added
    beside the key did not start moving the models that keep it.
    """
    fed = {A: True, B: False}
    absent = pyraichu.expand_model(plugin_spec(TWO_CLAUSES, None, fed=fed))
    stated = pyraichu.expand_model(plugin_spec(TWO_CLAUSES, "and", fed=fed))

    assert json.dumps(stated, sort_keys=True) == json.dumps(absent, sort_keys=True)


def test_a_condition_free_output_is_unaffected_by_the_refusal():
    """An output with no production condition at all still builds: the
    refusal is keyed on the mode being declared, not on the section.
    """
    spec = plugin_spec(None, None, fed={A: True})
    signal = spec["plugins"]["muscadet"]["objects"][-1]["flows_out"][0]
    del signal["var_prod_cond"]
    signal["var_prod_default"] = True

    assert _signal_holds(pyraichu.expand_model(spec)) is True


# --- 3. the control: the operand vocabulary is shared -----------------


@pytest.mark.parametrize(
    "operand,holds_when_fed",
    [
        (A, True),
        ({"name": A}, True),
        ({"name": A, "port": "in"}, True),
        ({"name": A, "negate": True}, False),
        ({"name": A, "op": ">", "value": 0}, True),
    ],
    ids=["bare", "mapping", "port-in", "negated", "compared"],
)
def test_a_structured_operand_reads_alike_on_both_routes(operand, holds_when_fed):
    """Each mapping form means on the plugin route what it means on the
    muscadet route. One group of one operand, so the two conventions cannot
    disagree and the operand is all that is measured.

    The negated operand is the discriminator: an operand dropped rather than
    read would agree with the plain one, and agreeing in BOTH feed states is
    what separates "read the same way" from "read at all".
    """
    for fed in (True, False):
        expected = holds_when_fed if fed else not holds_when_fed

        assert via_declaration([[operand]], "and", fed={A: fed}) is expected
        assert via_plugin([[operand]], "and", fed={A: fed}) is expected


def test_the_output_side_of_an_input_only_name_is_refused_on_both_routes():
    """Asking for a side the component does not carry is refused, and
    refused on both routes.

    The muscadet route KEEPS an explicit ``port: "out"`` exactly when the
    name is an input too, which is what makes the selection load-bearing
    rather than decorative; the plugin route hands it on. Neither may end up
    resolving a side that is not there.
    """
    operand = {"name": A, "port": "out"}

    for build in (via_declaration, via_plugin):
        with pytest.raises(ValueError, match=r"reads `a`"):
            build([[operand]], "and", fed={A: True})


def test_a_clashing_name_reads_the_output_side_on_both_routes():
    """The transit shape: a board that feeds on a name and republishes it.

    Here the two sides are different variables, so the explicit selection is
    load-bearing, and it is the case the muscadet route keeps the key for.
    Both routes must build a model that runs and agrees.
    """
    operand = {"name": A, "port": "out"}

    for fed in (True, False):
        declared = via_declaration([[operand]], "and", fed={A: fed}, republishes=True)
        plugged = via_plugin([[operand]], "and", fed={A: fed}, republishes=True)

        assert declared is plugged
