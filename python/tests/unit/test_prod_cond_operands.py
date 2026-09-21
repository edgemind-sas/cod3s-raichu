"""A production-condition operand that says more than a flow name.

muscadet writes a production condition as groups of operands, and an operand is
either a bare flow name or a mapping (``muscadet/obj.py``, ``apply_prod_cond``;
``muscadet/declare.py``, ``_prod_cond_spec``, which is what a declaration
actually carries). Three of the mapping's keys were refused here by name, and
the three refusals stood between a platform corpus and the engine:

- ``negate`` denies the operand, which is the ``¬`` an editor of libraries puts
  in front of one;
- ``op`` and ``value`` compare what the operand names against a threshold, in
  the very vocabulary a controller's rule guard already carries here, which is
  how the discrete and the continuous halves of a model speak to each other;
- ``port: "out"`` on a flow name the component declares **as an input too**.
  That last one was never a missing key -- it was a resolution rule. The
  default resolves the input side first, exactly as muscadet's does, and the
  layer refused rather than silently read the other side. What was missing is
  honouring an explicit selection, and the shape asking for it is the most
  ordinary one there is: a transit component, a board that feeds on and
  publishes that it does.

What this module pins, in that order, plus the control the three rest on: a
bare operand and an operand stating the default resolve as they always did, and
the document they build is unchanged, character for character. That is what
makes a red here a regression of the new reading rather than of the old one.

Oracle-free, like the rest of this directory: what is pinned is this engine's
reading of the declaration, not PyCATSHOO's agreement with it.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.declare as declare
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

#: The transit component's two flows: the one it takes in and republishes, and
#: the signal whose production is conditioned on the operand under measure.
FEED = "feed"
SIGNAL = "ctrl"


# --- the declaration under measure ------------------------------------


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


def _flow_out(name: str, prod_cond=None, prod_default: bool = False) -> dict:
    entry = {
        "cls": "FlowOut",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_default": prod_default,
        "var_prod_cond_inner_mode": "and",
        "negate": False,
    }
    if prod_cond is not None:
        entry["var_prod_cond"] = prod_cond
    return entry


def a_board(operand, *, republishes: bool = False, fed: bool = True) -> dict:
    """SOURCE -> BOARD, where the board's signal is conditioned on `operand`.

    `fed` says whether the source actually produces, which is what makes the
    same declaration answer "what does this operand mean" by running rather
    than by reading a keyword back. `republishes` adds the board's own ``feed``
    OUTPUT, the name clash the ported operand turns on -- the shape
    ``DATACENTER`` carries on ``tableau_elec`` and ``RBD_CTRL_REF`` on
    ``node``.
    """
    flows = [_flow_in(FEED), _flow_out(SIGNAL, prod_cond=[[operand]])]
    if republishes:
        flows.append(_flow_out(FEED, prod_cond=[[FEED]]))
    return {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "board",
        "components": {
            "SRC": {
                "name": "SRC",
                "cls": "ObjFlow",
                "failure_modes": [],
                "flows": [_flow_out(FEED, prod_default=fed)],
            },
            "BRD": {
                "name": "BRD",
                "cls": "ObjFlow",
                "failure_modes": [],
                "flows": flows,
            },
        },
        "connections": [
            {
                "source": "SRC",
                "source_box": f"{FEED}_out",
                "target": "BRD",
                "target_box": f"{FEED}_in",
                "flow": FEED,
            }
        ],
        "indicators": [],
    }


def signal_holds(spec: dict) -> bool:
    """Whether the board's signal is out at the initial instant of a run."""
    document = declare.build_document(spec)
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


# --- the control: what an operand said before it could say more -------


@pytest.mark.parametrize(
    "operand",
    [FEED, {"name": FEED}, {"name": FEED, "port": "in"}],
    ids=["bare", "mapping", "port-in"],
)
def test_an_operand_stating_the_default_reads_its_input(operand):
    for fed in (True, False):
        assert signal_holds(a_board(operand, fed=fed)) is fed


@pytest.mark.parametrize(
    "operand",
    [{"name": FEED}, {"name": FEED, "port": "in"}],
    ids=["mapping", "port-in"],
)
def test_an_operand_stating_the_default_builds_the_very_same_document(operand):
    """Not merely the same verdict: the same bytes.

    muscadet writes ``port`` on EVERY flow operand it hands back, so almost
    every operand of a real export states the default explicitly. Reducing
    those to the bare name is what keeps a corpus built before this reading
    existed byte-identical afterwards, and the identity is the measurement --
    equal verdicts on the two runs above would also hold for two different
    models that happen to agree at t = 0.
    """
    bare = declare.build_document(a_board(FEED))
    stated = declare.build_document(a_board(operand))

    assert json.dumps(stated, sort_keys=True) == json.dumps(bare, sort_keys=True)


# --- 1. the operand is denied -----------------------------------------


def test_a_negated_operand_denies_the_guard_it_reads():
    """muscadet's ``var_prod_cond_negate``, written inline: the board emits
    its signal exactly when its feed is ABSENT."""
    for fed in (True, False):
        assert signal_holds(a_board({"name": FEED, "negate": True}, fed=fed)) is not fed


def test_a_negated_operand_is_not_the_same_model_as_the_plain_one():
    """The counter-case to the pair above, which would also pass on a reader
    that dropped the key and inverted the fixture by accident."""
    plain = declare.build_document(a_board(FEED))
    denied = declare.build_document(a_board({"name": FEED, "negate": True}))

    assert json.dumps(denied, sort_keys=True) != json.dumps(plain, sort_keys=True)


# --- 2. the operand compares against a threshold ----------------------


@pytest.mark.parametrize(
    "op, value, expected_when_fed",
    [
        (">", 0.5, True),
        (">=", 1.0, True),
        ("<", 0.5, False),
        ("==", 1.0, True),
        ("!=", 0.0, True),
    ],
)
def test_a_comparison_reads_a_discrete_state_as_zero_or_one(
    op, value, expected_when_fed
):
    """muscadet thresholds such an operand on ``float(var_fed.value())``, so a
    state that is absent or present is compared as 0 or 1
    (``muscadet.flow._prod_cond_compare_reader``).

    It matters that this arrives as a boolean expression rather than as a
    comparison: this engine's ordering comparison is typed and refuses a
    boolean left side outright, so a condition written the muscadet way would
    otherwise build and then fail at its first evaluation.
    """
    operand = {"name": FEED, "op": op, "value": value}
    assert signal_holds(a_board(operand, fed=True)) is expected_when_fed
    assert signal_holds(a_board(operand, fed=False)) is not expected_when_fed


@pytest.mark.parametrize(
    "op, value, always",
    [(">=", 0.0, True), ("<", 0.0, False), (">", 1.0, False)],
)
def test_a_threshold_no_state_can_cross_is_the_constant_it_is(op, value, always):
    """A threshold outside ``[0, 1]`` decides the operand once and for all, and
    that is the honest reading rather than a refusal: muscadet runs such a
    condition, answering the same constant at every evaluation."""
    operand = {"name": FEED, "op": op, "value": value}
    for fed in (True, False):
        assert signal_holds(a_board(operand, fed=fed)) is always


def test_the_comparison_vocabulary_is_the_rule_guard_one():
    """Stated on the declaration rather than in a comment: the two directions
    of the discrete/continuous interoperation compare the same way."""
    assert set(declare.PROD_COND_COMPARISONS) == set(mu._CMP_OPS)


def test_an_operator_outside_that_vocabulary_is_refused_naming_it():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            a_board({"name": FEED, "op": "=~", "value": 1.0})["components"]["BRD"]
        )

    assert "=~" in str(raised.value)


def test_a_comparison_and_a_negation_on_one_operand_are_refused_together():
    """A comparison already yields a truth value, so it is denied by the
    opposite operator rather than beside it. muscadet refuses the pair at the
    same place (``muscadet.rules.check_operand_negation``), which is why
    carrying it here would carry a condition muscadet cannot read back."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            a_board({"name": FEED, "op": ">", "value": 0.5, "negate": True})[
                "components"
            ]["BRD"]
        )

    message = str(raised.value)
    assert "negate" in message or "negates" in message


def test_a_threshold_without_its_operator_is_refused():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_board({"name": FEED, "value": 1.0})["components"]["BRD"])

    assert "value" in str(raised.value)


def test_a_band_is_refused_naming_the_key():
    """``release`` is not in muscadet's operand vocabulary here, and the reason
    is the same one that keeps it out of there: a band has to be HELD between
    its two edges, and the variable this condition writes is rewritten at every
    evaluation. A rule guard carries one; this does not."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(
            a_board({"name": FEED, "op": ">", "value": 0.5, "release": 0.2})[
                "components"
            ]["BRD"]
        )

    assert "release" in str(raised.value)


# --- 2b. a threshold on a quantity that MOVES -------------------------

#: The filling volume the located crossing is measured on: empty at t = 0,
#: filled at a rate of 2, so it crosses 10 at t = 5 exactly. A closed form,
#: not a tolerance: nothing else in the model moves.
FILL_RATE = 2.0
CAPACITY = 100.0
THRESHOLD = 10.0
CROSSING = THRESHOLD / FILL_RATE


class _Filler(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="q", var_fed_default=FILL_RATE)


def _tank_class(operand):
    class Tank(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="q", var_demand_in_default=FILL_RATE)
            self.add_capacity(
                name="tank",
                flow="q",
                capacity=CAPACITY,
                side="in",
                content_init={"q": 0.0},
            )
            self.add_flow_out(name="full", var_prod_cond=[[operand]])

    return Tank


def _filling_tank(operand) -> mu.System:
    system = mu.System("threshold")
    system.add_component(_Filler, "SRC")
    system.add_component(_tank_class(operand), "TANK")
    system.connect("SRC", "q", "TANK", "q")
    return system


def test_a_threshold_on_a_moving_quantity_is_read_at_its_own_date():
    """The measurement the whole indirection exists for.

    The condition writes a variable through a sensitive function, and a
    sensitive function is re-run by a DISCRETE change: an attribute an ODE
    moves announces nothing of its own. Read straight off the quantity, this
    condition is evaluated at t = 0 and never again -- measured, the output
    stayed unfed for the whole mission with the volume at 16 against a
    threshold of 10. It reads the location of a watched threshold automaton
    instead, which is what muscadet hangs a sensitive method on for the same
    reason.
    """
    result = _filling_tank({"name": "tank", "op": ">=", "value": THRESHOLD}).simulate(
        t_max=12.0, samples=[1.0, CROSSING, CROSSING + CROSSING_TOL, 8.0]
    )

    assert sampled(result, "TANK_tank_content", CROSSING) == pytest.approx(THRESHOLD)
    assert sampled(result, "TANK_full_fed_out", 1.0) is False
    assert sampled(result, "TANK_full_fed_out", CROSSING + CROSSING_TOL) is True
    assert sampled(result, "TANK_full_fed_out", 8.0) is True


def test_the_reverse_threshold_falls_at_the_same_date():
    """The mirror, so the date above is the crossing and not the moment some
    other part of the model happened to move."""
    result = _filling_tank({"name": "tank", "op": "<", "value": THRESHOLD}).simulate(
        t_max=12.0, samples=[1.0, CROSSING + CROSSING_TOL, 8.0]
    )

    assert sampled(result, "TANK_full_fed_out", 1.0) is True
    assert sampled(result, "TANK_full_fed_out", CROSSING + CROSSING_TOL) is False
    assert sampled(result, "TANK_full_fed_out", 8.0) is False


def test_a_threshold_a_moving_quantity_cannot_be_located_on_is_refused():
    """An equality has no side a crossing is entered from, so the watched
    transition carrying it would hold only where a float landed exactly on the
    threshold. The refusal a capacity's discharge command already makes, for
    the same reason and in the same words."""
    with pytest.raises(ValueError) as raised:
        _filling_tank({"name": "tank", "op": "==", "value": THRESHOLD}).build_dict()

    assert "ordering comparison" in str(raised.value)


def test_a_boolean_operand_on_a_moving_quantity_is_refused_saying_what_to_write():
    """muscadet reads such an operand as ``!= 0``. Here the crossing has to be
    located to be read at its own date, and ``!= 0`` has no side to be entered
    from: locating it would mean choosing ``> 0`` or ``< 0``, which supposes
    the sign of the quantity in the modeller's place. So the shape is refused
    rather than read at whatever date the model happened to stop at next, and
    the refusal carries the spelling to use."""
    with pytest.raises(ValueError) as raised:
        _filling_tank({"name": "tank"}).build_dict()

    message = str(raised.value)
    assert "ordering comparison" in message
    assert "'op': '>'" in message


# --- 3. the operand selects the output side of a clashing name --------


def a_transit(operand, *, feeds_on: bool) -> dict:
    """One component carrying ``feed`` on BOTH sides, whose two sides
    DISAGREE: the output is produced unconditionally, the input is fed by
    nothing.

    The disagreement is what makes the resolution readable. The running
    example's board republishes what it takes in, so its two sides always
    agree and a reader resolving either one answers alike; this is that same
    shape with the agreement removed.
    """
    return {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "transit",
        "components": {
            "BRD": {
                "name": "BRD",
                "cls": "ObjFlow",
                "failure_modes": [],
                "flows": [
                    _flow_in(FEED),
                    _flow_out(FEED, prod_default=not feeds_on),
                    _flow_out(SIGNAL, prod_cond=[[operand]]),
                ],
            }
        },
        "connections": [],
        "indicators": [],
    }


def test_an_operand_on_the_output_side_of_a_clashing_name_reads_the_output():
    """The door the eight studies were waiting behind.

    ``{name: feed, port: out}`` on a component declaring ``feed`` as an input
    too: the input is fed by nothing and the output is produced all the same,
    so the two sides answer differently and the answer says which was read.
    """
    assert signal_holds(a_transit({"name": FEED, "port": "out"}, feeds_on=False)) is True


@pytest.mark.parametrize(
    "operand", [FEED, {"name": FEED, "port": "in"}], ids=["bare", "port-in"]
)
def test_the_default_resolution_still_reads_the_input_side_first(operand):
    """The other half, and the one that says the ported case measures the
    SELECTION: the same clashing declaration, and an operand that does not
    select reads the input, which is unfed."""
    assert signal_holds(a_transit(operand, feeds_on=False)) is False


def test_the_two_resolutions_are_two_different_models():
    ported = declare.build_document(a_transit({"name": FEED, "port": "out"}, feeds_on=False))
    default = declare.build_document(a_transit(FEED, feeds_on=False))

    assert json.dumps(ported, sort_keys=True) != json.dumps(default, sort_keys=True)


def test_the_output_side_of_a_name_carried_once_still_builds_the_bare_document():
    """``port: "out"`` on a flow this component carries ONLY as an output
    states the default, since the resolution falls through to the output side
    anyway. muscadet writes that spelling on every such operand, so reducing it
    is what leaves an export built before this reading unchanged."""
    ported = declare.build_document(a_board({"name": SIGNAL, "port": "out"}))
    bare = declare.build_document(a_board(SIGNAL))

    assert json.dumps(ported, sort_keys=True) == json.dumps(bare, sort_keys=True)


def test_an_operand_naming_no_flow_of_the_component_is_refused_naming_it():
    with pytest.raises(ValueError) as raised:
        declare.build_document(a_board({"name": "absent", "port": "out"}))

    assert "absent" in str(raised.value)
