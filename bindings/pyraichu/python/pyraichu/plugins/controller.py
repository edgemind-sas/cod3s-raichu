"""ObjCtrl: the controller, a peer of ObjFlow.

A muscadet model carries two natures of thing. ``ObjFlow`` transports a
**conserved quantity**: what leaves one component arrives at another, so a
flow enters allocation and the balance sweeps. A controller transports a
**reading or a signal**: nothing is conserved, nothing is allocated, and
republishing a value creates no matter. That is why ``ObjCtrl`` is a peer of
``ObjFlow`` and not a subclass of it, here as in muscadet.

Why the port exists
-------------------
Every automaton the muscadet layer derives comes from a declaration that
needs one (a failure mode, a capacity bound, a rule selection), and a mode
transition's guard reads only the target rule, never the source state, so a
mode holds no memory: a guard switches back the instant its condition stops
holding. The heated-tank regulation (pumps below 6, valve above 8) therefore
does not port: the level freezes at 6 and never cycles. What is missing is a
**declarable automaton with continuous guards and two-threshold memory**,
which is what :class:`CtrlBand` is.

What a controller declares
--------------------------
Two sections.

* ``controls_in``: **observation inputs**. An input is a measurement channel,
  so a controller reads a capacity level (``kind="level"``), the rate a
  continuous output delivers (``kind="rate"``) or the share one constituent
  is of what a volume holds (``kind="ratio"``) through one concept and one
  wire shape. The link carries **no quantity**: the ports it joins declare no
  channel, so the reader creates no demand and enters no allocation operator.
  An input declaring an ``aggregate`` reduces several sources to one value.

  **A ratio input is what a closed grammar costs, and where that cost is
  paid.** There is no division among the four operators below: a quotient
  carries no threshold anything could root-find, so admitting it would break
  the property the whole grammar exists for. A fraction is materialised by
  the volume that holds the constituents and reaches the controller as an
  ordinary reading, which a band then thresholds like any other.

* ``controls_out``: **outputs**, of two natures. A ``"bool"`` output is one
  boolean attribute exported on ``{name}_out``, exactly what a component's
  discrete in-flow imports, so a controller drives a source with no new
  mechanism on the driven side. A ``"value"`` output publishes a number on
  ``{name}_level_out``, which is what lets the output of one controller be
  the input of another, and what an instrument publishes to a voter.

**Why an input is not read by a sensitive function.** RAICHU derives *when*
to run a sensitive function from the attributes and states its expressions
read, and it runs no callback during numerical integration: a continuous
quantity moving inside an integration step announces nothing. A sensitive
function reading a level would therefore never re-evaluate between discrete
events, and the failure would be silent. A controller re-evaluates on the
**notification of the automata its grammar compiles to**, whose transitions
are ``watched``, so a crossing is located by root-finding rather than
noticed at whatever event happens to come next.

**What an output carries: a closed grammar, never a function.** An output
declares its value under ``emit``, and what may be written there is a
composition of exactly four operators: :class:`CtrlCompare` (a reading
against a threshold), :class:`CtrlBand` (two thresholds and a direction),
:class:`CtrlCombine` (booleans by and / or / not / k-of-n) and
:class:`CtrlRepublish` (a reading, times a gain). A Python callable is
refused there, and so is anything that is not one of the four.

The reason is the one the whole module turns on. The engine dates a crossing
exactly only on a form it **recognises**: a threshold it can root-find, an
edge it can watch. Nothing can read a threshold out of arbitrary Python, so
nothing could compile one to a watched transition, and the fallback (a
sensitive function on the reading) is precisely the silent failure above. A
closed grammar is what makes every form compilable.

**Every number the grammar carries is an attribute of the model.** A
threshold, a band's two edges and a republication's gain are component
attributes, not constants inlined into the expressions that read them. One
change, three consequences: two instances of one declaration can be tuned to
different thresholds, an indicator can name a threshold as its target, and a
**failure mode can move one**. Three further endpoints reach an output, each
an ordinary attribute an ``ObjFM`` names by its exact name:

============================== =========================================
``{output}_level_gain``        a value output's publication, scaled
``{output}_forced``            with ``{output}_forced_value``: that
``{output}_forced_value``      publication, replaced
``{output}_signal_available``  a boolean output, blinded
============================== =========================================

Where this port diverges from muscadet, and why
-----------------------------------------------
Three places, each forced by an engine that answers the same question
differently, none of them a weakening.

1. **A comparison reads its automaton's state, not the comparison again.**
   muscadet's compare closure recomputes the comparison live and uses the
   automaton only for the stop and the notification. Here the emission is a
   sensitive function whose triggers RAICHU derives from what it reads, so
   recomputing the comparison would make it trigger on the *reading*, which
   announces nothing between events. Reading the state instead makes the
   located crossing the notification, exactly as it already is for a band.

2. **Blinding needs no automaton of its own.** muscadet carries one two-state
   automaton per boolean output whose state *is* "this output is blinded",
   because a PyCATSHOO sensitive method must be attached to something that
   announces, and the returning edge is load-bearing: a signal variable is
   not reinitialised, so releasing the clamp puts nothing back. Here the
   emission reads ``{output}_signal_available`` directly and RAICHU derives
   the trigger from that read, on both edges. The automaton would carry no
   information the read does not already carry.

3. **``min`` and ``max`` aggregates are refused rather than approximated.**
   See :data:`CONTROL_AGGREGATIONS`.

The order the controllers run in
--------------------------------
muscadet derives it from a topological sort of the signal graph at a pre-run
step, once every connection exists. A plugin expands **one object at a
time** and has no pre-run step, so the order here follows declaration order,
which costs two rules a model has to keep:

* a controller reading another controller's value output is declared after
  it, so the reading is swept downstream of the publication;
* on a model that carries an ``evaluation_order``, an object that emits
  explicit equations of its own is declared **before** the controllers,
  because a controller takes the field over as it stands. An object landing
  after one leaves the order short of its steps, which the engine refuses by
  name at load rather than silently completing.

The rest is settled by construction, a controller's component being appended
after everything already in the model, so an input is always swept
downstream of the level it mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["expand_objctrl"]


# ----------------------------------------------------------------------
# The declaration vocabulary
# ----------------------------------------------------------------------

#: A measurement input reads a capacity **level**: an integrated state.
MEASUREMENT_LEVEL = "level"

#: A measurement input reads the **rate** a continuous output delivers.
MEASUREMENT_RATE = "rate"

#: A measurement input reads the **ratio** one constituent is of what a
#: volume holds: a dimensionless fraction, published per constituent by the
#: volume itself. It exists so that a controller can threshold a fraction,
#: the output grammar being closed at four operators, none of them
#: arithmetic.
MEASUREMENT_RATIO = "ratio"

#: Every nature a measurement input may be declared with.
MEASUREMENT_KINDS = (MEASUREMENT_LEVEL, MEASUREMENT_RATE, MEASUREMENT_RATIO)

#: The closed list an observation input's aggregation is chosen from, under
#: the names muscadet gives them.
#:
#: **``min`` and ``max`` are declarable and refused**, with the reason named.
#: The only variable-arity reader of an in port is the ``port_agg``
#: expression operator, whose aggregations are ``sum``, ``count``, ``all``,
#: ``any``, ``mean`` and ``median``: there is no minimum and no maximum, and
#: a measurement link carries no per-connection channel a fixed-arity
#: ``min`` expression could read one by one. Refused rather than
#: approximated by a sum or a mean, which would answer a different question
#: in silence.
#:
#: What each surviving name means is settled by the engine, notably the
#: **median of an even count**, which is the mean of the two central
#: readings and not a tie-break: muscadet's ``combine`` and RAICHU's
#: ``AggOp::Median`` agree on that, so an even count needs no caution here
#: beyond knowing that it interpolates.
CONTROL_AGGREGATIONS = ("min", "max", "mean", "median", "sum")

#: The aggregations this port compiles, and the ``port_agg`` operator each
#: becomes.
CONTROL_AGGREGATION_OPS = {"sum": "sum", "mean": "mean", "median": "median"}

#: The aggregations that are **sensitive to rank**, and therefore the ones a
#: crossing of two readings makes non-differentiable: a minimum or a maximum
#: changes argument, a median changes representative. With ``min`` and
#: ``max`` refused, ``median`` is the only one that reaches the engine.
#:
#: muscadet declares one watched two-state automaton per **pair** of sources
#: so the solver stops at the crossing. That is deliberately **not ported**:
#: the pair count is a function of how many connections reach the input, and
#: a per-object plugin expansion does not know it (the model's connections
#: are written beside the plugin section, and later objects may add more).
#: The reduced value stays continuous through such a point, so what is lost
#: is the derivative, not the reading.
AGGREGATION_KINK_POLICIES = ("min", "max", "median")

#: A controller output carrying a **boolean** signal, exported on
#: ``{name}_out``: the very port shape a discrete in-flow imports, so an
#: in-flow consumes a controller's output with no adapter.
CTRL_OUT_BOOL = "bool"

#: A controller output carrying a **number**, published on
#: ``{name}_level_out``. Indistinguishable from a capacity's own publication
#: by whoever observes it, which is what makes a chain of controllers
#: possible.
CTRL_OUT_VALUE = "value"

#: Every nature a controller output may be declared with.
CTRL_OUT_KINDS = (CTRL_OUT_BOOL, CTRL_OUT_VALUE)

#: A reading compared to a threshold. Compiles to a watched two-state
#: automaton.
CTRL_OP_COMPARE = "compare"

#: Two thresholds and a direction: a hysteresis band. Compiles to one
#: watched two-state automaton whose two edges are the two thresholds.
CTRL_OP_BAND = "band"

#: Booleans reduced by ``and``, ``or``, ``not`` or k-of-n. The one operator
#: whose operands are already discrete, and therefore the one that adds no
#: automaton of its own.
CTRL_OP_COMBINE = "combine"

#: A reading published as a number, multiplied by a gain. Compiles to an
#: explicit equation rather than to an automaton: it carries a quantity, and
#: a quantity has no crossing to date.
CTRL_OP_REPUBLISH = "republish"

#: The operators that answer a **boolean**, and therefore the ones a
#: :data:`CTRL_OUT_BOOL` output may emit.
CTRL_BOOL_OPERATORS = (CTRL_OP_COMPARE, CTRL_OP_BAND, CTRL_OP_COMBINE)

#: The operators that answer a **number**, and therefore the ones a
#: :data:`CTRL_OUT_VALUE` output may emit.
CTRL_VALUE_OPERATORS = (CTRL_OP_REPUBLISH,)

#: The closed list an output value is composed from.
CTRL_OPERATORS = CTRL_BOOL_OPERATORS + CTRL_VALUE_OPERATORS

#: A band detecting a reading that has risen **above** its activation level.
#: Its release edge then sits at or below that level.
CTRL_BAND_ABOVE = "above"

#: A band detecting a reading that has fallen **below** its activation
#: level. Its release edge then sits at or above that level.
CTRL_BAND_BELOW = "below"

#: Every direction a band may be declared in.
CTRL_BAND_DIRECTIONS = (CTRL_BAND_ABOVE, CTRL_BAND_BELOW)

#: The comparison of each band edge, per direction: the activation one
#: first, the release one second.
#:
#: The release edge is deliberately **strict**, exactly as muscadet's is and
#: for the same reason: the degenerate band, the two levels coinciding, then
#: leaves the two edges mutually exclusive, so the output switches at that
#: single level instead of having both edges hold at once.
CTRL_BAND_EDGE_OPERATORS = {
    CTRL_BAND_ABOVE: ("ge", "lt"),
    CTRL_BAND_BELOW: ("le", "gt"),
}

#: The comparison operators a :class:`CtrlCompare` may carry.
#:
#: ``eq`` and ``ne`` are **absent**, and the omission is the same one the
#: authoring layer makes on a rule guard: a controller input reads a
#: continuously-evolving quantity, an equality on such a quantity brackets
#: no crossing, and a watched transition needs an ordering comparison to
#: root-find. A model that means "has reached" writes ``ge`` or ``le``.
COMPARISON_OPERATORS = ("<", "<=", ">", ">=")

#: Comparison spelling to the engine's, and the ordering comparison that is
#: its exact negation: an edge out of a state is the complement of the edge
#: into it, and emitting the complement directly keeps both edges an
#: ordering comparison, which is what ``watched`` requires.
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
_NEGATED = {"lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}

#: What a comparison's threshold attribute is named from:
#: ``{output}{path}_threshold``, ``path`` being the node's **position** in
#: the output's tree, the very one the automata are named from. Two
#: comparisons on one input against two levels are therefore two attributes,
#: and a model rebuilt names them the same way.
CTRL_PARAM_THRESHOLD = "threshold"

#: What a band's activation edge is named from.
CTRL_PARAM_ACTIVATE = "activate"

#: What a band's release edge is named from.
CTRL_PARAM_RELEASE = "release"

#: The attribute a failure mode clamps to blind a **boolean** output:
#: ``{output}_signal_available``, created at True. False makes the output
#: carry its declared default (for a control port, no order at all) while
#: everything upstream of it goes on being right.
CTRL_AVAILABLE_SUFFIX = "signal_available"

#: The attribute a failure mode raises to force a **value** output:
#: ``{output}_forced``, created at False.
CTRL_FORCED_SUFFIX = "forced"

#: What a forced value output publishes instead of its reading:
#: ``{output}_forced_value``, created at 0. Two attributes and not one
#: because a number has no rest value a single flag could stand for, which
#: is exactly what a boolean output's declared default is, and why blinding
#: one takes no second attribute.
CTRL_FORCED_VALUE_SUFFIX = "forced_value"

#: What everything a value output publishes is multiplied by:
#: ``{output}_level_gain``. A gain of 0 is a dead instrument, a gain of 5 a
#: wild one, and a mode reaches both by clamping this one attribute, which
#: is why a republication declares its gain here and nowhere else.
CTRL_GAIN_SUFFIX = "level_gain"

#: Conjunction: every operand holds.
CTRL_LOGIC_AND = "and"

#: Disjunction: at least one operand holds.
CTRL_LOGIC_OR = "or"

#: Negation: the single operand does not hold.
CTRL_LOGIC_NOT = "not"

#: k-of-n: at least ``k`` of the operands hold. The voting shape a redundant
#: instrument set exists for.
CTRL_LOGIC_K = "k"

#: The closed list a combination is chosen from.
CTRL_LOGICS = (CTRL_LOGIC_AND, CTRL_LOGIC_OR, CTRL_LOGIC_NOT, CTRL_LOGIC_K)

#: Declaration keys an observation input reads.
CONTROL_IN_KEYS = ("name", "kind", "flows", "aggregate", "default")

#: Declaration keys a controller output reads.
CONTROL_OUT_KEYS = ("name", "kind", "default", "emit")

#: Declaration keys the object itself reads.
OBJCTRL_KEYS = ("type", "name", "controls_in", "controls_out")


# ----------------------------------------------------------------------
# Small expression helpers (core schema)
# ----------------------------------------------------------------------


def _float(value: float) -> dict[str, Any]:
    return {"op": "const", "value": {"kind": "float", "value": float(value)}}


def _bool_const(value: bool) -> dict[str, Any]:
    return {"op": "const", "value": {"kind": "bool", "value": bool(value)}}


def _attr(component: str, attribute: str) -> dict[str, Any]:
    return {"op": "attr", "attr": {"component": component, "attribute": attribute}}


def _state(component: str, automaton: str, state: str) -> dict[str, Any]:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


def _cmp(op: str, lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    return {"op": "cmp", "cmp": op, "lhs": lhs, "rhs": rhs}


def _float_attribute(name: str, init: float) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "float",
        "init": {"kind": "float", "value": float(init)},
    }


def _bool_attribute(name: str, init: bool) -> dict[str, Any]:
    return {"name": name, "kind": "bool", "init": {"kind": "bool", "value": bool(init)}}


# ----------------------------------------------------------------------
# The output grammar
# ----------------------------------------------------------------------


@dataclass
class CtrlNode:
    """One node of an output value: an operator and its operands.

    Never built directly by a model: :func:`build_ctrl_node` is the door,
    and it is what turns a declaration into a tree, refusing by name what
    the closed grammar does not carry.
    """

    #: True when this operator answers a boolean, False when it answers a
    #: number. A class fact, not a field: it is a property of the operator.
    IS_BOOLEAN = True

    op: str = ""

    def operand_nodes(self) -> list[CtrlNode]:
        """The sub-nodes this one reduces. Empty on a leaf."""
        return []

    def inputs_read(self) -> list[str]:
        """Every observation input this subtree reads, in declaration order."""
        found: list[str] = []
        for operand in self.operand_nodes():
            for name in operand.inputs_read():
                if name not in found:
                    found.append(name)
        return found


@dataclass
class CtrlCompare(CtrlNode):
    """A reading against a threshold.

    ``{"op": "compare", "input": "tank", "operator": ">=", "threshold": 5.0}``
    """

    op: str = CTRL_OP_COMPARE
    #: Name of the observation input whose reading is compared.
    input: str = ""
    #: One of :data:`COMPARISON_OPERATORS`.
    operator: str = ">="
    #: The level the reading is compared to, and the initial value of
    #: ``{output}{path}_threshold``: what the comparison reads is that
    #: attribute, so an instance may be tuned away from its declaration, an
    #: indicator may name it and a failure mode may move it.
    threshold: float = 0.0

    def inputs_read(self) -> list[str]:
        return [self.input]


@dataclass
class CtrlBand(CtrlNode):
    """A hysteresis band: two thresholds and the direction they are read in.

    ``{"op": "band", "input": "tank", "direction": "below",
    "activate": 6.0, "release": 8.0}``

    What a band buys over a comparison is the whole of why it is an operator
    of its own: a comparison switches back the instant its condition stops
    holding, so a montage gated on one chatters around a single level. A
    band holds between its two edges, which is what lets the quantity it
    controls actually move. That is also why the automaton's **state is the
    band's value**: no reading of the quantity alone can say whether the
    band is holding, because the band's whole business is to answer
    differently at the same level depending on where it came from.
    """

    op: str = CTRL_OP_BAND
    #: Name of the observation input whose reading is banded.
    input: str = ""
    #: ``"above"`` to activate when the reading rises past the activation
    #: level, ``"below"`` to activate when it falls past it. It is what
    #: fixes on which side of that level the release edge must sit.
    direction: str = CTRL_BAND_ABOVE
    #: The level the band switches on at, and the initial value of
    #: ``{output}{path}_activate``.
    activate: float = 0.0
    #: The level it switches off at, and the initial value of
    #: ``{output}{path}_release``. ``None``, the default, coincides with the
    #: activation level: the degenerate band, no hysteresis, and the two
    #: edges still mutually exclusive because the release comparison is
    #: strict.
    release: float | None = None

    def edge_operators(self) -> tuple[str, str]:
        """The activation and release comparisons of this band."""
        return CTRL_BAND_EDGE_OPERATORS[self.direction]

    def inputs_read(self) -> list[str]:
        return [self.input]


@dataclass
class CtrlCombine(CtrlNode):
    """Booleans reduced by and / or / not / k-of-n.

    ``{"op": "combine", "logic": "k", "k": 2, "operands": [...]}``
    """

    op: str = CTRL_OP_COMBINE
    #: One of :data:`CTRL_LOGICS`.
    logic: str = CTRL_LOGIC_AND
    #: How many operands must hold, for ``logic="k"`` and for it alone.
    #: Refused elsewhere rather than ignored: a ``k`` declared beside an
    #: ``or`` is a modeller who meant k-of-n.
    k: int | None = None
    #: The sub-nodes combined, already built.
    operands: list[CtrlNode] = field(default_factory=list)

    def operand_nodes(self) -> list[CtrlNode]:
        return list(self.operands)


@dataclass
class CtrlRepublish(CtrlNode):
    """A reading published as a number, multiplied by a gain.

    ``{"op": "republish", "input": "reading", "gain": 1.0}``

    The one operator that answers a quantity, and therefore the one that
    compiles to an explicit equation instead of to an automaton: what it
    carries is refreshed at every evaluation point, exactly as every other
    published measurement is, and it has no crossing for anything to date.
    That is also why a value output needs no automaton to be forced, where a
    boolean one would: the equation runs at every sweep, so a forcing flag
    is read the instant it is raised and the instant it is released.

    A gain of 0 annuls the **reading** and nothing else: what the channel
    observes goes on moving underneath it, which is the difference between a
    dead instrument and an empty tank. Forcing the publication to a number
    of its own is the other effect, and it is a different one.
    """

    IS_BOOLEAN = False

    op: str = CTRL_OP_REPUBLISH
    #: Name of the observation input whose reading is published.
    input: str = ""
    #: The factor everything published is multiplied by. Lands in
    #: ``{name}_level_gain``, the attribute a failure mode clamps.
    gain: float = 1.0

    def inputs_read(self) -> list[str]:
        return [self.input]


def _require_keys(where: str, spec: dict, accepted: tuple[str, ...]) -> None:
    """Refuse a declaration key nothing reads, naming it.

    A misspelt key is otherwise swallowed whole, and a controller silently
    missing a threshold is indistinguishable from one that never declared
    any.
    """
    unknown = sorted(set(spec) - set(accepted))
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        raise ValueError(
            f"{where} does not accept declaration key{plural} "
            f"{', '.join(repr(k) for k in unknown)}; it accepts "
            f"{', '.join(sorted(accepted))}"
        )


def _number(where: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: `{key}` is a number, got {value!r}")
    return float(value)


def build_ctrl_node(where: str, spec: Any) -> CtrlNode | None:
    """Turn one ``emit`` declaration into a grammar node, or refuse it.

    The single door onto the grammar, and therefore the single place a
    Python callable is turned away. Recursive: a combination's operands are
    built first, so a node always holds nodes and never raw mappings.

    ``None`` is an output nothing computes, which keeps the declared default
    the attribute was created with.
    """
    if spec is None:
        return None
    if isinstance(spec, CtrlNode):
        return spec

    operators = ", ".join(CTRL_OPERATORS)
    if not isinstance(spec, dict):
        raise ValueError(
            f"{where}: an output value is a composition of the closed "
            f"operators {operators}, written as a mapping carrying an 'op' "
            f"key, and {spec!r} is not one. A Python callable is refused here "
            "whatever it attests: nothing can read a threshold out of "
            "arbitrary Python, so nothing could compile one to a watched "
            "transition, and the fallback would be a sensitive function on a "
            "reading, which never re-evaluates between discrete events"
        )

    spec = dict(spec)
    op = spec.pop("op", None)
    if op == CTRL_OP_COMPARE:
        return _build_compare(where, spec)
    if op == CTRL_OP_BAND:
        return _build_band(where, spec)
    if op == CTRL_OP_COMBINE:
        return _build_combine(where, spec)
    if op == CTRL_OP_REPUBLISH:
        return _build_republish(where, spec)
    raise ValueError(f"{where}: unknown operator {op!r}, expected one of {operators}")


def _build_compare(where: str, spec: dict) -> CtrlCompare:
    _require_keys(where, spec, ("input", "operator", "threshold"))
    for key in ("input", "operator", "threshold"):
        if key not in spec:
            raise ValueError(f"{where}: a comparison declares `{key}`")
    operator = spec["operator"]
    if operator not in COMPARISON_OPERATORS:
        raise ValueError(
            f"{where}: a comparison operator is one of "
            f"{', '.join(COMPARISON_OPERATORS)}, got {operator!r}. An equality "
            "is absent on purpose: a controller reads a continuously-evolving "
            "quantity, and an equality on one brackets no crossing for the "
            "engine to locate"
        )
    return CtrlCompare(
        input=str(spec["input"]),
        operator=operator,
        threshold=_number(where, "threshold", spec["threshold"]),
    )


def _build_band(where: str, spec: dict) -> CtrlBand:
    _require_keys(where, spec, ("input", "direction", "activate", "release"))
    for key in ("input", "activate"):
        if key not in spec:
            raise ValueError(f"{where}: a band declares `{key}`")
    direction = spec.get("direction", CTRL_BAND_ABOVE)
    if direction not in CTRL_BAND_DIRECTIONS:
        raise ValueError(
            f"{where}: a band detects in one of "
            f"{', '.join(CTRL_BAND_DIRECTIONS)}, got {direction!r}"
        )
    activate = _number(where, "activate", spec["activate"])
    release = spec.get("release")
    if release is None:
        release = activate
    else:
        release = _number(where, "release", release)
        inverted = (
            release > activate if direction == CTRL_BAND_ABOVE else release < activate
        )
        if inverted:
            # An inverted band is not a subtle mistake with a subtle
            # consequence: a band detecting below 3 and releasing at 1 can
            # never release, because the reading has to fall to 1 while the
            # band is what stops it falling. The montage then latches on its
            # first activation and never speaks again.
            side = "below" if direction == CTRL_BAND_ABOVE else "above"
            raise ValueError(
                f"{where}: a band detecting {direction!r} {activate} releases "
                f"at or {side} it, not at {release}"
            )
    return CtrlBand(
        input=str(spec["input"]),
        direction=direction,
        activate=activate,
        release=release,
    )


def _build_combine(where: str, spec: dict) -> CtrlCombine:
    _require_keys(where, spec, ("logic", "k", "operands"))
    logic = spec.get("logic")
    if logic not in CTRL_LOGICS:
        raise ValueError(
            f"{where}: a combination is one of {', '.join(CTRL_LOGICS)}, "
            f"got {logic!r}"
        )
    operands = [
        build_ctrl_node(f"{where} operand {index}", operand)
        for index, operand in enumerate(spec.get("operands") or [])
    ]
    # The vacuous cases are refused rather than answered. `any([])` is False
    # and `all([])` is True, so an empty combination is a silent constant,
    # and a constant output is a controller that does nothing.
    if not operands:
        raise ValueError(f"{where}: a combination with no operand is a constant")
    for operand in operands:
        if operand is None or not operand.IS_BOOLEAN:
            raise ValueError(
                f"{where}: a combination reduces conditions, and operand "
                f"{getattr(operand, 'op', operand)!r} carries a number"
            )
    if logic == CTRL_LOGIC_NOT and len(operands) != 1:
        raise ValueError(f"{where}: a 'not' negates ONE operand, got {len(operands)}")
    k = spec.get("k")
    if logic != CTRL_LOGIC_K:
        if k is not None:
            raise ValueError(
                f"{where}: 'k' counts the operands of a {CTRL_LOGIC_K!r} "
                f"combination and has no meaning beside a {logic!r} one"
            )
        return CtrlCombine(logic=logic, k=None, operands=operands)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(
            f"{where}: a {CTRL_LOGIC_K!r} combination counts at least one "
            f"operand, got k={k!r}"
        )
    if k > len(operands):
        raise ValueError(
            f"{where}: a {CTRL_LOGIC_K!r} combination asks for {k} of "
            f"{len(operands)} operands, which can never hold"
        )
    return CtrlCombine(logic=logic, k=k, operands=operands)


def _build_republish(where: str, spec: dict) -> CtrlRepublish:
    _require_keys(where, spec, ("input", "gain"))
    if "input" not in spec:
        raise ValueError(f"{where}: a republication declares `input`")
    gain = 1.0 if "gain" not in spec else _number(where, "gain", spec["gain"])
    return CtrlRepublish(input=str(spec["input"]), gain=gain)


# ----------------------------------------------------------------------
# Observation inputs
# ----------------------------------------------------------------------


@dataclass
class _ControlIn:
    """One observation input: what it reads, and how it reduces its
    sources."""

    name: str
    kind: str
    flow: str | None
    aggregate: str | None
    #: What the attribute holds before the first sweep. One number and one
    #: spelling, where muscadet has ``level_default`` / ``rate_default`` /
    #: ``ratio_default``: the nature is already declared by ``kind``.
    #:
    #: It governs that first instant only. muscadet reads an unconnected
    #: reference as this default for the whole run, while ``port_agg``
    #: answers 0 on a port nothing feeds, so an input left unwired reads
    #: zero here from the first evaluation point on.
    default: float

    @property
    def attribute(self) -> str:
        """The attribute this input's reading lands in.

        The publisher's own naming, so the two sides line up: a capacity
        publishes ``{name}_level`` and ``{name}_level_{flow}``, a continuous
        output ``{name}_rate``, a volume ``{name}_ratio_{flow}``.
        """
        if self.kind == MEASUREMENT_RATE:
            return f"{self.name}_{MEASUREMENT_RATE}"
        if self.kind == MEASUREMENT_RATIO:
            return f"{self.name}_{MEASUREMENT_RATIO}_{self.flow}"
        if self.flow is None:
            return f"{self.name}_{MEASUREMENT_LEVEL}"
        return f"{self.name}_{MEASUREMENT_LEVEL}_{self.flow}"

    @property
    def port(self) -> str:
        """The in port the reading arrives on.

        One port per attribute, unlike muscadet's message box that carries
        several: RAICHU connects attribute to attribute, so a ratio has a
        port of its own where PyCATSHOO shared the level's box.
        """
        return f"{self.attribute}_in"

    @property
    def agg(self) -> str:
        """The ``port_agg`` aggregation this input reduces with."""
        return CONTROL_AGGREGATION_OPS[self.aggregate or "sum"]


def _build_control_in(where: str, spec: Any) -> _ControlIn:
    """One observation input, from its declaration.

    ``add_control_in(name, aggregate=None, kind="level", flows=[...])`` in
    muscadet's spelling, written here as the mapping a plugin object
    carries.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: an observation input is a mapping, got {spec!r}")
    _require_keys(where, spec, CONTROL_IN_KEYS)
    name = spec.get("name")
    if not name:
        raise ValueError(f"{where}: an observation input declares `name`")
    where = f"{where} input `{name}`"
    kind = spec.get("kind", MEASUREMENT_LEVEL)
    if kind not in MEASUREMENT_KINDS:
        raise ValueError(
            f"{where}: a measurement reads one of {', '.join(MEASUREMENT_KINDS)}, "
            f"got {kind!r}"
        )
    flows = list(spec.get("flows") or [])
    if kind == MEASUREMENT_RATE and flows:
        raise ValueError(
            f"{where}: it reads a rate, which has no constituent, so "
            f"{flows} cannot be read on it. A constituent is a substance a "
            "volume holds"
        )
    if kind == MEASUREMENT_RATIO and len(flows) != 1:
        raise ValueError(
            f"{where}: a ratio is the share of ONE constituent, so it names "
            f"exactly one in `flows`, got {flows}. A whole has no fraction of "
            "itself"
        )
    if kind == MEASUREMENT_LEVEL and len(flows) > 1:
        raise ValueError(
            f"{where}: an observation input reads ONE number, so it names at "
            f"most one constituent in `flows`, got {flows}. Declare a second "
            "input to read a second constituent"
        )
    aggregate = spec.get("aggregate")
    if aggregate is not None:
        if aggregate not in CONTROL_AGGREGATIONS:
            raise ValueError(
                f"{where}: an aggregation is one of "
                f"{', '.join(CONTROL_AGGREGATIONS)}, got {aggregate!r}"
            )
        if aggregate not in CONTROL_AGGREGATION_OPS:
            raise ValueError(
                f"{where}: `{aggregate}` is a declarable aggregation this "
                "engine cannot compute. The only variable-arity reader of an "
                "in port is the `port_agg` operator, whose aggregations are "
                "sum, count, all, any, mean and median: it has no minimum and "
                "no maximum, and a measurement link declares no per-connection "
                "channel a fixed-arity `min` expression could read one by one. "
                "Refused rather than answered by a sum or a mean, which would "
                f"be a different question ({', '.join(sorted(CONTROL_AGGREGATION_OPS))} "
                "are computed)"
            )
    return _ControlIn(
        name=str(name),
        kind=kind,
        flow=flows[0] if flows else None,
        aggregate=aggregate,
        default=_number(where, "default", spec.get("default", 0.0)),
    )


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------


@dataclass
class _ControlOut:
    """One emitted control signal: its nature, its default and its
    grammar."""

    name: str
    kind: str
    default: Any
    node: CtrlNode | None

    @property
    def attribute(self) -> str:
        """The attribute the signal lands in, and the one the out port
        exports."""
        if self.kind == CTRL_OUT_BOOL:
            return self.name
        return f"{self.name}_{MEASUREMENT_LEVEL}"

    @property
    def port(self) -> str:
        return f"{self.attribute}_out"


def _build_control_out(where: str, spec: Any) -> _ControlOut:
    """One emitted control signal, from its declaration.

    ``add_control_out(name, kind, emit=<node>)`` in muscadet's spelling.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: a control output is a mapping, got {spec!r}")
    _require_keys(where, spec, CONTROL_OUT_KEYS)
    name = spec.get("name")
    if not name:
        raise ValueError(f"{where}: a control output declares `name`")
    where = f"{where} output `{name}`"
    kind = spec.get("kind", CTRL_OUT_BOOL)
    if kind not in CTRL_OUT_KINDS:
        raise ValueError(
            f"{where}: a control output carries one of "
            f"{', '.join(CTRL_OUT_KINDS)}, got {kind!r}"
        )
    node = build_ctrl_node(where, spec.get("emit"))
    # A boolean output emits a condition and a value output a quantity, and
    # the grammar answers exactly one of the two per operator: naming the
    # nature is what makes the mismatch a refusal instead of a cast.
    if node is not None:
        if kind == CTRL_OUT_BOOL and not node.IS_BOOLEAN:
            raise ValueError(
                f"{where}: a {CTRL_OUT_BOOL!r} output carries a condition, and "
                f"{node.op!r} answers a number. The operators that answer a "
                f"condition are {', '.join(CTRL_BOOL_OPERATORS)}"
            )
        if kind == CTRL_OUT_VALUE and node.IS_BOOLEAN:
            raise ValueError(
                f"{where}: a {CTRL_OUT_VALUE!r} output carries a number, and "
                f"{node.op!r} answers a condition. The operators that answer a "
                f"number are {', '.join(CTRL_VALUE_OPERATORS)}"
            )
    if kind == CTRL_OUT_BOOL:
        default = bool(spec.get("default", False))
    else:
        default = _number(where, "default", spec.get("default", 0.0))
    return _ControlOut(name=str(name), kind=kind, default=default, node=node)


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------


class _Compiler:
    """One controller, compiled to core material.

    Holds what the walk over an output's tree accumulates: the attributes
    every number of the grammar becomes, the automata every threshold
    compiles to, and the expression the emission reads.
    """

    def __init__(self, name: str, inputs: dict[str, _ControlIn]):
        self.name = name
        self.inputs = inputs
        self.attributes: list[dict[str, Any]] = []
        self.automata: list[dict[str, Any]] = []

    # --- naming -------------------------------------------------------

    def base(self, out_name: str, path: str, suffix: str) -> str:
        """The name every part of one node's automaton, and every number it
        declares, is derived from: ``{output}{path}_{suffix}``.

        A threshold and the automaton dating its crossing are therefore read
        off the same position in the tree, and a model rebuilt names them
        the same way.
        """
        return f"{out_name}{path}_{suffix}"

    def channel(self, where: str, node: CtrlNode) -> _ControlIn:
        channel = self.inputs.get(node.input)
        if channel is None:
            raise ValueError(
                f"{where}: operator {node.op!r} reads observation input "
                f"`{node.input}`, which this controller does not declare "
                f"(it declares {sorted(self.inputs) or 'none'})"
            )
        return channel

    # --- numbers of the grammar --------------------------------------

    def param(self, out_name: str, path: str, suffix: str, value: float) -> str:
        """Create the attribute of one number the grammar declared, and
        answer its name.

        Created **before** the automaton and read by it: the guard the engine
        locates and the emission that reads the state must consult the same
        number, or the two would answer differently the moment anything moved
        the threshold.
        """
        basename = self.base(out_name, path, suffix)
        self.attributes.append(_float_attribute(basename, value))
        return basename

    # --- automata -----------------------------------------------------

    def two_state(
        self,
        base: str,
        states: tuple[str, str],
        guards: tuple[dict[str, Any], dict[str, Any]],
    ) -> str:
        """The **watched** two-state automaton one grammar node compiles to,
        and the name of its far state.

        Two transitions, both watched, so the engine root-finds the date the
        condition turns and stops the integration there instead of picking
        the change up at the following discrete event. The automaton starts
        in its resting state whatever the reading is: a watched transition
        whose guard already holds fires at the current instant, so the
        automaton settles at t = 0 rather than sitting on the wrong side of
        its own threshold until the reading came back and crossed it again.
        """
        rest, far = (f"{base}_{suffix}" for suffix in states)
        self.automata.append(
            {
                "name": base,
                "states": [rest, far],
                "init": rest,
                "transitions": [
                    {
                        "name": f"{base}_up",
                        "source": rest,
                        "targets": [far],
                        "guard": guards[0],
                        "distrib": "watched",
                    },
                    {
                        "name": f"{base}_down",
                        "source": far,
                        "targets": [rest],
                        "guard": guards[1],
                        "distrib": "watched",
                    },
                ],
            }
        )
        return far

    # --- the walk -----------------------------------------------------

    def read(self, out_name: str, node: CtrlNode, path: str) -> dict[str, Any]:
        """Compile one node to the expression the emission reads, appending
        the attributes and automata it needs.

        Depth first, so an operand's automata exist before the combination
        that reads them. ``path`` is what makes an automaton's name a
        function of its **position** in the tree: two comparisons on the same
        input against different thresholds are two automata, two threshold
        attributes, and they have to be tellable apart.
        """
        where = f"ObjCtrl `{self.name}` output `{out_name}`"
        if isinstance(node, CtrlCompare):
            channel = self.channel(where, node)
            threshold = self.param(
                out_name, path, CTRL_PARAM_THRESHOLD, node.threshold
            )
            reading = _attr(self.name, channel.attribute)
            level = _attr(self.name, threshold)
            holds = _CMP[node.operator]
            # The edge out of a state is the exact complement of the edge
            # into it, and the complement of an ordering comparison is
            # another one: emitted as a negation it would carry no
            # comparison for the engine to root-find.
            base = self.base(out_name, path, CTRL_OP_COMPARE)
            far = self.two_state(
                base,
                ("below", "above"),
                (
                    _cmp(holds, reading, level),
                    _cmp(_NEGATED[holds], reading, level),
                ),
            )
            # The state and not the comparison: see the module docstring.
            # Read live, the comparison would make this emission trigger on a
            # quantity that announces nothing between discrete events.
            return _state(self.name, base, far)

        if isinstance(node, CtrlBand):
            channel = self.channel(where, node)
            activate = self.param(
                out_name, path, CTRL_PARAM_ACTIVATE, node.activate
            )
            release = self.param(
                out_name, path, CTRL_PARAM_RELEASE, float(node.release)
            )
            activate_op, release_op = node.edge_operators()
            reading = _attr(self.name, channel.attribute)
            # Two edges, two guards, ONE automaton, and the automaton's
            # state IS the band's value. That is what makes the hysteresis a
            # property of the compiled form rather than of a rule somebody
            # has to remember.
            base = self.base(out_name, path, CTRL_OP_BAND)
            far = self.two_state(
                base,
                ("released", "activated"),
                (
                    _cmp(activate_op, reading, _attr(self.name, activate)),
                    _cmp(release_op, reading, _attr(self.name, release)),
                ),
            )
            return _state(self.name, base, far)

        if isinstance(node, CtrlCombine):
            operands = [
                self.read(out_name, operand, f"{path}_operand_{index}")
                for index, operand in enumerate(node.operands)
            ]
            if node.logic == CTRL_LOGIC_AND:
                return {"op": "bool", "bool_op": "and", "args": operands}
            if node.logic == CTRL_LOGIC_OR:
                return {"op": "bool", "bool_op": "or", "args": operands}
            if node.logic == CTRL_LOGIC_NOT:
                return {"op": "bool", "bool_op": "not", "args": operands}
            # k-of-n: the voting shape a redundant instrument set exists for,
            # counted rather than enumerated. Enumerating the k-subsets would
            # be the same answer at a combinatorial size, and it would put the
            # count in the shape of the expression instead of in a number.
            votes = [
                {"op": "if", "cond": operand, "then": _float(1.0), "otherwise": _float(0.0)}
                for operand in operands
            ]
            return _cmp("ge", {"op": "add", "args": votes}, _float(float(node.k)))

        channel = self.channel(where, node)
        return _attr(self.name, channel.attribute)


def expand_objctrl(spec: dict[str, Any], model: dict[str, Any]) -> tuple:
    """Expand one ``ObjCtrl`` object into core material.

    Returns ``(components, connections, indicators)``, plus the model-level
    evaluation order when the model carries one (see
    :func:`_evaluation_order`). The controller declares its own component and
    no connection: the wiring lives in the model's ``connections`` list, as
    it does for every other object of this plugin.
    """
    _require_keys("ObjCtrl", spec, OBJCTRL_KEYS)
    name = spec.get("name")
    if not name:
        raise ValueError("ObjCtrl: an object declares `name`")
    where = f"ObjCtrl `{name}`"

    inputs: dict[str, _ControlIn] = {}
    for declaration in spec.get("controls_in") or []:
        channel = _build_control_in(where, declaration)
        if channel.name in inputs:
            raise ValueError(
                f"{where}: observation input `{channel.name}` is declared twice"
            )
        inputs[channel.name] = channel

    outputs: list[_ControlOut] = []
    seen: set[str] = set()
    for declaration in spec.get("controls_out") or []:
        output = _build_control_out(where, declaration)
        if output.name in seen:
            raise ValueError(f"{where}: control output `{output.name}` is declared twice")
        seen.add(output.name)
        outputs.append(output)

    if not inputs and not outputs:
        raise ValueError(
            f"{where}: a controller declares at least one observation input or "
            "one control output; one declaring neither is an empty component"
        )

    attributes: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    indicators: list[dict[str, Any]] = []
    compiler = _Compiler(name, inputs)

    # The reading side. One attribute, one in port carrying no channel (so
    # the link exchanges no quantity and the reader enters no allocation
    # operator), and the explicit equation that sweeps the port into the
    # attribute at every evaluation point.
    for channel in inputs.values():
        attributes.append(_float_attribute(channel.attribute, channel.default))
        ports.append({"name": channel.port, "dir": "in"})
        equations.append(
            {
                "target": channel.attribute,
                "kind": "explicit",
                "expr": {
                    "op": "port_agg",
                    "port": {"component": name, "port": channel.port},
                    "agg": channel.agg,
                },
            }
        )

    # The emitting side.
    for output in outputs:
        if output.kind == CTRL_OUT_BOOL:
            attributes.append(_bool_attribute(output.attribute, bool(output.default)))
            available = f"{output.name}_{CTRL_AVAILABLE_SUFFIX}"
            attributes.append(_bool_attribute(available, True))
        else:
            attributes.append(_float_attribute(output.attribute, float(output.default)))
        ports.append(
            {"name": output.port, "dir": "out", "attr": output.attribute}
        )

        if output.node is None:
            # An output nothing computes keeps the value it was created
            # with, which is what a test drives and what a bare skeleton has
            # always done.
            continue

        value = compiler.read(output.name, output.node, "")

        if output.kind == CTRL_OUT_BOOL:
            # Blinding, read directly rather than through an automaton of its
            # own: RAICHU derives this function's triggers from what it reads,
            # so the availability attribute announces both edges, the
            # returning one included. A blinded output carries its declared
            # default, which for a control port is no order at all, while
            # everything upstream of it goes on being right.
            functions.append(
                {
                    "name": f"emit_{output.name}",
                    "effects": [
                        {
                            "target": {"component": name, "attribute": output.attribute},
                            "value": {
                                "op": "if",
                                "cond": _attr(name, available),
                                "then": value,
                                "otherwise": _bool_const(bool(output.default)),
                            },
                        }
                    ],
                }
            )
        else:
            gain = f"{output.name}_{CTRL_GAIN_SUFFIX}"
            forced = f"{output.name}_{CTRL_FORCED_SUFFIX}"
            forced_value = f"{output.name}_{CTRL_FORCED_VALUE_SUFFIX}"
            attributes.append(
                _float_attribute(gain, float(getattr(output.node, "gain", 1.0)))
            )
            attributes.append(_bool_attribute(forced, False))
            attributes.append(_float_attribute(forced_value, 0.0))
            # A forced output publishes its forced value, gain and all. One
            # publication path and one gain: routing a forced value around the
            # gain would make a mode that kills the gain of a forced
            # instrument a silent no-op.
            equations.append(
                {
                    "target": output.attribute,
                    "kind": "explicit",
                    "expr": {
                        "op": "mul",
                        "args": [
                            _attr(name, gain),
                            {
                                "op": "if",
                                "cond": _attr(name, forced),
                                "then": _attr(name, forced_value),
                                "otherwise": value,
                            },
                        ],
                    },
                }
            )

    component = {
        "name": name,
        # The grammar's own numbers come last, so the readings and the
        # signals keep the head of the list: what an ObjFM names, and what
        # an indicator observes, is what a model declared.
        "attributes": attributes + compiler.attributes,
        "ports": ports,
        "interfaces": [],
        "automata": compiler.automata,
        "sensitive_functions": functions,
        "equations": equations,
    }

    for attribute in component["attributes"]:
        indicators.append(
            {
                "name": f"{name}_{attribute['name']}",
                "target": "attribute",
                "attr": {"component": name, "attribute": attribute["name"]},
            }
        )

    order = _evaluation_order(model, component)
    if order is None:
        return [component], [], indicators
    return [component], [], indicators, {"evaluation_order": order}


def _evaluation_order(
    model: dict[str, Any], component: dict[str, Any]
) -> list[dict[str, str]] | None:
    """Extend the model's evaluation order with this controller's steps, or
    answer ``None`` when the model declares none.

    A model without the field keeps the **positional** sweep: components in
    declaration order, and inside each, the explicit equations in declaration
    order. A controller's component is appended after everything already in
    the model, so its readings are already swept downstream of the levels
    they mirror, and no order has to be written to say so.

    A model that *does* carry the field is a different matter: the order must
    cover the declared steps exactly, one entry each, so a controller landing
    in it has to be added or the whole model is refused. The existing order
    is kept as it stands, everything else declared is closed over (so a step
    some other object gained cannot fall out of the sweep silently), and this
    controller's steps go last, which is downstream of every level it can
    read.

    The field is **taken over** rather than written beside: a model-wide
    property has one writer, so the key is removed from the model here and
    handed back through the plugin contract, which is what lets a second
    controller extend what the first one left.
    """
    existing = model.get("evaluation_order")
    if existing is None:
        return None

    order: list[dict[str, str]] = []
    listed: set[tuple[str, str]] = set()

    def step(component_name: str, attribute: str) -> None:
        if (component_name, attribute) not in listed:
            listed.add((component_name, attribute))
            order.append({"component": component_name, "attribute": attribute})

    for entry in existing:
        step(entry["component"], entry["attribute"])
    for declared in list(model.get("components") or []) + [component]:
        for equation in declared.get("equations") or []:
            if equation.get("kind") == "explicit":
                step(declared["name"], equation["target"])
        for allocation in declared.get("allocations") or []:
            step(declared["name"], allocation["name"])

    model.pop("evaluation_order", None)
    return order
