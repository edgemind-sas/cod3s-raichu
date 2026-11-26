"""Vocabulary of the authoring layer: the keys, classes and states a declaration names, and the functions that read them."""

from __future__ import annotations

import itertools
import math
import operator
from types import MappingProxyType, SimpleNamespace
from typing import Any


#: Channels materialised per continuous connection: what the consumer
#: asks for, what remains available to it, and what it was allocated.
_CONTINUOUS_CHANNELS = ("demand", "capability", "alloc")


#: Mode a rule set sits in while no guard holds and it declares no
#: default rule. It consumes nothing and produces nothing there.
_NO_RULE = "none"


#: What a published rate channel may carry: the quantity that crossed,
#: or the quantity that could have. They differ wherever a supply is not
#: fully drawn, and which one an observer wants depends on what it is
#: watching for.
#:
#: A regulator deciding whether to call on a reserve wants the
#: **capability**: reading the delivery instead closes a loop through its
#: own decision, the reserve reducing the draw on the supply and the
#: lower draw calling for more reserve. A meter recording what a
#: component consumed wants the **delivery**, which is the thing it
#: measures.
RATE_DELIVERED = "delivered"


RATE_CAPABILITY = "capability"


RATE_QUANTITIES = (RATE_DELIVERED, RATE_CAPABILITY)


#: The rate of an output nothing derates: the neutral element of the
#: minimum the deratings fold by. A profile is the neutral element of a
#: **product** instead, which is why the two never fold together; an
#: output declaring no profile carries no profile variable at all, so
#: that second neutral element is never written down and has no constant
#: of its own here.
NOMINAL_RATE = 1.0


#: The fraction of an output nothing takes off it: the neutral element
#: of the **sum** the taps fold by. It is zero where a cap's neutral is
#: one, which is the whole reason a cap and a tap never share a map: read
#: as a cap, a tap of 0.1 leaves a tenth of the output instead of taking
#: a tenth off it, and nothing in the number distinguishes the two.
NO_TAP = 0.0


#: The one transfer law this layer can carry as data, matching what
#: muscadet declares. Everything else is a Python callable, which no
#: mapping serializes and no engine can attest continuous, so it is
#: refused rather than approximated.
_TRANSFER_CLASS = "ConductiveTransfer"


#: The time profile shapes this layer can carry as data, matching
#: muscadet's: a clamped sinusoid, a periodic square wave whose edges are
#: linear ramps, and a drawn periodic curve through points. Any other
#: profile is a Python callable, refused for the same reason as a
#: transfer law.
SINE_PROFILE = "SinusoidalProfile"


SQUARE_PROFILE = "TrapezoidProfile"


DRAWN_PROFILE = "PiecewiseLinearProfile"


PROFILE_CLASSES = (SINE_PROFILE, SQUARE_PROFILE, DRAWN_PROFILE)


#: The noise modifier every shape may carry: the standard deviation at the
#: nodes, their spacing, and the flow key making two flows' noises
#: independent. muscadet writes them only on a noisy profile.
NOISE_KEYS = ("noise_amplitude", "noise_time_scale", "noise_key")


#: Every key a declared sinusoid may carry, as DATA rather than as a
#: tuple buried in the parser: the vocabulary is confronted with what
#: muscadet's read-back writes rather than patched key by key after an
#: export is refused (`test_muscadet_declaration_vocabulary`).
#:
#: The six parameters of the clamped sinusoid, `name`, and the noise
#: modifier. `name` is not a parameter of the curve: muscadet serializes a
#: declared object by its constructor's own parameter names, and
#: `SinusoidalProfile.__init__` takes a `name` defaulted to the class
#: name, so EVERY exported sinusoid carries one whether or not a modeller
#: wrote it. A vocabulary that did not know the key refused the whole
#: document, which is the same shape of gate `mixtures` was on a
#: component. The two other shapes take no `name`, so they carry none.
PROFILE_KEYS = (
    "amplitude",
    "period",
    "phase_shift",
    "offset",
    "value_min",
    "value_max",
    "name",
    *NOISE_KEYS,
)


#: The vocabulary of each declarable shape, the sinusoid's being
#: :data:`PROFILE_KEYS`.
PROFILE_CLASS_KEYS = MappingProxyType(
    {
        SINE_PROFILE: PROFILE_KEYS,
        SQUARE_PROFILE: (
            "low",
            "high",
            "period",
            "high_duration",
            "ramp_duration",
            "phase_shift",
            "value_min",
            "value_max",
            *NOISE_KEYS,
        ),
        DRAWN_PROFILE: (
            "points",
            "period",
            "phase_shift",
            "value_min",
            "value_max",
            *NOISE_KEYS,
        ),
    }
)


#: The keys a square wave or a drawn curve cannot do without: muscadet's
#: constructors take them without a default.
_PROFILE_REQUIRED_KEYS = MappingProxyType(
    {
        SINE_PROFILE: (),
        SQUARE_PROFILE: ("low", "high", "period", "high_duration", "ramp_duration"),
        DRAWN_PROFILE: ("points", "period"),
    }
)


_FNV_OFFSET_BASIS = 0xCBF29CE484222325


_FNV_PRIME = 0x100000001B3


_MASK64 = (1 << 64) - 1


def flow_noise_key(component: str, flow: str) -> int:
    """The key of the noise on `flow` of `component`: the FNV-1a 64-bit
    hash of the UTF-8 bytes of ``"<component>/<flow>"``.

    muscadet computes the same key for a noisy profile declared without
    one, so two components of one class draw independent noises on every
    engine, and the key never depends on the process (as Python's salted
    ``hash`` would)."""
    value = _FNV_OFFSET_BASIS
    for byte in f"{component}/{flow}".encode():
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK64
    return value


#: Why a Python function is refused where a declared shape is expected.
#: It names the mechanism a discontinuous law would need rather than only
#: the rule it breaks: a guard read from inside the sweeps is evaluated
#: at the integration points the solver chooses, so a jump it has not
#: been told about is crossed inside a step and integrated over an
#: interval that partly precedes it.
_CONTINUITY_MESSAGE = (
    "continuity cannot be inspected, so it is declared: only the shapes "
    "this layer carries as data are accepted, and a Python function is "
    "not one of them. A discontinuous law (a thermostat switching at a "
    "threshold, a schedule, a lookup table) needs a boundary transition "
    "at each of its breakpoints so the solver stops AT the jump instead "
    "of crossing it inside an integration step; this layer derives no "
    "such transition from a callable, so it refuses the declaration "
    "rather than integrating the jump wrong"
)


#: Comparison operators a guard operand may carry, as the schema's
#: ``cmp`` tags.
_CMP_OPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


#: The same operators as Python predicates, for the one thing the schema tag
#: cannot do: decide a comparison whose left side is a **two-valued** read
#: before the model is written. See :func:`_boolean_comparison`.
_CMP_PREDICATES = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


#: The subset of them a **continuous** boundary may be located on. An
#: equality is not one: a crossing has no side to be located from, and
#: the guard would hold only if a float landed exactly on the threshold.
_ORDERING_OPS = ("<", "<=", ">", ">=")


#: The complement of each comparison. A guard is negated by flipping its
#: operators rather than by wrapping it in a `not`: the negation of `>=`
#: is the STRICT `<`, and only a strict comparison is tightened away from
#: the boundary. Wrapped, the two guards of a mode both hold exactly at
#: the threshold, and the mode chatters there instead of crossing it.
_NEGATED_OPS = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


#: Fraction of the volume a capacity must move back from a bound before
#: it leaves the matching location (KTD6). It is a declared parameter and
#: not a constant buried in the generator, because it is what bounds the
#: segment count: every bound entry and exit ends an integration segment,
#: a restart discards the adapted step size, and a capacity chattering on
#: its bound would drive the segment count towards the step count.
#:
#: **Chosen by measurement**, on the bound-riding scenario of
#: `test_muscadet_capacities.py` (volume 100, ten units offered, five
#: drawn, bound reached at t=20 and ridden to t=1000, `pyraichu` 0.9.0,
#: default numerics, ~10 000 solver steps).
#:
#: An *exactly* matched inflow costs nothing at any width: the allocation
#: hands the capacity back precisely the quantity it asked for, the two
#: floats cancel bit for bit, and the run takes **3 segments whatever the
#: width, zero included**. That exactness is not something to rely on
#: though: the flow resolution promises its tolerance (`1e-9` relative),
#: not exactness. So the scenario is matched only to that promise, the
#: pass-through demand sitting `5e-9` under what the capacity delivers,
#: and the width is measured against the drift that leaves:
#:
#: ==========  ========  ========================================
#: width       segments  what happens over 1000 units of time
#: ==========  ========  ========================================
#: 0           21053     the bound is re-crossed at every step
#: 1e-12       21053     under the drift, so no better
#: 1e-10       959       chattering, cheaper
#: 1e-9        99        at the flow tolerance, still chattering
#: 1e-8        11        nearly settled
#: 1e-7        3         the knee: the bound is entered once
#: **1e-6**    **3**     shipped, a decade past the knee
#: 1e-4        3         no cheaper, a hundred times the error
#: ==========  ========  ========================================
#:
#: The width is bounded from below by what it has to dominate, the flow
#: tolerance rather than the (ten times finer) event-location tolerance:
#: a band narrower than what the resolution itself promises cannot tell
#: "at the bound" from "off it". It is bounded from above by the error it
#: induces, a location held past the physical crossing suppressing inflow
#: or outflow across the band: **at most `width x volume` on the held
#: quantity**, so 1e-4 units on a volume of 100 and 1e-3 on a volume of
#: 1000, and nothing at all on the flow, which is exact inside the
#: location. 1e-6 sits a decade above the knee and four decades below a
#: visible error. Declare `hysteresis=` per capacity when a volume needs
#: a different trade; a genuine physical imbalance, an inflow that cannot
#: match the outflow, makes the capacity oscillate for reasons no width
#: settles, and is a modelling answer rather than a numerical one.
DEFAULT_HYSTERESIS = 1e-6


#: What a volume publishes as its serving capability while it holds
#: something and declares no `serve_rate` of its own: the spelling of
#: "this volume puts no ceiling on the rate it is drawn at".
#:
#: muscadet writes that as ``math.inf`` (`Capacity.serve_ceiling`), and
#: it is the right physics: a volume limits the TOTAL that can be taken
#: from it, not the rate, and the empty bound is what stops the draw. A
#: model document is JSON, which has no literal for an infinity, so the
#: quantity crossing the document is a magnitude no physical model
#: reaches instead. It is meant as the ceiling of a ``min`` against a
#: demand, never a term of a derivative and never compared for equality,
#: so its exact value is immaterial as long as no demand approaches it
#: and no coefficient multiplies it into an overflow -- 1e30 leaves 278
#: decades of headroom under `f64::MAX` for both.
#:
#: "Never a term of a derivative" is not left to this layer's care: an
#: unbounded demand (a volume with an infinite `fill_rate`) meeting an
#: unbounded ceiling makes it the rate a stock drains at, whose physics is
#: an instantaneous transfer no rate expresses. Every model this layer
#: builds therefore DECLARES the magnitude (:func:`model_level_keys`),
#: and the engine refuses, by name, any integrated rate reaching it.
UNBOUNDED_SERVICE = 1e30


def model_level_keys(evaluation_order: list[dict[str, str]]) -> dict[str, Any]:
    """The model-wide keys a continuous network carries: the sweep order
    it derives, and the magnitude it reserves for "unbounded"
    (:data:`UNBOUNDED_SERVICE`), which no integrated rate may reach. One
    function for the three routes that build such a model, so none of
    them can carry the order without the reservation."""
    return {"evaluation_order": evaluation_order, "unbounded_rate": UNBOUNDED_SERVICE}


#: Stand-in read where a flow is held by no volume at all, so that the
#: question "does the volume holding it transit?" has an answer without a
#: `None` check at every site that asks. Its `transmits` is False on
#: purpose: a flow no volume holds has no volume to pass anything on.
_NO_CAPACITY = SimpleNamespace(transmits=False)


#: The two locations of a capacity's discharge gate: the volume releases
#: in the first and nothing at all in the second. Named rather than
#: written at the four sites that use them, so a rule guard that reads
#: the gate (:meth:`ObjFlow._declared_automata` publishes it) and the
#: capability that branches on it cannot come to disagree.
SERVE_GATE_STATES = ("serving", "withheld")


#: The two locations of a mixture machine's run gate (muscadet's R52): the
#: machine draws in the first and nothing at all in the second.
RUN_GATE_STATES = ("running", "stopped")


#: How a discharge command combines its two levels, and muscadet's own
#: default (`Capacity.serve_cond_inner_mode`). The same convention a
#: production condition carries: ``"or"`` is a conjunction of
#: disjunctions, ``"and"`` a disjunction of conjunctions.
SERVE_COND_INNER_MODES = ("or", "and")


SERVE_COND_INNER_MODE_DEFAULT = "or"


#: How a production condition combines its two levels, and muscadet's own
#: default (``FlowDiscreteOut.var_prod_cond_inner_mode``): ``"or"`` reads
#: the list as a conjunction of disjunctions, ``"and"`` as a disjunction of
#: conjunctions, and a flat list is one clause per element either way.
PROD_COND_INNER_MODES = ("or", "and")


PROD_COND_INNER_MODE_DEFAULT = "or"


#: The most conjunctions a condition may expand to when its clauses are
#: multiplied out into the disjunctive form the expression is built in:
#: one per choice of an operand from each clause. Exact, and small for a
#: real condition; past this, the condition belongs in a rule guard.
MAX_PROD_COND_GROUPS = 256


def _prod_cond_inner_mode(flow: str, inner_mode: str) -> str:
    """`inner_mode`, refused unless it is one of muscadet's two."""
    if inner_mode not in PROD_COND_INNER_MODES:
        raise ValueError(
            f"flow `{flow}` declares `var_prod_cond_inner_mode`={inner_mode!r}; "
            f"a production condition combines its two levels one way or the "
            f"other, so it is "
            f"{' or '.join(repr(mode) for mode in PROD_COND_INNER_MODES)}"
        )
    return inner_mode


def prod_cond_unconditional(where: str, declared: Any, inner_mode: str) -> bool:
    """Whether a production condition made of EMPTY groups is unconditional.

    ``[[]]`` is what the platform's port editor seeded on every fresh output
    for a long time, and libraries built before still carry it, on sources and
    on outputs whose author never wrote the condition. muscadet's discrete
    family evaluates it as it evaluates any other (``muscadet/flow.py``,
    ``prod_cond_holds``), so its meaning follows `inner_mode`:

    - under ``"and"`` the groups are a disjunction of conjunctions, and an
      empty conjunction HOLDS: the output produces from the start, whatever
      its inputs and whatever its declared default (measured on the reference
      engine 2026-10-15). True is returned, and the caller builds the output
      that declares no condition and starts produced. No writer is the
      reference's behaviour too: a sensitive method subscribed to no operand
      runs at the initial evaluation and never again, so a mode that latches
      the production later keeps its latch;
    - under ``"or"`` an empty clause is FALSE and the output would never
      produce. That is a dead port no author writes on purpose, and the
      platform refuses the mode on an empty condition for this reason, so it
      is refused here, naming the way out.

    An empty group BESIDE a group naming an operand is refused on both
    readings: under ``"and"`` it would make the whole condition hold and the
    operand the author did name would say nothing, under ``"or"`` it would
    make it never hold. That is an editing slip, and a run would hide it.

    False is returned for a condition carrying no empty group, which is then
    read as it always was. A flat list (bare operands) carries no group at all.
    """
    if not declared or isinstance(declared, (str, dict)):
        return False
    groups = [group for group in declared if isinstance(group, (list, tuple))]
    empty = [group for group in groups if not group]
    if not empty:
        return False
    if len(empty) != len(declared):
        raise ValueError(
            f"{where} carries an empty group beside groups that name operands; "
            f"under `var_prod_cond_inner_mode`='and' it would make the whole "
            f"condition hold, under 'or' it would make it never hold, and "
            f"either way the operands it sits beside would say nothing. Remove "
            f"the empty group"
        )
    if inner_mode != "and":
        raise ValueError(
            f"{where} carries only empty groups under "
            f"`var_prod_cond_inner_mode`={inner_mode!r}; there an empty clause "
            f"is false and the output would never produce. Declare no condition "
            f"(`[]`) for an unconditional output, or name what it waits for"
        )
    return True


def prod_cond_disjunction(
    where: str, clauses: list[list[Any]], inner_mode: str
) -> list[list[Any]]:
    """A production condition's clauses, in disjunctive form.

    ``"and"`` clauses are ALREADY a disjunction of conjunctions and come back
    unchanged. ``"or"`` clauses are a conjunction of disjunctions, multiplied
    out: one conjunction per choice of a single operand from each clause.
    """
    if inner_mode not in PROD_COND_INNER_MODES:
        raise ValueError(
            f"{where} declares `var_prod_cond_inner_mode`={inner_mode!r}; a "
            f"production condition combines its two levels one way or the "
            f"other, so it is "
            f"{' or '.join(repr(mode) for mode in PROD_COND_INNER_MODES)}"
        )
    if inner_mode == "and":
        return [list(clause) for clause in clauses]
    width = 1
    for clause in clauses:
        width *= max(len(clause), 1)
    if width > MAX_PROD_COND_GROUPS:
        raise ValueError(
            f"{where} carries a production condition of {len(clauses)} "
            f"conjoined clauses expanding to {width} disjunctions, past the "
            f"{MAX_PROD_COND_GROUPS} this layer converts. Split the component "
            f"or state the condition through a rule guard"
        )
    return [list(choice) for choice in itertools.product(*clauses)]


#: The laws a temporised output's transition may wait on, and the
#: parameters each carries, in the engine's transition form.
_TEMPO_LAW_PARAMETERS = {"delay": ("time",), "exp": ("rate",), "uniform": ("low", "high")}


def _tempo_law(flow: str, direction: str, time: float, law: dict[str, Any] | None) -> dict[str, Any]:
    """The law one tempo transition waits on, in the engine's form.

    `time` is the short form of a fixed delay and `law` the general one;
    both at once for one direction contradict each other unless the time
    is the default. A negative parameter is refused: a delay that ends
    before it starts, or a rate that is no rate, is no law. A uniform wait
    is drawn in `[low, high]` and refuses an empty interval, as the core
    does on any transition.
    """
    if law is None:
        law = {"distrib": "delay", "time": time}
    elif time:
        raise ValueError(
            f"flow {flow!r}: `{direction}_law` and `{direction}_time` both declare the "
            f"{direction} transition's wait; declare one"
        )
    distrib = law.get("distrib")
    parameters = _TEMPO_LAW_PARAMETERS.get(distrib)
    if parameters is None:
        raise ValueError(
            f"flow {flow!r}: the {direction} law {law!r} is neither "
            f"{' nor '.join(sorted(_TEMPO_LAW_PARAMETERS))}"
        )
    if set(law) != {"distrib", *parameters}:
        carried = "`distrib`, " + ", ".join(f"`{name}`" for name in parameters)
        raise ValueError(f"flow {flow!r}: a {distrib} {direction} law carries {carried}, got {law!r}")
    values = {}
    for name in parameters:
        value = law[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(
                f"flow {flow!r}: the {direction} law's `{name}` must be a finite non-negative number, got {value!r}"
            )
        values[name] = float(value)
    if distrib == "uniform" and values["high"] <= values["low"]:
        raise ValueError(
            f"flow {flow!r}: the {direction} law's interval [{values['low']}, {values['high']}] is empty; "
            f"`high` must be strictly above `low`"
        )
    return {"distrib": distrib, **values}


#: What one entry of a rule set may carry, and what
#: :meth:`ObjFlow._parse_rule` refuses anything outside of. Named rather than
#: written inline at the check, so a reader of a muscadet declaration can
#: CONFRONT this surface with what a muscadet read-back writes instead of
#: restating it: see the cross-validation suite's
#: `test_muscadet_declaration_vocabulary.py`, which needs muscadet.
RULE_KEYS = frozenset({"name", "cond", "cons", "prod"})


#: The same, for one operand of a rule's guard
#: (:meth:`ObjFlow._parse_operand`).
RULE_OPERAND_KEYS = frozenset(
    {"name", "port", "automaton", "state", "negate", "op", "value", "release"}
)


#: What one operand of a **production condition** may carry
#: (:meth:`ObjFlow._parse_prod_cond_operand`). muscadet's own vocabulary there
#: (``muscadet/obj.py``, ``apply_prod_cond``), which is the guard's minus the
#: two keys that need a location to sit in: `automaton`/`state` gate on one,
#: and `release` widens a comparison into a band that has to be held between
#: its edges. The production variable is rewritten at every evaluation and
#: holds nothing, so both are refused rather than read past.
PROD_COND_OPERAND_KEYS = frozenset({"name", "port", "negate", "op", "value"})
