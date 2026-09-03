"""muscadet-style authoring layer over RAICHU (M3: `raichu-muscadet`).

Recovers muscadet's productivity idioms: smart flow components,
declarative failure modes, one-line connections, as a thin builder
that *generates a native RAICHU model* (ports/interfaces, sensitive
functions, expression trees).

The authoring surface mirrors `muscadet.ObjFlow`:

>>> import pyraichu.muscadet as mu
>>> class Block(mu.ObjFlow):
...     def add_flows(self):
...         self.add_flow_in(name="is_ok")
...         self.add_flow_out(name="is_ok", var_prod_cond=["is_ok"])
>>> system = mu.System("rbd")
>>> system.add_component(Block, "B1")
>>> ...
>>> system.connect("S", "is_ok", "B1", "is_ok")
>>> result = system.simulate(t_max=24.0)

Semantics generated per flow (mirroring `muscadet/flow.py`):

- flow in  ``f``: bool ``f_fed_in`` := aggregation (``or``/``and``/
  ``k >= n``) of the connected producers' ``f_fed_out``;
- flow out ``f``: bool ``f_fed_out`` := production condition (AND of
  the declared ``var_prod_cond`` in-flows, or the ``var_prod_default``
  constant) AND ``f_fed_available_out`` (driven by failure modes);
- failure modes: two-state automaton (``ok``/``nok``) with delay or
  exponential laws; ``nok`` forces ``f_fed_available_out`` to ``False``.

Continuous flows (mirroring `muscadet/flow_continuous.py`) carry a real
quantity instead of a boolean, over three channels resolved in three
bands: capability along the flow, demand back against it, production
along it again. Per continuous flow ``f``:

- flow in  ``f``: ``f_capability_in`` (what the producers say they
  could deliver), ``f_demand_in`` (what this consumer asks for) and
  ``f_fed_in`` (what it actually receives, the sum of the shares
  allocated to it). Unconnected, the two received channels read the
  declared ``var_in_default``;
- flow out ``f``: ``f_capability_out`` (``f_out_rate`` x the declared
  ``var_fed_default``), ``f_demand_out`` (the total asked of it) and
  ``f_fed_out`` (the total it delivered), plus one **allocation
  operator** splitting a shortage among its consumers under the
  declared policy.

The per-connection quantities live in the out port's ``demand``,
``capability`` and ``alloc`` channels, so a producer hands each
consumer its own share and publishes to each what remains once the
others are accounted for.

Capacities (mirroring `muscadet/capacity.py`) buffer one or more of
those flows in a shared volume. Per capacity ``c`` holding flow ``f``:

- ``c_content_f`` integrates what enters minus what leaves, and
  ``c_content`` totals the held flows;
- ``c_fill_f`` is that content's *weighted* share of the volume and
  ``c_fill`` their sum, so a volume holding two flows of different
  weights reports one fill consistent with both;
- an automaton ``c_bounds`` sits in ``empty``, ``partial`` or ``full``,
  its locations entered by **watched** transitions on the total fill.
  The bounds are locations rather than branches inside the derivative:
  a branch would make the right-hand side discontinuous without the
  solver being told, which is exactly what a watched transition exists
  to avoid;
- at the full bound the capacity asks upstream only for what it can
  still take, and at the empty bound it serves downstream only what
  currently transits through it. Neither bound is written into the
  flow's own declaration: the generator wraps whatever demand and
  whatever capability that flow ended up with, so the buffer is declared
  independently of the flows it holds and composes with a rule set on
  the same flow whichever of the two was declared first;
- the level and the fill are published on **read-only** out ports, total
  and per constituent. A reader declares ``add_measurement_in`` and is
  connected with ``System.connect_measurement``: the link carries no
  quantity, declares no channel and enters no allocation.

Transformation rules (mirroring `muscadet/rules.py`) turn some of those
flows into others. A component declares an ordered set of rules, each
with a guard, a map of consumed input coefficients and a map of produced
output coefficients; the set runs at the scale its scarcest input and its
least demanded output allow. Per rule set ``S``:

- ``S_capability_scale`` is the minimum of every consumed input's
  capability over its coefficient, and ``S_scale`` bounds that by every
  demanded output's demand over its own. Both are **continuous minima**,
  which is the whole encoding (KTD14): the trajectory keeps a kink where
  the limiting term changes identity, so no crossing transition per
  input pair is needed, and none is declared;
- the demand of a consumed input and the capability of a produced output
  are the set's own contributions, collected per flow while the model is
  generated and folded in there rather than written into either flow's
  declaration. The capacity bounds then wrap the result, which is what
  lets a rule and a buffer on one flow compose whichever is declared
  first;
- ``{flow}_produced_out`` is what the set actually made, as opposed to
  what it could make: it is what the allocation distributes, and it is
  what holds correlated outputs in their declared proportion;
- a guarded set carries a mode automaton ``S_mode``, one location per
  rule plus one meaning "no rule applies", whose transitions are
  **watched** when a guard reads a continuously-evolving quantity.

Transfer pairs (mirroring `muscadet/transfer.py`) move a quantity that
travels because a **gradient** drives it, which demand-pulled transport
is the wrong vehicle for. A pair names two continuous outputs and a
declared law returning a **signed** quantity whose sign this layer
routes, so a model never writes a direction clamp. Per pair ``p``:

- ``p_requested`` is what the law asks for and ``p_moved`` what the
  component was able to move, published side by side so a saturated
  transfer reads as a shortfall rather than as a plausible number;
- naming two flows is the **two-stream exchange**: ``p_moved`` is
  subtracted from one output's balance and added to the other's, on top
  of ``{flow}_transfer_base``, the base each would carry without the
  pair. The raw total is untouched, which is the whole of what a pair
  guarantees, and the magnitude is capped by what the origin carries;
- naming the same flow twice is the **metered conduit**: what crosses
  the component IS the computed quantity, in place of the rate the
  output was declared with. It asks upstream for what it is about to
  move, and a computed reversal crosses nothing.

Time profiles (mirroring `muscadet/profile.py`) and deratings (mirroring
`muscadet/derating.py`) are the two other things multiplying a continuous
output's production, and they compose differently on purpose::

    produced = what the rule (or the declared rate) makes
               x  {flow}_out_profile
               x  {flow}_effective_rate

- ``{flow}_out_profile`` is a declared **continuous** function of the
  clock, published read-only. A discontinuous one would need a watched
  transition at each of its breakpoints, which this layer derives from
  no callable, so a profile that is not one of the declarable shapes is
  refused rather than integrated across an unannounced jump;
- ``{flow}_effective_rate`` is the **minimum** over the shared
  ``{flow}_out_rate`` (the endpoint a mode declared outside this layer
  clamps) and one term per failure mode derating the output. The minimum
  is order-independent and safe on repair, and each term reads its
  mode's automaton **location**, which is what makes the return to
  nominal implicit: leaving the failing state restores the rate with
  nothing declared on the other side. There is no separate boolean gate
  on a continuous flow, so a rate of zero is what stops it entirely;
- the two are separate channels, never folded: a panel at 30 % of its
  profile that is also derated to 0.5 produces 15 %, where one shared
  variable folding by minimum would read 30 % and signal nothing.

Only what muscadet declares as **data** translates: the conductive law
over its two potential operands (a constant and a measurement reading)
and the clamped sinusoid. A Python function in either place is refused
by name (R8), which is deliberate rather than provisional: two of the
documented reference models compute a ratio of one volume's
constituents, which is neither operand form, and they are authored
natively instead of widening this vocabulary.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Type

from . import (
    Model,
    SimulationResult,
    McEstimates,
    load_model,
    monte_carlo,
    seal,
    simulate,
)

__all__ = ["DEFAULT_HYSTERESIS", "ObjFlow", "System"]

#: Channels materialised per continuous connection: what the consumer
#: asks for, what remains available to it, and what it was allocated.
_CONTINUOUS_CHANNELS = ("demand", "capability", "alloc")

#: Mode a rule set sits in while no guard holds and it declares no
#: default rule. It consumes nothing and produces nothing there.
_NO_RULE = "none"

#: The rate of an output nothing derates: the neutral element of the
#: minimum the deratings fold by. A profile is the neutral element of a
#: **product** instead, which is why the two never fold together; an
#: output declaring no profile carries no profile variable at all, so
#: that second neutral element is never written down and has no constant
#: of its own here.
NOMINAL_RATE = 1.0

#: The one transfer law and the one time profile this layer can carry as
#: data, matching what muscadet declares. Everything else in either place
#: is a Python callable, which no mapping serializes and no engine can
#: attest continuous, so it is refused rather than approximated (R8).
_TRANSFER_CLASS = "ConductiveTransfer"
_PROFILE_CLASS = "SinusoidalProfile"

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
    "threshold, a schedule, a lookup table) needs a watched transition "
    "at each of its breakpoints so the solver stops AT the jump instead "
    "of crossing it inside an integration step; this layer derives no "
    "such transition from a callable, so it refuses the declaration "
    "rather than integrating the jump wrong"
)

#: Comparison operators a guard operand may carry, as the schema's
#: ``cmp`` tags.
_CMP_OPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}

#: How many arcs the rule-cycle search may expand before it gives up.
#: The diagnostic is worth a large but finite walk over the rule graph,
#: never an unbounded one on a pathological topology.
_CYCLE_SEARCH_BUDGET = 200_000


class _CycleSearchExhausted(Exception):
    """The rule-cycle search spent its budget before it enumerated every
    cycle, so it saw only part of the graph.

    Raised out of the walk rather than returned, because a partial
    enumeration is not a shorter answer to the same question: a cycle
    that creates matter may sit in the part never reached, and a guard
    that quietly stops guarding is worse than no guard. The caller turns
    it into a refusal naming what the search covered."""

    def __init__(self, nodes: int, budget: int):
        super().__init__(f"{budget} arcs expanded over {nodes} nodes")
        self.nodes = nodes
        self.budget = budget

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


def _var(component: str, variable: str) -> dict[str, Any]:
    return {"op": "attr", "attr": {"component": component, "attribute": variable}}


def _state_active(component: str, automaton: str, state: str) -> dict[str, Any]:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


def _float(value: float) -> dict[str, Any]:
    return {"op": "const", "value": {"kind": "float", "value": float(value)}}


def _float_attribute(name: str, init: float) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "float",
        "init": {"kind": "float", "value": float(init)},
    }


def _sum(terms: list[dict[str, Any]]) -> dict[str, Any]:
    """The sum of `terms`, degrading to the term itself and to zero
    rather than emitting a one-argument or empty ``add``."""
    if not terms:
        return _float(0.0)
    if len(terms) == 1:
        return terms[0]
    return {"op": "add", "args": terms}


def _min(terms: list[dict[str, Any]]) -> dict[str, Any]:
    """The minimum of `terms`, degrading to the term itself rather than
    emitting a one-argument ``min``.

    This is the operator a rule's scale is written with (KTD14): it is
    continuous and piecewise linear, so the trajectory keeps a **kink**
    where the limiting term changes identity, and needs no crossing
    transition to be located. An empty list has no minimum and is a
    caller error rather than a silent zero."""
    if not terms:
        raise ValueError("a minimum needs at least one term")
    if len(terms) == 1:
        return terms[0]
    return {"op": "min", "args": terms}


def _bool(bool_op: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    """`terms` joined by `bool_op`, degrading to the term itself rather
    than emitting a one-argument ``bool``.

    The sibling of :func:`_sum` and :func:`_min`, and there for the same
    reason: a conjunction or a disjunction is written wherever one is
    needed, over a list whose length the caller rarely knows, and the
    one-term case is the common one."""
    if len(terms) == 1:
        return terms[0]
    return {"op": "bool", "bool_op": bool_op, "args": terms}


def _matches_flow(pattern: str, name: str) -> bool:
    """Whether an effect pattern names `name`, **matched whole**.

    A full match rather than a bare search, which is the whole of the
    rule: unanchored, ``"H2"`` would name ``H2O`` as well, and a
    declaration meant for one output would silently derate its
    neighbour. The pattern is **grouped** before it is matched, because
    alternation binds looser than anchoring: written ``^H2|O2$``, the
    two-flow pattern ``"H2|O2"`` reads as "starts with H2, or ends with
    O2" and names ``H2O``, ``XO2`` and ``H2X`` after all. Grouped, it
    names the two flows it was written for and nothing else."""
    return re.fullmatch(f"(?:{pattern})", name) is not None


def _negated(expr: dict[str, Any]) -> dict[str, Any]:
    """``-expr``, written as a subtraction from zero: the schema has no
    unary minus, and a `mul` by -1 reads worse at the same cost."""
    return {"op": "sub", "lhs": _float(0.0), "rhs": expr}


def _clamped_at_zero(expr: dict[str, Any]) -> dict[str, Any]:
    """``max(0, expr)``: a magnitude, where a negative value means the
    other direction and is handled by the sign routing rather than by
    letting it through as a quantity."""
    return {"op": "max", "args": [_float(0.0), expr]}


def _channel_attr(port: str, channel: str, edge: str) -> str:
    """The attribute the compiler materialises for one (connection,
    channel) pair, per the schema's naming rule."""
    return f"{port}__{channel}__{edge}"


def _edge_name(connection: dict[str, Any]) -> str:
    """The edge a connection is named after: its own ``name`` when it
    has one, its destination otherwise (the schema's default)."""
    if connection.get("name"):
        return str(connection["name"])
    destination = connection["to"]
    return f"{destination['component']}__{destination['port']}"


def _find_cycle(nodes, arcs):
    """The first cycle a depth-first walk closes over `arcs`, as a list
    of ``(node, label)`` steps, or ``None``.

    Declaration order in, declaration order out: the reported cycle is
    the first one the walk closes, so one model always yields one
    diagnostic."""
    unseen, active, done = 0, 1, 2
    mark = {node: unseen for node in nodes}
    path: list = []

    def walk(node):
        mark[node] = active
        for target, label in arcs.get(node, ()):
            path.append((node, label))
            state = mark.get(target, unseen)
            if state == active:
                start = next(
                    index for index, (seen, _) in enumerate(path) if seen == target
                )
                return path[start:]
            if state == unseen:
                found = walk(target)
                if found is not None:
                    return found
            path.pop()
        mark[node] = done
        return None

    for node in nodes:
        if mark[node] == unseen:
            found = walk(node)
            if found is not None:
                return found
    return None


@dataclass
class _FlowIn:
    name: str
    logic: str = "or"  # "or" | "and" | int k (k-out-of-n)
    k: int | None = None


@dataclass
class _FlowOut:
    name: str
    var_prod_default: bool = False
    # Flat list = one AND group; list-of-lists = DNF (outer-OR of
    # inner-AND groups, the platform-export `prod_cond` form).
    var_prod_cond: list[str] | list[list[str]] = field(default_factory=list)
    # muscadet `FlowOutTempo`: {"enable_time", "disable_time",
    # "init_enable"}: a disabled↔enabled automaton whose delayed
    # transitions are gated on the production condition (reset on
    # interruption); the flow is fed while `enabled`.
    tempo: dict[str, Any] | None = None
    # muscadet `FlowOutOnTrigger`: {"time_up", "time_down", "logic"},
    # a down↔up automaton on a dedicated trigger in-port with
    # *inhibition* logic (up while the trigger inputs are absent); the
    # flow is fed while `up` AND the production condition holds.
    trigger: dict[str, Any] | None = None


@dataclass
class _FlowContinuousIn:
    """A real-valued input: what it asks for, and what it reads when
    nothing is connected to it."""

    name: str
    var_demand_in_default: float = 0.0
    var_in_default: float = 0.0
    #: Replaces the constant demand with a declared expression, for a
    #: caller that has one to hand. It is an escape hatch onto the
    #: generated equation and nothing in this module assigns it: a rule
    #: set consuming the flow derives the demand instead, and a capacity
    #: holding the flow bounds whatever came out, both of them while the
    #: model is generated. It is read last of the three, so a conduit or
    #: a rule set on the same flow takes precedence over it.
    demand_expr: dict[str, Any] | None = None


@dataclass
class _Profile:
    """A clamped sinusoid of simulation time scaling an output.

    ``amplitude x sin(2 pi (t - phase_shift) / period) + offset``,
    clamped into ``[value_min, value_max]``. Continuous by construction,
    the clamp meeting the curve where it crosses the bound, which is why
    this shape carries no attestation of its own: the attestation is what
    a bare callable cannot give (see :data:`_CONTINUITY_MESSAGE`).

    ``value_max`` is ``None`` when the curve is unbounded above: a
    document cannot carry an infinity, and an upper clamp that never
    binds is better left out of the expression than written as a number
    standing in for one."""

    amplitude: float = 1.0
    period: float = 2 * math.pi
    phase_shift: float = 0.0
    offset: float = 0.0
    value_min: float = 0.0
    value_max: float | None = None

    def factor(self, time: float) -> float:
        """The factor at `time`, for the initial value of the published
        variable: the sweep overwrites it, and an init that already
        agrees with the curve keeps a document readable on its own."""
        angle = (time - self.phase_shift) * (2 * math.pi / self.period)
        value = max(self.value_min, self.amplitude * math.sin(angle) + self.offset)
        return value if self.value_max is None else min(value, self.value_max)


@dataclass
class _Transfer:
    """One declared transfer pair: two flow names and the conductive law
    moving a signed quantity between them.

    The equation says how much, the pair says between what. Naming the
    same flow twice is the **metered conduit**, where what crosses the
    component IS the computed quantity; naming two flows is the
    **two-stream exchange**, a signed delta on top of what both streams
    already carry. The two take different paths through the bands and
    cannot share one rule."""

    name: str
    flows: tuple[str, str]
    conductance: float
    #: Normalised potentials: ``("const", value, None)`` or
    #: ``("measurement", channel, flow or None)``, the two operand forms
    #: muscadet admits and the only two this layer resolves.
    potential_a: tuple[str, Any, Any]
    potential_b: tuple[str, Any, Any]

    @property
    def is_conduit(self) -> bool:
        """True when the pair meters one flow's transit rather than
        moving a quantity between two."""
        return self.flows[0] == self.flows[1]

    @property
    def source(self) -> str:
        """The flow a positive quantity leaves."""
        return self.flows[0]

    @property
    def destination(self) -> str:
        """The flow a positive quantity enters."""
        return self.flows[1]


@dataclass
class _FlowContinuousOut:
    """A real-valued output: what it could deliver, and how a shortage
    is split among its consumers."""

    name: str
    var_fed_default: float = 0.0
    allocation: str = "proportional"  # "proportional" | "shares" | "priority"
    #: Keyed by consumer component name, or by a ``(component, flow)``
    #: pair when one consumer reads this output over two flows.
    allocation_shares: dict[Any, float] | None = None
    allocation_priorities: dict[Any, float] | None = None
    #: Replaces ``out_rate x var_fed_default`` with a declared
    #: expression: the counterpart of `_FlowContinuousIn.demand_expr`,
    #: and the same escape hatch, assigned by nothing in this module. A
    #: rule set producing the flow, a conduit metering it, its profile
    #: and its deratings all reach the capability by other routes. It
    #: stands on its own where it is read: the branch taking it applies
    #: neither the output factors nor the declared default.
    capability_expr: dict[str, Any] | None = None
    #: The declared time profile multiplying what this output produces,
    #: or ``None``. It composes with the derating rate by **product**,
    #: never by minimum: an output at 0.3 of its profile that is also
    #: derated to 0.5 produces 0.15, not 0.3.
    profile: _Profile | None = None


@dataclass
class _RuleOperand:
    """One operand of a rule guard.

    Either a read of a flow or of a declared attribute (`name`, with
    `port` disambiguating a name carried on both sides), or an
    automaton-state gate (`automaton` + `state`). `op` and `value` turn
    the read into a numeric comparison; `negate` inverts a boolean one.
    """

    name: str | None = None
    port: str | None = None
    automaton: str | None = None
    state: str | None = None
    negate: bool = False
    op: str | None = None
    value: float | None = None

    def label(self) -> str:
        """The operand as it reads in a diagnostic."""
        if self.automaton is not None:
            return f"{self.automaton} in {self.state}"
        if self.op is not None:
            return f"{self.name} {self.op} {float(self.value):g}"
        return f"not {self.name}" if self.negate else str(self.name)


@dataclass
class _Rule:
    """One transformation rule: a guard, consumed input coefficients and
    produced output coefficients."""

    cond: list[_RuleOperand]
    cons: dict[str, float]
    prod: dict[str, float]
    name: str | None = None

    @property
    def is_default(self) -> bool:
        """A rule carrying no guard is the default of its set: it applies
        when no guarded rule matches."""
        return not self.cond


@dataclass
class _RuleSet:
    """An ordered set of transformation rules declared on a component.

    The rules do **not** live on an output flow: a reaction with
    correlated outputs cannot be stated one output at a time, and its
    limiting-reagent scale must be shared rather than duplicated."""

    name: str
    rules: list[_Rule]
    #: Share of a contested output's demand this set claims, keyed by
    #: produced flow. A **RAICHU extension** (R13): muscadet hands each
    #: set the whole of a shared output's demand and apportions nothing,
    #: so the two sets between them produce twice what is asked and drop
    #: the surplus. Normalised across the sets producing that output.
    apportionment: dict[str, float] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """Name of the automaton selecting the active rule."""
        return f"{self.name}_mode"

    @property
    def has_guards(self) -> bool:
        return any(not rule.is_default for rule in self.rules)

    @property
    def default_index(self) -> int | None:
        for index, rule in enumerate(self.rules):
            if rule.is_default:
                return index
        return None

    def label(self, index: int) -> str:
        """The state one rule occupies in the mode automaton."""
        return self.rules[index].name or f"rule_{index}"

    def states(self) -> list[str]:
        """The mode's locations: one per rule, plus one meaning "no rule
        applies" when the set declares no default."""
        names = [self.label(index) for index in range(len(self.rules))]
        if self.default_index is None:
            names.append(_NO_RULE)
        return names

    def consumed(self) -> list[str]:
        """Every flow any of the rules consumes, in declaration order."""
        return _ordered_union(rule.cons for rule in self.rules)

    def produced(self) -> list[str]:
        """Every flow any of the rules produces, in declaration order."""
        return _ordered_union(rule.prod for rule in self.rules)


def _ordered_union(maps) -> list[str]:
    """The keys of several mappings, deduplicated, first-seen order."""
    names: list[str] = []
    for mapping in maps:
        for name in mapping:
            if name not in names:
                names.append(name)
    return names


def _as_operand_list(where: str, value: Any) -> list[Any]:
    """A rule guard, normalised to a list of operand specifications.

    An absent guard is an empty conjunction, which is what makes its
    rule the default of its set."""
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(
        f"{where} carries a guard that is neither a name, a mapping nor a "
        f"list: {value!r}"
    )


def _as_coefficients(where: str, key: str, value: Any) -> dict[str, float]:
    """A ``cons`` or ``prod`` map, normalised.

    A list of flow names is the short form of the same names at
    coefficient 1, which is what most rules carry."""
    if value is None:
        return {}
    if isinstance(value, (list, tuple, set)):
        return {str(name): 1.0 for name in value}
    if isinstance(value, dict):
        return {str(name): float(coefficient) for name, coefficient in value.items()}
    raise ValueError(
        f"{where} carries a `{key}` map that is neither a mapping nor a list "
        f"of names: {value!r}"
    )


def _weighted_fill(component: str, capacity: _Capacity) -> dict[str, Any]:
    """The capacity's total weighted fill, written over the **integrated
    contents** rather than over the reported ``_fill`` variables.

    A bound is a property of the integrated state, and this expression is
    what the solver root-finds a crossing on: reading the explicit
    reporting variables instead would put a swept value between the
    solver and the state it is locating."""
    terms: list[dict[str, Any]] = []
    for entry in capacity.flows:
        content = _var(component, f"{capacity.name}_content_{entry.name}")
        terms.append(
            content
            if entry.weight == 1.0
            else {"op": "mul", "args": [content, _float(entry.weight)]}
        )
    return {"op": "div", "lhs": _sum(terms), "rhs": _float(capacity.volume)}


def _flow_fill(
    component: str, capacity: _Capacity, entry: _CapacityFlow
) -> dict[str, Any]:
    """One held flow's weighted share of the volume, written over its
    **integrated content** for the reason :func:`_weighted_fill` gives.

    The weight is written out even when it is 1, where the terms of the
    total drop it: there the factor sits inside a sum this layer keeps
    free of no-op multiplications, while here it *is* the whole
    expression, and one shape whatever the weight is what keeps the
    reported constituent readable against its declaration."""
    return {
        "op": "div",
        "lhs": {
            "op": "mul",
            "args": [
                _var(component, f"{capacity.name}_content_{entry.name}"),
                _float(entry.weight),
            ],
        },
        "rhs": _float(capacity.volume),
    }


def _fill_threshold(
    component: str, capacity: _Capacity, cmp_op: str, value: float
) -> dict[str, Any]:
    """One bound comparison on the total weighted fill."""
    return {
        "op": "cmp",
        "cmp": cmp_op,
        "lhs": _weighted_fill(component, capacity),
        "rhs": _float(value),
    }


def _level_alias(channel: str, flow: str | None = None) -> str:
    """The name a measurement channel publishes a level under.

    Publisher and observer both build their port names here on purpose:
    the two sides are matched by name, and a convention that drifted by
    one underscore would fail the connection rather than the alias."""
    return f"{channel}_level" if flow is None else f"{channel}_level_{flow}"


def _fill_alias(channel: str, flow: str | None = None) -> str:
    """The name a measurement channel publishes a weighted fill under.
    See :func:`_level_alias`."""
    return f"{channel}_fill" if flow is None else f"{channel}_fill_{flow}"


@dataclass
class _CapacityFlow:
    """One flow a capacity holds, and how much volume a unit of it
    occupies."""

    name: str
    weight: float = 1.0


@dataclass
class _Capacity:
    """A volume a component holds over one or more of its continuous
    flows."""

    name: str
    flows: list[_CapacityFlow]
    volume: float
    side: str
    content_init: dict[str, float] = field(default_factory=dict)
    #: What the capacity claims for itself on each held flow, over and
    #: above the demand already passing through it, for as long as it has
    #: room. ``0`` is a pure buffer; ``math.inf`` is "whatever the
    #: producer can deliver", which this layer resolves to the published
    #: capability since a document cannot carry an infinity.
    fill_rate: float = 0.0
    hysteresis: float = DEFAULT_HYSTERESIS

    @property
    def flow_names(self) -> list[str]:
        return [entry.name for entry in self.flows]

    def content_of(self, flow: str) -> float:
        return float(self.content_init.get(flow, 0.0))

    def initial_fill(self) -> float:
        return sum(self.content_of(e.name) * e.weight for e in self.flows) / self.volume

    def published(self) -> list[tuple[str, str]]:
        """The ``(port alias, exported attribute)`` pairs the capacity
        publishes: the two totals, then each constituent.

        The totals alone would not do. A volume holding water and heat is
        at a temperature of ``heat / water``, while the total level is
        their weighted sum, which is neither term; an observer wanting
        the ratio needs both, so both are published."""
        pairs = [
            (_level_alias(self.name), f"{self.name}_content"),
            (_fill_alias(self.name), f"{self.name}_fill"),
        ]
        for entry in self.flows:
            pairs.append(
                (
                    _level_alias(self.name, entry.name),
                    f"{self.name}_content_{entry.name}",
                )
            )
            pairs.append(
                (_fill_alias(self.name, entry.name), f"{self.name}_fill_{entry.name}")
            )
        return pairs


@dataclass
class _MeasurementIn:
    """The reading side of a measurement link: a read-only view of a
    published level, carrying no quantity."""

    name: str
    #: Constituents of the observed volume to read individually, beside
    #: the totals.
    flows: list[str] = field(default_factory=list)

    def channels(self) -> list[str]:
        """The variables (and, suffixed with ``_in``, the ports) this
        reader materialises, in the order the publisher lists them."""
        names = [_level_alias(self.name), _fill_alias(self.name)]
        for flow in self.flows:
            names.append(_level_alias(self.name, flow))
            names.append(_fill_alias(self.name, flow))
        return names


@dataclass
class _ContinuousEdge:
    """One continuous connection, as the resolution sees it: the edge
    the per-connection channels are materialised under, and the two
    flows it joins."""

    name: str
    producer: str
    flow_out: str
    consumer: str
    flow_in: str


@dataclass
class _FailureMode:
    name: str
    law: str  # "delay" | "exp"
    failure_param: float
    repair_param: float
    targets: list[str] = field(default_factory=list)  # affected out-flows
    failure_cond: str | None = None  # local variable gating the failure
    #: What this mode leaves of a continuous output while it stands, and
    #: what it leaves once repaired, keyed by flow name and resolved from
    #: the declared effect patterns. A flow absent from the repair map
    #: returns to :data:`NOMINAL_RATE`: the release is implicit, a
    #: derating having no per-step reset of its own.
    failure_deratings: dict[str, float] = field(default_factory=dict)
    repair_deratings: dict[str, float] = field(default_factory=dict)


class ObjFlow:
    """A muscadet-style smart flow component. Subclass and override
    :meth:`add_flows`."""

    def __init__(self, name: str):
        self.name = name
        self._init_declarations()
        self.add_flows()

    def _init_declarations(self) -> None:
        """Empty declaration lists, in one place: the serialized entry
        point (`pyraichu.plugins.muscadet`) builds an `ObjFlow` without
        running `__init__`, and must not have to know the list of
        lists."""
        self.flows_in: list[_FlowIn] = []
        self.flows_out: list[_FlowOut] = []
        self.flows_continuous_in: list[_FlowContinuousIn] = []
        self.flows_continuous_out: list[_FlowContinuousOut] = []
        self.capacities: list[_Capacity] = []
        self.measurements_in: list[_MeasurementIn] = []
        self.rule_sets: list[_RuleSet] = []
        self.transfers: list[_Transfer] = []
        self.failure_modes: list[_FailureMode] = []
        # Decoration a declaration may carry (`pyraichu.declare`). It reaches
        # no generated model: it is held so a declaration survives a round
        # trip through a live component rather than losing what a platform
        # export knows about the instance.
        self.label: str | None = None
        self.description: str | None = None
        self.metadata: dict[str, Any] = {}

    def add_flows(self) -> None:  # pragma: no cover - overridden
        """Declare flows (override in subclasses)."""

    def add_flow_in(self, name: str, logic: str | int = "or") -> None:
        """Declare an incoming flow aggregated with `logic`
        (``"or"``, ``"and"`` or an integer k for k-out-of-n)."""
        if isinstance(logic, int):
            self.flows_in.append(_FlowIn(name=name, logic="k", k=logic))
        else:
            self.flows_in.append(_FlowIn(name=name, logic=logic))

    def add_flow_out(
        self,
        name: str,
        var_prod_default: bool = False,
        var_prod_cond: list[str] | None = None,
    ) -> None:
        """Declare an outgoing flow, produced unconditionally
        (``var_prod_default=True``) or when the named in-flows are fed."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
            )
        )

    def add_flow_continuous_in(
        self,
        name: str,
        var_demand_in_default: float = 0.0,
        var_in_default: float = 0.0,
    ) -> None:
        """Declare a real-valued input, aggregating every incoming
        connection by sum.

        `var_demand_in_default` is what this consumer asks its
        producers for while nothing derives a demand for it (a pure
        consumer derives none); `var_in_default` is what the input
        reads, on both the received and the capability channel, when
        nothing is connected to it."""
        self._reject_flow_clash(name, "in")
        self.flows_continuous_in.append(
            _FlowContinuousIn(
                name=name,
                var_demand_in_default=float(var_demand_in_default),
                var_in_default=float(var_in_default),
            )
        )

    def add_flow_continuous_out(
        self,
        name: str,
        var_fed_default: float = 0.0,
        allocation: str = "proportional",
        allocation_shares: dict[Any, float] | None = None,
        allocation_priorities: dict[Any, float] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> None:
        """Declare a real-valued output delivering `var_fed_default`
        when asked without bound.

        `allocation` names the policy splitting a shortage among the
        consumers: ``"proportional"`` to their demands (the default),
        ``"shares"`` in the declared ratio, or ``"priority"`` in
        ascending rank. The two keyed policies read their parameter map
        keyed by consumer component name (or by a ``(component, flow)``
        pair when one consumer reads this output over two flows), never
        by connection position, so reordering a connection cannot
        re-attach a share to a different consumer.

        `profile` is a declared **continuous** function of simulation
        time multiplying what this output produces, as the mapping
        ``{"cls": "SinusoidalProfile", ...}`` carrying `amplitude`,
        `period`, `phase_shift`, `offset`, `value_min` and `value_max`.
        A Python callable is refused: continuity is an attestation this
        layer cannot make for the modeller, and it derives no watched
        transition from a function's breakpoints. The profile composes
        with the derating rate by **product**, never by minimum: an
        output at 0.3 of its profile that is also derated to 0.5
        produces 0.15."""
        if allocation not in ("proportional", "shares", "priority"):
            raise ValueError(
                f"ObjFlow `{self.name}`: continuous out-flow `{name}` declares "
                f"unknown allocation policy `{allocation}` (expected "
                "'proportional', 'shares' or 'priority')"
            )
        self._reject_flow_clash(name, "out")
        self.flows_continuous_out.append(
            _FlowContinuousOut(
                name=name,
                var_fed_default=float(var_fed_default),
                allocation=allocation,
                allocation_shares=(
                    dict(allocation_shares) if allocation_shares is not None else None
                ),
                allocation_priorities=(
                    dict(allocation_priorities)
                    if allocation_priorities is not None
                    else None
                ),
                profile=self._parse_profile(name, profile),
            )
        )

    # --- time profiles --------------------------------------------------

    def _parse_profile(self, flow: str, spec: Any) -> _Profile | None:
        """One declared time profile, normalised, or refused where it was
        written (R8).

        The only shape a mapping can carry is the clamped sinusoid: a
        bare callable is refused because the continuity flag is the one
        thing this layer cannot work out for itself, and a mapping naming
        the callable shape offers a serialised form that cannot be
        serialised."""
        where = f"ObjFlow `{self.name}`: continuous out-flow `{flow}`"
        if spec is None or isinstance(spec, _Profile):
            return spec
        if callable(spec):
            raise ValueError(
                f"{where} declares its time profile as a Python function; "
                f"{_CONTINUITY_MESSAGE}. Declare it as "
                f'{{"cls": "{_PROFILE_CLASS}", ...}}'
            )
        if not isinstance(spec, dict):
            raise ValueError(
                f"{where} declares {spec!r} as a time profile, which is not "
                f'one: a profile is {{"cls": "{_PROFILE_CLASS}", ...}}'
            )

        params = dict(spec)
        shape = params.pop("cls", None)
        if shape is None:
            raise ValueError(
                f"{where} declares a time profile with no `cls` key naming "
                f"its shape; the one declarable shape is `{_PROFILE_CLASS}`"
            )
        if shape != _PROFILE_CLASS:
            raise ValueError(
                f"{where} declares the time profile shape `{shape}`; the one "
                f"declarable shape is `{_PROFILE_CLASS}`, every other profile "
                f"being a Python function. {_CONTINUITY_MESSAGE}"
            )
        known = (
            "amplitude",
            "period",
            "phase_shift",
            "offset",
            "value_min",
            "value_max",
        )
        unknown = sorted(set(params) - set(known))
        if unknown:
            raise ValueError(
                f"{where} declares a time profile carrying unknown keys "
                f"{unknown} (expected {list(known)})"
            )

        period = float(params.get("period", 2 * math.pi))
        if not period > 0.0:
            raise ValueError(
                f"{where} declares a time profile of period {period:g}; the "
                "period is the duration of one cycle and must be strictly "
                "positive"
            )
        value_min = float(params.get("value_min", 0.0))
        if value_min < 0.0:
            raise ValueError(
                f"{where} declares a time profile clamped at value_min="
                f"{value_min:g}. A profile SCALES production, so a negative "
                "factor would mean a negative quantity, which no balance here "
                "models: use value_min=0 to cut the negative half-cycle, or "
                "lift the whole curve with an offset"
            )
        value_max = params.get("value_max")
        value_max = (
            None if value_max is None or value_max == math.inf else float(value_max)
        )
        if value_max is not None and value_max < value_min:
            raise ValueError(
                f"{where} declares a time profile whose clamps are the wrong "
                f"way round: value_min={value_min:g} exceeds "
                f"value_max={value_max:g}"
            )
        return _Profile(
            amplitude=float(params.get("amplitude", 1.0)),
            period=period,
            phase_shift=float(params.get("phase_shift", 0.0)),
            offset=float(params.get("offset", 0.0)),
            value_min=value_min,
            value_max=value_max,
        )

    # --- transfer pairs -------------------------------------------------

    def add_transfer(
        self,
        name: str,
        flows: list[str] | tuple[str, str] | None = None,
        equation: dict[str, Any] | None = None,
    ) -> None:
        """Declare a transfer pair: a computed quantity this component
        moves between two flows because a gradient drives it (muscadet
        `add_transfer`, R5).

        The rest of this layer transports quantities pulled by **demand**,
        which is the wrong vehicle for one that moves because a gradient
        makes it move: heat across an exchanger wall, moles through a
        membrane, charge through a resistance. A pair is that second
        vehicle, and the equation returns a **signed** quantity whose sign
        this layer routes, so a model never writes a direction clamp.

        `flows` names exactly two continuous **outputs**, source first: a
        pair writes both balances and a balance is written on the output
        side. Naming the same flow twice is the **metered conduit**, which
        additionally needs the input side, since it meters a transit and
        there is no transit to meter without one; what crosses the
        component is then the computed quantity, in place of the rate the
        output was declared with. Naming two flows is the **two-stream
        exchange**, a signed delta on top of what both streams carry.

        `equation` is the mapping ``{"cls": "ConductiveTransfer",
        "conductance": G, "potential_a": ..., "potential_b": ...}``,
        which is ``G x (potential_a - potential_b)``. A potential is a
        number, ``{"const": value}`` or ``{"measurement": name}``
        (optionally narrowed to one constituent with ``"flow"``): the two
        operand forms muscadet admits, and deliberately no more. A Python
        function is refused per R8, naming this component and this
        transfer: it carries no continuity attestation, no serialised
        form, and widening the vocabulary to buy one model would buy no
        migration path.

        Declare a pair **after** the flows and measurement channels it
        names: they are resolved here, so a misspelling is refused at
        declaration rather than reaching the engine as a dangling read."""
        where = f"ObjFlow `{self.name}`: transfer `{name}`"
        if any(existing.name == name for existing in self.transfers):
            raise ValueError(f"{where} is already declared")

        named = list(flows or [])
        if len(named) != 2:
            raise ValueError(
                f"{where} names {len(named)} flows: {named!r}. A pair names "
                "exactly two, source first; name the same flow twice for a "
                "metered conduit"
            )

        producible = {flow.name for flow in self.flows_continuous_out}
        consumable = {flow.name for flow in self.flows_continuous_in}
        for flow in dict.fromkeys(named):
            if flow not in producible:
                raise ValueError(
                    f"{where} moves `{flow}`, which this component declares as "
                    f"no continuous out-flow (it declares {sorted(producible)}). "
                    "A pair writes both balances, and a balance is written on "
                    "the output side"
                )

        if named[0] == named[1] and named[0] not in consumable:
            raise ValueError(
                f"{where} meters `{named[0]}`, which this component declares "
                f"as no continuous in-flow (it declares {sorted(consumable)}). "
                "A conduit meters a transit, and there is no transit to meter "
                "without an input"
            )

        conductance, potential_a, potential_b = self._parse_transfer_equation(
            where, equation
        )
        # The clash with a rule set and with a conduit is checked by
        # `_record`, which reads the whole component: written here, it
        # held in one declaration order only.
        self._record(
            self.transfers,
            _Transfer(
                name=name,
                flows=(named[0], named[1]),
                conductance=conductance,
                potential_a=potential_a,
                potential_b=potential_b,
            ),
        )

    def _parse_transfer_equation(self, where: str, spec: Any):
        """The declared equation, normalised, or refused where it was
        written (R8, KTD15).

        Only the conductive law is declarable, and only over muscadet's
        two potential operands. The refusal of everything else is
        load-bearing rather than provisional: two of the documented
        reference models compute a ratio of one volume's constituents,
        which is neither operand form, and they are authored natively
        instead of widening this vocabulary."""
        if callable(spec):
            raise ValueError(
                f"{where} declares its equation as a Python function; "
                f"{_CONTINUITY_MESSAGE}. Declare it as "
                f'{{"cls": "{_TRANSFER_CLASS}", ...}}'
            )
        if not isinstance(spec, dict):
            raise ValueError(
                f"{where} declares {spec!r} as its equation, which is not "
                f'one: an equation is {{"cls": "{_TRANSFER_CLASS}", ...}}'
            )

        params = dict(spec)
        law = params.pop("cls", None)
        if law is None:
            raise ValueError(
                f"{where} declares an equation with no `cls` key naming its "
                f"law; the one declarable law is `{_TRANSFER_CLASS}`"
            )
        if law != _TRANSFER_CLASS:
            raise ValueError(
                f"{where} declares the transfer law `{law}`; the one "
                f"declarable law is `{_TRANSFER_CLASS}`, every other law being "
                f"a Python function. {_CONTINUITY_MESSAGE}"
            )
        unknown = sorted(set(params) - {"conductance", "potential_a", "potential_b"})
        if unknown:
            raise ValueError(
                f"{where} declares an equation carrying unknown keys "
                f"{unknown} (expected ['conductance', 'potential_a', "
                "'potential_b'])"
            )
        if "conductance" not in params:
            raise ValueError(
                f"{where} declares a `{_TRANSFER_CLASS}` with no conductance; "
                "it is the proportionality constant the potential difference "
                "is multiplied by"
            )
        conductance = float(params["conductance"])
        if conductance < 0.0:
            raise ValueError(
                f"{where} declares a conductance of {conductance:g}; a "
                "negative conductance would drive the quantity UP its own "
                "gradient, which is not transport. Swap the two potentials to "
                "reverse the direction"
            )
        for key in ("potential_a", "potential_b"):
            if params.get(key) is None:
                raise ValueError(
                    f"{where} declares no `{key}`; the quantity moved is the "
                    "difference of the two potentials, so both are required"
                )
        return (
            conductance,
            self._parse_potential(where, "potential_a", params["potential_a"]),
            self._parse_potential(where, "potential_b", params["potential_b"]),
        )

    def _parse_potential(self, where: str, key: str, spec: Any):
        """One declared potential, in the two forms and only the two.

        Forming a ratio from a volume's own contents is **not** an
        operand form: a measurement channel already publishes each
        constituent it names, so a component dividing two of them
        publishes the intensive property itself, which is what a probe
        reporting a temperature from a heat and a mass is. The declarable
        law reads a potential; it does not compute one."""
        advice = (
            'Use a number, {"const": value} for a fixed potential, or '
            '{"measurement": name} (optionally with "flow") to read one over '
            "a measurement channel. A ratio of a volume's own constituents is "
            "not an operand: publish it as a reading and read that"
        )
        if isinstance(spec, bool):
            raise ValueError(f"{where}: `{key}` is {spec!r}, not a potential. {advice}")
        if isinstance(spec, (int, float)):
            return ("const", float(spec), None)
        if isinstance(spec, dict):
            if "const" in spec and "measurement" in spec:
                raise ValueError(
                    f"{where}: `{key}` declares both `const` and `measurement`. "
                    "Declare one or the other, rather than leaving the law "
                    "running against a frozen potential for the whole mission"
                )
            if "const" in spec:
                return ("const", float(spec["const"]), None)
            if "measurement" in spec:
                channel = str(spec["measurement"])
                link = next(
                    (m for m in self.measurements_in if m.name == channel), None
                )
                if link is None:
                    raise ValueError(
                        f"{where}: `{key}` reads measurement channel "
                        f"`{channel}`, which this component does not declare; "
                        "declare it with add_measurement_in before reading it"
                    )
                constituent = spec.get("flow")
                if constituent is not None and constituent not in link.flows:
                    raise ValueError(
                        f"{where}: `{key}` reads constituent `{constituent}` of "
                        f"measurement channel `{channel}`, which reads "
                        f"{link.flows}"
                    )
                return ("measurement", channel, constituent)
        raise ValueError(f"{where}: `{key}` is {spec!r}, not a potential. {advice}")

    # --- capacities ---------------------------------------------------

    def add_capacity(
        self,
        name: str,
        flow: str | dict[str, Any] | None = None,
        flows: list[str | dict[str, Any]] | None = None,
        capacity: float | None = None,
        side: str | None = None,
        content_init: dict[str, float] | None = None,
        fill_rate: float | None = 0.0,
        hysteresis: float | None = None,
    ) -> None:
        """Declare a volume this component holds over one or more of its
        continuous flows (muscadet `add_capacity`).

        A capacity is declared **independently of the flows it holds**, so
        a buffer can be added to an existing component without touching
        its flow logic. The flows must already be declared, so call this
        after the ``add_flow_continuous_*`` calls.

        `flow` is the single-flow short form of `flows`, which takes flow
        names or mappings carrying ``name`` and ``weight``; the weight
        (default 1) is how much volume one unit of that flow occupies, and
        `capacity` is the volume they **share**. `content_init` gives the
        initial raw quantity per held flow, an omitted flow starting
        empty. `fill_rate` is what the capacity claims for itself on each
        held flow, over and above the demand already passing through it,
        for as long as it has room: ``0`` is a pure buffer that never
        stocks up, ``math.inf`` is whatever the producer can deliver.
        `hysteresis` overrides :data:`DEFAULT_HYSTERESIS`, the fraction of
        the volume the content must move back from a bound before the
        capacity leaves it.

        The demand a **full** capacity carries upstream is what it can
        still take, which is what currently leaves it, capped by the
        demand already passing through it. That pass-through demand is
        whatever the flow would ask for without the volume: a rule set's
        derived demand when one consumes the flow, and the in-flow's
        declared ``var_demand_in_default`` otherwise. A tank that
        delivers five, that no rule consumes and that declares no
        pass-through demand asks for nothing once full, and therefore
        drains; declare it, and the buffer holds. The two compose in
        either declaration order: the bound is applied when the model is
        generated, not when the capacity is declared.

        `side` places the capacity upstream (``"in"``) or downstream
        (``"out"``) of the component's transformation rules, as muscadet
        does. It is validated and recorded, but **not yet discriminating**:
        this layer has no rules for it to sit either side of, so the
        bounds govern both ends of every held flow the component declares.
        A pure buffer is the same object read from either side, which is
        why one capacity per flow is enough here and a second one on the
        other side is refused rather than silently ignored.

        A volume over a flow a **rule set of this component** consumes or
        produces is refused, in either declaration order: the rule and
        the volume would each carry that flow, and the component would
        make matter out of the double count. A volume on another
        component, upstream or downstream of the rules, is the ordinary
        shape and conserves; see :meth:`_refuse_flows_carried_twice`."""
        declared_in = {f.name: f for f in self.flows_continuous_in}
        declared_out = {f.name: f for f in self.flows_continuous_out}

        if any(existing.name == name for existing in self.capacities):
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` is already declared"
            )
        if any(existing.name == name for existing in self.measurements_in):
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` clashes with the "
                "measurement link of that name"
            )
        if flow is not None and flows is not None:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` takes either `flow` "
                "(the single-flow short form) or `flows`, not both"
            )
        specs = flows if flows is not None else ([] if flow is None else [flow])
        if not specs:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` must hold at least one flow"
            )
        if capacity is None or not float(capacity) > 0.0:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` must declare a "
                f"strictly positive volume, got {capacity}"
            )
        # `None` is the declared spelling of an unbounded fill: a volume
        # that claims whatever its supplier can deliver while it is not
        # full. It exists because a document has to be able to say it and
        # JSON has no literal for an infinity, so the alternative was a
        # string coerced by accident.
        rate = float("inf") if fill_rate is None else float(fill_rate)
        if not rate >= 0.0:  # negative or NaN
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` must declare a fill "
                f"rate that is positive, zero, or null for unbounded, got "
                f"{fill_rate}"
            )
        width = DEFAULT_HYSTERESIS if hysteresis is None else float(hysteresis)
        if not 0.0 <= width < 1.0:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` must declare a "
                f"hysteresis width in [0, 1), got {hysteresis}"
            )

        entries: list[_CapacityFlow] = []
        for spec in specs:
            if isinstance(spec, str):
                entries.append(_CapacityFlow(name=spec))
            elif isinstance(spec, dict):
                if not spec.get("name"):
                    raise ValueError(
                        f"ObjFlow `{self.name}`: capacity `{name}` holds a "
                        f"mapping naming no flow: {spec!r}; a mapping carries "
                        "`name` and an optional `weight`"
                    )
                weight = float(spec.get("weight", 1.0))
                if not weight > 0.0:  # negative, zero or NaN
                    raise ValueError(
                        f"ObjFlow `{self.name}`: capacity `{name}` must declare "
                        f"a strictly positive weight for `{spec['name']}`, got "
                        f"{spec.get('weight')}. A weight is how much volume one "
                        "unit of that flow occupies, so a zero weight would "
                        "leave the volume unbounded"
                    )
                entries.append(_CapacityFlow(name=str(spec["name"]), weight=weight))
            else:
                raise ValueError(
                    f"ObjFlow `{self.name}`: capacity `{name}` holds an entry "
                    f"that is neither a flow name nor a mapping: {spec!r}"
                )
        held = [entry.name for entry in entries]
        duplicates = sorted({n for n in held if held.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` holds the same flow "
                f"twice: {', '.join(duplicates)}"
            )
        for entry in entries:
            if entry.name not in declared_in and entry.name not in declared_out:
                raise ValueError(
                    f"ObjFlow `{self.name}`: capacity `{name}` holds `"
                    f"{entry.name}`, which this component declares as no "
                    "continuous flow (declare the flow before the capacity)"
                )
            for existing in self.capacities:
                if entry.name in existing.flow_names:
                    raise ValueError(
                        f"ObjFlow `{self.name}`: flow `{entry.name}` is already "
                        f"held by capacity `{existing.name}`; a flow buffers in "
                        "one volume"
                    )
        unknown = sorted(set(content_init or {}) - set(held))
        if unknown:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` declares "
                f"`content_init` for {unknown}, which it does not hold"
            )

        # Side. Declared, it must be carried by every held flow; left out,
        # it is resolved from them and a flow carried both ways prefers
        # the input side, as muscadet resolves it.
        def sides_of(entry: _CapacityFlow) -> list[str]:
            return [
                carried
                for carried, declared in (("in", declared_in), ("out", declared_out))
                if entry.name in declared
            ]

        if side is not None:
            if side not in ("in", "out"):
                raise ValueError(
                    f"ObjFlow `{self.name}`: capacity `{name}` declares side "
                    f"`{side}`, expected 'in' or 'out'"
                )
            for entry in entries:
                if side not in sides_of(entry):
                    raise ValueError(
                        f"ObjFlow `{self.name}`: capacity `{name}` declares side "
                        f"`{side}`, which its held flow `{entry.name}` does not "
                        f"carry (it carries {sides_of(entry)})"
                    )
            resolved = side
        else:
            preferred = {sides_of(entry)[0] for entry in entries}
            if len(preferred) > 1:
                raise ValueError(
                    f"ObjFlow `{self.name}`: capacity `{name}` holds flows that "
                    "resolve to different sides; declare `side` explicitly"
                )
            resolved = preferred.pop()

        declaration = _Capacity(
            name=name,
            flows=entries,
            volume=float(capacity),
            side=resolved,
            content_init={
                key: float(value) for key, value in (content_init or {}).items()
            },
            fill_rate=rate,
            hysteresis=width,
        )
        self._record(self.capacities, declaration)

    # --- one flow, one carrier -----------------------------------------

    def _record(self, into: list, declaration: Any) -> None:
        """Record `declaration`, refusing it at once when it makes a flow
        cross this component twice.

        The check reads the whole component rather than the declaration
        alone, so the refusal does not depend on which half of a clashing
        pair was written first. A refused declaration is taken back out,
        leaving the component as it was."""
        into.append(declaration)
        try:
            self._refuse_flows_carried_twice()
        except ValueError:
            into.pop()
            raise

    def _refuse_flows_carried_twice(self) -> None:
        """Refuse every composition that would make one flow cross this
        component twice, in whichever order its halves were declared.

        A rule set, a metered conduit and a capacity each speak for what
        a flow carries: the rule transforms it, the conduit replaces its
        transit, the volume stores it. Two of them on one flow do not
        compose, they double-count, and what leaves the component stops
        matching what entered it. The three pairings are refused here
        rather than inside the three declaration methods, which is what
        makes them order-independent: each declaration runs this as an
        early diagnostic through :meth:`_record`, and :meth:`_build` runs
        it again on the finished component, so a declaration list built
        by other means is caught too."""
        me = f"ObjFlow `{self.name}`"
        carried = {
            flow
            for rule_set in self.rule_sets
            for flow in rule_set.consumed() + rule_set.produced()
        }
        metered = {pair.source for pair in self.transfers if pair.is_conduit}

        for capacity in self.capacities:
            clash = sorted(set(capacity.flow_names) & carried)
            if clash:
                raise ValueError(
                    f"{me}: capacity `{capacity.name}` holds `{clash[0]}`, "
                    "which a rule set of this component already consumes or "
                    "produces. A rule TRANSFORMS what crosses the component "
                    "and a volume STORES it, so the flow would cross twice, "
                    "once by the rule and once by the capacity, and the "
                    "component would make matter. Hold the flow in a volume "
                    "on another component, upstream or downstream of the "
                    "rules, which is where a buffer conserves"
                )

        for pair in self.transfers:
            if not pair.is_conduit or pair.source not in carried:
                continue
            raise ValueError(
                f"{me}: transfer `{pair.name}` meters `{pair.source}`, which a "
                "rule set of this component already consumes or produces. A "
                "conduit REPLACES what crosses the component, so the flow "
                "would cross twice, once by the rule and once by the pair. "
                "Meter a flow the rules leave alone, or express the metering "
                "in the rule itself"
            )

        for pair in self.transfers:
            if pair.is_conduit:
                continue
            clash = [
                flow for flow in (pair.source, pair.destination) if flow in metered
            ]
            if clash:
                raise ValueError(
                    f"{me}: transfer `{pair.name}` moves {clash}, which a "
                    "conduit of this component already meters. A conduit "
                    "replaced that flow's transit, so a delta on top of it has "
                    "no stream to sit on"
                )

    def _capacity_of(self, flow: str) -> _Capacity | None:
        """The volume holding `flow`, or ``None``. A flow buffers in one
        volume, which :meth:`add_capacity` enforces."""
        for capacity in self.capacities:
            if flow in capacity.flow_names:
                return capacity
        return None

    def _capacity_bounded_demand(
        self, flow: str, base: dict[str, Any]
    ) -> dict[str, Any]:
        """`base`, the demand a flow carries upstream, bounded by the
        volume holding it.

        Full, the capacity asks only for what it can still take, which is
        what currently leaves it; below its bound it claims its fill rate
        on top of the demand already passing through it. The bound is
        read from the automaton **location** rather than from the fill,
        so the decision is the one the located crossing made and never a
        second, unlocated comparison.

        Applied here rather than at declaration, so a rule set and a
        capacity naming one flow compose whichever is declared first:
        the rule derives the pass-through demand, the volume bounds it."""
        capacity = self._capacity_of(flow)
        if capacity is None:
            return base
        me = self.name
        outflow = (
            _var(me, f"{flow}_fed_out")
            if any(declared.name == flow for declared in self.flows_continuous_out)
            else _float(0.0)
        )
        if capacity.fill_rate == float("inf"):
            # "Whatever the producer can deliver", which a document
            # cannot carry as an infinity: the published capability is
            # that quantity, and the pass-through demand still gets
            # through when it exceeds it.
            claiming = {"op": "max", "args": [base, _var(me, f"{flow}_capability_in")]}
        elif capacity.fill_rate:
            claiming = {"op": "add", "args": [base, _float(capacity.fill_rate)]}
        else:
            claiming = base
        return {
            "op": "if",
            "cond": _state_active(me, f"{capacity.name}_bounds", "full"),
            "then": {"op": "min", "args": [base, outflow]},
            "otherwise": claiming,
        }

    def _capacity_bounded_capability(
        self, flow: str, nominal: dict[str, Any]
    ) -> dict[str, Any]:
        """`nominal`, what a flow could deliver, bounded by the volume
        holding it: empty, it serves downstream only what currently
        transits through it. See :meth:`_capacity_bounded_demand`."""
        capacity = self._capacity_of(flow)
        if capacity is None:
            return nominal
        me = self.name
        inflow = (
            _var(me, f"{flow}_fed_in")
            if any(declared.name == flow for declared in self.flows_continuous_in)
            else _float(0.0)
        )
        return {
            "op": "if",
            "cond": _state_active(me, f"{capacity.name}_bounds", "empty"),
            "then": {"op": "min", "args": [nominal, inflow]},
            "otherwise": nominal,
        }

    def add_measurement_in(self, name: str, flows: list[str] | None = None) -> None:
        """Declare the reading side of a measurement link (muscadet
        `add_measurement_in`).

        The observer reads a published level and fill through ordinary in
        ports carrying no channel: the link is read-only by construction,
        exchanges no quantity and enters no allocation. `name` matches the
        observed capacity's name, which is what lines the two sides up;
        `flows` names constituents of the observed volume to read beside
        the totals, and a constituent the publisher does not hold is
        refused at connection.

        Connect with :meth:`System.connect_measurement`."""
        if any(existing.name == name for existing in self.measurements_in):
            raise ValueError(
                f"ObjFlow `{self.name}`: measurement link `{name}` is already declared"
            )
        if any(existing.name == name for existing in self.capacities):
            raise ValueError(
                f"ObjFlow `{self.name}`: measurement link `{name}` clashes with "
                "the capacity of that name"
            )
        self.measurements_in.append(_MeasurementIn(name=name, flows=list(flows or [])))

    # --- transformation rules -------------------------------------------

    def add_rule_set(
        self,
        name: str,
        rules: list[dict[str, Any]],
        apportionment: dict[str, float] | None = None,
    ) -> None:
        """Declare an ordered set of transformation rules (muscadet
        `add_rules`).

        Each entry of `rules` is a mapping carrying ``cond`` (the guard),
        ``cons`` (consumed input coefficients) and ``prod`` (produced
        output coefficients), plus an optional ``name`` designating the
        rule in diagnostics. A coefficient map may also be given as a
        list of flow names, each taken at coefficient 1.

        The set runs at the scale its **scarcest input and least demanded
        output** allow: the minimum of every consumed input divided by
        its coefficient and of every demanded output divided by its own.
        One scale serves the whole rule, which is what holds correlated
        outputs in their declared proportion.

        Rules are **ordered**: the guarded ones are tried in declaration
        order and the first whose guard holds is selected. A rule with no
        guard is the **default** of its set and applies when no guarded
        rule matches; a set may carry at most one, and a set with none
        selects nothing (and then produces nothing) while no guard holds.

        A guard is a conjunction of operands. An operand is a mapping
        naming a flow or a declared attribute (``name``, with ``port``
        selecting the side when a name is carried both ways), optionally
        ``negate``d or compared through ``op`` and ``value``; or an
        automaton gate (``automaton`` and ``state``). A bare string is
        the short form of ``{"name": ...}``. A continuous flow is read
        through the quantity its producers **could** deliver rather than
        through what it was given: a guard on the delivered quantity
        would be circular, since the delivery follows the demand the
        rule derives from that very guard. A capacity is read through
        its **integrated** content, never through the swept reporting
        variable, for the reason :meth:`add_capacity` states.

        `apportionment` declares this set's share of a produced output's
        demand, and is required of every set producing into an output
        another set also produces into (R13). muscadet has no field for
        it: it is a documented RAICHU extension, and its absence is
        refused rather than defaulted, because how two reactions share
        one product is a modelling question. The shares of one output are
        normalised, so ``3`` and ``1`` split it three to one.

        Declare a rule set **after** the flows, capacities and failure
        modes its coefficients and guards name: they are resolved here,
        so a misspelling is refused at declaration rather than reaching
        the engine as a dangling read."""
        if any(existing.name == name for existing in self.rule_sets):
            raise ValueError(
                f"ObjFlow `{self.name}`: rule set `{name}` is already declared"
            )
        parsed = [
            self._parse_rule(name, index, spec)
            for index, spec in enumerate(rules or [])
        ]
        if not parsed:
            raise ValueError(
                f"ObjFlow `{self.name}`: rule set `{name}` declares no rule"
            )

        defaults = [index for index, rule in enumerate(parsed) if rule.is_default]
        if len(defaults) > 1:
            raise ValueError(
                f"ObjFlow `{self.name}`: rule set `{name}` carries "
                f"{len(defaults)} unguarded rules; a set carries at most one "
                "unguarded rule, its default, or the selected rule is undefined"
            )

        declaration = _RuleSet(name=name, rules=parsed)
        labels = [declaration.label(index) for index in range(len(parsed))]
        duplicates = sorted({n for n in labels if labels.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"ObjFlow `{self.name}`: rule set `{name}` names two rules "
                f"alike: {', '.join(duplicates)}"
            )
        if defaults == [] and _NO_RULE in labels:
            raise ValueError(
                f"ObjFlow `{self.name}`: rule set `{name}` names a rule "
                f"`{_NO_RULE}`, which is the mode it sits in while no guard "
                "holds; name the rule otherwise"
            )

        produced = declaration.produced()
        for flow, share in (apportionment or {}).items():
            if flow not in produced:
                raise ValueError(
                    f"ObjFlow `{self.name}`: rule set `{name}` declares an "
                    f"apportionment of `{flow}`, which it does not produce "
                    f"(it produces {produced})"
                )
            if not float(share) > 0.0:
                raise ValueError(
                    f"ObjFlow `{self.name}`: rule set `{name}` declares a share "
                    f"of `{flow}` that is not strictly positive: {share}"
                )
        declaration.apportionment = {
            flow: float(share) for flow, share in (apportionment or {}).items()
        }
        self._record(self.rule_sets, declaration)

    def _parse_rule(self, set_name: str, index: int, spec: Any) -> _Rule:
        """One entry of a rule set, normalised and resolved against the
        component's declarations."""
        where = f"ObjFlow `{self.name}`: rule set `{set_name}`, rule {index}"
        if not isinstance(spec, dict):
            raise ValueError(f"{where} is not a mapping: {spec!r}")
        unknown = sorted(set(spec) - {"name", "cond", "cons", "prod"})
        if unknown:
            raise ValueError(
                f"{where} carries unknown keys {unknown} (expected `name`, "
                "`cond`, `cons`, `prod`)"
            )

        rule = _Rule(
            name=spec.get("name"),
            cond=[
                self._parse_operand(where, operand)
                for operand in _as_operand_list(where, spec.get("cond"))
            ],
            cons=_as_coefficients(where, "cons", spec.get("cons")),
            prod=_as_coefficients(where, "prod", spec.get("prod")),
        )

        consumable = {flow.name for flow in self.flows_continuous_in}
        producible = {flow.name for flow in self.flows_continuous_out}
        for flow in rule.cons:
            if flow not in consumable:
                raise ValueError(
                    f"{where} consumes `{flow}`, which this component declares "
                    f"as no continuous in-flow (it declares {sorted(consumable)})"
                )
        for flow in rule.prod:
            if flow not in producible:
                raise ValueError(
                    f"{where} produces `{flow}`, which this component declares "
                    f"as no continuous out-flow (it declares "
                    f"{sorted(producible)})"
                )
        for operand in rule.cond:
            # Resolved here for its diagnostics; the expression itself is
            # rebuilt at build time, from the same resolution.
            continuous = self._operand_reads_continuously(where, operand)
            if operand.op in _CMP_OPS and continuous and operand.op not in _ORDERING_OPS:
                raise ValueError(
                    f"{where} compares the continuous quantity `{operand.name}` "
                    f"with `{operand.op}`; the mode transition carrying it is "
                    "watched, and a crossing is located on an ordering "
                    f"comparison ({', '.join(_ORDERING_OPS)})"
                )
        return rule

    def _parse_operand(self, where: str, spec: Any) -> _RuleOperand:
        """One guard operand, from its mapping or its short string form."""
        if isinstance(spec, str):
            if not spec.isidentifier():
                raise ValueError(
                    f"{where} carries the guard operand {spec!r}; the short "
                    "form is a bare flow name, and anything else is written "
                    "as a mapping with `name`, `op` and `value`"
                )
            return _RuleOperand(name=spec)
        if not isinstance(spec, dict):
            raise ValueError(
                f"{where} carries a guard operand that is neither a name nor "
                f"a mapping: {spec!r}"
            )
        unknown = sorted(
            set(spec) - {"name", "port", "automaton", "state", "negate", "op", "value"}
        )
        if unknown:
            raise ValueError(
                f"{where} carries a guard operand with unknown keys {unknown}"
            )
        operand = _RuleOperand(
            name=spec.get("name"),
            port=spec.get("port"),
            automaton=spec.get("automaton"),
            state=spec.get("state"),
            negate=bool(spec.get("negate", False)),
            op=spec.get("op"),
            value=spec.get("value"),
        )
        if (operand.name is None) == (operand.automaton is None):
            raise ValueError(
                f"{where} carries a guard operand that names neither a flow "
                "(`name`) nor an automaton gate (`automaton` and `state`), or "
                f"names both: {spec!r}"
            )
        if operand.automaton is not None and operand.state is None:
            raise ValueError(
                f"{where} gates on automaton `{operand.automaton}` without "
                "naming a state"
            )
        if operand.port is not None and operand.port not in ("in", "out"):
            raise ValueError(
                f"{where} carries a guard operand with side `{operand.port}`, "
                "expected 'in' or 'out'"
            )
        if operand.op is not None:
            if operand.op not in _CMP_OPS:
                raise ValueError(
                    f"{where} compares with `{operand.op}`, expected one of "
                    f"{', '.join(_CMP_OPS)}"
                )
            if operand.value is None:
                raise ValueError(
                    f"{where} compares `{operand.name}` with `{operand.op}` "
                    "and no value"
                )
            operand.value = float(operand.value)
        return operand

    def _operand_read(
        self, where: str, operand: _RuleOperand
    ) -> tuple[dict[str, Any], tuple[str, str] | None]:
        """The expression a guard operand reads, and the flow endpoint it
        reads it at when that read is a **rate**: a continuous flow
        quantity rather than an integrated level or a discrete state.

        The distinction is what the rate-comparison diagnostic rests on:
        a threshold on a level is broken by the integration that carries
        it, where a threshold on a rate is instantaneous and can close a
        loop with no fixpoint."""
        me = self.name
        if operand.automaton is not None:
            declared = self._declared_automata()
            if operand.automaton not in declared:
                raise ValueError(
                    f"{where} gates on automaton `{operand.automaton}`, which "
                    f"this component does not declare (it declares "
                    f"{sorted(declared)})"
                )
            if operand.state not in declared[operand.automaton]:
                raise ValueError(
                    f"{where} gates on state `{operand.state}` of automaton "
                    f"`{operand.automaton}`, whose states are "
                    f"{declared[operand.automaton]}"
                )
            return _state_active(me, operand.automaton, operand.state), None

        name = operand.name
        continuous_in = {flow.name for flow in self.flows_continuous_in}
        continuous_out = {flow.name for flow in self.flows_continuous_out}
        discrete_in = {flow.name for flow in self.flows_in}
        discrete_out = {flow.name for flow in self.flows_out}
        sides = ("in", "out") if operand.port is None else (operand.port,)
        for side in sides:
            if side == "in":
                if name in continuous_in:
                    return _var(me, f"{name}_capability_in"), (name, "in")
                if name in discrete_in:
                    return _var(me, f"{name}_fed_in"), None
            else:
                if name in continuous_out:
                    return _var(me, f"{name}_capability_out"), (name, "out")
                if name in discrete_out:
                    return _var(me, f"{name}_fed_out"), None

        level = self._capacity_level(name)
        if level is not None:
            return level, None
        for measurement in self.measurements_in:
            if name in measurement.channels():
                return _var(me, name), None
        raise ValueError(
            f"{where} reads `{name}`, which this component declares as no "
            "flow, capacity level or measurement channel"
        )

    def _capacity_level(self, name: str) -> dict[str, Any] | None:
        """The expression a capacity level reads under, or ``None`` when
        `name` designates no level of this component.

        Written over the **integrated contents** rather than over the
        swept reporting variables, for the reason
        :func:`_weighted_fill` states: a guard on a level is located by
        root-finding, and reading the reporting variable would put a
        swept value between the solver and the state it locates."""
        me = self.name
        for capacity in self.capacities:
            contents = [
                _var(me, f"{capacity.name}_content_{entry.name}")
                for entry in capacity.flows
            ]
            if name in (capacity.name, f"{capacity.name}_content"):
                return _sum(contents)
            if name == f"{capacity.name}_fill":
                return _weighted_fill(me, capacity)
            for entry, content in zip(capacity.flows, contents):
                if name == f"{capacity.name}_content_{entry.name}":
                    return content
                if name == f"{capacity.name}_fill_{entry.name}":
                    return _flow_fill(me, capacity, entry)
        return None

    def _measures(self, name: str | None) -> bool:
        """Whether `name` designates a measurement channel of this
        component.

        Every channel a link materialises reads a **published capacity
        level**: the level and the weighted fill of the observed volume,
        plus the same pair per constituent read
        (:meth:`_MeasurementIn.channels` against
        :meth:`_Capacity.published`). Each of them is an integrated
        content or a sweep over integrated contents, so a measurement is
        continuous whichever channel it names and the predicate needs no
        case distinction."""
        return any(name in link.channels() for link in self.measurements_in)

    def _operand_reads_continuously(self, where: str, operand: _RuleOperand) -> bool:
        """Whether a guard operand reads a continuously-evolving
        quantity: a flow rate, a level this component holds, or a level
        it observes over a measurement link.

        One answer to that question, so the two things resting on it
        cannot drift apart: an equality on such a quantity is refused
        (:data:`_ORDERING_OPS`), and the mode transition carrying the
        guard is declared **watched**, so the crossing is located instead
        of noticed at whatever event happens to come next.

        A level read over a measurement link is as continuous as it is on
        the component holding it: the link carries an ODE target, and the
        reader's mirror of it is swept immediately after the level it
        mirrors at every evaluation point (see :meth:`_evaluation_order`).
        Read as neither a rate nor a level, it produced an `inst`
        transition, and the rule then switched at the next event some
        other part of the model produced rather than at the crossing."""
        _, rate = self._operand_read(where, operand)
        return (
            rate is not None
            or self._capacity_level(operand.name) is not None
            or self._measures(operand.name)
        )

    def _declared_automata(self) -> dict[str, list[str]]:
        """Every automaton this component generates, with its states: the
        gates a guard may read."""
        automata: dict[str, list[str]] = {}
        for mode in self.failure_modes:
            automata[mode.name] = ["ok", "nok"]
        for capacity in self.capacities:
            automata[f"{capacity.name}_bounds"] = ["empty", "partial", "full"]
        for flow in self.flows_out:
            if flow.tempo is not None:
                automata[f"{flow.name}_tempo"] = ["disabled", "enabled"]
            if flow.trigger is not None:
                automata[f"{flow.name}_trigger"] = ["down", "up"]
        for rule_set in self.rule_sets:
            if rule_set.has_guards:
                automata[rule_set.mode] = rule_set.states()
        return automata

    def _rule_operand_expr(self, operand: _RuleOperand) -> dict[str, Any]:
        """One guard operand as a boolean expression."""
        read, _ = self._operand_read(f"ObjFlow `{self.name}`", operand)
        if operand.op is not None:
            return {
                "op": "cmp",
                "cmp": _CMP_OPS[operand.op],
                "lhs": read,
                "rhs": _float(operand.value),
            }
        if operand.negate:
            return {"op": "bool", "bool_op": "not", "args": [read]}
        return read

    def _rule_operand_negation(self, operand: _RuleOperand) -> dict[str, Any]:
        """The operand denied, with a comparison **complemented** rather
        than wrapped in a `not`: see :data:`_NEGATED_OPS`."""
        if operand.op is not None:
            return self._rule_operand_expr(
                replace(operand, op=_NEGATED_OPS[operand.op])
            )
        return self._rule_operand_expr(replace(operand, negate=not operand.negate))

    def _rule_guard_negation(self, rule: _Rule) -> dict[str, Any]:
        """A rule's guard denied: the disjunction of its denied operands,
        by De Morgan, so every comparison in it is complemented."""
        return _bool(
            "or", [self._rule_operand_negation(operand) for operand in rule.cond]
        )

    def _rule_guard_expr(self, rule: _Rule) -> dict[str, Any]:
        """A rule's guard: the conjunction of its operands."""
        return _bool("and", [self._rule_operand_expr(operand) for operand in rule.cond])

    def _rule_reads_continuously(self, rule: _Rule) -> bool:
        """Whether a rule's guard compares a continuously-evolving
        quantity, which is what makes its selection a located crossing
        rather than a discrete decision."""
        for operand in rule.cond:
            if operand.op is None:
                continue
            if self._operand_reads_continuously(f"ObjFlow `{self.name}`", operand):
                return True
        return False

    def _rule_selection_guard(
        self, rule_set: _RuleSet, target: str
    ) -> tuple[dict[str, Any], bool]:
        """When the mode enters `target`, and whether that entry is a
        located crossing.

        Rules are ordered, so a guarded rule is selected when its own
        guard holds and no earlier one does; the default (or the "no
        rule" mode) is selected when no guard holds at all."""
        guarded = [
            (index, rule)
            for index, rule in enumerate(rule_set.rules)
            if not rule.is_default
        ]
        selected = next(
            (
                index
                for index in range(len(rule_set.rules))
                if rule_set.label(index) == target
            ),
            None,
        )
        earlier = [
            rule for index, rule in guarded if selected is None or index < selected
        ]
        terms = []
        if selected is not None and not rule_set.rules[selected].is_default:
            terms.append(self._rule_guard_expr(rule_set.rules[selected]))
            reading = [rule_set.rules[selected]] + earlier
        else:
            reading = [rule for _, rule in guarded]
            earlier = reading
        for rule in earlier:
            terms.append(self._rule_guard_negation(rule))
        return _bool("and", terms), any(
            self._rule_reads_continuously(rule) for rule in reading
        )

    def _rule_mode_automaton(self, rule_set: _RuleSet) -> dict[str, Any]:
        """The automaton selecting the active rule of a guarded set.

        Selection is an automaton rather than a branch inside the scale
        for two reasons: the coefficient maps are then **frozen within a
        mode**, which breaks the algebraic coupling between the demand
        and the production bands; and a transition carrying a continuous
        threshold is declared **watched**, so the solver stops the
        integration at the crossing instead of noticing it at the
        following step."""
        states = rule_set.states()
        default = rule_set.default_index
        transitions: list[dict[str, Any]] = []
        for source in states:
            for target in states:
                if source == target:
                    continue
                guard, located = self._rule_selection_guard(rule_set, target)
                transition: dict[str, Any] = {
                    "name": f"{rule_set.name}_{source}_to_{target}",
                    "source": source,
                    "targets": [target],
                    "guard": guard,
                }
                if located:
                    transition["distrib"] = "watched"
                else:
                    transition["distrib"] = "inst"
                    transition["probs"] = []
                transitions.append(transition)
        return {
            "name": rule_set.mode,
            "states": states,
            # The set starts in its default, or in "no rule applies"
            # when it declares none: the guards read quantities the
            # initial fixpoint resolves, so the mode that holds at t = 0
            # is entered there rather than guessed here.
            "init": _NO_RULE if default is None else rule_set.label(default),
            "transitions": transitions,
        }

    def _rule_choice(
        self, rule_set: _RuleSet, per_rule: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """`per_rule[i]` selected by the mode, falling through to the
        default rule's entry, or to zero when the set declares none.

        The branch is on the **mode state**, a discrete quantity that
        only changes at a located transition, so nothing continuous is
        hidden inside it."""
        if not rule_set.has_guards:
            return per_rule[0]
        default = rule_set.default_index
        if default is not None and all(entry == per_rule[0] for entry in per_rule):
            return per_rule[0]
        result = _float(0.0) if default is None else per_rule[default]
        for index in reversed(range(len(rule_set.rules))):
            if index == default:
                continue
            result = {
                "op": "if",
                "cond": _state_active(self.name, rule_set.mode, rule_set.label(index)),
                "then": per_rule[index],
                "otherwise": result,
            }
        return result

    def _apportionment_shares(self) -> dict[str, dict[str, float]]:
        """Per produced flow, the normalised share of its demand each
        rule set claims. Only a flow two sets produce into is contested;
        a sole producer takes the whole of it.

        A contested output every set does not declare a share of is
        **refused** here (R13): handing each set the whole of that
        output's demand makes the two of them produce twice what is
        asked and drop the surplus, which no balance records. How two
        reactions share one product is a modelling question, so it is
        asked rather than defaulted."""
        producers: dict[str, list[_RuleSet]] = {}
        for rule_set in self.rule_sets:
            for flow in rule_set.produced():
                producers.setdefault(flow, []).append(rule_set)
        shares: dict[str, dict[str, float]] = {}
        for flow, sets in producers.items():
            if len(sets) < 2:
                continue
            silent = [s.name for s in sets if flow not in s.apportionment]
            if silent:
                raise ValueError(
                    f"ObjFlow `{self.name}`: rule sets "
                    f"{', '.join(f'`{s.name}`' for s in sets)} all produce into "
                    f"`{flow}`, and {', '.join(f'`{name}`' for name in silent)} "
                    "declare no `apportionment` of it; a contested output must "
                    "say how its demand is shared"
                )
            declared = {s.name: s.apportionment[flow] for s in sets}
            total = sum(declared.values())
            shares[flow] = {name: value / total for name, value in declared.items()}
        return shares

    def _rule_capability_scale(self, rule: _Rule) -> dict[str, Any]:
        """The scale a rule could run at, set by its scarcest input: the
        minimum of what each consumed flow's producers could deliver,
        divided by that flow's coefficient.

        A coefficient of zero is the catalyst idiom: the rule names the
        flow so a guard can read it but draws none of it, so it is not
        limited by it. A rule nothing constrains, having no consumed
        flow at all, runs at its nominal scale of 1 and produces exactly
        its declared quantities."""
        terms = [
            {
                "op": "div",
                "lhs": _var(self.name, f"{flow}_capability_in"),
                "rhs": _float(coefficient),
            }
            for flow, coefficient in rule.cons.items()
            if coefficient > 0
        ]
        return _min(terms) if terms else _float(1.0)

    def _rule_demand_scale(
        self,
        rule_set: _RuleSet,
        rule: _Rule,
        connected_out: set[str],
        shares: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """The scale a rule actually runs at: its capability scale,
        bounded by what its outputs are asked for.

        The output bound is a **minimum**, not a maximum: the outputs of
        one rule are correlated by construction, so a scale above what
        any one of them can take would produce a surplus of the others
        that has nowhere to go and would be recorded by no balance.

        An output **no connection reads** does not take part: nobody is
        asking is not the same statement as somebody asking for nothing,
        and only the second is a bound. A rule left with no reading
        output at all therefore runs at its capability scale, exactly as
        a rule producing nothing does."""
        terms = [_var(self.name, f"{rule_set.name}_capability_scale")]
        for flow, coefficient in rule.prod.items():
            if coefficient <= 0 or flow not in connected_out:
                continue
            demand: dict[str, Any] = _var(self.name, f"{flow}_demand_out")
            share = shares.get(flow, {}).get(rule_set.name)
            if share is not None:
                demand = {"op": "mul", "args": [demand, _float(share)]}
            terms.append({"op": "div", "lhs": demand, "rhs": _float(coefficient)})
        return _min(terms)

    def _reject_flow_clash(self, name: str, direction: str) -> None:
        """A boolean and a continuous flow of the same name and
        direction would claim one port: refuse it here rather than let
        the engine report a duplicate port."""
        declared = (
            {f.name for f in self.flows_in} | {f.name for f in self.flows_continuous_in}
            if direction == "in"
            else {f.name for f in self.flows_out}
            | {f.name for f in self.flows_continuous_out}
        )
        if name in declared:
            raise ValueError(
                f"ObjFlow `{self.name}`: flow `{name}` is already declared "
                f"as an {direction}-flow of this component"
            )

    def add_delay_failure_mode(
        self,
        name: str,
        failure_time: float,
        repair_time: float,
        targets: list[str] | None = None,
        failure_cond: str | None = None,
        failure_effects: list[tuple[str, Any]] | None = None,
        repair_effects: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Deterministic failure/repair mode driving the out-flows'
        availability (all out-flows unless `targets` names some).
        `failure_cond` names a local variable gating the failure
        (muscadet's `cond_occ_12`).

        `failure_effects` and `repair_effects` **derate** continuous
        outputs: see :meth:`_resolve_deratings` for their shape and for
        why the return to nominal needs no declaration."""
        self.failure_modes.append(
            _FailureMode(
                name=name,
                law="delay",
                failure_param=failure_time,
                repair_param=repair_time,
                targets=list(targets or []),
                failure_cond=failure_cond,
                failure_deratings=self._resolve_deratings(
                    name, "failure_effects", failure_effects
                ),
                repair_deratings=self._resolve_deratings(
                    name, "repair_effects", repair_effects
                ),
            )
        )

    # --- deratings ------------------------------------------------------

    def _resolve_deratings(
        self, mode: str, key: str, effects: list[tuple[str, Any]] | None
    ) -> dict[str, float]:
        """The continuous outputs a mode's declared effects derate, and
        what each is left at (R6, R18).

        An effect is a ``(pattern, value)`` pair, the pattern being a
        regular expression matched **anchored** against both the flow
        name and the ``{flow}_fed_out`` alias, which is the 1.x spelling
        of an effect on an output. Anchored, and not by chance: unanchored,
        ``"H2"`` would name ``H2O`` as well and a declaration meant for
        one output would silently derate its neighbour.

        The value is what the mode LEAVES of the output, in ``[0, 1]``;
        ``False`` is the muscadet idiom for a total loss and reads as
        zero. There is no separate boolean gate on a continuous flow, so
        a rate of zero is what stops production entirely.

        A pattern reaching no continuous output is **refused** rather
        than dropped: an effect that reaches nothing builds, runs to
        completion and reports the availability figures of a plant whose
        modelled failure never happened, which is the defect class this
        refusal exists to close. A discrete out-flow is gated through
        `targets`, not here."""
        resolved: dict[str, float] = {}
        where = f"ObjFlow `{self.name}`: failure mode `{mode}`"
        for entry in effects or []:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    f"{where} declares the {key} entry {entry!r}; an effect is "
                    "a (pattern, value) pair"
                )
            pattern, value = entry
            try:
                matched = [
                    flow.name
                    for flow in self.flows_continuous_out
                    if _matches_flow(pattern, flow.name)
                    or _matches_flow(pattern, f"{flow.name}_fed_out")
                ]
            except re.error as error:
                raise ValueError(
                    f"{where} declares the {key} pattern `{pattern}`, which is "
                    f"not a regular expression: {error}"
                ) from None
            if not matched:
                declared = [flow.name for flow in self.flows_continuous_out]
                raise ValueError(
                    f"{where} declares the {key} pattern `{pattern}`, which "
                    f"names no continuous out-flow of this component (it "
                    f"declares {declared}). An effect reaching nothing is a "
                    "silent no-op; a discrete out-flow is gated through "
                    "`targets` instead"
                )
            rate = float(value)
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"{where} declares the {key} pattern `{pattern}` at "
                    f"{rate:g}; a derating is what the mode LEAVES of the "
                    "output, so it lies in [0, 1]"
                )
            for flow in matched:
                resolved[flow] = rate
        return resolved

    def _deratings_on(self, flow: str) -> list[tuple[_FailureMode, float, float]]:
        """The modes derating `flow`, with what each leaves of it while
        it stands and once repaired.

        The repair value defaults to :data:`NOMINAL_RATE`: a mode owns
        its derating, so it owns the release, and a derating has no
        per-step reset of its own. A mode declaring a repair effect keeps
        that value instead, a mode returning degraded rather than as-new
        being a legitimate model."""
        return [
            (
                mode,
                mode.failure_deratings.get(flow, NOMINAL_RATE),
                mode.repair_deratings.get(flow, NOMINAL_RATE),
            )
            for mode in self.failure_modes
            if flow in mode.failure_deratings or flow in mode.repair_deratings
        ]

    # --- what a continuous output's production is multiplied by ---------

    def _conduit_on(self, flow: str) -> _Transfer | None:
        """The conduit metering `flow`, or ``None``. A flow is metered by
        at most one, :meth:`add_transfer` refusing a second claim on it."""
        for pair in self.transfers:
            if pair.is_conduit and pair.source == flow:
                return pair
        return None

    def _produced_by_rules(self) -> set[str]:
        """The continuous outputs whose delivery is something other than
        their declared rate: what a rule set makes, and what a conduit
        meters across.

        These are exactly the outputs carrying a ``{flow}_produced_out``,
        what was actually made as opposed to what could have been, and it
        is that quantity the allocation distributes. Derived here rather
        than at each of the two places that need it, because it is a
        property of the component's own declarations and neither of them
        knows anything the other does not."""
        return {
            flow for rule_set in self.rule_sets for flow in rule_set.produced()
        } | {pair.source for pair in self.transfers if pair.is_conduit}

    def _has_transfer_delta(self, flow: str) -> bool:
        """Whether a two-stream pair adds a signed delta to `flow`."""
        return any(
            not pair.is_conduit and flow in (pair.source, pair.destination)
            for pair in self.transfers
        )

    def _apply_transfer_delta(
        self, flow: str, base: dict[str, Any], until: _Transfer | None = None
    ) -> dict[str, Any]:
        """`base` with the two-stream pairs' signed deltas on top, in
        declaration order, stopping before `until`.

        A pair subtracts what it moves from the origin's balance and adds
        it to the target's, so the component's raw total is untouched:
        that is the whole of what a pair guarantees. `until` is what makes
        several pairs on one stream compose **sequentially**: each is
        capped by the balance the earlier ones left, so two pairs cannot
        between them relieve a stream of more than it carries."""
        expr = base
        for pair in self.transfers:
            if pair is until:
                break
            if pair.is_conduit:
                continue
            moved = _var(self.name, f"{pair.name}_moved")
            if pair.source == flow:
                expr = {"op": "sub", "lhs": expr, "rhs": moved}
            if pair.destination == flow:
                expr = {"op": "add", "args": [expr, moved]}
        return expr

    def _has_production_base(self, flow: str) -> bool:
        """Whether `flow` publishes a ``{flow}_produced_base``: a stream a
        pair sits on **and** a rule set produces, whose two bases differ."""
        return self._has_transfer_delta(flow) and any(
            flow in rule_set.produced() for rule_set in self.rule_sets
        )

    def _stream_balance(self, flow: str, pair: _Transfer) -> dict[str, Any]:
        """What `flow` still carries when `pair` runs: its base, adjusted
        by the pairs declared before it. A single pair on a stream reads
        the base itself, which is the common case.

        On a stream a rule set produces, the base is the **lesser** of
        what the flow could deliver and what the rule actually made. The
        two differ as soon as a downstream demand holds the production
        below the capability, and capping on the capability alone
        creates matter: the origin loses what it has and the shortfall
        is clamped away, while the target still gains the whole moved
        quantity. Both sides must move one and the same number, so both
        read the same cap."""
        base = _var(self.name, f"{flow}_transfer_base")
        if self._has_production_base(flow):
            base = _min([base, _var(self.name, f"{flow}_produced_base")])
        return self._apply_transfer_delta(flow, base, until=pair)

    def _output_factors(self, flow: _FlowContinuousOut) -> list[dict[str, Any]]:
        """What this output's production is multiplied by, in order: the
        rate its failure modes leave of it, then its time profile.

        The two are separate channels on purpose (R6). Deratings
        compose by **minimum**, which is what makes them order-independent
        and safe on repair; a profile is not a competing degradation but
        the size of the thing being degraded, so it composes by
        **product**. Folded into one variable, a panel at 30 % irradiance
        also derated to 0.5 would read 0.5 instead of 0.15, and nothing
        would signal it."""
        factors = [
            _var(self.name, f"{flow.name}_effective_rate")
            if self._deratings_on(flow.name)
            else _var(self.name, f"{flow.name}_out_rate")
        ]
        if flow.profile is not None:
            factors.append(_var(self.name, f"{flow.name}_out_profile"))
        return factors

    def _conduit_crossing(self, conduit: _Transfer) -> dict[str, Any]:
        """What a conduit is about to move: the computed quantity, scaled
        by what its output's deratings and profile leave of it.

        Scaled, and not raw, because this is *also* what it asks its
        supplier for: a dead output produces nothing and must therefore
        draw nothing, or the difference is taken from the supplier and
        lost. A negative quantity crosses nothing, a conduit's direction
        being its connection's, so a computed reversal asks for nothing
        rather than for a magnitude it will not move."""
        flow = next(
            declared
            for declared in self.flows_continuous_out
            if declared.name == conduit.source
        )
        return {
            "op": "mul",
            "args": self._output_factors(flow)
            + [_clamped_at_zero(_var(self.name, f"{conduit.name}_requested"))],
        }

    def _effective_rate_expr(self, flow: str) -> dict[str, Any]:
        """The minimum over the shared ``{flow}_out_rate`` and the modes
        derating this output (R6).

        The two mechanisms compose rather than compete: a mode declared
        outside this layer clamps the shared variable, a mode declared
        here contributes a term of its own, and the deeper of the two is
        what the output produces at. Each mode's term reads its automaton
        **location** rather than a variable it wrote, which is what makes
        the return to nominal implicit: leaving the failing state
        restores the rate with nothing declared on the other side."""
        terms = [_var(self.name, f"{flow}_out_rate")]
        for mode, failed, repaired in self._deratings_on(flow):
            terms.append(
                {
                    "op": "if",
                    "cond": _state_active(self.name, mode.name, "nok"),
                    "then": _float(failed),
                    "otherwise": _float(repaired),
                }
            )
        return _min(terms)

    def _profile_expr(self, profile: _Profile) -> dict[str, Any]:
        """The clamped sinusoid, as an expression of simulation time."""
        angle = {
            "op": "mul",
            "args": [
                {
                    "op": "sub",
                    "lhs": {"op": "time"},
                    "rhs": _float(profile.phase_shift),
                },
                _float(2 * math.pi / profile.period),
            ],
        }
        wave: dict[str, Any] = {
            "op": "add",
            "args": [
                {
                    "op": "mul",
                    "args": [_float(profile.amplitude), {"op": "sin", "arg": angle}],
                },
                _float(profile.offset),
            ],
        }
        clamped: dict[str, Any] = {
            "op": "max",
            "args": [_float(profile.value_min), wave],
        }
        if profile.value_max is None:
            return clamped
        return {"op": "min", "args": [clamped, _float(profile.value_max)]}

    def _potential_expr(self, potential: tuple[str, Any, Any]) -> dict[str, Any]:
        """One resolved potential, as the expression reading it."""
        kind, first, second = potential
        if kind == "const":
            return _float(first)
        return _var(self.name, _level_alias(first, second))

    def add_flow_out_tempo(
        self,
        name: str,
        enable_time: float = 0.0,
        disable_time: float = 0.0,
        init_enable: bool = False,
        var_prod_default: bool = False,
        var_prod_cond: list[str] | None = None,
    ) -> None:
        """muscadet `FlowOutTempo`: the flow feeds while a
        disabled↔enabled automaton sits in `enabled`; the enable
        (resp. disable) transition is a delay of `enable_time`
        (`disable_time`) guarded on the production condition (resp. its
        negation), reset on interruption."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                tempo={
                    "enable_time": enable_time,
                    "disable_time": disable_time,
                    "init_enable": init_enable,
                },
            )
        )

    def add_flow_out_on_trigger(
        self,
        name: str,
        trigger_time_up: float = 0.0,
        trigger_time_down: float = 0.0,
        trigger_logic: str | int = "or",
        var_prod_default: bool = False,
        var_prod_cond: list[str] | None = None,
    ) -> None:
        """muscadet `FlowOutOnTrigger`: the flow feeds while a down↔up
        automaton sits in `up`, with *inhibition* logic: `up` is armed
        while the trigger aggregate (`"and"`, `"or"` or k-out-of-n over
        the `{name}_trigger_in` port) is false, `down` while it is
        true; both transitions are delays, reset on interruption."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                trigger={
                    "time_up": trigger_time_up,
                    "time_down": trigger_time_down,
                    "logic": trigger_logic,
                },
            )
        )

    def add_exp_failure_mode(
        self,
        name: str,
        failure_rate: float,
        repair_rate: float,
        targets: list[str] | None = None,
        failure_cond: str | None = None,
        failure_effects: list[tuple[str, Any]] | None = None,
        repair_effects: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Exponential failure/repair mode (statistical regime).

        `failure_effects` and `repair_effects` derate continuous outputs,
        exactly as on :meth:`add_delay_failure_mode`."""
        self.failure_modes.append(
            _FailureMode(
                name=name,
                law="exp",
                failure_param=failure_rate,
                repair_param=repair_rate,
                targets=list(targets or []),
                failure_cond=failure_cond,
                failure_deratings=self._resolve_deratings(
                    name, "failure_effects", failure_effects
                ),
                repair_deratings=self._resolve_deratings(
                    name, "repair_effects", repair_effects
                ),
            )
        )

    # --- model generation --------------------------------------------

    def _build(self, connected_out: set[str] | None = None) -> dict[str, Any]:
        """The native component this declaration generates.

        `connected_out` names the continuous out-flows at least one
        connection reads, which only the system knows: an output nobody
        reads is not a bound on the rule producing it. Left out, every
        declared out-flow counts as read.

        One helper per declared construct, appending into five shared
        lists. The call order is not presentational: those lists become
        the document's JSON arrays, so it is what a reader of the
        generated model sees, and the rule sets have to run before the
        continuous flows that fold their contributions in."""
        # Re-run rather than trusted: the declaration methods run this
        # too, but a component whose lists were filled by other means
        # would otherwise reach the generator with a flow crossing it
        # twice, and a conservation guard has to hold at the point the
        # document is written.
        self._refuse_flows_carried_twice()

        variables: list[dict] = []
        ports: list[dict] = []
        functions: list[dict] = []
        automata: list[dict] = []

        self._build_flows_in(variables, ports, functions)
        self._build_failure_modes(automata)
        self._build_flows_out(variables, ports, functions, automata)

        # Continuous flows. What is emitted here is what the component
        # knows on its own; everything that reads the connection list
        # (the per-edge channels, the allocation operator, the two
        # aggregated in-channels) is emitted by `System.build_dict`.
        equations: list[dict] = []
        rule_demand, rule_capability, rule_production = self._build_rule_sets(
            connected_out, variables, automata, equations
        )
        self._build_continuous_in(rule_demand, variables, ports, equations)
        self._build_continuous_out(
            rule_capability, rule_production, variables, ports, equations
        )
        self._build_transfers(variables, equations)
        self._build_capacities(variables, ports, equations, automata)
        self._build_measurements(variables, ports, equations)

        return {
            "name": self.name,
            "attributes": variables,
            "ports": ports,
            "interfaces": [],
            "automata": automata,
            "sensitive_functions": functions,
            "equations": equations,
        }

    def _build_flows_in(
        self, variables: list[dict], ports: list[dict], functions: list[dict]
    ) -> None:
        """The boolean in-flows: one `{flow}_fed_in` per declaration, an
        in port, and the sensitive function aggregating that port under
        the declared logic (any, all, or k of n)."""
        me = self.name
        for flow in self.flows_in:
            fed_in = f"{flow.name}_fed_in"
            variables.append(
                {"name": fed_in, "kind": "bool", "init": {"kind": "bool", "value": False}}
            )
            ports.append({"name": f"{flow.name}_in", "dir": "in"})
            agg: dict[str, Any]
            port_ref = {"component": me, "port": f"{flow.name}_in"}
            if flow.logic == "and":
                agg = {"op": "port_agg", "port": port_ref, "agg": "all"}
            elif flow.logic == "k":
                agg = {
                    "op": "cmp",
                    "cmp": "ge",
                    "lhs": {"op": "port_agg", "port": port_ref, "agg": "sum"},
                    "rhs": {"op": "const", "value": {"kind": "int", "value": flow.k}},
                }
            else:
                agg = {"op": "port_agg", "port": port_ref, "agg": "any"}
            functions.append(
                {
                    "name": f"update_{fed_in}",
                    "effects": [
                        {"target": {"component": me, "attribute": fed_in}, "value": agg}
                    ],
                }
            )

    def _build_failure_modes(self, automata: list[dict]) -> None:
        """The ok/nok automaton of each declared failure mode, its two
        transitions carrying the delay or the rate the law asks for.

        The deratings a mode declares are not written here: they are read
        off this automaton's **location** by the outputs they bear on,
        which is what makes the return to nominal implicit."""
        me = self.name
        for mode in self.failure_modes:
            failure_transition: dict[str, Any] = {
                "name": "failure",
                "source": "ok",
                "targets": ["nok"],
                "distrib": mode.law,
                ("time" if mode.law == "delay" else "rate"): mode.failure_param,
            }
            if mode.failure_cond is not None:
                failure_transition["guard"] = _var(me, mode.failure_cond)
            automata.append(
                {
                    "name": mode.name,
                    "states": ["ok", "nok"],
                    "init": "ok",
                    "transitions": [
                        failure_transition,
                        {
                            "name": "repair",
                            "source": "nok",
                            "targets": ["ok"],
                            "distrib": mode.law,
                            ("time" if mode.law == "delay" else "rate"): mode.repair_param,
                        },
                    ],
                }
            )

    def _build_flows_out(
        self,
        variables: list[dict],
        ports: list[dict],
        functions: list[dict],
        automata: list[dict],
    ) -> None:
        """The boolean out-flows: what the failure modes leave available,
        the production condition, and the tempo or trigger automaton
        gating the delivery when one is declared."""
        me = self.name
        for flow in self.flows_out:
            fed_out = f"{flow.name}_fed_out"
            # muscadet-aligned name (`FlowOut.var_fed_available_out`): real
            # studies' failure_effects target `{flow}_fed_available_out`.
            available = f"{flow.name}_fed_available_out"
            variables.append(
                {"name": fed_out, "kind": "bool", "init": {"kind": "bool", "value": False}}
            )
            variables.append(
                {"name": available, "kind": "bool", "init": {"kind": "bool", "value": True}}
            )
            ports.append({"name": f"{flow.name}_out", "dir": "out", "attr": fed_out})

            # Availability follows the failure-mode automata targeting
            # this flow (all of them must sit in `ok`).
            relevant = [
                m for m in self.failure_modes if not m.targets or flow.name in m.targets
            ]
            if relevant:
                avail_expr = _bool(
                    "and", [_state_active(me, m.name, "ok") for m in relevant]
                )
                functions.append(
                    {
                        "name": f"update_{available}",
                        "effects": [
                            {
                                "target": {"component": me, "attribute": available},
                                "value": avail_expr,
                            }
                        ],
                    }
                )

            # Production condition. `var_prod_cond` is either a flat list
            # (one AND group: the historical form) or a DNF list-of-lists
            # (outer-OR of inner-AND groups: the platform-export
            # `prod_cond` form). A referenced flow resolves to this
            # component's `_fed_in` (in-flow) or `_fed_out` (out-flow:
            # the diagnostic-mirror pattern; the fixpoint handles the
            # intra-component dependency without a topological sort).
            in_names = {f.name for f in self.flows_in}
            out_names = {f.name for f in self.flows_out}

            def prod_ref(cond: str) -> dict:
                if cond in in_names:
                    return _var(me, f"{cond}_fed_in")
                if cond in out_names:
                    return _var(me, f"{cond}_fed_out")
                raise ValueError(
                    f"ObjFlow `{me}`: production condition of "
                    f"`{flow.name}` references unknown flow `{cond}` "
                    "(neither an in-flow nor an out-flow of this component)"
                )

            if flow.var_prod_cond:
                groups = (
                    flow.var_prod_cond
                    if all(isinstance(g, list) for g in flow.var_prod_cond)
                    else [flow.var_prod_cond]
                )
                prod_expr: dict[str, Any] = _bool(
                    "or",
                    [
                        _bool("and", [prod_ref(cond) for cond in group])
                        for group in groups
                    ],
                )
            else:
                prod_expr = {
                    "op": "const",
                    "value": {"kind": "bool", "value": bool(flow.var_prod_default)},
                }

            if flow.tempo is not None:
                # FlowOutTempo: fed while `enabled`; the production
                # condition only gates the (delayed, reset) enable and
                # disable transitions: a lost condition keeps feeding
                # until the disable delay elapses.
                aut = f"{flow.name}_tempo"
                automata.append(
                    {
                        "name": aut,
                        "states": ["disabled", "enabled"],
                        "init": "enabled" if flow.tempo["init_enable"] else "disabled",
                        "transitions": [
                            {
                                "name": f"{flow.name}_enable",
                                "source": "disabled",
                                "targets": ["enabled"],
                                "guard": prod_expr,
                                "distrib": "delay",
                                "time": float(flow.tempo["enable_time"]),
                            },
                            {
                                "name": f"{flow.name}_disable",
                                "source": "enabled",
                                "targets": ["disabled"],
                                "guard": {
                                    "op": "bool",
                                    "bool_op": "not",
                                    "args": [prod_expr],
                                },
                                "distrib": "delay",
                                "time": float(flow.tempo["disable_time"]),
                            },
                        ],
                    }
                )
                gate_terms = [_state_active(me, aut, "enabled")]
            elif flow.trigger is not None:
                # FlowOutOnTrigger: inhibition logic, `up` arms while
                # the trigger aggregate is false; fed while `up` AND
                # the production condition holds.
                aut = f"{flow.name}_trigger"
                port_name = f"{flow.name}_trigger_in"
                ports.append({"name": port_name, "dir": "in"})
                port_ref = {"component": me, "port": port_name}
                logic = flow.trigger["logic"]
                if logic == "and":
                    trigger_agg: dict[str, Any] = {
                        "op": "port_agg",
                        "port": port_ref,
                        "agg": "all",
                    }
                elif logic == "or":
                    trigger_agg = {"op": "port_agg", "port": port_ref, "agg": "any"}
                elif isinstance(logic, int):
                    trigger_agg = {
                        "op": "cmp",
                        "cmp": "ge",
                        "lhs": {"op": "port_agg", "port": port_ref, "agg": "sum"},
                        "rhs": {"op": "const", "value": {"kind": "int", "value": logic}},
                    }
                else:
                    raise ValueError(
                        "trigger logic must be 'and', 'or', or a positive integer"
                    )
                automata.append(
                    {
                        "name": aut,
                        "states": ["down", "up"],
                        "init": "down",
                        "transitions": [
                            {
                                "name": f"{flow.name}_trigger_up",
                                "source": "down",
                                "targets": ["up"],
                                "guard": {
                                    "op": "bool",
                                    "bool_op": "not",
                                    "args": [trigger_agg],
                                },
                                "distrib": "delay",
                                "time": float(flow.trigger["time_up"]),
                            },
                            {
                                "name": f"{flow.name}_trigger_down",
                                "source": "up",
                                "targets": ["down"],
                                "guard": trigger_agg,
                                "distrib": "delay",
                                "time": float(flow.trigger["time_down"]),
                            },
                        ],
                    }
                )
                gate_terms = [_state_active(me, aut, "up"), prod_expr]
            else:
                gate_terms = [prod_expr]

            fed_expr = {
                "op": "bool",
                "bool_op": "and",
                "args": gate_terms + [_var(me, available)],
            }
            functions.append(
                {
                    "name": f"update_{fed_out}",
                    "effects": [
                        {
                            "target": {"component": me, "attribute": fed_out},
                            "value": fed_expr,
                        }
                    ],
                }
            )

    def _build_rule_sets(
        self,
        connected_out: set[str] | None,
        variables: list[dict],
        automata: list[dict],
        equations: list[dict],
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
    ]:
        """The two scales of each rule set, its mode automaton when it is
        guarded, and the per-flow contributions the flows themselves fold
        in afterwards.

        Those contributions are the return value, keyed by flow: what
        each set demands of an input it consumes, and what it makes an
        output capable of and what it actually produced there. They are
        handed back rather than written into the flow declarations, so a
        rule set and a capacity on one flow compose whichever was
        declared first: the flow folds the contributions in, and the
        capacity bounds the result."""
        me = self.name
        reading = (
            {flow.name for flow in self.flows_continuous_out}
            if connected_out is None
            else set(connected_out)
        )
        shares = self._apportionment_shares()
        rule_demand: dict[str, list[dict[str, Any]]] = {}
        rule_capability: dict[str, list[dict[str, Any]]] = {}
        rule_production: dict[str, list[dict[str, Any]]] = {}
        for rule_set in self.rule_sets:
            capability_scale = f"{rule_set.name}_capability_scale"
            scale = f"{rule_set.name}_scale"
            variables.append(_float_attribute(capability_scale, 0.0))
            variables.append(_float_attribute(scale, 0.0))
            if rule_set.has_guards:
                automata.append(self._rule_mode_automaton(rule_set))
            equations.append(
                {
                    "target": capability_scale,
                    "kind": "explicit",
                    "expr": self._rule_choice(
                        rule_set,
                        [self._rule_capability_scale(rule) for rule in rule_set.rules],
                    ),
                }
            )
            equations.append(
                {
                    "target": scale,
                    "kind": "explicit",
                    "expr": self._rule_choice(
                        rule_set,
                        [
                            self._rule_demand_scale(rule_set, rule, reading, shares)
                            for rule in rule_set.rules
                        ],
                    ),
                }
            )
            for flow in rule_set.consumed():
                coefficient = self._rule_choice(
                    rule_set, [_float(rule.cons.get(flow, 0.0)) for rule in rule_set.rules]
                )
                rule_demand.setdefault(flow, []).append(
                    {"op": "mul", "args": [coefficient, _var(me, scale)]}
                )
            for flow in rule_set.produced():
                coefficient = self._rule_choice(
                    rule_set, [_float(rule.prod.get(flow, 0.0)) for rule in rule_set.rules]
                )
                rule_capability.setdefault(flow, []).append(
                    {"op": "mul", "args": [coefficient, _var(me, capability_scale)]}
                )
                rule_production.setdefault(flow, []).append(
                    {"op": "mul", "args": [coefficient, _var(me, scale)]}
                )

        return rule_demand, rule_capability, rule_production

    def _build_continuous_in(
        self,
        rule_demand: dict[str, list[dict[str, Any]]],
        variables: list[dict],
        ports: list[dict],
        equations: list[dict],
    ) -> None:
        """The continuous in-flows: their three channels, their in port,
        and the demand this component carries upstream on each.

        The demand has one source of the four, in this order of
        precedence: a conduit metering the flow, the rule sets consuming
        it, a declared `demand_expr`, and failing all three the declared
        constant. Whichever it is, the volume holding the flow bounds
        it."""
        for continuous_in in self.flows_continuous_in:
            flow_name = continuous_in.name
            variables.append(
                _float_attribute(f"{flow_name}_fed_in", continuous_in.var_in_default)
            )
            variables.append(
                _float_attribute(
                    f"{flow_name}_demand_in", continuous_in.var_demand_in_default
                )
            )
            variables.append(
                _float_attribute(
                    f"{flow_name}_capability_in", continuous_in.var_in_default
                )
            )
            ports.append({"name": f"{flow_name}_in", "dir": "in"})
            conduit = self._conduit_on(flow_name)
            if conduit is not None:
                # A conduit asks for what it is about to move: it
                # replaced its flow's transit, so nothing else claims
                # that input and the demand carried upstream is the
                # computed quantity itself.
                base = self._conduit_crossing(conduit)
            elif flow_name in rule_demand:
                base = _sum(rule_demand[flow_name])
            elif continuous_in.demand_expr is not None:
                base = continuous_in.demand_expr
            else:
                base = _float(continuous_in.var_demand_in_default)
            equations.append(
                {
                    "target": f"{flow_name}_demand_in",
                    "kind": "explicit",
                    "expr": self._capacity_bounded_demand(flow_name, base),
                }
            )

    def _build_continuous_out(
        self,
        rule_capability: dict[str, list[dict[str, Any]]],
        rule_production: dict[str, list[dict[str, Any]]],
        variables: list[dict],
        ports: list[dict],
        equations: list[dict],
    ) -> None:
        """The continuous out-flows: the rate their deratings leave, the
        profile scaling them, their three channels, the out port carrying
        the per-connection channels, and what they could deliver.

        What could be delivered and what was actually made are two
        equations, not one: they differ as soon as a rule has several
        outputs, and it is the second that the allocation distributes."""
        me = self.name
        for continuous_out in self.flows_continuous_out:
            flow_name = continuous_out.name
            # muscadet's shared `{flow}_out_rate`: the derating endpoint,
            # created with the flow and left at its nominal 1 by this
            # layer, so a mode that derates the flow has one place to
            # write and nothing to fight over.
            variables.append(_float_attribute(f"{flow_name}_out_rate", NOMINAL_RATE))
            if self._deratings_on(flow_name):
                # What the failure modes bearing on this output leave of
                # it, folded by minimum with the shared rate above.
                variables.append(
                    _float_attribute(f"{flow_name}_effective_rate", NOMINAL_RATE)
                )
                equations.append(
                    {
                        "target": f"{flow_name}_effective_rate",
                        "kind": "explicit",
                        "expr": self._effective_rate_expr(flow_name),
                    }
                )
            if continuous_out.profile is not None:
                # A read-only publication of the factor the production
                # sweep applies: writing it has no effect, the next
                # evaluation overwriting it from the curve.
                variables.append(
                    _float_attribute(
                        f"{flow_name}_out_profile",
                        continuous_out.profile.factor(0.0),
                    )
                )
                equations.append(
                    {
                        "target": f"{flow_name}_out_profile",
                        "kind": "explicit",
                        "expr": self._profile_expr(continuous_out.profile),
                    }
                )
            variables.append(_float_attribute(f"{flow_name}_capability_out", 0.0))
            variables.append(_float_attribute(f"{flow_name}_demand_out", 0.0))
            variables.append(_float_attribute(f"{flow_name}_fed_out", 0.0))
            ports.append(
                {
                    "name": f"{flow_name}_out",
                    "dir": "out",
                    "attr": f"{flow_name}_fed_out",
                    "channels": [{"name": channel} for channel in _CONTINUOUS_CHANNELS],
                }
            )
            factors = self._output_factors(continuous_out)
            conduit = self._conduit_on(flow_name)
            if conduit is not None:
                # A metered conduit REPLACES what the flow would carry:
                # the computed quantity is what crosses, bounded by what
                # the producers upstream say they could deliver. A
                # negative quantity crosses nothing, a conduit's
                # direction being its connection's.
                nominal = {
                    "op": "min",
                    "args": [
                        self._conduit_crossing(conduit),
                        _clamped_at_zero(_var(me, f"{flow_name}_capability_in")),
                    ],
                }
            elif flow_name in rule_capability:
                nominal = {
                    "op": "mul",
                    "args": factors + [_sum(rule_capability[flow_name])],
                }
            elif continuous_out.capability_expr is not None:
                nominal = continuous_out.capability_expr
            else:
                nominal = {
                    "op": "mul",
                    "args": factors + [_float(continuous_out.var_fed_default)],
                }
            if self._has_transfer_delta(flow_name):
                # The base a two-stream pair sits on, named rather than
                # inlined: the pair caps what it moves by what the origin
                # stream carries, so the two readings must be the same
                # one. A stream cannot be relieved of more than it has.
                base = f"{flow_name}_transfer_base"
                variables.append(_float_attribute(base, 0.0))
                equations.append(
                    {"target": base, "kind": "explicit", "expr": nominal}
                )
                nominal = self._apply_transfer_delta(flow_name, _var(me, base))
            equations.append(
                {
                    "target": f"{flow_name}_capability_out",
                    "kind": "explicit",
                    "expr": self._capacity_bounded_capability(flow_name, nominal),
                }
            )
            if conduit is not None:
                # What actually crossed, as opposed to what could: the
                # allocation distributes this, so a conduit whose supply
                # fell short hands on what arrived and no more.
                variables.append(_float_attribute(f"{flow_name}_produced_out", 0.0))
                equations.append(
                    {
                        "target": f"{flow_name}_produced_out",
                        "kind": "explicit",
                        "expr": self._capacity_bounded_capability(
                            flow_name,
                            {
                                "op": "min",
                                "args": [
                                    self._conduit_crossing(conduit),
                                    _clamped_at_zero(_var(me, f"{flow_name}_fed_in")),
                                ],
                            },
                        ),
                    }
                )
            elif flow_name in rule_production:
                # What the rule actually makes, as opposed to what it
                # could make if asked without bound. The two differ as
                # soon as a rule has several outputs: the capability of
                # each is its own, where the production of all of them
                # shares the scale the least demanded one allows, which
                # is what keeps correlated outputs in proportion.
                variables.append(_float_attribute(f"{flow_name}_produced_out", 0.0))
                produced: dict[str, Any] = {
                    "op": "mul",
                    "args": factors + [_sum(rule_production[flow_name])],
                }
                if self._has_transfer_delta(flow_name):
                    # What the rule made, named rather than inlined for
                    # the reason the capability base is: a pair caps what
                    # it moves by this quantity as well, and the cap and
                    # the balance it is taken on must be the same
                    # reading. The clamp that follows is left in place
                    # against rounding; it is no longer what keeps the
                    # balance non-negative, the cap is.
                    made = f"{flow_name}_produced_base"
                    variables.append(_float_attribute(made, 0.0))
                    equations.append(
                        {"target": made, "kind": "explicit", "expr": produced}
                    )
                    produced = _clamped_at_zero(
                        self._apply_transfer_delta(flow_name, _var(me, made))
                    )
                equations.append(
                    {
                        "target": f"{flow_name}_produced_out",
                        "kind": "explicit",
                        "expr": self._capacity_bounded_capability(flow_name, produced),
                    }
                )

    def _build_transfers(self, variables: list[dict], equations: list[dict]) -> None:
        """Transfer pairs: what the gradient asks for, and what the
        component was able to move.

        Both are published, so a saturated transfer reads as a shortfall
        rather than as a plausible number nothing distinguishes from a
        satisfied one."""
        me = self.name
        for pair in self.transfers:
            variables.append(_float_attribute(f"{pair.name}_requested", 0.0))
            variables.append(_float_attribute(f"{pair.name}_moved", 0.0))
            requested = _var(me, f"{pair.name}_requested")
            equations.append(
                {
                    "target": f"{pair.name}_requested",
                    "kind": "explicit",
                    "expr": {
                        "op": "mul",
                        "args": [
                            _float(pair.conductance),
                            {
                                "op": "sub",
                                "lhs": self._potential_expr(pair.potential_a),
                                "rhs": self._potential_expr(pair.potential_b),
                            },
                        ],
                    },
                }
            )
            if pair.is_conduit:
                # What crossed is what the flow delivered: a conduit
                # replaced that flow's transit, so the two are one.
                moved = _var(me, f"{pair.source}_fed_out")
            else:
                # The sign is the direction (R5): a model never writes a
                # direction clamp, this reads the sign and caps the
                # magnitude by what the origin stream carries.
                moved = {
                    "op": "if",
                    "cond": {
                        "op": "cmp",
                        "cmp": "ge",
                        "lhs": requested,
                        "rhs": _float(0.0),
                    },
                    "then": {
                        "op": "min",
                        "args": [
                            requested,
                            _clamped_at_zero(
                                self._stream_balance(pair.source, pair)
                            ),
                        ],
                    },
                    "otherwise": _negated(
                        {
                            "op": "min",
                            "args": [
                                _negated(requested),
                                _clamped_at_zero(
                                    self._stream_balance(pair.destination, pair)
                                ),
                            ],
                        }
                    ),
                }
            equations.append(
                {"target": f"{pair.name}_moved", "kind": "explicit", "expr": moved}
            )

    def _build_capacities(
        self,
        variables: list[dict],
        ports: list[dict],
        equations: list[dict],
        automata: list[dict],
    ) -> None:
        """Capacities: one integrated content per held flow, the fills and
        the two totals swept off them, the bounds automaton, and the
        read-only ports publishing the level.

        The bounds are automaton locations entered by watched transitions
        on the total weighted fill, never a branch inside the derivative:
        a branch would make the right-hand side discontinuous without the
        solver being told, and a bound would be stepped over instead of
        located."""
        me = self.name
        continuous_in_names = {f.name for f in self.flows_continuous_in}
        continuous_out_names = {f.name for f in self.flows_continuous_out}

        for capacity in self.capacities:
            contents: list[dict[str, Any]] = []
            fills: list[dict[str, Any]] = []
            for entry in capacity.flows:
                content = f"{capacity.name}_content_{entry.name}"
                fill = f"{capacity.name}_fill_{entry.name}"
                initial = capacity.content_of(entry.name)
                variables.append(_float_attribute(content, initial))
                variables.append(
                    _float_attribute(fill, initial * entry.weight / capacity.volume)
                )
                contents.append(_var(me, content))
                fills.append(_var(me, fill))

                # What arrives minus what leaves, over the U7 channels.
                arriving = (
                    _var(me, f"{entry.name}_fed_in")
                    if entry.name in continuous_in_names
                    else _float(0.0)
                )
                leaving = (
                    _var(me, f"{entry.name}_fed_out")
                    if entry.name in continuous_out_names
                    else _float(0.0)
                )
                equations.append(
                    {
                        "target": content,
                        "kind": "ode",
                        "expr": {"op": "sub", "lhs": arriving, "rhs": leaving},
                    }
                )
                equations.append(
                    {
                        "target": fill,
                        "kind": "explicit",
                        "expr": _flow_fill(me, capacity, entry),
                    }
                )

            initial_fill = capacity.initial_fill()
            variables.append(
                _float_attribute(
                    f"{capacity.name}_content",
                    sum(capacity.content_of(e.name) for e in capacity.flows),
                )
            )
            variables.append(_float_attribute(f"{capacity.name}_fill", initial_fill))
            # The two totals are swept, not integrated: summing the
            # contents is exact at every evaluation point, where a second
            # integration would drift away from its own constituents.
            equations.append(
                {
                    "target": f"{capacity.name}_content",
                    "kind": "explicit",
                    "expr": _sum(contents),
                }
            )
            equations.append(
                {
                    "target": f"{capacity.name}_fill",
                    "kind": "explicit",
                    "expr": _sum(fills),
                }
            )

            width = capacity.hysteresis
            automata.append(
                {
                    "name": f"{capacity.name}_bounds",
                    "states": ["empty", "partial", "full"],
                    "init": (
                        "empty"
                        if initial_fill <= 0.0
                        else "full"
                        if initial_fill >= 1.0
                        else "partial"
                    ),
                    # A bound is entered at the bound itself and left only
                    # once the content has moved back across the hysteresis
                    # band: without it a capacity resolved *on* its bound
                    # re-crosses it at once, and every crossing ends an
                    # integration segment.
                    #
                    # The four guards are **inclusive**, which is what a
                    # bound is: a tank whose fill reaches exactly 1 is
                    # full, not nearly full. This is worth stating
                    # because it briefly was not true. An exactly
                    # representable crossing (a tank of 100 filling at 5
                    # reaches its bound at t=20, on a step boundary) used
                    # to be missed for the rest of the run, and these
                    # guards were written strict to route around it. The
                    # engine now re-establishes the crossing scan's
                    # precondition after every accepted step, so the
                    # inclusive form locates, and the workaround is gone
                    # rather than merely unused.
                    "transitions": [
                        {
                            "name": f"{capacity.name}_leave_empty",
                            "source": "empty",
                            "targets": ["partial"],
                            "distrib": "watched",
                            "guard": _fill_threshold(me, capacity, "ge", width),
                        },
                        {
                            "name": f"{capacity.name}_reach_empty",
                            "source": "partial",
                            "targets": ["empty"],
                            "distrib": "watched",
                            "guard": _fill_threshold(me, capacity, "le", 0.0),
                        },
                        {
                            "name": f"{capacity.name}_reach_full",
                            "source": "partial",
                            "targets": ["full"],
                            "distrib": "watched",
                            "guard": _fill_threshold(me, capacity, "ge", 1.0),
                        },
                        {
                            "name": f"{capacity.name}_leave_full",
                            "source": "full",
                            "targets": ["partial"],
                            "distrib": "watched",
                            "guard": _fill_threshold(me, capacity, "le", 1.0 - width),
                        },
                    ],
                }
            )

            # The level, published read-only (R7): an out port carrying no
            # channel, so a reader exchanges no quantity and enters no
            # allocation operator.
            for alias, attribute in capacity.published():
                ports.append({"name": f"{alias}_out", "dir": "out", "attr": attribute})

    def _build_measurements(
        self, variables: list[dict], ports: list[dict], equations: list[dict]
    ) -> None:
        """The reading side of each measurement link: one variable and one
        in port per published channel, each swept off its port.

        The ports carry no channel, so the link exchanges no quantity and
        the reader enters no allocation operator."""
        me = self.name
        for measurement in self.measurements_in:
            for variable in measurement.channels():
                port = f"{variable}_in"
                variables.append(_float_attribute(variable, 0.0))
                ports.append({"name": port, "dir": "in"})
                equations.append(
                    {
                        "target": variable,
                        "kind": "explicit",
                        "expr": {
                            "op": "port_agg",
                            "port": {"component": me, "port": port},
                            "agg": "sum",
                        },
                    }
                )


class System:
    """A muscadet-style system: add components, connect flows, simulate
    through the RAICHU engine."""

    def __init__(self, name: str):
        self.name = name
        self.comp: dict[str, ObjFlow] = {}
        self._connections: list[dict] = []

    def add_component(self, cls: Type[ObjFlow], name: str) -> ObjFlow:
        """Instantiate `cls` under `name` and register it."""
        component = cls(name)
        self.comp[name] = component
        return component

    def add_declared_component(
        self, spec: dict[str, Any], classes: dict[str, type] | None = None
    ) -> ObjFlow:
        """Build a component from a declaration held in DATA and register
        it (`pyraichu.declare`).

        The counterpart of :meth:`add_component`, and deliberately beside
        it rather than folded into it: that one instantiates a class and
        takes no parameters, where a declaration carries its own sections
        and its class's initialisation parameters, and has to impose the
        order those sections are declared in.

        `spec` is muscadet's own component declaration, section for
        section: ``flows``, ``capacities``, ``measurements_in``,
        ``failure_modes``, ``rules`` and ``transfers``, plus ``cls`` and
        ``params`` naming a component class and its own declaration.
        Anything the vocabulary cannot carry is refused by name, before
        anything is built. See :func:`pyraichu.declare.check_spec` to
        validate a batch without building any of them."""
        from . import declare

        return declare.build_component(self, spec, classes=classes)

    def connect(self, source: str, flow_out: str, target: str, flow_in: str) -> None:
        """Connect `source`'s out-flow to `target`'s in-flow."""
        self._connections.append(
            {
                "from": {"component": source, "port": f"{flow_out}_out"},
                "to": {"component": target, "port": f"{flow_in}_in"},
            }
        )

    def connect_trigger(self, source: str, target: str, flow: str) -> None:
        """Connect `source`'s out-flow to `target`'s trigger in-port
        (muscadet `connect_trigger`)."""
        self._connections.append(
            {
                "from": {"component": source, "port": f"{flow}_out"},
                "to": {"component": target, "port": f"{flow}_trigger_in"},
            }
        )

    def connect_measurement(
        self,
        holder: str,
        capacity: str,
        observer: str,
        measurement: str | None = None,
    ) -> None:
        """Wire a published level to a reader (muscadet's measurement
        link, R7).

        `capacity` names the volume on `holder`, `measurement` the reading
        channel on `observer` (the capacity's own name by default, which
        is what makes the two sides line up). Both totals are wired, plus
        the constituents the reader named; a constituent the volume does
        not hold is refused here, naming what it does hold.

        The link carries **no quantity**: the ports it joins declare no
        channel, so the reader creates no demand, enters no allocation
        operator and cannot change the flow answer."""
        channel = measurement if measurement is not None else capacity
        held = self.comp.get(holder)
        reader = self.comp.get(observer)
        if held is None or reader is None:
            missing = holder if held is None else observer
            raise ValueError(
                f"System `{self.name}`: measurement link names unknown "
                f"component `{missing}`"
            )
        volume = next((c for c in held.capacities if c.name == capacity), None)
        if volume is None:
            raise ValueError(
                f"System `{self.name}`: component `{holder}` declares no "
                f"capacity `{capacity}`"
            )
        link = next((m for m in reader.measurements_in if m.name == channel), None)
        if link is None:
            raise ValueError(
                f"System `{self.name}`: component `{observer}` declares no "
                f"measurement link `{channel}`"
            )
        unknown = [flow for flow in link.flows if flow not in volume.flow_names]
        if unknown:
            raise ValueError(
                f"System `{self.name}`: measurement link `{channel}` of "
                f"`{observer}` reads {unknown}, which capacity `{capacity}` of "
                f"`{holder}` does not hold (it holds {volume.flow_names})"
            )

        pairs = [
            (_level_alias(capacity), _level_alias(channel)),
            (_fill_alias(capacity), _fill_alias(channel)),
        ]
        for flow in link.flows:
            pairs.append((_level_alias(capacity, flow), _level_alias(channel, flow)))
            pairs.append((_fill_alias(capacity, flow), _fill_alias(channel, flow)))
        for published, read in pairs:
            self.connect(holder, published, observer, read)

    def auto_connect(self, source: str, target: str) -> None:
        """Connect every same-named (out, in) flow pair: the muscadet
        convenience. Boolean and continuous flows pair within their own
        family, since a connection carries one or the other."""
        pairs = (
            (self.comp[source].flows_out, self.comp[target].flows_in),
            (
                self.comp[source].flows_continuous_out,
                self.comp[target].flows_continuous_in,
            ),
        )
        for flows_out, flows_in in pairs:
            for flow_out in flows_out:
                for flow_in in flows_in:
                    if flow_out.name == flow_in.name:
                        self.connect(source, flow_out.name, target, flow_in.name)

    # --- continuous network ------------------------------------------

    def _continuous_edges(self) -> list[_ContinuousEdge]:
        """The continuous connections, resolved against the two
        declarations they join.

        A connection joining a continuous flow to a boolean one is
        refused here, naming both ends: it would otherwise reach the
        engine as a type mismatch on an aggregation."""
        edges: list[_ContinuousEdge] = []
        for connection in self._connections:
            source, destination = connection["from"], connection["to"]
            producer = self.comp.get(source["component"])
            consumer = self.comp.get(destination["component"])
            flow_out = next(
                (
                    flow.name
                    for flow in (producer.flows_continuous_out if producer else [])
                    if f"{flow.name}_out" == source["port"]
                ),
                None,
            )
            flow_in = next(
                (
                    flow.name
                    for flow in (consumer.flows_continuous_in if consumer else [])
                    if f"{flow.name}_in" == destination["port"]
                ),
                None,
            )
            if flow_out is None and flow_in is None:
                continue
            if flow_out is None or flow_in is None:
                stranger = source if flow_out is None else destination
                if stranger["component"] not in self.comp:
                    raise ValueError(
                        f"System `{self.name}`: connection "
                        f"{source['component']}.{source['port']} -> "
                        f"{destination['component']}.{destination['port']} "
                        f"carries a continuous flow into `{stranger['component']}`, "
                        "which this system did not build. The continuous "
                        "network is resolved over the components it declares, "
                        "so a quantity crossing into another one is accounted "
                        "for nowhere"
                    )
                raise ValueError(
                    f"System `{self.name}`: connection "
                    f"{source['component']}.{source['port']} -> "
                    f"{destination['component']}.{destination['port']} joins a "
                    "continuous flow to a boolean one; a connection carries "
                    "one or the other"
                )
            edges.append(
                _ContinuousEdge(
                    name=_edge_name(connection),
                    producer=source["component"],
                    flow_out=flow_out,
                    consumer=destination["component"],
                    flow_in=flow_in,
                )
            )
        return edges

    def _allocation_params(
        self,
        edges: list[_ContinuousEdge],
        declared: dict[Any, float],
        keyword: str,
    ) -> list[dict[str, Any]]:
        """A ``{consumer: value}`` declaration as the schema's
        ConsumerParam list, keyed by destination port.

        Keying by destination rather than by position is what keeps a
        share attached to its consumer when a connection is inserted or
        reordered, so the authoring key is the consumer, never an
        index."""
        producer, flow = edges[0].producer, edges[0].flow_out
        params: list[dict[str, Any]] = []
        matched: set[Any] = set()
        for edge in edges:
            key = next(
                (
                    candidate
                    for candidate in ((edge.consumer, edge.flow_in), edge.consumer)
                    if candidate in declared
                ),
                None,
            )
            if key is None:
                raise ValueError(
                    f"System `{self.name}`: continuous out-flow `{flow}` of "
                    f"`{producer}` declares `{keyword}` without an entry for "
                    f"consumer `{edge.consumer}`; a keyed policy must cover "
                    "every connection of the flow"
                )
            matched.add(key)
            params.append(
                {
                    "to": {"component": edge.consumer, "port": f"{edge.flow_in}_in"},
                    "value": float(declared[key]),
                }
            )
        unknown = [key for key in declared if key not in matched]
        if unknown:
            raise ValueError(
                f"System `{self.name}`: continuous out-flow `{flow}` of "
                f"`{producer}` declares `{keyword}` for {unknown}, which no "
                "connection of the flow ends at"
            )
        return params

    def _emit_continuous_network(
        self, components: list[dict[str, Any]], edges: list[_ContinuousEdge]
    ) -> list[dict[str, str]] | None:
        """Add the connection-dependent material to `components` and
        return the evaluation order it needs, or ``None`` when the
        system declares no continuous flow.

        The equations emitted here are the ones a component cannot write
        on its own, because they name the connections: the per-edge
        demand and capability channels and the two totals on a producer,
        the two aggregated channels on a consumer.

        `edges` is the resolved connection graph, taken as a parameter
        rather than derived again: :meth:`build_dict` has already
        resolved it, and re-deriving it here would re-run every
        continuous/boolean compatibility refusal for a second answer that
        cannot differ, nothing between the two touching the connections
        or the flow declarations.
        """
        if not any(
            component.flows_continuous_in or component.flows_continuous_out
            for component in self.comp.values()
        ):
            return None

        by_name = {component["name"]: component for component in components}
        out_edges: dict[tuple[str, str], list[_ContinuousEdge]] = {}
        in_edges: dict[tuple[str, str], list[_ContinuousEdge]] = {}
        for edge in edges:
            out_edges.setdefault((edge.producer, edge.flow_out), []).append(edge)
            in_edges.setdefault((edge.consumer, edge.flow_in), []).append(edge)

        for name, obj in self.comp.items():
            component = by_name[name]
            produced_by_rules = obj._produced_by_rules()
            for flow in obj.flows_continuous_out:
                port = f"{flow.name}_out"
                served = out_edges.get((name, flow.name), [])
                capability = _var(name, f"{flow.name}_capability_out")

                # What remains for one consumer once the others are
                # accounted for (R11): a supply read by several
                # consumers publishes to each what is left, never its
                # whole capability, so neither sizes itself as though
                # the other were absent.
                for index, edge in enumerate(served):
                    others = [
                        _var(name, _channel_attr(port, "alloc", other.name))
                        for position, other in enumerate(served)
                        if position != index
                    ]
                    remaining = (
                        capability
                        if not others
                        else {"op": "sub", "lhs": capability, "rhs": _sum(others)}
                    )
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "capability", edge.name),
                            "kind": "explicit",
                            "expr": {"op": "max", "args": [_float(0.0), remaining]},
                        }
                    )

                # Demand back-propagation: each consumer's demand,
                # published on the edge that serves it.
                for edge in served:
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "demand", edge.name),
                            "kind": "explicit",
                            "expr": _var(edge.consumer, f"{edge.flow_in}_demand_in"),
                        }
                    )
                component["equations"].append(
                    {
                        "target": f"{flow.name}_demand_out",
                        "kind": "explicit",
                        "expr": _sum(
                            [
                                _var(name, _channel_attr(port, "demand", edge.name))
                                for edge in served
                            ]
                        ),
                    }
                )
                component["equations"].append(
                    {
                        "target": f"{flow.name}_fed_out",
                        "kind": "explicit",
                        "expr": _sum(
                            [
                                _var(name, _channel_attr(port, "alloc", edge.name))
                                for edge in served
                            ]
                        ),
                    }
                )

                # An output nobody reads distributes nothing: no
                # operator, and the two totals above stand at zero.
                if not served:
                    continue
                allocation: dict[str, Any] = {
                    "name": f"{flow.name}_alloc",
                    "port": port,
                    # What is there to distribute: the capability of a
                    # flow nothing transforms, and what the rule
                    # actually made when one does.
                    "available": (
                        _var(name, f"{flow.name}_produced_out")
                        if flow.name in produced_by_rules
                        else capability
                    ),
                    "demand": "demand",
                    "allocated": "alloc",
                    "policy": flow.allocation,
                }
                if flow.allocation == "shares":
                    allocation["shares"] = self._allocation_params(
                        served, flow.allocation_shares or {}, "allocation_shares"
                    )
                elif flow.allocation == "priority":
                    allocation["priorities"] = self._allocation_params(
                        served,
                        flow.allocation_priorities or {},
                        "allocation_priorities",
                    )
                component.setdefault("allocations", []).append(allocation)

            for flow in obj.flows_continuous_in:
                port_ref = {"component": name, "port": f"{flow.name}_in"}
                connected = bool(in_edges.get((name, flow.name)))
                for channel, target in (
                    ("capability", f"{flow.name}_capability_in"),
                    ("alloc", f"{flow.name}_fed_in"),
                ):
                    component["equations"].append(
                        {
                            "target": target,
                            "kind": "explicit",
                            # Unconnected, the input supplies its
                            # declared constant and nothing else.
                            "expr": (
                                {
                                    "op": "port_agg",
                                    "port": port_ref,
                                    "agg": "sum",
                                    "channel": channel,
                                }
                                if connected
                                else _float(flow.var_in_default)
                            ),
                        }
                    )

        return self._evaluation_order(components, out_edges, in_edges)

    def _flow_order(
        self, edges: dict[tuple[str, str], list[_ContinuousEdge]]
    ) -> list[str]:
        """The components in flow-graph topological order (producers
        before their consumers), ties broken by declaration order.

        A conservative flow network is cyclic by nature, so a cycle is
        expected and legal: the nodes it holds back are released in
        declaration order rather than refused.

        Kahn's algorithm over two heaps of **declaration indices**, which
        is what makes both tie-breaks one statement rather than two: take
        the lowest index whose producers are all placed, and when a cycle
        leaves none of them, the lowest index still unplaced. Lowest
        index *is* declared first, so neither tie-break has to be
        expressed a second time. Placed nodes are left in the heaps and
        skipped on the way out, which costs one comparison rather than a
        removal."""
        names = list(self.comp)
        position = {name: index for index, name in enumerate(names)}
        successors: list[list[int]] = [[] for _ in names]
        indegree = [0] * len(names)
        for group in edges.values():
            for edge in group:
                if edge.producer == edge.consumer:
                    continue
                successors[position[edge.producer]].append(position[edge.consumer])
                indegree[position[edge.consumer]] += 1

        ready = [index for index in range(len(names)) if indegree[index] == 0]
        heapq.heapify(ready)
        # Every node, in declaration order, which is already a valid
        # heap: the source the cycle fallback draws from.
        unplaced = list(range(len(names)))
        placed = [False] * len(names)
        order: list[str] = []
        while len(order) < len(names):
            while ready and placed[ready[0]]:
                heapq.heappop(ready)
            if ready:
                chosen = heapq.heappop(ready)
            else:
                while placed[unplaced[0]]:
                    heapq.heappop(unplaced)
                chosen = heapq.heappop(unplaced)
            placed[chosen] = True
            order.append(names[chosen])
            # A node released by a cycle break still frees its
            # successors, exactly as one taken in order does.
            for successor in successors[chosen]:
                indegree[successor] -= 1
                if indegree[successor] == 0 and not placed[successor]:
                    heapq.heappush(ready, successor)
        return order

    def _evaluation_order(
        self,
        components: list[dict[str, Any]],
        out_edges: dict[tuple[str, str], list[_ContinuousEdge]],
        in_edges: dict[tuple[str, str], list[_ContinuousEdge]],
    ) -> list[dict[str, str]]:
        """The three-band sweep order: capability along the flow, demand
        back against it, production along it again.

        The engine sweeps the explicit equations once per evaluation
        point, so this order *is* part of the answer: swept the other
        way round, a consumer would size its demand on the capability
        the previous evaluation point left behind."""
        order: list[dict[str, str]] = []
        listed: set[tuple[str, str]] = set()

        def step(component: str, attribute: str) -> None:
            if (component, attribute) not in listed:
                listed.add((component, attribute))
                order.append({"component": component, "attribute": attribute})

        topological = self._flow_order(out_edges)

        def unconnected_inputs(suffix: str) -> None:
            """An input no producer feeds appears in no producer's band,
            and its constant still has to be swept."""
            for name, obj in self.comp.items():
                for flow in obj.flows_continuous_in:
                    if not in_edges.get((name, flow.name)):
                        step(name, f"{flow.name}_{suffix}")

        # 0. Held levels, before anything reads them. A capacity's fills
        # and its two totals read the integrated contents only, so they
        # are swept first, and an observer reads what a capacity has just
        # published.
        for name, obj in self.comp.items():
            for capacity in obj.capacities:
                for entry in capacity.flows:
                    step(name, f"{capacity.name}_fill_{entry.name}")
                step(name, f"{capacity.name}_fill")
                step(name, f"{capacity.name}_content")
        for name, obj in self.comp.items():
            for measurement in obj.measurements_in:
                for variable in measurement.channels():
                    step(name, variable)
        # What multiplies a production, and what a gradient asks for:
        # neither reads the flow network, both are read by it. The
        # transfers come after the measurements, whose readings are the
        # potentials a conductive law is written over.
        for name, obj in self.comp.items():
            for flow in obj.flows_continuous_out:
                if obj._deratings_on(flow.name):
                    step(name, f"{flow.name}_effective_rate")
                if flow.profile is not None:
                    step(name, f"{flow.name}_out_profile")
            for pair in obj.transfers:
                step(name, f"{pair.name}_requested")

        # 1. Capability, along the flow. An input no producer feeds
        # supplies a constant, and a rule set may read it: swept first,
        # so a scale never sizes itself on a constant the same pass has
        # not reached.
        unconnected_inputs("capability_in")
        for producer in topological:
            # A rule set turns the capability of what it consumes into
            # the capability of what it produces, so its scale is swept
            # between the two.
            for rule_set in self.comp[producer].rule_sets:
                step(producer, f"{rule_set.name}_capability_scale")
            # A two-stream pair caps what it moves by what its origin
            # carries, so both bases are swept before the pair, and the
            # pair before the capabilities it adjusts.
            for flow in self.comp[producer].flows_continuous_out:
                if self.comp[producer]._has_transfer_delta(flow.name):
                    step(producer, f"{flow.name}_transfer_base")
            for pair in self.comp[producer].transfers:
                if not pair.is_conduit:
                    step(producer, f"{pair.name}_moved")
            for flow in self.comp[producer].flows_continuous_out:
                port = f"{flow.name}_out"
                step(producer, f"{flow.name}_capability_out")
                served = out_edges.get((producer, flow.name), [])
                for edge in served:
                    step(producer, _channel_attr(port, "capability", edge.name))
                for edge in served:
                    step(edge.consumer, f"{edge.flow_in}_capability_in")

        # 2. Demand, back against the flow.
        for producer in reversed(topological):
            for flow in self.comp[producer].flows_continuous_out:
                port = f"{flow.name}_out"
                served = out_edges.get((producer, flow.name), [])
                for edge in served:
                    step(edge.consumer, f"{edge.flow_in}_demand_in")
                for edge in served:
                    step(producer, _channel_attr(port, "demand", edge.name))
                step(producer, f"{flow.name}_demand_out")
            # The rule's own scale: after its outputs have been asked,
            # and before its inputs ask, which a producer's own visit
            # (later in this reversed walk) is what steps.
            for rule_set in self.comp[producer].rule_sets:
                step(producer, f"{rule_set.name}_scale")
        unconnected_inputs("demand_in")

        # 3. Production, along the flow again.
        for producer in topological:
            produced_by_rules = self.comp[producer]._produced_by_rules()
            for flow in self.comp[producer].flows_continuous_out:
                served = out_edges.get((producer, flow.name), [])
                if flow.name in produced_by_rules:
                    step(producer, f"{flow.name}_produced_out")
                if served:
                    step(producer, f"{flow.name}_alloc")
                step(producer, f"{flow.name}_fed_out")
                for edge in served:
                    step(edge.consumer, f"{edge.flow_in}_fed_in")
            # What a conduit moved is what its flow delivered, so it is
            # read once that delivery is settled.
            for pair in self.comp[producer].transfers:
                if pair.is_conduit:
                    step(producer, f"{pair.name}_moved")
        unconnected_inputs("fed_in")

        # The order must cover the declared steps exactly, and the
        # engine refuses an omission rather than completing it: close it
        # over what the components actually declare, so a step this
        # layer gains later cannot fall out of the sweep silently.
        for component in components:
            for equation in component["equations"]:
                if equation["kind"] == "explicit":
                    step(component["name"], equation["target"])
            for allocation in component.get("allocations", []):
                step(component["name"], allocation["name"])
        return order

    # --- rule diagnostics ---------------------------------------------

    def _validate_rules(self, edges: list[_ContinuousEdge]) -> None:
        """The three refusals the rule vocabulary makes sayable, in
        declaration order: a contested output nobody apportions, a loop
        of rate comparisons, and a self-feeding cycle that creates
        matter."""
        if not any(obj.rule_sets for obj in self.comp.values()):
            return
        for obj in self.comp.values():
            obj._apportionment_shares()
        self._refuse_rate_comparison_loops(edges)
        self._refuse_unbounded_rule_cycles(edges)

    def _rule_graph(
        self, edges: list[_ContinuousEdge]
    ) -> dict[tuple[str, str, str], list[tuple[tuple[str, str, str], float, Any]]]:
        """The graph matter travels on: one node per flow endpoint
        ``(component, flow, side)``, one edge per connection (carrying
        the quantity unchanged) and one per ``(consumed, produced)`` pair
        of a rule (carrying the ratio of their coefficients).

        Every rule of every set contributes, not only the active one: a
        guard is a run-time fact, and a diagnostic that only held for the
        rule that happens to be selected would be no diagnostic."""
        successors: dict[
            tuple[str, str, str], list[tuple[tuple[str, str, str], float, Any]]
        ] = {}
        for edge in edges:
            successors.setdefault((edge.producer, edge.flow_out, "out"), []).append(
                ((edge.consumer, edge.flow_in, "in"), 1.0, None)
            )
        for name, obj in self.comp.items():
            for rule_set in obj.rule_sets:
                for index, rule in enumerate(rule_set.rules):
                    for consumed, taken in rule.cons.items():
                        if taken <= 0:
                            continue
                        for produced, made in rule.prod.items():
                            if made <= 0:
                                continue
                            successors.setdefault(
                                (name, consumed, "in"), []
                            ).append(
                                (
                                    (name, produced, "out"),
                                    made / taken,
                                    (name, rule_set.name, index),
                                )
                            )
        return successors

    def _elementary_cycles(self, successors):
        """The elementary cycles of the rule graph, as lists of
        ``(source, target, gain, rule)`` steps.

        A depth-first walk from each node over the nodes that follow it
        in declaration order, which enumerates every cycle exactly once.
        The walk is bounded: the diagnostic is worth a large but finite
        search, never an unbounded one on a pathological graph. Spending
        that budget raises :class:`_CycleSearchExhausted` instead of
        ending the walk, since the budget is shared across every start
        node and a graph large enough to exhaust it would otherwise
        yield nothing and read exactly like a graph with no cycle."""
        # First-seen order, which is declaration order here and is what
        # `rank` below turns into the "follows me" test: `dict.fromkeys`
        # keeps it while deduplicating in one pass rather than scanning
        # the list already built at every node.
        seen: list[tuple[str, str, str]] = []
        for source, arcs in successors.items():
            seen.append(source)
            seen.extend(target for target, _, _ in arcs)
        nodes = list(dict.fromkeys(seen))
        rank = {node: index for index, node in enumerate(nodes)}
        budget = [_CYCLE_SEARCH_BUDGET]

        def walk(start, node, path, on_path):
            for target, gain, rule in successors.get(node, []):
                if budget[0] <= 0:
                    raise _CycleSearchExhausted(len(nodes), _CYCLE_SEARCH_BUDGET)
                budget[0] -= 1
                step = (node, target, gain, rule)
                if target == start:
                    yield path + [step]
                elif rank[target] > rank[start] and target not in on_path:
                    on_path.add(target)
                    yield from walk(start, target, path + [step], on_path)
                    on_path.discard(target)

        for start in nodes:
            yield from walk(start, start, [], {start})

    def _refuse_rate_comparison_loops(self, edges: list[_ContinuousEdge]) -> None:
        """Refuse a loop of rate comparisons (R15).

        A guard comparing a **rate** is instantaneous: it reads a
        quantity the very sweep it takes part in produces. When two rule
        sets each guard on a rate the other drives, the selection has no
        fixpoint: each mode is the answer to the other, and the pair
        chatters or settles on whichever was evaluated first. A guard on
        an integrated **level** closes no such loop, because the
        integration carries the value across the sweep, which is why the
        two kinds of comparison are told apart rather than counted
        together."""
        carried: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
        for edge in edges:
            carried.setdefault((edge.producer, edge.flow_out, "out"), []).append(
                (edge.consumer, edge.flow_in, "in")
            )
        influence: dict[tuple[str, str], set] = {}
        reads: dict[tuple[str, str], list] = {}
        for name, obj in self.comp.items():
            for rule_set in obj.rule_sets:
                key = (name, rule_set.name)
                # What this set drives *before another rule set does*:
                # its own outputs and the inputs they feed. A rate that
                # travels further does so through another set, which is
                # then the next node of the loop rather than a hop
                # inside this one.
                driven = {(name, flow, "out") for flow in rule_set.produced()}
                for seed in list(driven):
                    driven.update(carried.get(seed, ()))
                influence[key] = driven
                compared = []
                for rule in rule_set.rules:
                    for operand in rule.cond:
                        if operand.op is None:
                            continue
                        _, rate = obj._operand_read(f"ObjFlow `{name}`", operand)
                        if rate is not None:
                            compared.append((name, rate[0], rate[1]))
                reads[key] = compared

        arcs: dict[tuple[str, str], list[tuple[tuple[str, str], tuple]]] = {}
        for driver, driven_nodes in influence.items():
            for reader, endpoints in reads.items():
                shared = [node for node in endpoints if node in driven_nodes]
                if shared:
                    arcs.setdefault(driver, []).append((reader, shared[0]))

        cycle = _find_cycle(list(influence), arcs)
        if cycle is None:
            return
        rendered = " -> ".join(
            f"rule set `{node[1]}` of `{node[0]}` (driving the rate "
            f"`{endpoint[1]}`)"
            for node, endpoint in cycle
        )
        raise ValueError(
            f"System `{self.name}`: the rule guards of {rendered} form a loop "
            "of rate comparisons: each set drives a rate the next one is "
            "selected on, so the selection has no fixpoint. Compare an integrated "
            "level instead, which the integration carries across the sweep."
        )

    def _refuse_unbounded_rule_cycles(self, edges: list[_ContinuousEdge]) -> None:
        """Refuse a rule cycle that creates matter from nothing (R25).

        A cycle whose coefficient product exceeds 1 makes more of a thing
        than it consumed, and if all of its inputs originate inside it
        nothing bounds what it makes. **The same cycle fed by a finite
        external input is bounded by that input and is a model**, not an
        error: it is refused only when it is closed on itself. A capacity
        anywhere on the cycle breaks it too, the integration carrying the
        quantity across instead of resolving it instantaneously.

        **What this verdict does not settle.** A rule cycle is also a
        cycle of *capability* equations, and the engine refuses those on
        its own account: a rule's output capability is derived from its
        inputs', so a cycle of rules closes a chain of explicit
        equations whatever its coefficients. A cycle this diagnostic
        accepts therefore still meets that refusal, and today a capacity
        on the cycle does not lift it either, since a buffered input is
        read through the upstream capability rather than through what
        the volume can serve. Making the volume replace the flow it
        buffers is what would close that gap, and it is a decision about
        what "available" means, not a detail of this diagnostic."""
        feeding: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
        for edge in edges:
            feeding.setdefault((edge.consumer, edge.flow_in, "in"), set()).add(
                (edge.producer, edge.flow_out, "out")
            )
        try:
            for cycle in self._elementary_cycles(self._rule_graph(edges)):
                applied = [step[3] for step in cycle if step[3] is not None]
                if not applied:
                    continue
                gain = 1.0
                for step in cycle:
                    gain *= step[2]
                if gain <= 1.0 + 1e-12:
                    continue
                on_cycle = {step[0] for step in cycle}
                if any(
                    self.comp[component]._capacity_of(flow)
                    for component, flow, _ in on_cycle
                ):
                    continue
                if self._cycle_is_fed_externally(applied, on_cycle, feeding):
                    continue
                path = " -> ".join(
                    f"`{component}`.`{flow}`" for component, flow, _ in
                    [step[0] for step in cycle] + [cycle[0][0]]
                )
                owners: list[tuple[str, str]] = []
                for component, rule_set, _ in applied:
                    if (component, rule_set) not in owners:
                        owners.append((component, rule_set))
                named = ", ".join(
                    f"rule set `{rule_set}` of `{component}`"
                    for component, rule_set in owners
                )
                raise ValueError(
                    f"System `{self.name}`: {named} form the cycle {path}, whose "
                    f"coefficient product is {gain:g} and whose inputs all "
                    "originate inside it: it would create matter from nothing. "
                    "Feed the cycle from outside it, or break it with a capacity."
                )
        except _CycleSearchExhausted as exhausted:
            raise ValueError(
                f"System `{self.name}`: the rule-cycle search stopped after "
                f"expanding its budget of {exhausted.budget} arcs over "
                f"{exhausted.nodes} flow endpoints, before it had enumerated "
                "every cycle. The refusal of a cycle that creates matter from "
                "nothing therefore cannot be given, and the model is refused "
                "rather than built on a guard that stopped guarding. Break the "
                "rule graph into fewer interconnected flows, or interpose a "
                "capacity, which breaks any cycle it sits on."
            ) from None

    def _cycle_is_fed_externally(self, applied, on_cycle, feeding) -> bool:
        """Whether anything outside the cycle feeds it, which is what
        bounds what it can make.

        Two ways in: a rule on the cycle consumes a flow the cycle does
        not carry (connected or standing at its declared default, either
        way a finite quantity), or a flow the cycle does carry is also
        fed by a producer that is not on it."""
        for component, set_name, index in applied:
            obj = self.comp[component]
            rule_set = next(s for s in obj.rule_sets if s.name == set_name)
            for flow, taken in rule_set.rules[index].cons.items():
                if taken > 0 and (component, flow, "in") not in on_cycle:
                    return True
        for node in on_cycle:
            if node[2] != "in":
                continue
            if any(producer not in on_cycle for producer in feeding.get(node, ())):
                return True
        return False

    def generate(
        self, foreign: list[dict[str, Any]] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]] | None]:
        """Build every registered component and add the material only the
        connection list can supply, answering the components and the
        evaluation order they need (``None`` when nothing is continuous).

        `foreign` names components of the SAME model this system did not
        build: none when :meth:`build_dict` writes the whole document, and
        the rest of the model when the serialized plugin path
        (`pyraichu.plugins.muscadet`) drives this system over a document
        that also carries controllers, failure-mode objects or hand-written
        components. They receive no continuous material, since they declare
        no flow this layer resolves, but the evaluation order **closes over
        them**: the order must cover the declared steps exactly, so a step
        some other object emitted has to be swept or the whole model is
        refused.

        This is the one continuous generation. :meth:`build_dict` and the
        plugin reach the per-edge equations, the R11 netting, the allocation
        operators and the three-band order through it, so the two authoring
        surfaces cannot answer one model differently."""
        edges = self._continuous_edges()
        self._validate_rules(edges)
        reading: dict[str, set[str]] = {}
        for edge in edges:
            reading.setdefault(edge.producer, set()).add(edge.flow_out)
        components = [
            obj._build(reading.get(name, set())) for name, obj in self.comp.items()
        ]
        order = self._emit_continuous_network(components + list(foreign or []), edges)
        return components, order

    def indicators(self, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One indicator per observable variable of `components`, named
        `comp_var` as muscadet does.

        `components` are the ones THIS system built (:meth:`generate`): a
        component it did not build declares no flow it can read, and naming
        one here would observe it by the shape of its variable names."""
        # A capacity's own variables end in a held flow's name, not in a
        # channel suffix, so they are named rather than matched: a level
        # generated and never observable would be of no use.
        held: dict[str, set[str]] = {}
        for name, obj in self.comp.items():
            observable: set[str] = set()
            for rule_set in obj.rule_sets:
                observable.add(f"{rule_set.name}_scale")
                observable.add(f"{rule_set.name}_capability_scale")
            for capacity in obj.capacities:
                observable.add(f"{capacity.name}_content")
                observable.add(f"{capacity.name}_fill")
                for entry in capacity.flows:
                    observable.add(f"{capacity.name}_content_{entry.name}")
                    observable.add(f"{capacity.name}_fill_{entry.name}")
            for measurement in obj.measurements_in:
                observable.update(measurement.channels())
            for flow in obj.flows_continuous_out:
                if obj._deratings_on(flow.name):
                    observable.add(f"{flow.name}_effective_rate")
                if flow.profile is not None:
                    observable.add(f"{flow.name}_out_profile")
                if obj._has_transfer_delta(flow.name):
                    observable.add(f"{flow.name}_transfer_base")
                if obj._has_production_base(flow.name):
                    observable.add(f"{flow.name}_produced_base")
            for pair in obj.transfers:
                observable.add(f"{pair.name}_requested")
                observable.add(f"{pair.name}_moved")
            held[name] = observable
        indicators: list[dict[str, Any]] = []
        for component in components:
            for variable in component["attributes"]:
                # The boolean suffixes plus the continuous channels a
                # flow resolves over: generated and never observed
                # otherwise.
                if variable["name"].endswith(
                    (
                        "_fed_in",
                        "_fed_out",
                        "_demand_in",
                        "_demand_out",
                        "_capability_in",
                        "_capability_out",
                        "_produced_out",
                    )
                ) or variable["name"] in held.get(component["name"], ()):
                    indicators.append(
                        {
                            "name": f"{component['name']}_{variable['name']}",
                            "target": "attribute",
                            "attr": {
                                "component": component["name"],
                                "attribute": variable["name"],
                            },
                        }
                    )
        return indicators

    def build_dict(self) -> dict[str, Any]:
        """Generate the native RAICHU model as a plain dict, with one
        indicator per flow variable (muscadet naming: `comp_var`),
        also the fixture-generation entry point.

        A system carrying continuous flows needs the evaluation order
        and the allocation operators, so its document is **sealed** in
        the format envelope; a purely boolean system uses baseline
        constructs only and keeps the bare body it has always had."""
        components, evaluation_order = self.generate()
        body: dict[str, Any] = {
            "name": self.name,
            "components": components,
            "connections": self._connections,
            "indicators": self.indicators(components),
        }
        if evaluation_order is None:
            return body
        body["evaluation_order"] = evaluation_order
        # The feature list is derived from the body by the engine, never
        # composed here: it cannot lag what the body holds.
        return seal(body)

    def build_model(self) -> Model:
        """Generate and validate the native RAICHU model."""
        return load_model(json.dumps(self.build_dict()))

    def simulate(self, t_max: float, **kwargs: Any) -> SimulationResult:
        """One trajectory through the RAICHU engine."""
        return simulate(self.build_model(), t_max=t_max, **kwargs)

    def monte_carlo(
        self, nb_runs: int, t_max: float, samples: list[float], **kwargs: Any
    ) -> McEstimates:
        """Monte-Carlo estimation through the RAICHU driver."""
        return monte_carlo(
            self.build_model(), nb_runs=nb_runs, t_max=t_max, samples=samples, **kwargs
        )
