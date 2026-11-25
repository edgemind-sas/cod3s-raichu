"""The records a component declaration is parsed into, and the naming helpers of rules and capacities."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ._expr import _float, _sum, _var
from ._vocabulary import (
    _NO_RULE,
    DEFAULT_HYSTERESIS,
    DRAWN_PROFILE,
    PROD_COND_INNER_MODE_DEFAULT,
    RUN_GATE_STATES,
    SERVE_COND_INNER_MODE_DEFAULT,
    SERVE_GATE_STATES,
    SINE_PROFILE,
)


@dataclass
class _FlowIn:
    name: str
    logic: str = "or"  # "or" | "and" | int k (k-out-of-n)
    k: int | None = None
    # muscadet `FlowIn.var_in_default`: what the input reads while nothing
    # is connected to it (`True` marks an always-fed boundary input).
    # `None` = the muscadet default (`False`). Honoured only as long as no
    # producer publishes on the flow: `_build` writes the emptiness into
    # the aggregating expression, so the declaration decides out of
    # connection and the aggregate decides once a wire is there.
    var_in_default: bool | None = None


@dataclass
class _FlowOut:
    name: str
    var_prod_default: bool = False
    # Flat list = one AND group; list-of-lists = DNF (outer-OR of
    # inner-AND groups, the platform-export `prod_cond` form). An operand
    # is a bare flow name or the mapping muscadet writes back, carrying
    # `PROD_COND_OPERAND_KEYS`: `port` to select a side the default
    # resolution would not, `negate` to deny the operand, `op`/`value` to
    # compare what it names against a threshold.
    var_prod_cond: list[Any] | list[list[Any]] = field(default_factory=list)
    # muscadet's `var_prod_cond_inner_mode`: which level of the list above
    # is the disjunction. See :data:`PROD_COND_INNER_MODES`.
    var_prod_cond_inner_mode: str = PROD_COND_INNER_MODE_DEFAULT
    # muscadet's `FlowOut.negate`: the output publishes the NEGATION of what
    # it would deliver, `not (prod and is_active and available)`.
    negate: bool = False
    # muscadet `FlowOutTempo`: {"enable", "disable", "init_enable"}: a
    # disabled↔enabled automaton whose two transitions, each on its own
    # law (a delay or an exponential, in the engine's transition form),
    # are gated on the production condition (reset on interruption); the
    # flow is fed while `enabled`.
    tempo: dict[str, Any] | None = None
    # muscadet `FlowOutOnTrigger`: {"time_up", "time_down", "logic"},
    # a down↔up automaton on a dedicated trigger in-port with
    # *inhibition* logic (up while the trigger inputs are absent); the
    # flow is fed while `up` AND the production condition holds.
    trigger: dict[str, Any] | None = None
    # muscadet `FlowOut.var_is_active_default`: service-function dormancy
    # through a `var_is_active` factor the caller's effects drive. `None`
    # = no such factor: the generated model carries no `var_is_active`
    # and stays byte-identical to what it has always been.
    var_is_active_default: bool | None = None
    # muscadet `FlowOut.var_fed_available_out_init`: the availability
    # gate's initial value. `None` = the muscadet default (available).
    var_fed_available_out_init: bool | None = None
    # muscadet `FlowOut.var_fed_available_out_reset`: whether the gate is
    # reinitialized to that seed at every step. `None` and `True` are
    # muscadet's default and this engine's own semantics -- the gate is
    # re-derived at every fixpoint pass. `False` asks for a gate that
    # MEMORISES, which this engine has no per-variable reset to switch
    # off: it is honoured where nothing writes the gate, which is every
    # port of a library that declares the control and never touches it,
    # and refused where a failure mode derives it (see `_build_flows_out`).
    var_fed_available_out_reset: bool | None = None


@dataclass
class _FlowContinuousIn:
    """A real-valued input: what it asks for, and what it reads when
    nothing is connected to it."""

    name: str
    var_demand_in_default: float = 0.0
    var_in_default: float = 0.0
    #: Which quantity this flow publishes as an observable rate channel,
    #: or ``None`` for none. Declared and not inferred: a rate is a
    #: quantity of the flow network, and publishing every one of them
    #: would put a port on every flow of every model for the few an
    #: observer reads.
    publish_rate: str | None = None
    #: Replaces the constant demand with a declared expression, for a
    #: caller that has one to hand. It is an escape hatch onto the
    #: generated equation and nothing in this module assigns it: a rule
    #: set consuming the flow derives the demand instead, and a capacity
    #: holding the flow bounds whatever came out, both of them while the
    #: model is generated. It is read last of the three, so a conduit or
    #: a rule set on the same flow takes precedence over it.
    demand_expr: dict[str, Any] | None = None
    #: What a mixture group asks on this input, ``R . m_f / S`` read off the
    #: volume the group draws from: set by
    #: :meth:`System._resolve_mixture_groups` when the model is generated,
    #: and kept apart from `demand_expr`, which stays the caller's.
    mixture_demand: dict[str, Any] | None = None
    #: A declared time profile scaling what this input asks for, or
    #: ``None``. The counterpart of the output's, and the same shape: a
    #: demand that varies with the clock is as ordinary as a production
    #: that does, and the asymmetry was an accident of which side was
    #: built first.
    profile: _Profile | None = None


@dataclass
class _Profile:
    """A declared time profile scaling a flow: a base curve of simulation
    time, plus an optional smoothed noise, clamped into
    ``[value_min, value_max]``.

    The base is one of three shapes, named by `shape` (muscadet's class
    name):

    - the sinusoid, ``amplitude x sin(2 pi (t - phase_shift) / period) +
      offset``;
    - the square wave, ``low`` rising over ``ramp_duration`` to ``high``,
      held ``high_duration``, falling back over ``ramp_duration`` and
      ``low`` for the rest of the period, the rise starting at
      ``phase_shift``;
    - the drawn curve, ``points`` joined by segments over one period, the
      last point joined to the first one period later, read at
      ``t - phase_shift``.

    The square wave is drawn as the four points of :meth:`drawn_points`,
    so both periodic shapes reach the engine as one piecewise-linear node.
    Every shape is continuous by construction, the clamp meeting the curve
    where it crosses the bound, which is why a shape carries no
    attestation of its own: the attestation is what a bare callable cannot
    give (see :data:`_CONTINUITY_MESSAGE`).

    The noise (``noise_amplitude > 0``) adds nodes every
    ``noise_time_scale`` carrying a standard normal drawn once per replica
    for the flow key ``noise_key``, smoothly interpolated, and null at
    ``t = 0``: the factor at the initial instant is the noise-free one.

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
    #: The label muscadet carries on a sinusoid, defaulting to the shape's
    #: own class name exactly as `SinusoidalProfile.__init__` does.
    #: Decoration, like a component's `label`: it reaches no generated
    #: model, and it is kept so a declaration survives the round trip
    #: rather than losing what a modeller wrote on the curve. What it
    #: buys on the way is the refusals below, which name the curve at
    #: fault instead of only the flow carrying it.
    name: str = SINE_PROFILE
    #: Which of :data:`PROFILE_CLASSES` the base curve is.
    shape: str = SINE_PROFILE
    #: The square wave's levels and durations.
    low: float = 0.0
    high: float = 0.0
    high_duration: float = 0.0
    ramp_duration: float = 0.0
    #: The drawn curve's ``[instant, factor]`` points over one period.
    points: list[list[float]] | None = None
    noise_amplitude: float = 0.0
    noise_time_scale: float | None = None
    noise_key: int | None = None

    @property
    def has_noise(self) -> bool:
        return self.noise_amplitude > 0.0

    def drawn_points(self) -> list[list[float]]:
        """The periodic piecewise-linear points of a square wave or a drawn
        curve: the declared points, or for a square wave ``[0, low]``,
        ``[ramp, high]``, ``[ramp + plateau, high]``,
        ``[2 ramp + plateau, low]``, low after that.

        A point that would repeat the instant of the one before it (no
        plateau) or fall on the period itself (the fall ending with the
        period, where the curve rejoins its first point by itself) is
        left out: the instants of a periodic shape are strictly increasing
        and below the period."""
        if self.shape == DRAWN_PROFILE:
            return [list(point) for point in self.points or []]
        ramp = self.ramp_duration
        candidates = [
            [0.0, self.low],
            [ramp, self.high],
            [ramp + self.high_duration, self.high],
            [2 * ramp + self.high_duration, self.low],
        ]
        points: list[list[float]] = []
        for point in candidates:
            if point[0] >= self.period or (points and point[0] <= points[-1][0]):
                continue
            points.append(point)
        return points

    def base(self, time: float) -> float:
        """The base curve at `time`, before the noise and the clamp."""
        if self.shape == SINE_PROFILE:
            angle = (time - self.phase_shift) * (2 * math.pi / self.period)
            return self.amplitude * math.sin(angle) + self.offset
        points = self.drawn_points()
        u = (time - self.phase_shift) % self.period
        index = max(bisect.bisect_right([p[0] for p in points], u), 1) - 1
        x0, y0 = points[index]
        x1, y1 = points[index + 1] if index + 1 < len(points) else (self.period, points[0][1])
        return y0 + (y1 - y0) * (u - x0) / (x1 - x0)

    def factor(self, time: float) -> float:
        """The noise-free factor at `time`, for the initial value of the
        published variable: the noise is null at the initial instant, the
        sweep overwrites the value, and an init that already agrees with
        the curve keeps a document readable on its own."""
        value = max(self.value_min, self.base(time))
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
    #: or ``None``. It composes with the failure-mode rates by
    #: **product**, never by minimum: an output at 0.3 of its profile
    #: that is also capped at 0.5 produces 0.15, not 0.3.
    profile: _Profile | None = None
    #: A ceiling on what this output can deliver per unit time, or
    #: ``None``. It bounds the quantity WHEREVER it comes from: a rule
    #: set producing the flow, a declared rate, a conduit metering it. On
    #: a rule-produced output it also bounds the scale the rule runs at,
    #: which is what makes the demand carried upstream the demand for
    #: what will actually be made rather than for what was asked.
    #:
    #: Distinct from a cap declared by a failure mode, which is a
    #: FRACTION of nominal and belongs to the mode that declares it: this
    #: is an absolute quantity and a property of the equipment. The two
    #: compose by minimum, the binding one winning, as two ceilings do.
    max_rate: float | None = None
    #: Which quantity this flow publishes as an observable rate channel,
    #: or ``None`` for none. See :class:`_FlowContinuousIn`.
    publish_rate: str | None = None


@dataclass
class _RuleOperand:
    """One operand of a rule guard.

    Either a read of a flow or of a declared attribute (`name`, with
    `port` disambiguating a name carried on both sides), or an
    automaton-state gate (`automaton` + `state`). `op` and `value` turn
    the read into a numeric comparison; `negate` inverts a boolean one.

    `release` turns that comparison into a **band**: `value` is the
    threshold at which the rule is entered, `release` the one at which it
    is left, and between the two the mode holds whatever it already was.
    A single threshold makes the rule switch at the exact point where the
    read crosses it, which is right for a quantity nothing in the model
    pushes back on, and wrong for one the rule itself moves: there the
    two states each produce the condition that justifies the other and
    the mode has no fixpoint. The band is the standard cure, and the cost
    of crossing it is what gives the cycle a physical period instead of a
    numerical one.
    """

    name: str | None = None
    port: str | None = None
    automaton: str | None = None
    state: str | None = None
    negate: bool = False
    op: str | None = None
    value: float | None = None
    release: float | None = None

    def label(self) -> str:
        """The operand as it reads in a diagnostic."""
        if self.automaton is not None:
            return f"{self.automaton} in {self.state}"
        if self.op is not None:
            band = (
                f" (until {float(self.release):g})"
                if self.release is not None
                else ""
            )
            return f"{self.name} {self.op} {float(self.value):g}{band}"
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
        content = _var(component, _content_attribute(capacity.name, entry.name))
        terms.append(
            content
            if entry.weight == 1.0
            else {"op": "mul", "args": [content, _float(entry.weight)]}
        )
    return {"op": "div", "lhs": _sum(terms), "rhs": _float(capacity.volume)}


def _fill_claim(
    capacity: _Capacity, asked: dict[str, Any], unbounded: dict[str, Any]
) -> dict[str, Any]:
    """What a volume with room asks for: `asked`, plus its `fill_rate`.

    An unbounded fill rate is "whatever there is", which a document
    cannot carry as an infinity: `unbounded` stands for it, and `asked`
    still gets through when it exceeds it. A fill rate of zero claims
    nothing beyond `asked`."""
    if capacity.fill_rate == float("inf"):
        return {"op": "max", "args": [asked, unbounded]}
    if capacity.fill_rate:
        return {"op": "add", "args": [asked, _float(capacity.fill_rate)]}
    return asked


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
                _var(component, _content_attribute(capacity.name, entry.name)),
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


def _rate_alias(channel: str) -> str:
    """The name a flow publishes the rate it carries under.

    One name for both directions on purpose: what an observer wants is
    the quantity crossing the wire, and which side of it the publisher
    sits on is the publisher's business."""
    return f"{channel}_rate"


def _derating_variable(mode: str, flow: str) -> str:
    """The variable a standalone mode clamps to derate `flow`: muscadet's
    own name for it (``muscadet.derating``), so the effect the mode
    declares names the same variable on both sides."""
    return f"{mode}_derating_{flow}"


def _ratio_alias(channel: str, flow: str) -> str:
    """The name a measurement channel publishes a constituent's ratio
    under. See :func:`_level_alias`.

    A ratio has no total: the ratios of a volume sum to one, so a
    channel-wide ratio would be the constant 1 and say nothing.""" 
    return f"{channel}_ratio_{flow}"


def _content_attribute(capacity: str, flow: str | None = None) -> str:
    """The attribute a capacity holds its content in: the raw quantity of
    the whole volume, or of one constituent of it.

    Written down in one place because it is the one capacity name the
    reference layer does NOT share: muscadet holds the same quantity in
    ``{c}_qty`` and ``{c}_qty_{f}`` (``muscadet.capacity``). An indicator
    is named against the variable muscadet created, so a reader
    translating one reaches this function
    (:func:`pyraichu.muscadet.declare.capacity_content_variables`) rather than
    restating the spelling and drifting the day it changes."""
    return f"{capacity}_content" if flow is None else f"{capacity}_content_{flow}"


def _publishes_ratios(flows: Any) -> bool:
    """Whether a volume holding these constituents publishes a ratio per
    constituent: only one holding MORE THAN ONE.

    A module-level rule and not only a property of the declaration,
    because the reader that translates an observation has to answer the
    same question off a document (:func:`pyraichu.muscadet.declare.
    capacity_absent_variables`): muscadet publishes ``{c}_ratio_{f}`` on
    every volume, so on a single-constituent one the name exists there and
    not here, and what says so is exactly this rule."""
    return len(flows) > 1


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
    #: Whether the volume passes on what it does not hold back: whether
    #: the **through-path exists at all**, which ``side`` cannot say. A
    #: buffer declares both ports and transits; a reservoir has no way in
    #: and an accumulator no way out, so neither has a through-path
    #: whatever it holds, and both come out of an exported document with
    #: the same ``side``. Not a third rate beside :attr:`fill_rate` and
    #: :attr:`serve_rate`: those two say HOW MUCH, one claimed and one
    #: capped, where this one says only WHETHER.
    #:
    #: **False by default, where muscadet's own field defaults to True**,
    #: and the difference is deliberate. muscadet's solver never reads
    #: this field: it takes the through-path from the wiring, so its
    #: default moves no model of its own. Here the field DECIDES, so a
    #: default of True would silently turn every volume ever written
    #: against this engine into a buffer. Measured on the cross-validation
    #: corpus, that moved five documents nothing compares, 172 series of
    #: the plant among them. A volume transits when it says so.
    transmits: bool = False
    #: The ceiling on what may leave the volume, whatever it holds.
    #: ``math.inf`` -- the default, and muscadet's -- is a volume that
    #: serves whatever is asked of it while it is stocked, which is what
    #: makes the volume ITSELF the source of what the flow delivers.
    #:
    #: Published as a **variable** per held flow rather than folded into
    #: the capability as a constant, under the name muscadet gives it
    #: (:meth:`ceiling_of`), so a failure mode has something to clamp.
    serve_rate: float = float("inf")
    #: The discharge command, as groups of guard operands: when it does
    #: not hold, the volume delivers nothing, on the branch that transits
    #: as on the branch that serves from stock. Empty -- the default --
    #: is an uncommanded volume, which is every volume that says nothing.
    serve_cond: list[list[_RuleOperand]] = field(default_factory=list)
    #: How the two levels of :attr:`serve_cond` combine, exactly as
    #: muscadet's ``serve_cond_inner_mode``: ``"or"`` -- muscadet's
    #: default -- is a conjunction of disjunctions, ``"and"`` the other
    #: way round.
    serve_cond_inner_mode: str = SERVE_COND_INNER_MODE_DEFAULT
    hysteresis: float = DEFAULT_HYSTERESIS
    #: The held constituent the atmosphere supplies when the volume is OPEN
    #: (muscadet's R53), or ``None`` for a closed volume. An open volume
    #: stays full: the make-up enters at the rate it runs short and the
    #: excess vents at the composition, so no arrival is refused.
    makeup_flow: str | None = None
    #: Whether a mixture group draws this volume (muscadet's R51): set by
    #: :meth:`System._resolve_mixture_groups` when the model is generated,
    #: never declared. A volume drawn as a mixture serves each held flow
    #: the composed request it was asked, what merely transits entering
    #: the composition first.
    serves_mixture: bool = False

    @property
    def flow_names(self) -> list[str]:
        return [entry.name for entry in self.flows]

    @property
    def commanded(self) -> bool:
        """Whether a discharge command gates what this volume releases."""
        return bool(self.serve_cond)

    @property
    def is_open(self) -> bool:
        """Whether the volume is open to the atmosphere (R53)."""
        return self.makeup_flow is not None

    # The command protocol, shared with `_MixtureIn`: one compiler for the
    # two places the operand vocabulary commands something (R49, R52).

    @property
    def command(self) -> list[list[_RuleOperand]]:
        return self.serve_cond

    @property
    def command_inner_mode(self) -> str:
        return self.serve_cond_inner_mode

    @property
    def gate_states(self) -> tuple[str, str]:
        return SERVE_GATE_STATES

    @property
    def command_owner(self) -> str:
        return f"capacity `{self.name}`"

    #: What the command decides, in the refusals that name it.
    command_noun = "discharge"

    @property
    def gate(self) -> str:
        """Name of the automaton carrying the discharge command.

        muscadet tags the same states ``{capacity}_serve``, and for the
        same reason: flows and capacities are two namespaces where the
        engine's states are one, so ``add_capacity(name=X, flow=X)`` --
        the most natural spelling there is -- would otherwise collide
        with the flow's own gate.
        """
        return f"{self.name}_serve"

    def ceiling_of(self, flow: str) -> str:
        """Name of the variable holding this volume's ceiling on `flow`.

        muscadet declares the ceiling **per held flow**
        (``Capacity.add_variables``), which is what makes a multi-flow
        volume's total the sum of its constituents' rather than one
        figure shared between them. The spelling is muscadet's, because
        the point of publishing it is that a failure mode written against
        muscadet clamps it by that name.
        """
        return f"{self.name}_serve_rate_{flow}"

    def content_of(self, flow: str) -> float:
        return float(self.content_init.get(flow, 0.0))

    def initial_fill(self) -> float:
        return sum(self.content_of(e.name) * e.weight for e in self.flows) / self.volume

    def initial_ratio(self, flow: str) -> float:
        """What fraction of the initial content one constituent is, and
        zero for a volume that starts empty: the same convention the
        swept equation carries."""
        total = sum(self.content_of(e.name) for e in self.flows)
        return self.content_of(flow) / total if total > 0.0 else 0.0

    def published(self) -> list[tuple[str, str]]:
        """The ``(port alias, exported attribute)`` pairs the capacity
        publishes: the two totals, then each constituent.

        The totals alone would not do. A volume holding water and heat is
        at a temperature of ``heat / water``, while the total level is
        their weighted sum, which is neither term; an observer wanting
        the ratio needs both, so both are published.

        A volume holding **more than one** constituent publishes each
        one's ratio as well, what fraction of the whole it is. That is a
        third quantity and not a convenience: the observer's output
        grammar carries no arithmetic, so a controller thresholding a
        fraction can only read one that is already published. A volume
        holding a single constituent publishes none, its ratio being
        identically one wherever it holds anything."""
        pairs = [
            (_level_alias(self.name), _content_attribute(self.name)),
            (_fill_alias(self.name), f"{self.name}_fill"),
        ]
        for entry in self.flows:
            pairs.append(
                (
                    _level_alias(self.name, entry.name),
                    _content_attribute(self.name, entry.name),
                )
            )
            pairs.append(
                (_fill_alias(self.name, entry.name), f"{self.name}_fill_{entry.name}")
            )
            if self.publishes_ratios:
                pairs.append(
                    (
                        _ratio_alias(self.name, entry.name),
                        f"{self.name}_ratio_{entry.name}",
                    )
                )
        return pairs

    @property
    def publishes_ratios(self) -> bool:
        """Whether this volume publishes a ratio per constituent: only a
        volume holding more than one. The rule itself is
        :func:`_publishes_ratios`, which a reader answering the same
        question off a DOCUMENT reaches too."""
        return _publishes_ratios(self.flows)


@dataclass
class _MeasurementIn:
    """The reading side of a measurement link: a read-only view of a
    published level, carrying no quantity."""

    name: str
    #: Constituents of the observed volume to read individually, beside
    #: the totals.
    flows: list[str] = field(default_factory=list)

    #: Whether the observed volume publishes a ratio per constituent,
    #: which only a volume holding more than one does. Declared on the
    #: reader because the reader is built before the pairing is known,
    #: and set from the publisher when the two are matched.
    ratios: bool = False

    def channels(self) -> list[str]:
        """The variables (and, suffixed with ``_in``, the ports) this
        reader materialises, in the order the publisher lists them."""
        names = [_level_alias(self.name), _fill_alias(self.name)]
        for flow in self.flows:
            names.append(_level_alias(self.name, flow))
            names.append(_fill_alias(self.name, flow))
            if self.ratios:
                names.append(_ratio_alias(self.name, flow))
        return names


@dataclass
class _MeasurementOut:
    """The publishing side of a measurement link on a component that is
    not a capacity: an instrument (muscadet's ``add_measurement_out``, R37).

    It publishes under the very aliases a capacity uses, so an observer
    cannot tell a republisher from a volume, and multiplies everything it
    publishes by ``{name}_level_gain``, the endpoint a failure mode clamps
    to make one instrument lie. The source is a capacity or a measurement
    channel of the same component; ``None`` leaves the published level a
    plain variable nothing refreshes."""

    name: str
    source: str | None = None
    #: Constituents republished beside the totals, each with its level,
    #: its fill and its share.
    flows: list[str] = field(default_factory=list)
    level_default: float = 0.0
    fill_default: float = 0.0
    ratio_default: float = 0.0
    gain_default: float = 1.0

    @property
    def gain(self) -> str:
        return f"{self.name}_level_gain"

    @property
    def flow_names(self) -> list[str]:
        return list(self.flows)

    @property
    def publishes_ratios(self) -> bool:
        """An instrument publishes a share for every constituent it names,
        as muscadet's does: it carries what the volume behind it would."""
        return bool(self.flows)

    def published(self) -> list[tuple[str, str]]:
        """The ``(port alias, exported attribute)`` pairs, in the order a
        capacity lists its own: the totals, then each constituent. The
        attribute IS the alias: the instrument holds nothing else."""
        aliases = [_level_alias(self.name), _fill_alias(self.name)]
        for flow in self.flows:
            aliases += [
                _level_alias(self.name, flow),
                _fill_alias(self.name, flow),
                _ratio_alias(self.name, flow),
            ]
        return [(alias, alias) for alias in aliases]


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
class _Effects:
    """What one mode's declared effects do to the continuous outputs
    they name.

    Two maps and not one, because a **cap** and a **tap** are opposite
    conventions over the same interval: a cap says what the output is
    left at, a tap says what fraction is taken off it. They also compose
    differently (a cap by minimum, a tap by sum), so folding them
    together would need a conversion at every read and would silently
    pick one convention for a number declared in the other."""

    #: What each named output is left at, in [0, 1]. Neutral:
    #: :data:`NOMINAL_RATE`.
    caps: dict[str, float] = field(default_factory=dict)
    #: What fraction is taken off each named output, in [0, 1]. Neutral:
    #: :data:`NO_TAP`.
    taps: dict[str, float] = field(default_factory=dict)
    #: Where a tap delivers what it takes, keyed by the tapped out-flow.
    routes: dict[str, str] = field(default_factory=dict)


@dataclass
class _FailureMode:
    name: str
    law: str  # "delay" | "exp"
    failure_param: float
    repair_param: float
    targets: list[str] = field(default_factory=list)  # affected out-flows
    failure_cond: str | None = None  # local variable gating the failure
    #: What this mode does to the continuous outputs it names, while it
    #: stands and once repaired, resolved from the declared effect
    #: patterns. A flow absent from the repair side returns to the
    #: neutral element of its channel (:data:`NOMINAL_RATE` for a cap,
    #: :data:`NO_TAP` for a tap): the release is implicit, neither
    #: channel having a per-step reset of its own.
    failure: _Effects = field(default_factory=_Effects)
    repair: _Effects = field(default_factory=_Effects)


@dataclass
class _MixtureIn:
    """Continuous inputs a component draws together, at ONE volumetric rate
    (muscadet's ``add_mixture_in``, R51). What each flow then carries is
    fixed by the composition of the volume drawn from, not declared.

    The rate is published as a variable a failure mode clamps by name, and
    the machine may be commanded on and off (muscadet's R52), in the
    vocabulary and through the compiler a capacity's discharge command
    uses."""

    name: str
    flows: list[str]
    flow_rate: float
    #: The on/off command, as groups of guard operands. Empty -- the
    #: default -- runs the machine whenever its rate allows.
    run_cond: list[list[_RuleOperand]] = field(default_factory=list)
    #: How the two levels of :attr:`run_cond` combine, as muscadet's
    #: ``run_cond_inner_mode``.
    run_cond_inner_mode: str = SERVE_COND_INNER_MODE_DEFAULT

    @property
    def rate_attribute(self) -> str:
        """Name of the variable holding the group's rate: muscadet's
        ``{group}_flow_rate``, so a mode written against muscadet clamps
        it by that name."""
        return f"{self.name}_flow_rate"

    @property
    def commanded(self) -> bool:
        return bool(self.run_cond)

    @property
    def gate(self) -> str:
        """Name of the automaton carrying the command, tagged ``_run`` as
        muscadet tags the group's threshold automata, so it cannot collide
        with a flow's gate or a capacity's ``_serve``."""
        return f"{self.name}_run"

    @property
    def command(self) -> list[list[_RuleOperand]]:
        return self.run_cond

    @property
    def command_inner_mode(self) -> str:
        return self.run_cond_inner_mode

    @property
    def gate_states(self) -> tuple[str, str]:
        return RUN_GATE_STATES

    @property
    def command_owner(self) -> str:
        return f"mixture group `{self.name}`"

    command_noun = "machine"


class _MixtureSource(NamedTuple):
    """Where a mixture-fed input comes from: the supplier, its output, the
    mixed volume serving it, and whether that volume has a way in for the
    flow (a held constituent may be stored and never fed)."""

    supplier: str
    flow_out: str
    volume: str
    has_inlet: bool
