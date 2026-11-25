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
- flow out ``f``: bool ``f_fed_out`` := production condition (the
  declared ``var_prod_cond`` over the component's own flows, or the
  ``var_prod_default`` constant) AND ``f_fed_available_out`` (driven by
  failure modes);
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
consumer its own share. The capability is published whole to every
consumer, as muscadet sums it (R-20): what each receives is the
split's to decide, and what a rule makes is bounded by that receipt.

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
- **a volume that TRANSITS is what its output delivers.** Between the two
  bounds it serves whatever is asked, capped by the declared
  ``serve_rate`` and by nothing else, which is what makes the stock and
  not the flow's declared rate the source of what crosses; and it carries
  the demand of its own consumers upstream, so it smooths a shortage
  instead of swallowing what arrives, ``fill_rate`` being what it claims
  over and above that. ``transmits`` is whether the volume has a
  through-path at all, which ``side`` cannot say: a buffer and a
  reservoir come out of an exported document with the same one. It is
  **declared and never inferred, and it is off by default**, so a volume
  written against this layer is a store until it says otherwise;
- **the discharge is commandable.** ``serve_cond`` is a condition on
  which the volume releases anything at all, in the operand vocabulary a
  rule guard carries, compiled into a two-state automaton ``c_serve``. It
  gates the CEILING and therefore both bounds, so a volume standing down
  holds back what it stores and what merely crosses it; and what it asks
  upstream for the transit is capped by that same ceiling, a volume not
  filling out of a demand it cannot honour. What it claims for itself is
  not, so a tank with a ``fill_rate`` goes on charging while it stands
  down. The ceiling itself is a **variable per held flow**,
  ``c_serve_rate_f``, which is what a failure mode clamps by name;
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

Time profiles (mirroring `muscadet/profile.py`) and failure-mode effects
(mirroring `muscadet/derating.py`) are the other things multiplying a
continuous output's production, and they compose differently on purpose::

    produced = what the rule (or the declared rate) makes
               x  {flow}_out_profile
               x  {flow}_effective_rate      <- the caps, by minimum
               x  {flow}_tap_rate            <- the taps, by sum

- ``{flow}_out_profile`` is a declared **continuous** function of the
  clock, published read-only. A discontinuous one would need a watched
  transition at each of its breakpoints, which this layer derives from
  no callable, so a profile that is not one of the declarable shapes is
  refused rather than integrated across an unannounced jump;
- ``{flow}_effective_rate`` is the **minimum** over the shared
  ``{flow}_out_rate`` (the endpoint a mode declared outside this layer
  clamps) and one term per failure mode **capping** the output. A cap
  says what the output is left at, so simultaneous caps do not add up:
  the binding one wins, which is what a derated capacity means and what
  makes the fold order-independent and safe on repair;
- ``{flow}_tap_rate`` is ``1 -`` the **sum** of what each failure mode
  **taps** off the output. A tap says what fraction is taken away, and
  simultaneous taps are parallel draws on one stream: what leaves by one
  does not pass the other, so two leaks of 0.1 and 0.2 leave 0.7. Folded
  by minimum they would leave 0.8 and the second leak would be free,
  which is the arithmetic this channel exists to avoid;
- a routed tap (``{"tap": x, "to": "other"}``) delivers what it takes to
  another continuous out-flow of the component, so the fraction is moved
  rather than extinguished and the component's balance closes. Both
  sides read one published quantity, ``{flow}_pre_tap_*``, what the
  stream carried before its taps drew;
- the taps on one output are **one stage**: they all draw on that same
  quantity, which is what makes them sum. Two stages, losses taken and
  then further losses taken on what is left, are two streams and not one
  output tapped twice, and they compose by product between them. A model
  that folds both stages onto one output sums across them and gets a
  different number;
- the demand channel is not divided by the tap rate. A plant that does
  not know about its leak produces what it was asked for and delivers
  less, and the shortfall is the consumer's; making it up is regulation,
  which is declared as a controller and not inferred from a leak;
- each term reads its mode's automaton **location** rather than a
  variable it wrote, which is what makes the return to nominal implicit
  on both channels: leaving the failing state restores the contribution
  with nothing declared on the other side. There is no separate boolean
  gate on a continuous flow, so a cap of zero is what stops it entirely;
- the channels are never folded together: a panel at 30 % of its profile
  that is also capped at 0.5 produces 15 %, where one shared variable
  folding by minimum would read 30 % and signal nothing.

Only what muscadet declares as **data** translates: the conductive law
over its two potential operands (a constant and a measurement reading)
and the clamped sinusoid. A Python function in either place is refused
by name (R8), which is deliberate rather than provisional: two of the
documented reference models compute a ratio of one volume's
constituents, which is neither operand form, and they are authored
natively instead of widening this vocabulary.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
import json
import math
import operator
import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from types import SimpleNamespace
from typing import Any, Mapping, NamedTuple, Type

from .. import (
    Model,
    SimulationResult,
    McEstimates,
    UnboundedRateError,
    load_model,
    model_body,
    monte_carlo,
    seal,
    simulate,
)
from ..indicators import GENERATED_INDICATORS

from ._vocabulary import (
    _CMP_OPS,
    _CMP_PREDICATES,
    _CONTINUITY_MESSAGE,
    _CONTINUOUS_CHANNELS,
    _FNV_OFFSET_BASIS,
    _FNV_PRIME,
    _MASK64,
    _NEGATED_OPS,
    _NO_CAPACITY,
    _NO_RULE,
    _ORDERING_OPS,
    _prod_cond_inner_mode,
    _PROFILE_REQUIRED_KEYS,
    _tempo_law,
    _TEMPO_LAW_PARAMETERS,
    _TRANSFER_CLASS,
    DEFAULT_HYSTERESIS,
    DRAWN_PROFILE,
    flow_noise_key,
    MAX_PROD_COND_GROUPS,
    model_level_keys,
    NO_TAP,
    NOISE_KEYS,
    NOMINAL_RATE,
    prod_cond_disjunction,
    PROD_COND_INNER_MODE_DEFAULT,
    PROD_COND_INNER_MODES,
    PROD_COND_OPERAND_KEYS,
    prod_cond_unconditional,
    PROFILE_CLASS_KEYS,
    PROFILE_CLASSES,
    PROFILE_KEYS,
    RATE_CAPABILITY,
    RATE_DELIVERED,
    RATE_QUANTITIES,
    RULE_KEYS,
    RULE_OPERAND_KEYS,
    RUN_GATE_STATES,
    SERVE_COND_INNER_MODE_DEFAULT,
    SERVE_COND_INNER_MODES,
    SERVE_GATE_STATES,
    SINE_PROFILE,
    SQUARE_PROFILE,
    UNBOUNDED_SERVICE,
)
from ._expr import (
    _bool,
    _boolean_comparison,
    _clamped_at_zero,
    _const_bool,
    _float,
    _float_attribute,
    _matches_flow,
    _max,
    _min,
    _negated,
    _renamed_reads,
    _state_active,
    _sum,
    _var,
)
from ._records import (
    _as_coefficients,
    _as_operand_list,
    _Capacity,
    _CapacityFlow,
    _content_attribute,
    _ContinuousEdge,
    _derating_variable,
    _Effects,
    _FailureMode,
    _fill_alias,
    _fill_claim,
    _fill_threshold,
    _flow_fill,
    _FlowContinuousIn,
    _FlowContinuousOut,
    _FlowIn,
    _FlowOut,
    _level_alias,
    _MeasurementIn,
    _MeasurementOut,
    _MixtureIn,
    _MixtureSource,
    _ordered_union,
    _Profile,
    _publishes_ratios,
    _rate_alias,
    _ratio_alias,
    _Rule,
    _RuleOperand,
    _RuleSet,
    _Transfer,
    _weighted_fill,
)
from ._graph import (
    _channel_attr,
    _CycleSearchExhausted,
    _edge_name,
    _find_cycle,
    _strongly_connected,
    connected_in_flows,
)
from ._flow_profiles import _ProfilesMixin
from ._flow_measure import _MeasurementsMixin
from ._flow_discrete import _DiscreteFlowsMixin
from ._flow_modes import _FailureModesMixin
from ._flow_transfers import _TransfersMixin
from ._flow_mixture import _MixturesMixin
from ._flow_rule_guards import _RuleGuardsMixin
from ._flow_rule_scales import _RuleScalesMixin
from ._flow_capacity import _CapacitiesMixin
from ._flow_continuous import _ContinuousFlowsMixin


__all__ = ["DEFAULT_HYSTERESIS", "ObjFlow", "System"]


#: How many arcs the rule-cycle search may expand before it gives up.
#: The diagnostic is worth a large but finite walk over the rule graph,
#: never an unbounded one on a pathological topology.
_CYCLE_SEARCH_BUDGET = 200_000


def _stocked_capacity(model: Model, variable: str) -> str | None:
    """The capacity whose content `variable` (``component.attribute``) is.

    Read from the model rather than from a naming guess alone: the attribute
    must be the capacity's total content or one of its per-flow contents, and
    the component must carry that capacity's bounds automaton. ``None`` for
    anything else, so the explanation below never names a volume that is not
    there.
    """
    component_name, _, attribute = variable.partition(".")
    body = model_body(json.loads(model.json))
    for component in body.get("components", []):
        if component["name"] != component_name:
            continue
        for automaton in component.get("automata", []):
            name = automaton["name"]
            if not name.endswith("_bounds"):
                continue
            capacity = name[: -len("_bounds")]
            total = _content_attribute(capacity)
            if attribute == total or attribute.startswith(f"{total}_"):
                return capacity
    return None


def explain_unbounded_rate(error: UnboundedRateError, model: Model) -> UnboundedRateError:
    """The engine's refusal, completed in muscadet's vocabulary.

    The engine names the variable and says "bound one of the two"; a modeller
    needs the capacity and the key. When the variable is a capacity's content,
    the stock of that capacity is what drains without bound, and a finite
    ``serve_rate`` on it is what states the rate (with it, RAICHU and the
    reference engine agree). Any other variable is returned unchanged.
    """
    capacity = _stocked_capacity(model, error.variable)
    if capacity is None:
        return error
    component = error.variable.partition(".")[0]
    message = (
        f"{error}. In muscadet terms: capacity `{capacity}` of `{component}` "
        f"holds a stock with no ceiling on what leaves it (`serve_rate` left "
        f"unbounded) while what it feeds takes without bound (for instance an "
        f"infinite `fill_rate` downstream). Declare a finite `serve_rate` on capacity "
        f"`{capacity}`, the rate its stock may leave at, or bound the demand "
        f"downstream. The reference engine runs such a model at a rate the "
        f"model does not state (one rule activity per time unit, measured on "
        f"a stock beside rules), so its answer depends on the study's time "
        f"unit"
    )
    explained = UnboundedRateError(
        message, error.variable, error.time, error.rate, error.unbounded
    )
    # What the engine set on the instance travels on: a campaign's replica
    # failure carries `replica`, `nb_runs` and `seed`, the coordinates that
    # replay it.
    carried = vars(explained)
    explained.__dict__.update(
        {key: value for key, value in vars(error).items() if key not in carried}
    )
    return explained


def run_explained(run: Any, model: Model, **kwargs: Any) -> Any:
    """``run(model, **kwargs)``, with an unbounded-rate refusal explained
    by :func:`explain_unbounded_rate`. The one wrapper every muscadet entry
    point that runs a model goes through."""
    try:
        return run(model, **kwargs)
    except UnboundedRateError as error:
        raise explain_unbounded_rate(error, model) from error


class ObjFlow(
    _ProfilesMixin,
    _MeasurementsMixin,
    _DiscreteFlowsMixin,
    _FailureModesMixin,
    _TransfersMixin,
    _MixturesMixin,
    _RuleGuardsMixin,
    _RuleScalesMixin,
    _CapacitiesMixin,
    _ContinuousFlowsMixin,
):
    """A muscadet-style smart flow component. Subclass and override
    :meth:`add_flows`."""

    #: The continuous inputs a connection feeds from an output another
    #: component serves as a mixture (:meth:`_mixed_volume`), each with
    #: ``(supplier, its output, the mixed volume)``: what the cascade reads
    #: to expect how much of it will actually arrive. Only the system knows
    #: it: set by :meth:`_build`, read afterwards by the system's sweep
    #: order and release as well, and empty on a component never built.
    _mixture_sources: Mapping[str, _MixtureSource] = MappingProxyType({})

    #: The continuous inputs at least one connection feeds, set by
    #: :meth:`_build`: what the rules may draw of a wired input is what
    #: its suppliers allocated it (:meth:`_draw_budget`).
    _wired_in: frozenset[str] = frozenset()

    def __init__(self, name: str):
        self.name = name
        self._init_declarations()
        self.add_flows()

    def _init_declarations(self) -> None:
        """Empty declaration lists, in one place: the serialized entry
        point (`pyraichu.muscadet.plugin`) builds an `ObjFlow` without
        running `__init__`, and must not have to know the list of
        lists."""
        self.flows_in: list[_FlowIn] = []
        self.flows_out: list[_FlowOut] = []
        self.flows_continuous_in: list[_FlowContinuousIn] = []
        self.flows_continuous_out: list[_FlowContinuousOut] = []
        self.capacities: list[_Capacity] = []
        self.measurements_in: list[_MeasurementIn] = []
        self.measurements_out: list[_MeasurementOut] = []
        self.mixtures: list[_MixtureIn] = []
        self.rule_sets: list[_RuleSet] = []
        self.transfers: list[_Transfer] = []
        self.failure_modes: list[_FailureMode] = []
        # The (mode, flow) pairs a mode declared OUTSIDE this component
        # derates (`add_derating`). Each owns a variable here, folded by
        # minimum into the output's effective rate.
        self.external_deratings: list[tuple[str, str]] = []
        # Decoration a declaration may carry (`pyraichu.muscadet.declare`). It reaches
        # no generated model: it is held so a declaration survives a round
        # trip through a live component rather than losing what a platform
        # export knows about the instance.
        self.label: str | None = None
        self.description: str | None = None
        self.metadata: dict[str, Any] = {}

    def add_flows(self) -> None:  # pragma: no cover - overridden
        """Declare flows (override in subclasses)."""

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
        matching what entered it. The pairings are refused here rather
        than inside the declaration methods, which is what makes them
        order-independent: each declaration runs this as an early
        diagnostic through :meth:`_record`, and :meth:`_build` runs it
        again on the finished component, so a declaration list built by
        other means is caught too.

        **A volume and a rule set on one flow are NOT such a pair when
        the volume sits on one side of the rules**, as muscadet places
        it. Upstream (``side="in"``) the flow enters the volume and the
        rule's draw is the volume's only outflow; downstream
        (``side="out"``) the rule's production is the volume's only
        inflow and what the consumers take its only outflow. Either way
        the flow crosses once, the volume standing between the boundary
        and the rule, and what is refused is only what leaves that
        reading ambiguous:

        - a held flow the component **also declares on the other side**:
          the volume would be the rules' supply and the flow's own way
          out at once, and what leaves it would have two readings;
        - a volume upstream of the rules on a flow they **produce**, or
          downstream on a flow they **consume**: there is no rule on the
          far side of it to draw from it or to fill it."""
        me = f"ObjFlow `{self.name}`"
        consumed = {
            flow for rule_set in self.rule_sets for flow in rule_set.consumed()
        }
        produced = {
            flow for rule_set in self.rule_sets for flow in rule_set.produced()
        }
        carried = consumed | produced
        metered = {pair.source for pair in self.transfers if pair.is_conduit}

        # Two rule sets drawing one flow from a volume upstream of the
        # rules are each bounded by the whole of what arrives once the
        # volume is empty, so together they would draw it twice. How they
        # share it is not decided here, so the shape is refused.
        for capacity in self.capacities:
            if capacity.side != "in":
                continue
            for flow in capacity.flow_names:
                drawing = [
                    rule_set.name
                    for rule_set in self.rule_sets
                    if flow in rule_set.consumed()
                ]
                if len(drawing) > 1:
                    raise ValueError(
                        f"{me}: rule sets "
                        f"{', '.join(f'`{name}`' for name in drawing)} all draw "
                        f"`{flow}` from capacity `{capacity.name}` upstream of the "
                        "rules. Once the volume is empty each would be allowed "
                        "the whole of what arrives, so together they would draw "
                        "it twice; draw the flow from one rule set, or hold it "
                        "in one volume per set"
                    )
        declared_in = {flow.name for flow in self.flows_continuous_in}
        declared_out = {flow.name for flow in self.flows_continuous_out}

        for capacity in self.capacities:
            for flow in sorted(set(capacity.flow_names) & carried):
                if flow in declared_in and flow in declared_out:
                    raise ValueError(
                        f"{me}: capacity `{capacity.name}` holds `{flow}`, "
                        "which a rule set of this component transforms and "
                        "which the component declares on both sides. The "
                        "volume would be the rules' supply and the flow's own "
                        "transit at once, so what leaves it has two readings "
                        "and the flow would cross twice. Hold a flow the "
                        "rules transform only on the side they transform it "
                        "on, or give the transit a flow of its own"
                    )
                wrong_way = (
                    flow in produced if capacity.side == "in" else flow in consumed
                )
                if wrong_way:
                    where, verb = (
                        ("upstream", "produce")
                        if capacity.side == "in"
                        else ("downstream", "consume")
                    )
                    raise ValueError(
                        f"{me}: capacity `{capacity.name}` sits {where} of "
                        f"the rules (side `{capacity.side}`) and holds "
                        f"`{flow}`, which a rule set of this component "
                        f"{verb}s. A volume beside the rules is the far end "
                        "of what they draw or make, so on this side it has "
                        "no rule to face and the flow would cross twice"
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

    def _refuse_ill_formed_taps(self) -> None:
        """Refuse every tap declaration that would not conserve.

        Run from :meth:`_build` and **not** from :meth:`_record`, unlike
        the other conservation guard: a tap names the output it delivers
        into, which the modeller is free to declare after the mode, so
        an early diagnostic here would refuse a well-formed component for
        the order its lines were written in.

        Five refusals, each closing a way for matter to appear or vanish
        without saying so:

        - a route naming something other than a continuous out-flow of
          this component, or naming the tapped flow itself;
        - the two sides of one tap routing to different outputs, which
          would move the fraction elsewhere on repair;
        - taps on one output that can together take more than all of it,
          which is what lets :meth:`_tap_rate_expr` do without a clamp;
        - a receiving output that also produces, which would add what it
          was handed to what it makes and count the fraction twice;
        - a cycle in the routing, an output feeding back into a stream it
          is fed from, which has no fixpoint the sweep could reach."""
        me = f"ObjFlow `{self.name}`"
        declared = [flow.name for flow in self.flows_continuous_out]
        routes: dict[str, str] = {}

        for flow in declared:
            for mode, _, _ in self._taps_on(flow):
                sides = {
                    side
                    for side in (
                        mode.failure.routes.get(flow),
                        mode.repair.routes.get(flow),
                    )
                    if side is not None
                }
                if len(sides) > 1:
                    raise ValueError(
                        f"{me}: failure mode `{mode.name}` taps `{flow}` to "
                        f"{sorted(sides)}, one destination on each side. A "
                        "route belongs to the tap and not to the state it is "
                        "read in; declare the same `to` on both sides, or on "
                        "one of them only"
                    )
                if not sides:
                    continue
                destination = sides.pop()
                if destination == flow:
                    raise ValueError(
                        f"{me}: failure mode `{mode.name}` taps `{flow}` to "
                        "itself. A tap moves a fraction OUT of its stream, so "
                        "delivering it back to that same stream takes nothing "
                        "off it; drop the `to` if the fraction is meant to "
                        "leave the system"
                    )
                if destination not in declared:
                    raise ValueError(
                        f"{me}: failure mode `{mode.name}` taps `{flow}` to "
                        f"`{destination}`, which is not a continuous out-flow "
                        f"of this component (it declares {declared}). What a "
                        "tap takes is delivered on an output of the component "
                        "it is taken from; connect that output to wherever the "
                        "fraction goes next"
                    )
                routes[flow] = destination

        for flow in declared:
            taken = sum(
                max(failed, repaired) for _, failed, repaired in self._taps_on(flow)
            )
            if taken > NOMINAL_RATE:
                raise ValueError(
                    f"{me}: the failure modes tapping `{flow}` can together "
                    f"take {taken:g} of it. Taps are parallel draws on one "
                    "stream and compose by sum, so they cannot between them "
                    "take more than the stream carries"
                )

        produced = {
            flow for rule_set in self.rule_sets for flow in rule_set.produced()
        } | {pair.source for pair in self.transfers if pair.is_conduit}
        for destination in set(routes.values()):
            flow = next(
                declared_out
                for declared_out in self.flows_continuous_out
                if declared_out.name == destination
            )
            reason = None
            if destination in produced:
                reason = "a rule set of this component already produces it"
            elif flow.capability_expr is not None:
                reason = "it carries a declared capability of its own"
            elif flow.var_fed_default:
                reason = f"it is declared at a rate of {flow.var_fed_default:g}"
            elif self._has_transfer_delta(destination):
                reason = "a transfer pair already adds to it"
            if reason is not None:
                raise ValueError(
                    f"{me}: `{destination}` receives what a tap takes, and "
                    f"{reason}. A receiving output carries what was taken off "
                    "another stream and nothing else, or the fraction is "
                    "counted twice; declare a separate output for it"
                )

        for start in routes:
            seen = [start]
            node = routes.get(start)
            while node is not None:
                if node in seen:
                    raise ValueError(
                        f"{me}: the taps route in a cycle, {' -> '.join(seen)} "
                        f"-> {node}. An output cannot be fed a fraction of a "
                        "stream it feeds; the sweep has no order that settles "
                        "both"
                    )
                seen.append(node)
                node = routes.get(node)

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

    # --- model generation --------------------------------------------

    def _build(
        self,
        connected_out: set[str] | None = None,
        connected_in: set[str] | None = None,
        ring_torn_out: set[str] | None = None,
        mixture_fed_in: dict[str, _MixtureSource] | None = None,
        wired_in: set[str] | None = None,
    ) -> dict[str, Any]:
        """The native component this declaration generates.

        `connected_out` names the continuous out-flows at least one
        connection reads, which only the system knows: an output nobody
        reads is not a bound on the rule producing it. Left out, every
        declared out-flow counts as read.

        `ring_torn_out` names the continuous out-flows feeding a volume
        on a recirculation ring (:meth:`System._ring_torn_outputs`),
        again something only the system knows. Such an output is read,
        but it no longer bounds the rule producing it: the volume at the
        other end absorbs what the two ends momentarily disagree on, and
        that is what makes the ring solvable at all. Left out, nothing is
        torn.

        `mixture_fed_in` names the continuous inputs a connection feeds
        from an output another component serves as a mixture
        (:meth:`_mixed_volume`), which only the system knows too. A rule
        consuming one runs at the draw what arrived allows
        (:meth:`_rule_draw_scale`). Kept on the component, since the
        system's sweep order and release read it after the build. Left
        out, no input is.

        `wired_in` names the continuous in-flows at least one connection
        feeds. What the rules may draw of one is what its suppliers
        allocated it, while one nothing feeds supplies its declared
        constant (:meth:`_draw_budget`). Left out, every declared
        continuous in-flow counts as wired.

        `connected_in` names the in-ports at least one connection feeds,
        the same kind of knowledge at the other end of the wire, and it
        is read for the trigger port alone: a `FlowOutOnTrigger` whose
        trigger nothing feeds is armed from the initial instant, so the
        automaton starts in `up` rather than crossing a zero-delay
        transition to get there. Left out, every declared trigger counts
        as connected. The BOOLEAN in-flows read none of it: what one
        answers out of connection is written into its own expression
        (see :meth:`_build_flows_in`).

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
        # Not re-run but run once, and only here: a tap names an output
        # that may legitimately be declared after the mode taking it.
        self._refuse_ill_formed_taps()
        self._mixture_sources = dict(mixture_fed_in or {})
        self._wired_in = (
            {flow.name for flow in self.flows_continuous_in}
            if wired_in is None
            else set(wired_in)
        )

        variables: list[dict] = []
        ports: list[dict] = []
        functions: list[dict] = []
        automata: list[dict] = []

        self._build_flows_in(variables, ports, functions)
        self._build_failure_modes(automata)
        self._build_flows_out(variables, ports, functions, automata, connected_in)

        # Continuous flows. What is emitted here is what the component
        # knows on its own; everything that reads the connection list
        # (the per-edge channels, the allocation operator, the two
        # aggregated in-channels) is emitted by `System.build_dict`.
        equations: list[dict] = []
        rule_demand, rule_capability, rule_production = self._build_rule_sets(
            connected_out, variables, automata, equations, ring_torn_out
        )
        self._build_continuous_in(
            rule_demand, connected_out, variables, ports, equations
        )
        self._build_continuous_out(
            rule_capability, rule_production, variables, ports, equations
        )
        self._build_transfers(variables, equations)
        self._build_capacities(variables, ports, equations, automata)
        self._build_mixtures(variables, automata)
        self._build_measurements(variables, ports, equations)
        self._build_measurements_out(variables, ports, equations)

        return {
            "name": self.name,
            "attributes": variables,
            "ports": ports,
            "interfaces": [],
            "automata": automata,
            "sensitive_functions": functions,
            "equations": equations,
        }


class System:
    """A muscadet-style system: add components, connect flows, simulate
    through the RAICHU engine."""

    def __init__(self, name: str, generated_indicators: bool = True):
        self.name = name
        self.comp: dict[str, ObjFlow] = {}
        self._connections: list[dict] = []
        #: Whether the document this system writes asks for the generated
        #: indicator set (:data:`pyraichu.indicators.GENERATED_INDICATORS`).
        #: **True** here and false in the document format, which is not a
        #: contradiction: a system built class by class is the muscadet
        #: authoring surface, whose models have always been observed variable
        #: by variable, and a document written by hand declares what it wants.
        self.generated_indicators = generated_indicators

    def add_component(self, cls: Type[ObjFlow], name: str) -> ObjFlow:
        """Instantiate `cls` under `name` and register it."""
        component = cls(name)
        self.comp[name] = component
        return component

    def add_declared_component(
        self, spec: dict[str, Any], classes: dict[str, type] | None = None
    ) -> ObjFlow:
        """Build a component from a declaration held in DATA and register
        it (`pyraichu.muscadet.declare`).

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
        anything is built. See :func:`pyraichu.muscadet.declare.check_spec` to
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
        # The publisher is a volume or an instrument republishing one: the two
        # publish under the same aliases, so the observer cannot tell them
        # apart and neither can the wiring.
        volume = next(
            (c for c in held.capacities if c.name == capacity), None
        ) or next((m for m in held.measurements_out if m.name == capacity), None)
        if volume is None:
            raise ValueError(
                f"System `{self.name}`: component `{holder}` declares no "
                f"capacity or published measurement `{capacity}`"
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

        # The reader materialises its channels from its own declaration,
        # and only the pairing knows whether the volume it reads publishes
        # ratios. Set before the connections are made, so the variables
        # exist by the time they are wired.
        link.ratios = volume.publishes_ratios

        pairs = [
            (_level_alias(capacity), _level_alias(channel)),
            (_fill_alias(capacity), _fill_alias(channel)),
        ]
        for flow in link.flows:
            pairs.append((_level_alias(capacity, flow), _level_alias(channel, flow)))
            pairs.append((_fill_alias(capacity, flow), _fill_alias(channel, flow)))
            if volume.publishes_ratios:
                pairs.append(
                    (_ratio_alias(capacity, flow), _ratio_alias(channel, flow))
                )
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

    def _releasing_inputs(
        self, in_edges: dict[tuple[str, str], list[_ContinuousEdge]]
    ) -> set[tuple[str, str]]:
        """The ``(component, flow)`` inputs that return what they do not
        need to their suppliers.

        Every supplier of one input is asked the input's whole demand, so
        together they may offer more than it needs. A **rule** consumes
        its need and no more, and muscadet's `release_unused_supply`
        hands the surplus back pro rata of what each supplier offered:
        without it the input kept everything offered and the difference
        vanished, a battery beside a bus emptying at twice the rate its
        electrolyser consumed.

        **A single supplier releases too.** The rule runs at the draw its
        scarcest arrival allows (:meth:`ObjFlow._rule_draw_scale`), which
        may be less than the scale its demand was published at: a recipe
        limited by one input consumes less of the others than it asked
        for, and the rest goes back rather than vanishing. The need is
        therefore what the rules consume at their draw,
        ``{flow}_consumed_in``, never the demand they published.

        Which inputs release is :meth:`ObjFlow._releases_surplus`: one a
        volume upstream of the rules holds is filled by what arrives
        instead, as muscadet's does.

        **A pure sink fed twice or more releases too** (muscadet 6.1.0,
        :meth:`ObjFlow._is_sink`): each supplier serves it its whole
        published demand, so it kept the sum of the offers, two sources of
        9 and 3 delivering 12 to a demand of 6. Its need is that demand.
        A single supplier already serves no more than it, so it releases
        nothing and the input is left as it was."""
        releasing: set[tuple[str, str]] = set()
        for (name, flow), edges in in_edges.items():
            obj = self.comp.get(name)
            if obj is None:
                continue
            if obj._releases_surplus(flow) or (
                len(edges) >= 2
                and obj._capacity_of(flow) is None
                and obj._is_sink(flow)
            ):
                releasing.add((name, flow))
        return releasing

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
        releasing = self._releasing_inputs(in_edges)

        def held_empty(edge: _ContinuousEdge) -> dict[str, Any] | None:
            """Whether `edge` leaves a volume that has run out of the flow
            and so passes on what arrives, as a location of that volume, or
            ``None`` when the supplier holds no such volume. What `transmits`
            says does not enter: an empty volume with a way in serves what
            crosses it either way (:meth:`ObjFlow._capacity_bounded_capability`)."""
            supplier = self.comp[edge.producer]
            capacity = supplier._capacity_of(edge.flow_out)
            if capacity is None or not any(
                declared.name == edge.flow_out
                for declared in supplier.flows_continuous_in
            ):
                return None
            return supplier._held_empty(capacity, edge.flow_out)

        def delivered(name: str, port: str, edge: _ContinuousEdge) -> dict[str, Any]:
            """What `edge` actually hands its consumer: its allocation,
            less its share of the surplus a releasing input returns."""
            channel = (
                "delivered" if (edge.consumer, edge.flow_in) in releasing else "alloc"
            )
            return _var(name, _channel_attr(port, channel, edge.name))

        for name, obj in self.comp.items():
            component = by_name[name]
            produced_by_rules = obj._produced_outputs()
            mixed = obj._mixed_outputs()
            for flow in obj.flows_continuous_out:
                port = f"{flow.name}_out"
                served = out_edges.get((name, flow.name), [])
                capability = _var(name, f"{flow.name}_capability_out")

                # What the output could deliver, published WHOLE to every
                # consumer, as muscadet's `incoming_capability` sums it
                # (R-20): a capability is asked before any demand exists
                # to apportion it against, so it is not apportioned. Two
                # consumers of one source each read all of it and size
                # their claim as if the other were absent; the split
                # settles what each receives, and what a rule makes is
                # bounded by that receipt (`ObjFlow._rule_draw_scale`).
                #
                # Netted per consumer (the capability less the others'
                # allocations), it made each consumer's claim depend on
                # the others' shares: an unbounded fill claim weighed at
                # what was left after a rival rather than at the supply,
                # and a rule bounded by it ratcheted down with its own
                # share.
                for edge in served:
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "capability", edge.name),
                            "kind": "explicit",
                            "expr": capability,
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
                        "expr": _sum([delivered(name, port, edge) for edge in served]),
                    }
                )
                # Into a releasing input, each edge delivers its
                # allocation scaled by what the consumer accepts, on a
                # channel of its own so the consumer can sum it.
                released = [
                    edge for edge in served if (edge.consumer, edge.flow_in) in releasing
                ]
                if released:
                    declared_port = next(
                        declared for declared in component["ports"]
                        if declared["name"] == port
                    )
                    declared_port["channels"].append({"name": "delivered"})
                for edge in released:
                    share: dict[str, Any] = _var(
                        edge.consumer, f"{edge.flow_in}_accepted_share"
                    )
                    empty = held_empty(edge)
                    if empty is not None:
                        share = {
                            "op": "if",
                            "cond": empty,
                            "then": _var(
                                edge.consumer, f"{edge.flow_in}_accepted_held_share"
                            ),
                            "otherwise": share,
                        }
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "delivered", edge.name),
                            "kind": "explicit",
                            "expr": {
                                "op": "mul",
                                "args": [
                                    _var(name, _channel_attr(port, "alloc", edge.name)),
                                    share,
                                ],
                            },
                        }
                    )

                # An output nobody reads distributes nothing: no
                # operator, and the two totals above stand at zero.
                if not served:
                    continue
                # What is there to distribute: the capability of a flow
                # nothing transforms, what the rule actually made when
                # one does, and a mixed volume's pooled draw
                # (:meth:`ObjFlow._pooled_serve`).
                available = (
                    _var(name, f"{flow.name}_produced_out")
                    if flow.name in produced_by_rules
                    else _var(name, f"{flow.name}_pooled_out")
                    if flow.name in mixed
                    else capability
                )
                # What the split weighs each consumer at: its demand AS
                # DECLARED (muscadet 5.24.0), so consumers in parallel on
                # a saturating supply are all served one fraction of what
                # they need: 12 between claims of 6 and 18 is 3 and 9.
                # Truncated at the quantity available, as muscadet did
                # from 5.11.0 to 5.23.0, the same claims weighed 6 and 12
                # and the small consumer was favoured. An unbounded claim
                # is the limit of a growing one: it takes everything and
                # the finite claims beside it are served nothing.
                declared_port = next(
                    declared for declared in component["ports"]
                    if declared["name"] == port
                )
                declared_port["channels"].append({"name": "claim"})
                # A volume drawn as a MIXTURE hands each machine its composed
                # share scaled to what it delivers (R52), muscadet's
                # `allocate_output`. The split weighing claims as declared,
                # the scaling gives what the raw claims would: 100 and 1
                # behind a ceiling of 50.5 are served 50 and 0.5 either way.
                # The total demand is a demand-band quantity and the
                # delivery reads it, so no band reads a later one.
                drawn = flow.name in mixed and mixed[flow.name].serves_mixture
                total = _var(name, f"{flow.name}_demand_out")
                for edge in served:
                    asked = _var(name, _channel_attr(port, "demand", edge.name))
                    claim = (
                        {
                            "op": "if",
                            "cond": {
                                "op": "cmp",
                                "cmp": "gt",
                                "lhs": total,
                                "rhs": _float(0.0),
                            },
                            "then": {
                                "op": "div",
                                "lhs": {"op": "mul", "args": [asked, available]},
                                "rhs": total,
                            },
                            "otherwise": _float(0.0),
                        }
                        if drawn
                        else asked
                    )
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "claim", edge.name),
                            "kind": "explicit",
                            "expr": claim,
                        }
                    )
                allocation: dict[str, Any] = {
                    "name": f"{flow.name}_alloc",
                    "port": port,
                    "available": available,
                    "demand": "claim",
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
                # A DECLARED supply replaces the aggregation: an input
                # nobody feeds reads `var_in_default`, and no sum over a
                # port answers that. Declaring nothing is the other case
                # and is not the same one: the input then reads zero,
                # which is exactly what the sum over its empty port
                # answers, so the two spellings carry the same number and
                # only one of them says WHERE the number comes from.
                #
                # Writing the constant there would erase the port from
                # the document, and with it the only thing that tells a
                # forgotten wire from a deliberate boundary value. That
                # is what `unfed_triggers` reads to name the wire, and it
                # is why the boolean inputs above keep their aggregation
                # under the same condition.
                declared = flow.var_in_default != 0.0
                releases = (name, flow.name) in releasing
                if releases:
                    # The share of what was offered the rule accepts: its
                    # need over the offers when they exceed it, all of it
                    # otherwise. Never a division by zero: a ratio is only
                    # taken where the offers exceed a need >= 0.
                    #
                    # A transiting volume that has RUN OUT passes on what
                    # arrives first, and the other suppliers share what is
                    # left. Scaled with them, it would hand on less than
                    # it receives, refill, leave its empty bound, offer
                    # its whole request, drain, and chatter on the bound
                    # for as long as the rule runs: the reference has the
                    # same dynamic, hidden by its fixed step. Passing on
                    # what arrives is that sliding motion's own average,
                    # the content held at zero.
                    offered = obj._offered(flow.name)
                    # A rule needs what it consumes at its draw, a sink
                    # the demand it published.
                    need = _var(
                        name,
                        f"{flow.name}_demand_in"
                        if obj._is_sink(flow.name)
                        else f"{flow.name}_consumed_in",
                    )
                    held = [
                        {
                            "op": "if",
                            "cond": empty,
                            "then": _var(
                                edge.producer,
                                _channel_attr(f"{edge.flow_out}_out", "alloc", edge.name),
                            ),
                            "otherwise": _float(0.0),
                        }
                        for edge in in_edges[(name, flow.name)]
                        for empty in [held_empty(edge)]
                        if empty is not None
                    ]

                    def ratio(wanted, available):
                        return {
                            "op": "if",
                            "cond": {"op": "cmp", "cmp": "gt", "lhs": available, "rhs": wanted},
                            "then": {"op": "div", "lhs": wanted, "rhs": available},
                            "otherwise": _float(1.0),
                        }

                    if held:
                        passed = _var(name, f"{flow.name}_accepted_held")
                        component["attributes"].append(
                            _float_attribute(f"{flow.name}_accepted_held", 0.0)
                        )
                        component["attributes"].append(
                            _float_attribute(f"{flow.name}_accepted_held_share", 1.0)
                        )
                        component["equations"] += [
                            {
                                "target": f"{flow.name}_accepted_held",
                                "kind": "explicit",
                                "expr": _sum(held),
                            },
                            {
                                "target": f"{flow.name}_accepted_held_share",
                                "kind": "explicit",
                                "expr": ratio(need, passed),
                            },
                        ]
                        # What the other suppliers are left to cover, and
                        # what they offered.
                        need = _clamped_at_zero({"op": "sub", "lhs": need, "rhs": passed})
                        offered = {"op": "sub", "lhs": offered, "rhs": passed}
                    component["attributes"].append(
                        _float_attribute(f"{flow.name}_accepted_share", 1.0)
                    )
                    component["equations"].append(
                        {
                            "target": f"{flow.name}_accepted_share",
                            "kind": "explicit",
                            "expr": ratio(need, offered),
                        }
                    )
                for channel, target in (
                    ("capability", f"{flow.name}_capability_in"),
                    ("delivered" if releases else "alloc", f"{flow.name}_fed_in"),
                ):
                    component["equations"].append(
                        {
                            "target": target,
                            "kind": "explicit",
                            "expr": (
                                _float(flow.var_in_default)
                                if declared and not connected
                                else {
                                    "op": "port_agg",
                                    "port": port_ref,
                                    "agg": "sum",
                                    "channel": channel,
                                }
                            ),
                        }
                    )

        return self._evaluation_order(components, out_edges, in_edges, releasing)

    def _flow_order(
        self, edges: dict[tuple[str, str], list[_ContinuousEdge]]
    ) -> list[str]:
        """The components in flow-graph topological order (producers
        before their consumers), ties broken by declaration order.

        A conservative flow network is cyclic by nature, so a cycle is
        expected and legal: the nodes it holds back are released in
        declaration order rather than refused.

        Kahn's algorithm over a heap of **declaration indices**: take the
        lowest index whose producers are all placed, and when a cycle
        leaves none of them, the lowest index **on a cycle** whose
        producers off that cycle are all placed. Lowest index *is*
        declared first, so the tie-break is never expressed a second
        time. The break releases a member of the ring and never a
        component merely downstream of it: released early, that one
        would be visited before one of its own suppliers, and every
        input it sums would read that supplier a sweep late."""
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

        ring_of, _ = _strongly_connected(successors)
        # What each node still waits for from OUTSIDE its own ring: at a
        # stall, the source rings of what is left have none, and a
        # singleton at zero would have been ready.
        from_outside = [0] * len(names)
        for producer, targets in enumerate(successors):
            for consumer in targets:
                if ring_of[producer] != ring_of[consumer]:
                    from_outside[consumer] += 1

        ready = [index for index in range(len(names)) if indegree[index] == 0]
        heapq.heapify(ready)
        placed = [False] * len(names)
        order: list[str] = []
        while len(order) < len(names):
            while ready and placed[ready[0]]:
                heapq.heappop(ready)
            if ready:
                chosen = heapq.heappop(ready)
            else:
                chosen = next(
                    index
                    for index in range(len(names))
                    if not placed[index] and from_outside[index] == 0
                )
            placed[chosen] = True
            order.append(names[chosen])
            # A node released by a cycle break still frees its
            # successors, exactly as one taken in order does.
            for successor in successors[chosen]:
                indegree[successor] -= 1
                if ring_of[chosen] != ring_of[successor]:
                    from_outside[successor] -= 1
                if indegree[successor] == 0 and not placed[successor]:
                    heapq.heappush(ready, successor)
        return order

    def _measurement_order(self, step) -> None:
        """Sweep the measurement readers and the instruments in the order
        their readings flow: a reader after the publication it reads, an
        instrument after the channel it republishes. With no instrument in
        the model this is the readers in declaration order, as before.

        A reading that feeds back into its own source through instruments
        has no order to be swept in, and is refused naming what it joins."""
        publisher_of = {
            (c["to"]["component"], c["to"]["port"]): (
                c["from"]["component"],
                c["from"]["port"],
            )
            for c in self._connections
        }
        instruments = {
            (name, publication.name): publication
            for name, obj in self.comp.items()
            for publication in obj.measurements_out
        }
        instrument_of_port = {
            (name, f"{alias}_out"): key
            for key, publication in instruments.items()
            for name in (key[0],)
            for alias, _ in publication.published()
        }
        readers = [
            (name, link)
            for name, obj in self.comp.items()
            for link in obj.measurements_in
        ]
        done_readers: set[tuple[str, str]] = set()
        done_instruments = {
            key for key, publication in instruments.items() if publication.source is None
        }

        def reader_ready(name: str, link: _MeasurementIn) -> bool:
            for variable in link.channels():
                publisher = publisher_of.get((name, f"{variable}_in"))
                instrument = instrument_of_port.get(publisher) if publisher else None
                if instrument is not None and instrument not in done_instruments:
                    return False
            return True

        def instrument_ready(key: tuple[str, str], publication: _MeasurementOut) -> bool:
            name = key[0]
            if any(link.name == publication.source for link in self.comp[name].measurements_in):
                return (name, publication.source) in done_readers
            return True

        progress = True
        while progress:
            progress = False
            for name, link in readers:
                if (name, link.name) in done_readers or not reader_ready(name, link):
                    continue
                for variable in link.channels():
                    step(name, variable)
                done_readers.add((name, link.name))
                progress = True
            for key, publication in instruments.items():
                if key in done_instruments or not instrument_ready(key, publication):
                    continue
                for _, attribute in publication.published():
                    step(key[0], attribute)
                done_instruments.add(key)
                progress = True
        stuck = [f"{n}.{link.name}" for n, link in readers if (n, link.name) not in done_readers]
        stuck += [f"{k[0]}.{k[1]}" for k in instruments if k not in done_instruments]
        if stuck:
            raise ValueError(
                f"System `{self.name}`: the measurement readings {stuck} feed "
                "back into their own sources through republishing instruments, "
                "so no order can sweep them"
            )

    def _evaluation_order(
        self,
        components: list[dict[str, Any]],
        out_edges: dict[tuple[str, str], list[_ContinuousEdge]],
        in_edges: dict[tuple[str, str], list[_ContinuousEdge]],
        releasing: set[tuple[str, str]],
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

        def after_last_supplier(publish, skip=frozenset()):
            """An input sums what ALL its suppliers publish, so it is
            swept once the last of them is, not at the first: a later
            supplier would otherwise be read at the previous evaluation's
            value, which the discrete fixpoint hides and a varying source
            exposes. In a ring the last supplier comes after the consumer,
            and the input is still swept after it: summed at the
            consumer's own visit, the input would read the ring one
            evaluation late in the balance it integrates, where only the
            pooled draw reads it late now (`_transit`)."""
            awaiting = {
                key: len(edges) for key, edges in in_edges.items() if key not in skip
            }

            def supplied(edge: _ContinuousEdge) -> None:
                key = (edge.consumer, edge.flow_in)
                if key in awaiting:
                    awaiting[key] -= 1
                    if awaiting[key] == 0:
                        publish(*key)

            return supplied

        capability_supplied = after_last_supplier(
            lambda consumer, flow_in: step(consumer, f"{flow_in}_capability_in")
        )

        def unconnected_inputs(suffix: str) -> None:
            """An input no producer feeds appears in no producer's band,
            and what it reads still has to be swept: its declared
            constant, or the aggregation over its empty port."""
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
                step(name, _content_attribute(capacity.name))
                # After the total they divide by, and before the
                # observers below read them.
                if capacity.publishes_ratios:
                    for entry in capacity.flows:
                        step(name, f"{capacity.name}_ratio_{entry.name}")
        self._measurement_order(step)
        # What multiplies a production, and what a gradient asks for:
        # neither reads the flow network, both are read by it. The
        # transfers come after the measurements, whose readings are the
        # potentials a conductive law is written over.
        for name, obj in self.comp.items():
            for flow in obj.flows_continuous_in:
                if flow.profile is not None:
                    step(name, f"{flow.name}_in_profile")
            for flow in obj.flows_continuous_out:
                if obj._derated(flow.name):
                    step(name, f"{flow.name}_effective_rate")
                if obj._taps_on(flow.name):
                    step(name, f"{flow.name}_tap_rate")
                if flow.profile is not None:
                    step(name, f"{flow.name}_out_profile")
            for pair in obj.transfers:
                step(name, f"{pair.name}_requested")

        # 1. Capability, along the flow. An input no producer feeds
        # supplies a fixed quantity, and a rule set may read it: swept
        # first, so a scale never sizes itself on a value the same pass
        # has not reached.
        unconnected_inputs("capability_in")
        for name, obj in self.comp.items():
            for flow in obj.flows_continuous_in:
                if flow.publish_rate == RATE_CAPABILITY:
                    step(name, _rate_alias(flow.name))
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
            # What each tapped stream carried before its taps drew, for
            # every stream of the component at once and ahead of the
            # capabilities: an output that RECEIVES a tap reads the base
            # of the output it is fed from, and the two are ordered by
            # the routing, not by which was declared first.
            for flow_name in self.comp[producer]._tap_ordered_outputs():
                if self.comp[producer]._taps_on(flow_name):
                    step(producer, f"{flow_name}_pre_tap_capability_out")
            for flow in self.comp[producer].flows_continuous_out:
                port = f"{flow.name}_out"
                step(producer, f"{flow.name}_capability_out")
                # A channel carrying what COULD cross belongs to this
                # band and not to the delivery band: read after the
                # delivery, it would close a loop through the observer's
                # own decision, which is the reason the two quantities
                # are distinguishable at all.
                if flow.publish_rate == RATE_CAPABILITY:
                    step(producer, _rate_alias(flow.name))
                served = out_edges.get((producer, flow.name), [])
                for edge in served:
                    step(producer, _channel_attr(port, "capability", edge.name))
                for edge in served:
                    capability_supplied(edge)

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
            # A volume downstream of the rules claims on what its
            # consumers asked, and the scale reads the claim.
            for claimed in self.comp[producer]._claimed_outputs().values():
                step(producer, claimed)
            # The rule's own scale: after its outputs have been asked,
            # and before its inputs ask, which a producer's own visit
            # (later in this reversed walk) is what steps.
            for rule_set in self.comp[producer].rule_sets:
                step(producer, f"{rule_set.name}_scale")
        unconnected_inputs("demand_in")

        # 3. Production, along the flow again. What a releasing input
        # accepts reads the offers of ALL its suppliers, and a later
        # supplier may be reached after an earlier one's own delivery
        # (a battery fed by the bus beside it), so that share, the
        # deliveries it scales and the totals summing them are swept at
        # the end of the band. Nothing else in the band reads them.
        deferred: list[tuple[str, dict[str, Any], list[_ContinuousEdge]]] = []

        def publish_fed(consumer: str, flow_in: str) -> None:
            """What one consumer was fed, with its published rate when
            declared."""
            step(consumer, f"{flow_in}_fed_in")
            if any(
                declared.name == flow_in and declared.publish_rate == RATE_DELIVERED
                for declared in self.comp[consumer].flows_continuous_in
            ):
                step(consumer, _rate_alias(flow_in))

        # A releasing input is swept at the end of the band, below.
        fed = after_last_supplier(publish_fed, skip=releasing)

        def delivery(producer: str, flow, served: list[_ContinuousEdge]) -> None:
            """What an output delivered, then what each consumer it
            serves was fed."""
            step(producer, f"{flow.name}_fed_out")
            if flow.publish_rate == RATE_DELIVERED:
                step(producer, _rate_alias(flow.name))
            for edge in served:
                fed(edge)

        for producer in topological:
            # What the rules draw reads what arrived, settled by the
            # producers visited before this one, in declaration order
            # since each set is offered what the earlier ones left;
            # everything this component makes reads the draw.
            for rule_set in self.comp[producer].rule_sets:
                if self.comp[producer]._runs_at_a_draw(rule_set):
                    step(producer, f"{rule_set.name}_draw_scale")
            # What the rules consume of an input that releases its
            # surplus, at the draw just swept: the release at the end of
            # the band measures the offers against it.
            for flow in self.comp[producer].flows_continuous_in:
                if self.comp[producer]._releases_surplus(flow.name):
                    step(producer, f"{flow.name}_consumed_in")
            for flow_name in self.comp[producer]._claimed_outputs():
                step(producer, f"{flow_name}_made_out")
            # A mixed volume's pooled excess reads what arrived, settled
            # by the suppliers visited before this component, and the
            # demands of the band before; what each output hands out
            # reads the excess, and the allocations below read that.
            mixed = self.comp[producer]._mixed_outputs()
            # An input nobody feeds transits its declared constant, and
            # is otherwise swept only at the end of the band: taken now,
            # the excess never reads it a sweep late.
            for flow in self.comp[producer].flows_continuous_in:
                if flow.name in mixed and not in_edges.get((producer, flow.name)):
                    step(producer, f"{flow.name}_fed_in")
            for capacity in self.comp[producer].capacities:
                if any(held is capacity for held in mixed.values()):
                    step(producer, self.comp[producer]._beyond_attribute(capacity))
            for flow_name in mixed:
                step(producer, f"{flow_name}_pooled_out")
            produced_by_rules = self.comp[producer]._produced_outputs()
            for flow_name in self.comp[producer]._tap_ordered_outputs():
                if (
                    self.comp[producer]._taps_on(flow_name)
                    and flow_name in produced_by_rules
                ):
                    step(producer, f"{flow_name}_pre_tap_produced_out")
            for flow in self.comp[producer].flows_continuous_out:
                served = out_edges.get((producer, flow.name), [])
                if flow.name in produced_by_rules:
                    step(producer, f"{flow.name}_produced_out")
                if served:
                    # The claims read what is available, settled just
                    # above, and the split reads the claims.
                    for edge in served:
                        step(producer, _channel_attr(f"{flow.name}_out", "claim", edge.name))
                    step(producer, f"{flow.name}_alloc")
                if any((edge.consumer, edge.flow_in) in releasing for edge in served):
                    # Only what the release scales waits: every OTHER
                    # consumer of this output reads its plain allocation,
                    # settled now, and is swept here, before its own
                    # visit reads what arrived.
                    for edge in served:
                        if (edge.consumer, edge.flow_in) not in releasing:
                            fed(edge)
                    deferred.append((producer, flow, served))
                    continue
                delivery(producer, flow, served)
            # What a conduit moved is what its flow delivered, so it is
            # read once that delivery is settled.
            waiting = {flow.name for owner, flow, _ in deferred if owner == producer}
            for pair in self.comp[producer].transfers:
                if pair.is_conduit and pair.source not in waiting:
                    step(producer, f"{pair.name}_moved")
        # Every offer is settled now: what each releasing input accepts,
        # then what each of its suppliers delivered, and the totals.
        position = {name: index for index, name in enumerate(self.comp)}
        for name, flow_name in sorted(
            releasing, key=lambda key: (position[key[0]], key[1])
        ):
            for attribute in (
                f"{flow_name}_accepted_held",
                f"{flow_name}_accepted_held_share",
            ):
                if any(
                    equation["target"] == attribute
                    for component in components
                    if component["name"] == name
                    for equation in component["equations"]
                ):
                    step(name, attribute)
            step(name, f"{flow_name}_accepted_share")
        # Every delivery and every supplier's total first, and only then
        # what each releasing input was fed: an input sums the deliveries
        # of ALL its suppliers, so reading it after the first one would
        # take the others from the previous evaluation.
        for producer, flow, served in deferred:
            port = f"{flow.name}_out"
            for edge in served:
                if (edge.consumer, edge.flow_in) in releasing:
                    step(producer, _channel_attr(port, "delivered", edge.name))
            step(producer, f"{flow.name}_fed_out")
            if flow.publish_rate == RATE_DELIVERED:
                step(producer, _rate_alias(flow.name))
            # What a conduit moved is what its flow delivered.
            for pair in self.comp[producer].transfers:
                if pair.is_conduit and pair.source == flow.name:
                    step(producer, f"{pair.name}_moved")
        for producer, flow, served in deferred:
            for edge in served:
                if (edge.consumer, edge.flow_in) in releasing:
                    publish_fed(edge.consumer, edge.flow_in)
        unconnected_inputs("fed_in")

        # What the atmosphere adds to an OPEN volume reads the deliveries,
        # every one of them now swept (R53), and is read by nothing but the
        # contents' derivatives.
        for name, obj in self.comp.items():
            for capacity in obj.capacities:
                if not capacity.is_open:
                    continue
                for entry in capacity.flows:
                    step(name, f"{capacity.name}_vent_{entry.name}")
                step(name, f"{capacity.name}_makeup")

        # The order must cover the declared steps exactly, and the
        # engine refuses an omission rather than completing it: close it
        # over what the components actually declare, so a step this
        # layer gains later cannot fall out of the sweep silently.
        #
        # What the closure adds is swept in DEPENDENCY order, not in
        # declaration order: a controller declared before the sensor it
        # reads would otherwise compare the reading the previous sweep
        # left behind, and a crossing it watches would be located at the
        # first scan point past the true date (4/3 h fired at 1.30625 h
        # on the H2 plant) or loop on the boundary.
        for name, target in self._dependency_ordered(
            [
                (component["name"], equation)
                for component in components
                for equation in component["equations"]
                if equation["kind"] == "explicit"
                and (component["name"], equation["target"]) not in listed
            ],
            components,
        ):
            step(name, target)
        for component in components:
            for allocation in component.get("allocations", []):
                step(component["name"], allocation["name"])
        return order

    def _dependency_ordered(
        self,
        pending: list[tuple[str, dict[str, Any]]],
        components: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """`pending` explicit equations as ``(component, target)``, each
        after every other pending one it reads, ties and cycles kept in
        the order given.

        A read is an attribute the expression names, or the attribute an
        out-port publishes into an in-port the expression aggregates
        (``port_agg`` with no channel), resolved through the connections.
        A per-connection channel is a quantity of the flow network, swept
        by its own band, and is not followed here."""
        published = {
            (component["name"], port["name"]): port.get("attr")
            for component in components
            for port in component.get("ports", [])
            if port.get("dir") == "out"
        }
        feeding: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for connection in self._connections:
            source, destination = connection["from"], connection["to"]
            attribute = published.get((source["component"], source["port"]))
            if attribute is not None:
                feeding.setdefault(
                    (destination["component"], destination["port"]), []
                ).append((source["component"], attribute))

        def reads(expr: Any, found: set[tuple[str, str]]) -> None:
            if isinstance(expr, dict):
                if expr.get("op") == "attr":
                    found.add((expr["attr"]["component"], expr["attr"]["attribute"]))
                elif expr.get("op") == "port_agg" and not expr.get("channel"):
                    port = expr["port"]
                    found.update(feeding.get((port["component"], port["port"]), []))
                for value in expr.values():
                    reads(value, found)
            elif isinstance(expr, list):
                for value in expr:
                    reads(value, found)

        keys = [(name, equation["target"]) for name, equation in pending]
        position = {key: index for index, key in enumerate(keys)}
        readers: list[list[int]] = [[] for _ in keys]
        missing = [0] * len(keys)
        for index, (_, equation) in enumerate(pending):
            found: set[tuple[str, str]] = set()
            reads(equation["expr"], found)
            for read in found:
                source = position.get(read)
                if source is not None and source != index:
                    readers[source].append(index)
                    missing[index] += 1
        # A cycle among these equations is released as a block, in
        # declaration order, and only once everything it reads from
        # outside the block is placed: its strongly connected components
        # (Tarjan) are ordered by Kahn's algorithm, lowest declaration
        # first. A real cycle is refused by the engine anyway; what this
        # keeps is an order that stays complete and never places an
        # equation before one it reads that is not on its cycle.
        # `readers` points from a read equation to its readers; Tarjan
        # walks the reversed edges (reader -> what it reads), which keeps
        # the numbering the order has always had.
        reads_of: list[list[int]] = [[] for _ in keys]
        for source, targets in enumerate(readers):
            for reader in targets:
                reads_of[reader].append(source)
        component_of, members = _strongly_connected(reads_of)
        needs = [0] * len(members)
        followers: list[set[int]] = [set() for _ in members]
        for source, targets in enumerate(readers):
            for reader in targets:
                a, b = component_of[source], component_of[reader]
                if a != b and b not in followers[a]:
                    followers[a].add(b)
                    needs[b] += 1
        ready = [(block[0], number) for number, block in enumerate(members) if needs[number] == 0]
        heapq.heapify(ready)
        ordered: list[tuple[str, str]] = []
        while ready:
            _, number = heapq.heappop(ready)
            ordered.extend(keys[member] for member in members[number])
            for follower in followers[number]:
                needs[follower] -= 1
                if needs[follower] == 0:
                    heapq.heappush(ready, (members[follower][0], follower))
        return ordered

    # --- rule diagnostics ---------------------------------------------

    def _validate_rules(self, edges: list[_ContinuousEdge]) -> None:
        """The three refusals the rule vocabulary makes sayable, in
        declaration order: a contested output nobody apportions, a loop
        of rate comparisons, and a self-feeding cycle that creates
        matter.

        A **commanded capacity** enters here for the second alone: a
        discharge command compares rates in the same vocabulary a rule
        guard does, and it drives the flow it releases, so it closes the
        same loop with no fixpoint. The third stays the rule
        vocabulary's, and not only because it would answer nothing
        without one: its search walks the whole flow graph under a budget
        it REFUSES the model for spending, and a model with no rule has
        no cycle it could report anyway, every connection carrying its
        quantity unchanged."""
        commanded = any(
            capacity.commanded
            for obj in self.comp.values()
            for capacity in obj.capacities
        ) or any(
            group.commanded for obj in self.comp.values() for group in obj.mixtures
        )
        rules = any(obj.rule_sets for obj in self.comp.values())
        if not rules and not commanded:
            return
        for obj in self.comp.values():
            obj._apportionment_shares()
        self._refuse_rate_comparison_loops(edges)
        if rules:
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
        together.

        **A commanded capacity is a node of the same graph**, and the
        same reasoning word for word: its discharge command compares
        rates in the rule guard's vocabulary and it drives what it
        releases, so a command reading a rate its own discharge moves
        has no more of a fixpoint than a rule selected on one. muscadet
        documents that angle as a blind spot of its own loop detectors
        (#172, "a comparison on a rate is not seen"); here it is the one
        place a capacity had to be written into.

        The two kinds of node close loops with each other as readily as
        with their own kind, which is why they share one graph rather
        than being looked for twice."""
        carried: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
        for edge in edges:
            carried.setdefault((edge.producer, edge.flow_out, "out"), []).append(
                (edge.consumer, edge.flow_in, "in")
            )

        # One node per switching decision, keyed by what it is so the
        # diagnostic can name it: `(component, what, name)`.
        influence: dict[tuple[str, str, str], set] = {}
        reads: dict[tuple[str, str, str], list] = {}

        def declare(key, driven_flows, operands, obj, name):
            # What this decision drives *before another one does*: its
            # own outputs and the inputs they feed. A rate that travels
            # further does so through another decision, which is then
            # the next node of the loop rather than a hop inside this one.
            driven = {(name, flow, "out") for flow in driven_flows}
            for seed in list(driven):
                driven.update(carried.get(seed, ()))
            influence[key] = driven
            compared = []
            for operand in operands:
                if operand.op is None:
                    continue
                _, rate = obj._operand_read(f"ObjFlow `{name}`", operand)
                if rate is not None:
                    compared.append((name, rate[0], rate[1]))
            reads[key] = compared

        for name, obj in self.comp.items():
            for rule_set in obj.rule_sets:
                declare(
                    (name, "rule set", rule_set.name),
                    rule_set.produced(),
                    [operand for rule in rule_set.rules for operand in rule.cond],
                    obj,
                    name,
                )
            delivered = {flow.name for flow in obj.flows_continuous_out}
            for capacity in obj.capacities:
                if not capacity.commanded:
                    continue
                declare(
                    (name, "capacity", capacity.name),
                    [flow for flow in capacity.flow_names if flow in delivered],
                    [
                        operand
                        for group in capacity.serve_cond
                        for operand in group
                    ],
                    obj,
                    name,
                )
            # A commanded MACHINE drives what the volume it draws delivers
            # (R52): the producer's outputs feeding the group, and the
            # group's own inputs they arrive on. Written out rather than
            # through `declare`, whose driven rates are the component's own
            # outputs; a machine has none.
            for group in obj.mixtures:
                if not group.commanded:
                    continue
                key = (name, "mixture group", group.name)
                feeding = [
                    edge
                    for edge in edges
                    if edge.consumer == name and edge.flow_in in group.flows
                ]
                influence[key] = {
                    (edge.producer, edge.flow_out, "out") for edge in feeding
                } | {(name, edge.flow_in, "in") for edge in feeding}
                compared = []
                for operand in (op for row in group.run_cond for op in row):
                    if operand.op is None:
                        continue
                    _, rate = obj._operand_read(f"ObjFlow `{name}`", operand)
                    if rate is not None:
                        compared.append((name, rate[0], rate[1]))
                reads[key] = compared

        arcs: dict[tuple[str, str, str], list[tuple[tuple[str, str, str], tuple]]] = {}
        for driver, driven_nodes in influence.items():
            for reader, endpoints in reads.items():
                shared = [node for node in endpoints if node in driven_nodes]
                if shared:
                    arcs.setdefault(driver, []).append((reader, shared[0]))

        cycle = _find_cycle(list(influence), arcs)
        if cycle is None:
            return
        rendered = " -> ".join(
            f"{node[1]} `{node[2]}` of `{node[0]}` (driving the rate "
            f"`{endpoint[1]}`)"
            for node, endpoint in cycle
        )
        raise ValueError(
            f"System `{self.name}`: the guards of {rendered} form a loop "
            "of rate comparisons: each drives a rate the next one switches "
            "on, so the selection has no fixpoint. Compare an integrated "
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

    def _ring_torn_outputs(self, edges: list[_ContinuousEdge]) -> dict[str, set[str]]:
        """Per component, the rule outputs whose connection closes a
        recirculation ring on a volume: the edge the ring is torn at.

        A fan drawing air out of a room and blowing it back in is a ring
        of instantaneous values: the fan's rule is bounded by the demand
        on its return output, which is the room's inbound demand, which
        carries the fan's own demand for room air through the transit.
        The engine refuses such a chain as having no solution, and
        rightly so while nothing on it integrates.

        A volume on the ring changes that, and muscadet accepts the ring
        for it (a ring with no volume is its R30 refusal). It tears the
        edge **entering** the volume and reads the torn quantities one
        evaluation late. The same edge is torn here at its **other end**:
        the rule producing into it is no longer bounded by the volume's
        demand, and the volume absorbs whatever the two ends momentarily
        disagree on, which is what a volume is. No late read is needed,
        so the sweep keeps its consistent solution where the reference
        shifts the volume's level once by one integration stage.

        An edge is torn when it feeds a flow its consumer holds in a
        volume, its producer makes that flow by a rule, and the flow
        graph leads from the volume back to that output: through the
        connections, the rules (consumed to produced), the pass-throughs
        and the transiting volumes. A ring closing through a producer
        that is not a rule (a plain pipe) is left alone, so the engine's
        refusal still names it: tearing it would need the pipe's demand
        cut, which is another statement.

        **The tear is taken only where the volume provably absorbs what
        the two ends disagree on** (:meth:`_tear_conserves`), since a
        connection never delivers more than its consumer asks: a torn
        rule making more than the volume accepts would lose the surplus,
        and one left with no other bound would run unbounded. Every other
        ring keeps the engine's refusal."""
        successors: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
        for node, steps in self._rule_graph(edges).items():
            successors.setdefault(node, set()).update(step[0] for step in steps)
        for name, obj in self.comp.items():
            declared_in = {flow.name for flow in obj.flows_continuous_in}
            declared_out = {flow.name for flow in obj.flows_continuous_out}
            transiting = {
                flow
                for capacity in obj.capacities
                if capacity.transmits
                for flow in capacity.flow_names
            }
            for flow in declared_in & declared_out:
                if flow in transiting or obj._passes_through(flow):
                    successors.setdefault((name, flow, "in"), set()).add(
                        (name, flow, "out")
                    )

        def reaches(start, goal) -> bool:
            seen, pending = {start}, [start]
            while pending:
                node = pending.pop()
                if node == goal:
                    return True
                for following in successors.get(node, ()):
                    if following not in seen:
                        seen.add(following)
                        pending.append(following)
            return False

        torn: dict[str, set[str]] = {}
        for edge in edges:
            producer = self.comp.get(edge.producer)
            consumer = self.comp.get(edge.consumer)
            if producer is None or consumer is None:
                continue
            held = {
                flow for capacity in consumer.capacities for flow in capacity.flow_names
            }
            made = {
                flow for rule_set in producer.rule_sets for flow in rule_set.produced()
            }
            if edge.flow_in not in held or edge.flow_out not in made:
                continue
            if reaches(
                (edge.consumer, edge.flow_in, "in"),
                (edge.producer, edge.flow_out, "out"),
            ) and self._tear_conserves(edge, edges):
                torn.setdefault(edge.producer, set()).add(edge.flow_out)
        return torn

    def _tear_conserves(
        self, edge: _ContinuousEdge, edges: list[_ContinuousEdge]
    ) -> bool:
        """Whether tearing the ring at `edge` keeps every quantity
        bounded and conserved, which holds for a direct recirculation:

        - the torn output feeds the volume and nothing else, so no other
          consumer loses the bound its demand put on the rule;
        - the volume hands the same flow straight back to the producer,
          and every rule making the torn output draws at least as much of
          that returned flow as it makes, so it never blows back more
          than the volume let out, which the volume always accepts
          (full, it accepts exactly what leaves it);
        - every such rule makes something else a consumer reads, which
          still bounds it once the torn output no longer does.

        A ventilated room is exactly this: the fan returns the air it
        drew and exhausts to the atmosphere. A fan that only recirculates,
        one that makes more air than it draws, or a return read off the
        ring fails a condition and keeps the engine's refusal."""
        producer = self.comp[edge.producer]
        if [
            other
            for other in edges
            if other.producer == edge.producer and other.flow_out == edge.flow_out
        ] != [edge]:
            return False
        returned = {
            back.flow_in
            for back in edges
            if back.producer == edge.consumer
            and back.consumer == edge.producer
            and back.flow_out == edge.flow_in
        }
        if not returned:
            return False
        bounding = {
            other.flow_out
            for other in edges
            if other.producer == edge.producer and other.flow_out != edge.flow_out
        }
        for rule_set in producer.rule_sets:
            for rule in rule_set.rules:
                made = rule.prod.get(edge.flow_out, 0.0)
                if made <= 0:
                    continue
                drawn = max((rule.cons.get(flow, 0.0) for flow in returned), default=0.0)
                if drawn < made:
                    return False
                if not any(
                    rule.prod.get(flow, 0.0) > 0 for flow in bounding
                ):
                    return False
        return True

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

    def _resolve_mixture_groups(self, edges: list[_ContinuousEdge]) -> None:
        """Bind every mixture group to the one volume that composes for it
        (muscadet's `resolve_mixture_groups`, R51), and write what the group
        asks on each of its inputs: ``R . m_f / S``, ``S`` being the volume's
        weighted occupied volume, read off its integrated contents.

        Several groups may draw one volume (muscadet's R52): each asks its own
        rate at the composition, and the producer's split serves them.

        Refused by name, as muscadet refuses them: a group whose flows do not
        all arrive from ONE capacity of ONE producer, a group leaving out a
        constituent of that volume, a producer serving any held flow of it
        to a consumer that is not one of its groups, a volume drawn by
        several groups whose outputs allocate other than proportionally, a
        flow a rule of the consumer also consumes, and a FED volume drawn as a
        mixture at ``fill_rate=0`` (muscadet 5.18.0: asking only its composed
        share, it would never admit a constituent it does not hold, nor start
        when empty). Recomputed from scratch
        at every generation, so a model generated twice binds the same
        way."""
        for obj in self.comp.values():
            for capacity in obj.capacities:
                capacity.serves_mixture = False
            for flow in obj.flows_continuous_in:
                flow.mixture_demand = None
        # (producer, capacity name) -> the INPUTS drawing it as a mixture,
        # `(consumer, flow_in)` pairs, which is what the exclusivity rule
        # exempts, and the groups, which the allocation rule counts. Keyed on
        # the input and not on the consumer: a connection here joins ports,
        # so a machine may wire an UNGROUPED input to a held output too, and
        # that one is a per-flow request like any other.
        drawers: dict[tuple[str, str], set[tuple[str, str]]] = {}
        bound: dict[tuple[str, str], list[str]] = {}
        for consumer, obj in self.comp.items():
            for group in obj.mixtures:
                for edge in edges:
                    if edge.consumer == consumer and edge.flow_in in group.flows:
                        supplier = self.comp[edge.producer]
                        held = supplier._capacity_of(edge.flow_out)
                        if held is not None:
                            drawers.setdefault((edge.producer, held.name), set()).add(
                                (consumer, edge.flow_in)
                            )
        for consumer, obj in self.comp.items():
            for group in obj.mixtures:
                where = f"System `{self.name}`: mixture group `{consumer}.{group.name}`"
                feeding = {
                    flow: [e for e in edges if e.consumer == consumer and e.flow_in == flow]
                    for flow in group.flows
                }
                producers = {e.producer for found in feeding.values() for e in found}
                if any(len(found) != 1 for found in feeding.values()) or len(producers) != 1:
                    raise ValueError(
                        f"{where}: its flows must all arrive from one capacity of "
                        f"one producer, and they arrive from "
                        f"{ {f: [e.producer for e in found] for f, found in feeding.items()} }"
                    )
                producer = producers.pop()
                supplier = self.comp[producer]
                outputs = {flow: found[0].flow_out for flow, found in feeding.items()}
                held = [supplier._capacity_of(out) for out in outputs.values()]
                names = {c.name if c is not None else None for c in held}
                if len(names) != 1 or None in names:
                    raise ValueError(
                        f"{where}: its flows must all arrive from one capacity of "
                        f"one producer, and `{producer}` holds them in "
                        f"{sorted(n or 'no capacity' for n in names)}"
                    )
                capacity = held[0]
                left_out = [
                    flow
                    for flow in capacity.flow_names
                    if flow not in outputs.values()
                ]
                if left_out:
                    raise ValueError(
                        f"{where}: capacity `{capacity.name}` of `{producer}` also "
                        f"holds {left_out}, which the group does not draw. A "
                        "machine moves a volume of whatever the room holds, so a "
                        "group names every constituent of the volume it draws"
                    )
                mixtures = drawers.get((producer, capacity.name), set())
                others = [
                    e for e in edges
                    if e.producer == producer
                    and e.flow_out in capacity.flow_names
                    and (e.consumer, e.flow_in) not in mixtures
                ]
                if others:
                    raise ValueError(
                        f"{where}: `{producer}` also serves "
                        f"{sorted({e.consumer for e in others})} out of the volume "
                        "the group draws, and a volume drawn as a mixture serves "
                        "mixture groups alone: a composed share against a per-flow "
                        "request is not arbitrated"
                    )
                bound.setdefault((producer, capacity.name), []).append(
                    f"{consumer}.{group.name}"
                )
                consumed = {
                    flow for rule_set in obj.rule_sets for flow in rule_set.consumed()
                }
                ruled = sorted(set(group.flows) & consumed)
                if ruled:
                    raise ValueError(
                        f"{where}: {ruled} are consumed by a rule of `{consumer}` "
                        "too, and an input has one demand"
                    )
                capacity.serves_mixture = True
                occupied = _sum(
                    [
                        {
                            "op": "mul",
                            "args": [
                                _var(producer, _content_attribute(capacity.name, entry.name)),
                                _float(entry.weight),
                            ],
                        }
                        for entry in capacity.flows
                    ]
                )
                # The rate is READ, never written in: a failure mode may have
                # clamped it (R52), and a mode writing below zero states a
                # direction, which is the connection's and not the rate's.
                # The command BRANCHES on it, never multiplies.
                rate: dict[str, Any] = _clamped_at_zero(
                    _var(consumer, group.rate_attribute)
                )
                if group.commanded:
                    rate = {
                        "op": "if",
                        "cond": _state_active(consumer, group.gate, RUN_GATE_STATES[0]),
                        "then": rate,
                        "otherwise": _float(0.0),
                    }
                for flow, out in outputs.items():
                    share = {
                        "op": "div",
                        "lhs": _var(producer, _content_attribute(capacity.name, out)),
                        "rhs": occupied,
                    }
                    demand = {
                        "op": "if",
                        "cond": {"op": "cmp", "cmp": "gt", "lhs": occupied, "rhs": _float(0.0)},
                        "then": {"op": "mul", "args": [rate, share]},
                        "otherwise": _float(0.0),
                    }
                    target = next(f for f in obj.flows_continuous_in if f.name == flow)
                    target.mixture_demand = demand
        for (producer, capacity_name), groups in bound.items():
            supplier = self.comp[producer]
            capacity = next(c for c in supplier.capacities if c.name == capacity_name)
            fed = sorted(
                {
                    e.flow_in
                    for e in edges
                    if e.consumer == producer and e.flow_in in capacity.flow_names
                }
            )
            if fed and not capacity.fill_rate:
                raise ValueError(
                    f"System `{self.name}`: capacity `{capacity_name}` of "
                    f"`{producer}` is drawn as a mixture ({', '.join(groups)}) "
                    f"and fed on {', '.join(fed)}, with fill_rate=0. At that "
                    "rate it asks upstream only what the machines take at its "
                    "composition, so a constituent it does not hold would never "
                    "enter and an empty volume would never fill. Declare a "
                    "fill_rate: math.inf takes whatever the producers deliver"
                )
            if len(groups) < 2:
                continue
            for out in supplier.flows_continuous_out:
                if out.name not in capacity.flow_names:
                    continue
                if out.allocation != "proportional":
                    raise ValueError(
                        f"System `{self.name}`: capacity `{capacity_name}` of "
                        f"`{producer}` is drawn by several mixture groups "
                        f"({', '.join(groups)}), and its output `{out.name}` "
                        f"declares allocation={out.allocation!r}. Machines on one "
                        "volume are served pro rata of their composed shares, "
                        "which is the proportional allocation: declare none on "
                        "that output"
                    )

    def generate(
        self, foreign: list[dict[str, Any]] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]] | None]:
        """Build every registered component and add the material only the
        connection list can supply, answering the components and the
        evaluation order they need (``None`` when nothing is continuous).

        `foreign` names components of the SAME model this system did not
        build: none when :meth:`build_dict` writes the whole document, and
        the rest of the model when the serialized plugin path
        (`pyraichu.muscadet.plugin`) drives this system over a document
        that also carries controllers, failure-mode objects or hand-written
        components. They receive no continuous material, since they declare
        no flow this layer resolves, but the evaluation order **closes over
        them**: the order must cover the declared steps exactly, so a step
        some other object emitted has to be swept or the whole model is
        refused.

        This is the one continuous generation. :meth:`build_dict` and the
        plugin reach the per-edge equations, the allocation
        operators and the three-band order through it, so the two authoring
        surfaces cannot answer one model differently."""
        edges = self._continuous_edges()
        self._validate_rules(edges)
        self._resolve_mixture_groups(edges)
        reading: dict[str, set[str]] = {}
        for edge in edges:
            reading.setdefault(edge.producer, set()).add(edge.flow_out)
        torn = self._ring_torn_outputs(edges)
        # The inputs fed from an output served as a mixture: a rule
        # consuming one runs at the draw what arrived allows.
        mixture_fed: dict[str, dict[str, _MixtureSource]] = {}
        for edge in edges:
            supplier = self.comp[edge.producer]
            volume = supplier._mixed_volume(edge.flow_out)
            if volume is not None:
                mixture_fed.setdefault(edge.consumer, {})[edge.flow_in] = _MixtureSource(
                    edge.producer,
                    edge.flow_out,
                    volume.name,
                    any(
                        declared.name == edge.flow_out
                        for declared in supplier.flows_continuous_in
                    ),
                )
        wired: dict[str, set[str]] = {}
        for edge in edges:
            wired.setdefault(edge.consumer, set()).add(edge.flow_in)
        components = [
            obj._build(
                reading.get(name, set()),
                connected_in_flows(self._connections, name),
                torn.get(name, set()),
                mixture_fed.get(name, {}),
                wired.get(name, set()),
            )
            for name, obj in self.comp.items()
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
                if obj._runs_at_a_draw(rule_set):
                    observable.add(f"{rule_set.name}_draw_scale")
            for flow_name, claimed in obj._claimed_outputs().items():
                observable.add(claimed)
                observable.add(f"{flow_name}_made_out")
            # A mixed volume's pooled excess and what each output hands
            # out, and what a rule fed by one consumes at its draw.
            mixed = obj._mixed_outputs()
            for flow_name, capacity in mixed.items():
                observable.add(obj._beyond_attribute(capacity))
                observable.add(f"{flow_name}_pooled_out")
            for flow in obj.flows_continuous_in:
                if obj._fed_by_a_mixture(flow.name):
                    observable.add(f"{flow.name}_consumed_in")
            for capacity in obj.capacities:
                observable.add(_content_attribute(capacity.name))
                observable.add(f"{capacity.name}_fill")
                for entry in capacity.flows:
                    observable.add(_content_attribute(capacity.name, entry.name))
                    observable.add(f"{capacity.name}_fill_{entry.name}")
                    if capacity.publishes_ratios:
                        observable.add(f"{capacity.name}_ratio_{entry.name}")
                    if capacity.is_open:
                        observable.add(f"{capacity.name}_vent_{entry.name}")
                if capacity.is_open:
                    observable.add(f"{capacity.name}_makeup")
            for measurement in obj.measurements_in:
                observable.update(measurement.channels())
            for flow in obj.flows_continuous_in:
                if flow.profile is not None:
                    observable.add(f"{flow.name}_in_profile")
                if flow.publish_rate is not None:
                    observable.add(_rate_alias(flow.name))
            for flow in obj.flows_continuous_out:
                if flow.publish_rate is not None:
                    observable.add(_rate_alias(flow.name))
                if obj._derated(flow.name):
                    observable.add(f"{flow.name}_effective_rate")
                observable.update(obj._external_deratings_on(flow.name))
                if obj._taps_on(flow.name):
                    observable.add(f"{flow.name}_tap_rate")
                    observable.add(f"{flow.name}_pre_tap_capability_out")
                    observable.add(f"{flow.name}_pre_tap_produced_out")
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
        constructs only and keeps the bare body it has always had.

        The generated set is governed by
        :data:`~pyraichu.indicators.GENERATED_INDICATORS`, and the key is
        **written into the body** rather than left implicit. Two reasons, and
        the second is the load-bearing one:

        * a document says what it holds, so a reader of the file knows why it
          carries eleven indicators and not one, without knowing which
          authoring surface wrote it;
        * the plugin expansion and this builder write **one document** for one
          model (`test_plugin_continuous.py`), and they cannot if one of them
          states the key and the other relies on a default. The identity is
          the evidence that there is one implementation of the continuous
          semantics and not two, which is worth more than the saved line.
        """
        components, evaluation_order = self.generate()
        body: dict[str, Any] = {
            "name": self.name,
            "components": components,
            "connections": self._connections,
            "indicators": (
                self.indicators(components) if self.generated_indicators else []
            ),
            GENERATED_INDICATORS: self.generated_indicators,
        }
        if evaluation_order is None:
            return body
        body.update(model_level_keys(evaluation_order))
        # The feature list is derived from the body by the engine, never
        # composed here: it cannot lag what the body holds.
        return seal(body)

    def build_model(self) -> Model:
        """Generate and validate the native RAICHU model."""
        return load_model(json.dumps(self.build_dict()))

    def simulate(self, t_max: float, **kwargs: Any) -> SimulationResult:
        """One trajectory through the RAICHU engine."""
        return run_explained(simulate, self.build_model(), t_max=t_max, **kwargs)

    def monte_carlo(
        self, nb_runs: int, t_max: float, samples: list[float], **kwargs: Any
    ) -> McEstimates:
        """Monte-Carlo estimation through the RAICHU driver."""
        return run_explained(
            monte_carlo,
            self.build_model(),
            nb_runs=nb_runs,
            t_max=t_max,
            samples=samples,
            **kwargs,
        )
