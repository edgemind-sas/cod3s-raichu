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
from ._system_mixtures import _SystemMixturesMixin
from ._system_rules import _SystemRulesMixin
from ._system_network import _SystemNetworkMixin


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


class System(
    _SystemMixturesMixin,
    _SystemRulesMixin,
    _SystemNetworkMixin,
):
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
