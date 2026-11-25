"""Transformation rules of a component: scales, apportionment and draws."""

from __future__ import annotations

from typing import Any

from ._expr import (
    _clamped_at_zero,
    _float,
    _float_attribute,
    _max,
    _min,
    _renamed_reads,
    _state_active,
    _sum,
    _var,
)
from ._records import _Rule, _RuleSet
from ._vocabulary import NOMINAL_RATE, UNBOUNDED_SERVICE


class _RuleScalesMixin:
    """Methods of :class:`ObjFlow` transformation rules of a component: scales, apportionment and draws."""

    def _rule_coefficient(
        self, rule_set: _RuleSet, flow: str, side: str
    ) -> dict[str, Any]:
        """The coefficient on `flow` of whichever rule of `rule_set` is
        selected: its ``cons`` (`side` ``"cons"``) or its ``prod``, zero
        for a rule that does not name the flow."""
        return self._rule_choice(
            rule_set,
            [_float(getattr(rule, side).get(flow, 0.0)) for rule in rule_set.rules],
        )

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

    def _contested_producers(self) -> dict[str, list[_RuleSet]]:
        """Per output two or more rule sets produce into, those sets in
        declaration order. A sole producer contests nothing and is left
        out."""
        producers: dict[str, list[_RuleSet]] = {}
        for rule_set in self.rule_sets:
            for flow in rule_set.produced():
                producers.setdefault(flow, []).append(rule_set)
        return {flow: sets for flow, sets in producers.items() if len(sets) > 1}

    def _apportionment_shares(self) -> dict[str, dict[str, float]]:
        """Per contested output **every** producer apportions, the
        normalised share of its demand each rule set claims.

        An output no producer apportions is not here: its demand
        **cascades** through the sets in declaration order (see
        :meth:`_cascaded_producers`). An output apportioned by some of
        its producers only is **refused** (R13): a share on one set and
        none on the other says neither a ratio nor an order."""
        shares: dict[str, dict[str, float]] = {}
        for flow, sets in self._contested_producers().items():
            silent = [s.name for s in sets if flow not in s.apportionment]
            if len(silent) == len(sets):
                continue
            if silent:
                raise ValueError(
                    f"ObjFlow `{self.name}`: rule sets "
                    f"{', '.join(f'`{s.name}`' for s in sets)} all produce into "
                    f"`{flow}`, and {', '.join(f'`{name}`' for name in silent)} "
                    "declare no `apportionment` of it while the others do; "
                    "declare a share on every producer, or on none of them "
                    "to have the demand served in declaration order"
                )
            declared = {s.name: s.apportionment[flow] for s in sets}
            total = sum(declared.values())
            shares[flow] = {name: value / total for name, value in declared.items()}
        return shares

    def _cascaded_producers(self) -> dict[str, list[_RuleSet]]:
        """Per contested output no producer apportions, its producers in
        declaration order: the first is asked the whole demand, and each
        next one only what those before it left unserved.

        muscadet hands **every** active set the whole demand, delivers
        at most that demand and drops the surplus, which no balance
        records: two sets asked 50 each draw 100 for 50 delivered. Here
        the component draws exactly what it delivers. The order is
        muscadet's own production order, the declaration order, so the
        two agree whenever an earlier set cannot serve the whole demand,
        which is how a trace leak and an air renewal share one exhaust."""
        return {
            flow: sets
            for flow, sets in self._contested_producers().items()
            if not any(flow in s.apportionment for s in sets)
        }

    def _rule_capability_scale(
        self, rule: _Rule, exclude: str | None = None
    ) -> dict[str, Any]:
        """The scale a rule could run at, set by its scarcest input: the
        minimum of what each consumed flow's producers could deliver,
        divided by that flow's coefficient.

        `exclude` leaves one consumed flow out, which is the scale the
        rule's OTHER inputs could sustain (muscadet's `get_supply_scale`):
        what the claim on that flow is bounded by. Nothing else bounding
        it, the scale is unbounded and not nominal.

        A coefficient of zero is the catalyst idiom: the rule names the
        flow so a guard can read it but draws none of it, so it is not
        limited by it. A rule nothing constrains, having no consumed
        flow at all, runs at its nominal scale of 1 and produces exactly
        its declared quantities.

        An output declaring a `max_rate` bounds this scale as well, at
        `max_rate / coefficient`. Here and not only on the delivered
        quantity, because this scale is what the derived demand is built
        from: bounded only at the output, the component would go on
        asking its suppliers for what it was asked for while making the
        lesser quantity its ceiling allows, and the difference would be
        drawn upstream and lost.

        An input held in a volume **upstream of the rules** is read
        through that volume: whatever the ceiling allows while the flow
        is stocked, what could arrive once it has run out
        (:meth:`_rule_supply`). The stock is the rule's supply, and the
        capability upstream is only what refills it."""

        def supply(flow: str) -> dict[str, Any]:
            capability = _var(self.name, f"{flow}_capability_in")
            capacity = self._beside_rules(flow, "in")
            if capacity is None:
                return capability
            return self._rule_supply(capacity, flow, capability)

        terms = [
            {"op": "div", "lhs": supply(flow), "rhs": _float(coefficient)}
            for flow, coefficient in rule.cons.items()
            if coefficient > 0 and flow != exclude
        ]
        terms += [
            _float(ceiling / coefficient)
            for flow, coefficient in rule.prod.items()
            if coefficient > 0
            for ceiling in [self._max_rate_on(flow)]
            if ceiling is not None
        ]
        if terms:
            return _min(terms)
        return _float(1.0 if exclude is None else UNBOUNDED_SERVICE)

    def _max_rate_on(self, flow: str) -> float | None:
        """The ceiling declared on `flow`, or ``None``. Looked up rather
        than threaded through, a rule naming a flow by name and not by
        the declaration that carries it."""
        for declared in self.flows_continuous_out:
            if declared.name == flow:
                return declared.max_rate
        return None

    def _rule_demand_scale(
        self,
        rule_set: _RuleSet,
        rule: _Rule,
        connected_out: set[str],
        shares: dict[str, dict[str, float]],
        cascades: dict[str, list[_RuleSet]],
        claiming: str | None = None,
        torn: set[str] | None = None,
    ) -> dict[str, Any]:
        """The scale a rule actually runs at: its capability scale,
        bounded by what its outputs are asked for.

        With `claiming` naming a consumed flow, the scale the rule CLAIMS
        that flow at instead: the same bound by its outputs, and the
        capability of its other inputs only (muscadet 5.24.0). An input is
        not bounded by its own supplier: its claim is the need behind it,
        which is the weight that need is given where the supply is shared.

        The output bound is a **minimum**, not a maximum: the outputs of
        one rule are correlated by construction, so a scale above what
        any one of them can take would produce a surplus of the others
        that has nowhere to go and would be recorded by no balance.

        An output **no connection reads** does not take part: nobody is
        asking is not the same statement as somebody asking for nothing,
        and only the second is a bound. A rule left with no reading
        output at all, or producing nothing, runs at its **nominal**
        scale of 1 where its inputs allow it, as muscadet's does: it
        claims its declared coefficients and no more, so a machine
        nothing is connected behind does not draw its supplier's whole
        capability, nor take a rival's share of it.

        An output in `torn` is another matter: something does read it,
        on the edge a ring is cut at, and its demand is only left out of
        this band. A rule whose every asked output is torn keeps its
        capability scale."""
        ran = _var(self.name, f"{rule_set.name}_capability_scale")
        terms = [
            ran if claiming is None else self._rule_capability_scale(rule, claiming)
        ]
        for flow, coefficient in rule.prod.items():
            if coefficient <= 0 or flow not in connected_out:
                continue
            demand: dict[str, Any] = _var(self.name, f"{flow}_demand_out")
            share = shares.get(flow, {}).get(rule_set.name)
            if share is not None:
                demand = {"op": "mul", "args": [demand, _float(share)]}
            earlier = cascades.get(flow, [])
            earlier = earlier[: earlier.index(rule_set)] if rule_set in earlier else []
            if earlier:
                # What the sets declared before this one already make of
                # the flow, at the scale they actually run at: each is
                # bounded by the demand left to it, so the remainder
                # never goes negative on its own, and the clamp only
                # absorbs rounding.
                served = _sum(
                    [
                        {
                            "op": "mul",
                            "args": [
                                self._rule_coefficient(before, flow, "prod"),
                                self._expected_scale(before),
                            ],
                        }
                        for before in earlier
                    ]
                )
                demand = _clamped_at_zero({"op": "sub", "lhs": demand, "rhs": served})
            terms.append({"op": "div", "lhs": demand, "rhs": _float(coefficient)})
        if len(terms) == 1:
            if any(
                coefficient > 0 and flow in (torn or ())
                for flow, coefficient in rule.prod.items()
            ):
                # Read on a torn edge only: the scale it runs at.
                return ran
            # Nothing asks: the nominal scale, bounded by the inputs.
            terms.append(_float(1.0))
        return _min(terms)

    def _expected_scale(self, rule_set: _RuleSet) -> dict[str, Any]:
        """The scale `rule_set` can be EXPECTED to run at, from the demand
        band: its scale, bounded by what a mixed volume feeding one of its
        inputs will leave it.

        A mixed volume (:meth:`_mixed_volume`) passes what transits and a
        share of the excess pro rata of its content, so a set asking it
        for 50 of a trace constituent receives about half a unit, and
        runs at that draw. The declaration-order cascade that follows
        must subtract what the set will MAKE, not what it asked for:
        subtracting the request left the next set asked for nothing, and
        the ventilation exhausted 0.39 where it exhausts 50.

        The volume's offer is estimated on the needs of this set and of
        the sets declared before it on the same input,
        ``transit + max(need - transit, 0) x ratio`` capped at that need,
        which is the pooled delivery exactly when the volume's other held
        flows are covered by what transits them (a ventilated room, whose
        returned air covers the air asked); otherwise it under-states it,
        and the next set is asked a little more, never less. The earlier
        sets are served out of it first, each taking at most its own
        need and drawing it at its uptake factor, since the draw shares
        the input in declaration order (:meth:`_rule_draw_scale`): a
        natural draft declared after a forced extraction is expected to
        make nothing while the extraction runs, so the demand it leaves
        on a contested output is handed on to the next producer, and
        everything once a mode has cut the extraction's output. The
        transit is read from the capability band, what could arrive, so
        this reads nothing the production band has yet to settle."""
        scale = _var(self.name, f"{rule_set.name}_scale")
        bounds = [scale]
        position = self.rule_sets.index(rule_set)
        for flow in rule_set.consumed():
            source = self._mixture_sources.get(flow)
            if source is None or not self._fed_by_a_mixture(flow):
                continue
            coefficient = self._rule_coefficient(rule_set, flow, "cons")
            need = {"op": "mul", "args": [coefficient, scale]}
            earlier = [
                before
                for before in self.rule_sets[:position]
                if flow in before.consumed() and self._runs_at_a_draw(before)
            ]
            asked = _sum(
                [
                    {
                        "op": "mul",
                        "args": [
                            self._rule_coefficient(before, flow, "cons"),
                            _var(self.name, f"{before.name}_scale"),
                        ],
                    }
                    for before in earlier
                ]
                + [need]
            )
            # A constituent the volume holds but is never fed transits
            # nothing, as `_transit` reads it on the volume's side.
            transit = (
                _var(source.supplier, f"{source.flow_out}_capability_in")
                if source.has_inlet
                else _float(0.0)
            )
            excess = _clamped_at_zero({"op": "sub", "lhs": asked, "rhs": transit})
            offered = _min(
                [
                    asked,
                    {
                        "op": "add",
                        "args": [
                            transit,
                            {
                                "op": "mul",
                                "args": [
                                    excess,
                                    _var(
                                        source.supplier,
                                        f"{source.volume}_ratio_{source.flow_out}",
                                    ),
                                ],
                            },
                        ],
                    },
                ]
            )
            # The earlier sets are served first out of that offer, each
            # taking at most its own need of what is left, and drawing it
            # at its uptake factor (:meth:`_rule_draw_scale`).
            for before in earlier:
                wanted = {
                    "op": "mul",
                    "args": [
                        self._rule_coefficient(before, flow, "cons"),
                        _var(self.name, f"{before.name}_scale"),
                    ],
                }
                taken = _min([wanted, offered])
                uptake = self._uptake_factor(before)
                if uptake is not None:
                    taken = {"op": "mul", "args": [taken, uptake]}
                offered = _clamped_at_zero({"op": "sub", "lhs": offered, "rhs": taken})
            expected = _min([need, offered])
            bounds.append(
                {
                    "op": "if",
                    "cond": {"op": "cmp", "cmp": "gt", "lhs": coefficient, "rhs": _float(0.0)},
                    "then": {"op": "div", "lhs": expected, "rhs": coefficient},
                    "otherwise": scale,
                }
            )
        return _min(bounds)

    def _build_rule_sets(
        self,
        connected_out: set[str] | None,
        variables: list[dict],
        automata: list[dict],
        equations: list[dict],
        ring_torn_out: set[str] | None = None,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
    ]:
        """The two scales of each rule set, its mode automaton when it is
        guarded, and the per-flow contributions the flows themselves fold
        in afterwards. An output in `ring_torn_out` takes no part in the
        demand bound on the scale, for the reason :meth:`_build` gives.

        Those contributions are the return value, keyed by flow: what
        each set demands of an input it consumes, and what it makes an
        output capable of and what it actually produced there. They are
        handed back rather than written into the flow declarations, so a
        rule set and a capacity on one flow compose whichever was
        declared first: the flow folds the contributions in, and the
        capacity bounds the result.

        A volume **beside the rules** changes two things here and leaves
        the coefficients alone, as muscadet does:

        - downstream (``side="out"``), the consumers' demand on the held
          output is replaced by the volume's claim on it,
          ``{flow}_claimed_out`` (:meth:`_capacity_claim_out`), before the
          scale is taken on it: the rule is asked to make what is taken
          plus what the volume claims for itself;
        - upstream (``side="in"``), the volume is what the rules draw
          from (:meth:`_rule_supply` on what arrived).

        Every set that draws a continuous input runs at
        ``{set}_draw_scale`` (:meth:`_rule_draw_scale`): its scale bounded
        by what its inputs actually supply, shared in declaration order
        between the sets consuming one input. The scale itself stays the
        NEED the demand band publishes upstream; the draw is what the
        production band makes, and what empties a volume."""
        me = self.name
        claimed = self._claimed_outputs()
        # A volume downstream of the rules bounds them through its claim
        # whether or not anything reads the output: a tank nobody draws
        # from yet still stops the rules filling it once full.
        reading = (
            (
                {flow.name for flow in self.flows_continuous_out}
                if connected_out is None
                else set(connected_out)
            )
            | set(claimed)
        ) - set(ring_torn_out or ())
        shares = self._apportionment_shares()
        cascades = self._cascaded_producers()
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
                    "expr": _renamed_reads(
                        self._rule_choice(
                            rule_set,
                            [
                                self._rule_demand_scale(
                                    rule_set,
                                    rule,
                                    reading,
                                    shares,
                                    cascades,
                                    torn=ring_torn_out,
                                )
                                for rule in rule_set.rules
                            ],
                        ),
                        me,
                        claimed,
                    ),
                }
            )
            made = scale
            if self._runs_at_a_draw(rule_set):
                made = f"{rule_set.name}_draw_scale"
                variables.append(_float_attribute(made, 0.0))
                equations.append(
                    {
                        "target": made,
                        "kind": "explicit",
                        "expr": self._rule_choice(
                            rule_set,
                            [
                                self._rule_draw_scale(rule_set, rule, _var(me, scale))
                                for rule in rule_set.rules
                            ],
                        ),
                    }
                )
            for flow in rule_set.consumed():
                coefficient = self._rule_coefficient(rule_set, flow, "cons")
                # The claim on an input is the need behind it, bounded by
                # the rule's other inputs and never by its own supplier.
                claim_scale = _renamed_reads(
                    self._rule_choice(
                        rule_set,
                        [
                            self._rule_demand_scale(
                                rule_set,
                                rule,
                                reading,
                                shares,
                                cascades,
                                flow,
                                torn=ring_torn_out,
                            )
                            for rule in rule_set.rules
                        ],
                    ),
                    me,
                    claimed,
                )
                rule_demand.setdefault(flow, []).append(
                    {"op": "mul", "args": [coefficient, claim_scale]}
                )
            for flow in rule_set.produced():
                coefficient = self._rule_coefficient(rule_set, flow, "prod")
                rule_capability.setdefault(flow, []).append(
                    {"op": "mul", "args": [coefficient, _var(me, capability_scale)]}
                )
                rule_production.setdefault(flow, []).append(
                    {"op": "mul", "args": [coefficient, _var(me, made)]}
                )

        # What the rules actually consume of each input that releases its
        # surplus, at their draw: the need the release measures the offers
        # against, so the suppliers keep what was not consumed.
        for flow in sorted(
            {
                flow
                for rule_set in self.rule_sets
                for flow in rule_set.consumed()
                if self._releases_surplus(flow)
            }
        ):
            variables.append(_float_attribute(f"{flow}_consumed_in", 0.0))
            equations.append(
                {
                    "target": f"{flow}_consumed_in",
                    "kind": "explicit",
                    "expr": _sum(
                        [
                            self._set_draw(rule_set, flow)
                            for rule_set in self.rule_sets
                            if flow in rule_set.consumed()
                        ]
                    ),
                }
            )

        # The claims the scales above read, one per output held in a
        # volume downstream of the rules. Written after the loop because
        # an unbounded fill claims what the rules could make, which is
        # every set's capability contribution to that output.
        for flow, target in claimed.items():
            capacity = self._beside_rules(flow, "out")
            variables.append(_float_attribute(target, 0.0))
            equations.append(
                {
                    "target": target,
                    "kind": "explicit",
                    "expr": self._capacity_claim_out(
                        capacity, flow, _sum(rule_capability.get(flow, []))
                    ),
                }
            )

        return rule_demand, rule_capability, rule_production

    def _runs_at_a_draw(self, rule_set: _RuleSet) -> bool:
        """Whether `rule_set` actually draws a continuous input, and so
        runs at a draw scale of its own (:meth:`_rule_draw_scale`): some
        rule of it consumes a continuous in-flow of this component at a
        positive coefficient.

        A set naming its inputs at a zero coefficient only (the catalyst
        idiom) draws nothing, and one consuming nothing makes its
        declared quantities: both run at their scale."""
        continuous = {flow.name for flow in self.flows_continuous_in}
        return any(
            coefficient > 0 and flow in continuous
            for rule in rule_set.rules
            for flow, coefficient in rule.cons.items()
        )

    def _draw_budget(self, flow: str) -> dict[str, Any]:
        """What the rule sets here may draw of `flow` between them, in
        the production band: what actually reaches them, never what
        could.

        - A volume **upstream of the rules** lets them take whatever its
          ceiling allows while the flow is stocked, and once it has run
          out what arrived and no more (:meth:`_rule_supply`).
        - A **wired** input reads what its suppliers allocated it, before
          any release hands part of it back: the release is settled at
          the end of the band and reads the draws, so the draws cannot
          read it. Through a single supplier and no release this is
          ``{flow}_fed_in`` exactly.
        - An input **no connection feeds** reads ``{flow}_fed_in``, its
          declared constant (an empty port sums to zero, which is not
          what such an input supplies)."""
        capacity = self._beside_rules(flow, "in")
        if capacity is not None:
            return self._rule_supply(capacity, flow, _var(self.name, f"{flow}_fed_in"))
        if flow in self._wired_in:
            return self._offered(flow)
        return _var(self.name, f"{flow}_fed_in")

    def _rule_draw_scale(
        self, rule_set: _RuleSet, rule: _Rule, scale: dict[str, Any]
    ) -> dict[str, Any]:
        """The scale `rule` actually runs at, in the production band: its
        `scale` (the need the demand band published upstream), bounded
        by what each input it consumes actually supplies it, over the
        coefficient.

        **What arrived, not what could** (muscadet's `rule_scale` on
        `get_input_available`). The demand band sizes a claim on the
        suppliers' capability, which is published unapportioned (R-20):
        two consumers of one source each read the whole of it. Producing
        at that scale would make what never arrived, so the production
        is bounded here by the budget of each input
        (:meth:`_draw_budget`), and an input that delivered less than it
        was asked caps the whole set, its other legs included.

        **Several sets on one input share it, in declaration order**
        (muscadet R-17): each is offered what the sets declared before it
        left, and what an earlier set leaves is what it actually drew,
        its uptake factor included (R-13, :meth:`_set_draw`). A set whose
        outputs a mode derated to zero therefore takes nothing from a
        sibling declared after it, which keeps its whole demand. An
        unbounded budget is never spent down: it crosses this layer as
        :data:`UNBOUNDED_SERVICE`, a magnitude no draw moves."""
        terms = [scale]
        position = self.rule_sets.index(rule_set)
        for flow, coefficient in rule.cons.items():
            if coefficient <= 0 or not self._is_continuous_in(flow):
                continue
            remaining = self._draw_budget(flow)
            earlier = [
                self._set_draw(before, flow)
                for before in self.rule_sets[:position]
                if flow in before.consumed() and self._runs_at_a_draw(before)
            ]
            if earlier:
                remaining = _clamped_at_zero(
                    {"op": "sub", "lhs": remaining, "rhs": _sum(earlier)}
                )
            terms.append({"op": "div", "lhs": remaining, "rhs": _float(coefficient)})
        return _min(terms)

    def _uptake_factor(self, rule_set: _RuleSet) -> dict[str, Any] | None:
        """What a set's draw on a volume is scaled by when a failure mode
        derates what it produces (muscadet R-13, ``get_uptake_factor``),
        or None when no output it produces is derated.

        The **maximum** over the outputs the set produces of what their
        factors leave of them, clamped at one. Maximum and not minimum: a
        derating is a loss on the leg it bears on, never a saving on the
        reagents the other legs still consume, so cutting one output of a
        two-output reaction to zero must not cut the draw to zero. An
        output no mode derates counts as nominal.

        Applied to the DRAW only, never to `made`: the production is
        already multiplied by the output's own factors downstream, so
        folding this in there would apply a derating of 0.5 twice. A fan
        whose exhaust a mode cuts to zero then stops drawing the room it
        ventilates, which is what lets the room fill."""
        produced = {
            flow.name: flow
            for flow in self.flows_continuous_out
            if flow.name in rule_set.produced()
        }
        if not any(self._derated(name) for name in produced):
            return None
        legs = [
            {"op": "mul", "args": self._output_factors(flow)}
            if self._derated(name)
            else _float(NOMINAL_RATE)
            for name, flow in produced.items()
        ]
        return _min([_max(legs), _float(NOMINAL_RATE)])

    def _set_draw(self, rule_set: _RuleSet, flow: str) -> dict[str, Any]:
        """What one set draws of `flow` from a volume upstream: its
        coefficient times its draw scale, times its uptake factor when a
        mode derates what it produces."""
        args = [
            self._rule_coefficient(rule_set, flow, "cons"),
            _var(self.name, f"{rule_set.name}_draw_scale"),
        ]
        uptake = self._uptake_factor(rule_set)
        if uptake is not None:
            args.append(uptake)
        return {"op": "mul", "args": args}

    def _rule_draw(self, flow: str) -> dict[str, Any]:
        """What the rule sets draw of `flow` from the volume upstream of
        them: each consuming set's coefficient times its draw scale. The
        volume's only outflow."""
        terms = []
        for rule_set in self.rule_sets:
            if flow not in rule_set.consumed():
                continue
            terms.append(self._set_draw(rule_set, flow))
        return _sum(terms)
