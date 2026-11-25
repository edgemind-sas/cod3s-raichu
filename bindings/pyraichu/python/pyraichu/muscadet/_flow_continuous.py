"""Continuous flows of a component: declaration and emission."""

from __future__ import annotations

from typing import Any

from ._expr import _clamped_at_zero, _float, _float_attribute, _min, _sum, _var
from ._records import _FlowContinuousIn, _FlowContinuousOut, _rate_alias
from ._vocabulary import (
    _CONTINUOUS_CHANNELS,
    _NO_CAPACITY,
    NOMINAL_RATE,
    RATE_DELIVERED,
    RATE_QUANTITIES,
    UNBOUNDED_SERVICE,
)


class _ContinuousFlowsMixin:
    """Methods of :class:`ObjFlow` continuous flows of a component: declaration and emission."""

    def add_flow_continuous_in(
        self,
        name: str,
        var_demand_in_default: float = 0.0,
        var_in_default: float = 0.0,
        profile: dict[str, Any] | None = None,
        publish_rate: bool | str | None = None,
    ) -> None:
        """Declare a real-valued input, aggregating every incoming
        connection by sum.

        `var_demand_in_default` is what this consumer asks its
        producers for while nothing derives a demand for it (a pure
        consumer derives none); `var_in_default` is what the input
        reads, on both the received and the capability channel, when
        nothing is connected to it.

        `profile` is a declared **continuous** function of simulation
        time multiplying what this input asks for, in the same shape and
        under the same refusal as an output's: a demand that varies with
        the clock is as ordinary as a production that does. It scales the
        demand whatever the demand is derived from, so a rule set
        consuming the flow is scaled with it.

        `publish_rate` materialises the quantity this input receives as
        an observable channel, `{flow}_rate` on a port of its own, so a
        controller can threshold what a component consumes. Declared and
        not implied: publishing every rate would put a port on every flow
        of every model for the few an observer reads."""
        self._reject_flow_clash(name, "in")
        self.flows_continuous_in.append(
            _FlowContinuousIn(
                name=name,
                var_demand_in_default=float(var_demand_in_default),
                var_in_default=float(var_in_default),
                profile=self._parse_profile(name, profile, side="in"),
                publish_rate=self._rate_quantity(name, publish_rate),
            )
        )

    def _rate_quantity(self, flow: str, declared: Any) -> str | None:
        """Which quantity a flow's rate channel carries, normalised.

        ``True`` is the delivered quantity, which is what the channel
        carried when it had no choice, so a declaration written before
        the distinction existed still means what it meant."""
        if declared is None or declared is False:
            return None
        if declared is True:
            return RATE_DELIVERED
        if declared not in RATE_QUANTITIES:
            raise ValueError(
                f"ObjFlow `{self.name}`: continuous flow `{flow}` declares "
                f"`publish_rate`={declared!r}; a rate channel carries one of "
                f"{list(RATE_QUANTITIES)} (or `true` for "
                f"`{RATE_DELIVERED}`)"
            )
        return declared

    def add_flow_continuous_out(
        self,
        name: str,
        var_fed_default: float = 0.0,
        allocation: str = "proportional",
        allocation_shares: dict[Any, float] | None = None,
        allocation_priorities: dict[Any, float] | None = None,
        profile: dict[str, Any] | None = None,
        max_rate: float | None = None,
        publish_rate: bool | str | None = None,
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
        ``{"cls": <shape>, ...}``, the shape one of
        :data:`PROFILE_CLASSES` with the keys of
        :data:`PROFILE_CLASS_KEYS`: a sinusoid (`amplitude`, `period`,
        `phase_shift`, `offset`, and a `name` labelling the curve, which
        muscadet writes on every exported sinusoid and this layer carries
        back), a square wave with ramped edges (`low`, `high`, `period`,
        `high_duration`, `ramp_duration`, `phase_shift`) or a drawn
        periodic curve (`points`, `period`, `phase_shift`). Every shape
        takes the clamp `value_min`/`value_max` and an optional smoothed
        noise (`noise_amplitude`, `noise_time_scale`, `noise_key`, the
        key defaulting to :func:`flow_noise_key` of this flow).
        A Python callable is refused: continuity is an attestation this
        layer cannot make for the modeller, and it derives no watched
        transition from a function's breakpoints. The profile composes
        with the derating rate by **product**, never by minimum: an
        output at 0.3 of its profile that is also derated to 0.5
        produces 0.15.

        `max_rate` is a ceiling on what this output can deliver per unit
        time: the equipment cannot make more, whatever it is fed and
        whatever is asked of it. On an output a rule set produces it also
        bounds the scale the rule runs at, so what the component asks its
        own suppliers for is what it will actually make and not what it
        was asked for. It is an absolute quantity, where a failure mode's
        cap is a fraction of nominal; the two compose by minimum, as two
        ceilings do.

        `publish_rate` materialises the quantity this output delivers as
        an observable channel, `{flow}_rate` on a port of its own, so a
        controller can threshold what a component produces. It carries
        what actually crossed and not what could have: an observer
        watching a supply wants what arrived."""
        if max_rate is not None and max_rate < 0.0:
            raise ValueError(
                f"ObjFlow `{self.name}`: continuous out-flow `{name}` declares "
                f"`max_rate`={max_rate:g}; a ceiling on a delivered quantity "
                "is not negative. A ceiling of zero is an output that "
                "delivers nothing"
            )
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
                max_rate=None if max_rate is None else float(max_rate),
                publish_rate=self._rate_quantity(name, publish_rate),
            )
        )

    def _produced_outputs(self) -> set[str]:
        """The continuous outputs whose delivery is something other than
        their declared rate: what a rule set makes, what a conduit meters
        across, what a pass-through hands on, and what a routed tap hands
        to a receiving output.

        These are exactly the outputs carrying a ``{flow}_produced_out``,
        what was actually made as opposed to what could have been, and it
        is that quantity the allocation distributes. Derived here rather
        than at each of the places that need it, because it is a property
        of the component's own declarations and none of them knows
        anything the others do not.

        A receiver inherits the distinction from its sources: it carries
        a production only where at least one stream feeding it publishes
        one, since a fraction of what was merely *available* upstream is
        not something that was made. Reached by fixpoint because a tap
        may feed an output that is itself tapped; the walk is monotone,
        so it terminates whether or not the routing graph is acyclic, and
        :meth:`_refuse_ill_formed_taps` is what rules the cycle out."""
        produced = (
            {flow for rule_set in self.rule_sets for flow in rule_set.produced()}
            | {pair.source for pair in self.transfers if pair.is_conduit}
            | {
                flow.name
                for flow in self.flows_continuous_out
                if self._passes_through(flow.name)
            }
        )
        while True:
            grown = produced | {
                flow.name
                for flow in self.flows_continuous_out
                if any(
                    source in produced
                    for source, _, _, _ in self._tapped_into(flow.name)
                )
            }
            if grown == produced:
                return produced
            produced = grown

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
            if self._derated(flow.name)
            else _var(self.name, f"{flow.name}_out_rate")
        ]
        if flow.profile is not None:
            factors.append(_var(self.name, f"{flow.name}_out_profile"))
        return factors

    def _bounded_by_max_rate(
        self, flow: _FlowContinuousOut, expr: dict[str, Any]
    ) -> dict[str, Any]:
        """`expr` bounded by the output's declared ceiling, or `expr`
        itself where none is declared.

        Applied to the delivered quantity and not only to the rule scale,
        because an output need not come from a rule at all: a declared
        rate, a conduit's crossing and a routed tap's delivery each reach
        this point by their own branch, and a ceiling that bound only one
        of them would be a property of the declaration it was written
        beside rather than of the equipment."""
        if flow.max_rate is None:
            return expr
        return _min([expr, _float(flow.max_rate)])

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
        # A standalone mode clamps a variable of its own rather than the
        # shared one, so two of them on one output fold like two caps.
        terms += [_var(self.name, name) for name in self._external_deratings_on(flow)]
        for mode, failed, repaired in self._caps_on(flow):
            terms.append(self._mode_term(mode, failed, repaired))
        return _min(terms)

    def _releases_surplus(self, flow: str) -> bool:
        """Whether the input `flow` hands back to its suppliers what the
        rules did not consume of it: muscadet's `release_unused_supply`.

        The demand that fetched a rule's inputs is sized on what each
        could deliver, before the scale the rule actually runs at is
        known, so a recipe limited by one input is offered more of the
        others than it consumes. Whatever the number of suppliers, the
        rule takes its consumption at its draw and the rest goes back:
        without it, the surplus of the inputs the scarce one limits
        vanished, and a tank supplying one of them emptied faster than
        the recipe justifies.

        Three inputs keep what they are offered, as muscadet's do: one no
        connection feeds, one no rule consumes, and one a volume of this
        component holds, which is filled by what arrives instead. A pure
        sink releases too, against its demand rather than a consumption:
        :meth:`System._releasing_inputs` decides it, since it needs the
        number of suppliers."""
        return (
            flow in self._wired_in
            and self._capacity_of(flow) is None
            and any(flow in rule_set.consumed() for rule_set in self.rule_sets)
        )

    def _is_sink(self, flow: str) -> bool:
        """Whether the continuous input `flow` ends here: no rule set
        consumes it, it does not cross the component, no conduit meters
        it, no mixture group draws it and no mixed volume feeds it.

        muscadet's `sink_input_flows`. What such an input needs is its
        published demand, ``{flow}_demand_in``, which the release reads
        where a rule input reads ``{flow}_consumed_in``. A sink fed by a mixed volume is
        left out: what it is handed is a share of a pooled excess, which
        the release does not measure against a demand."""
        return (
            any(declared.name == flow for declared in self.flows_continuous_in)
            and not any(flow in rule_set.consumed() for rule_set in self.rule_sets)
            and not self._passes_through(flow)
            and self._conduit_on(flow) is None
            and not any(flow in group.flows for group in self.mixtures)
            and flow not in self._mixture_sources
        )

    def _offered(self, flow: str) -> dict[str, Any]:
        """What the suppliers of `flow` hand this input, before any of it
        is handed back: the sum of their allocations on it."""
        return {
            "op": "port_agg",
            "port": {"component": self.name, "port": f"{flow}_in"},
            "agg": "sum",
            "channel": "alloc",
        }

    def _is_continuous_in(self, flow: str) -> bool:
        """Whether `flow` names a continuous in-flow of this component."""
        return any(declared.name == flow for declared in self.flows_continuous_in)

    def _build_continuous_in(
        self,
        rule_demand: dict[str, list[dict[str, Any]]],
        connected_out: set[str] | None,
        variables: list[dict],
        ports: list[dict],
        equations: list[dict],
    ) -> None:
        """The continuous in-flows: their three channels, their in port,
        and the demand this component carries upstream on each.

        The demand has one source of the five, in this order of
        precedence: a conduit metering the flow, the rule sets consuming
        it, a declared `demand_expr`, the output the flow passes through
        to, and failing all four the declared constant. Whichever it is,
        a declared time profile scales it and the volume holding the flow
        bounds it, in that order: the profile says how much is wanted, the
        volume says how much of that there is room for."""
        for continuous_in in self.flows_continuous_in:
            flow_name = continuous_in.name
            variables.append(
                _float_attribute(f"{flow_name}_fed_in", continuous_in.var_in_default)
            )
            # An infinite declared demand is "whatever there is", which a
            # document cannot carry as an infinity.
            declared_demand = min(
                continuous_in.var_demand_in_default, UNBOUNDED_SERVICE
            )
            variables.append(
                _float_attribute(f"{flow_name}_demand_in", declared_demand)
            )
            variables.append(
                _float_attribute(
                    f"{flow_name}_capability_in", continuous_in.var_in_default
                )
            )
            if continuous_in.publish_rate is not None:
                variables.append(
                    _float_attribute(
                        _rate_alias(flow_name), continuous_in.var_in_default
                    )
                )
                ports.append(
                    {
                        "name": f"{_rate_alias(flow_name)}_out",
                        "dir": "out",
                        "attr": _rate_alias(flow_name),
                    }
                )
                equations.append(
                    {
                        "target": _rate_alias(flow_name),
                        "kind": "explicit",
                        "expr": _var(
                            self.name,
                            f"{flow_name}_fed_in"
                            if continuous_in.publish_rate == RATE_DELIVERED
                            else f"{flow_name}_capability_in",
                        ),
                    }
                )
            if continuous_in.profile is not None:
                # A read-only publication of the factor the demand sweep
                # applies, as on an output: writing it has no effect, the
                # next evaluation overwriting it from the curve.
                variables.append(
                    _float_attribute(
                        f"{flow_name}_in_profile",
                        continuous_in.profile.factor(0.0),
                    )
                )
                equations.append(
                    {
                        "target": f"{flow_name}_in_profile",
                        "kind": "explicit",
                        "expr": self._profile_expr(continuous_in.profile),
                    }
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
            elif continuous_in.mixture_demand is not None:
                base = continuous_in.mixture_demand
            elif continuous_in.demand_expr is not None:
                base = continuous_in.demand_expr
            elif self._passes_through(flow_name):
                # A pass-through asks for what its output is asked for,
                # and an output nothing reads asks for nothing: carried
                # as its declared default instead, it would take a share
                # of a supply a wired rival needs. A ceiling on the output
                # bounds the ask as well, for the same reason.
                read = connected_out is None or flow_name in connected_out
                base = _float(0.0)
                if read:
                    base = _var(self.name, f"{flow_name}_demand_out")
                    ceiling = self._max_rate_on(flow_name)
                    if ceiling is not None:
                        base = _min([base, _float(ceiling)])
            else:
                base = _float(declared_demand)
            if continuous_in.profile is not None:
                base = {
                    "op": "mul",
                    "args": [base, _var(self.name, f"{flow_name}_in_profile")],
                }
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
        """The continuous out-flows: the rate their caps leave, the
        fraction their taps take off, the profile scaling them, their
        three channels, the out port carrying the per-connection
        channels, and what they could deliver.

        What could be delivered and what was actually made are two
        equations, not one: they differ as soon as a rule has several
        outputs, and it is the second that the allocation distributes.

        A tapped output publishes a third quantity, ``{flow}_pre_tap_*``,
        which is what it carried before its taps drew on it. Named rather
        than inlined for two reasons that are one: the receiving output
        reads it to know what it was handed, so both sides of a routed
        tap divide the same number, and what left the stream is legible
        beside what stayed on it."""
        me = self.name
        produced_outputs = self._produced_outputs()
        for continuous_out in self.flows_continuous_out:
            flow_name = continuous_out.name
            # muscadet's shared `{flow}_out_rate`: the derating endpoint,
            # created with the flow and left at its nominal 1 by this
            # layer, so a mode that derates the flow has one place to
            # write and nothing to fight over.
            variables.append(_float_attribute(f"{flow_name}_out_rate", NOMINAL_RATE))
            # One variable per standalone mode derating this output, at
            # nominal until its mode writes it.
            for name in self._external_deratings_on(flow_name):
                variables.append(_float_attribute(name, NOMINAL_RATE))
            if self._derated(flow_name):
                # What the failure modes capping this output leave of it,
                # folded by minimum with the shared rate above.
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
            if self._taps_on(flow_name):
                # What the failure modes tapping this output leave of it
                # once each has taken its fraction: a separate channel
                # from the cap above and never folded into it, the two
                # composing differently and meaning opposite things.
                variables.append(
                    _float_attribute(f"{flow_name}_tap_rate", NOMINAL_RATE)
                )
                equations.append(
                    {
                        "target": f"{flow_name}_tap_rate",
                        "kind": "explicit",
                        "expr": self._tap_rate_expr(flow_name),
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
            if continuous_out.publish_rate is not None:
                # On a port of its own, carrying whichever of the two
                # quantities was declared. Swept after the one it
                # reports, which the order below places.
                variables.append(_float_attribute(_rate_alias(flow_name), 0.0))
                ports.append(
                    {
                        "name": f"{_rate_alias(flow_name)}_out",
                        "dir": "out",
                        "attr": _rate_alias(flow_name),
                    }
                )
                equations.append(
                    {
                        "target": _rate_alias(flow_name),
                        "kind": "explicit",
                        "expr": _var(
                            me,
                            f"{flow_name}_fed_out"
                            if continuous_out.publish_rate == RATE_DELIVERED
                            else f"{flow_name}_capability_out",
                        ),
                    }
                )
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
            replaced_by_a_volume = False
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
            elif self._passes_through(flow_name):
                # What could arrive could cross, scaled by the output's
                # own factors as any output is.
                nominal = {
                    "op": "mul",
                    "args": factors
                    + [_clamped_at_zero(_var(me, f"{flow_name}_capability_in"))],
                }
            elif self._tapped_into(flow_name):
                # An output that receives: what it carries is what the
                # taps routed into it took off their own streams, and
                # nothing else. `_refuse_ill_formed_taps` is what
                # guarantees "nothing else", a receiver being allowed no
                # production of its own.
                nominal = _sum(self._tap_delivery(flow_name, "capability"))
            elif continuous_out.capability_expr is not None:
                nominal = continuous_out.capability_expr
            else:
                nominal = {
                    "op": "mul",
                    "args": factors + [_float(continuous_out.var_fed_default)],
                }
                # A volume that TRANSITS and whose component declares no
                # production of its own is the source of what this output
                # delivers, so it replaces the rate rather than being
                # capped by it: the product above is identically zero
                # whatever the deratings scale it by, and a ceiling of
                # zero would leave the output delivering nothing however
                # full the volume was. Every capacity muscadet exports
                # arrives exactly so.
                #
                # THREE conditions, and no two of them would do.
                #
                # A rate that says something is a ceiling the volume
                # respects, transit or not. A zero rate on a volume that
                # declares no through-path is a SHUT OUTLET, a model
                # somebody means: `h2_battery_stop` of the cross-
                # validation corpus is a battery stopped by a rate of
                # zero that must go on filling and deliver nothing. And a
                # volume the component gives no way IN has no
                # through-path to use whatever it declares -- three
                # documents of that corpus hold one, and reading them as
                # buffers published an unbounded capability where they
                # had published none.
                held = self._capacity_of(flow_name) or _NO_CAPACITY
                replaced_by_a_volume = (
                    continuous_out.var_fed_default == 0.0
                    and held.transmits
                    and any(
                        declared.name == flow_name
                        for declared in self.flows_continuous_in
                    )
                )
            if continuous_out.max_rate is not None:
                # A declared ceiling is the equipment's, not the
                # declaration's, so it outlives the volume replacing the
                # rate underneath it.
                replaced_by_a_volume = False
            nominal = self._bounded_by_max_rate(continuous_out, nominal)
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
                # A pair moved a computed quantity onto the stream: what
                # the flow carries is no longer the bare declared rate,
                # so the volume bounds it rather than replacing it.
                replaced_by_a_volume = False
            equations.append(
                {
                    "target": f"{flow_name}_capability_out",
                    "kind": "explicit",
                    "expr": self._tap_split(
                        flow_name,
                        "capability",
                        self._capacity_bounded_capability(
                            flow_name, nominal, replaces=replaced_by_a_volume
                        ),
                        variables,
                        equations,
                    ),
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
            elif self._passes_through(flow_name):
                # What crossed, as opposed to what could: what actually
                # arrived, scaled by the output's factors. A scarce supply
                # is handed on as it is, and the allocation distributes
                # this rather than the capability.
                variables.append(_float_attribute(f"{flow_name}_produced_out", 0.0))
                equations.append(
                    {
                        "target": f"{flow_name}_produced_out",
                        "kind": "explicit",
                        "expr": self._tap_split(
                            flow_name,
                            "produced",
                            self._bounded_by_max_rate(
                                continuous_out,
                                {
                                    "op": "mul",
                                    "args": factors
                                    + [
                                        _clamped_at_zero(
                                            _var(me, f"{flow_name}_fed_in")
                                        )
                                    ],
                                },
                            ),
                            variables,
                            equations,
                        ),
                    }
                )
            elif self._tapped_into(flow_name) and flow_name in produced_outputs:
                # A receiver whose sources publish what they actually
                # made: what it was handed is a fraction of that, never
                # of what they merely could have made.
                variables.append(_float_attribute(f"{flow_name}_produced_out", 0.0))
                equations.append(
                    {
                        "target": f"{flow_name}_produced_out",
                        "kind": "explicit",
                        "expr": self._tap_split(
                            flow_name,
                            "produced",
                            self._capacity_bounded_capability(
                                flow_name,
                                _sum(self._tap_delivery(flow_name, "produced")),
                            ),
                            variables,
                            equations,
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
                produced: dict[str, Any] = self._bounded_by_max_rate(
                    continuous_out,
                    {
                        "op": "mul",
                        "args": factors + [_sum(rule_production[flow_name])],
                    },
                )
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
                held = self._beside_rules(flow_name, "out")
                if held is None:
                    delivered = self._capacity_bounded_capability(flow_name, produced)
                else:
                    # Downstream of the rules, what they made FILLS the
                    # volume, named so the volume's balance reads the same
                    # quantity, and the consumers are served from the
                    # volume: their whole request while it is stocked,
                    # what the rules are making once it is empty
                    # (muscadet's `serve_limit`, with the rule in the
                    # inlet's place).
                    made_out = f"{flow_name}_made_out"
                    variables.append(_float_attribute(made_out, 0.0))
                    equations.append(
                        {"target": made_out, "kind": "explicit", "expr": produced}
                    )
                    request = self._bounded_by_max_rate(
                        continuous_out,
                        _min(
                            [
                                _var(me, f"{flow_name}_demand_out"),
                                self._capacity_ceiling(held, flow_name),
                            ]
                        ),
                    )
                    delivered = {
                        "op": "if",
                        "cond": self._held_empty(held, flow_name),
                        "then": _min([request, _var(me, made_out)]),
                        "otherwise": request,
                    }
                equations.append(
                    {
                        "target": f"{flow_name}_produced_out",
                        "kind": "explicit",
                        "expr": self._tap_split(
                            flow_name,
                            "produced",
                            delivered,
                            variables,
                            equations,
                        ),
                    }
                )
