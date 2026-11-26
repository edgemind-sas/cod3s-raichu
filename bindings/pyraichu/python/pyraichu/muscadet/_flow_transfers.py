"""Transfer pairs of a component: declaration, crossings and emission."""

from __future__ import annotations

from typing import Any

from ._expr import _clamped_at_zero, _float, _float_attribute, _min, _negated, _var
from ._records import _level_alias, _Transfer
from ._vocabulary import _CONTINUITY_MESSAGE, _TRANSFER_CLASS


class _TransfersMixin:
    """Methods of :class:`ObjFlow` transfer pairs of a component: declaration, crossings and emission."""

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

    def _conduit_on(self, flow: str) -> _Transfer | None:
        """The conduit metering `flow`, or ``None``. A flow is metered by
        at most one, :meth:`add_transfer` refusing a second claim on it."""
        for pair in self.transfers:
            if pair.is_conduit and pair.source == flow:
                return pair
        return None

    def _passes_through(self, flow: str) -> bool:
        """Whether `flow` crosses this component unchanged, input to output.

        muscadet's identity transfer (`get_identity_transfer_flows`, its
        R31): a continuous flow declared on BOTH sides and named by no
        rule set is handed on as it arrives, so a plain pipe needs no
        ceremonial same-in-same-out rule. Without it the output published
        its declared rate, zero on such a component, and everything
        downstream read zero.

        Every other way of saying what the component does with the flow
        wins, because each one already is that statement: a rule set
        consuming or producing it, a conduit metering it, a volume holding
        it (transit or not, which `transmits` decides), a tap routing into
        it, a declared equation for either side. A stream a two-stream pair
        sits on is left out as well: its delta would have to be taken on
        what crossed, which this layer does not yet name."""
        inflow = next(
            (
                declared
                for declared in self.flows_continuous_in
                if declared.name == flow
            ),
            None,
        )
        out = next(
            (
                declared
                for declared in self.flows_continuous_out
                if declared.name == flow
            ),
            None,
        )
        if inflow is None or out is None:
            return False
        if inflow.demand_expr is not None or out.capability_expr is not None:
            return False
        if any(
            flow in rule_set.consumed() or flow in rule_set.produced()
            for rule_set in self.rule_sets
        ):
            return False
        return (
            self._conduit_on(flow) is None
            and self._capacity_of(flow) is None
            and not self._tapped_into(flow)
            and not self._has_transfer_delta(flow)
        )

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

    def _potential_expr(self, potential: tuple[str, Any, Any]) -> dict[str, Any]:
        """One resolved potential, as the expression reading it."""
        kind, first, second = potential
        if kind == "const":
            return _float(first)
        return _var(self.name, _level_alias(first, second))

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
