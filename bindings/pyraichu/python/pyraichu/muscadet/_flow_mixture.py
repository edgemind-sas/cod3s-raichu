"""Mixtures of a component and the discharge or run command that gates them."""

from __future__ import annotations

import math
from typing import Any

from ._expr import _bool, _clamped_at_zero, _float_attribute, _min, _sum, _var
from ._records import _Capacity, _MixtureIn, _RuleOperand
from ._vocabulary import (
    _CMP_OPS,
    _ORDERING_OPS,
    SERVE_COND_INNER_MODE_DEFAULT,
    SERVE_COND_INNER_MODES,
)


class _MixturesMixin:
    """Methods of :class:`ObjFlow` mixtures of a component and the discharge or run command that gates them."""

    # --- the discharge command ------------------------------------------

    def _parse_command(
        self,
        where: str,
        declared: Any,
        negates: list | None,
        compares: list | None,
        what: str = "discharge command",
        key: str = "serve_cond",
    ) -> list[list[_RuleOperand]]:
        """A discharge command, as groups of parsed guard operands; a
        machine's run command too (R52), which is the same vocabulary
        under `key` = ``run_cond``.

        The shape follows muscadet's: a list of groups, each group a list
        of operands, and a bare name or mapping is the one-operand
        one-group short form. A FLAT list is read as one group per
        element, which is what the layer underneath reads a flat
        production condition as too, so `["a", "b"]` is two clauses of one
        operand rather than one clause of two.

        `negates` and `compares` are the parallel matrices muscadet lifts
        a negation and a comparison out of its operands into. They are
        folded back onto the operand they belong to rather than carried
        beside it: one condition has one spelling here, and a matrix cell
        contradicting what its own operand says is refused instead of one
        of the two quietly winning.
        """
        if not declared:
            return []
        if isinstance(declared, (str, dict)):
            declared = [declared]
        if not isinstance(declared, (list, tuple)):
            raise ValueError(
                f"{where} carries a {what} that is neither a name, "
                f"an operand nor a list: {declared!r}"
            )

        groups: list[list[_RuleOperand]] = []
        for row, group in enumerate(declared):
            specs = group if isinstance(group, (list, tuple)) else [group]
            if not specs:
                raise ValueError(
                    f"{where} carries an empty group in its {what}; "
                    "a group that reads nothing says nothing about when it "
                    "holds"
                )
            operands = []
            for column, spec in enumerate(specs):
                operands.append(
                    self._parse_operand(
                        f"{where}, {what}",
                        self._command_operand_spec(
                            where, spec, row, column, negates, compares, what, key
                        ),
                    )
                )
            groups.append(operands)
        return groups

    @staticmethod
    def _command_operand_spec(
        where: str,
        spec: Any,
        row: int,
        column: int,
        negates: list | None,
        compares: list | None,
        what: str = "discharge command",
        key: str = "serve_cond",
    ) -> Any:
        """One operand's declaration, with its cells of the two parallel
        matrices folded in, refusing a cell that contradicts it.

        Folded into the DECLARATION rather than onto the parsed operand,
        so a comparison lifted out into a matrix goes through the same
        validation an inline one does -- the operator, the value, and the
        band that widens it -- instead of a second copy of it here.

        muscadet stores a resolved condition as its operands plus the two
        matrices, so a document written against that model may carry the
        negation and the comparison there rather than inline. Both
        spellings declare one thing; a cell saying something else than its
        own operand would leave which of the two holds to the order they
        happened to be read in.
        """

        def cell(matrix: list | None) -> Any:
            try:
                return matrix[row][column]  # type: ignore[index]
            except (IndexError, KeyError, TypeError):
                return None

        # A negation matrix is SPARSE -- muscadet attaches it only once
        # some operand is negated -- so a False cell says nothing and
        # never denies an inline `negate`.
        negated = bool(cell(negates))
        compare = cell(compares)
        if not negated and compare is None:
            return spec

        merged = {"name": spec} if isinstance(spec, str) else spec
        if not isinstance(merged, dict):
            raise ValueError(
                f"{where} carries a {what} operand that is neither "
                f"a name nor a mapping: {spec!r}"
            )
        merged = dict(merged)
        if negated:
            merged["negate"] = True
        if compare is not None:
            if not isinstance(compare, dict) or "op" not in compare:
                raise ValueError(
                    f"{where} carries the {what} comparison "
                    f"{compare!r} at [{row}][{column}]; a cell of "
                    f"`{key}_compare` carries `op` and `value`"
                )
            declared = merged.get("op")
            if declared is not None and (
                declared != compare["op"]
                or float(merged.get("value")) != float(compare.get("value"))
            ):
                raise ValueError(
                    f"{where} compares `{merged.get('name')}` inline as "
                    f"`{declared} {merged.get('value')}` and in "
                    f"`{key}_compare` as `{compare['op']} "
                    f"{compare.get('value')}`; one condition has one "
                    "comparison, so the two spellings state the same one or "
                    "only one of them is written"
                )
            merged["op"] = compare["op"]
            merged["value"] = compare.get("value")
        return merged

    def _resolve_command(self, carrier: Any) -> None:
        """Resolve a recorded command -- a capacity's discharge command, a
        mixture machine's run command -- against what this component
        declares, refusing a misspelling here rather than letting it reach
        the engine as a dangling read.

        The same two diagnostics a rule guard gets, for the same reasons:
        an operand naming nothing is refused, and an EQUALITY on a
        continuously-evolving quantity is refused because the transition
        carrying it is boundary and a crossing is located on an ordering
        comparison.
        """
        where = f"ObjFlow `{self.name}`: {carrier.command_owner}"
        for group in carrier.command:
            for operand in group:
                continuous = self._operand_reads_continuously(where, operand)
                if (
                    operand.op in _CMP_OPS
                    and continuous
                    and operand.op not in _ORDERING_OPS
                ):
                    raise ValueError(
                        f"{where} commands its {carrier.command_noun} on the "
                        f"continuous quantity `{operand.name}` with "
                        f"`{operand.op}`; the gate transition carrying it is "
                        "boundary, and a crossing is located on an ordering "
                        f"comparison ({', '.join(_ORDERING_OPS)})"
                    )

    def _command_reads_continuously(self, carrier: Any) -> bool:
        """Whether the command compares a continuously-evolving quantity,
        which is what makes its gate a located crossing rather than a
        discrete decision."""
        where = f"ObjFlow `{self.name}`: {carrier.command_owner}"
        return any(
            operand.op is not None
            and self._operand_reads_continuously(where, operand)
            for group in carrier.command
            for operand in group
        )

    def _command_expr(self, carrier: Any) -> dict[str, Any]:
        """A command -- a capacity's discharge, a machine's run -- as a
        boolean expression.

        The two levels are combined as the carrier's inner mode says, and
        the nesting is written as it stands rather than normalised: the
        gate is an automaton guard, which carries an `and` of `or`s
        directly, where a discrete production condition has to be
        expanded into a disjunction of flow-name conjunctions because
        that is all the layer underneath it reads.
        """
        outer, inner = (
            ("and", "or")
            if carrier.command_inner_mode == "or"
            else ("or", "and")
        )
        return _bool(
            outer,
            [
                _bool(inner, [self._rule_operand_expr(operand) for operand in group])
                for group in carrier.command
            ],
        )

    def _command_negation(self, carrier: Any) -> dict[str, Any]:
        """The command denied, by De Morgan over both levels and with
        every comparison **complemented** rather than wrapped in a `not`,
        for the reason :data:`_NEGATED_OPS` states.

        Denied at the far edge of every band it carries: this is the
        guard of the transition leaving the state the band holds, so a
        command entered at `value` is left at `release` and holds
        between the two.
        """
        outer, inner = (
            ("or", "and")
            if carrier.command_inner_mode == "or"
            else ("and", "or")
        )
        return _bool(
            outer,
            [
                _bool(
                    inner,
                    [
                        self._rule_operand_negation(operand, releasing=True)
                        for operand in group
                    ],
                )
                for group in carrier.command
            ],
        )

    def _command_gate_automaton(self, carrier: Any) -> dict[str, Any]:
        """The two-state automaton carrying a command: a capacity's
        discharge (``{capacity}_serve``) or a machine's run (``{group}_run``).

        An automaton rather than a branch inside the served quantity, and
        for the two reasons a rule set's mode is one: the decision is
        frozen between crossings, and a transition carrying a continuous
        threshold is declared **boundary**, so the solver stops AT the
        crossing instead of noticing it at the following step. muscadet
        compiles the same condition the same way, under the same
        ``{capacity}_serve`` tag (`Capacity.add_serve_cond_automata`).

        It starts in its closed state, which is not a guess about the command:
        the guards read quantities the initial fixpoint resolves, so the
        location that holds at t = 0 is entered there, exactly as a rule
        set's mode enters its own.
        """
        on, off = carrier.gate_states
        located = self._command_reads_continuously(carrier)

        def transition(name: str, source: str, target: str, guard: dict) -> dict:
            entry: dict[str, Any] = {
                "name": name,
                "source": source,
                "targets": [target],
                "guard": guard,
            }
            if located:
                entry["distrib"] = "boundary"
            else:
                entry["distrib"] = "inst"
                entry["probs"] = []
            return entry

        return {
            "name": carrier.gate,
            "states": [on, off],
            "init": off,
            "transitions": [
                transition(
                    f"{carrier.gate}_opens",
                    off,
                    on,
                    self._command_expr(carrier),
                ),
                transition(
                    f"{carrier.gate}_shuts",
                    on,
                    off,
                    self._command_negation(carrier),
                ),
            ],
        }

    def _mixed_volume(self, flow: str) -> _Capacity | None:
        """The volume serving output `flow` as a MIXTURE, or ``None``.

        muscadet's `draw_from_capacity` (measured on 5.6.0) does not
        serve each output of a volume holding several flows from its own
        stock. It pools what the consumers ask beyond what transits and
        hands each held flow a share of that excess in proportion to its
        raw content (:meth:`_pooled_serve`). That is the reading of a
        volume **downstream of the transfer** (``side="out"``) holding
        more than one flow, and of no other: a single-flow volume has
        nothing to mix, one upstream of the transfer serves each flow
        from its own stock, and a volume beside this component's rules
        is served by the rules (:meth:`_beside_rules`), none of whose
        held flows may be carried by a rule here for the pooled draw to
        apply. An output whose delivery something else computes (a rule,
        a conduit, a pass-through, a routed tap) is not the volume's to
        pool either."""
        capacity = self._capacity_of(flow)
        if capacity is None or len(capacity.flows) < 2 or capacity.side != "out":
            return None
        if not any(declared.name == flow for declared in self.flows_continuous_out):
            return None
        if any(
            held in rule_set.consumed() or held in rule_set.produced()
            for rule_set in self.rule_sets
            for held in capacity.flow_names
        ):
            return None
        if flow in self._produced_outputs():
            return None
        return capacity

    def _mixed_outputs(self) -> dict[str, _Capacity]:
        """Per output served as a mixture, the volume serving it."""
        return {
            flow.name: capacity
            for flow in self.flows_continuous_out
            for capacity in [self._mixed_volume(flow.name)]
            if capacity is not None
        }

    def _beyond_attribute(self, capacity: _Capacity) -> str:
        """Name of the variable holding what the consumers of a mixed
        volume ask beyond what transits it (:meth:`_pooled_serve`)."""
        return f"{capacity.name}_beyond"

    def _pooled_excess(self, capacity: _Capacity) -> dict[str, Any]:
        """``beyond``: what the consumers of `capacity` ask of it over and
        above what transits it, summed over every held output,
        ``sum_g max(req_g - transit_g, 0)``.

        ``req_g`` is the demand published on output ``g``, uncapped by
        the stock, exactly as muscadet publishes it: the capability band
        is left as it was (a stocked flow still offers its ceiling), so
        the consumers size their demand on it and the pooled draw only
        decides what is handed out."""
        mixed = self._mixed_outputs()
        return _sum(
            [
                _clamped_at_zero(
                    {
                        "op": "sub",
                        "lhs": _var(self.name, f"{flow}_demand_out"),
                        "rhs": self._transit(flow),
                    }
                )
                for flow in capacity.flow_names
                if mixed.get(flow) is capacity
            ]
        )

    def _pooled_serve(self, capacity: _Capacity, flow: str) -> dict[str, Any]:
        """What a mixed volume hands the consumers of `flow`: muscadet
        5.6.0 `draw_from_capacity`, the branch without a declared
        mixture, measured term for term.

        While `flow` is stocked, what transits it plus its share of the
        pooled excess, the share being its RAW content over the raw total
        (``{capacity}_ratio_{flow}``: the weights change the fill and not
        the split, measured identical with a hydrogen weight of 2), and
        never more than the capability, which carries the ceiling, the
        discharge command and the output's own factors. The allocation
        caps it by what was asked, which is the ``min(req_f, ...)`` of
        the reference. Once `flow` has run out, the capability itself,
        unchanged: an empty constituent passes on what arrives, as it
        did before the mixture was read.

        Consequence worth stating: hydrogen asked alone out of a room
        of air leaves at ``req * h / (h + a)`` rather than at ``req``,
        so it is never exhausted in finite time (``h - h0 + a ln(h/h0)
        = -req t`` for a constant ``a``)."""
        me = self.name
        capability = _var(me, f"{flow}_capability_out")
        if capacity.serves_mixture:
            # Drawn by a mixture group: the request is already composed
            # (``R . m_f / S``), and what transits enters the composition
            # rather than passing straight on (muscadet 5.6.0,
            # `draw_from_capacity`, the branch serving a mixture).
            return _min([capability, _var(me, f"{flow}_demand_out")])
        share = {
            "op": "mul",
            "args": [
                _var(me, self._beyond_attribute(capacity)),
                _var(me, f"{capacity.name}_ratio_{flow}"),
            ],
        }
        stocked = _min(
            [capability, {"op": "add", "args": [self._transit(flow), share]}]
        )
        return {
            "op": "if",
            "cond": self._held_empty(capacity, flow),
            "then": capability,
            "otherwise": stocked,
        }

    def add_mixture_in(
        self,
        name: str,
        flows: list[str],
        flow_rate: float,
        run_cond: Any = None,
        run_cond_negate: list | None = None,
        run_cond_compare: list | None = None,
        run_cond_inner_mode: str | None = None,
    ) -> None:
        """Declare continuous inputs drawn TOGETHER at one volumetric rate
        (muscadet `add_mixture_in`, R51): a ventilation extractor, a pump on
        a line carrying a mixture.

        `flow_rate` is ONE rate for the whole group and a VOLUME per unit of
        time: what leaves the volume drawn from per constituent is
        ``flow_rate . m_f / sum_g (m_g . w_g)``, fixed by that volume's
        composition and weights, so the volume extracted is exactly
        `flow_rate`. The flows must already be declared as continuous
        inputs; which volume composes for the group is settled when the
        model is generated, from the connections, and refused by name when
        the group's flows do not all arrive from one capacity of one
        producer serving nobody else.

        The rate is published as ``{name}_flow_rate`` (muscadet's R52), a
        variable a failure mode clamps by name and a repair releases back to
        `flow_rate`. `run_cond` commands the machine on and off in the
        vocabulary of a capacity's `serve_cond`, with the same two parallel
        matrices and inner mode; empty, the machine runs whenever its rate
        allows."""
        where = f"ObjFlow `{self.name}`: mixture group `{name}`"
        if any(existing.name == name for existing in self.mixtures):
            raise ValueError(f"{where} is already declared")
        flows = [flows] if isinstance(flows, str) else list(flows or [])
        if not flows:
            raise ValueError(f"{where} draws no flow")
        declared = {flow.name for flow in self.flows_continuous_in}
        missing = [flow for flow in flows if flow not in declared]
        if missing:
            raise ValueError(
                f"{where} draws {missing}, which this component declares as no "
                "continuous input (declare the inputs first)"
            )
        outputs = {out.name for out in self.flows_continuous_out}
        carried = sorted(flow for flow in flows if flow in outputs)
        if carried:
            raise ValueError(
                f"{where} draws {carried}, also a continuous OUTPUT of this "
                "component. A machine drawing a mixture and carrying it onward "
                "destroys matter as soon as its outlet asks for less than it "
                "draws, which muscadet refuses by the same reason; declare a "
                "terminal extractor"
            )
        grouped = {flow for group in self.mixtures for flow in group.flows}
        twice = [flow for flow in flows if flow in grouped]
        if twice or len(set(flows)) != len(flows):
            raise ValueError(
                f"{where} draws a flow another group, or itself, already draws: "
                f"{twice or flows}"
            )
        if (
            isinstance(flow_rate, bool)
            or not isinstance(flow_rate, (int, float))
            or not math.isfinite(flow_rate)
            or flow_rate < 0
        ):
            raise ValueError(
                f"{where} declares `flow_rate`={flow_rate!r}; it is one finite, "
                "non-negative volume per unit of time for the whole group"
            )
        inner_mode = (
            SERVE_COND_INNER_MODE_DEFAULT
            if run_cond_inner_mode is None
            else run_cond_inner_mode
        )
        if inner_mode not in SERVE_COND_INNER_MODES:
            raise ValueError(
                f"{where} declares `run_cond_inner_mode`={run_cond_inner_mode!r}; "
                "a run command combines its two levels one way or the other, so "
                f"it is {' or '.join(repr(mode) for mode in SERVE_COND_INNER_MODES)}"
            )
        command = self._parse_command(
            where,
            run_cond,
            run_cond_negate,
            run_cond_compare,
            what="run command",
            key="run_cond",
        )
        group = _MixtureIn(
            name=name,
            flows=flows,
            flow_rate=float(flow_rate),
            run_cond=command,
            run_cond_inner_mode=inner_mode,
        )
        self.mixtures.append(group)
        try:
            self._resolve_command(group)
        except ValueError:
            self.mixtures.remove(group)
            raise

    def _fed_by_a_mixture(self, flow: str) -> bool:
        """Whether the rules here consume `flow` as another component's
        mixed volume serves it (:meth:`_mixed_volume`).

        Such an input may receive LESS than the rules asked while its
        supplier is still stocked: the volume hands out a share of the
        pooled excess, not the request. A rule producing at its scale
        regardless would make what it never received, which is what the
        draw below prevents. A flow this component holds in a volume of
        its own is not concerned: that volume is what the rules draw
        from, and it absorbs the difference."""
        return (
            flow in self._mixture_sources
            and self._capacity_of(flow) is None
            and any(flow in rule_set.consumed() for rule_set in self.rule_sets)
        )

    def _build_mixtures(self, variables: list[dict], automata: list[dict]) -> None:
        """Mixture groups (muscadet's R52): the published rate of each, and
        the run gate of a commanded one.

        The rate is a variable with NO equation: a failure mode is its only
        writer, and its initial value is the declared `flow_rate`, which is
        what a held effect releases back to. The gate is posted as an
        automaton, beside the rate rather than inside the demand, for the
        reason a capacity's discharge gate is: a threshold inlined in a
        quantity closes a cycle among variables alone, which nobody reports.
        What the group asks of the volume is written by
        :meth:`System._resolve_mixture_groups`, which knows the volume."""
        for group in self.mixtures:
            variables.append(_float_attribute(group.rate_attribute, group.flow_rate))
            if group.commanded:
                automata.append(self._command_gate_automaton(group))
