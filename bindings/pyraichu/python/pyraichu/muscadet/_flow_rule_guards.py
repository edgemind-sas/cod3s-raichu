"""Transformation rules of a component: parsing, operands and guards."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ._expr import _bool, _boolean_comparison, _float, _state_active, _var
from ._records import _as_coefficients, _as_operand_list, _Rule, _RuleOperand, _RuleSet
from ._vocabulary import (
    _CMP_OPS,
    _NEGATED_OPS,
    _NO_RULE,
    _ORDERING_OPS,
    RULE_KEYS,
    RULE_OPERAND_KEYS,
    RUN_GATE_STATES,
    SERVE_GATE_STATES,
)


class _RuleGuardsMixin:
    """Methods of :class:`ObjFlow` transformation rules of a component: parsing, operands and guards."""

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
        demand where another set also produces into it (R13). muscadet
        has no field for it: it is a documented RAICHU extension. The
        shares of one output are normalised, so ``3`` and ``1`` split it
        three to one. An output **none** of its producers apportions is
        served in declaration order instead, each set asked only what the
        sets before it left unserved, so the component never draws more
        than it delivers (muscadet asks every set for the whole demand and
        drops the surplus). A share on some producers and not on the
        others is refused.

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
        unknown = sorted(set(spec) - RULE_KEYS)
        if unknown:
            raise ValueError(
                f"{where} carries unknown keys {unknown} (expected "
                f"{', '.join('`' + key + '`' for key in sorted(RULE_KEYS))})"
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
        unknown = sorted(set(spec) - RULE_OPERAND_KEYS)
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
            release=spec.get("release"),
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
        if operand.release is not None:
            self._check_band(where, operand)
        return operand

    @staticmethod
    def _check_band(where: str, operand: _RuleOperand) -> None:
        """Refuse a hysteresis band that is not one.

        Every refusal here is a band that would look declared and behave
        as though it were not: silently reverting to a single threshold is
        the one outcome worth ruling out, since the reason for declaring a
        band at all is a mode that chatters without it."""
        if operand.op is None:
            raise ValueError(
                f"{where} declares `release`={float(operand.release):g} on "
                f"`{operand.name}` without comparing it; a band widens a "
                "comparison, so it needs `op` and `value`"
            )
        if operand.op not in _ORDERING_OPS:
            raise ValueError(
                f"{where} declares a band on `{operand.name} {operand.op}`; a "
                f"band needs an ordering comparison ({', '.join(_ORDERING_OPS)}), "
                "since it says which side the rule is left on"
            )
        operand.release = float(operand.release)
        if operand.release == operand.value:
            raise ValueError(
                f"{where} declares `release` equal to `value` "
                f"({float(operand.value):g}) on `{operand.name}`; that is a "
                "band of zero width, which is the single threshold it was "
                "meant to replace: widen it or drop the key"
            )
        rising = operand.op in (">", ">=")
        if rising and operand.release > operand.value:
            raise ValueError(
                f"{where} enters on `{operand.name} {operand.op} "
                f"{float(operand.value):g}` and would leave at "
                f"{float(operand.release):g}, above it; a rule entered on the "
                "way up is left on the way down, so `release` sits BELOW "
                "`value`"
            )
        if not rising and operand.release < operand.value:
            raise ValueError(
                f"{where} enters on `{operand.name} {operand.op} "
                f"{float(operand.value):g}` and would leave at "
                f"{float(operand.release):g}, below it; a rule entered on the "
                "way down is left on the way up, so `release` sits ABOVE "
                "`value`"
            )

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
            if capacity.commanded:
                automata[capacity.gate] = list(SERVE_GATE_STATES)
        for group in self.mixtures:
            if group.commanded:
                automata[group.gate] = list(RUN_GATE_STATES)
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
        """One guard operand as a boolean expression.

        A comparison against a **two-valued** read is settled here rather than
        emitted (:func:`_boolean_comparison`): this engine's ordering
        comparison is typed, so `ctrl > 0.5` on a discrete flow would build a
        model that fails at its first evaluation where muscadet reads it as
        `float(var_fed) > 0.5`. It is the same vocabulary on both sides of the
        discrete/continuous interoperation, so it is resolved in one place
        rather than in the direction that happened to need it first."""
        where = f"ObjFlow `{self.name}`"
        read, _ = self._operand_read(where, operand)
        if operand.op is not None:
            if not self._operand_reads_continuously(where, operand):
                return _boolean_comparison(read, operand.op, operand.value)
            return {
                "op": "cmp",
                "cmp": _CMP_OPS[operand.op],
                "lhs": read,
                "rhs": _float(operand.value),
            }
        if operand.negate:
            return {"op": "bool", "bool_op": "not", "args": [read]}
        return read

    def _rule_operand_negation(
        self, operand: _RuleOperand, releasing: bool = False
    ) -> dict[str, Any]:
        """The operand denied, with a comparison **complemented** rather
        than wrapped in a `not`: see :data:`_NEGATED_OPS`.

        `releasing` denies the operand at its band's far edge instead of
        at the edge it was entered on. It is set on the rule the mode is
        currently in and on no other: an earlier rule the mode might move
        UP to is still tested at its own entry threshold, so a band widens
        the rule that holds it and never the ones competing for it."""
        if operand.op is not None:
            threshold = (
                operand.value
                if not releasing or operand.release is None
                else operand.release
            )
            return self._rule_operand_expr(
                replace(operand, op=_NEGATED_OPS[operand.op], value=threshold)
            )
        return self._rule_operand_expr(replace(operand, negate=not operand.negate))

    def _rule_guard_negation(
        self, rule: _Rule, releasing: bool = False
    ) -> dict[str, Any]:
        """A rule's guard denied: the disjunction of its denied operands,
        by De Morgan, so every comparison in it is complemented."""
        return _bool(
            "or",
            [
                self._rule_operand_negation(operand, releasing)
                for operand in rule.cond
            ],
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
        self, rule_set: _RuleSet, target: str, source: str | None = None
    ) -> tuple[dict[str, Any], bool]:
        """When the mode enters `target`, and whether that entry is a
        located crossing.

        Rules are ordered, so a guarded rule is selected when its own
        guard holds and no earlier one does; the default (or the "no
        rule" mode) is selected when no guard holds at all.

        `source` is the mode being left. It matters only to a rule that
        declares a **band**: the rule the mode is in is denied at its
        release threshold rather than at its entry one, which is what
        makes the mode hold inside the band instead of switching on the
        first crossing."""
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
        held = (
            next(
                (
                    rule
                    for index, rule in guarded
                    if rule_set.label(index) == source
                ),
                None,
            )
            if source is not None
            else None
        )
        for rule in earlier:
            terms.append(self._rule_guard_negation(rule, releasing=rule is held))
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
                guard, located = self._rule_selection_guard(
                    rule_set, target, source
                )
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
