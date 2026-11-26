"""Discrete (boolean) flows of a component and their production conditions."""

from __future__ import annotations

from typing import Any

from ._expr import _bool, _state_active, _var
from ._records import _FlowIn, _FlowOut, _RuleOperand
from ._vocabulary import (
    _ORDERING_OPS,
    _prod_cond_inner_mode,
    _tempo_law,
    prod_cond_disjunction,
    PROD_COND_INNER_MODE_DEFAULT,
    PROD_COND_OPERAND_KEYS,
    prod_cond_unconditional,
    RULE_OPERAND_KEYS,
)


class _DiscreteFlowsMixin:
    """Methods of :class:`ObjFlow` discrete (boolean) flows of a component and their production conditions."""

    def add_flow_in(
        self, name: str, logic: str | int = "or", var_in_default: bool | None = None
    ) -> None:
        """Declare an incoming flow aggregated with `logic`
        (``"or"``, ``"and"`` or an integer k for k-out-of-n).

        `var_in_default` is what the input reads while nothing is
        connected to it (``True`` marks an always-fed boundary input).
        It is carried on the FOUR logics, k-out-of-n included: dropping
        it there used to be silent, so a caller who had declared a
        boundary input got the opposite answer and no diagnostic."""
        if isinstance(logic, int):
            self.flows_in.append(
                _FlowIn(name=name, logic="k", k=logic, var_in_default=var_in_default)
            )
        else:
            self.flows_in.append(
                _FlowIn(name=name, logic=logic, var_in_default=var_in_default)
            )

    def add_flow_out(
        self,
        name: str,
        var_prod_default: bool = False,
        var_prod_cond: list[Any] | None = None,
        var_is_active_default: bool | None = None,
        var_fed_available_out_init: bool | None = None,
        var_fed_available_out_reset: bool | None = None,
        var_prod_cond_inner_mode: str = PROD_COND_INNER_MODE_DEFAULT,
        negate: bool = False,
    ) -> None:
        """Declare an outgoing flow, produced unconditionally
        (``var_prod_default=True``) or when the declared condition holds.

        `var_prod_cond` is read as muscadet reads it: a list of clauses (a
        flat list is one clause per element), combined as
        `var_prod_cond_inner_mode` says. ``"or"``, muscadet's default, is
        every clause holding, a clause holding when any of its operands does;
        ``"and"`` is any clause holding, a clause holding when all of its
        operands do. So ``[["a"], ["b"]]`` is ``a and b`` by default and
        ``a or b`` under ``"and"``, and a flat ``["a", "b"]`` is ``a and b``
        by default, as it always was. An operand is a bare flow
        name -- read on the INPUT side first, exactly as muscadet resolves one
        -- or the mapping carrying :data:`PROD_COND_OPERAND_KEYS`: `port` to
        select a side the default would not have picked (the transit pattern,
        where the component carries one name both ways), `negate` to deny the
        operand, `op` and `value` to compare what it names against a threshold
        in a rule guard's own comparison vocabulary.

        The three dormancy knobs are muscadet's: `var_is_active_default`
        adds a ``var_is_active`` factor to the delivery (the flow stays
        unfed until an effect sets it), `var_fed_available_out_init`
        seeds the availability gate, and `var_fed_available_out_reset`
        says whether that gate falls back on the seed at every step.
        Left ``None``, the generated model carries none of them.

        `negate` is muscadet's `FlowOut.negate`: the output publishes the
        negation of what it would deliver, ``not (production and active and
        available)``, the inverter of a logic chain."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                var_prod_cond_inner_mode=_prod_cond_inner_mode(
                    name, var_prod_cond_inner_mode
                ),
                var_is_active_default=var_is_active_default,
                var_fed_available_out_init=var_fed_available_out_init,
                var_fed_available_out_reset=var_fed_available_out_reset,
                negate=bool(negate),
            )
        )

    # --- production conditions ------------------------------------------

    def _parse_prod_cond(
        self, where: str, declared: Any, inner_mode: str = PROD_COND_INNER_MODE_DEFAULT
    ) -> list[list[_RuleOperand]]:
        """A declared production condition, as the groups of resolved
        operands the expression reads: groups OR-ed, operands AND-ed.

        The declaration is read as muscadet reads it: a list of clauses, a
        FLAT list being one clause per element, combined as `inner_mode`
        says (:data:`PROD_COND_INNER_MODES`), then put in disjunctive form by
        :func:`prod_cond_disjunction`. A flat list under ``"or"`` therefore
        comes out as the one conjunction it always was here, and the
        expression built from it is unchanged character for character.

        A MIXED list (groups beside bare operands) keeps its old reading, one
        group whose list element no operand parser accepts, so it is refused
        by the operand check rather than silently re-read.
        """
        if all(isinstance(group, (list, tuple)) for group in declared):
            clauses = [list(group) for group in declared]
        elif not any(isinstance(group, (list, tuple)) for group in declared):
            clauses = [[operand] for operand in declared]
        else:
            clauses = [list(declared)]
        for clause in clauses:
            # Unreachable from `_build_flows_out`, which reads a condition of
            # empty groups only through `prod_cond_unconditional` first and
            # refuses an empty group beside an operand there. Kept for any
            # other caller: an empty clause multiplied out under "or" would
            # make the condition vanish rather than fail.
            if not clause:
                raise ValueError(
                    f"{where} carries an empty group; a group that reads "
                    "nothing says nothing about when the output produces"
                )
        groups = prod_cond_disjunction(where, clauses, inner_mode)
        parsed: list[list[_RuleOperand]] = []
        for group in groups:
            specs = group if isinstance(group, (list, tuple)) else [group]
            parsed.append(
                [self._parse_prod_cond_operand(where, spec) for spec in specs]
            )
        return parsed

    def _parse_prod_cond_operand(self, where: str, spec: Any) -> _RuleOperand:
        """One production-condition operand, in the form the expression
        reads it.

        The operand vocabulary is a rule guard's minus the two keys
        :data:`PROD_COND_OPERAND_KEYS` leaves out, and it is read by the same
        parser, so a comparison is spelled once for both directions of the
        discrete/continuous interoperation.

        Two things happen here that a guard does not need.

        **A negated comparison is refused.** A comparison already yields a
        truth value, so it is denied by the opposite operator rather than
        beside it; muscadet refuses the pair at the same place
        (``muscadet.rules.check_operand_negation``).

        **A boolean operand naming a continuously-evolving quantity is
        refused**, where muscadet reads one as ``!= 0`` (its R46
        normalisation). The divergence is deliberate and neither half of it is
        free:

        * read as it stands, the quantity would be looked at whenever the
          condition happens to be evaluated, which is at DISCRETE epochs only.
          A quantity an ODE moves announces no change of its own, so the
          condition would be settled at t = 0 and never again -- measured, on
          a volume filling past its threshold with the output left unfed for
          the whole mission;
        * turned into a located crossing, it would have to become ``> 0`` or
          ``< 0``, since ``!= 0`` has no side to be entered from. Choosing
          between them means **supposing the sign of the quantity** in the
          modeller's place, on a rate that may well take both.

        So the shape is refused and the refusal says what to write: on a
        quantity that only rises, ``{"op": ">", "value": 0}`` says the same
        thing AND says which side it is entered from. The comparison it asks
        for is then boundary by the threshold automaton below, which a boolean
        read never got on either engine.
        """
        if isinstance(spec, dict):
            refused = sorted(set(spec) & (RULE_OPERAND_KEYS - PROD_COND_OPERAND_KEYS))
            if refused:
                raise ValueError(
                    f"{where} carries the production-condition operand key(s) "
                    f"{refused}; a production condition reads what crosses this "
                    "component, and `automaton`/`state` gate on a location "
                    "while `release` needs one to hold a band between its two "
                    "edges. The production variable is rewritten at every "
                    "evaluation and holds neither. State the condition through "
                    "a rule guard, which does"
                )
            unknown = sorted(set(spec) - PROD_COND_OPERAND_KEYS)
            if unknown:
                raise ValueError(
                    f"{where} carries a production-condition operand with "
                    f"unknown keys {unknown}; an operand carries "
                    f"{sorted(PROD_COND_OPERAND_KEYS)}"
                )

        operand = self._parse_operand(where, spec)
        if operand.negate and operand.op is not None:
            raise ValueError(
                f"{where} both negates and compares `{operand.name}`; a "
                "comparison already yields a truth value, so it is denied by "
                "the opposite operator rather than beside it"
            )
        if self._operand_reads_continuously(where, operand):
            if operand.op is None:
                raise ValueError(
                    f"{where} reads `{operand.name}` as a boolean, and what it "
                    "names is a continuously-evolving quantity. muscadet reads "
                    "such an operand as `!= 0`; here the condition is read at "
                    "discrete epochs, so the crossing has to be LOCATED to be "
                    "read at its own date, and a crossing is located on an "
                    "ordering comparison. Compare it explicitly: `{'op': '>', "
                    "'value': 0}` says the same thing of a quantity that only "
                    "rises, and says which side it is entered from"
                )
            if operand.op not in _ORDERING_OPS:
                raise ValueError(
                    f"{where} conditions on the continuous quantity "
                    f"`{operand.name}` with `{operand.op}`; the transition "
                    "carrying it is boundary, and a crossing is located on an "
                    f"ordering comparison ({', '.join(_ORDERING_OPS)})"
                )
        return operand

    @staticmethod
    def _prod_cond_threshold(flow_name: str, row: int, column: int) -> tuple[str, str, str]:
        """The automaton one continuous comparison of a production condition
        is boundary by, and its two locations."""
        base = f"{flow_name}_cond_{row}_{column}"
        return f"{base}_threshold", f"{base}_below", f"{base}_above"

    def _prod_cond_expr(
        self, where: str, flow_name: str, groups: list[list[_RuleOperand]]
    ) -> dict[str, Any]:
        """A parsed production condition as a boolean expression: the
        disjunction of its groups, each the conjunction of its operands.

        An operand comparing a **continuously-evolving** quantity reads the
        LOCATION of its threshold automaton rather than the quantity itself,
        and that indirection is the whole mechanism rather than a detour. This
        expression writes a variable through a propagation, and a
        propagation is re-run by a *discrete* change: an attribute an
        ODE moves announces nothing of its own, so a condition reading the
        quantity directly is evaluated at t = 0 and never again -- measured,
        on a volume filling past its threshold with the condition left False
        for the whole mission. Reading the location subscribes the function to
        the automaton (`state_triggers`), whose boundary transitions fire AT the
        crossing. muscadet wires the same thing the other way round, hanging a
        sensitive method on the same automaton
        (`add_prod_cond_threshold_automata`).

        The two agree everywhere but at t = 0, where the automaton is still in
        the location it was declared in: the initial fixpoint settles it, as it
        settles a capacity's discharge gate and a rule set's mode.
        """
        return _bool(
            "or",
            [
                _bool(
                    "and",
                    [
                        self._prod_cond_operand_expr(
                            where, flow_name, row, column, operand
                        )
                        for column, operand in enumerate(group)
                    ],
                )
                for row, group in enumerate(groups)
            ],
        )

    def _prod_cond_operand_expr(
        self,
        where: str,
        flow_name: str,
        row: int,
        column: int,
        operand: _RuleOperand,
    ) -> dict[str, Any]:
        """One production-condition operand as a boolean expression."""
        if operand.op is not None and self._operand_reads_continuously(where, operand):
            automaton, _, above = self._prod_cond_threshold(flow_name, row, column)
            return _state_active(self.name, automaton, above)
        return self._rule_operand_expr(operand)

    def _prod_cond_threshold_automata(
        self, where: str, flow_name: str, groups: list[list[_RuleOperand]]
    ) -> list[dict[str, Any]]:
        """One two-state automaton per comparison a production condition makes
        against a **continuously-evolving** quantity.

        The automaton is not what the condition reads: the expression above
        reads the quantity itself, so the two can never disagree on a value.
        What it contributes is the **date**. A quantity moving inside an
        integration step announces no change of its own, so nothing would
        re-evaluate the condition at the crossing and the output would start
        (or stop) producing at whatever event happened to come next. Declaring
        the two transitions `boundary` makes the solver stop AT the crossing,
        which is what a rule set's mode buys on the other direction of the
        interoperation and what muscadet builds here under the same reasoning
        (``muscadet.flow.add_prod_cond_threshold_automata``).

        A comparison on a DISCRETE read needs none of it: that read changes
        only at an event, which is already a stop.
        """
        automata = []
        for row, group in enumerate(groups):
            for column, operand in enumerate(group):
                if operand.op is None or not self._operand_reads_continuously(
                    where, operand
                ):
                    continue
                automaton, below, above = self._prod_cond_threshold(
                    flow_name, row, column
                )
                automata.append(
                    {
                        "name": automaton,
                        "states": [below, above],
                        # The location the comparison designates cannot be
                        # known here: the initial fixpoint resolves the
                        # quantity, and the boundary transition settles the
                        # automaton there. Exactly how a capacity's discharge
                        # gate enters its own.
                        "init": below,
                        "transitions": [
                            {
                                "name": f"{automaton}_cross_up",
                                "source": below,
                                "targets": [above],
                                "guard": self._rule_operand_expr(operand),
                                "distrib": "boundary",
                            },
                            {
                                "name": f"{automaton}_cross_down",
                                "source": above,
                                "targets": [below],
                                "guard": self._rule_operand_negation(operand),
                                "distrib": "boundary",
                            },
                        ],
                    }
                )
        return automata

    def add_flow_out_tempo(
        self,
        name: str,
        enable_time: float = 0.0,
        disable_time: float = 0.0,
        init_enable: bool = False,
        var_prod_default: bool = False,
        var_prod_cond: list[Any] | None = None,
        var_fed_available_out_init: bool | None = None,
        var_fed_available_out_reset: bool | None = None,
        enable_law: dict[str, Any] | None = None,
        disable_law: dict[str, Any] | None = None,
        var_prod_cond_inner_mode: str = PROD_COND_INNER_MODE_DEFAULT,
    ) -> None:
        """muscadet `FlowOutTempo`: the flow feeds while a
        disabled↔enabled automaton sits in `enabled`; the enable
        (resp. disable) transition is guarded on the production
        condition (resp. its negation), reset on interruption.

        Each transition waits on a law. `enable_time` / `disable_time`
        are the short form of a fixed delay; `enable_law` /
        `disable_law` take the law itself, in the engine's transition
        form: ``{"distrib": "delay", "time": t}``,
        ``{"distrib": "exp", "rate": r}`` or
        ``{"distrib": "uniform", "low": a, "high": b}``, the last two
        drawing the switch-on (switch-off) instant. A law and a non-zero time for
        the same direction contradict each other and are refused.

        The availability gate's two knobs are the base class's and are
        read here for the same reason muscadet reads them here: the
        temporised delivery still ANDs ``{name}_fed_available_out``."""
        enable = _tempo_law(name, "enable", enable_time, enable_law)
        disable = _tempo_law(name, "disable", disable_time, disable_law)
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                var_prod_cond_inner_mode=_prod_cond_inner_mode(
                    name, var_prod_cond_inner_mode
                ),
                var_fed_available_out_init=var_fed_available_out_init,
                var_fed_available_out_reset=var_fed_available_out_reset,
                tempo={"enable": enable, "disable": disable, "init_enable": init_enable},
            )
        )

    def add_flow_out_on_trigger(
        self,
        name: str,
        trigger_time_up: float = 0.0,
        trigger_time_down: float = 0.0,
        trigger_logic: str | int = "or",
        var_prod_default: bool = False,
        var_prod_cond: list[Any] | None = None,
        var_fed_available_out_init: bool | None = None,
        var_fed_available_out_reset: bool | None = None,
        var_prod_cond_inner_mode: str = PROD_COND_INNER_MODE_DEFAULT,
    ) -> None:
        """muscadet `FlowOutOnTrigger`: the flow feeds while a down↔up
        automaton sits in `up`, with *inhibition* logic: `up` is armed
        while the trigger aggregate (`"and"`, `"or"` or k-out-of-n over
        the `{name}_trigger_in` port) is false, `down` while it is
        true; both transitions are delays, reset on interruption.

        The availability gate's two knobs are the base class's, read
        here as on the temporised output: the delivery still ANDs
        ``{name}_fed_available_out``."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                var_prod_cond_inner_mode=_prod_cond_inner_mode(
                    name, var_prod_cond_inner_mode
                ),
                var_fed_available_out_init=var_fed_available_out_init,
                var_fed_available_out_reset=var_fed_available_out_reset,
                trigger={
                    "time_up": trigger_time_up,
                    "time_down": trigger_time_down,
                    "logic": trigger_logic,
                },
            )
        )

    def _build_flows_in(
        self,
        variables: list[dict],
        ports: list[dict],
        functions: list[dict],
    ) -> None:
        """The boolean in-flows: one `{flow}_fed_in` per declaration, an
        in port, and the propagation aggregating that port under
        the declared logic (any, all, or k of n).

        muscadet answers an in-flow nothing feeds with its declared
        `var_in_default` (`FlowDiscreteIn`, default `False`), and the
        aggregates alone do not: `all` over an empty port is the vacuous
        truth, and `sum >= 0` over one is true as well. So the emptiness
        is written INTO the expression, as `count(p) >= 1`, exactly as
        the trigger port next door writes it and for the same reason --
        a rule written on the port means the same thing whatever is
        wired to it, and a model rebuilt with one connection more has
        nothing to regenerate.

        Reading it the other way round -- dropping the aggregation of an
        input the connection list does not name -- is what this used to
        do, and it was wrong twice: it needed system knowledge the
        component does not have, and it left the bare aggregation in
        place on every input that declared no default, where the
        initialization axiom overwrote the `False` seed with the vacuous
        truth at t = 0.

        `or` keeps its bare `any`: an `any` that answers true already
        names a connection, so the emptiness needs no term of its own
        there. `and` and k do need one, and muscadet's own dissymmetry is
        the tell: it passes the default as an ARGUMENT of the aggregate
        for `and` and `or` (`andValue`, `orValue`) and as an early return
        guarded by `nbCnx() == 0` for an integer k."""
        me = self.name
        for flow in self.flows_in:
            fed_in = f"{flow.name}_fed_in"
            variables.append(
                {
                    "name": fed_in,
                    "kind": "bool",
                    "init": {"kind": "bool", "value": bool(flow.var_in_default)},
                }
            )
            ports.append({"name": f"{flow.name}_in", "dir": "in"})
            agg: dict[str, Any]
            port_ref = {"component": me, "port": f"{flow.name}_in"}
            connected = {
                "op": "cmp",
                "cmp": "ge",
                "lhs": {"op": "port_agg", "port": port_ref, "agg": "count"},
                "rhs": {"op": "const", "value": {"kind": "int", "value": 1}},
            }
            if flow.logic == "and":
                agg = {
                    "op": "bool",
                    "bool_op": "and",
                    "args": [
                        {"op": "port_agg", "port": port_ref, "agg": "all"},
                        connected,
                    ],
                }
            elif flow.logic == "k":
                agg = {
                    "op": "bool",
                    "bool_op": "and",
                    "args": [
                        {
                            "op": "cmp",
                            "cmp": "ge",
                            "lhs": {"op": "port_agg", "port": port_ref, "agg": "sum"},
                            "rhs": {
                                "op": "const",
                                "value": {"kind": "int", "value": flow.k},
                            },
                        },
                        connected,
                    ],
                }
            else:
                agg = {"op": "port_agg", "port": port_ref, "agg": "any"}
            if flow.var_in_default:
                # A declared always-fed boundary input: what it reads out
                # of connection is the declaration, not the aggregate.
                # The seed says the same thing, but the seed alone does
                # not survive the initialization axiom, which runs every
                # propagation once.
                agg = {
                    "op": "bool",
                    "bool_op": "or",
                    "args": [
                        agg,
                        {"op": "bool", "bool_op": "not", "args": [connected]},
                    ],
                }
            functions.append(
                {
                    "name": f"update_{fed_in}",
                    "effects": [
                        {"target": {"component": me, "attribute": fed_in}, "value": agg}
                    ],
                }
            )

    def _build_flows_out(
        self,
        variables: list[dict],
        ports: list[dict],
        functions: list[dict],
        automata: list[dict],
        connected_in: set[str] | None = None,
    ) -> None:
        """The boolean out-flows: what the failure modes leave available,
        the production condition, and the tempo or trigger automaton
        gating the delivery when one is declared.

        `connected_in` is read for one thing only, and at the trigger: a
        `FlowOutOnTrigger` whose trigger port nothing feeds is armed from
        the initial instant, so its automaton is declared `up` rather
        than reaching it through a transition of zero delay.

        Availability is *derived* here (the conjunction of the mode states
        targeting the flow), not clamped by effects as muscadet's is, so a
        seeded `var_fed_available_out_init` on a flow that carries failure
        modes is recomputed at t = 0. A dormancy seed therefore needs the
        study's own effect on `{flow}_fed_available_out` to be held: the
        reinitialization clamp falls back on the declared initial value,
        which is this seed.

        Production, on the other hand, is a VARIABLE and not an
        expression, exactly as muscadet's ``var_prod_available`` is, and
        the reason is the one the variable exists for over there: the
        function that writes it is emitted only when a production
        condition is declared, so a flow WITHOUT one has no writer at all
        and a failure mode may latch it to start a dormant output. An
        inline expression has nothing to latch. With a condition the
        function rewrites the variable at every evaluation and a mode's
        effect is erased by it, which is what muscadet's sensitive method
        does to the same variable."""
        me = self.name
        for flow in self.flows_out:
            # Empty groups only, under "and": the output that declares no
            # condition and starts produced (see `prod_cond_unconditional`).
            unconditional = prod_cond_unconditional(
                f"ObjFlow `{me}`: production condition of `{flow.name}`",
                flow.var_prod_cond,
                flow.var_prod_cond_inner_mode,
            )
            prod_cond = [] if unconditional else flow.var_prod_cond
            prod_default = True if unconditional else flow.var_prod_default
            fed_out = f"{flow.name}_fed_out"
            # muscadet-aligned name (`FlowOut.var_fed_available_out`): real
            # studies' failure_effects target `{flow}_fed_available_out`.
            available = f"{flow.name}_fed_available_out"
            variables.append(
                {"name": fed_out, "kind": "bool", "init": {"kind": "bool", "value": False}}
            )
            variables.append(
                {
                    "name": available,
                    "kind": "bool",
                    "init": {
                        "kind": "bool",
                        "value": bool(
                            flow.var_fed_available_out_init
                            if flow.var_fed_available_out_init is not None
                            else True
                        ),
                    },
                }
            )
            # muscadet's `FlowOut.var_prod_available`, and a variable for
            # the same reason: it is the one attribute of a discrete
            # output a failure mode may WRITE to start a dormant
            # function. Emitted on every discrete output, whether or not
            # a production condition is declared, because what tells the
            # dormant case apart is the absence of the WRITER below, not
            # the absence of the variable.
            prod_available = f"{flow.name}_prod_available"
            variables.append(
                {
                    "name": prod_available,
                    "kind": "bool",
                    "init": {"kind": "bool", "value": bool(prod_default)},
                }
            )
            # Service-function dormancy through `var_is_active`: the third
            # factor of muscadet's delivery (`var_prod AND var_is_active
            # AND var_fed_available_out`), present only when a caller
            # seeds it, so ordinary flows carry nothing new.
            is_active = f"{flow.name}_is_active"
            if flow.var_is_active_default is not None:
                variables.append(
                    {
                        "name": is_active,
                        "kind": "bool",
                        "init": {"kind": "bool", "value": bool(flow.var_is_active_default)},
                    }
                )
            ports.append({"name": f"{flow.name}_out", "dir": "out", "attr": fed_out})

            # Availability follows the failure-mode automata targeting
            # this flow (all of them must sit in `ok`).
            relevant = [
                m for m in self.failure_modes if not m.targets or flow.name in m.targets
            ]
            if relevant and flow.var_fed_available_out_reset is False:
                # A gate declared PERSISTENT and DERIVED at once has no
                # faithful expansion: persistence asks the gate to keep
                # whatever it was last written to, and the writer below
                # recomputes it from the mode states at every pass, so
                # the gate would come back up on repair instead of
                # latching. Refused by name: a divergence a study has no
                # way of noticing is worse than a model it cannot run. A
                # mode declared BESIDE the component is latched instead
                # (`pyraichu.muscadet.plugin.
                # _latch_held_writes_on_persistent_gates`), writing the
                # gate on its own edges; a mode declared inside it has no
                # automaton of its own to carry them.
                raise ValueError(
                    f"ObjFlow `{me}`: the availability gate "
                    f"`{available}` is declared persistent "
                    f"(`var_fed_available_out_reset=False` on flow "
                    f"`{flow.name}`), and the failure mode(s) "
                    f"{sorted(m.name for m in relevant)} derive it. This "
                    f"engine has no per-variable reset to switch off: the "
                    f"gate is recomputed from the mode states at every "
                    f"evaluation, so it would come back up on repair "
                    f"instead of memorising. Declare a reinitialized gate "
                    f"(`var_fed_available_out_reset=True`), or take the "
                    f"flow out of the mode's targets"
                )
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
            # `prod_cond` form). An operand is a bare flow name or the
            # mapping muscadet writes back, and it is read by the very
            # parser a rule guard's operands go through: the default
            # resolution searches the INPUT side first, exactly as
            # muscadet's does, and `port` selects the other one where the
            # component carries the name both ways (the transit pattern:
            # a board that feeds on and says so). An in-flow reads
            # `_fed_in`, an out-flow `_fed_out` -- the diagnostic-mirror
            # pattern; the fixpoint handles the intra-component
            # dependency without a topological sort.
            if prod_cond:
                where = (
                    f"ObjFlow `{me}`: production condition of `{flow.name}`"
                )
                groups = self._parse_prod_cond(
                    where, prod_cond, flow.var_prod_cond_inner_mode
                )
                functions.append(
                    {
                        "name": f"update_{prod_available}",
                        "effects": [
                            {
                                "target": {
                                    "component": me,
                                    "attribute": prod_available,
                                },
                                "value": self._prod_cond_expr(
                                    where, flow.name, groups
                                ),
                            }
                        ],
                    }
                )
                automata.extend(
                    self._prod_cond_threshold_automata(where, flow.name, groups)
                )
            # No production condition, no writer: the variable keeps
            # whatever it holds, which is its declared initial value
            # until a failure mode writes it. That gap IS the dormant
            # function, and it is muscadet's own (`update_sensitive_
            # methods` subscribes the production method to the operands
            # of `var_prod_cond`, so an empty condition leaves it
            # unsubscribed and the variable unwritten).

            # What the delivery and the tempo automaton read from here
            # on: the variable, never the condition that writes it.
            # Reading the condition again would re-derive the production
            # at the point of use and step over a mode's latch.
            prod_expr: dict[str, Any] = _var(me, prod_available)

            if flow.tempo is not None:
                # FlowOutTempo: fed while `enabled`; the production
                # condition only gates the (delayed, reset) enable and
                # disable transitions: a lost condition keeps feeding
                # until the disable delay elapses.
                aut = f"{flow.name}_tempo"
                transitions = [
                    {
                        "name": f"{flow.name}_enable",
                        "source": "disabled",
                        "targets": ["enabled"],
                        "guard": prod_expr,
                        **flow.tempo["enable"],
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
                        **flow.tempo["disable"],
                    },
                ]
                automata.append(
                    {
                        "name": aut,
                        "states": ["disabled", "enabled"],
                        "init": "enabled" if flow.tempo["init_enable"] else "disabled",
                        # An exponential wait at rate 0 never ends: the
                        # reference engine reads it as a transition that
                        # never fires, and the engine takes no zero rate,
                        # so it is not emitted.
                        "transitions": [t for t in transitions if t.get("rate") != 0.0],
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
                    # An UNCONNECTED trigger port reads `False`, not the
                    # vacuous truth of a conjunction over nothing.
                    # muscadet aggregates that port with an explicit
                    # no-connection value and passes `False` for the three
                    # logics (`flow.py`: `andValue(False)`,
                    # `orValue(False)`, `sumValue(0) >= k`), so a trigger
                    # nothing feeds INHIBITS NOTHING and the backup runs.
                    # `any` and `sum` already answer that on an empty
                    # port; `all` does not, so the emptiness is written
                    # into the expression rather than left to the
                    # aggregate's own convention -- and written as a
                    # condition on the port, not on the connection list,
                    # so the guard means the same thing whatever is wired
                    # to it.
                    trigger_agg: dict[str, Any] = {
                        "op": "bool",
                        "bool_op": "and",
                        "args": [
                            {"op": "port_agg", "port": port_ref, "agg": "all"},
                            {
                                "op": "cmp",
                                "cmp": "ge",
                                "lhs": {
                                    "op": "port_agg",
                                    "port": port_ref,
                                    "agg": "count",
                                },
                                "rhs": {
                                    "op": "const",
                                    "value": {"kind": "int", "value": 1},
                                },
                            },
                        ],
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
                # ... and it inhibits nothing FROM THE INITIAL INSTANT.
                # With nothing connected the aggregate is false for ever:
                # the up transition is enabled at t = 0 and the down one
                # never is, so `down` is a state of zero duration. The
                # reference engine fires that zero-delay transition inside
                # the initial instant and already reads `up` at t = 0;
                # declaring the initial state as `up` is what makes the
                # two agree there, and it is exact rather than cosmetic
                # since no connection can ever bring the trigger down
                # again. A declared `time_up` is a real wait, so it keeps
                # its transition: the automaton starts `down` and rises at
                # that date on both engines.
                armed_from_the_start = (
                    connected_in is not None
                    and f"{flow.name}_trigger" not in connected_in
                    and float(flow.trigger["time_up"]) == 0.0
                )
                automata.append(
                    {
                        "name": aut,
                        "states": ["down", "up"],
                        "init": "up" if armed_from_the_start else "down",
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

            if flow.var_is_active_default is not None:
                gate_terms = gate_terms + [_var(me, is_active)]

            fed_expr = {
                "op": "bool",
                "bool_op": "and",
                "args": gate_terms + [_var(me, available)],
            }
            if flow.negate:
                fed_expr = {"op": "bool", "bool_op": "not", "args": [fed_expr]}
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
