"""Capacities (stocks) of a component: declaration, bounds and emission."""

from __future__ import annotations

from typing import Any

from ._expr import (
    _clamped_at_zero,
    _float,
    _float_attribute,
    _min,
    _negated,
    _state_active,
    _sum,
    _var,
)
from ._records import (
    _Capacity,
    _CapacityFlow,
    _content_attribute,
    _fill_claim,
    _fill_threshold,
    _flow_fill,
    _weighted_fill,
)
from ._vocabulary import (
    DEFAULT_HYSTERESIS,
    SERVE_COND_INNER_MODE_DEFAULT,
    SERVE_COND_INNER_MODES,
    SERVE_GATE_STATES,
    UNBOUNDED_SERVICE,
)


class _CapacitiesMixin:
    """Methods of :class:`ObjFlow` capacities (stocks) of a component: declaration, bounds and emission."""

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
        transmits: bool = False,
        serve_rate: float | None = None,
        serve_cond: Any = None,
        serve_cond_negate: list | None = None,
        serve_cond_compare: list | None = None,
        serve_cond_inner_mode: str | None = None,
        hysteresis: float | None = None,
        makeup_flow: str | None = None,
    ) -> None:
        """Declare a volume this component holds over one or more of its
        continuous flows (muscadet `add_capacity`).

        `makeup_flow` opens the volume to the atmosphere (muscadet's R53)
        and names the held constituent the outside supplies: the volume
        stays full, the make-up entering at the rate it runs short and the
        excess venting at the composition. What `content_init` leaves of the
        volume is the make-up at t=0. What it asks upstream follows
        `fill_rate` as for any volume: a room fed by leaks declares it
        unbounded.

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

        **A volume on an output is what that output delivers**, in place
        of the rate the flow was declared with, wherever the component
        declares no production of its own: while it holds something it
        then serves whatever is asked of it, capped by `serve_rate` and by
        nothing else. That is what a volume MEANS on an output, and it is
        what muscadet's `Capacity.serve_limit` computes; a capacity over a
        flow whose declared rate is zero -- how every exported muscadet
        capacity comes out, its component producing nothing -- would
        otherwise deliver zero however full it was. Declare `serve_rate`
        to give the volume a rate of its own; it defaults to unbounded, as
        muscadet's does. An out-flow rate that *says* something stays a
        ceiling the volume respects.

        **The ceiling is a VARIABLE, one per held flow**, published under
        the name muscadet gives it, ``{capacity}_serve_rate_{flow}``. A
        constant folded into the capability would leave a failure mode
        written to throttle the discharge with nothing to clamp, which is
        precisely how muscadet's own modes reach it; and muscadet declares
        it per held flow, so a multi-flow volume releases the SUM of its
        constituents' ceilings rather than one figure shared between them.

        `serve_cond` is the **discharge command**: a condition on which
        the volume releases anything at all. It is written in the operand
        vocabulary a RULE GUARD carries -- ``{name, port, automaton,
        state, negate, op, value, release}`` -- which is the one that
        carries the comparisons muscadet's own `serve_cond` documents, and
        not the flow-name-only vocabulary of a discrete production
        condition. Two levels, combined as `serve_cond_inner_mode` says:
        ``"or"``, muscadet's default, reads the outer list as a
        conjunction of disjunctions, ``"and"`` the other way round.
        `serve_cond_negate` and `serve_cond_compare` are the parallel
        matrices muscadet lifts a negation and a comparison OUT of its
        operands into; they are folded back onto the operand they belong
        to, so the two spellings declare one condition and a contradiction
        between them is refused rather than silently resolved.

        **The command gates BOTH branches**, which is what muscadet's
        `Capacity.serve_ceiling` does by answering zero outright: a
        commanded volume standing down holds back what it stores AND what
        merely crosses it. Gating the stored branch alone would leave a
        buffer at rest passing its inlet through to its outlet.

        `release` on a comparison is a **band**, a RAICHU extension
        muscadet has no field for: the command is entered at `value` and
        left at `release`, and between the two it holds whatever it
        already was. It is what a command reading a level its own
        discharge moves needs, since without it the two states each
        produce the condition that justifies the other and the period of
        the cycle is numerical rather than physical.

        The condition is compiled into a two-state automaton
        (``{capacity}_serve``, :data:`SERVE_GATE_STATES`) rather than
        folded into the served quantity, which is how muscadet compiles it
        too (`Capacity.add_serve_cond_automata`) and what makes it visible
        to the pre-run loop diagnostic: a threshold inlined in a quantity
        closes a cycle among variables alone, which reads as an algebraic
        loop and is reported by nobody.

        `transmits` is whether the volume passes on what it does not hold
        back: whether the **through-path exists at all**, which `side`
        cannot say, since a buffer and a reservoir come out of an exported
        document with the same one. True is a volume with a way in and a
        way out, which once empty still lets through what currently
        crosses it, so a shortage is smoothed rather than swallowed, and
        which carries its consumers' demand upstream on top of its own
        claim.

        False -- **the default here** -- is a pure store: it serves what it
        holds and nothing more, it asks upstream only for what its own
        inlet declares, and its out-flow rate stays its ceiling **even at
        zero**. A rate of zero on a volume that declares no through-path
        is a SHUT OUTLET, a model somebody means, where the same zero on
        one that does is a component producing nothing of its own.

        **muscadet's own field defaults to True, and this one does not.**
        muscadet's solver never reads it: it takes the through-path from
        the wiring, so its default moves no model of its own. Here the
        field decides, so a default of True would reinterpret every volume
        already written against this layer. Every document muscadet
        exports carries the key explicitly, at both values, so nothing is
        lost by making a volume say it.

        A through-path needs **two ports**, so a volume this component
        gives the flow no way in keeps its declared rate whatever it
        declares: there is no path for it to use.

        The demand a **full** capacity carries upstream is what it can
        still take, which is what currently leaves it, capped by the
        demand already passing through it. That pass-through demand is
        what its consumers ask of it where the volume transits, and
        otherwise whatever the flow would ask for without the volume: a
        rule set's derived demand when one consumes the flow, and the
        in-flow's declared ``var_demand_in_default`` failing that. A tank
        that delivers five, that does not transit, that no rule consumes
        and that declares no pass-through demand asks for nothing once
        full, and therefore drains; declare either the demand or the
        transit, and the buffer holds. The two compose in either
        declaration order: the bound is applied when the model is
        generated, not when the capacity is declared.

        `side` places the capacity upstream (``"in"``) or downstream
        (``"out"``) of the component's transformation rules, as muscadet
        does, and left out it is resolved as muscadet resolves it: the
        side the held flow is carried on, ``"in"`` for a flow carried on
        both. On a flow no rule set carries it does not discriminate: the
        bounds govern both ends of every held flow the component declares,
        a pure buffer being the same object read from either side, which
        is why one capacity per flow is enough here and a second one on
        the other side is refused rather than silently ignored.

        **On a flow a rule set of this component carries, the side is the
        whole reading**, and the rules keep their declared coefficients;
        the volume changes only what crosses the boundary:

        - ``"in"``, on a flow the rules consume: what arrives fills the
          volume, and the rules draw from it, whatever they need while
          that flow is stocked and what arrives once it has run out. What
          is published upstream is the rules' need, claimed by the volume
          as any volume claims (fill rate while there is room, the need
          alone once full);
        - ``"out"``, on a flow the rules produce: the consumers' demand is
          replaced by the volume's claim on it before the rule scale is
          taken, what the rules make fills the volume, and the consumers
          are served from it, their whole request while it is stocked and
          what the rules make once it is empty.

        A volume holding several such flows judges emptiness **per
        constituent** and fullness on the shared total, as muscadet does.
        `transmits` says nothing on such a flow: the through-path is the
        rule. Refused, in either declaration order, is only what would
        make the flow cross twice: a held flow the rules transform that
        the component ALSO declares on the other side, and a volume on
        the side the rules do not face; see
        :meth:`_refuse_flows_carried_twice`."""
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
        # Same spelling as `fill_rate` above and for the same reason: a
        # document has no literal for an infinity, so the unbounded
        # ceiling -- the default, and muscadet's -- is written as null.
        ceiling = float("inf") if serve_rate is None else float(serve_rate)
        if not ceiling >= 0.0:  # negative or NaN
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` must declare a serve "
                f"rate that is positive, zero, or null for unbounded, got "
                f"{serve_rate}"
            )
        if not isinstance(transmits, bool):
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` must declare "
                f"`transmits` as a boolean, got {transmits!r}. It says WHETHER "
                "the volume has a through-path, not how much crosses it; the "
                "rate is `serve_rate`"
            )
        inner_mode = (
            SERVE_COND_INNER_MODE_DEFAULT
            if serve_cond_inner_mode is None
            else serve_cond_inner_mode
        )
        if inner_mode not in SERVE_COND_INNER_MODES:
            raise ValueError(
                f"ObjFlow `{self.name}`: capacity `{name}` declares "
                f"`serve_cond_inner_mode`={serve_cond_inner_mode!r}; a "
                f"discharge command combines its two levels one way or the "
                f"other, so it is "
                f"{' or '.join(repr(mode) for mode in SERVE_COND_INNER_MODES)}"
            )
        command = self._parse_command(
            f"ObjFlow `{self.name}`: capacity `{name}`",
            serve_cond,
            serve_cond_negate,
            serve_cond_compare,
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

        content = {key: float(value) for key, value in (content_init or {}).items()}
        if makeup_flow is not None:
            where = f"ObjFlow `{self.name}`: capacity `{name}`"
            held = [entry.name for entry in entries]
            if makeup_flow not in held:
                raise ValueError(
                    f"{where}: makeup_flow `{makeup_flow}` names a constituent "
                    f"the volume does not hold ({held}). The outside supplies "
                    "one of the constituents already in the volume"
                )
            weights = {entry.name: entry.weight for entry in entries}
            room = float(capacity) - sum(
                quantity * weights[flow] for flow, quantity in content.items()
            )
            # Written into the content rather than kept aside, exactly as
            # muscadet writes it: the declaration then states what the
            # volume holds, and a document read back carries it.
            if room > 0.0:
                content[makeup_flow] = (
                    content.get(makeup_flow, 0.0) + room / weights[makeup_flow]
                )

        declaration = _Capacity(
            name=name,
            flows=entries,
            volume=float(capacity),
            side=resolved,
            content_init=content,
            fill_rate=rate,
            transmits=transmits,
            serve_rate=ceiling,
            serve_cond=command,
            serve_cond_inner_mode=inner_mode,
            hysteresis=width,
            makeup_flow=makeup_flow,
        )
        self._record(self.capacities, declaration)
        # The command is resolved once the volume is RECORDED, and the
        # order is what makes a reserve floor declarable: an operand
        # naming this capacity's own level reads the contents this
        # declaration is what creates. A refused command takes the whole
        # declaration back out, as `_record` does with its own refusal.
        try:
            self._resolve_command(declaration)
        except ValueError:
            self.capacities.remove(declaration)
            raise

    def _capacity_of(self, flow: str) -> _Capacity | None:
        """The volume holding `flow`, or ``None``. A flow buffers in one
        volume, which :meth:`add_capacity` enforces."""
        for capacity in self.capacities:
            if flow in capacity.flow_names:
                return capacity
        return None

    def _beside_rules(self, flow: str, side: str) -> _Capacity | None:
        """The volume holding `flow` on `side` of this component's rules,
        or ``None``: a volume the rules draw from (``"in"``) or fill
        (``"out"``) on that flow.

        Only a flow a rule set carries has a volume BESIDE the rules; a
        volume on any other flow is the ordinary buffer or store, with
        the boundary on both of its ends. Which way the rules carry it is
        what :meth:`_refuse_flows_carried_twice` has already settled, so
        this reads the side and not the direction."""
        capacity = self._capacity_of(flow)
        if capacity is None or capacity.side != side:
            return None
        if not any(
            flow in rule_set.consumed() or flow in rule_set.produced()
            for rule_set in self.rule_sets
        ):
            return None
        return capacity

    def _transit(self, flow: str) -> dict[str, Any]:
        """What currently arrives on `flow`, zero where the component
        declares no way in for it.

        On a recirculation ring (a ventilated room whose fan blows the air
        back) the returning flow is settled AFTER this volume's visit, so
        the pooled draw reads it one evaluation late. That is deliberate
        and measured: the reference reads the same edge one evaluation
        late (its ring cut), and reading it forward instead moved the H2
        showcase's room away from the reference for no physical gain; with
        this read the room agrees to within 0.009 at every instant."""
        if any(declared.name == flow for declared in self.flows_continuous_in):
            return _var(self.name, f"{flow}_fed_in")
        return _float(0.0)

    def _stock_automaton(self, capacity: _Capacity, flow: str) -> str:
        """Name of the automaton judging whether `flow` alone is stocked
        in a volume holding several constituents."""
        return f"{capacity.name}_stock_{flow}"

    def _held_empty(self, capacity: _Capacity, flow: str) -> dict[str, Any]:
        """Whether `flow` has run out in `capacity`, as a location.

        **Judged per constituent**, as muscadet's `serve_limit` judges it
        on the flow's own quantity: a volume whose hydrogen has filled it
        holds no oxygen for that, and serving the oxygen rule from stock
        it does not have would drive its content below zero. A volume
        holding a single flow asks its bounds automaton, whose total IS
        that flow; one holding several asks the per-constituent automaton
        :meth:`_build_capacities` posts for every held flow. Both are locations entered by located crossings, never a
        comparison read inside a quantity."""
        if len(capacity.flows) == 1:
            return _state_active(self.name, f"{capacity.name}_bounds", "empty")
        return _state_active(
            self.name, self._stock_automaton(capacity, flow), "empty"
        )

    def _rule_supply(
        self, capacity: _Capacity, flow: str, arrived: dict[str, Any]
    ) -> dict[str, Any]:
        """What a volume upstream of the rules lets them take of `flow`:
        whatever is asked while that flow is stocked, capped by the
        ceiling alone, and once it has run out only `arrived`.

        muscadet's `serve_limit` on the rules' side of the volume. The
        capability sweep reads it with `arrived` the capability upstream,
        which is what COULD arrive; the production sweep reads it with
        what DID, which is what bounds the draw."""
        return self._served_while_stocked(capacity, flow, arrived)

    def _served_while_stocked(
        self, capacity: _Capacity, flow: str, incoming: dict[str, Any]
    ) -> dict[str, Any]:
        """What `capacity` lets out of `flow`: its ceiling while that
        flow is stocked, and once it has run out no more than `incoming`,
        what currently reaches the volume. Emptiness is judged on the
        flow's own content (:meth:`_held_empty`)."""
        ceiling = self._capacity_ceiling(capacity, flow)
        return {
            "op": "if",
            "cond": self._held_empty(capacity, flow),
            "then": _min([ceiling, incoming]),
            "otherwise": ceiling,
        }

    def _capacity_ceiling(self, capacity: _Capacity, flow: str) -> dict[str, Any]:
        """The ceiling on what `flow` may leave `capacity` at, right now:
        muscadet's `Capacity.serve_ceiling`, term for term.

        **Read from a variable** rather than written in as the declared
        number: a failure mode may have clamped it by the time this is
        evaluated, which is what a rate that can be throttled has to be.

        **The command composes with it by BRANCHING**, not by product,
        and the alternative is not a near miss: an unbounded ceiling
        crosses this layer as :data:`UNBOUNDED_SERVICE`, and a magnitude
        times zero is a quantity nobody downstream reads as "closed".
        A command that does not hold answers zero outright.

        One expression, read by the two sides of the volume: what it
        serves and what it carries upstream are capped by the same
        ceiling, as muscadet's `serve_limit` and `demand_claim` are.
        """
        ceiling: dict[str, Any] = _var(self.name, capacity.ceiling_of(flow))
        if not capacity.commanded:
            return ceiling
        return {
            "op": "if",
            "cond": _state_active(self.name, capacity.gate, SERVE_GATE_STATES[0]),
            "then": ceiling,
            "otherwise": _float(0.0),
        }

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

        **A volume that transits asks upstream for what is asked of it**,
        which is the other half of what makes it a buffer: it smooths a
        shortage rather than swallowing what arrives. That pass-through
        demand is read from the out side (``{flow}_demand_out``), which
        the demand band settles before it reaches this component's own
        inputs, and it is taken as the GREATER of it and whatever the
        flow would have asked on its own. The declared
        ``var_demand_in_default`` was the only way to state a
        pass-through before the out side could be read, so a model that
        declared one keeps it as a floor rather than losing it, and
        neither is added to the other: they are two statements of one
        quantity, and adding them would ask twice for one transit.

        **What it asks for the transit is capped by what it can
        release**, ceiling and discharge command alike: a volume does not
        fill out of a demand it cannot honour. What it claims for itself
        is not, that claim being what a `fill_rate` says and the ceiling
        having nothing to say about it.

        Applied here rather than at declaration, so a rule set and a
        capacity naming one flow compose whichever is declared first:
        the rule derives the pass-through demand, the volume bounds it."""
        capacity = self._capacity_of(flow)
        if capacity is None:
            return base
        me = self.name
        delivers = any(declared.name == flow for declared in self.flows_continuous_out)
        outflow = _var(me, f"{flow}_fed_out") if delivers else _float(0.0)
        if capacity.transmits and delivers and self._conduit_on(flow) is None:
            # A conduit is excluded because it already REPLACED the
            # transit: its crossing is `base` here, and asking for the
            # downstream demand beside it would ask for the same passage
            # twice.
            base = {"op": "max", "args": [base, _var(me, f"{flow}_demand_out")]}
        # **The transit a volume asks for is capped by what it can
        # release**, which is muscadet's `demand_claim` and the rule R-20
        # already wrote for the other side: a volume does not fill out of
        # a demand it cannot honour. Without it a buffer at the
        # documented `fill_rate=0` -- "it never stocks up" -- becomes an
        # accumulator the moment a ceiling is declared or a command
        # stands its discharge down: measured here on a commanded volume,
        # the two engines agreed on what reached the consumer and parted
        # on what the volume went on drawing behind it. What the volume
        # claims for ITSELF is not capped, and rightly: that is the
        # `fill_rate` below, a claim the ceiling has nothing to say about.
        base = {"op": "min", "args": [base, self._capacity_ceiling(capacity, flow)]}
        # An unbounded fill claims without bound, as muscadet publishes
        # it (`fill_rate = inf` is an infinite demand): the producer's
        # split weighs every claim as declared, so the volume takes the
        # whole supply and a finite rival beside it is served nothing.
        # Written as the capability this input reads, it weighed as
        # whatever capability reached it, which is no claim muscadet
        # makes.
        claiming = _fill_claim(capacity, base, _float(UNBOUNDED_SERVICE))
        # Full, a volume upstream of the rules asks for what the rules
        # draw from it, which is `base` itself: full is stocked, and a
        # stocked volume serves the rules their whole need. Written as
        # `base` rather than as the draw, which reads what arrived and
        # would close the demand band on the production band.
        accepted = (
            base
            if self._beside_rules(flow, "in") is not None
            else {"op": "min", "args": [base, outflow]}
        )
        # An OPEN volume is never full (muscadet's R53): what exceeds what
        # leaves is vented, so its claim is the ordinary one, `fill_rate`
        # included, and the throttling of a full volume never applies. A
        # forced unbounded claim drained a stocked tank feeding the room at
        # its whole content per unit of time, measured in muscadet.
        if capacity.is_open:
            return claiming
        return {
            "op": "if",
            "cond": _state_active(me, f"{capacity.name}_bounds", "full"),
            "then": accepted,
            "otherwise": claiming,
        }

    def _capacity_claim_out(
        self, capacity: _Capacity, flow: str, capability: dict[str, Any]
    ) -> dict[str, Any]:
        """What a volume downstream of the rules asks them to make of
        `flow`: muscadet's `demand_claim` on the consumers' demand, in
        the flow's own units, before the rule scale is taken on it.

        What the consumers ask, capped by the ceiling, plus the fill rate
        while there is room; unbounded fill is "whatever the rule can
        make", `capability`: this claim is read by the component's own
        rule scale, never weighed at a split, so the quantity it stands
        for is known here. Full, the claim is the consumers' demand
        alone: the stock serves it whole, so what may still enter is
        exactly what leaves, and reading the delivery here instead would
        close the demand band on the production band."""
        asked = _min(
            [_var(self.name, f"{flow}_demand_out"), self._capacity_ceiling(capacity, flow)]
        )
        claiming = _fill_claim(capacity, asked, capability)
        # Never full when open (R53): the ordinary claim, never the throttle.
        if capacity.is_open:
            return claiming
        return {
            "op": "if",
            "cond": _state_active(self.name, f"{capacity.name}_bounds", "full"),
            "then": asked,
            "otherwise": claiming,
        }

    def _capacity_bounded_capability(
        self, flow: str, nominal: dict[str, Any], *, replaces: bool = False
    ) -> dict[str, Any]:
        """What a flow can deliver once the volume holding it has its say.

        Mirrors muscadet's `Capacity.serve_limit`: while the volume holds
        something it serves whatever is asked, capped by the declared
        ``serve_rate`` and by nothing else; once empty it serves only what
        currently transits through it, and a volume that does not transit
        serves nothing at all. `transmits` is exactly the EXISTENCE of
        that empty branch, which is why it is a predicate and not a third
        rate.

        A declared `serve_cond` gates the ceiling itself, and therefore
        both branches, exactly as muscadet's `serve_ceiling` does by
        answering zero before either of them is reached.

        `replaces` says which of the two readings of `nominal` applies,
        and the call site is what knows:

        - **True** -- `nominal` is the rate the flow was declared with,
          and the volume REPLACES it. A volume on an output is what that
          output delivers, so a tank whose component produces nothing of
          its own still serves from its stock. Reading the declared rate
          as a second ceiling underneath is what made every exported
          muscadet capacity deliver zero: the exporting component
          produces nothing, so it writes ``var_fed_default: 0.0``, and a
          ceiling of zero leaves zero however full the volume is;
        - **False** -- `nominal` is a quantity something else already
          computed and that genuinely crosses the component: a metered
          conduit's crossing, what taps routed in, an explicitly declared
          capability. The volume BOUNDS that rather than replacing it,
          the component having a production of its own to bound.

        Applied here rather than at declaration, for the reason
        :meth:`_capacity_bounded_demand` gives."""
        capacity = self._capacity_of(flow)
        if capacity is None:
            return nominal
        me = self.name
        if self._beside_rules(flow, "out") is not None:
            # Downstream of the rules, `nominal` is what the rules could
            # make, and it plays the inlet: a stocked volume serves up to
            # its ceiling, an empty one what the rules could hand it.
            return self._served_while_stocked(capacity, flow, nominal)
        # What currently transits, which is what an empty volume passes
        # on. `fed_in` and not the published capability: what crosses is
        # what actually arrived, which is the quantity muscadet's own
        # `get_inflow` reads. A volume the component gives no way in
        # transits nothing.
        #
        # **Read whatever `transmits` says**, which is not an oversight.
        # muscadet's own `serve_limit` takes this branch from the WIRING
        # and not from the field, and the field is never written False on
        # a volume that has a way in: `ports` `in` and `out` are the only
        # two it comes from, and neither declares both sides. So gating
        # this on it would be equivalent on every document muscadet can
        # write, and would silently move the volumes already written
        # against this engine, which used to pass on what crossed them
        # here whatever else they declared.
        inflow = (
            _var(me, f"{flow}_fed_in")
            if any(declared.name == flow for declared in self.flows_continuous_in)
            else _float(0.0)
        )
        # Empty is judged on THIS flow's own content, not on the total:
        # a room full of air holds no hydrogen for it, and serving the
        # hydrogen from stock it does not have drove that constituent's
        # content below zero.
        served = self._served_while_stocked(capacity, flow, inflow)
        return served if replaces else {"op": "min", "args": [nominal, served]}

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
                _var(me, _content_attribute(capacity.name, entry.name))
                for entry in capacity.flows
            ]
            if name in (capacity.name, _content_attribute(capacity.name)):
                return _sum(contents)
            if name == f"{capacity.name}_fill":
                return _weighted_fill(me, capacity)
            for entry, content in zip(capacity.flows, contents):
                if name == _content_attribute(capacity.name, entry.name):
                    return content
                if name == f"{capacity.name}_fill_{entry.name}":
                    return _flow_fill(me, capacity, entry)
        return None

    def _claimed_outputs(self) -> dict[str, str]:
        """Per output held in a volume downstream of the rules, the
        variable carrying the volume's claim on it: what the rule scales
        read in place of the consumers' demand."""
        return {
            flow.name: f"{flow.name}_claimed_out"
            for flow in self.flows_continuous_out
            if self._beside_rules(flow.name, "out") is not None
        }

    def _capacity_transit(
        self, flow: str, continuous_in: set[str], continuous_out: set[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """What arrives at and what leaves a held flow's content, over the
        U7 channels. Beside the rules, the rules play the far end: upstream
        their draw is the outflow, downstream their production is the
        inflow."""
        me = self.name
        if self._beside_rules(flow, "out") is not None:
            arriving = _var(me, f"{flow}_made_out")
        elif flow in continuous_in:
            arriving = _var(me, f"{flow}_fed_in")
        else:
            arriving = _float(0.0)
        if self._beside_rules(flow, "in") is not None:
            leaving = self._rule_draw(flow)
        elif flow in continuous_out:
            leaving = _var(me, f"{flow}_fed_out")
        else:
            leaving = _float(0.0)
        return arriving, leaving

    def _capacity_surplus(
        self, capacity: _Capacity, continuous_in: set[str], continuous_out: set[str]
    ) -> dict[str, Any]:
        """``D = sum_f w_f (in_f - out_f)``, the volume the transit adds to
        `capacity` per unit of time (muscadet's `Capacity.breathe`, R53).

        Read off the deliveries, which the sweep settles before anything
        reads this, and never fed back into a demand or a capability: that
        would close the demand band on the production band."""
        terms = []
        for entry in capacity.flows:
            arriving, leaving = self._capacity_transit(
                entry.name, continuous_in, continuous_out
            )
            terms.append(
                {
                    "op": "mul",
                    "args": [
                        {"op": "sub", "lhs": arriving, "rhs": leaving},
                        _float(entry.weight),
                    ],
                }
            )
        return _sum(terms)

    def _capacity_vent(
        self,
        capacity: _Capacity,
        entry: Any,
        continuous_in: set[str],
        continuous_out: set[str],
    ) -> dict[str, Any]:
        """What an OPEN volume vents of one constituent: the excess of what
        arrives over what leaves, ``max(D, 0)``, at the composition
        ``m_f / sum_g m_g w_g`` (R53). The make-up and the vent together
        cancel ``D`` exactly, so the occupied volume never moves."""
        me = self.name
        occupied = _sum(
            [
                {
                    "op": "mul",
                    "args": [
                        _var(me, _content_attribute(capacity.name, held.name)),
                        _float(held.weight),
                    ],
                }
                for held in capacity.flows
            ]
        )
        content = _clamped_at_zero(_var(me, _content_attribute(capacity.name, entry.name)))
        excess = _clamped_at_zero(
            self._capacity_surplus(capacity, continuous_in, continuous_out)
        )
        return {
            "op": "if",
            "cond": {"op": "cmp", "cmp": "gt", "lhs": occupied, "rhs": _float(0.0)},
            "then": {
                "op": "div",
                "lhs": {"op": "mul", "args": [excess, content]},
                "rhs": occupied,
            },
            "otherwise": _float(0.0),
        }

    def _build_capacities(
        self,
        variables: list[dict],
        ports: list[dict],
        equations: list[dict],
        automata: list[dict],
    ) -> None:
        """Capacities: one integrated content per held flow, the fills and
        the two totals swept off them, the ceiling on each held flow, the
        bounds automaton, the discharge gate where one is commanded, and
        the read-only ports publishing the level.

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
            ratios: list[tuple[str, str]] = []
            # The declared ceiling, as a VARIABLE per held flow and under
            # muscadet's own name for it. A constant folded into the
            # capability leaves a failure mode written to throttle the
            # discharge with nothing to clamp; and a rate a state of
            # charge can suspend but a mode cannot reach is not a rate.
            # No equation defines it: a mode is its only writer, and its
            # initial value is what the declaration said.
            capacity_ceiling = (
                UNBOUNDED_SERVICE
                if capacity.serve_rate == float("inf")
                else capacity.serve_rate
            )
            for entry in capacity.flows:
                content = _content_attribute(capacity.name, entry.name)
                fill = f"{capacity.name}_fill_{entry.name}"
                initial = capacity.content_of(entry.name)
                variables.append(_float_attribute(content, initial))
                variables.append(
                    _float_attribute(fill, initial * entry.weight / capacity.volume)
                )
                variables.append(
                    _float_attribute(capacity.ceiling_of(entry.name), capacity_ceiling)
                )
                contents.append(_var(me, content))
                fills.append(_var(me, fill))

                # What arrives minus what leaves, over the U7 channels,
                # plus what the atmosphere adds to an OPEN volume (R53).
                arriving, leaving = self._capacity_transit(
                    entry.name, continuous_in_names, continuous_out_names
                )
                rate: dict[str, Any] = {"op": "sub", "lhs": arriving, "rhs": leaving}
                if capacity.is_open:
                    vent = f"{capacity.name}_vent_{entry.name}"
                    variables.append(_float_attribute(vent, 0.0))
                    equations.append(
                        {
                            "target": vent,
                            "kind": "explicit",
                            "expr": self._capacity_vent(
                                capacity, entry, continuous_in_names, continuous_out_names
                            ),
                        }
                    )
                    rate = {"op": "sub", "lhs": rate, "rhs": _var(me, vent)}
                    if entry.name == capacity.makeup_flow:
                        rate = {
                            "op": "add",
                            "args": [rate, _var(me, f"{capacity.name}_makeup")],
                        }
                equations.append({"target": content, "kind": "ode", "expr": rate})
                equations.append(
                    {
                        "target": fill,
                        "kind": "explicit",
                        "expr": _flow_fill(me, capacity, entry),
                    }
                )
                if capacity.publishes_ratios:
                    ratio = f"{capacity.name}_ratio_{entry.name}"
                    variables.append(
                        _float_attribute(ratio, capacity.initial_ratio(entry.name))
                    )
                    ratios.append((ratio, content))

            if capacity.is_open:
                makeup = f"{capacity.name}_makeup"
                variables.append(_float_attribute(makeup, 0.0))
                weight = next(
                    e.weight for e in capacity.flows if e.name == capacity.makeup_flow
                )
                equations.append(
                    {
                        "target": makeup,
                        "kind": "explicit",
                        "expr": {
                            "op": "div",
                            "lhs": _clamped_at_zero(
                                _negated(
                                    self._capacity_surplus(
                                        capacity, continuous_in_names, continuous_out_names
                                    )
                                )
                            ),
                            "rhs": _float(weight),
                        },
                    }
                )

            initial_fill = capacity.initial_fill()
            variables.append(
                _float_attribute(
                    _content_attribute(capacity.name),
                    sum(capacity.content_of(e.name) for e in capacity.flows),
                )
            )
            variables.append(_float_attribute(f"{capacity.name}_fill", initial_fill))
            # The two totals are swept, not integrated: summing the
            # contents is exact at every evaluation point, where a second
            # integration would drift away from its own constituents.
            equations.append(
                {
                    "target": _content_attribute(capacity.name),
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
            # After the total, which they divide by. An empty volume has
            # a total of zero and every ratio reads 0 there: nothing is
            # no fraction of nothing, and the alternative is a division
            # the sweep would have to answer with a non-number.
            for ratio, content in ratios:
                equations.append(
                    {
                        "target": ratio,
                        "kind": "explicit",
                        "expr": {
                            "op": "if",
                            "cond": {
                                "op": "cmp",
                                "cmp": "gt",
                                "lhs": _var(me, _content_attribute(capacity.name)),
                                "rhs": _float(0.0),
                            },
                            "then": {
                                "op": "div",
                                "lhs": _var(me, content),
                                "rhs": _var(me, _content_attribute(capacity.name)),
                            },
                            "otherwise": _float(0.0),
                        },
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

            # Whether each constituent is stocked, for a volume holding
            # more than one (:meth:`_held_empty`), whether it transits or
            # sits beside the rules. Left when the constituent's own share
            # passes the hysteresis band, entered when its content reaches
            # zero: the same inclusive, located guards as the bounds
            # above, read on one content instead of the total.
            if len(capacity.flows) > 1:
                for entry in capacity.flows:
                    stock = self._stock_automaton(capacity, entry.name)
                    share = _flow_fill(me, capacity, entry)
                    automata.append(
                        {
                            "name": stock,
                            "states": ["empty", "stocked"],
                            "init": (
                                "stocked"
                                if capacity.content_of(entry.name) > 0.0
                                else "empty"
                            ),
                            "transitions": [
                                {
                                    "name": f"{stock}_leave_empty",
                                    "source": "empty",
                                    "targets": ["stocked"],
                                    "distrib": "watched",
                                    "guard": {
                                        "op": "cmp",
                                        "cmp": "ge",
                                        "lhs": share,
                                        "rhs": _float(width),
                                    },
                                },
                                {
                                    "name": f"{stock}_reach_empty",
                                    "source": "stocked",
                                    "targets": ["empty"],
                                    "distrib": "watched",
                                    "guard": {
                                        "op": "cmp",
                                        "cmp": "le",
                                        "lhs": share,
                                        "rhs": _float(0.0),
                                    },
                                },
                            ],
                        }
                    )

            # A volume serving its outputs as a mixture: the pooled
            # excess, then what each held output hands out
            # (:meth:`_pooled_serve`). Both belong to the production band,
            # swept at this component's visit once what arrives is
            # settled and before the allocations distribute them.
            pooled = [
                flow
                for flow, held in self._mixed_outputs().items()
                if held is capacity
            ]
            if pooled:
                beyond = self._beyond_attribute(capacity)
                variables.append(_float_attribute(beyond, 0.0))
                equations.append(
                    {
                        "target": beyond,
                        "kind": "explicit",
                        "expr": self._pooled_excess(capacity),
                    }
                )
                for flow in pooled:
                    variables.append(_float_attribute(f"{flow}_pooled_out", 0.0))
                    equations.append(
                        {
                            "target": f"{flow}_pooled_out",
                            "kind": "explicit",
                            "expr": self._pooled_serve(capacity, flow),
                        }
                    )

            # The discharge gate, where the volume declares one. Posted
            # beside the bounds rather than inside the served quantity so
            # that the pre-run loop diagnostic can see it: a threshold
            # inlined in a quantity closes a cycle among variables alone,
            # which is read as an algebraic loop and reported by nobody.
            if capacity.commanded:
                automata.append(self._command_gate_automaton(capacity))

            # The level, published read-only (R7): an out port carrying no
            # channel, so a reader exchanges no quantity and enters no
            # allocation operator.
            for alias, attribute in capacity.published():
                ports.append({"name": f"{alias}_out", "dir": "out", "attr": attribute})
