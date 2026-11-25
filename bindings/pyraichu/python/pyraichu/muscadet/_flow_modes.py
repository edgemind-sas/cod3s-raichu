"""Failure modes, deratings and taps of a component."""

from __future__ import annotations

import re
from typing import Any

from ._expr import _float, _float_attribute, _matches_flow, _state_active, _sum, _var
from ._records import _derating_variable, _Effects, _FailureMode
from ._vocabulary import NO_TAP, NOMINAL_RATE


class _FailureModesMixin:
    """Methods of :class:`ObjFlow` failure modes, deratings and taps of a component."""

    def add_delay_failure_mode(
        self,
        name: str,
        failure_time: float,
        repair_time: float,
        targets: list[str] | None = None,
        failure_cond: str | None = None,
        failure_effects: list[tuple[str, Any]] | None = None,
        repair_effects: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Deterministic failure/repair mode driving the out-flows'
        availability (all out-flows unless `targets` names some).
        `failure_cond` names a local variable gating the failure
        (muscadet's `cond_occ_12`).

        `failure_effects` and `repair_effects` **derate** continuous
        outputs: see :meth:`_resolve_effects` for their shape, for the
        cap/tap distinction, and for why the return to nominal needs no
        declaration."""
        self.failure_modes.append(
            _FailureMode(
                name=name,
                law="delay",
                failure_param=failure_time,
                repair_param=repair_time,
                targets=list(targets or []),
                failure_cond=failure_cond,
                failure=self._resolve_effects(
                    name, "failure_effects", failure_effects
                ),
                repair=self._resolve_effects(
                    name, "repair_effects", repair_effects
                ),
            )
        )

    # --- deratings ------------------------------------------------------

    def add_derating(self, mode: str, flow: str) -> str:
        """Allocate the derating variable a mode declared OUTSIDE this
        component clamps on one of its continuous outputs, and return its
        name, ``{mode}_derating_{flow}``.

        muscadet's ``ObjFlow.add_derating`` (format 1.0.3, the
        ``deratings`` section): a standalone mode targets the component,
        and its effects write this variable by name. The variable starts
        at the nominal 1 and joins the output's effective rate by
        **minimum**, beside the shared ``{flow}_out_rate`` and the modes
        declared on the component, so two standalone modes on one output
        compose like any two caps and a repair of one leaves the other's
        cap standing. A mode owns its variable, so it owns the release:
        nothing here writes it after allocation.

        Allocating the same pair twice is a no-op, as in muscadet, so a
        declaration read back and replayed is not refused for its own
        echo. A flow that is not a continuous output of this component is
        refused: a derating reaching nothing would build, run, and report
        a plant whose modelled failure never happened."""
        outputs = [declared.name for declared in self.flows_continuous_out]
        if flow not in outputs:
            raise ValueError(
                f"ObjFlow `{self.name}`: derating for mode `{mode}` names `{flow}`, "
                f"which is not a continuous output of this component. Its "
                f"continuous outputs are {outputs or 'none'}"
            )
        if (mode, flow) not in self.external_deratings:
            self.external_deratings.append((mode, flow))
        return _derating_variable(mode, flow)

    def _external_deratings_on(self, flow: str) -> list[str]:
        """The variables the standalone modes derating `flow` clamp."""
        return [
            _derating_variable(mode, derated)
            for mode, derated in self.external_deratings
            if derated == flow
        ]

    def _derated(self, flow: str) -> bool:
        """Whether `flow` carries an effective rate of its own: a mode
        declared here caps it, or a standalone mode derates it."""
        return bool(self._caps_on(flow)) or bool(self._external_deratings_on(flow))

    def _resolve_effects(
        self, mode: str, key: str, effects: list[tuple[str, Any]] | None
    ) -> _Effects:
        """What a mode's declared effects do to the continuous outputs
        they name (R6, R18).

        An effect is a ``(pattern, value)`` pair, the pattern being a
        regular expression matched **anchored** against both the flow
        name and the ``{flow}_fed_out`` alias, which is the 1.x spelling
        of an effect on an output. Anchored, and not by chance: unanchored,
        ``"H2"`` would name ``H2O`` as well and a declaration meant for
        one output would silently derate its neighbour.

        The value says which of the two things a mode can do to an output
        it names, and the two are not interchangeable:

        - a bare number, or the equivalent ``{"cap": x}``, is a **cap**:
          what the mode LEAVES of the output, in ``[0, 1]``. ``False`` is
          the muscadet idiom for a total loss and reads as zero. Caps
          compose by **minimum**, the binding constraint winning, which
          is what a derated capacity is: a valve stuck at 0.8 and a pump
          degraded to 0.9 leave 0.8 of the output between them;
        - ``{"tap": x}`` is a **tap**: what fraction the mode TAKES OFF
          the output, in ``[0, 1]``. Taps compose by **sum**, being
          parallel draws on one stream: leaks of 0.1 and 0.2 leave 0.7,
          where a minimum would leave 0.8 and make the second leak free.
          ``{"tap": x, "to": "other"}`` delivers what it takes to another
          continuous out-flow of this component, which is what closes the
          mass balance; without ``to`` the fraction leaves the system, a
          loss to the environment being a legitimate model.

        There is no separate boolean gate on a continuous flow, so a cap
        of zero is what stops production entirely.

        A pattern reaching no continuous output is **refused** rather
        than dropped: an effect that reaches nothing builds, runs to
        completion and reports the availability figures of a plant whose
        modelled failure never happened, which is the defect class this
        refusal exists to close. A discrete out-flow is gated through
        `targets`, not here."""
        resolved = _Effects()
        where = f"ObjFlow `{self.name}`: failure mode `{mode}`"
        for entry in effects or []:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    f"{where} declares the {key} entry {entry!r}; an effect is "
                    "a (pattern, value) pair"
                )
            pattern, value = entry
            matched = self._effect_targets(where, key, pattern)
            kind, rate, destination = self._effect_value(where, key, pattern, value)
            if kind == "cap":
                for flow in matched:
                    resolved.caps[flow] = rate
                continue
            if destination is not None and len(matched) > 1:
                raise ValueError(
                    f"{where} declares the {key} pattern `{pattern}` as a tap "
                    f"delivering to `{destination}`, but the pattern names "
                    f"{matched}. A routed tap moves matter out of ONE stream, "
                    "so its pattern names a single output; declare one effect "
                    "per tapped flow"
                )
            for flow in matched:
                resolved.taps[flow] = rate
                if destination is not None:
                    resolved.routes[flow] = destination
        return resolved

    def _effect_targets(self, where: str, key: str, pattern: Any) -> list[str]:
        """The continuous out-flows an effect pattern names, refusing a
        pattern that names none."""
        try:
            matched = [
                flow.name
                for flow in self.flows_continuous_out
                if _matches_flow(pattern, flow.name)
                or _matches_flow(pattern, f"{flow.name}_fed_out")
            ]
        except re.error as error:
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}`, which is "
                f"not a regular expression: {error}"
            ) from None
        if not matched:
            declared = [flow.name for flow in self.flows_continuous_out]
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}`, which "
                f"names no continuous out-flow of this component (it "
                f"declares {declared}). An effect reaching nothing is a "
                "silent no-op; a discrete out-flow is gated through "
                "`targets` instead"
            )
        return matched

    def _effect_value(
        self, where: str, key: str, pattern: Any, value: Any
    ) -> tuple[str, float, str | None]:
        """One effect's value, as ``(kind, rate, destination)`` with kind
        ``"cap"`` or ``"tap"``.

        The spelling carries which convention is meant, never the
        magnitude: nothing in the number 0.1 distinguishes a tenth left
        from a tenth taken, so a value that does not say is a cap by the
        1.x reading and never a guess made on the figure."""
        if not isinstance(value, dict):
            return ("cap", self._effect_rate(where, key, pattern, "cap", value), None)
        unknown = sorted(set(value) - {"cap", "tap", "to"})
        if unknown:
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}` with the "
                f"unknown key(s) {unknown}; an effect carries `cap` (what it "
                "leaves), `tap` (what it takes off) and, on a tap, `to` "
                "(where what it takes is delivered)"
            )
        if ("cap" in value) == ("tap" in value):
            both = "both" if "cap" in value else "neither"
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}` with {both} "
                "`cap` and `tap`. An effect either CAPS the output at what it "
                "leaves or TAPS a fraction off it; the two are opposite "
                "conventions over the same interval, so one effect declares "
                "exactly one of them"
            )
        if "cap" in value:
            if "to" in value:
                raise ValueError(
                    f"{where} declares the {key} pattern `{pattern}` as a cap "
                    "carrying `to`. A cap bounds what the output delivers and "
                    "moves nothing, so it has nothing to deliver anywhere; "
                    "`to` belongs to a tap"
                )
            return (
                "cap",
                self._effect_rate(where, key, pattern, "cap", value["cap"]),
                None,
            )
        destination = value.get("to")
        if destination is not None and not isinstance(destination, str):
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}` with "
                f"`to` = {destination!r}; a route names one continuous "
                "out-flow of this component"
            )
        return (
            "tap",
            self._effect_rate(where, key, pattern, "tap", value["tap"]),
            destination,
        )

    def _effect_rate(
        self, where: str, key: str, pattern: Any, kind: str, value: Any
    ) -> float:
        """One effect's fraction, refused outside ``[0, 1]``. The message
        states the convention the number is read in, which is the whole
        of what separates the two kinds."""
        meaning = (
            "what the mode LEAVES of the output"
            if kind == "cap"
            else "what fraction the mode TAKES OFF the output"
        )
        try:
            rate = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}` at "
                f"{value!r}; a {kind} is {meaning}, so it is a number in "
                "[0, 1]"
            ) from None
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"{where} declares the {key} pattern `{pattern}` at "
                f"{rate:g}; a {kind} is {meaning}, so it lies in [0, 1]"
            )
        return rate

    def _caps_on(self, flow: str) -> list[tuple[_FailureMode, float, float]]:
        """The modes capping `flow`, with what each leaves of it while it
        stands and once repaired.

        The repair value defaults to :data:`NOMINAL_RATE`: a mode owns
        its cap, so it owns the release, and a cap has no per-step reset
        of its own. A mode declaring a repair effect keeps that value
        instead, a mode returning degraded rather than as-new being a
        legitimate model."""
        return [
            (
                mode,
                mode.failure.caps.get(flow, NOMINAL_RATE),
                mode.repair.caps.get(flow, NOMINAL_RATE),
            )
            for mode in self.failure_modes
            if flow in mode.failure.caps or flow in mode.repair.caps
        ]

    def _taps_on(self, flow: str) -> list[tuple[_FailureMode, float, float]]:
        """The modes tapping `flow`, with what fraction each takes off it
        while it stands and once repaired.

        The repair value defaults to :data:`NO_TAP` for the reason a
        cap's defaults to :data:`NOMINAL_RATE`: the release belongs to
        the mode, and this channel's neutral element is zero."""
        return [
            (
                mode,
                mode.failure.taps.get(flow, NO_TAP),
                mode.repair.taps.get(flow, NO_TAP),
            )
            for mode in self.failure_modes
            if flow in mode.failure.taps or flow in mode.repair.taps
        ]

    def _tap_route(self, flow: str, mode: _FailureMode) -> str | None:
        """Where `mode`'s tap on `flow` delivers, or ``None`` when the
        fraction leaves the system.

        Read from either side: the route is a property of the tap, not of
        the state it is read in. Two sides naming different destinations
        is refused by :meth:`_refuse_ill_formed_taps`, so the two lookups
        cannot disagree by the time this is read."""
        return mode.failure.routes.get(flow) or mode.repair.routes.get(flow)

    def _tap_ordered_outputs(self) -> list[str]:
        """The continuous out-flows, ordered so that a stream comes
        before every output its taps deliver into.

        Declaration order will not do: an output that receives a tap
        reads the base of the stream it is fed from, and nothing stops a
        modeller from declaring the receiver first. Chained taps then
        settle in the wrong order and the receiver sweeps on a base its
        source has not written yet. The walk terminates because
        :meth:`_refuse_ill_formed_taps` has already ruled out a cycle."""
        routes: dict[str, str] = {}
        for flow in self.flows_continuous_out:
            for mode, _, _ in self._taps_on(flow.name):
                destination = self._tap_route(flow.name, mode)
                if destination is not None:
                    routes[flow.name] = destination

        ordered: list[str] = []
        placed: set[str] = set()

        def place(flow: str) -> None:
            if flow in placed:
                return
            placed.add(flow)
            for source, destination in routes.items():
                if destination == flow:
                    place(source)
            ordered.append(flow)

        for flow in self.flows_continuous_out:
            place(flow.name)
        return ordered

    def _tapped_into(
        self, destination: str
    ) -> list[tuple[str, _FailureMode, float, float]]:
        """The taps delivering into `destination`: the stream each draws
        from, the mode owning it, and its two fractions.

        Walked in declaration order of the streams and then of the modes,
        so what the receiving output sums is ordered by the declaration
        and not by a dictionary."""
        found: list[tuple[str, _FailureMode, float, float]] = []
        for flow in self.flows_continuous_out:
            for mode, failed, repaired in self._taps_on(flow.name):
                if self._tap_route(flow.name, mode) == destination:
                    found.append((flow.name, mode, failed, repaired))
        return found

    def _tap_split(
        self,
        flow: str,
        channel: str,
        expr: dict[str, Any],
        variables: list[dict],
        equations: list[dict],
    ) -> dict[str, Any]:
        """`expr` with this output's taps taken off it, publishing what
        it carried before they drew as ``{flow}_pre_tap_{channel}_out``.

        Published, and not inlined, because the outputs a tap delivers
        into read it: both sides of a routed tap then divide one and the
        same number, so what one stream loses is exactly what the others
        gain and the component's balance closes by construction rather
        than by two expressions agreeing. An output nothing taps is
        returned untouched and publishes nothing extra."""
        if not self._taps_on(flow):
            return expr
        pre_tap = f"{flow}_pre_tap_{channel}_out"
        variables.append(_float_attribute(pre_tap, 0.0))
        equations.append({"target": pre_tap, "kind": "explicit", "expr": expr})
        return {
            "op": "mul",
            "args": [_var(self.name, pre_tap), _var(self.name, f"{flow}_tap_rate")],
        }

    def _tap_delivery(self, destination: str, channel: str) -> list[dict[str, Any]]:
        """What each tap routed into `destination` hands it: the fraction
        it takes, of what its own stream carried before any tap drew.

        A source publishing what it actually made is read on that
        quantity; one that publishes only what it could deliver is read
        on that, which is what it delivers. The two cannot be mixed up:
        the channel a source is read on is decided by
        :meth:`_produced_outputs`, the same predicate that decides
        whether the source publishes a production at all."""
        produced = self._produced_outputs()
        terms = []
        for source, mode, failed, repaired in self._tapped_into(destination):
            made = channel == "produced" and source in produced
            carried = "produced" if made else "capability"
            terms.append(
                {
                    "op": "mul",
                    "args": [
                        _var(self.name, f"{source}_pre_tap_{carried}_out"),
                        self._mode_term(mode, failed, repaired),
                    ],
                }
            )
        return terms

    def _mode_term(
        self, mode: _FailureMode, failed: float, repaired: float
    ) -> dict[str, Any]:
        """What one mode contributes to a channel, read from its
        automaton **location** rather than from a variable it wrote.

        That is what makes the return to nominal implicit on both
        channels: leaving the failing state restores the contribution
        with nothing declared on the other side."""
        return {
            "op": "if",
            "cond": _state_active(self.name, mode.name, "nok"),
            "then": _float(failed),
            "otherwise": _float(repaired),
        }

    def _tap_rate_expr(self, flow: str) -> dict[str, Any]:
        """What the taps LEAVE of `flow`: one minus the sum of what each
        mode takes off it (R6).

        A sum and not a minimum, because taps are parallel draws on one
        stream: what leaves by one does not pass the other, so two leaks
        of 0.1 and 0.2 leave 0.7. Folded by minimum they would leave 0.8,
        and the second leak would cost nothing. The sum is bounded at one
        by :meth:`_refuse_ill_formed_taps`, at build time and over all
        the modes at once, so this expression needs no clamp and the
        component cannot be relieved of more than it carries."""
        taken = [
            self._mode_term(mode, failed, repaired)
            for mode, failed, repaired in self._taps_on(flow)
        ]
        return {"op": "sub", "lhs": _float(NOMINAL_RATE), "rhs": _sum(taken)}

    def add_exp_failure_mode(
        self,
        name: str,
        failure_rate: float,
        repair_rate: float,
        targets: list[str] | None = None,
        failure_cond: str | None = None,
        failure_effects: list[tuple[str, Any]] | None = None,
        repair_effects: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Exponential failure/repair mode (statistical regime).

        `failure_effects` and `repair_effects` derate continuous outputs,
        exactly as on :meth:`add_delay_failure_mode`."""
        self.failure_modes.append(
            _FailureMode(
                name=name,
                law="exp",
                failure_param=failure_rate,
                repair_param=repair_rate,
                targets=list(targets or []),
                failure_cond=failure_cond,
                failure=self._resolve_effects(
                    name, "failure_effects", failure_effects
                ),
                repair=self._resolve_effects(
                    name, "repair_effects", repair_effects
                ),
            )
        )

    def _build_failure_modes(self, automata: list[dict]) -> None:
        """The ok/nok automaton of each declared failure mode, its two
        transitions carrying the delay or the rate the law asks for (one
        only for an exponential mode whose repair rate is zero, which is
        non-repairable).

        The deratings a mode declares are not written here: they are read
        off this automaton's **location** by the outputs they bear on,
        which is what makes the return to nominal implicit."""
        me = self.name
        for mode in self.failure_modes:
            failure_transition: dict[str, Any] = {
                "name": "failure",
                "source": "ok",
                "targets": ["nok"],
                "distrib": mode.law,
                ("time" if mode.law == "delay" else "rate"): mode.failure_param,
            }
            if mode.failure_cond is not None:
                failure_transition["guard"] = _var(me, mode.failure_cond)
            transitions = [failure_transition]
            # An exponential repair at rate zero never fires: the mode is
            # non-repairable, and `nok` is absorbing. muscadet builds that
            # edge with its zero rate (the reference engine never draws it);
            # the engine here refuses a zero rate, so the edge is left out,
            # which is the same trajectory.
            if not (mode.law == "exp" and float(mode.repair_param) == 0.0):
                transitions.append(
                    {
                        "name": "repair",
                        "source": "nok",
                        "targets": ["ok"],
                        "distrib": mode.law,
                        ("time" if mode.law == "delay" else "rate"): mode.repair_param,
                    }
                )
            automata.append(
                {
                    "name": mode.name,
                    "states": ["ok", "nok"],
                    "init": "ok",
                    "transitions": transitions,
                }
            )
