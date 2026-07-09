"""Causal-journal queries (M5).

The engine records a structured causal journal when asked
(``simulate(model, t_max=..., journal=True)``): transition firings,
sensitive-function triggers, variable changes (with their writing
function), schedulings, reschedulings (`reschedule_modifiable`) and drops
(`drop_disabled` / source left). This module answers the three
explainability questions, *from the journal alone*:

- :meth:`JournalQuery.why_not_fired` — why didn't transition X fire?
- :meth:`JournalQuery.who_changed` — who modified variable V?
- :meth:`JournalQuery.cascade_after` — what cascade followed event E?

>>> result = pyraichu.simulate(model, t_max=6.0, journal=True)
>>> query = pyraichu.JournalQuery(result.journal)
>>> print(query.why_not_fired("w_reset.job.finish"))
`w_reset.job.finish` did not fire: scheduled at t=0 for t=6, dropped
at t=4 (its guard turned false — reset policy, drop_disabled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "JournalQuery",
    "TransitionHistory",
    "AttributeChange",
    "Cascade",
]


def _value(raw: Any) -> Any:
    """Unwrap the serialized `{kind, value}` form of engine values."""
    if isinstance(raw, dict) and set(raw) == {"kind", "value"}:
        return raw["value"]
    return raw


_DROP_PROSE = {
    "guard_false": "its guard turned false — reset policy, drop_disabled",
    "guard_paused": "its guard turned false — countdown paused, resume policy",
    "source_left": "its automaton left the source state",
}


@dataclass
class TransitionHistory:
    """Everything the journal knows about one transition."""

    transition: str
    #: Firing dates.
    fired: list[float] = field(default_factory=list)
    #: `(time, planned firing date)` — schedulings and reschedulings.
    schedules: list[tuple[float, float]] = field(default_factory=list)
    #: `(time, reason)` — drops, reason in guard_false / guard_paused /
    #: source_left.
    drops: list[tuple[float, str]] = field(default_factory=list)
    #: Planned firing date still pending when the run ended, if any.
    still_pending: float | None = None

    @property
    def explanation(self) -> str:
        name = f"`{self.transition}`"
        if self.fired:
            dates = ", ".join(f"t={t:g}" for t in self.fired)
            return f"{name} fired at {dates}."
        if not self.schedules and not self.drops:
            return (
                f"{name} never appears in the journal: it was never "
                "armed (check the transition name, whether its source "
                "state is ever entered, and whether its guard can hold "
                "there)."
            )
        parts = [f"{name} did not fire:"]
        for when, planned in self.schedules:
            parts.append(f"scheduled at t={when:g} for t={planned:g},")
        for when, reason in self.drops:
            prose = _DROP_PROSE.get(reason, reason)
            parts.append(f"dropped at t={when:g} ({prose}),")
        if self.still_pending is not None:
            parts.append(
                f"and was still pending for t={self.still_pending:g} "
                "when the run ended."
            )
        text = " ".join(parts).rstrip(",")
        return text + ("" if text.endswith(".") else ".")

    def __str__(self) -> str:
        return self.explanation


@dataclass
class AttributeChange:
    """One write to a variable, with its full causal context."""

    time: float
    attribute: str
    old: Any
    new: Any
    #: Sensitive function that wrote the value.
    cause: str
    #: Transition whose firing (directly or through the cascade)
    #: triggered the write — `"initialization"` for the t = 0 fixpoint.
    trigger: str

    def __str__(self) -> str:
        return (
            f"t={self.time:g}: `{self.attribute}` {self.old!r} -> "
            f"{self.new!r} by `{self.cause}` (after `{self.trigger}`)"
        )


@dataclass
class Cascade:
    """Direct consequences of one transition firing: the sensitive
    functions triggered, the variables rewritten and the schedule
    updates, up to the next firing."""

    transition: str
    time: float
    from_state: str
    to_state: str
    functions: list[str] = field(default_factory=list)
    changes: list[AttributeChange] = field(default_factory=list)
    #: `(transition, planned firing date)` — schedulings and
    #: reschedulings caused by this event.
    scheduled: list[tuple[str, float]] = field(default_factory=list)
    #: `(transition, reason)` — drops caused by this event.
    dropped: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"t={self.time:g}: `{self.transition}` "
            f"({self.from_state} -> {self.to_state})"
        ]
        for change in self.changes:
            lines.append(
                f"  - `{change.attribute}` {change.old!r} -> "
                f"{change.new!r} by `{change.cause}`"
            )
        for name, planned in self.scheduled:
            lines.append(f"  - scheduled `{name}` for t={planned:g}")
        for name, reason in self.dropped:
            prose = _DROP_PROSE.get(reason, reason)
            lines.append(f"  - dropped `{name}` ({prose})")
        return "\n".join(lines)


class JournalQuery:
    """Query layer over the causal journal of one simulation run.

    Build from ``result.journal`` (requires ``journal=True`` at
    simulation time — the journal is zero-cost when disabled and
    therefore empty here otherwise).
    """

    def __init__(self, journal: list[dict[str, Any]]):
        if not journal:
            raise ValueError(
                "empty journal — run simulate(..., journal=True)"
            )
        self._journal = journal

    # -- inventories ---------------------------------------------------

    def transitions(self) -> list[str]:
        """Transition names appearing anywhere in the journal."""
        names = {
            record["transition"]
            for record in self._journal
            if "transition" in record
        }
        return sorted(names)

    def attributes(self) -> list[str]:
        """Variable names appearing in change records."""
        names = {
            record["attribute"]
            for record in self._journal
            if record["record"] == "attribute_changed"
        }
        return sorted(names)

    # -- the three explainability questions -----------------------------

    def why_not_fired(self, transition: str) -> TransitionHistory:
        """Why didn't `transition` fire? (Or when did it, if it did.)"""
        history = TransitionHistory(transition=transition)
        pending: float | None = None
        for record in self._journal:
            if record.get("transition") != transition:
                continue
            kind = record["record"]
            if kind == "transition_fired":
                history.fired.append(record["time"])
                pending = None
            elif kind in ("transition_scheduled", "transition_rescheduled"):
                history.schedules.append((record["time"], record["firing_at"]))
                pending = record["firing_at"]
            elif kind == "transition_dropped":
                history.drops.append((record["time"], record["reason"]))
                pending = None
        history.still_pending = pending
        return history

    def who_changed(self, attribute: str) -> list[AttributeChange]:
        """Every write to `variable`, with the writing function and the
        transition firing it reacted to."""
        changes: list[AttributeChange] = []
        trigger = "initialization"
        for record in self._journal:
            if record["record"] == "transition_fired":
                trigger = record["transition"]
            elif (
                record["record"] == "attribute_changed"
                and record["attribute"] == attribute
            ):
                changes.append(
                    AttributeChange(
                        time=record["time"],
                        attribute=attribute,
                        old=_value(record["old"]),
                        new=_value(record["new"]),
                        cause=record["cause"],
                        trigger=trigger,
                    )
                )
        return changes

    def cascade_after(self, transition: str, occurrence: int = 0) -> Cascade:
        """Direct consequences of the `occurrence`-th firing of
        `transition`: everything the journal records from that firing
        (excluded) to the next firing (excluded)."""
        seen = -1
        start = None
        for index, record in enumerate(self._journal):
            if (
                record["record"] == "transition_fired"
                and record["transition"] == transition
            ):
                seen += 1
                if seen == occurrence:
                    start = index
                    break
        if start is None:
            raise ValueError(
                f"`{transition}` fired {seen + 1} time(s) — no "
                f"occurrence #{occurrence} in the journal"
            )
        fired = self._journal[start]
        cascade = Cascade(
            transition=transition,
            time=fired["time"],
            from_state=fired["from"],
            to_state=fired["to"],
        )
        for record in self._journal[start + 1 :]:
            kind = record["record"]
            if kind == "transition_fired":
                break
            if kind == "function_triggered":
                cascade.functions.append(record["function"])
            elif kind == "attribute_changed":
                cascade.changes.append(
                    AttributeChange(
                        time=record["time"],
                        attribute=record["attribute"],
                        old=_value(record["old"]),
                        new=_value(record["new"]),
                        cause=record["cause"],
                        trigger=transition,
                    )
                )
            elif kind in ("transition_scheduled", "transition_rescheduled"):
                cascade.scheduled.append(
                    (record["transition"], record["firing_at"])
                )
            elif kind == "transition_dropped":
                cascade.dropped.append((record["transition"], record["reason"]))
        return cascade
