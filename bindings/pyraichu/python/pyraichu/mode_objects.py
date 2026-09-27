"""The muscadet-plugin object a cod3s mode or event declaration means.

A two-state component -- a failure mode, or a standalone event -- is declared
in cod3s' own vocabulary: the keyword arguments ``cod3s.ObjMode2S`` /
``ObjFMExp`` / ``ObjFMDelay`` / ``ObjFMInst`` / ``ObjEvent`` take, spelled in
data. This module is the one translation from that wire to the objects
:mod:`pyraichu.plugins.muscadet` expands, and it holds the whole of the law
matrix: three occurrence kinds on each of the two directions, the three
behaviours, the per-common-cause-order parameter vectors, and the on-demand
draw.

Its caller is :mod:`pyraichu.declare`, which reads a muscadet system
declaration and hands each two-state component's wire through here rather than
writing a second, lesser expansion of the same semantics. It lives apart from
that reader because the two vocabularies are not the same one: ``declare``
reads what muscadet writes, this module reads what cod3s' constructors take,
and a declaration is translated from the first into the second before anything
is built.

Anything the wire carries and this translation cannot build raises a typed
:class:`TranslationError` naming the offending field: a mode that translates is
a mode whose semantics are covered.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TranslationError",
    "event_object",
    "failure_mode_object",
]


class TranslationError(ValueError):
    """A cod3s mode wire uses a construct this translation does not cover."""


def _require(mapping: dict, key: str, *, where: str) -> Any:
    """Fetch a required key with a typed, contextual error (never a raw
    KeyError: the fail-fast contract of this module)."""
    try:
        return mapping[key]
    except KeyError:
        raise TranslationError(f"{where}: missing required key {key!r}") from None


# --- the occurrence and return laws ------------------------------------------

_FM_LAWS = {"ObjFMExp": ("exp", "rate"), "ObjFMDelay": ("delay", "time")}

# Native `cls: ObjMode2S` wire (the production translator's vocabulary,
# `study_translator_service.py:_emit_objmode2s_spec`): per-direction law
# dicts, the value key each law carries and the parameter variable name it
# is bound to (lambda/ttf/gamma on the occurrence, mu/ttr/repair_gamma on
# the repair). The names are decorative here (RAICHU bakes the values, it
# has no parameter variables), but a name that disagrees with the declared
# law is a wire inconsistency worth refusing rather than ignoring.
_LAW_KINDS = ("exp", "delay", "inst")
_LAW_VALUE_KEY = {"exp": "rate", "delay": "time", "inst": "prob"}
_OCC_PARAM_NAMES = {"exp": "lambda", "delay": "ttf", "inst": "gamma"}
_REP_PARAM_NAMES = {"exp": "mu", "delay": "ttr", "inst": "repair_gamma"}


def _one_shot_effects(fm: dict, spec: dict, pairs: tuple[tuple[str, str], ...]) -> None:
    """Carry a mode's one-shot (trans-based) effects into the plugin spec.

    They are written ONCE, on the firing edge of the occurrence or of the
    return (a transition's `effects`), where the state effects are held
    while the state lasts. The plugin's expansion checks the shapes cod3s
    itself refuses (common cause, an on-demand law, a behaviour with no
    symmetric edge pair, a variable driven both ways)."""
    for source, target in pairs:
        if fm.get(source):
            spec[target] = dict(fm[source])


def _order_law(kind: str, raw: Any, where: str) -> dict | None:
    """One entry of a native law vector → the plugin's per-order law dict,
    `None` for an inactive order (cod3s `ModeLaw*.is_active_value`):
    `None` is the explicit inactive-order marker, a zero *exp* rate is
    inactive too (an order-0 rate builds no automaton, a zero repair rate
    leaves `occ` absorbing), while a zero *delay* is active and immediate
    and an *inst* prob of 0 is a valid never-drawing mode."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise TranslationError(f"{where}: invalid {_LAW_VALUE_KEY[kind]} value {raw!r}") from None
    if kind == "exp" and value <= 0.0:
        return None
    if kind == "inst":
        if not 0.0 <= value <= 1.0:
            raise TranslationError(f"{where}: inst prob must be within [0, 1], got {value}")
        return {"law": "inst", "prob": value}
    if value < 0.0:
        raise TranslationError(f"{where}: {kind} law value must be >= 0, got {value}")
    return {"law": kind, _LAW_VALUE_KEY[kind]: value}


def _direction_laws(law: Any, n_targets: int, where: str) -> tuple[str, list[dict | None]]:
    """One native per-direction law (`{cls, rate|time|prob: [vector]}`) →
    `(kind, per-order law list)`. The vector is the per-CC-order one, padded
    to the target count with inactive orders (cod3s pads `exp` with 0.0 and
    `delay`/`inst` with `None`, both read as inactive here)."""
    if not isinstance(law, dict):
        raise TranslationError(f"{where}: law {law!r} must be a {{cls, value}} dict")
    kind = law.get("cls")
    if kind not in _LAW_KINDS:
        raise TranslationError(
            f"{where}: law cls {kind!r} not supported (expected one of {list(_LAW_KINDS)})"
        )
    raw = law.get(_LAW_VALUE_KEY[kind])
    if raw is None:
        raise TranslationError(
            f"{where}: `{kind}` law carries no {_LAW_VALUE_KEY[kind]!r} value"
        )
    vector = raw if isinstance(raw, list) else [raw]
    if len(vector) > n_targets:
        raise TranslationError(
            f"{where}: the {kind} law vector has {len(vector)} entries for "
            f"{n_targets} target(s) (one entry per CC order, at most one per target)"
        )
    return kind, [
        _order_law(kind, vector[i] if i < len(vector) else None, where) for i in range(n_targets)
    ]


def _check_param_names(fm: dict, occ_kind: str, rep_kind: str, where: str) -> None:
    """`occ_param_name` / `not_occ_param_name` name the parameter variable
    each direction's law is bound to; the platform derives them from the
    law, so a mismatch is a wire inconsistency."""
    for key, kind, expected in (
        ("occ_param_name", occ_kind, _OCC_PARAM_NAMES),
        ("not_occ_param_name", rep_kind, _REP_PARAM_NAMES),
    ):
        names = fm.get(key)
        if names is None:
            continue
        if list(names) != [expected[kind]]:
            raise TranslationError(
                f"{where}: {key} {names!r} disagrees with the {kind!r} law "
                f"(expected {[expected[kind]]})"
            )


#: Sentinel for "this alias has no schema default", distinct from a legacy
#: spelling whose default legitimately *is* ``None``.
_NO_DEFAULT = object()


def _alias(
    fm: dict,
    native: str,
    legacy: str,
    where: str,
    *,
    legacy_default: Any = _NO_DEFAULT,
) -> Any:
    """Read one field under its native and its legacy spelling.

    Both present and disagreeing is a refusal (cod3s refuses a double set
    the same way); otherwise the native one wins and the legacy one is the
    fallback.

    ``legacy_default`` is what keeps that refusal honest on a **serialised**
    study. A platform study.yaml is dumped from a Pydantic model, so it
    carries the schema default of every field the author never touched: the
    presence of a key proves nothing about intent. cod3s declares
    ``failure_cond`` and ``repair_cond`` with a default of ``True``
    ("always fireable"), and the platform writes the real condition under
    the native spelling, so every mode carrying a repair condition arrived
    here with ``not_occ_cond`` set AND ``repair_cond: True`` beside it, and
    was refused for a contradiction nobody had written (COD3S #175). A
    legacy spelling left at its declared default therefore carries no
    intent and does not contradict anything.
    """

    def _meaningful(key: str) -> bool:
        # A spelling carries intent when it is present AND says something
        # its default does not already say.
        return key in fm and not (
            legacy_default is not _NO_DEFAULT and fm[key] == legacy_default
        )

    if _meaningful(native) and _meaningful(legacy) and fm[native] != fm[legacy]:
        raise TranslationError(
            f"{where}: both {native!r} and its legacy alias {legacy!r} are "
            "set with different values: set exactly one"
        )
    if _meaningful(native):
        return fm[native]
    if _meaningful(legacy):
        return fm[legacy]
    if legacy_default is not _NO_DEFAULT:
        # Neither spelling says anything. Absent and "left at the default"
        # are the same statement, so answer as if the field were absent:
        # otherwise a serialised study would build a spec that a hand-written
        # one does not, for a field nobody set.
        return None
    return fm[native] if native in fm else fm.get(legacy)


# --- a failure mode ----------------------------------------------------------

def failure_mode_object(fm: dict) -> dict:
    """The muscadet-plugin object a cod3s failure-mode declaration means.

    The one translation from the cod3s mode wire to
    :mod:`pyraichu.plugins.muscadet`: a muscadet system declaration carries
    that vocabulary for a standalone mode -- it is the mode class's own
    constructor keywords either way -- and :mod:`pyraichu.declare` reads it
    through here rather than writing a second, lesser expansion of its own.

    What the caller owes is a normalised mapping: the ``cls`` the class table
    knows, one value per common-cause order rather than a document's nested
    lists, and condition leaves carrying the object they watch. What it gets
    back is an ``ObjFM`` or ``ObjFMInst`` object, named after the mode.

    Raises
    ------
    TranslationError
        For a mode class or a construct this expansion does not cover.
    """
    return _translate_failure_mode(fm)


def _translate_failure_mode(fm: dict) -> dict:
    cls = fm.get("cls", "ObjFMExp")
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"
    if cls in _FM_LAWS:
        return _translate_legacy_fm(fm, cls)
    if cls == "ObjMode2S":
        return _translate_objmode2s(fm)
    if cls == "ObjFMInst":
        return _translate_objfminst(fm)
    raise TranslationError(f"{where}: cls {cls!r} not supported")


def _translate_legacy_fm(fm: dict, cls: str) -> dict:
    """Legacy `ObjFMExp` / `ObjFMDelay` dialect: one scalar per CC order,
    the occurrence law carried by `cls` itself."""
    law, key = _FM_LAWS[cls]
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"

    def order_law(p):
        # cod3s marks an INACTIVE common-cause order with a zero rate
        # (`is_occ_law_*_active` = param > 0, `drop_inactive_automata`):
        # normalise to None so the plugin drops the order. Exp only: a
        # zero *delay* is a legitimate immediate transition.
        if p is None or (law == "exp" and float(p) <= 0.0):
            return None
        return {"law": law, key: float(p)}

    spec = {
        "type": "ObjFM",
        "name": _require(fm, "fm_name", where=where),
        "targets": list(_require(fm, "targets", where=where)),
        "behaviour": fm.get("behaviour", "internal"),
        "failure": [order_law(p) for p in _require(fm, "failure_param", where=where)],
        "repair": [order_law(p) for p in _require(fm, "repair_param", where=where)],
        "failure_effects": dict(fm.get("failure_effects") or {}),
    }
    if fm.get("repair_effects"):
        spec["repair_effects"] = dict(fm["repair_effects"])
    _one_shot_effects(
        fm,
        spec,
        (
            ("failure_effects_trans", "failure_effects_trans"),
            ("repair_effects_trans", "repair_effects_trans"),
        ),
    )
    for cond in ("failure_cond", "repair_cond"):
        if cond in fm:
            spec[cond] = fm[cond]
    for state in ("failure_state", "repair_state"):
        if state in fm:
            spec[state] = fm[state]
    return spec


def _translate_objmode2s(fm: dict) -> dict:
    """Native `cls: ObjMode2S` wire → the plugin `ObjFM` / `ObjFMInst`
    spec: a normalisation, not a second expansion path. Everything the
    production translator emits has a reader here, the one-shot
    `*_effects_trans` included (written on the firing edge)."""
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"
    targets = list(_require(fm, "targets", where=where))
    n = len(targets)
    occ_kind, failure = _direction_laws(_require(fm, "occ_law", where=where), n, where)
    rep_kind, repair = _direction_laws(_require(fm, "not_occ_law", where=where), n, where)
    _check_param_names(fm, occ_kind, rep_kind, where)

    behaviour = fm.get("behaviour", "internal")
    on_demand = occ_kind == "inst"

    spec: dict[str, Any] = {
        "type": _on_demand_expander(on_demand, behaviour),
        "name": _require(fm, "fm_name", where=where),
        "targets": targets,
        "behaviour": behaviour,
        "failure": failure,
        "repair": repair,
        # State effects: `occ_effects` / `not_occ_effects` are the same
        # state-clamped reading the plugin holds while in `occ` / `rep`.
        "failure_effects": dict(fm.get("occ_effects") or {}),
    }
    if fm.get("not_occ_effects"):
        spec["repair_effects"] = dict(fm["not_occ_effects"])
    _one_shot_effects(
        fm,
        spec,
        (
            ("occ_effects_trans", "failure_effects_trans"),
            ("not_occ_effects_trans", "repair_effects_trans"),
        ),
    )

    # Conditions: `failure_cond` is the wire alias of `occ_cond`, and the
    # repair face is `not_occ_cond` (`repair_cond` being its legacy
    # spelling, which the pre-native dialects carry).
    occ_cond = _alias(fm, "occ_cond", "failure_cond", where, legacy_default=True)
    if occ_cond is not None:
        spec["failure_cond"] = occ_cond
    not_occ_cond = _alias(fm, "not_occ_cond", "repair_cond", where, legacy_default=True)
    if not_occ_cond is not None:
        spec["repair_cond"] = not_occ_cond

    # State names and, for an on-demand occurrence, the parked micro-state
    # a lost draw waits in (cod3s' ObjFMInst grammar:
    # `not_<failure_state>`, pinned by the platform). The absent
    # `not_occ_state` falls back on `rep` and deliberately not on cod3s'
    # engine default (`not_occ`): on an on-demand mode that name is the
    # parked state's, and the two would collide into one state.
    spec["failure_state"] = fm.get("occ_state", "occ")
    spec["repair_state"] = fm.get("not_occ_state", "rep")
    if on_demand and "occ_parked_state" in fm:
        spec["absorb_state"] = fm["occ_parked_state"]
    return spec


def _on_demand_expander(on_demand: bool, behaviour: str) -> str:
    """Which plugin object type an occurrence law routes to.

    An on-demand (`inst`) occurrence has two readers, and the behaviour
    decides which one. `ObjFMInst` is the dedicated on-demand expander and
    builds the `internal` behaviour only, which is why an external mode
    used to be refused here rather than silently built internal. `ObjFM`
    is the unified one: it reads the whole 3x3 law matrix, the `inst`
    cell included, under all three behaviours, so an external on-demand
    mode goes there instead.

    `internal` deliberately keeps the reader it has always had. The two
    expanders build the same states and the same draw / re-arm / return
    edges, but not the same monitoring: `ObjFM` carries the mission's
    sequence events on those edges and `ObjFMInst` does not, so moving
    `internal` over would change what every already-validated on-demand
    study reports. It costs nothing to leave where it is, since the
    behaviour an external mode needs is exactly what `ObjFMInst` lacks.
    """
    if not on_demand:
        return "ObjFM"
    return "ObjFMInst" if behaviour == "internal" else "ObjFM"


def _legacy_inst_gamma(value: Any, where: str) -> dict | None:
    """One `failure_param` entry of the legacy on-demand dialect → the
    per-order occurrence law dict. `None` stays the inactive-order marker;
    an inst prob of 0 is a valid never-drawing order, as on the native
    wire."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("prob", value.get("gamma"))
    return {"law": "inst", "prob": _inst_probability(value, where)}


def _inst_probability(raw: Any, where: str) -> float:
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        raise TranslationError(f"{where}: invalid inst prob {raw!r}") from None
    if not 0.0 <= prob <= 1.0:
        raise TranslationError(f"{where}: inst prob must be within [0, 1], got {prob}")
    return prob


def _legacy_inst_return(value: Any, where: str) -> dict | None:
    """One `repair_param` entry of the legacy on-demand dialect → the
    per-order return law dict. The dialect writes an exponential rate, and
    a rate of 0 is cod3s' inactive marker (`is_occ_law_repair_active`
    false: `occ` stays absorbing), so it reads as no return edge at all."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise TranslationError(f"{where}: invalid repair rate {value!r}") from None
    if rate <= 0.0:
        return None
    return {"law": "exp", "rate": rate}


def _translate_objfminst(fm: dict) -> dict:
    """Legacy on-demand dialect (`cls: ObjFMInst`): the same 3-state
    Bernoulli expansion as the native inst cell, under the historical
    failure/repair vocabulary and scalar per-order gammas."""
    for key in ("failure_effects_trans", "repair_effects_trans", "occ_effects_trans", "not_occ_effects_trans"):
        if fm.get(key):
            raise TranslationError(
                f"failure mode `{fm.get('fm_name', '<unnamed>')}`: `{key}` on an on-demand "
                "(inst) mode: a one-shot effect on a branching draw edge is refused, "
                "as cod3s refuses it (`_validate_trans_effects`)"
            )
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"
    behaviour = fm.get("behaviour", "internal")
    expander = _on_demand_expander(True, behaviour)

    # Same activity convention as the native wire: `None` is the explicit
    # inactive-order marker, any other value passes through (a number or a
    # `{distrib: inst, prob: …}` dict).
    failure = list(_require(fm, "failure_param", where=where))
    repair = list(_require(fm, "repair_param", where=where))
    if expander == "ObjFM":
        # The unified expander reads per-order LAW DICTS, where this
        # dialect writes bare numbers: a gamma on the occurrence face, an
        # exponential rate on the return one. Spelling them out is what
        # keeps the two dialects one mode, and it is a translation, not a
        # second reading of the law.
        failure = [_legacy_inst_gamma(value, where) for value in failure]
        repair = [_legacy_inst_return(value, where) for value in repair]
    spec: dict[str, Any] = {
        "type": expander,
        "name": _require(fm, "fm_name", where=where),
        "targets": list(_require(fm, "targets", where=where)),
        "behaviour": behaviour,
        "failure": failure,
        "repair": repair,
        "failure_effects": dict(fm.get("failure_effects") or {}),
    }
    if fm.get("repair_effects"):
        spec["repair_effects"] = dict(fm["repair_effects"])
    for cond in ("failure_cond", "repair_cond"):
        if cond in fm:
            spec[cond] = fm[cond]
    for state in ("failure_state", "repair_state"):
        if state in fm:
            spec[state] = fm[state]
    if "occ_parked_state" in fm:
        spec["absorb_state"] = fm["occ_parked_state"]
    return spec


# --- an event ----------------------------------------------------------------

#: The six comparisons an event's ``cond_operator`` may name, and the two
#: truth functions its logics may. Spelled alike by cod3s, by muscadet's
#: ``MODE_OPERATORS`` / ``MODE_LOGIC`` and by the plugin's own ``_OPE`` /
#: ``_LOGIC``, so nothing is translated here -- but a spelling outside them
#: reaches the plugin as a bare `KeyError` naming a dict nobody declared,
#: which is precisely what this module exists not to do.
_EVENT_OPERATORS = ("==", "!=", "<", "<=", ">", ">=")
_EVENT_LOGIC = ("all", "any")

#: The keys an event carries beside its name and its condition, each spelled
#: the way ``cod3s.ObjEvent.__init__`` takes it and read by
#: :func:`pyraichu.plugins.muscadet._expand_objevent` under the same name.
_EVENT_KEYS = (
    "inner_logic",
    "outer_logic",
    "cond_operator",
    "cond_value",
    "tempo_occ",
    "tempo_not_occ",
    "event_aut_name",
    "occ_state_name",
    "not_occ_state_name",
)


def event_object(ev: dict) -> dict:
    """The muscadet-plugin object a cod3s EVENT declaration means.

    The sibling of :func:`failure_mode_object`, and here for the same reason:
    a muscadet system declaration carries that vocabulary for a standalone
    event -- it is ``cod3s.ObjEvent``'s own constructor keywords either way --
    and :mod:`pyraichu.declare` reads it through here rather than writing a
    second, lesser expansion.

    Named apart from :func:`failure_mode_object` rather than folded into it:
    an event is not a failure mode. It has no target, no effect and no
    common-cause order, so the two share no key beyond the condition, and one
    function reading both would refuse a mode's fault in an event's
    vocabulary.

    What the caller owes is the declaration as cod3s spells it, with its
    condition leaves carrying the object they watch. What it gets back is an
    ``ObjEvent`` object, named after the event.

    Raises
    ------
    TranslationError
        For a missing name or condition, or for a comparison or a truth
        function spelled outside the ones the three layers share.
    """
    where = f"event `{ev.get('name', '<unnamed>')}`"
    spec = {
        "type": "ObjEvent",
        "name": _require(ev, "name", where=where),
        "cond": _require(ev, "cond", where=where),
    }
    for key in _EVENT_KEYS:
        if key in ev:
            spec[key] = ev[key]

    operator = spec.get("cond_operator", "==")
    if operator not in _EVENT_OPERATORS:
        raise TranslationError(
            f"{where}: cond_operator {operator!r} is not one of "
            f"{list(_EVENT_OPERATORS)}"
        )
    for key in ("inner_logic", "outer_logic"):
        if key in spec and spec[key] not in _EVENT_LOGIC:
            raise TranslationError(
                f"{where}: {key} {spec[key]!r} is not one of "
                f"{list(_EVENT_LOGIC)}; a declaration carries the NAME of a "
                f"truth function, never the function"
            )
    for key in ("tempo_occ", "tempo_not_occ"):
        if key in spec:
            spec[key] = _as_float(spec[key], where=where, what=key)
    return spec


def _as_float(raw: Any, *, where: str, what: str) -> float:
    """One numeric field of an event's wire, typed (a declaration is
    serialised, so it carries whatever the writer put there)."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise TranslationError(f"{where}: {what} must be a number, got {raw!r}")
    try:
        return float(raw)
    except ValueError:
        raise TranslationError(f"{where}: {what} must be a number, got {raw!r}") from None
