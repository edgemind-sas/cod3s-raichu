"""COD3S-platform translator (F3b): model export + study → RAICHU model.

The platform persists a study in two **disjoint** artefacts:

- the **model export** (JSON): topology only, a `MUSCADET`-type KB of
  component templates (interfaces with `input_logic` / `prod_cond` DNF) and
  UUID-keyed component instances + connections, with per-instance attribute
  overrides;
- the **study** (YAML → dict): the dynamics, `failure_modes` (ObjFM),
  `events` (ObjEvent feared events), `targets`, `indicators` and the
  Monte-Carlo `simulation` parameters.

:func:`translate` fuses both into one RAICHU plugin-spec model (ObjFlow /
ObjFM / ObjEvent objects expanded by :mod:`pyraichu.plugins.muscadet`) plus
the run configuration. The topology semantics mirror muscadet's
`importers/cod3s_platform.py` (the reference importer) but the translator is
self-contained: no muscadet or PyCATSHOO dependency.

Scope: what real platform safety studies use, `classic` flows, `input_logic`
or/and/k, `prod_cond` DNF (outer-OR / inner-AND), instance overrides
`logic`/`logic_in` and `init`/`prod_init`, ObjFMExp/ObjFMDelay, ObjEvent.
Anything outside raises a typed :class:`TranslationError` (fail fast, never
a silently-wrong model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TranslationError", "Translation", "translate", "translate_export", "translate_study"]


class TranslationError(ValueError):
    """A platform artefact uses a construct the translator does not cover."""


# Instance-attribute roles: current platform vocabulary and its legacy
# (pre-2026-05) spelling, normalised to the current one. Observable roles
# are runtime variables, not configuration: ignored.
_LEGACY_ROLES = {
    "logic": "logic_in",
    "init": "prod_init",
    "state": "fed_in",
    "availability": "is_available",
}
_OBSERVABLE_ROLES = {"is_available", "fed_out", "fed_in", "is_active"}
#: The roles applied to a flow, each with the direction it applies to and
#: the flow-entry key it becomes (the muscadet `FlowIn` / `FlowOut`
#: parameter it is passed to). `logic_in` is the input-side exception: it
#: coerces as a logic (`and` / `or` / k), not as a boolean. The tempo,
#: capacity and controller roles belong to the continuous bridges (wave 2)
#: and stay refused by name.
_LOGIC_ROLE = "logic_in"
_BOOL_ROLES = {
    "var_in_default": ("input", "var_in_default"),
    "prod_init": ("output", "var_prod_default"),
    # Service-function dormancy: `active_init` is muscadet's
    # `var_is_active_default` (kept for the var_is_active path, not a
    # platform UI role), `fed_available_init` the user-facing one, the
    # availability gate's initial value.
    "active_init": ("output", "var_is_active_default"),
    "fed_available_init": ("output", "var_fed_available_out_init"),
}
_KNOWN_OVERRIDES = {_LOGIC_ROLE, *_BOOL_ROLES}

_SUPPORTED_EXPORT_MAJORS = {1, 3}


@dataclass
class Translation:
    """Result of :func:`translate`: the RAICHU model + the run config."""

    #: Plugin-spec model dict, ready for ``pyraichu.load_model``.
    model: dict[str, Any]
    #: Monte-Carlo run configuration from the study's ``simulation`` section
    #: (``nb_runs``, ``samples``, ``seed``, ``time_unit``…), empty without a study.
    simulation: dict[str, Any] = field(default_factory=dict)
    #: Indicator measures requested by the study, per indicator name
    #: (e.g. ``{"doors_unsecured_occ": ["nb-occurrences", "sojourn-time"]}``).
    measures: dict[str, list[str]] = field(default_factory=dict)


# --- model export (topology) -------------------------------------------------


def _check_version(payload: dict) -> None:
    """Reject exports of an unsupported major. An absent version is
    tolerated (platform DB dumps carry none): the per-construct fail-fast
    checks still guard the actual content."""
    version = payload.get("export_version")
    if version is None:
        return
    major = str(version).split(".", 1)[0]
    if not major.isdigit() or int(major) not in _SUPPORTED_EXPORT_MAJORS:
        raise TranslationError(
            f"unsupported export_version {version!r} "
            f"(supported majors: {sorted(_SUPPORTED_EXPORT_MAJORS)})"
        )


def _resolve_kb(payload: dict) -> dict:
    """The embedded KB lives under `kb_embedded` (versioned exports), a
    top-level `kb` (platform DB dumps) or `model.kb`: first with
    component_templates wins."""
    for kb in (
        payload.get("kb_embedded"),
        payload.get("kb"),
        payload.get("model", {}).get("kb"),
    ):
        if isinstance(kb, dict) and kb.get("component_templates"):
            return kb["component_templates"]
    raise TranslationError("export carries no KB component_templates")


def _parse_interface(template_name: str, interface: dict) -> dict:
    """One KB interface → a flow dict `{direction, name, …}` (muscadet
    `_parse_interface` semantics, restricted to the supported scope)."""
    name = interface.get("name")
    if not name:
        raise TranslationError(f"KB `{template_name}`: interface missing 'name'")
    if "logic" in interface:
        raise TranslationError(
            f"KB `{template_name}`, interface `{name}`: legacy 'logic' field, "
            "re-export from a post-3.0.0 platform (input_logic / prod_cond)"
        )
    direction = (interface.get("port_type") or {}).get("general")
    if direction == "input":
        return {"direction": "input", "name": name, "logic": interface.get("input_logic", "or")}
    if direction != "output":
        raise TranslationError(
            f"KB `{template_name}`, interface `{name}`: unsupported "
            f"port_type.general={direction!r}"
        )
    flow_type = interface.get("flow_type") or "classic"
    if flow_type != "classic":
        raise TranslationError(
            f"KB `{template_name}`, interface `{name}`: flow_type "
            f"{flow_type!r} not supported by the translator (classic only)"
        )
    if interface.get("negate"):
        raise TranslationError(
            f"KB `{template_name}`, interface `{name}`: negate=true not supported"
        )
    inner = interface.get("logic_inner_mode", "and")
    if inner != "and":
        raise TranslationError(
            f"KB `{template_name}`, interface `{name}`: logic_inner_mode "
            f"{inner!r} not supported (outer-OR/inner-AND only)"
        )
    prod_cond = interface.get("prod_cond") or []
    for group in prod_cond:
        # DNF only: a flat `["power_in"]` would otherwise be exploded into
        # one group per character and fail far downstream, on a flow name
        # that does not exist.
        if not isinstance(group, list):
            raise TranslationError(
                f"KB `{template_name}`, interface `{name}`: prod_cond must be a "
                f"list of AND groups (OR of lists), got {group!r} at top level"
            )
    return {
        "direction": "output",
        "name": name,
        "prod_cond": [list(group) for group in prod_cond],
    }


def _require(mapping: dict, key: str, *, where: str) -> Any:
    """Fetch a required key with a typed, contextual error (never a raw
    KeyError: the fail-fast contract of this module)."""
    try:
        return mapping[key]
    except KeyError:
        raise TranslationError(f"{where}: missing required key {key!r}") from None


def _coerce_bool(raw: Any, *, where: str) -> bool:
    """Strict boolean coercion: the platform persists attribute values as
    strings, so Python truthiness would turn \"false\" into True (the exact
    pitfall the reference importer's `_parse_init_value` guards against)."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        return raw.strip().lower() == "true"
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    raise TranslationError(f"{where}: invalid boolean override {raw!r}")


def _coerce_logic(raw: Any, *, where: str) -> str | int:
    """'and' | 'or' | int k (the platform persists k as a decimal string)."""
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped in ("and", "or"):
            return stripped
        if stripped.isdigit() and int(stripped) >= 1:
            return int(stripped)
    elif isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
        return raw
    raise TranslationError(f"{where}: invalid logic override {raw!r} (expected 'and', 'or' or k >= 1)")


def _instance_overrides(comp: dict) -> dict[tuple[str, str], Any]:
    """Index a component instance's attributes by (flow, normalised role):
    observables ignored, unknown roles rejected. Applying a role is the
    caller's job, and so is the check that the flow's direction is the one
    the role applies to."""
    out: dict[tuple[str, str], Any] = {}
    for attr in comp.get("attributes") or []:
        name, value = attr.get("name"), attr.get("value")
        role = attr.get("role")
        if not name or role is None or value is None:
            continue  # role-less / valueless entries: KB defaults apply
        role = _LEGACY_ROLES.get(role, role)
        if role in _OBSERVABLE_ROLES:
            continue
        if role not in _KNOWN_OVERRIDES:
            raise TranslationError(
                f"component `{comp.get('name')}`: unknown attribute role {role!r} on `{name}`"
            )
        out[(name, role)] = value
    return out


def _bool_role(
    role: str,
    overrides: dict[tuple[str, str], Any],
    claimed: set[tuple[str, str]],
    *,
    cname: str,
    fname: str,
    direction: str,
) -> dict[str, Any]:
    """One boolean override role → its `{key: value}` flow-entry pair,
    empty when the instance carries none. A role applied to the other
    direction is a snapshot corruption, refused rather than coerced into
    something plausible."""
    value = overrides.get((fname, role))
    if value is None:
        return {}
    expected, key = _BOOL_ROLES[role]
    if direction != expected:
        raise TranslationError(
            f"component `{cname}`, flow `{fname}`: override role {role!r} "
            f"expects a {expected} flow but `{fname}` is {direction}"
        )
    claimed.add((fname, role))
    return {key: _coerce_bool(value, where=f"component `{cname}`, flow `{fname}` {role}")}


def _logic_role(
    overrides: dict[tuple[str, str], Any],
    flow: dict,
    claimed: set[tuple[str, str]],
    *,
    cname: str,
) -> Any:
    """The `logic_in` override of one input flow, or the KB's own."""
    fname = flow["name"]
    logic = overrides.get((fname, _LOGIC_ROLE), flow["logic"])
    if (fname, _LOGIC_ROLE) in overrides:
        claimed.add((fname, _LOGIC_ROLE))
    return _coerce_logic(logic, where=f"{cname}.{fname}")


def _refuse_unclaimed_overrides(
    overrides: dict[tuple[str, str], Any],
    claimed: set[tuple[str, str]],
    flows: list[dict],
    *,
    cname: str,
) -> None:
    """Every override an instance carries must be claimed by a flow of the
    class: one left over names a flow that does not exist, or a role
    applied to the other direction. Either way the value would be dropped
    in silence and the instance would run at its KB defaults with nothing
    in the model saying so."""
    directions = {flow["name"]: flow["direction"] for flow in flows}
    for fname, role in sorted(set(overrides) - claimed):
        expected = _BOOL_ROLES[role][0] if role in _BOOL_ROLES else "input"
        if fname not in directions:
            raise TranslationError(
                f"component `{cname}`: override role {role!r} names the flow "
                f"`{fname}`, which its class does not declare"
            )
        raise TranslationError(
            f"component `{cname}`, flow `{fname}`: override role {role!r} "
            f"expects a {expected} flow but `{fname}` is {directions[fname]}"
        )


def translate_export(payload: dict) -> dict[str, Any]:
    """Model export → RAICHU plugin-spec model (topology only): one
    ObjFlow object per component instance, connections resolved from UUIDs
    to `{component, port}` (out-flow port `{itf}_out` → in-flow `{itf}_in`)."""
    _check_version(payload)
    templates = _resolve_kb(payload)
    flows_by_class: dict[str, list[dict]] = {
        cls: [_parse_interface(cls, itf) for itf in (tpl.get("interfaces") or {}).values()]
        for cls, tpl in templates.items()
    }

    elements = payload.get("model", {}).get("elements", {})
    objects: list[dict] = []
    names_by_uuid: dict[str, str] = {}
    for uuid, comp in (elements.get("components") or {}).items():
        cname, cls = comp.get("name"), comp.get("class_name")
        if cls not in flows_by_class:
            raise TranslationError(f"component `{cname}`: unknown KB class `{cls}`")
        names_by_uuid[uuid] = cname
        overrides = _instance_overrides(comp)
        claimed: set[tuple[str, str]] = set()
        flows_in, flows_out = [], []
        for flow in flows_by_class[cls]:
            fname = flow["name"]
            if flow["direction"] == "input":
                flows_in.append(
                    {
                        "name": fname,
                        "logic": _logic_role(overrides, flow, claimed, cname=cname),
                        **_bool_role(
                            "var_in_default",
                            overrides,
                            claimed,
                            cname=cname,
                            fname=fname,
                            direction="input",
                        ),
                    }
                )
            else:
                entry: dict[str, Any] = {"name": fname, "var_prod_cond": flow["prod_cond"]}
                for role in ("prod_init", "active_init", "fed_available_init"):
                    entry.update(
                        _bool_role(
                            role, overrides, claimed, cname=cname, fname=fname, direction="output"
                        )
                    )
                flows_out.append(entry)
        _refuse_unclaimed_overrides(overrides, claimed, flows_by_class[cls], cname=cname)
        objects.append(
            {"type": "ObjFlow", "name": cname, "flows_in": flows_in, "flows_out": flows_out}
        )

    connections = []
    for uuid, conn in (elements.get("connections") or {}).items():
        try:
            src = names_by_uuid[conn["component_source"]]
            dst = names_by_uuid[conn["component_target"]]
        except KeyError as missing:
            raise TranslationError(
                f"connection `{uuid}` references unknown component {missing}"
            ) from None
        where = f"connection `{uuid}`"
        itf_src = _require(conn, "interface_source", where=where)
        itf_dst = _require(conn, "interface_target", where=where)
        connections.append(
            {
                "from": {"component": src, "port": f"{itf_src}_out"},
                "to": {"component": dst, "port": f"{itf_dst}_in"},
            }
        )

    return {
        "name": payload.get("model", {}).get("name", "cod3s_platform_model"),
        "plugins": {"muscadet": {"objects": objects}},
        "components": [],
        "connections": connections,
        "indicators": [],
    }


# --- study (dynamics) --------------------------------------------------------

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


def _one_shot_effects_refused(fm: dict, where: str) -> None:
    """One-shot (trans-based) effects have no RAICHU construct yet: the
    engine's effects are held (re-evaluated to a fixpoint), never pulsed at
    a transition. Losing one silently would build a model whose pulse is
    gone, so they are refused, naming the mode and the attribute."""
    for key in ("occ_effects_trans", "not_occ_effects_trans"):
        bucket = fm.get(key)
        if bucket:
            raise TranslationError(
                f"{where}: one-shot (trans-based) effects are not supported: "
                f"`{key}` carries {sorted(bucket)} and RAICHU has no "
                "transition-scoped effect yet (its effects are held while "
                "the state lasts). Drop the one-shot effects from the mode "
                "or keep the study on the platform engine"
            )


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


def _alias(fm: dict, native: str, legacy: str, where: str) -> Any:
    """Read one field under its native and its legacy spelling (cod3s
    refuses a double set the same way): both present must agree, otherwise
    the native one wins and the legacy one is the fallback."""
    if native in fm and legacy in fm and fm[native] != fm[legacy]:
        raise TranslationError(
            f"{where}: both {native!r} and its legacy alias {legacy!r} are "
            "set with different values: set exactly one"
        )
    return fm[native] if native in fm else fm.get(legacy)


def _translate_failure_mode(fm: dict) -> dict:
    cls = fm.get("cls", "ObjFMExp")
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"
    _one_shot_effects_refused(fm, where)
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
    production translator emits has a reader here; what it emits and what
    the engine cannot express yet (`*_effects_trans`) is refused."""
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"
    targets = list(_require(fm, "targets", where=where))
    n = len(targets)
    occ_kind, failure = _direction_laws(_require(fm, "occ_law", where=where), n, where)
    rep_kind, repair = _direction_laws(_require(fm, "not_occ_law", where=where), n, where)
    _check_param_names(fm, occ_kind, rep_kind, where)

    behaviour = fm.get("behaviour", "internal")
    on_demand = occ_kind == "inst"
    if on_demand and behaviour != "internal":
        # The on-demand expander builds the internal behaviour only: an
        # external on-demand mode would be silently built internal.
        raise TranslationError(
            f"{where}: an on-demand (inst) occurrence is only expanded with "
            f"the `internal` behaviour, got {behaviour!r}"
        )

    spec: dict[str, Any] = {
        "type": "ObjFMInst" if on_demand else "ObjFM",
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

    # Conditions: `failure_cond` is the wire alias of `occ_cond`, and the
    # repair face is `not_occ_cond` (`repair_cond` being its legacy
    # spelling, which the pre-native dialects carry).
    occ_cond = _alias(fm, "occ_cond", "failure_cond", where)
    if occ_cond is not None:
        spec["failure_cond"] = occ_cond
    not_occ_cond = _alias(fm, "not_occ_cond", "repair_cond", where)
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


def _translate_objfminst(fm: dict) -> dict:
    """Legacy on-demand dialect (`cls: ObjFMInst`): the same 3-state
    Bernoulli expansion as the native inst cell, under the historical
    failure/repair vocabulary and scalar per-order gammas."""
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"
    behaviour = fm.get("behaviour", "internal")
    if behaviour != "internal":
        raise TranslationError(
            f"{where}: an on-demand (inst) failure mode is only expanded with "
            f"the `internal` behaviour, got {behaviour!r}"
        )

    # Same activity convention as the native wire: `None` is the explicit
    # inactive-order marker, any other value passes through (a number or a
    # `{distrib: inst, prob: …}` dict).
    spec: dict[str, Any] = {
        "type": "ObjFMInst",
        "name": _require(fm, "fm_name", where=where),
        "targets": list(_require(fm, "targets", where=where)),
        "behaviour": behaviour,
        "failure": list(_require(fm, "failure_param", where=where)),
        "repair": list(_require(fm, "repair_param", where=where)),
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


def _translate_event(ev: dict, target_names: set[str]) -> dict:
    where = f"event `{ev.get('name', '<unnamed>')}`"
    spec = {
        "type": "ObjEvent",
        "name": _require(ev, "name", where="study events"),
        "cond": _require(ev, "cond", where=where),
    }
    for key in (
        "inner_logic",
        "outer_logic",
        "cond_operator",
        "cond_value",
        "tempo_occ",
        "tempo_not_occ",
        "event_aut_name",
        "occ_state_name",
        "not_occ_state_name",
    ):
        if key in ev:
            spec[key] = ev[key]
    if ev["name"] in target_names:
        spec["target"] = True
    return spec


def _strip_anchors(pattern: str, *, where: str) -> str:
    """The study's indicator selectors are regexes; the translator only
    supports exact-name patterns (`^name$` or a bare literal)."""
    if not isinstance(pattern, str):
        raise TranslationError(f"{where}: selector must be a string, got {pattern!r}")
    literal = pattern.removeprefix("^").removesuffix("$")
    if any(ch in literal for ch in ".*+?[](){}|\\"):
        raise TranslationError(f"{where}: regex pattern {pattern!r} not supported (exact names only)")
    return literal


def _as_float(raw: Any, *, where: str, what: str) -> float:
    """One numeric schedule field, typed (the platform persists YAML
    numbers, but a hand-edited study carries whatever was typed)."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise TranslationError(f"{where}: {what} must be a number, got {raw!r}")
    try:
        return float(raw)
    except ValueError:
        raise TranslationError(f"{where}: {what} must be a number, got {raw!r}") from None


def _as_count(raw: Any, *, where: str, what: str) -> int:
    """One integer schedule field (`nvalues`)."""
    value = _as_float(raw, where=where, what=what)
    if not value.is_integer():
        raise TranslationError(f"{where}: {what} must be an integer, got {raw!r}")
    return int(value)


def _range_instants(entry: dict, where: str) -> list[float]:
    """One `{start, end, nvalues}` range → its instants, cod3s
    `InstantLinearRange.get_instants_list`: `nvalues <= 1` yields `[end]`,
    otherwise `nvalues` evenly spaced points from `start` to `end`, both
    bounds included. The last point is written as `end` exactly (as
    numpy's `linspace` does) rather than left to accumulated rounding."""
    start = _as_float(entry["start"], where=where, what="start")
    end = _as_float(entry["end"], where=where, what="end")
    nvalues = _as_count(entry["nvalues"], where=where, what="nvalues")
    if nvalues < 1:
        raise TranslationError(f"{where}: nvalues must be >= 1, got {entry['nvalues']!r}")
    if end < start:
        raise TranslationError(f"{where}: end ({end}) must be >= start ({start})")
    if nvalues == 1:
        return [end]
    step = (end - start) / (nvalues - 1)
    points = [start + i * step for i in range(nvalues)]
    points[-1] = end
    return points


def _schedule_instants(entries: Any) -> list[float]:
    """The study's observation schedule → the flat, time-ordered list of
    instants the engine samples at.

    Three shapes reach this module: a bare number, the single-instant
    `{"instant": t}` and the range `{"start", "end", "nvalues"}`. The
    range is a first-class platform feature (`ScheduleEntry`, `nvalues`
    up to 1e5) and the engine takes a flat list, so the expansion happens
    here, once, with cod3s' own semantics. Across entries the instants
    are sorted and deduplicated: a range overlapping a hand-written
    instant must not sample twice."""
    instants: list[float] = []
    for index, entry in enumerate(entries or []):
        where = f"study simulation.schedule[{index}]"
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            instants.append(float(entry))
        elif isinstance(entry, dict):
            if entry.get("instant") is not None:
                clash = sorted(set(entry) & {"start", "end", "nvalues"})
                if clash:
                    raise TranslationError(
                        f"{where}: 'instant' is exclusive with {clash} "
                        "(a single instant or a range, never both)"
                    )
                if set(entry) != {"instant"}:
                    raise TranslationError(
                        f"{where}: unknown key(s) {sorted(set(entry) - {'instant'})} "
                        "next to 'instant'"
                    )
                instants.append(_as_float(entry["instant"], where=where, what="instant"))
            else:
                missing = sorted(k for k in ("start", "end", "nvalues") if k not in entry)
                if missing:
                    raise TranslationError(
                        f"{where}: a schedule entry is either {{'instant': t}} or "
                        f"{{'start', 'end', 'nvalues'}}: missing {missing}"
                    )
                extra = sorted(set(entry) - {"start", "end", "nvalues"})
                if extra:
                    raise TranslationError(f"{where}: unknown key(s) {extra}")
                instants += _range_instants(entry, where)
        else:
            raise TranslationError(
                f"{where}: unsupported schedule entry {entry!r} (a number, "
                "{'instant': t} or {'start', 'end', 'nvalues'})"
            )
    return sorted(set(instants))


#: The indicator vocabulary the raichu engine serves, i.e. the contract the
#: platform runner documents (`cod3s-run-study-raichu`, `_MEASURE_SERIES` /
#: `_STAT_SERIES`): `nb-occurrences`, `sojourn-time` and `value`, over mean
#: and stddev. `had_value` and the min/max/quantile statistics cod3s computes
#: have no raichu counterpart, so a study asking for one is refused here
#: rather than answered with a hole in `indicators.csv`. (The runner's own
#: `[warn]`-skip covers a study edited after its translation; a translation
#: that can already see the mismatch has no excuse to.)
_MEASURES = ("nb-occurrences", "sojourn-time", "value")
_STATS = ("mean", "stddev")

#: Comparison operators of an indicator predicate, wire spelling → the core
#: expression tree's `cmp` operator (the same mapping the muscadet plugin's
#: `_leaf` applies to condition clauses).
_PREDICATE_OPS = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def _predicate(ind: dict, where: str) -> dict | None:
    """The optional `operator`/`value_test` predicate of a direct variable
    measurement. Both keys come together and the tested value must be one
    the engine carries: a string state is not (see `_const_value`)."""
    operator, value_test = ind.get("operator"), ind.get("value_test")
    if (operator is None) != (value_test is None):
        raise TranslationError(
            f"{where}: 'operator' and 'value_test' come together "
            "(a predicate is both a comparison and the value it tests)"
        )
    if operator is None:
        return None
    if operator not in _PREDICATE_OPS:
        raise TranslationError(
            f"{where}: operator {operator!r} not supported (expected one of {sorted(_PREDICATE_OPS)})"
        )
    if isinstance(value_test, str) or not isinstance(value_test, (bool, int, float)):
        raise TranslationError(
            f"{where}: value_test {value_test!r} is not a value the engine carries "
            "(a boolean, an integer or a float: raichu has no string state yet)"
        )
    return {"operator": operator, "value": value_test}


def _const_value(value: Any) -> dict:
    """One tested value → a core constant expression node (bool, int or
    float; the kinds the engine's `Value` carries)."""
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    else:
        kind, value = "float", float(value)
    return {"op": "const", "value": {"kind": kind, "value": value}}


def _measure_component(ref: dict, predicate: dict) -> dict:
    """A VAR indicator's predicate → the one-attribute component holding it.

    The engine records a raw attribute (`IndicatorTarget::Attribute`), so a
    predicate (`{comp}.{attr} <op> value`) has no direct construct: this
    mirror recomputes the comparison whenever the observed attribute changes
    and the indicator reads it. Observation only: no automaton and no
    monitored transition, so it leaves no trace in the events nor in the
    minimal sequences."""
    name = f"measure_{ref['component']}_{ref['attribute']}"
    return {
        "name": name,
        "attributes": [
            {"name": "value", "kind": "bool", "init": {"kind": "bool", "value": False}}
        ],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [
            {
                "name": "update_value",
                "effects": [
                    {
                        "target": {"component": name, "attribute": "value"},
                        "value": {
                            "op": "cmp",
                            "cmp": _PREDICATE_OPS[predicate["operator"]],
                            "lhs": {"op": "attr", "attr": dict(ref)},
                            "rhs": _const_value(predicate["value"]),
                        },
                    }
                ],
            }
        ],
        "equations": [],
    }


def _mode_state_registry(objects: list[dict]) -> dict[str, dict]:
    """Every mode/target pair → what raichu names it, keyed by the carrier
    name cod3s writes in a `mode_ref` clause.

    cod3s builds an ObjFM as a component named `{target}__{fm_name}` whose
    failure state is `occ`, and an inter-mode condition reads exactly that
    (`study_translator_service.py:_resolve_cond_objs` rewrites a `mode_ref`
    clause into `{"attr": "occ", "obj": "{instance}__{fm}"}`). raichu names
    the component after the mode alone and gives every active common-cause
    order its own automaton, so the reference has to be resolved to the
    order-1 automaton of the *named instance*: `fm__cc_i` and its
    suffixed state. An `external` mode is the mirror case: the automaton
    is grafted into the target itself, unsuffixed."""
    registry: dict[str, dict] = {}
    for spec in objects:
        if spec.get("type") not in ("ObjFM", "ObjFMInst"):
            continue
        targets = spec.get("targets") or []
        entry = {
            "component": spec["name"],
            "behaviour": spec.get("behaviour", "internal"),
            "failure_state": spec.get("failure_state", "occ"),
            "repair_state": spec.get("repair_state", "rep"),
        }
        for index, target in enumerate(targets):
            registry[f"{target}__{spec['name']}"] = {
                **entry, "target": target, "index": index, "count": len(targets)
            }
    return registry


def _mode_ref_leaf(leaf: dict, registry: dict[str, dict], *, where: str) -> dict:
    """One condition leaf: rewritten when it reads a mode's state, checked
    for a value the engine can carry in every case, and passed through
    otherwise."""
    value = leaf.get("value")
    if isinstance(value, str):
        raise TranslationError(
            f"{where}: the clause tests {value!r}, a string, and raichu has no "
            "string state to compare it with (its values are boolean, integer "
            "and float; the string discrete state is a documented extension "
            "that needs an engine change). Test a boolean or a number, or keep "
            "the study on the platform engine"
        )
    obj, attr = leaf.get("obj"), leaf.get("attr")
    if not isinstance(obj, str) or not isinstance(attr, str) or "__" not in obj:
        return leaf
    hit = registry.get(obj)
    if hit is None:
        raise TranslationError(
            f"{where}: the clause reads state {attr!r} of `{obj}`, and no "
            "failure mode of this study is declared on such a carrier: a "
            "dangling inter-mode reference, which a variable read would only "
            "report far downstream as an unknown attribute"
        )
    if attr == hit["failure_state"]:
        state = hit["failure_state"]
    elif attr == hit["repair_state"]:
        state = hit["repair_state"]
    else:
        raise TranslationError(
            f"{where}: the clause reads `{obj}`.{attr}, and the mode "
            f"`{hit['component']}` has no such state (its states are "
            f"`{hit['failure_state']}` and `{hit['repair_state']}`)"
        )
    ope = leaf.get("ope", "==")
    if ope not in ("==", "!="):
        raise TranslationError(
            f"{where}: a mode reference only compares with '==' or '!=', got {ope!r}"
        )
    if not isinstance(value, bool):
        raise TranslationError(
            f"{where}: a mode reference tests a boolean state, got {value!r}"
        )
    if hit["behaviour"] in ("external", "external_rep_indep"):
        # An external mode grafts its mirror into the target: the state
        # lives on the target component, under the mode's own automaton,
        # with no common-cause suffix (the mirror is unsuffixed).
        return {
            "obj": hit["target"], "automaton": hit["component"], "state": state,
            "ope": ope, "value": value,
        }
    suffix = "" if hit["count"] == 1 else f"__cc_{hit['index'] + 1}"
    return {
        "obj": hit["component"], "automaton": f"fm{suffix}", "state": f"{state}{suffix}",
        "ope": ope, "value": value,
    }


def _normalise_clauses(cond: Any, registry: dict[str, dict], *, where: str) -> Any:
    """Walk one condition specification and rewrite every leaf: the shapes
    are the cod3s ones (`bool`, one leaf, a list of leaves, a list of
    lists), which the muscadet plugin's `_cond_groups` accepts as is."""
    if isinstance(cond, dict):
        return _mode_ref_leaf(cond, registry, where=where)
    if not isinstance(cond, list):
        return cond
    return [
        _mode_ref_leaf(clause, registry, where=where)
        if isinstance(clause, dict)
        else [
            _mode_ref_leaf(leaf, registry, where=where) if isinstance(leaf, dict) else leaf
            for leaf in clause
        ] if isinstance(clause, list) else clause
        for clause in cond
    ]


def translate_study(study: dict) -> tuple[list[dict], list[dict], dict, dict[str, list[str]]]:
    """Study dict → ``(plugin objects, indicators, simulation, measures)``.

    An indicator entry carrying a `predicate` (a `VAR` target with an
    `operator`/`value_test`) is not loadable as is: :func:`translate`
    materialises the mirror component it reads instead."""
    target_names = {
        _require(t, "name", where="study targets")
        for t in study.get("targets") or []
        if t.get("enabled", True)
    }
    events = [e for e in study.get("events") or [] if e.get("enabled", True)]
    event_auts = {
        _require(e, "name", where="study events"): e.get("event_aut_name", "ev")
        for e in events
    }

    objects = [
        _translate_failure_mode(fm)
        for fm in study.get("failure_modes") or []
        if fm.get("enabled", True)
    ]
    objects += [_translate_event(ev, target_names) for ev in events]

    # Clause dialect: a mode reference spelled `{"attr": "occ", "obj":
    # "{instance}__{fm}"}` is a *state* read and compiles to a state test
    # on the automaton the referenced mode builds; a clause value the
    # engine cannot carry is refused. One pass over every condition of the
    # study, after all the modes are known (a condition may reference any
    # of them), and before anything expands.
    registry = _mode_state_registry(objects)
    for spec in objects:
        where = f"condition of `{spec['name']}`"
        if spec.get("type") == "ObjEvent":
            spec["cond"] = _normalise_clauses(spec["cond"], registry, where=where)
            continue
        for key in ("failure_cond", "repair_cond"):
            if key in spec:
                spec[key] = _normalise_clauses(spec[key], registry, where=where)

    indicators: list[dict] = []
    measures: dict[str, list[str]] = {}
    # The target indexed per engine indicator name, so a second study
    # indicator reaching the same name is checked against the first one:
    # the engine keys its indicators by name, so two different targets
    # under one name would silently answer for each other.
    seen: dict[str, dict] = {}
    for ind in study.get("indicators") or []:
        if not ind.get("enabled", True):
            continue
        where = f"indicator on {ind.get('component')!r}"
        attr_type = ind.get("attr_type")
        if attr_type not in ("ST", "VAR"):
            raise TranslationError(
                f"{where}: attr_type {attr_type!r} not supported (ST or VAR)"
            )
        component = _strip_anchors(_require(ind, "component", where=where), where=where)
        raw_attr = _alias(ind, "attr_name", "var", where)
        if raw_attr is None:
            raise TranslationError(f"{where}: missing required key 'attr_name'")
        attribute = _strip_anchors(raw_attr, where=where)
        # An absent measure is the trajectory itself: the sampled value at
        # each schedule instant, which is exactly the `value` measure.
        measure = ind.get("measure") or "value"
        if measure not in _MEASURES:
            raise TranslationError(
                f"{where}: measure {ind.get('measure')!r} has no raichu counterpart "
                f"(supported: {list(_MEASURES)}); an absent measure reads as 'value'"
            )
        stats = ind.get("stats") or ["mean"]
        unknown = [s for s in stats if s not in _STATS]
        if unknown:
            raise TranslationError(
                f"{where}: stats {unknown} have no raichu counterpart "
                f"(supported: {list(_STATS)})"
            )

        predicate: dict | None = None
        if attr_type == "ST":
            if component not in event_auts:
                raise TranslationError(
                    f"{where}: component `{component}` is not a declared event "
                    "(state indicators read an ObjEvent's occ / not_occ)"
                )
            name = f"{component}_{attribute}"
            entry = {
                "name": name,
                "target": "state",
                "component": component,
                "automaton": event_auts[component],
                "state": attribute,
            }
        else:
            predicate = _predicate(ind, where)
            name = f"{component}_{attribute}"
            entry = {
                "name": name,
                "target": "attribute",
                "attr": {"component": component, "attribute": attribute},
            }
            if predicate is not None:
                entry["predicate"] = predicate
        previous = seen.get(name)
        if previous is not None and (
            {k: v for k, v in previous.items() if k != "name"}
            != {k: v for k, v in entry.items() if k != "name"}
        ):
            raise TranslationError(
                f"{where}: indicator name {name!r} is already taken by a "
                "different target; the engine keys its indicators by name, so "
                "the two would answer for each other (rename one of them)"
            )
        seen[name] = entry
        if name not in measures:
            indicators.append(entry)
            measures[name] = []
        if measure not in measures[name]:
            measures[name].append(measure)

    simulation = dict(study.get("simulation") or {})
    if "schedule" in simulation:
        simulation["samples"] = _schedule_instants(simulation.pop("schedule"))
    return objects, indicators, simulation, measures


# --- fusion ------------------------------------------------------------------


def _materialise_predicates(model: dict, indicators: list[dict]) -> None:
    """Point every predicate-carrying indicator at its mirror component,
    added to the model. Done here rather than in :func:`translate_study`
    because the mirror is a *component*: only the fusion site owns the
    model the component goes into."""
    for indicator in indicators:
        predicate = indicator.pop("predicate", None)
        if predicate is None:
            continue
        ref = indicator["attr"]
        mirror = _measure_component(ref, predicate)
        model["components"].append(mirror)
        indicator["attr"] = {"component": mirror["name"], "attribute": "value"}


def translate(export: dict, study: dict | None = None) -> Translation:
    """Fuse the model export (topology) and the study (dynamics) into one
    RAICHU plugin-spec model + the Monte-Carlo run configuration."""
    model = translate_export(export)
    if study is None:
        return Translation(model=model)
    objects, indicators, simulation, measures = translate_study(study)
    model["plugins"]["muscadet"]["objects"].extend(objects)
    _materialise_predicates(model, indicators)
    model["indicators"].extend(indicators)
    if study.get("name"):
        model["name"] = study["name"]
    return Translation(model=model, simulation=simulation, measures=measures)
