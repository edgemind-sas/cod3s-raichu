"""COD3S-platform translator (F3b): model export + study → RAICHU model.

The platform persists a study in two **disjoint** artefacts:

- the **model export** (JSON): topology only — a `MUSCADET`-type KB of
  component templates (interfaces with `input_logic` / `prod_cond` DNF) and
  UUID-keyed component instances + connections, with per-instance attribute
  overrides;
- the **study** (YAML → dict): the dynamics — `failure_modes` (ObjFM),
  `events` (ObjEvent feared events), `targets`, `indicators` and the
  Monte-Carlo `simulation` parameters.

:func:`translate` fuses both into one RAICHU plugin-spec model (ObjFlow /
ObjFM / ObjEvent objects expanded by :mod:`pyraichu.plugins.muscadet`) plus
the run configuration. The topology semantics mirror muscadet's
`importers/cod3s_platform.py` (the reference importer) but the translator is
self-contained — no muscadet or PyCATSHOO dependency.

Scope: what real platform safety studies use — `classic` flows, `input_logic`
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
# are runtime variables, not configuration — ignored.
_LEGACY_ROLES = {
    "logic": "logic_in",
    "init": "prod_init",
    "state": "fed_in",
    "availability": "is_available",
}
_OBSERVABLE_ROLES = {"is_available", "fed_out", "fed_in", "is_active"}
_SUPPORTED_OVERRIDES = {"logic_in", "prod_init"}
_KNOWN_OVERRIDES = _SUPPORTED_OVERRIDES | {"var_in_default", "active_init", "fed_available_init"}

_SUPPORTED_EXPORT_MAJORS = {1, 3}


@dataclass
class Translation:
    """Result of :func:`translate`: the RAICHU model + the run config."""

    #: Plugin-spec model dict, ready for ``pyraichu.load_model``.
    model: dict[str, Any]
    #: Monte-Carlo run configuration from the study's ``simulation`` section
    #: (``nb_runs``, ``samples``, ``seed``, ``time_unit``…) — empty without a study.
    simulation: dict[str, Any] = field(default_factory=dict)
    #: Indicator measures requested by the study, per indicator name
    #: (e.g. ``{"doors_unsecured_occ": ["nb-occurrences", "sojourn-time"]}``).
    measures: dict[str, list[str]] = field(default_factory=dict)


# --- model export (topology) -------------------------------------------------


def _check_version(payload: dict) -> None:
    """Reject exports of an unsupported major. An absent version is
    tolerated (platform DB dumps carry none) — the per-construct fail-fast
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
    top-level `kb` (platform DB dumps) or `model.kb` — first with
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
            f"KB `{template_name}`, interface `{name}`: legacy 'logic' field — "
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
    return {
        "direction": "output",
        "name": name,
        "prod_cond": [list(group) for group in interface.get("prod_cond") or []],
    }


def _require(mapping: dict, key: str, *, where: str) -> Any:
    """Fetch a required key with a typed, contextual error (never a raw
    KeyError — the fail-fast contract of this module)."""
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
    observables ignored, unknown roles rejected, unsupported-but-known ones
    rejected only when they carry a value."""
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
        if role not in _SUPPORTED_OVERRIDES:
            raise TranslationError(
                f"component `{comp.get('name')}`: override role {role!r} on "
                f"`{name}` not supported by the translator"
            )
        out[(name, role)] = value
    return out


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
        flows_in, flows_out = [], []
        for flow in flows_by_class[cls]:
            fname = flow["name"]
            if flow["direction"] == "input":
                logic = overrides.get((fname, "logic_in"), flow["logic"])
                flows_in.append(
                    {"name": fname, "logic": _coerce_logic(logic, where=f"{cname}.{fname}")}
                )
            else:
                entry: dict[str, Any] = {"name": fname, "var_prod_cond": flow["prod_cond"]}
                init = overrides.get((fname, "prod_init"))
                if init is not None:
                    entry["var_prod_default"] = _coerce_bool(
                        init, where=f"component `{cname}`, flow `{fname}` prod_init"
                    )
                flows_out.append(entry)
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


def _translate_failure_mode(fm: dict) -> dict:
    cls = fm.get("cls", "ObjFMExp")
    if cls not in _FM_LAWS:
        raise TranslationError(f"failure mode `{fm.get('fm_name')}`: cls {cls!r} not supported")
    law, key = _FM_LAWS[cls]
    where = f"failure mode `{fm.get('fm_name', '<unnamed>')}`"

    def order_law(p):
        # cod3s marks an INACTIVE common-cause order with a zero rate
        # (`is_occ_law_*_active` = param > 0, `drop_inactive_automata`) —
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
    literal = pattern.removeprefix("^").removesuffix("$")
    if any(ch in literal for ch in ".*+?[](){}|\\"):
        raise TranslationError(f"{where}: regex pattern {pattern!r} not supported (exact names only)")
    return literal


def translate_study(study: dict) -> tuple[list[dict], list[dict], dict, dict[str, list[str]]]:
    """Study dict → ``(plugin objects, indicators, simulation, measures)``."""
    target_names = {
        t["name"] for t in study.get("targets") or [] if t.get("enabled", True)
    }
    events = [e for e in study.get("events") or [] if e.get("enabled", True)]
    event_auts = {e["name"]: e.get("event_aut_name", "ev") for e in events}

    objects = [
        _translate_failure_mode(fm)
        for fm in study.get("failure_modes") or []
        if fm.get("enabled", True)
    ]
    objects += [_translate_event(ev, target_names) for ev in events]

    indicators: list[dict] = []
    measures: dict[str, list[str]] = {}
    for ind in study.get("indicators") or []:
        if not ind.get("enabled", True):
            continue
        where = f"indicator on {ind.get('component')!r}"
        if ind.get("attr_type") != "ST":
            raise TranslationError(f"{where}: attr_type {ind.get('attr_type')!r} not supported (ST only)")
        component = _strip_anchors(_require(ind, "component", where=where), where=where)
        state = _strip_anchors(_require(ind, "attr_name", where=where), where=where)
        if component not in event_auts:
            raise TranslationError(
                f"{where}: component `{component}` is not a declared event "
                "(only ObjEvent state indicators are supported)"
            )
        name = f"{component}_{state}"
        if name not in measures:
            indicators.append(
                {
                    "name": name,
                    "target": "state",
                    "component": component,
                    "automaton": event_auts[component],
                    "state": state,
                }
            )
            measures[name] = []
        measure = ind.get("measure")
        if measure and measure not in measures[name]:
            measures[name].append(measure)

    simulation = dict(study.get("simulation") or {})
    if "schedule" in simulation:
        simulation["samples"] = [
            float(entry["instant"]) if isinstance(entry, dict) else float(entry)
            for entry in simulation.pop("schedule")
        ]
    return objects, indicators, simulation, measures


# --- fusion ------------------------------------------------------------------


def translate(export: dict, study: dict | None = None) -> Translation:
    """Fuse the model export (topology) and the study (dynamics) into one
    RAICHU plugin-spec model + the Monte-Carlo run configuration."""
    model = translate_export(export)
    if study is None:
        return Translation(model=model)
    objects, indicators, simulation, measures = translate_study(study)
    model["plugins"]["muscadet"]["objects"].extend(objects)
    model["indicators"].extend(indicators)
    if study.get("name"):
        model["name"] = study["name"]
    return Translation(model=model, simulation=simulation, measures=measures)
