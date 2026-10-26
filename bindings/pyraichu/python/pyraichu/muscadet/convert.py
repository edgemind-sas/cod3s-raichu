"""muscadet declaration -> component type library documents.

A muscadet system declaration (the document :mod:`pyraichu.muscadet.declare`
reads, which the COD3S platform exports) describes every component one by
one. This module rewrites it as the three documents of a component type
library: a library holding one type per knowledge-base class, a system
instantiating and connecting those types, and a study activating the modes
and declaring the feared events.

Every reading rule is the declaration reader's own: a flow component goes
through :func:`declare._plan`, a two-state component through
:func:`declare.mode_object`, a gate through :func:`declare.logic_gate_object`,
and a condition through the plugin's :func:`plugin._cond_tree`. A second
reader written here would drift from the first.

What has no counterpart in the library format is refused by name, per
component, and never dropped: :class:`Conversion.refusals` lists each
refusal with the construct that blocks it. A declaration with one refusal
yields no documents, since a library that silently lacks a component is a
different model.

>>> from pyraichu.muscadet.convert import convert
>>> conversion = convert(declaration)
>>> if conversion.converted:
...     model = conversion.load()
"""

from __future__ import annotations

import copy
import functools
import json
from dataclasses import dataclass, field
from typing import Any

from . import declare, plugin

#: Format version of the three documents written here.
DOCUMENT_FORMAT = 1

#: Classes too generic to name a type: their components become singleton
#: types named after the instance.
GENERIC_CLASSES = frozenset({"ObjFlow", "ObjCtrl", "ObjLogicGate"})

#: Knowledge-base classes whose behaviour is known to be carried entirely by
#: the declaration (KTD11 of the plan). A class enters this list only through
#: a model that is green on the muscadet corpus bench against the reference
#: engine; each entry names that model. All of them were measured at muscadet
#: 5.6.0, bench verdict recorded on 2026-09-21. A component of any other
#: Python class may hold behaviour the declaration does not say, so both
#: RAICHU routes would agree on a model the class does not describe.
#:
#: Keyed by class name, which is all a declaration records: a same-named
#: class of another module is not the audited one, and nothing here can tell
#: them apart.
AUDITED_CLASSES: dict[str, str] = {
    "ObjFlow": "the bare component: everything it does is declared",
    "ObjLogicGate": "the portable gate: everything it does is declared",
    "AirConditioning": "examples/datacenter/datacenter_02.py",
    "AlimElec": "examples/isimu/cyber_3comp.py",
    "BackupPump": "examples/isimu/power_plant.py",
    "Block": "examples/rbd_01/rbd_01.py",
    "CoolingSource": "examples/isimu/datacenter_lite.py",
    "Generator": "examples/datacenter/datacenter_02.py",
    "Grid": "examples/isimu/power_plant.py",
    "HMI": "examples/isimu/power_plant.py",
    "MainPump": "examples/isimu/power_plant.py",
    "Plant": "examples/isimu/power_plant.py",
    "PowerSource": "examples/isimu/datacenter_lite.py",
    "ProcessIndustriel": "examples/isimu/cyber_3comp.py",
    "Pump": "examples/datacenter/datacenter_02.py",
    "Server": "examples/isimu/datacenter_lite.py",
    "Serveur": "examples/isimu/cyber_3comp.py",
    "Source": "examples/rbd_01/rbd_01.py",
    "SourceTrigger": "examples/isimu/trigger_source.py",
    "Target": "examples/rbd_01/rbd_01.py",
    "TargetK2": "examples/isimu/rbd_kn.py",
}

#: Keys of a failure-mode object this converter reads.
_MODE_KEYS = frozenset(
    {
        "type",
        "name",
        "mode_name",
        "targets",
        "behaviour",
        "failure",
        "repair",
        "failure_effects",
        "failure_cond",
        "repair_cond",
        "failure_state",
        "repair_state",
        "cond_inner_logic",
        "cond_outer_logic",
    }
)

#: Keys of an event object this converter reads.
_EVENT_KEYS = frozenset(
    {
        "type",
        "name",
        "cond",
        "inner_logic",
        "outer_logic",
        "cond_operator",
        "cond_value",
        "tempo_occ",
        "tempo_not_occ",
    }
)

#: The automaton and states every feared event carries, on both routes.
_EVENT_AUTOMATON = "ev"
_EVENT_STATES = ("occ", "not_occ")

#: The two states of a converted mode automaton (`raichu-library`'s own
#: names), and the defaults the declaration route names its states with.
_OCC, _REP = "occ", "rep"


@dataclass(frozen=True)
class Refusal:
    """One construct the library format cannot express."""

    component: str
    construct: str
    detail: str

    def __str__(self) -> str:
        return f"{self.component}: {self.construct} ({self.detail})"


@dataclass
class Conversion:
    """The documents a declaration converts to, or why it does not.

    ``library``, ``system`` and ``study`` are ``None`` when ``refusals`` is
    not empty. ``rename`` maps each transition of the declaration route that
    the library route carries under another name (``"Srv__mdc_a.fm.failure"``
    -> ``"Srv.mdc_a.occ"``); feared events keep their names and map to
    themselves. ``unverified`` names the components whose class may carry
    behaviour outside the declaration (see :data:`AUDITED_CLASSES`).
    """

    library: dict | None
    system: dict | None
    study: dict | None
    refusals: list[Refusal] = field(default_factory=list)
    rename: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    @property
    def converted(self) -> bool:
        return not self.refusals

    def load(self):
        """The expanded library model, ready to simulate."""
        from ..library import load_system

        if not self.converted:
            raise ValueError(
                "the declaration did not convert: "
                + "; ".join(str(refusal) for refusal in self.refusals)
            )
        return load_system(self.system, [self.library], self.study)


class _Refused(Exception):
    """A component the converter cannot express; carries the refusal."""

    def __init__(self, construct: str, detail: str):
        super().__init__(f"{construct}: {detail}")
        self.construct = construct
        self.detail = detail


# ---------------------------------------------------------------------------
# Flow components and gates: what each instance uses
# ---------------------------------------------------------------------------


def _flow_in_use(name: str, keywords: dict) -> dict:
    logic = keywords.pop("logic", "or")
    default = keywords.pop("var_in_default", False)
    if isinstance(logic, bool) or not (logic in ("and", "or") or isinstance(logic, int)):
        raise _Refused("flow input logic", f"input `{name}` declares logic {logic!r}")
    with_: dict[str, Any] = {"name": name}
    if isinstance(logic, int):
        with_["logic"] = "k"
        with_["k"] = logic
    else:
        with_["logic"] = logic
    with_["default"] = bool(default)
    return {"fragment": "std.bool_flow_in", "with": with_}


def _operand_attribute(output: str, operand: Any, inputs: set[str], outputs: set[str]) -> str:
    """The attribute a production operand reads, resolved as the authoring
    layer resolves it: an input of that name first, else an output."""
    port = None
    if isinstance(operand, dict):
        if set(operand) - {"name", "port"}:
            raise _Refused(
                "structured production operand",
                f"output `{output}` reads {operand!r}; the std output reads flow states only",
            )
        port, operand = operand.get("port"), operand["name"]
    if port != "out" and operand in inputs:
        return f"{operand}_fed_in"
    if operand in outputs:
        return f"{operand}_fed_out"
    raise _Refused("production operand", f"output `{output}` reads `{operand}`, no discrete flow")


def _flow_out_use(name: str, keywords: dict, inputs: set[str], outputs: set[str]) -> dict:
    if keywords.pop("negate", False):
        raise _Refused("negated output (`negate`)", f"output `{name}`")
    if not keywords.pop("var_fed_available_out_reset", True):
        raise _Refused(
            "availability not reset (`var_fed_available_out_reset` false)",
            f"output `{name}`",
        )
    available = bool(keywords.pop("var_fed_available_out_init", True))
    produced = bool(keywords.pop("var_prod_default", False))
    keywords.pop("var_prod_cond_inner_mode", None)
    condition = keywords.pop("var_prod_cond", None)
    if not condition:
        return {
            "fragment": "std.bool_source_out",
            "with": {"name": name, "prod_init": produced, "fed_available_init": available},
        }
    groups = []
    for group in condition:
        groups.append([_operand_attribute(name, o, inputs, outputs) for o in group])
    return {
        "fragment": "std.bool_flow_out",
        "with": {"name": name, "prod_cond": groups, "fed_available_init": available},
    }


def _flow_uses(spec: dict, name: str) -> tuple[list[dict], list[tuple[str, dict]]]:
    """The std fragments a flow component's declaration means, in order, and
    the failure modes it declares in its own `failure_modes` section."""
    try:
        _, _, _, calls = declare._plan(spec, declare.COMPONENT_CLASSES)
    except declare.ComponentSpecError as error:
        raise _Refused("declaration refused by the reader", str(error)) from error
    inputs = {c.keywords.get("name", c.entry) for c in calls if c.method == "add_flow_in"}
    outputs = {c.keywords.get("name", c.entry) for c in calls if c.method == "add_flow_out"}
    uses, embedded = [], []
    for call in calls:
        if call.section == "failure_modes" and call.method in _EMBEDDED_LAWS:
            embedded.append((call.method, dict(call.keywords)))
            continue
        if call.section != "flows":
            raise _Refused(f"section `{call.section}`", f"entry {call.entry!r}")
        keywords = dict(call.keywords)
        flow = keywords.pop("name", call.entry)
        if call.method == "add_flow_in":
            uses.append(_flow_in_use(flow, keywords))
        elif call.method == "add_flow_out":
            uses.append(_flow_out_use(flow, keywords, inputs, outputs))
        else:
            entry = next(
                (e for e in spec.get("flows") or [] if e.get("name") == flow), {}
            )
            raise _Refused(
                f"flow class `{entry.get('cls', call.method)}`", f"flow `{flow}`"
            )
        if keywords:
            raise _Refused(
                "flow keyword", f"flow `{flow}` declares {sorted(keywords)}"
            )
    return uses, embedded


#: The authoring calls of a `failure_modes` entry and the law each builds.
_EMBEDDED_LAWS = {
    "add_exp_failure_mode": ("exp", "rate", "failure_rate", "repair_rate"),
    "add_delay_failure_mode": ("delay", "time", "failure_time", "repair_time"),
}


@dataclass(frozen=True)
class _EmbeddedMode:
    """A mode of a component's own `failure_modes` section, converted."""

    component: str
    mode: str
    entry: dict
    activation: dict
    rename: dict[str, str]


def _embedded_mode(component: str, method: str, keywords: dict, uses: list[dict]) -> _EmbeddedMode:
    """A mode of a component's own `failure_modes` section: its catalogue
    entry, its activation and its transition renames.

    The authoring layer builds it as an `ok`/`nok` automaton named after the
    mode, its failure guarded by one local variable, and cuts the
    availability of every target output while `nok`."""
    law, field_name, failure_key, repair_key = _EMBEDDED_LAWS[method]
    mode = keywords.pop("name")
    targets = keywords.pop("targets", None) or []
    cond = keywords.pop("failure_cond", None)
    failure = keywords.pop(failure_key)
    repair = keywords.pop(repair_key)
    for key in ("failure_effects", "repair_effects"):
        if keywords.pop(key, None):
            raise _Refused(f"failure mode `{key}`", f"mode `{mode}` derates a continuous output")
    if not targets:
        # An empty target list means every output to the authoring layer,
        # and the declaration route refuses it at system scale.
        raise _Refused("failure mode gating no output", f"mode `{mode}` names no target output")
    if keywords:
        raise _Refused("failure mode keyword", f"mode `{mode}` declares {sorted(keywords)}")
    outputs = {
        use["with"]["name"]: use["with"].get("fed_available_init", True)
        for use in uses
        if use["fragment"] in ("std.bool_flow_out", "std.bool_source_out")
    }
    for target in targets:
        if target not in outputs:
            raise _Refused("failure mode target", f"mode `{mode}` cuts `{target}`, no discrete output")
        if outputs[target] is not True:
            raise _Refused(
                "failure mode on an output initially unavailable",
                f"mode `{mode}` cuts `{target}`",
            )
    entry: dict[str, Any] = {"name": mode, "nature": "failure"}
    if targets:
        entry["effects"] = [
            {"attribute": f"{target}_fed_available_out", "value": False} for target in sorted(targets)
        ]
    activation: dict[str, Any] = {
        "mode": mode,
        "instances": [component],
        "occurrence": {"distrib": law, field_name: float(failure)},
    }
    # An exponential repair at rate zero is no repair (the authoring layer
    # leaves the edge out).
    if not (law == "exp" and float(repair) == 0.0):
        activation["repair"] = {"distrib": law, field_name: float(repair)}
    if cond is not None:
        activation["condition"] = {"op": "attr", "attr": {"component": component, "attribute": cond}}
    rename = {
        f"{component}.{mode}.failure": f"{component}.{mode}.{_OCC}",
        f"{component}.{mode}.repair": f"{component}.{mode}.{_REP}",
    }
    return _EmbeddedMode(component, mode, entry, activation, rename)


def _gate_use(
    spec: dict, name: str, components: dict
) -> tuple[list[dict], list[dict]]:
    """A gate's fragment and the connections that feed it.

    A gate leaf reading ``<obj>.<flow>_fed_out == true`` reads exactly what
    the out port ``<obj>.<flow>_out`` exports, so it becomes a connection
    to the gate's input. Any other leaf reads an attribute by name, which a
    type cannot do.
    """
    gate = declare.logic_gate_object(spec, name, components)
    outputs = gate["out_elements"]
    if len(outputs) != 1:
        raise _Refused("gate with several out elements", f"out elements {outputs}")
    out = outputs[0]
    sources = []
    for group in gate["cond"]:
        if len(group) != 1:
            raise _Refused("gate group of several leaves", f"group {group!r}")
        (leaf,) = group
        source = components.get(leaf.get("obj"))
        attr = leaf.get("attr", "")
        flow = attr[: -len("_fed_out")] if attr.endswith("_fed_out") else None
        if (
            "attr" not in leaf
            or leaf.get("value") is not True
            or leaf.get("ope", "==") != "=="
            or source is None
            or declare.component_kind(source, leaf["obj"]) != declare.COMPONENT_KIND_FLOW
            or flow not in declare._discrete_out_flows(source)
        ):
            raise _Refused(
                "gate leaf read by name",
                f"leaf {leaf!r} is not an equality to true on a discrete output",
            )
        sources.append({"from": f"{leaf['obj']}.{flow}_out", "to": f"{name}.{out}_in"})
    with_: dict[str, Any] = {"name": out, "logic": gate["kind"]}
    if gate["kind"] == "k":
        with_["k"] = gate["k"]
    return [{"fragment": "std.logic_gate", "with": with_}], sources


#: Which parameters of a std fragment hold a VALUE (a type parameter when it
#: differs between instances); every other parameter is structure.
_VALUE_PARAMETERS = {
    "std.bool_flow_in": ("default", "k"),
    "std.bool_flow_out": ("fed_available_init",),
    "std.bool_source_out": ("prod_init", "fed_available_init"),
    "std.logic_gate": ("k",),
}


def _targetable(uses: list[dict]) -> set[str]:
    """The attributes a mode may write on an instance using `uses`."""
    attributes = set()
    for use in uses:
        name = use["with"]["name"]
        if use["fragment"] in ("std.bool_flow_out", "std.bool_source_out"):
            attributes.add(f"{name}_fed_available_out")
        if use["fragment"] == "std.bool_source_out":
            attributes.add(f"{name}_prod_available")
    return attributes


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModeRef:
    """Where the library route keeps a declared mode."""

    target: str
    mode: str
    failure_state: str
    repair_state: str


def _rewrite_leaves(cond: Any, modes: dict[str, _ModeRef], gates: dict[str, str]) -> Any:
    """Point every leaf of a condition at what the library route holds.

    A state leaf on a declared mode (its own ``fm`` automaton, or the mirror
    an external mode grafts on its target) reads the mode automaton of the
    target instance; a leaf on a gate's ``result`` reads the gate input.
    """
    if isinstance(cond, bool):
        return cond
    if isinstance(cond, dict):
        cond = [[cond]]
    elif cond and all(isinstance(leaf, dict) for leaf in cond):
        cond = [cond]
    rewritten = []
    for group in cond:
        out = []
        for leaf in group:
            leaf = dict(leaf)
            obj = leaf.get("obj")
            if "automaton" in leaf:
                ref = (
                    modes.get(f"{obj}.{leaf['automaton']}")
                    or modes.get(obj)
                    or modes.get(leaf["automaton"])
                )
                if ref is not None:
                    # The mode's own automaton names its states exactly; the
                    # mirror an external mode grafts on its target prefixes
                    # them with `<mode>__`.
                    state = leaf["state"].rsplit("__", 1)[-1]
                    if state == ref.failure_state:
                        state = _OCC
                    elif state == ref.repair_state:
                        state = _REP
                    else:
                        raise _Refused(
                            "condition on a mode state", f"state `{leaf['state']}` of `{obj}`"
                        )
                    leaf.update(obj=ref.target, automaton=ref.mode, state=state)
                elif leaf.get("automaton") != _EVENT_AUTOMATON:
                    raise _Refused(
                        "condition on an automaton", f"`{obj}.{leaf['automaton']}`"
                    )
            elif obj in gates:
                if leaf.get("attr") != "result":
                    raise _Refused("condition on a gate attribute", f"`{obj}.{leaf.get('attr')}`")
                leaf["attr"] = f"{gates[obj]}_fed_in"
            out.append(leaf)
        rewritten.append(out)
    return rewritten


def _condition(
    cond: Any,
    *,
    carrier: str | None,
    modes: dict[str, _ModeRef],
    gates: dict[str, str],
    inner: str = "all",
    outer: str = "any",
    operator: str = "==",
    value: bool = True,
) -> dict:
    """A declared condition as a core expression, through the plugin's own
    reading (`carrier` is what a leaf naming no `obj` reads)."""
    return plugin._cond_tree(
        _rewrite_leaves(cond, modes, gates),
        inner,
        outer,
        operator,
        value,
        carrier=carrier,
        where="a converted condition",
    )


# ---------------------------------------------------------------------------
# Modes and events
# ---------------------------------------------------------------------------


def _law(spec: Any, what: str) -> dict:
    kind = spec.get("law") or spec.get("cls") or spec.get("distrib")
    if kind == "inst":
        raise _Refused("on-demand law (`inst`)", what)
    try:
        return plugin._law(spec)
    except ValueError as error:
        raise _Refused("mode law", f"{what}: {error}") from error


def _check_keys(obj: dict, known: frozenset, what: str) -> None:
    extra = sorted(
        key for key, value in obj.items() if key not in known and value not in (None, {}, [], False)
    )
    if extra:
        raise _Refused(f"{what} key `{extra[0]}`", f"declares {extra}")


def _order_laws(laws: list, targets: list[str], key: str) -> Any:
    """The order-1 law of a mode, refusing an active common-cause order."""
    for order, law in enumerate(laws[1:], start=2):
        if law is not None:
            raise _Refused(
                f"common-cause order {order}",
                f"`{key}` declares a law for {order} targets out of {len(targets)}",
            )
    return laws[0] if laws else None


def _mode_states(obj: dict) -> tuple[str, str]:
    """The failure and repair state names of a declared mode."""
    return obj.get("failure_state") or _OCC, obj.get("repair_state") or _REP


def _mode_transitions(name: str, obj: dict, targets: list[str]) -> list[tuple[str, str, str]]:
    """(declaration transition, target, which) of a mode, as the plugin
    names them: `failure`/`repair` alone, `occ__cc_i` per target otherwise."""
    failure_state, repair_state = _mode_states(obj)
    if len(targets) == 1:
        return [
            (f"{name}.fm.failure", targets[0], _OCC),
            (f"{name}.fm.repair", targets[0], _REP),
        ]
    names = []
    for index, target in enumerate(targets, start=1):
        suffix = f"__cc_{index}"
        names.append((f"{name}.fm{suffix}.{failure_state}{suffix}", target, _OCC))
        names.append((f"{name}.fm{suffix}.{repair_state}{suffix}", target, _REP))
    return names


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _type_key(spec: dict, name: str) -> str:
    """The class a component's type is named after (KTD9)."""
    metadata = spec.get("metadata") or {}
    class_name = metadata.get("class_name") if isinstance(metadata, dict) else None
    if isinstance(class_name, str) and class_name:
        return class_name
    source = _source_class(spec)
    if source not in GENERIC_CLASSES:
        return source
    return name


def _source_class(spec: dict) -> str:
    """The knowledge-base class a component was read from."""
    source = spec.get(declare.SOURCE_CLS_KEY) or spec.get("cls")
    return source if isinstance(source, str) and source else "ObjFlow"


def _clean_name(name: str) -> str:
    return name.replace(".", "_").replace("$", "_")


def _masked(uses: list[dict]) -> str:
    """The structure of a use list: every value parameter masked."""
    masked = []
    for use in uses:
        values = _VALUE_PARAMETERS.get(use["fragment"], ())
        masked.append(
            {
                "fragment": use["fragment"],
                "with": {k: ("<value>" if k in values else v) for k, v in use["with"].items()},
            }
        )
    return json.dumps(masked, sort_keys=True)


@dataclass
class _Variant:
    structure: str
    uses: list[dict]
    modes: dict[str, dict] = field(default_factory=dict)
    members: list[str] = field(default_factory=list)


def _build_types(
    instances: dict[str, dict], notes: list[str]
) -> tuple[list[dict], list[dict]]:
    """One type per class, split into variants where the structure differs;
    values differing between instances become parameters."""
    variants: dict[str, list[_Variant]] = {}
    for name, info in instances.items():
        structure = _masked(info["uses"])
        family = variants.setdefault(info["key"], [])
        home = None
        for variant in family:
            if variant.structure != structure:
                continue
            if any(
                variant.modes.get(mode, entry) != entry for mode, entry in info["modes"].items()
            ):
                continue
            home = variant
            break
        if home is None:
            home = _Variant(structure, info["uses"])
            family.append(home)
        home.modes.update(info["modes"])
        home.members.append(name)

    types, system_instances = [], {}
    taken: set[str] = set()
    for key, family in variants.items():
        if len(family) > 1:
            notes.append(
                f"class `{key}` splits into {len(family)} variant types "
                "(different flows or a mode the members declare differently)"
            )
        for index, variant in enumerate(family, start=1):
            base = _clean_name(key) if index == 1 else f"{_clean_name(key)}_v{index}"
            type_name, suffix = base, 2
            while type_name in taken:
                type_name, suffix = f"{base}_{suffix}", suffix + 1
            taken.add(type_name)
            uses, parameters, values = _parameterise(variant, instances)
            type_doc: dict[str, Any] = {"name": type_name, "uses": uses}
            if parameters:
                type_doc["parameters"] = parameters
            if variant.modes:
                type_doc["modes"] = [variant.modes[m] for m in sorted(variant.modes)]
            if key != type_name or variant.members != [key]:
                type_doc["metadata"] = {"class": key}
            types.append(type_doc)
            for member in variant.members:
                instance: dict[str, Any] = {"name": member, "type": type_name}
                if values.get(member):
                    instance["parameters"] = values[member]
                system_instances[member] = instance
    ordered = [system_instances[name] for name in instances]
    return types, ordered


def _parameterise(
    variant: _Variant, instances: dict[str, dict]
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    uses = copy.deepcopy(variant.uses)
    parameters: list[dict] = []
    values: dict[str, dict] = {member: {} for member in variant.members}
    taken: set[str] = set()
    for index, use in enumerate(uses):
        for key in _VALUE_PARAMETERS.get(use["fragment"], ()):
            if key not in use["with"]:
                continue
            seen = [instances[m]["uses"][index]["with"].get(key) for m in variant.members]
            if all(value == seen[0] for value in seen):
                continue
            parameter = f"{use['with']['name']}_{key}"
            while parameter in taken:
                parameter += "_"
            taken.add(parameter)
            first = seen[0]
            kind = "bool" if isinstance(first, bool) else (
                "integer" if isinstance(first, int) else "number"
            )
            parameters.append({"name": parameter, "kind": kind})
            use["with"][key] = {"$param": parameter}
            for member, value in zip(variant.members, seen):
                values[member][parameter] = value
    return uses, parameters, values


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------


def _components(declaration: dict) -> dict[str, dict]:
    components = declaration.get("components") or {}
    if isinstance(components, list):
        components = {c.get("name"): c for c in components}
    return components


def convert(
    declaration: dict,
    *,
    targets: list[str] | tuple[str, ...] = (),
    schedule: list[float] | None = None,
    nb_runs: int | None = None,
    seed: int = 0,
) -> Conversion:
    """Convert a muscadet system declaration into library documents.

    `targets` names the events a sequence campaign ends on (they become
    targets of their feared event); `schedule`, when given, writes the
    study's simulation settings (horizon = last instant).
    """
    components = _components(declaration)
    refusals: list[Refusal] = []
    notes: list[str] = []
    rename: dict[str, str] = {}

    def refuse(component: str, error: _Refused) -> None:
        refusals.append(Refusal(component, error.construct, error.detail))

    # --- flow components, gates and controllers -------------------------
    instances: dict[str, dict] = {}
    gates: dict[str, str] = {}
    connections: list[dict] = []
    unverified: list[str] = []
    embedded_modes: list[_EmbeddedMode] = []
    for name, spec in components.items():
        kind = declare.component_kind(spec, name)
        if kind == declare.COMPONENT_KIND_TWO_STATE_MODE:
            continue
        if _clean_name(name) != name:
            refuse(name, _Refused("component name", "a name may not contain `.` or `$`"))
            continue
        if _source_class(spec) not in AUDITED_CLASSES:
            unverified.append(name)
        try:
            if kind == declare.COMPONENT_KIND_CONTROLLER:
                raise _Refused("controller", "controllers have no library counterpart yet")
            if kind == declare.COMPONENT_KIND_LOGIC_GATE:
                uses, feeds = _gate_use(spec, name, components)
                gates[name] = uses[0]["with"]["name"]
                connections.extend(feeds)
            else:
                uses, embedded = _flow_uses(spec, name)
                for method, keywords in embedded:
                    embedded_modes.append(_embedded_mode(name, method, keywords, uses))
        except _Refused as error:
            refuse(name, error)
            continue
        instances[name] = {"key": _type_key(spec, name), "uses": uses, "modes": {}}
    if any(spec.get(declare.DERIVED_OUT_AUTOMATA_KEY) for spec in components.values()):
        notes.append(
            "`create_default_out_automata` is observation only (the ok/nok state "
            "pairs indicators may read); the library derives no such pair"
        )

    # --- modes: where each one lives ------------------------------------
    mode_objects: dict[str, dict] = {}
    mode_refs: dict[str, _ModeRef] = {}
    events: list[dict] = []
    for name, spec in components.items():
        if declare.component_kind(spec, name) != declare.COMPONENT_KIND_TWO_STATE_MODE:
            continue
        try:
            obj = declare.mode_object(spec, components, name)
        except declare.ComponentSpecError as error:
            refuse(name, _Refused("declaration refused by the reader", str(error)))
            continue
        if obj["type"] == "ObjEvent":
            events.append(obj)
            continue
        mode_objects[name] = obj

    # Modes of a component's own `failure_modes` section.
    activations: list[dict] = []
    used: dict[str, set[str]] = {}
    for own in embedded_modes:
        if own.component not in instances:
            continue
        instances[own.component]["modes"][own.mode] = own.entry
        used.setdefault(own.component, set()).add(own.mode)
        activations.append(own.activation)
        rename.update(own.rename)
        mode_refs[f"{own.component}.{own.mode}"] = _ModeRef(own.component, own.mode, "nok", "ok")

    # Mode names, unique per instance.
    mode_names: dict[str, str] = {}
    for name, obj in mode_objects.items():
        base = _clean_name(obj.get("mode_name") or name)
        mode_targets = obj.get("targets") or []
        chosen = base
        if any(chosen in used.get(t, set()) for t in mode_targets):
            chosen = _clean_name(name)
            notes.append(f"mode `{name}` is named `{chosen}`: `{base}` is taken on its target")
        for target in mode_targets:
            used.setdefault(target, set()).add(chosen)
        mode_names[name] = chosen
        if len(mode_targets) == 1:
            mode_refs[name] = _ModeRef(mode_targets[0], chosen, *_mode_states(obj))

    for name, obj in mode_objects.items():
        try:
            activations.extend(
                _convert_mode(
                    name, obj, mode_names[name], instances, mode_refs, gates, rename, notes
                )
            )
        except _Refused as error:
            refuse(name, error)

    # --- feared events --------------------------------------------------
    feared = []
    target_set = set(targets)
    for obj in events:
        name = obj["name"]
        try:
            _check_keys(obj, _EVENT_KEYS, "event")
            condition = _condition(
                obj["cond"],
                carrier=None,
                modes=mode_refs,
                gates=gates,
                inner=obj.get("inner_logic", "all"),
                outer=obj.get("outer_logic", "any"),
                operator=obj.get("cond_operator", "=="),
                value=obj.get("cond_value", True),
            )
        except (_Refused, ValueError, KeyError) as error:
            if not isinstance(error, _Refused):
                error = _Refused("event condition", str(error))
            refuse(name, error)
            continue
        event: dict[str, Any] = {"name": name, "condition": condition}
        for key in ("tempo_occ", "tempo_not_occ"):
            if obj.get(key):
                event[key] = float(obj[key])
        if name in target_set:
            event["target"] = True
        feared.append(event)
        for state in _EVENT_STATES:
            rename[f"{name}.{_EVENT_AUTOMATON}.{state}"] = f"{name}.{_EVENT_AUTOMATON}.{state}"
    missing_targets = sorted(target_set - {e["name"] for e in feared})
    if missing_targets:
        notes.append(f"targets naming no converted event: {missing_targets}")

    # --- connections ----------------------------------------------------
    for entry in declaration.get("connections") or []:
        connections.append(
            {
                "from": f"{entry['source']}.{entry['source_box']}",
                "to": f"{entry['target']}.{entry['target_box']}",
            }
        )

    if declaration.get("indicators"):
        notes.append(
            f"{len(declaration['indicators'])} platform indicators are not converted: "
            "the study declares none"
        )

    if refusals:
        return Conversion(None, None, None, refusals, {}, notes, unverified)

    # --- documents ------------------------------------------------------
    types, system_instances = _build_types(instances, notes)
    model_name = declaration.get("name") or "converted"
    system_name = _clean_name(model_name)
    library_name = f"{system_name}_types"
    library = {
        "raichu_library": {"format": DOCUMENT_FORMAT},
        "library": {
            "name": library_name,
            "version": "1.0.0",
            "imports": ["std"],
            "description": f"Types converted from the muscadet declaration `{model_name}`.",
            "types": types,
        },
    }
    system = {
        "raichu_system": {"format": DOCUMENT_FORMAT},
        "system": {
            "name": system_name,
            "library": {"name": library_name, "version": "1.0.0"},
            "instances": system_instances,
            "connections": connections,
        },
    }
    study_body: dict[str, Any] = {
        "name": f"{system_name}_study",
        "system": system_name,
        "modes": activations,
        "feared_events": feared,
    }
    if schedule:
        study_body["simulation"] = {
            "method": "monte_carlo",
            "nb_runs": int(nb_runs or 1),
            "horizon": float(max(schedule)),
            "instants": [float(t) for t in schedule],
            "seed": int(seed),
        }
    study = {"raichu_study": {"format": DOCUMENT_FORMAT}, "study": study_body}
    return Conversion(library, system, study, [], rename, notes, unverified)


def _convert_mode(
    name: str,
    obj: dict,
    mode: str,
    instances: dict[str, dict],
    mode_refs: dict[str, _ModeRef],
    gates: dict[str, str],
    rename: dict[str, str],
    notes: list[str],
) -> list[dict]:
    """A declared mode as a catalogue entry on each target and one study
    activation per target."""
    _check_keys(obj, _MODE_KEYS, "mode")
    behaviour = obj.get("behaviour")
    if behaviour not in (None, "internal", "external"):
        raise _Refused(f"mode behaviour `{behaviour}`", "only internal and external modes")
    targets = obj.get("targets") or []
    if not targets:
        raise _Refused("mode with no target", "a mode acts on at least one component")
    missing = [t for t in targets if t not in instances]
    if missing:
        raise _Refused("mode on a refused component", f"targets {missing}")

    failure = _order_laws(obj.get("failure") or [], targets, "failure")
    if failure is None:
        # No order carries a law (a zero rate reads as none): the declaration
        # route builds no automaton either, so the mode is not activated.
        notes.append(f"mode `{name}` declares no active occurrence law and is not activated")
        return []
    occurrence = _law(failure, "occurrence")
    repair_cond = obj.get("repair_cond", True)
    repair = None
    if repair_cond is False:
        repair = None
    elif repair_cond in (True, None):
        repair_law = _order_laws(obj.get("repair") or [], targets, "repair")
        repair = None if repair_law is None else _law(repair_law, "repair")
    else:
        raise _Refused("repair condition", "the study format has no repair condition")

    effects = []
    for attribute, value in sorted((obj.get("failure_effects") or {}).items()):
        if not isinstance(value, bool):
            raise _Refused("non-boolean mode effect", f"`{attribute}` = {value!r}")
        effects.append({"attribute": attribute, "value": value})
    for target in targets:
        allowed = _targetable(instances[target]["uses"])
        for effect in effects:
            if effect["attribute"] not in allowed:
                raise _Refused(
                    "effect on a non-targetable attribute",
                    f"`{target}.{effect['attribute']}`",
                )
    entry: dict[str, Any] = {"name": mode, "nature": "failure"}
    if effects:
        entry["effects"] = effects
    for target in targets:
        known = instances[target]["modes"].get(mode)
        if known is not None and known != entry:
            raise _Refused("mode name clash", f"`{mode}` on `{target}`")
        instances[target]["modes"][mode] = entry

    cond = obj.get("failure_cond", True)
    activations = []
    for target in targets:
        activation: dict[str, Any] = {
            "mode": mode,
            "instances": [target],
            "occurrence": occurrence,
        }
        if repair is not None:
            activation["repair"] = repair
        if cond is not True and cond is not None:
            try:
                activation["condition"] = _condition(
                    cond,
                    carrier=target,
                    modes=mode_refs,
                    gates=gates,
                    inner=obj.get("cond_inner_logic", "all"),
                    outer=obj.get("cond_outer_logic", "any"),
                )
            except (ValueError, KeyError) as error:
                raise _Refused("mode condition", str(error)) from error
        activations.append(activation)
    for declared, target, which in _mode_transitions(name, obj, targets):
        rename[declared] = f"{target}.{mode}.{which}"
    return activations


# ---------------------------------------------------------------------------
# Two-route comparison
# ---------------------------------------------------------------------------
#
# Both routes read the same declaration, so they must answer the same values.
# A shared seed cannot show it: the declaration route places each mode in its
# own component and the library route inside its instance, so the two draw
# their random numbers in different orders. Instead, one trajectory is drawn
# on the declaration route and REPLAYED on both in deferred mode, where no
# stochastic transition fires by itself: each recorded stochastic event is
# fired at its recorded date on both routes, and after every firing the armed
# transitions must agree on their law and bookkeeping. Comparing the law is
# what catches a wrongly converted rate, which a forced date alone would not.


@dataclass(frozen=True)
class RouteComparison:
    """The verdict of comparing the declaration route with the library route.

    ``level`` says what the verdict rests on: ``exact`` (no stochastic
    transition, one deterministic trajectory), ``replay`` (stochastic
    trajectories replayed in deferred mode, every converted random
    transition fired at least once), ``replay-partial`` (the replay found no
    divergence, but the transitions in ``unexercised`` never fired, so their
    effects were never observed) or ``statistical`` (attribute means over
    independent runs, a two-sample z-test per observation, which only shows
    that no difference was detected at that sample size).

    ``identical`` says no difference was found. Only ``exact`` and
    ``replay`` make it a proof of equality: a partial replay is reported as
    not identical, and a statistical pass only bounds the difference by the
    test's power.
    """

    identical: bool
    level: str
    detail: str
    trajectories: int
    unexercised: tuple[str, ...] = ()


class _Divergence(Exception):
    pass


def declaration_model(declaration: dict):
    """The declaration route's model of `declaration`."""
    from .. import load_model

    return load_model(json.dumps(declare.build_document(declaration)))


@functools.lru_cache(maxsize=8)
def _parsed_body(text: str) -> dict:
    document = json.loads(text)
    return document.get("model", document)


def _body(model) -> dict:
    return _parsed_body(model.json)


def _observed(library_model, declared_model) -> tuple[list[str], list[str]]:
    """Attributes and automata both routes carry under the same name: every
    attribute of the library model the declaration model also holds, and the
    feared events' automata."""
    declared = _body(declared_model)
    known = {
        f"{c['name']}.{a['name']}" for c in declared["components"] for a in c.get("attributes", [])
    }
    automata = {
        f"{c['name']}.{a['name']}" for c in declared["components"] for a in c.get("automata", [])
    }
    attributes, states = [], []
    for component in _body(library_model)["components"]:
        for attribute in component.get("attributes", []):
            qualified = f"{component['name']}.{attribute['name']}"
            if qualified in known:
                attributes.append(qualified)
        for automaton in component.get("automata", []):
            qualified = f"{component['name']}.{automaton['name']}"
            if automaton["name"] == _EVENT_AUTOMATON and qualified in automata:
                states.append(qualified)
    return attributes, states


def _snapshot(session, attributes: list[str], states: list[str]) -> dict[str, Any]:
    values = {name: session.attribute(name) for name in attributes}
    values.update({name: session.state(name) for name in states})
    return values


def _first_difference(left: dict, right: dict) -> str:
    for key, value in left.items():
        if value != right.get(key):
            return f"`{key}`: declaration {value!r}, library {right.get(key)!r}"
    return "values differ"


#: Laws that fire without a random date.
_DATED_LAWS = frozenset({"delay", "watched", "inst"})


def _stochastic_names(model) -> set[str]:
    """Qualified names of the transitions drawing a random date."""
    return {
        f"{component['name']}.{automaton['name']}.{transition['name']}"
        for component in _body(model)["components"]
        for automaton in component.get("automata", [])
        for transition in automaton.get("transitions", [])
        if transition.get("distrib", "delay") not in _DATED_LAWS
    }


def _replay_covers(model) -> bool:
    """Whether deferred mode can replay every random draw of `model`: an
    instantaneous transition with branch probabilities draws its branch at
    once, which deferred mode does not defer."""
    return not any(
        transition.get("distrib") == "inst" or len(transition.get("targets", [])) > 1
        for component in _body(model)["components"]
        for automaton in component.get("automata", [])
        for transition in automaton.get("transitions", [])
    )


def _armed(session, rename: dict[str, str] | None) -> dict[str, tuple]:
    armed = {}
    for entry in session.deferred():
        name = entry["transition"]
        if rename is not None:
            if name not in rename:
                raise _Divergence(f"armed transition `{name}` has no library counterpart")
            name = rename[name]
        armed[name] = (
            json.dumps(entry["law"], sort_keys=True),
            entry["armed_at"],
            entry["age"],
            entry["paused"],
            entry["hazard"],
        )
    return armed


def _drawn_trajectory(declared_model, seed: int, t_max: float) -> list[tuple[float, str]]:
    """One trajectory drawn on the declaration route: (date, transition)."""
    from .. import interactive

    drawn = interactive(declared_model, t_max=t_max, seed=seed)
    drawn.advance_to(t_max)
    return [(event.time, event.transition) for event in drawn.history()]


def _replay(
    declared_model,
    library_model,
    rename: dict[str, str],
    stochastic: set[str],
    plan: list[tuple[float, str]],
    t_max: float,
    attributes: list[str],
    states: list[str],
    *,
    recorded: bool,
) -> set[str]:
    """Replay `plan` on both routes in deferred mode and return the
    declaration transitions fired.

    A `recorded` plan is a drawn trajectory, every event in it, of which
    the random ones are fired at their dates. Otherwise the plan lists only
    random transitions to force: each is fired at its date once both routes
    have advanced there, and skipped when the declaration route does not
    have it armed then."""
    from .. import interactive

    sessions = [
        interactive(declared_model, t_max=t_max, deferred_draws=True),
        interactive(library_model, t_max=t_max, deferred_draws=True),
    ]

    def compare(when: str) -> None:
        left = _snapshot(sessions[0], attributes, states)
        right = _snapshot(sessions[1], attributes, states)
        if left != right:
            raise _Divergence(f"{when}: {_first_difference(left, right)}")
        armed = [_armed(sessions[0], rename), _armed(sessions[1], None)]
        if armed[0] != armed[1]:
            names = sorted(set(armed[0]) | set(armed[1]))
            differing = next(n for n in names if armed[0].get(n) != armed[1].get(n))
            raise _Divergence(
                f"{when}: armed transition `{differing}` differs: declaration "
                f"{armed[0].get(differing)}, library {armed[1].get(differing)}"
            )

    # Observe between two distinct event dates, where every route has
    # settled; fire each draw once the events before it have fired, then let
    # the same-date consequences (zero delays) fire before reading.
    previous = 0.0
    compare("at t=0")
    for date, transition in plan:
        if date > previous:
            middle = (previous + date) / 2.0
            for session in sessions:
                session.advance_to(middle)
            compare(f"at t={middle}")
            previous = date
        if transition not in stochastic:
            continue
        if not recorded:
            for session in sessions:
                session.advance_to(date)
            if transition not in {d["transition"] for d in sessions[0].deferred()}:
                continue
        try:
            sessions[0].fire_deferred_at(transition, date)
        except Exception as error:
            raise RuntimeError(
                f"the declaration route cannot replay its own draw `{transition}` at {date}: {error}"
            ) from error
        mapped = rename.get(transition)
        if mapped is None:
            raise _Divergence(f"fired transition `{transition}` has no library counterpart")
        try:
            sessions[1].fire_deferred_at(mapped, date)
        except Exception as error:
            raise _Divergence(f"the library route cannot fire `{mapped}` at {date}: {error}") from error
    if t_max > previous:
        for session in sessions:
            session.advance_to((previous + t_max) / 2.0)
        compare(f"at t={(previous + t_max) / 2.0}")
        for session in sessions:
            session.advance_to(t_max)

    # Same-date events are compared as a set: the contract is the converged
    # states and the event dates, not the propagation order inside a date.
    renamed = set(rename.values())
    histories = [
        sorted((e.time, rename[e.transition]) for e in sessions[0].history() if e.transition in rename),
        sorted((e.time, e.transition) for e in sessions[1].history() if e.transition in renamed),
    ]
    if histories[0] != histories[1]:
        diverging = next(
            (a, b) for a, b in zip(histories[0] + [None], histories[1] + [None]) if a != b
        )
        raise _Divergence(f"event histories part at {diverging}")
    return {e.transition for e in sessions[0].history() if e.transition in stochastic}


#: Seeds of the library route in a statistical comparison are shifted by this
#: much, so the two samples are independent.
_LIBRARY_SEED_OFFSET = 1_000_003


def _statistical(
    declared_model,
    library_model,
    seeds: list[int],
    t_max: float,
    instants: list[float],
    attributes: list[str],
    alpha: float,
) -> tuple[bool, str]:
    """Two-sample z-test on the mean of every observed attribute at every
    instant, Šidák-corrected over the comparisons."""
    import math

    from .. import interactive

    def samples(model, offset: int) -> list[list[float]]:
        rows = []
        for seed in seeds:
            session = interactive(model, t_max=t_max, seed=seed + offset)
            row = []
            for instant in instants:
                session.advance_to(instant)
                row.extend(float(session.attribute(name)) for name in attributes)
            rows.append(row)
        return rows

    left, right = samples(declared_model, 0), samples(library_model, _LIBRARY_SEED_OFFSET)
    columns = len(left[0]) if left else 0
    if columns == 0:
        return False, "nothing compared: the two routes share no observed attribute"
    threshold = 1.0 - (1.0 - alpha) ** (1.0 / columns)
    n = float(len(seeds))
    for column in range(columns):
        a = [row[column] for row in left]
        b = [row[column] for row in right]
        mean_a, mean_b = sum(a) / n, sum(b) / n
        var_a = sum((x - mean_a) ** 2 for x in a) / max(n - 1.0, 1.0)
        var_b = sum((x - mean_b) ** 2 for x in b) / max(n - 1.0, 1.0)
        spread = math.sqrt(var_a / n + var_b / n)
        if spread == 0.0:
            if mean_a != mean_b:
                return False, f"column {column}: constant {mean_a} against {mean_b}"
            continue
        z = abs(mean_a - mean_b) / spread
        p_value = math.erfc(z / math.sqrt(2.0))
        if p_value < threshold:
            name = attributes[column % len(attributes)]
            instant = instants[column // len(attributes)]
            return False, f"`{name}` at t={instant}: mean {mean_a} against {mean_b} (p={p_value:.3g})"
    return True, f"{columns} means agree at alpha={alpha} (Šidák)"


#: Fewest runs per route a statistical comparison accepts.
STATISTICAL_MIN_RUNS = 30


def compare_routes(
    declaration: dict,
    conversion: Conversion,
    *,
    t_max: float,
    seeds: list[int] | range | None = None,
    method: str = "auto",
    instants: list[float] | None = None,
    alpha: float = 0.05,
) -> RouteComparison:
    """Compare the declaration route with the library route of `conversion`.

    `method` is ``auto`` (replay when deferred mode covers the model,
    statistical otherwise), ``replay`` or ``statistical``. `seeds` defaults
    to 10 drawn trajectories for a replay and 200 runs per route for a
    statistical comparison, which refuses fewer than
    :data:`STATISTICAL_MIN_RUNS`.

    A replay fires, after the drawn trajectories, every converted random
    transition they never fired, once, at a third of the horizon (and its
    return at two thirds), so each mode's effects are observed at least
    once. One still unfired (not armed at that date) is listed in
    ``unexercised`` and the level is ``replay-partial``.
    """
    declared_model = declaration_model(declaration)
    library_model = conversion.load()
    attributes, states = _observed(library_model, declared_model)
    if method == "auto":
        method = "replay" if _replay_covers(library_model) else "statistical"
    if method == "statistical":
        seeds = list(range(200) if seeds is None else seeds)
        if len(seeds) < STATISTICAL_MIN_RUNS:
            raise ValueError(
                f"a statistical comparison needs at least {STATISTICAL_MIN_RUNS} runs per route, got {len(seeds)}"
            )
        identical, detail = _statistical(
            declared_model,
            library_model,
            seeds,
            t_max,
            instants or [t_max],
            attributes,
            alpha,
        )
        return RouteComparison(identical, "statistical", detail, len(seeds))
    if method != "replay":
        raise ValueError(f"unknown comparison method {method!r}")

    declared_draws = _stochastic_names(declared_model)
    stochastic = bool(declared_draws or _stochastic_names(library_model))
    seeds = list(range(10) if seeds is None else seeds) if stochastic else [0]
    level = "replay" if stochastic else "exact"
    replay = functools.partial(
        _replay,
        declared_model,
        library_model,
        conversion.rename,
        declared_draws,
        t_max=t_max,
        attributes=attributes,
        states=states,
    )
    fired: set[str] = set()
    for seed in seeds:
        try:
            fired |= replay(_drawn_trajectory(declared_model, seed, t_max), recorded=True)
        except _Divergence as divergence:
            return RouteComparison(False, level, f"seed {seed}: {divergence}", len(seeds))
    returns = {
        name: declared
        for declared, name in conversion.rename.items()
        if declared in declared_draws
    }
    for transition in sorted(declared_draws & set(conversion.rename) - fired):
        if transition in fired:
            continue
        plan = [(t_max / 3.0, transition)]
        target = conversion.rename[transition]
        if target.endswith(f".{_OCC}"):
            back = returns.get(f"{target[: -len(_OCC)]}{_REP}")
            if back is not None:
                plan.append((2.0 * t_max / 3.0, back))
        try:
            fired |= replay(plan, recorded=False)
        except _Divergence as divergence:
            return RouteComparison(
                False, level, f"forcing `{transition}`: {divergence}", len(seeds)
            )
    unexercised = tuple(sorted(declared_draws & set(conversion.rename) - fired))
    observed = f"{len(attributes)} attributes and {len(states)} event states"
    if unexercised:
        return RouteComparison(
            False,
            "replay-partial",
            f"{observed} agree, but {len(unexercised)} transitions never fired: {list(unexercised)[:5]}",
            len(seeds),
            unexercised,
        )
    return RouteComparison(True, level, f"{observed} agree on every trajectory", len(seeds))
