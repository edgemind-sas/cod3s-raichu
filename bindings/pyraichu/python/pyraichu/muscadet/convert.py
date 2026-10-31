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
import collections
import functools
import itertools
import json
import math
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
        "repair_effects",
        "failure_effects_trans",
        "repair_effects_trans",
        "failure_cond",
        "repair_cond",
        "failure_state",
        "repair_state",
        "absorb_state",
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
#: The state an on-demand mode parks in after a lost draw (`raichu-library`).
_PARKED = "parked"


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
    themselves. An external mode's mirror automaton, which the declaration
    route grafts on each target one zero-delay step behind the mode and
    which carries its effects, maps onto the library mode's effect step
    (``effect_step``). A comparison counts every fired transition. ``unverified`` names the components whose
    class may carry behaviour outside the declaration (see
    :data:`AUDITED_CLASSES`). ``inert`` names the declaration route's
    transitions that can never fire because nothing drives them (the mirror
    of an external mode with no occurrence law), which the library route
    does not build.
    """

    library: dict | None
    system: dict | None
    study: dict | None
    refusals: list[Refusal] = field(default_factory=list)
    rename: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    inert: list[str] = field(default_factory=list)

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


class _RefusedAll(Exception):
    """Every construct of one component the converter cannot express."""

    def __init__(self, refusals: list[_Refused]):
        super().__init__("; ".join(str(r) for r in refusals))
        self.refusals = refusals


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
    layer resolves it: an input of that name first, else an output. A
    negated operand reads the attribute `std.bool_not` publishes,
    `<attribute>_not` (:data:`_NEGATION_SUFFIX`)."""
    port, negated = None, False
    if isinstance(operand, dict):
        if set(operand) - {"name", "port", "negate"}:
            raise _Refused(
                "structured production operand",
                f"output `{output}` reads {operand!r}; the std output reads flow states and their negation only",
            )
        port, negated, operand = operand.get("port"), bool(operand.get("negate")), operand["name"]
    if port != "out" and operand in inputs:
        attribute = f"{operand}_fed_in"
    elif operand in outputs:
        attribute = f"{operand}_fed_out"
    else:
        raise _Refused("production operand", f"output `{output}` reads `{operand}`, no discrete flow")
    return f"{attribute}{_NEGATION_SUFFIX}" if negated else attribute


#: Suffix of the attribute `std.bool_not` publishes for a negated operand.
_NEGATION_SUFFIX = "_not"


def _negations(uses: list[dict]) -> list[dict]:
    """The `std.bool_not` uses the production conditions of `uses` read."""
    negated = sorted(
        {
            operand[: -len(_NEGATION_SUFFIX)]
            for use in uses
            for group in use["with"].get("prod_cond", [])
            for operand in group
            if operand.endswith(_NEGATION_SUFFIX)
        }
    )
    return [{"fragment": "std.bool_not", "with": {"of": attribute}} for attribute in negated]


def _flow_out_use(name: str, keywords: dict, inputs: set[str], outputs: set[str]) -> dict:
    # A negated output delivers the negation of its whole delivery (the
    # declaration route wraps it in `not`); production and availability are
    # unchanged.
    negate = bool(keywords.pop("negate", False))
    # A persistent availability (not reset every step) is the same document
    # as long as no mode writes it: `_persistent_outputs` keeps the list and
    # `_mode_effects` refuses a mode writing one.
    keywords.pop("var_fed_available_out_reset", None)
    available = bool(keywords.pop("var_fed_available_out_init", True))
    produced = bool(keywords.pop("var_prod_default", False))
    keywords.pop("var_prod_cond_inner_mode", None)
    condition = keywords.pop("var_prod_cond", None)
    if not condition:
        return {
            "fragment": "std.bool_source_out",
            "with": {
                "name": name,
                "prod_init": produced,
                "fed_available_init": available,
                "negate": negate,
            },
        }
    groups = []
    for group in condition:
        groups.append([_operand_attribute(name, o, inputs, outputs) for o in group])
    return {
        "fragment": "std.bool_flow_out",
        "with": {
            "name": name,
            "prod_cond": groups,
            "fed_available_init": available,
            "negate": negate,
        },
    }


def _tempo_law(output: str, direction: str, law: dict) -> dict[str, Any]:
    """A temporised output's wait as the fragment's two parameters."""
    kind = law.get("distrib")
    if kind == "delay":
        return {f"{direction}_law": "delay", f"{direction}_value": float(law["time"])}
    if kind == "exp":
        rate = float(law.get("rate", 0.0))
        if rate == 0.0:
            # A wait at rate zero never ends: the authoring layer builds no
            # transition, and neither does the fragment's `never` law.
            return {f"{direction}_law": "never"}
        if rate > 0.0:
            return {f"{direction}_law": "exp", f"{direction}_value": rate}
    raise _Refused("temporised output law", f"output `{output}` waits on {law!r} to {direction}")


def _tempo_out_use(name: str, keywords: dict, inputs: set[str], outputs: set[str]) -> dict:
    """muscadet `FlowOutTempo`: the output feeds while its tempo automaton is
    enabled, which waits on its own laws once the production condition holds
    or fails."""
    enable = _tempo_law(name, "enable", keywords.pop("enable_law", {"distrib": "delay", "time": 0.0}))
    disable = _tempo_law(name, "disable", keywords.pop("disable_law", {"distrib": "delay", "time": 0.0}))
    init_enable = bool(keywords.pop("init_enable", False))
    plain = _flow_out_use(name, keywords, inputs, outputs)
    with_: dict[str, Any] = {"name": name}
    if plain["fragment"] == "std.bool_flow_out":
        with_["prod_cond"] = plain["with"]["prod_cond"]
    else:
        with_["conditioned"] = False
        with_["prod_init"] = plain["with"]["prod_init"]
    with_["fed_available_init"] = plain["with"]["fed_available_init"]
    with_["init_enable"] = init_enable
    with_.update(enable)
    with_.update(disable)
    return {"fragment": "std.bool_flow_out_tempo", "with": with_}


def _trigger_out_use(name: str, keywords: dict, inputs: set[str], outputs: set[str]) -> dict:
    """muscadet `FlowOutOnTrigger`: the output feeds while its trigger
    automaton is `up`, which it reaches once the trigger aggregate has been
    false for `time_up`. Whether it starts `up` is a system fact (nothing
    feeds the trigger port), set by :func:`_arm_triggers` once the
    connections are known."""
    time_up = float(keywords.pop("trigger_time_up", 0.0) or 0.0)
    time_down = float(keywords.pop("trigger_time_down", 0.0) or 0.0)
    logic = keywords.pop("trigger_logic", "or")
    plain = _flow_out_use(name, keywords, inputs, outputs)
    if plain["with"].get("negate"):
        raise _Refused("negated triggered output", f"output `{name}`")
    with_: dict[str, Any] = {"name": name}
    if plain["fragment"] == "std.bool_flow_out":
        with_["prod_cond"] = plain["with"]["prod_cond"]
    else:
        with_["conditioned"] = False
        with_["prod_init"] = plain["with"]["prod_init"]
    with_["fed_available_init"] = plain["with"]["fed_available_init"]
    if logic in ("and", "or"):
        with_["logic"] = logic
    elif isinstance(logic, int) and not isinstance(logic, bool) and logic >= 1:
        with_["logic"] = "k"
        with_["k"] = logic
    else:
        raise _Refused("trigger logic", f"output `{name}` aggregates its trigger with {logic!r}")
    with_["time_up"] = time_up
    with_["time_down"] = time_down
    with_["armed"] = False
    return {"fragment": "std.bool_flow_out_trigger", "with": with_}


def _arm_triggers(instances: dict[str, dict], connections: list[dict]) -> None:
    """A triggered output whose trigger port nothing feeds and whose
    `time_up` is zero starts `up` on the declaration route (the trigger is
    absent for ever): the library states it with `armed`."""
    fed = {connection["to"] for connection in connections}
    for name, instance in instances.items():
        for use in instance["uses"]:
            if use["fragment"] == "std.bool_flow_out_trigger":
                with_ = use["with"]
                wired = f"{name}.{with_['name']}_trigger_in" in fed
                with_["armed"] = not wired and with_["time_up"] == 0.0


def _flow_uses(
    spec: dict, name: str, network: _Network
) -> tuple[list[dict], list[tuple[str, dict]], dict[str, str]]:
    """The std fragments a flow component's declaration means, in order, the
    failure modes it declares in its own `failure_modes` section, and the
    transitions its continuous part carries under the same names on both
    routes."""
    try:
        _, _, _, calls = declare._plan(spec, declare.COMPONENT_CLASSES)
    except declare.ComponentSpecError as error:
        raise _Refused("declaration refused by the reader", str(error)) from error
    inputs = {c.keywords.get("name", c.entry) for c in calls if c.method == "add_flow_in"}
    outputs = {
        c.keywords.get("name", c.entry)
        for c in calls
        if c.method in ("add_flow_out", "add_flow_out_tempo")
    }
    uses, embedded, refused = [], [], []
    for call in calls:
        if call.section == "failure_modes" and call.method in _EMBEDDED_LAWS:
            embedded.append((call.method, dict(call.keywords)))
            continue
        if _is_continuous(call):
            continue
        try:
            uses.append(_flow_call_use(spec, call, inputs, outputs))
        except _Refused as error:
            refused.append(error)
    rename: dict[str, str] = {}
    if any(_is_continuous(call) for call in calls):
        try:
            continuous, rename = _continuous_uses(spec, name, network)
            uses += continuous
        except _Refused as error:
            refused.append(error)
        except _RefusedAll as errors:
            refused.extend(errors.refusals)
    if refused:
        # Every blocking construct of the component, not only the first:
        # the coverage matrix counts what blocks a model alone.
        raise _RefusedAll(refused)
    return uses + _negations(uses), embedded, rename


def _flow_call_use(spec: dict, call: Any, inputs: set[str], outputs: set[str]) -> dict:
    """The std fragment one `flows` entry means."""
    if call.section != "flows":
        raise _Refused(f"section `{call.section}`", f"entry {call.entry!r}")
    keywords = dict(call.keywords)
    flow = keywords.pop("name", call.entry)
    if call.method == "add_flow_in":
        use = _flow_in_use(flow, keywords)
    elif call.method == "add_flow_out":
        use = _flow_out_use(flow, keywords, inputs, outputs)
    elif call.method == "add_flow_out_tempo":
        use = _tempo_out_use(flow, keywords, inputs, outputs)
    elif call.method == "add_flow_out_on_trigger":
        use = _trigger_out_use(flow, keywords, inputs, outputs)
    else:
        entry = next((e for e in spec.get("flows") or [] if e.get("name") == flow), {})
        raise _Refused(f"flow class `{entry.get('cls', call.method)}`", f"flow `{flow}`")
    if keywords:
        raise _Refused("flow keyword", f"flow `{flow}` declares {sorted(keywords)}")
    return use


# ---------------------------------------------------------------------------
# Continuous flows: what a component's network part uses
# ---------------------------------------------------------------------------

#: The authoring methods of the continuous flow classes.
_CONTINUOUS_METHODS = ("add_flow_continuous_in", "add_flow_continuous_out")
#: Sections read whole, per component, by :func:`_continuous_uses`.
_CONTINUOUS_SECTIONS = ("capacities", "rules")
#: Construct named for a volume the declaration route serves as a mixture
#: (:meth:`pyraichu.muscadet.ObjFlow._mixed_volume`): pooled serving is a
#: generic format mechanism the library does not have yet.
_POOLED = "pooled serving (mixed volume)"


@dataclass(frozen=True)
class _Network:
    """What the connections say about each continuous end: how many
    connections feed an input, and which inputs (component, flow) an output
    feeds."""

    suppliers: dict[tuple[str, str], int]
    consumers: dict[tuple[str, str], list[tuple[str, str]]]
    wired: dict[str, set[str]]

    @classmethod
    def of(cls, declaration: dict, components: dict) -> _Network:
        suppliers: dict[tuple[str, str], int] = collections.Counter()
        consumers: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for entry in declaration.get("connections") or []:
            source, box = entry.get("source"), str(entry.get("source_box") or "")
            target, inbox = entry.get("target"), str(entry.get("target_box") or "")
            if inbox.endswith("_in"):
                suppliers[(target, inbox[: -len("_in")])] += 1
            if box.endswith("_out") and inbox.endswith("_in"):
                consumers.setdefault((source, box[: -len("_out")]), []).append((target, inbox[: -len("_in")]))
        wired = declare._wired_rate_channels(components, declaration.get("connections"))
        return cls(dict(suppliers), consumers, wired)


def _is_continuous(call: Any) -> bool:
    return call.method in _CONTINUOUS_METHODS or call.section in _CONTINUOUS_SECTIONS


def _profile_refusal(profile: Any, where: str) -> list[_Refused]:
    """`std.time_profile` clamps from below only."""
    if profile is not None and profile.value_max is not None and not math.isinf(profile.value_max):
        return [_Refused("profile upper bound", f"{where} clamps its profile at {profile.value_max}")]
    return []


def _profile_uses(target: str, profile: Any) -> list[dict]:
    """`std.time_profile` for a declared sinusoidal profile."""
    return [
        {
            "fragment": "std.time_profile",
            "with": {
                "target": target,
                "amplitude": float(profile.amplitude),
                "phase_shift": float(profile.phase_shift),
                "period": float(profile.period),
                "offset": float(profile.offset),
                "value_min": float(profile.value_min),
                "initial": float(profile.factor(0.0)),
            },
        }
    ]


def _allocation(flow: Any, consumers: list[tuple[str, str]]) -> dict[str, Any]:
    """The `continuous_flow_out` parameters of an output's split policy."""
    if flow.allocation == "proportional":
        return {}
    if flow.allocation != "shares":
        raise _Refused(f"allocation `{flow.allocation}`", f"output `{flow.name}`")
    declared = dict(flow.allocation_shares or {})
    names = [component for component, _ in consumers]
    if sorted(declared) != sorted(set(names)) or len(names) != len(set(names)):
        raise _Refused(
            "allocation shares not naming each consumer once",
            f"output `{flow.name}` shares {sorted(declared)} over consumers {names}",
        )
    return {
        "allocation": "shares",
        "shares": [
            {"to": {"component": component, "port": f"{inflow}_in"}, "value": float(declared[component])}
            for component, inflow in consumers
        ],
    }


def _rule_set_uses(
    rule_set: Any, inputs: dict[str, Any], outputs: dict[str, Any], bool_inputs: set[str]
) -> list[dict]:
    """A rule set with one guarded rule and one idle rule, as `std.rule_set`
    and its `rule_consumes` / `rule_produces`."""
    where = f"rule set `{rule_set.name}`"
    if rule_set.apportionment:
        raise _Refused("rule set apportionment", where)
    guarded = [rule for rule in rule_set.rules if not rule.is_default]
    idle = [rule for rule in rule_set.rules if rule.is_default]
    if not guarded:
        raise _Refused("always-on rule set", f"{where} has no guarded rule")
    if len(guarded) != 1 or len(idle) != 1:
        raise _Refused(
            "multi-rule rule set", f"{where} has {len(guarded)} guarded and {len(idle)} idle rules"
        )
    active, rest = guarded[0], idle[0]
    if any(rate != 0.0 for rate in [*rest.cons.values(), *rest.prod.values()]):
        raise _Refused("idle rule moving a flow", f"{where}, rule `{rest.name}`")
    if not active.name or not rest.name:
        raise _Refused("unnamed rule", where)
    if set(rest.cons) - set(active.cons) or set(rest.prod) - set(active.prod):
        raise _Refused("idle rule naming a flow the active rule does not", where)
    conditions: list[str] = []
    band: dict[str, Any] = {}
    for operand in active.cond:
        if operand.automaton is not None or operand.negate:
            raise _Refused("rule guard operand", f"{where} reads {operand.label()}")
        if operand.op is None and operand.name in bool_inputs and operand.port in (None, "in"):
            conditions.append(operand.name)
            continue
        if (
            operand.op == ">="
            and operand.release is not None
            and operand.name in inputs
            and operand.port in (None, "in")
            and not band
        ):
            band = {
                "threshold": "band",
                "threshold_flow": operand.name,
                "activate": float(operand.value),
                "release": float(operand.release),
            }
            continue
        raise _Refused("rule guard operand", f"{where} reads {operand.label()}")
    consumes, produces = list(active.cons), list(active.prod)
    for flow in consumes:
        if flow not in inputs:
            raise _Refused("rule consuming no continuous input", f"{where} consumes `{flow}`")
    for flow in produces:
        if flow not in outputs:
            raise _Refused("rule producing no continuous output", f"{where} produces `{flow}`")
    capped = [flow for flow in produces if outputs[flow].max_rate is not None]
    uses = [
        {
            "fragment": "std.rule_set",
            "with": {
                "name": rule_set.name,
                "active": active.name,
                "idle": rest.name,
                "consumes": consumes,
                "produces": produces,
                "capped": capped,
                "conditions": conditions,
                **band,
            },
        }
    ]
    for flow in consumes:
        uses.append(
            {
                "fragment": "std.rule_consumes",
                "with": {
                    "rule_set": rule_set.name,
                    "active": active.name,
                    "flow": flow,
                    "rate": float(active.cons[flow]),
                },
            }
        )
    for flow in produces:
        with_: dict[str, Any] = {
            "rule_set": rule_set.name,
            "active": active.name,
            "flow": flow,
            "rate": float(active.prod[flow]),
        }
        if flow in capped:
            with_["capped"] = True
            with_["max_rate"] = float(outputs[flow].max_rate)
        uses.append({"fragment": "std.rule_produces", "with": with_})
    return uses


def _continuous_uses(spec: dict, name: str, network: _Network) -> tuple[list[dict], dict[str, str]]:
    """The std fragments a component's continuous part means, and the
    transitions they carry under the declaration route's names.

    The declaration is read by the declaration route's own reader
    (:func:`declare.build_component`) and the component it builds is asked
    what each flow is (a pass-through, a volume's, a rule set's, a mixture),
    so no reading rule is written twice. Only the shapes a
    plugin-versus-library bench proves are mapped; every other one is
    refused by name, all of them at once."""
    from .. import muscadet as authoring

    spec = declare._publish_wired_rates(spec, network.wired)
    try:
        obj = declare.build_component(authoring.System(name="convert"), spec)
    except declare.ComponentSpecError as error:
        raise _Refused("declaration refused by the reader", str(error)) from error

    inputs = {flow.name: flow for flow in obj.flows_continuous_in}
    outputs = {flow.name: flow for flow in obj.flows_continuous_out}
    bool_inputs = {flow.name for flow in obj.flows_in}
    refused: list[_Refused] = []
    uses: list[dict] = []
    rename: dict[str, str] = {}

    # Volumes first: a mixture is the one refusal no other reading explains.
    for capacity in obj.capacities:
        where = f"volume `{capacity.name}`"
        mixed = [flow for flow in capacity.flow_names if flow in outputs and obj._mixed_volume(flow) is not None]
        if mixed:
            refused.append(_Refused(_POOLED, f"{where} serves {mixed} as a mixture"))
        elif capacity.side != "out":
            refused.append(_Refused("input-side volume", where))
        elif len(capacity.flows) != 1:
            refused.append(_Refused("multi-flow volume", where))
        elif capacity.flow_names[0] not in inputs or capacity.flow_names[0] not in outputs:
            refused.append(_Refused("volume not on an input and an output", where))
        elif obj._beside_rules(capacity.flow_names[0], "out") is not None:
            refused.append(_Refused("volume beside a rule set", where))
        elif capacity.transmits:
            refused.append(_Refused("transmitting volume (`transmits`)", where))
        else:
            # Measured: once the volume leaves a bound, the two routes date
            # that departure 3e-11 apart and their contents part at 1e-11
            # relative, the algebraic loop through the volume's demand being
            # solved in a different order on each route. That is beyond the
            # comparison's 1e-12; volumes wait for the library to order that
            # loop as the declaration route does.
            refused.append(_Refused("output-side volume", where))

    produced_by: dict[str, list[str]] = {}
    consumed_by: dict[str, list[str]] = {}
    for rule_set in obj.rule_sets:
        for flow in rule_set.produced():
            produced_by.setdefault(flow, []).append(rule_set.name)
        for flow in rule_set.consumed():
            consumed_by.setdefault(flow, []).append(rule_set.name)
    for flow, sets in produced_by.items():
        if len(sets) > 1:
            refused.append(_Refused("output produced by two rule sets", f"output `{flow}` by {sets}"))
    for flow, sets in consumed_by.items():
        if len(sets) > 1:
            refused.append(_Refused("input consumed by two rule sets", f"input `{flow}` by {sets}"))

    for flow in inputs.values():
        where = f"input `{flow.name}`"
        if flow.var_in_default:
            refused.append(_Refused("value of an unconnected continuous input", where))
        if flow.demand_expr is not None:
            refused.append(_Refused("demand equation", where))
        if flow.mixture_demand is not None:
            refused.append(_Refused(_POOLED, f"{where} draws a mixture"))
        sink = not (obj._passes_through(flow.name) or flow.name in consumed_by)
        if flow.profile is not None and not sink:
            refused.append(_Refused("profile on an input that is not a sink", where))
        refused += _profile_refusal(flow.profile, where)
        if flow.publish_rate is not None and flow.publish_rate != authoring.RATE_DELIVERED:
            refused.append(_Refused(f"published input rate `{flow.publish_rate}`", where))
    for flow in outputs.values():
        where = f"output `{flow.name}`"
        if flow.capability_expr is not None:
            refused.append(_Refused("capability equation", where))
        source = not (obj._passes_through(flow.name) or flow.name in produced_by)
        if flow.profile is not None and not source:
            refused.append(_Refused("profile on an output that is not a source", where))
        refused += _profile_refusal(flow.profile, where)
        if flow.max_rate is not None and flow.name not in produced_by:
            refused.append(_Refused("maximum rate on an output no rule set produces", where))
        if flow.publish_rate is not None and flow.publish_rate != authoring.RATE_CAPABILITY:
            refused.append(_Refused(f"published output rate `{flow.publish_rate}`", where))
        if obj._tapped_into(flow.name):
            refused.append(_Refused("tap routed into an output", where))

    rule_uses: list[dict] = []
    for rule_set in obj.rule_sets:
        try:
            rule_uses += _rule_set_uses(rule_set, inputs, outputs, bool_inputs)
        except _Refused as error:
            refused.append(error)
            continue
        guarded = next(rule for rule in rule_set.rules if not rule.is_default)
        idle = next(rule for rule in rule_set.rules if rule.is_default)
        for a, b in ((guarded.name, idle.name), (idle.name, guarded.name)):
            qualified = f"{name}.{rule_set.mode}.{rule_set.name}_{a}_to_{b}"
            rename[qualified] = qualified
    if refused:
        raise _RefusedAll(refused)

    for flow in inputs.values():
        uses.append(
            {
                "fragment": "std.continuous_flow_in",
                "with": {"name": flow.name, "demand": float(flow.var_demand_in_default)},
            }
        )
        if flow.name in consumed_by and network.suppliers.get((name, flow.name), 0) >= 2:
            uses.append({"fragment": "std.surplus_release", "with": {"name": flow.name}})
        if flow.publish_rate is not None:
            uses.append({"fragment": "std.rate_publisher", "with": {"name": flow.name, "of": authoring.RATE_DELIVERED}})
        if obj._passes_through(flow.name) or flow.name in consumed_by:
            continue
        if flow.profile is not None:
            uses += _profile_uses(f"{flow.name}_in_profile", flow.profile)
        uses.append(
            {
                "fragment": "std.continuous_sink",
                "with": {
                    "name": flow.name,
                    "demand": float(flow.var_demand_in_default),
                    "profiled": flow.profile is not None,
                },
            }
        )
    uses += rule_uses
    for flow in outputs.values():
        consumers = network.consumers.get((name, flow.name), [])
        out: dict[str, Any] = {"name": flow.name, **_allocation(flow, consumers)}
        if obj._passes_through(flow.name):
            uses.append({"fragment": "std.pass_through", "with": {"name": flow.name}})
            out["available"] = "produced"
        elif flow.name in produced_by:
            out["available"] = "produced"
        else:
            if flow.profile is not None:
                uses += _profile_uses(f"{flow.name}_out_profile", flow.profile)
            uses.append(
                {
                    "fragment": "std.continuous_source",
                    "with": {
                        "name": flow.name,
                        "rate": float(flow.var_fed_default),
                        "profiled": flow.profile is not None,
                    },
                }
            )
        if flow.publish_rate is not None:
            uses.append({"fragment": "std.rate_publisher", "with": {"name": flow.name, "of": authoring.RATE_CAPABILITY}})
        uses.append({"fragment": "std.continuous_flow_out", "with": out})
    return uses, rename


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


def _embedded_mode(
    component: str, method: str, keywords: dict, uses: list[dict], persistent: frozenset[str] = frozenset()
) -> _EmbeddedMode:
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
        if use["fragment"] in _OUTPUT_FRAGMENTS
    }
    for target in targets:
        if target not in outputs:
            raise _Refused("failure mode target", f"mode `{mode}` cuts `{target}`, no discrete output")
        if outputs[target] is not True:
            raise _Refused(
                "failure mode on an output initially unavailable",
                f"mode `{mode}` cuts `{target}`",
            )
        if f"{target}_fed_available_out" in persistent:
            raise _Refused(_PERSISTENT, f"mode `{mode}` writes `{component}.{target}_fed_available_out`")
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
    "std.bool_flow_out": ("fed_available_init", "negate"),
    "std.bool_source_out": ("prod_init", "fed_available_init", "negate"),
    "std.logic_gate": ("k",),
    "std.bool_flow_out_tempo": (
        "prod_init",
        "fed_available_init",
        "init_enable",
        "enable_value",
        "disable_value",
    ),
    "std.continuous_flow_in": ("demand",),
    "std.continuous_source": ("rate",),
    "std.continuous_sink": ("demand",),
    "std.time_profile": ("amplitude", "phase_shift", "period", "offset", "value_min", "initial"),
    "std.volume_flow": ("capacity", "content_init", "demand", "fill_rate"),
    "std.volume_outlet": ("rate",),
    "std.volume_total": ("capacity",),
    "std.rule_set": ("activate", "release"),
    "std.rule_consumes": ("rate",),
    "std.rule_produces": ("rate", "max_rate"),
}

#: The parameter naming a use, for the type parameter a value becomes.
_USE_LABELS = ("name", "flow", "volume", "target", "rule_set")

#: The fragments carrying a discrete output.
_OUTPUT_FRAGMENTS = (
    "std.bool_flow_out",
    "std.bool_source_out",
    "std.bool_flow_out_tempo",
    "std.bool_flow_out_trigger",
)
#: The output fragments whose production a flag `conditioned` may leave to a mode.
_CONDITIONED_FRAGMENTS = ("std.bool_flow_out_tempo", "std.bool_flow_out_trigger")


def _persistent_outputs(spec: dict) -> set[str]:
    """The availabilities of `spec`'s outputs that the declaration route
    never resets (`var_fed_available_out_reset` false)."""
    return {
        f"{entry.get('name')}_fed_available_out"
        for entry in spec.get("flows") or []
        if entry.get("var_fed_available_out_reset") is False
    }


_PERSISTENT = "availability not reset (`var_fed_available_out_reset` false)"


def _initial_values(uses: list[dict]) -> dict[str, bool]:
    """The initial value of each output availability, as the std fragments
    seed it from the declaration's `var_fed_available_out_init`. Only the
    availability: a production is written by its condition or seeded from
    `var_prod_default`, which no fragment parameter of a conditioned output
    carries, so no rest or no-op is ever read off it."""
    return {
        f"{use['with']['name']}_fed_available_out": bool(use["with"].get("fed_available_init", True))
        for use in uses
        if use["fragment"] in _OUTPUT_FRAGMENTS
    }


def _refuse_shared_no_ops(instances: dict[str, dict], embedded: list, refuse, notes: list[str]) -> None:
    """A dropped no-op effect is exact only while no other mode writes the
    attribute another value: the declaration route then chains the writers
    in declaration order, the no-op among them."""
    for target, instance in instances.items():
        for attribute, value, mode in instance.get("no_ops", []):
            writers = [
                effect
                for entry in instance["modes"].values()
                for effect in entry.get("effects", [])
                if effect["attribute"] == attribute and effect["value"] != value
            ] + [
                effect
                for own in embedded
                if own.component == target
                for effect in own.entry.get("effects", [])
                if effect["attribute"] == attribute and effect["value"] != value
            ]
            if writers:
                refuse(
                    mode,
                    _Refused(
                        "no-op mode effect on a shared attribute",
                        f"mode `{mode}` holds `{target}.{attribute}` at its initial value, another mode writes it",
                    ),
                )
            else:
                notes.append(f"mode `{mode}`: the effect on `{target}.{attribute}` holds its initial value, a no-op")


def _targetable(uses: list[dict]) -> set[str]:
    """The attributes a mode may write on an instance using `uses`."""
    attributes = set()
    for use in uses:
        if use["fragment"] not in _OUTPUT_FRAGMENTS:
            continue
        name = use["with"]["name"]
        attributes.add(f"{name}_fed_available_out")
        # The production is a mode's to write only where no condition writes it.
        if use["fragment"] == "std.bool_source_out" or (
            use["fragment"] in _CONDITIONED_FRAGMENTS and use["with"].get("conditioned") is False
        ):
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
    #: An external mode, whose mirror is the library mode's effect step.
    external: bool = False


def _rewrite_leaves(cond: Any, modes: dict[str, _ModeRef], gates: dict[str, str]) -> Any:
    """Point every leaf of a condition at what the library route holds.

    A state leaf on a declared mode reads the mode automaton of the target
    instance for its own ``fm`` automaton, and the mode's effect step for the
    mirror an external mode grafts on its target, which moves one step
    later; a leaf on a gate's ``result`` reads the gate input.
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
                    mirror = "__" in leaf["state"]
                    state = leaf["state"].rsplit("__", 1)[-1]
                    if state == ref.failure_state:
                        state = _OCC
                    elif state == ref.repair_state:
                        state = _REP
                    else:
                        raise _Refused(
                            "condition on a mode state", f"state `{leaf['state']}` of `{obj}`"
                        )
                    automaton = ref.mode
                    if mirror and ref.external:
                        automaton = f"{ref.mode}__effect"
                    leaf.update(obj=ref.target, automaton=automaton, state=state)
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


def _law(spec: Any, what: str, *, occurrence: bool = False) -> dict:
    """A declared mode law in the study's spelling. An occurrence may be on
    demand (`inst`: the probability the mode occurs on each rising edge of
    its condition); a repair is timed."""
    kind = spec.get("law") or spec.get("cls") or spec.get("distrib")
    if kind == "inst":
        if not occurrence:
            raise _Refused("on-demand repair law (`inst`)", what)
        prob = spec.get("prob", spec.get("gamma"))
        if not isinstance(prob, (int, float)) or isinstance(prob, bool) or not 0.0 <= prob <= 1.0:
            raise _Refused("on-demand law (`inst`)", f"{what}: probability {prob!r}")
        return {"distrib": "inst", "probs": [float(prob)]}
    try:
        return plugin._law(spec)
    except ValueError as error:
        raise _Refused("mode law", f"{what}: {error}") from error


def _unknown_keys(obj: dict, known: frozenset, what: str) -> list[_Refused]:
    """One refusal per key this converter does not read."""
    extra = sorted(
        key for key, value in obj.items() if key not in known and value not in (None, {}, [], False)
    )
    return [_Refused(f"{what} key `{key}`", f"declares {extra}") for key in extra]


def _check_keys(obj: dict, known: frozenset, what: str) -> None:
    unknown = _unknown_keys(obj, known, what)
    if unknown:
        raise unknown[0]


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


def _declared_laws(obj: dict, targets: list[str]) -> tuple[list, list]:
    """A mode's per-order occurrence and repair laws. An `ObjFMInst` object
    writes them as bare numbers (each occurrence a probability, the repair
    an exponential rate broadcast over the orders, zero meaning none), the
    other modes as law objects."""
    failures = obj.get("failure") or []
    repairs = obj.get("repair") or []
    # The declaration route builds the orders 1 to n of n targets only, and
    # ignores laws listed past them.
    n = len(targets)
    if obj.get("type") != "ObjFMInst":
        return list(failures)[:n], list(repairs)[:n]
    failures = failures if isinstance(failures, list) else [failures]
    failures = [
        {"law": "inst", "prob": f} if isinstance(f, (int, float)) and not isinstance(f, bool) else f
        for f in failures
    ]
    if not isinstance(repairs, list):
        repairs = [repairs] * len(targets)
    repairs = [
        {"law": "exp", "rate": float(r)} if isinstance(r, (int, float)) and not isinstance(r, bool) else r
        for r in repairs
    ]
    return failures[:n], repairs[:n]


def _mode_effects(
    obj: dict,
    mode: str,
    targets: list[str],
    instances: dict[str, dict],
    problems: list,
    *,
    stepped: bool = False,
    on_demand: bool = False,
    overlapping: bool = False,
    repaired: bool = True,
    common_cause: bool = False,
    notes: list[str] | None = None,
) -> dict:
    """The catalogue entry of a mode on its targets, checking its effects.

    Held effects (`failure_effects`, rest values in `repair_effects`) become
    the entry's `effects`; one-shot writes (`failure_effects_trans`,
    `repair_effects_trans`) its `edge_effects`, as the declaration route
    writes them on the mode's edges (plugin `_check_one_shot_effects`,
    `_edge_effects`). Two shapes the declaration route absorbs into the held
    effect are held here too (`plugin._absorbed_pulses`): held while failed
    and handed back once at the repair, and held at rest and written once
    entering the failure. A held effect on a persistent gate (never reset)
    with no rest value latches on the declaration route: it is written once
    entering the failure and never restored, which is an occurrence edge
    effect (`plugin._latch_held_writes_on_persistent_gates`); with a rest
    value equal to the initial value it alternates with the mode, which is
    the held effect itself.

    `stepped` says the effects ride on an effect step (an external mode),
    `on_demand` that the occurrence is a draw, `overlapping` that several
    combinations of a common cause cover one target without a step, and
    `repaired` that every automaton of the mode has a repair edge, and
    `common_cause` that the mode becomes a group of combinations."""
    name = obj.get("name")
    failure = dict(obj.get("failure_effects") or {})
    rest_values = dict(obj.get("repair_effects") or {})
    pulse_on = dict(obj.get("failure_effects_trans") or {})
    pulse_off = dict(obj.get("repair_effects_trans") or {})
    same_face = sorted((set(failure) & set(pulse_on)) | (set(rest_values) & set(pulse_off)))
    if same_face:
        problems.append(
            _Refused(
                "held and one-shot write on the same face",
                f"mode `{name}` holds and writes once {same_face} on one face",
            )
        )
    # Held at rest and written once entering the failure: the write is the
    # failure value of a held effect whose rest is the rest value.
    rest_shape = sorted((set(pulse_on) & set(rest_values)) - set(failure))
    for attribute in rest_shape:
        failure[attribute] = pulse_on.pop(attribute)
        # The declaration route exempts only the hand-back from its refusal of
        # a one-shot write on a gate reset every step.
        reset = [t for t in targets if attribute not in instances[t].get("persistent", ())]
        if reset:
            problems.append(
                _Refused(
                    "one-shot write on an availability reset every step",
                    f"mode `{name}` writes `{attribute}` once on {reset}; the declaration route undoes it at the next step",
                )
            )
    # Held while failed and handed back once at the repair: the hand-back is
    # the held effect's return to its initial value, which it must be.
    handback = sorted(set(pulse_off) & set(failure))
    for attribute in handback:
        value = pulse_off.pop(attribute)
        for target in targets:
            initial = _initial_values(instances[target]["uses"]).get(attribute)
            if initial is not None and initial != value:
                problems.append(
                    _Refused(
                        "hand-back value different from the initial value",
                        f"`{target}.{attribute}` is handed back at {value!r}, starts at {initial!r}",
                    )
                )
    effects = []
    for attribute, value in sorted(failure.items()):
        if not isinstance(value, bool):
            problems.append(_Refused("non-boolean mode effect", f"`{attribute}` = {value!r}"))
            continue
        effects.append({"attribute": attribute, "value": value})
    edge_effects = []
    for face, pulses in (("occurrence", pulse_on), ("repair", pulse_off)):
        for attribute, value in sorted(pulses.items()):
            if not isinstance(value, bool):
                problems.append(_Refused("non-boolean mode effect", f"`{attribute}` = {value!r}"))
                continue
            edge_effects.append({"attribute": attribute, "value": value, "on": face})

    def persistent(attribute: str) -> bool:
        return all(attribute in instances[t].get("persistent", ()) for t in targets)

    # An effect holding the attribute's initial value on every target is a
    # no-op on the declaration route (held at that value while failed, at
    # rest at it too). The library has no such effect: it is dropped, and
    # `_refuse_shared_no_ops` refuses it if another mode writes there.
    no_ops = [
        effect
        for effect in effects
        if all(
            _initial_values(instances[t]["uses"]).get(effect["attribute"]) == effect["value"] for t in targets
        )
    ]
    for effect in no_ops:
        effects.remove(effect)
        for target in targets:
            instances[target].setdefault("no_ops", []).append((effect["attribute"], effect["value"], name))
    held = {effect["attribute"] for effect in effects} | {effect["attribute"] for effect in no_ops}
    latched: set[str] = set()
    for effect in list(effects):
        attribute = effect["attribute"]
        if not persistent(attribute):
            if any(attribute in instances[t].get("persistent", ()) for t in targets):
                problems.append(
                    _Refused(_PERSISTENT, f"mode `{name}` writes `{attribute}`, persistent on some targets only")
                )
            continue
        absorbed = attribute in rest_shape or attribute in handback
        if overlapping and not absorbed:
            problems.append(
                _Refused(
                    "held write on a persistent gate over overlapping combinations",
                    f"mode `{name}` holds `{attribute}` while any of several combinations is failed",
                )
            )
        elif (absorbed or attribute in rest_values) and on_demand and not stepped and attribute not in handback:
            # The declaration route writes it on the edges entering and
            # leaving `occ`, and a draw that may park enters it on a branch.
            problems.append(
                _Refused(
                    "held write on a persistent gate under an on-demand draw",
                    f"mode `{name}` holds `{attribute}` on a state a draw that may park enters",
                )
            )
        elif absorbed or attribute in rest_values:
            # Alternates with the mode: the held effect, its rest checked below.
            continue
        elif on_demand and not stepped:
            problems.append(
                _Refused(
                    "latched write on an on-demand draw",
                    f"mode `{name}` latches `{attribute}` on a draw that may park",
                )
            )
        else:
            effects.remove(effect)
            latched.add(attribute)
            edge_effects.append({"attribute": attribute, "value": effect["value"], "on": "occurrence"})
    # Writes on a persistent gate, for the check across modes
    # (`_refuse_shared_persistent_writes`); a dropped no-op still holds.
    for target in targets:
        shared = instances[target].setdefault("held_writers", [])
        for effect in effects + no_ops:
            attribute = effect["attribute"]
            shape = "handback" if attribute in handback else "rest" if attribute in rest_shape else None
            shared.append((attribute, shape, effect["value"], name))
        for edge in edge_effects:
            if edge["attribute"] in latched:
                # A latch is a held write on the declaration route.
                shared.append((edge["attribute"], None, edge["value"], name))
        record = instances[target].setdefault("persistent_writers", [])
        for effect in effects + no_ops:
            if effect["attribute"] in instances[target].get("persistent", ()):
                record.append(
                    (effect["attribute"], "handback" if effect["attribute"] in handback else "held", name)
                )
        for edge in edge_effects:
            # A latch is a held write on the declaration route.
            record.append((edge["attribute"], "held" if edge["attribute"] in latched else "edge", name))
    pulsed = {e["attribute"] for e in edge_effects}
    for target in targets:
        allowed = _targetable(instances[target]["uses"])
        for effect in effects + edge_effects:
            if effect["attribute"] not in allowed:
                problems.append(
                    _Refused("effect on a non-targetable attribute", f"`{target}.{effect['attribute']}`")
                )
        for attribute in sorted(pulsed):
            if attribute in allowed and attribute not in instances[target].get("persistent", ()):
                problems.append(
                    _Refused(
                        "one-shot write on an availability reset every step",
                        f"mode `{name}` writes `{target}.{attribute}` once; the declaration route undoes it at the next step",
                    )
                )
        # The declaration route holds a failure effect's attribute at its
        # `repair_effects` value at rest, else at its initial value; the
        # library holds it at its initial value. A rest value on an
        # attribute the mode does not hold is never written there.
        initial = _initial_values(instances[target]["uses"])
        for attribute, value in sorted(rest_values.items()):
            if attribute not in held and attribute in instances[target].get("persistent", ()):
                # The declaration route refuses a repair effect on a gate
                # that memorises with no failure counterpart.
                problems.append(
                    _Refused(_PERSISTENT, f"mode `{name}` rests `{target}.{attribute}`, holding no failure on it")
                )
            if attribute in held and initial.get(attribute) != value:
                problems.append(
                    _Refused(
                        "held rest value different from the initial value",
                        f"`{target}.{attribute}` rests at {value!r}, starts at {initial.get(attribute)!r}",
                    )
                )
    if overlapping:
        both = sorted(
            {e["attribute"] for e in edge_effects if e["on"] == "occurrence"}
            & {e["attribute"] for e in edge_effects if e["on"] == "repair"}
        )
        if both:
            problems.append(
                _Refused(
                    "one-shot writes on both faces over overlapping combinations",
                    f"mode `{name}` writes {both} on both faces",
                )
            )
    dropped = [e for e in edge_effects if e["on"] == "repair"] if not (repaired or stepped) else []
    if dropped:
        # An effect step always has its repair edge; without one, a mode's
        # repair write lands on its own repair edge.
        if not common_cause:
            # No repair edge: the write never happens on either route.
            for edge in dropped:
                edge_effects.remove(edge)
            if notes is not None:
                notes.append(f"mode `{name}` has no repair: its repair writes never fire")
        else:
            problems.append(
                _Refused(
                    "repair write on a common-cause order without repair",
                    f"mode `{name}` writes {sorted(e['attribute'] for e in dropped)} at a repair some order lacks",
                )
            )
    entry: dict[str, Any] = {"name": mode, "nature": "failure"}
    if effects:
        entry["effects"] = effects
    if edge_effects:
        entry["edge_effects"] = sorted(edge_effects, key=lambda e: (e["on"], e["attribute"]))
    for target in targets:
        known = instances[target]["modes"].get(mode)
        if known is not None and known != entry:
            problems.append(_Refused("mode name clash", f"`{mode}` on `{target}`"))
    return entry


def _refuse_ill_posed_sharing(instances: dict[str, dict], refuse) -> None:
    """Several modes holding one attribute, one of them through a one-shot
    write it absorbs, have no answer on the declaration route
    (`plugin._refuse_ill_posed_one_shot_sharing`, persistent or not): a mode
    holding it at rest and writing it once entering the failure beside any
    other writer, a hand-back beside a holder without one, and hand-backs of
    different clamp values."""
    for target, instance in instances.items():
        by_attribute: dict[str, list[tuple[str | None, Any, str]]] = {}
        for attribute, shape, value, mode in instance.get("held_writers", []):
            by_attribute.setdefault(attribute, []).append((shape, value, mode))
        for attribute, writers in sorted(by_attribute.items()):
            modes = sorted({mode for _, _, mode in writers})
            shapes = [shape for shape, _, _ in writers]
            if len(modes) < 2 or not any(shapes):
                continue
            if "rest" in shapes:
                reason = "one holds it at rest and writes it once entering the failure"
            elif None in shapes:
                reason = "some hand it back at their repair, others hold it without handing it back"
            elif len({json.dumps(value) for _, value, _ in writers}) > 1:
                reason = "they hand it back with different clamp values"
            else:
                continue
            for mode in modes:
                refuse(
                    mode,
                    _Refused(
                        "ill-posed sharing of a handed-back attribute",
                        f"`{target}.{attribute}` is written by {modes}: {reason}",
                    ),
                )


def _refuse_shared_persistent_writes(instances: dict[str, dict], refuse) -> None:
    """A persistent gate memorises, so the declaration route refuses two of
    its sharings (`plugin._latch_held_writes_on_persistent_gates`): a held
    write beside a one-shot write, and several modes holding it, unless
    each holds it and hands it back at its repair."""
    for target, instance in instances.items():
        by_attribute: dict[str, list[tuple[str, str]]] = {}
        for attribute, kind, mode in instance.get("persistent_writers", []):
            by_attribute.setdefault(attribute, []).append((kind, mode))
        for attribute, writers in sorted(by_attribute.items()):
            holders = sorted({mode for kind, mode in writers if kind != "edge"})
            if any(kind == "edge" for kind, _ in writers) and holders:
                for mode in holders:
                    refuse(
                        mode,
                        _Refused(
                            "held and one-shot writes on one persistent gate",
                            f"`{target}.{attribute}` is held by {holders} and written once by another mode",
                        ),
                    )
            elif len(holders) > 1 and any(kind == "held" for kind, _ in writers):
                for mode in holders:
                    refuse(
                        mode,
                        _Refused(
                            "persistent gate held by several modes",
                            f"`{target}.{attribute}` is held by {holders}",
                        ),
                    )


def _convert_group(
    name: str,
    obj: dict,
    mode: str,
    targets: list[str],
    failures: list,
    repairs: list,
    instances: dict[str, dict],
    rename: dict[str, str],
    problems: list,
) -> dict:
    """A mode with an active order above one, as a study common-cause group:
    one combination per active order, the mirror an external mode grafts on
    each target mapped onto the member's effect step."""
    group = _clean_name(name)
    for key in ("failure_cond", "repair_cond"):
        if obj.get(key) not in (True, None) and not (key == "repair_cond" and obj[key] is False):
            problems.append(
                _Refused("common-cause group condition", f"`{key}` on a mode of {len(targets)} targets")
            )
    orders = []
    for index, law in enumerate(failures):
        if law is None:
            continue
        entry: dict[str, Any] = {"order": index + 1}
        try:
            entry["occurrence"] = _law(law, "occurrence", occurrence=True)
        except _Refused as error:
            problems.append(error)
            continue
        repair_law = repairs[index] if index < len(repairs) and obj.get("repair_cond") is not False else None
        if repair_law is not None:
            try:
                repair = _law(repair_law, "repair")
            except _Refused as error:
                problems.append(error)
                repair = None
            if repair is not None and not (repair["distrib"] == "exp" and float(repair.get("rate", 0.0)) == 0.0):
                entry["repair"] = repair
        orders.append(entry)
    built = [order["order"] for order in orders]
    entry = _mode_effects(
        obj,
        mode,
        targets,
        instances,
        problems,
        stepped=obj.get("behaviour") == "external",
        on_demand=any(order["occurrence"]["distrib"] == "inst" for order in orders),
        overlapping=obj.get("behaviour") != "external"
        and sum(math.comb(len(targets) - 1, k - 1) for k in built) > 1,
        repaired=all("repair" in order for order in orders),
        common_cause=True,
    )
    if problems:
        raise _RefusedAll(problems)
    for target in targets:
        instances[target]["modes"][mode] = entry
    spec: dict[str, Any] = {"name": group, "mode": mode, "instances": list(targets), "orders": orders}
    if obj.get("behaviour") == "external":
        spec["effect_step"] = "zero_delay"
    inst_object = obj.get("type") == "ObjFMInst"
    for order in orders:
        k = order["order"]
        on_demand = order["occurrence"]["distrib"] == "inst"
        for comb in itertools.combinations(range(len(targets)), k):
            library = f"{group}.cc_{'_'.join(str(i + 1) for i in comb)}"
            for declared, which in _combination_transitions(name, obj, comb, len(targets), on_demand, inst_object):
                rename[declared] = f"{library}.{which}"
    if obj.get("behaviour") == "external":
        failure_state, repair_state = _mode_states(obj)
        mode_name = obj.get("mode_name") or name
        for target in targets:
            for state, which in ((failure_state, _OCC), (repair_state, _REP)):
                rename[f"{target}.{name}.{mode_name}__{state}"] = f"{target}.{group}__effect.{which}"
    return spec


def _combination_transitions(
    name: str, obj: dict, comb: tuple, order_max: int, on_demand: bool, inst_object: bool
) -> list[tuple[str, str]]:
    """(declaration transition, library transition) of one combination, as
    the plugin names them (`_expand_objfm`, `plugin._ccf_suffix`)."""
    failure_state, repair_state = _mode_states(obj)
    default_parked = "not_occ" if inst_object else f"not_{failure_state}"
    parked = obj.get("absorb_state") or default_parked
    suffix = plugin._ccf_suffix(comb, order_max)
    automaton = f"{name}.fm{suffix}"
    names = []
    if on_demand:
        names.append((f"{automaton}.{failure_state}{suffix}", _OCC))
        names.append((f"{automaton}.{parked}{suffix}", _PARKED))
    elif suffix:
        names.append((f"{automaton}.{failure_state}{suffix}", _OCC))
    else:
        names.append((f"{automaton}.failure", _OCC))
    repair = f"{automaton}.{repair_state}{suffix}" if suffix or inst_object else f"{automaton}.repair"
    names.append((repair, _REP))
    return names


def _mode_transitions(
    name: str, obj: dict, targets: list[str], on_demand: bool, inst_object: bool = False
) -> list[tuple[str, str, str]]:
    """(declaration transition, target, library transition) of a mode with
    order 1 only, as the plugin names them: `failure`/`repair` alone,
    `occ__cc_i` per target otherwise; an on-demand draw is named after the
    state it enters, and its re-arm after the state it parks in. An
    `ObjFMInst` names its repair after the state it returns to, and parks in
    `not_occ` by default."""
    return [
        (declared, target, which)
        for index, target in enumerate(targets)
        for declared, which in _combination_transitions(
            name, obj, (index,), len(targets), on_demand, inst_object
        )
    ]


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
            label = next(use["with"][k] for k in _USE_LABELS if isinstance(use["with"].get(k), str))
            parameter = f"{label}_{key}"
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
    network = _Network.of(declaration, components)
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
                uses, embedded, continuous_rename = _flow_uses(spec, name, network)
                rename.update(continuous_rename)
                own, refused = [], []
                for method, keywords in embedded:
                    try:
                        own.append(
                            _embedded_mode(name, method, keywords, uses, frozenset(_persistent_outputs(spec)))
                        )
                    except _Refused as error:
                        refused.append(error)
                if refused:
                    raise _RefusedAll(refused)
                embedded_modes.extend(own)
        except _Refused as error:
            refuse(name, error)
            continue
        except _RefusedAll as errors:
            for error in errors.refusals:
                refuse(name, error)
            continue
        instances[name] = {
            "key": _type_key(spec, name),
            "uses": uses,
            "modes": {},
            "persistent": _persistent_outputs(spec),
        }
        for use in uses:
            if use["fragment"] == "std.bool_flow_out_tempo":
                # Both routes name the tempo automaton and its waits alike.
                flow = use["with"]["name"]
                for direction in ("enable", "disable"):
                    qualified = f"{name}.{flow}_tempo.{flow}_{direction}"
                    rename[qualified] = qualified
            if use["fragment"] == "std.bool_flow_out_trigger":
                # And the trigger automaton and its two delays.
                flow = use["with"]["name"]
                for direction in ("up", "down"):
                    qualified = f"{name}.{flow}_trigger.{flow}_trigger_{direction}"
                    rename[qualified] = qualified
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
            mode_refs[name] = _ModeRef(
                mode_targets[0],
                chosen,
                *_mode_states(obj),
                external=obj.get("behaviour") == "external",
            )

    groups: list[dict] = []
    inert: list[str] = []
    for name, obj in mode_objects.items():
        try:
            activations.extend(
                _convert_mode(
                    name, obj, mode_names[name], instances, mode_refs, gates, rename, notes, groups, inert
                )
            )
        except _Refused as error:
            refuse(name, error)
        except _RefusedAll as errors:
            for error in errors.refusals:
                refuse(name, error)

    _refuse_shared_no_ops(instances, embedded_modes, refuse, notes)
    _refuse_ill_posed_sharing(instances, refuse)
    _refuse_shared_persistent_writes(instances, refuse)

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

    _arm_triggers(instances, connections)

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
    if groups:
        study_body["common_cause_groups"] = groups
    if schedule:
        study_body["simulation"] = {
            "method": "monte_carlo",
            "nb_runs": int(nb_runs or 1),
            "horizon": float(max(schedule)),
            "instants": [float(t) for t in schedule],
            "seed": int(seed),
        }
    study = {"raichu_study": {"format": DOCUMENT_FORMAT}, "study": study_body}
    return Conversion(library, system, study, [], rename, notes, unverified, inert)


def _convert_mode(
    name: str,
    obj: dict,
    mode: str,
    instances: dict[str, dict],
    mode_refs: dict[str, _ModeRef],
    gates: dict[str, str],
    rename: dict[str, str],
    notes: list[str],
    groups: list[dict],
    inert: list[str],
) -> list[dict]:
    """A declared mode as a catalogue entry on each target and one study
    activation per target, or a common-cause group when an order above one
    is active. Every construct that blocks the mode is named
    (:class:`_RefusedAll`), not only the first met."""
    problems = _unknown_keys(obj, _MODE_KEYS, "mode")
    behaviour = obj.get("behaviour")
    if behaviour not in (None, "internal", "external"):
        problems.append(
            _Refused(f"mode behaviour `{behaviour}`", "only internal and external modes")
        )
    targets = obj.get("targets") or []
    if not targets:
        raise _RefusedAll([*problems, _Refused("mode with no target", "a mode acts on at least one component")])
    missing = [t for t in targets if t not in instances]
    if missing:
        # Nothing more can be read off a refused target.
        raise _RefusedAll([*problems, _Refused("mode on a refused component", f"targets {missing}")])

    inst_object = obj.get("type") == "ObjFMInst"
    failures, repairs = _declared_laws(obj, targets)
    if any(law is not None for law in failures[1:]):
        groups.append(
            _convert_group(name, obj, mode, targets, failures, repairs, instances, rename, problems)
        )
        return []
    failure = repair_law = None
    try:
        failure = _order_laws(failures, targets, "failure")
    except _Refused as error:
        problems.append(error)
    repair_cond = obj.get("repair_cond", True)
    if repair_cond is not False:
        try:
            repair_law = _order_laws(repairs, targets, "repair")
        except _Refused as error:
            if all(p.construct != error.construct for p in problems):
                problems.append(error)
    if failure is None and not problems:
        # No order carries a law (a zero rate reads as none): the declaration
        # route builds no automaton either, so the mode is not activated.
        notes.append(f"mode `{name}` declares no active occurrence law and is not activated")
        if behaviour == "external":
            # The declaration route still grafts the mirror on each target,
            # monitored; no automaton drives it.
            failure_state, repair_state = _mode_states(obj)
            mode_name = obj.get("mode_name") or name
            inert.extend(
                f"{target}.{name}.{mode_name}__{state}"
                for target in targets
                for state in (failure_state, repair_state)
            )
        return []
    occurrence = repair = None
    try:
        occurrence = None if failure is None else _law(failure, "occurrence", occurrence=True)
    except _Refused as error:
        problems.append(error)
    try:
        repair = None if repair_law is None else _law(repair_law, "repair")
    except _Refused as error:
        problems.append(error)
    if repair is not None and repair["distrib"] == "exp" and float(repair.get("rate", 0.0)) == 0.0:
        # An exponential repair at rate zero never fires: no repair.
        repair = None

    entry = _mode_effects(
        obj,
        mode,
        targets,
        instances,
        problems,
        stepped=behaviour == "external",
        on_demand=occurrence is not None and occurrence["distrib"] == "inst",
        repaired=repair is not None,
        notes=notes,
    )

    cond = obj.get("failure_cond", True)
    conditioned_repair = repair is not None and repair_cond not in (True, None)
    logic = {
        "modes": mode_refs,
        "gates": gates,
        "inner": obj.get("cond_inner_logic", "all"),
        "outer": obj.get("cond_outer_logic", "any"),
    }
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
                activation["condition"] = _condition(cond, carrier=target, **logic)
            except (ValueError, KeyError) as error:
                problems.append(_Refused("mode condition", str(error)))
        if conditioned_repair:
            # Read like the occurrence condition: leaves with no `obj` on the
            # target, the same inner and outer logic (plugin `_expand_objfm`).
            try:
                activation["repair_condition"] = _condition(repair_cond, carrier=target, **logic)
            except (ValueError, KeyError) as error:
                problems.append(_Refused("repair condition", str(error)))
        if behaviour == "external":
            # Its effects ride one zero-delay step behind it, as on the
            # declaration route.
            activation["effect_step"] = "zero_delay"
        activations.append(activation)
    if problems:
        raise _RefusedAll(problems)
    for target in targets:
        instances[target]["modes"][mode] = entry
    on_demand = occurrence["distrib"] == "inst"
    for declared, target, which in _mode_transitions(name, obj, targets, on_demand, inst_object):
        rename[declared] = f"{target}.{mode}.{which}"
    if behaviour == "external":
        # The mirror automaton the plugin grafts on each target, named after
        # the mode object, its states after the mode (`plugin._expand_objfm`),
        # is the library mode's effect step.
        failure_state, repair_state = _mode_states(obj)
        mode_name = obj.get("mode_name") or name
        for target in targets:
            for state, which in ((failure_state, _OCC), (repair_state, _REP)):
                rename[f"{target}.{name}.{mode_name}__{state}"] = f"{target}.{mode}__effect.{which}"
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
    effects were never observed), ``tolerance`` (an exact or replayed
    comparison in which some values or event dates agreed only within the
    float-reordering tolerance of 1e-12, named in ``detail``) or
    ``statistical`` (attribute means over independent runs, a two-sample
    z-test per observation, which only shows that no difference was
    detected at that sample size). A comparison
    stops at ``structure``, not identical, when both routes would not record
    the same sequence events (monitored transitions, their kinds and cycle
    groups), before any trajectory runs.

    ``identical`` says no difference was found. Only ``exact``, ``replay``
    and ``tolerance`` make it a proof of equality: a partial replay is
    reported as not identical, and a statistical pass only bounds the
    difference by the test's power.
    """

    identical: bool
    level: str
    detail: str
    trajectories: int
    unexercised: tuple[str, ...] = ()


class _Divergence(Exception):
    pass


class _Unordered(Exception):
    """A draw the replay cannot place within its instant: its law's mean
    delay is below the floating-point resolution of its date, so the drawn
    trajectory ordered it among same-date events by arming wave, not by date,
    which deferred mode does not record."""


def _within_the_instant(law_json: str, date: float) -> bool:
    import math

    law = json.loads(law_json)
    rate = law.get("rate") if law.get("distrib") == "exp" else None
    return rate is not None and float(rate) > 0.0 and 1.0 / float(rate) < math.ulp(max(date, 1.0))


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


#: Laws that fire without a random date.
_DATED_LAWS = frozenset({"delay", "watched", "inst"})


def _is_choice(transition: dict) -> bool:
    """An instantaneous transition drawing one of several branches."""
    if transition.get("distrib") != "inst" or len(transition.get("targets", [])) < 2:
        return False
    probs = list(transition.get("probs", []))
    probs.append(1.0 - sum(probs))
    return sum(1 for p in probs if p > 0.0) > 1


def _targets(model) -> dict[str, list[str]]:
    """The destination states of every transition, by qualified name."""
    return {
        f"{component['name']}.{automaton['name']}.{transition['name']}": transition.get("targets", [])
        for component in _body(model)["components"]
        for automaton in component.get("automata", [])
        for transition in automaton.get("transitions", [])
    }


def _stochastic_names(model) -> set[str]:
    """Qualified names of the transitions a random number decides: a
    random date, or a random branch."""
    return {
        f"{component['name']}.{automaton['name']}.{transition['name']}"
        for component in _body(model)["components"]
        for automaton in component.get("automata", [])
        for transition in automaton.get("transitions", [])
        if transition.get("distrib", "delay") not in _DATED_LAWS or _is_choice(transition)
    }


def _replay_covers(model) -> bool:
    """Whether deferred mode can replay every random draw of `model`: it
    holds random dates and random branches of instantaneous transitions,
    not a timed law with several branches."""
    return not any(
        len(transition.get("targets", [])) > 1 and transition.get("distrib") != "inst"
        for component in _body(model)["components"]
        for automaton in component.get("automata", [])
        for transition in automaton.get("transitions", [])
    )


def _armed(session, rename: dict[str, str] | None, date: float | None = None) -> dict[str, tuple]:
    """The armed random transitions of `session`, by library name; with
    `date`, each running age is read at that date, so two sessions whose
    clocks stand at different instants of the same span compare alike."""
    armed = {}
    shift = 0.0 if date is None else date - session.time
    for entry in session.deferred():
        name = entry["transition"]
        if rename is not None:
            if name not in rename:
                raise _Divergence(f"armed transition `{name}` has no library counterpart")
            name = rename[name]
        armed[name] = (
            json.dumps(entry["law"], sort_keys=True),
            entry["armed_at"],
            entry["age"] + (0.0 if entry["paused"] else shift),
            entry["paused"],
            entry["hazard"],
        )
    return armed


#: Positions of the times (armed date, age) in an :func:`_armed` entry.
_ARMED_TIMES = (1, 2)


def _same_value(a, b, *, time: bool) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        # Deterministic float arithmetic done in another order (a running
        # age read at a shifted date): equal up to the contract's 1e-12,
        # relative. A time near zero also gets 1e-12 absolute; a cumulative
        # hazard has no scale and gets none.
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12 if time else 0.0)
    return a == b


#: Float-reordering tolerance of the comparison: both routes run the same
#: engine and integrator, so a value or a date may differ only by the order
#: in which deterministic arithmetic was done (the contract's 1e-12), never
#: by the integrator tolerance meant for two different engines.
_TOLERANCE = 1e-12


def _within_tolerance(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)


def _tolerated(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    """The keys of `left` whose value differs from `right`'s within float
    reordering only (1e-12 relative, or absolute near zero). Raises
    `ValueError` naming the first key that differs beyond it."""
    tolerated = set()
    for key, value in left.items():
        other = right.get(key)
        if value == other:
            continue
        if (
            isinstance(value, float)
            and isinstance(other, float)
            and _within_tolerance(value, other)
        ):
            tolerated.add(key)
            continue
        raise ValueError(f"`{key}`: declaration {value!r}, library {other!r}")
    return tolerated


def _same_entry(a: tuple | None, b: tuple | None) -> bool:
    if a is None or b is None:
        return a is b
    return len(a) == len(b) and all(
        _same_value(x, y, time=i in _ARMED_TIMES) for i, (x, y) in enumerate(zip(a, b))
    )


def _same_armed(left: dict[str, tuple], right: dict[str, tuple]) -> bool:
    """Whether two routes arm the same transitions with the same laws,
    dates and ages, floats compared within the float tolerance."""
    return left.keys() == right.keys() and all(_same_entry(left[n], right[n]) for n in left)


#: Position of the running age in an :func:`_armed` entry: it is read at a
#: shifted date on both routes, so its reordering is the reading's, not a
#: difference between the routes.
_ARMED_AGE = 2


def _armed_tolerated(left: dict[str, tuple], right: dict[str, tuple]) -> set[str]:
    """`armed <name>` for every transition both routes arm alike only within
    the float tolerance on its armed date or hazard (entries already found
    the same by :func:`_same_entry`)."""
    return {
        f"armed {name}"
        for name, entry in left.items()
        if (other := right.get(name)) is not None
        and any(i != _ARMED_AGE and x != y for i, (x, y) in enumerate(zip(entry, other)))
    }


def _armed_by_stepping(session, name: str, date: float, limit: int = 10_000) -> bool:
    """Fire the events due by `date`, one at a time, until `name` is armed.
    False when it is still not armed once none is left by then."""
    for _ in range(limit):
        if any(d["transition"] == name for d in session.deferred()):
            return True
        dates = [f.date for f in session.fireable() if f.date is not None and f.date == f.date]
        if not dates or min(dates) > date:
            return False
        if session.step() is None:
            # Stopped before a held choice, or nothing left by then.
            return any(d["transition"] == name for d in session.deferred())
    raise RuntimeError(f"`{name}` still unarmed after {limit} same-date events at {date}")


def _follow(session, wanted, date: float) -> None:
    """Fire, at `date` and in the order of `wanted`, the transitions of
    `wanted` that `session` has not fired yet at that date, while each is
    armed and due; stop at the first that is not."""
    remaining = list(wanted)
    for e in session.history():
        if e.time == date and e.transition in remaining:
            remaining.remove(e.transition)
    for name in remaining:
        due = any(
            f.transition == name and f.date is not None and f.date <= date
            for f in session.fireable()
        )
        if not due:
            return
        session.fire(name)


def _net_changes(history, names: dict[str, str]) -> list[tuple[float, str]]:
    """`(date, last transition)` of every automaton whose state differs at
    the end of a date from its state at its start, transitions read through
    `names`."""
    spans: dict[tuple[float, str], list] = {}
    for e in history:
        if e.transition in names:
            automaton = names[e.transition].rsplit(".", 1)[0]
            spans.setdefault((e.time, automaton), []).append(e)
    return sorted(
        (date, names[events[-1].transition])
        for (date, _), events in spans.items()
        if events[0].from_state != events[-1].to_state
    )


def _drawn_trajectory(declared_model, seed: int, t_max: float) -> list[tuple[float, str, str]]:
    """One trajectory drawn on the declaration route: (date, transition,
    state entered)."""
    from .. import interactive

    drawn = interactive(declared_model, t_max=t_max, seed=seed)
    drawn.advance_to(t_max)
    return [(event.time, event.transition, event.to_state) for event in drawn.history()]


#: Where, besides the midpoint, both routes are observed between two event
#: dates, as fractions of the interval: (3 - sqrt 5) / 2 and 1 / sqrt 2,
#: neither in a rational ratio with the midpoint.
_SIDE_FRACTIONS = ((3.0 - 5.0**0.5) / 2.0, 2.0**-0.5)


def _evolves(model) -> bool:
    """Whether a value of `model` can change between two events: it has an
    ODE or reads the time. Otherwise every value holds between events, and
    the midpoint observes all there is."""
    def reads_time(node: Any) -> bool:
        if isinstance(node, dict):
            return node.get("op") == "time" or any(reads_time(v) for v in node.values())
        if isinstance(node, list):
            return any(reads_time(v) for v in node)
        return False

    components = _body(model)["components"]
    return any(
        equation.get("kind") == "ode" for component in components for equation in component.get("equations", [])
    ) or reads_time(components)


def _replay(
    declared_model,
    library_model,
    rename: dict[str, str],
    stochastic: set[str],
    plan: list[tuple[float, str, str | None]],
    t_max: float,
    attributes: list[str],
    states: list[str],
    *,
    recorded: bool,
    prefer: frozenset[str] = frozenset(),
    tolerated: set[str] | None = None,
) -> set[str]:
    """Replay `plan` on both routes in deferred mode and return the
    declaration transitions fired.

    A `recorded` plan is a drawn trajectory, every event in it, of which
    the random ones are fired at their dates, a random branch taking the
    state the trajectory entered. Otherwise the plan lists only random
    transitions to force: each is fired at its date once both routes have
    advanced there (a branch taking its first destination), and skipped
    when the declaration route does not have it armed then. A forced entry
    naming no transition fires one of the random transitions armed then on
    the declaration route: one of `prefer` if any, else the least fired so
    far, so that successive entries explore the model."""
    from .. import interactive

    sessions = [
        interactive(declared_model, t_max=t_max, deferred_draws=True),
        interactive(library_model, t_max=t_max, deferred_draws=True),
    ]
    side_fractions = _SIDE_FRACTIONS if _evolves(declared_model) or _evolves(library_model) else ()

    def compare(when: str) -> None:
        left = _snapshot(sessions[0], attributes, states)
        right = _snapshot(sessions[1], attributes, states)
        if left != right:
            try:
                near = _tolerated(left, right)
            except ValueError as difference:
                raise _Divergence(f"{when}: {difference}") from None
            if tolerated is not None:
                tolerated.update(near)
        armed = [_armed(sessions[0], rename), _armed(sessions[1], None)]
        if not _same_armed(armed[0], armed[1]):
            names = sorted(set(armed[0]) | set(armed[1]))
            differing = next(n for n in names if not _same_entry(armed[0].get(n), armed[1].get(n)))
            raise _Divergence(
                f"{when}: armed transition `{differing}` differs: declaration "
                f"{armed[0].get(differing)}, library {armed[1].get(differing)}"
            )
        if tolerated is not None:
            tolerated.update(_armed_tolerated(armed[0], armed[1]))

    def held(session) -> str | None:
        """The held choice `session` stopped before, if any."""
        due = {f.transition for f in session.fireable() if f.date is not None and f.date <= session.time}
        return next((d["transition"] for d in session.deferred() if d["transition"] in due), None)

    def advance(target: float) -> None:
        """Advance both routes to `target`. An advance stops before a held
        choice, at the place a free run would draw it: a recorded plan draws
        them itself, so a stop is a draw one route makes and the other does
        not; a forced plan draws each, first branch, on both routes."""
        while True:
            for session in sessions:
                session.advance_to(target)
            stopped = [held(session) for session in sessions]
            if stopped == [None, None]:
                return
            if recorded or stopped[0] is None or rename.get(stopped[0]) != stopped[1]:
                where = "declaration" if stopped[0] is not None else "library"
                raise _Divergence(
                    f"at t={sessions[0].time if stopped[0] else sessions[1].time}: the {where} "
                    f"route holds a draw the other does not make at that point: "
                    f"declaration {stopped[0]}, library {stopped[1]}"
                )
            sessions[0].fire_deferred_at(stopped[0], sessions[0].time, 0)
            sessions[1].fire_deferred_at(stopped[1], sessions[1].time, 0)

    def observe(start: float, end: float) -> None:
        """Compare both routes strictly between two event dates, where every
        route has settled: at the midpoint, on the replayed trajectory, and
        at side instants, each on a branch that is then undone. A stop
        restarts the integrator, so stopping the replay at the side instants
        would change what is compared after them; a branch leaves it as it
        was. The side instants are incommensurate with the midpoint, so a
        continuous input differing between the routes (a profile's period
        or phase) cannot agree at all of them by aliasing."""
        saved = [session.snapshot() for session in sessions] if side_fractions else []
        for fraction in side_fractions:
            instant = start + fraction * (end - start)
            advance(instant)
            compare(f"at t={instant}")
            for session, snapshot in zip(sessions, saved):
                session.restore(snapshot)
        middle = (start + end) / 2.0
        advance(middle)
        compare(f"at t={middle}")

    # Observe between two distinct event dates; fire each draw once the
    # events before it have fired, then let the same-date consequences
    # (zero delays) fire before reading.
    previous = 0.0
    compare("at t=0")
    branches = [_targets(declared_model), _targets(library_model)]
    recorded_here: list[str] = []
    explored: collections.Counter = collections.Counter()
    for date, transition, entered in plan:
        if date > previous:
            observe(previous, date)
            previous = date
            recorded_here = []
        if transition is None:
            advance(date)
            candidates = sorted(
                d["transition"]
                for d in sessions[0].deferred()
                if d["transition"] in stochastic and d["transition"] in rename
            )
            if not candidates:
                continue
            transition = min(candidates, key=lambda c: (c not in prefer, explored[c], c))
            explored[transition] += 1
        if transition not in stochastic:
            recorded_here.append(transition)
            continue
        if recorded:
            # Reproduce, at this date, what the drawn trajectory fired before
            # this draw, in its order.
            _follow(sessions[0], recorded_here, date)
        recorded_here.append(transition)
        mapped = rename.get(transition)
        # Armed by a same-date consequence (a zero delay) of an earlier draw:
        # step each route through the events due at this date until the draw
        # is armed. Stepping, not advancing over the whole date: a later
        # consequence in the same instant (a zero-delay repair) may disarm it
        # again, as the drawn trajectory resolved it in between.
        armed = any(d["transition"] == transition for d in sessions[0].deferred())
        if not armed and not recorded:
            armed = _armed_by_stepping(sessions[0], transition, date)
        if not recorded and not armed:
            continue
        if mapped is None:
            raise _Divergence(f"fired transition `{transition}` has no library counterpart")
        # The library route resolves the date on its own first, as a free
        # run does: its effects need no mirror hop, so a same-date consequence
        # (a zero-delay repair withdrawing a solicitation) may come before the
        # draw there and after it on the declaration route. Imposing the
        # declaration's order would hide exactly that difference.
        sessions[1].advance_to(date)
        # The law each route is about to resolve: a branch choice is armed
        # only within its instant, so the observations between dates
        # never see it.
        left = _armed(sessions[0], rename, date).get(mapped)
        right = _armed(sessions[1], None, date).get(mapped)
        if not _same_entry(left, right) and (left is None or right is None):
            law = (left or right)[0]
            if _within_the_instant(law, date):
                raise _Unordered(
                    f"at t={date}: `{mapped}` draws within its instant ({law}): "
                    "its place among the same-date events is not replayable"
                )
        if not _same_entry(left, right):
            raise _Divergence(
                f"at t={date}: armed transition `{mapped}` differs: declaration {left}, library {right}"
            )
        if tolerated is not None and left is not None:
            tolerated.update(_armed_tolerated({mapped: left}, {mapped: right}))
        targets = branches[0].get(transition, [])
        branch = None
        if len(targets) > 1:
            branch = targets.index(entered) if entered in targets else 0
        try:
            sessions[0].fire_deferred_at(transition, date, branch)
        except Exception as error:
            raise RuntimeError(
                f"the declaration route cannot replay its own draw `{transition}` at {date}: {error}"
            ) from error
        try:
            sessions[1].fire_deferred_at(mapped, date, branch)
        except Exception as error:
            raise _Divergence(f"the library route cannot fire `{mapped}` at {date}: {error}") from error
    if t_max > previous:
        observe(previous, t_max)
        advance(t_max)

    # Every fired transition is accounted for, renamed onto the other route.
    # Otherwise the comparison below, which reads the renamed transitions
    # only, would not see it.
    renamed = set(rename.values())
    unaccounted = [
        ("declaration route", sorted({e.transition for e in sessions[0].history()} - set(rename))),
        ("library route", sorted({e.transition for e in sessions[1].history()} - renamed)),
    ]
    for route, names in unaccounted:
        if names:
            listed = ", ".join(f"`{n}`" for n in names[:5])
            raise _Divergence(f"the {route} fired transitions with no counterpart: {listed}")

    # The contract is the converged states and the event dates, not the
    # propagation order inside a date: each route's history is reduced to
    # the net change of each automaton at each date, an automaton that left
    # a state and came back within one date counting for nothing.
    histories = [
        _net_changes(sessions[0].history(), rename),
        _net_changes(sessions[1].history(), {name: name for name in renamed}),
    ]
    if histories[0] != histories[1]:
        same_length = len(histories[0]) == len(histories[1])
        for a, b in zip(histories[0] + [None], histories[1] + [None]):
            if a == b:
                continue
            if (
                same_length
                and a is not None
                and b is not None
                and a[1] == b[1]
                and _within_tolerance(a[0], b[0])
            ):
                if tolerated is not None:
                    tolerated.add(f"date of {a[1]}")
                continue
            raise _Divergence(f"event histories part at {(a, b)}")
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


def _can_fire(transition: dict) -> bool:
    """False for a transition no trajectory can fire, which records no
    event: a constant false guard, or an exponential law of rate zero."""
    guard = transition.get("guard")
    if guard == {"op": "const", "value": {"kind": "bool", "value": False}}:
        return False
    return not (
        transition.get("distrib") == "exp"
        and float(transition.get("rate", 0.0) or 0.0) == 0.0
        and transition.get("rate_expr") is None
    )


def _sequence_events(model, *, every: bool = False) -> dict[str, tuple[str | None, tuple[str, str | None]]]:
    """The monitored transitions of `model` (every transition that can fire
    with `every`), `component.automaton.transition`, each with its declared
    kind and its (component, cycle group) class."""
    events = {}
    for component in _body(model)["components"]:
        for automaton in component.get("automata", []):
            for transition in automaton["transitions"]:
                if (every or transition.get("monitored")) and _can_fire(transition):
                    name = f"{component['name']}.{automaton['name']}.{transition['name']}"
                    events[name] = (
                        transition.get("kind"),
                        (component["name"], transition.get("cycle_group")),
                    )
    return events


def _sequence_difference(
    declared_model, library_model, rename: dict[str, str], inert: frozenset[str] = frozenset()
) -> str | None:
    """Why both routes would not record the same sequence events, or None.

    Every monitored transition of the declaration, renamed, must be a
    monitored transition of the library with the same kind, and the cycle
    groups must split them into the same classes (their labels may differ,
    as transition names do). The library may record a transition the
    declaration route carries unmonitored (a mode of a component's own
    section): PyCATSHOO records what the study's `monitorTransition`
    patterns select, so that flag is a convention of the declaration route,
    not of the reference; a library event with no declared counterpart at
    all is a difference."""
    declared = {
        rename.get(name, name): event
        for name, event in _sequence_events(declared_model).items()
        if name not in inert
    }
    carried = {rename.get(name, name) for name in _sequence_events(declared_model, every=True)}
    library = _sequence_events(library_model)
    only_declared = sorted(set(declared) - set(library))
    only_library = sorted(set(library) - carried)
    if only_declared or only_library:
        return (
            "sequence events differ: recorded by the declaration route only "
            f"{only_declared[:5]}, by the library route only {only_library[:5]}"
        )
    kinds = sorted(name for name in declared if declared[name][0] != library[name][0])
    if kinds:
        name = kinds[0]
        return f"sequence events differ: `{name}` is a {declared[name][0]} on the declaration route, a {library[name][0]} on the library route"

    def classes(events) -> set[frozenset[str]]:
        grouped: dict[tuple, set[str]] = {}
        for name, (_, key) in events.items():
            if name in declared and key[1] is not None:
                grouped.setdefault(key, set()).add(name)
        return {frozenset(names) for names in grouped.values()}

    split = classes(declared) ^ classes(library)
    if split:
        return f"sequence events differ: cycle groups split {sorted(sorted(c) for c in split)[:4]} differently"
    return None


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
    return at two thirds; a repair at two thirds, after its occurrence at a
    third), so each mode's effects are observed at least once. One still unfired (not armed at that date) is listed in
    ``unexercised`` and the level is ``replay-partial``.
    """
    declared_model = declaration_model(declaration)
    library_model = conversion.load()
    attributes, states = _observed(library_model, declared_model)
    difference = _sequence_difference(
        declared_model, library_model, conversion.rename, frozenset(conversion.inert)
    )
    if difference is not None:
        return RouteComparison(False, "structure", difference, 0)
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
    tolerated: set[str] = set()
    replay = functools.partial(
        _replay,
        declared_model,
        library_model,
        conversion.rename,
        declared_draws,
        t_max=t_max,
        attributes=attributes,
        states=states,
        tolerated=tolerated,
    )
    fired: set[str] = set()
    for seed in seeds:
        try:
            fired |= replay(_drawn_trajectory(declared_model, seed, t_max), recorded=True)
        except _Divergence as divergence:
            return RouteComparison(False, level, f"seed {seed}: {divergence}", len(seeds))
        except _Unordered as unordered:
            return RouteComparison(False, "replay-partial", f"seed {seed}: {unordered}", len(seeds))
        except RuntimeError as unfaithful:
            # The declaration route could not replay its own draw: the
            # replay proves nothing for this trajectory, which is no proof.
            return RouteComparison(
                False, "replay-partial", f"seed {seed}: replay not faithful: {unfaithful}", len(seeds)
            )
    by_library_name = {
        name: declared
        for declared, name in conversion.rename.items()
        if declared in declared_draws
    }
    for transition in sorted(declared_draws & set(conversion.rename) - fired):
        if transition in fired:
            continue
        target = conversion.rename[transition]
        plan = [(t_max / 3.0, transition, None)]
        if target.endswith(f".{_OCC}"):
            back = by_library_name.get(f"{target[: -len(_OCC)]}{_REP}")
            if back is not None:
                plan.append((2.0 * t_max / 3.0, back, None))
        elif target.endswith(f".{_REP}"):
            # A repair is armed only once its mode has occurred, which this
            # fresh trajectory has not: force the occurrence first.
            first = by_library_name.get(f"{target[: -len(_REP)]}{_OCC}")
            if first is not None:
                plan = [(t_max / 3.0, first, None), (2.0 * t_max / 3.0, transition, None)]
        try:
            fired |= replay(plan, recorded=False)
        except _Divergence as divergence:
            return RouteComparison(
                False, level, f"forcing `{transition}`: {divergence}", len(seeds)
            )
        except _Unordered as unordered:
            return RouteComparison(
                False, "replay-partial", f"forcing `{transition}`: {unordered}", len(seeds)
            )
        except RuntimeError as unfaithful:
            return RouteComparison(
                False, "replay-partial", f"forcing `{transition}`: replay not faithful: {unfaithful}", len(seeds)
            )
    remaining = declared_draws & set(conversion.rename) - fired
    if remaining:
        # Some draws need others first (a condition only a rare draw makes
        # true): explore, firing one armed draw at each of evenly spaced
        # dates, the ones still unfired first.
        plan = [(t_max * k / 41.0, None, None) for k in range(1, 41)]
        try:
            fired |= replay(plan, recorded=False, prefer=frozenset(remaining))
        except _Divergence as divergence:
            return RouteComparison(False, level, f"exploring: {divergence}", len(seeds))
        except _Unordered as unordered:
            return RouteComparison(False, "replay-partial", f"exploring: {unordered}", len(seeds))
        except RuntimeError as unfaithful:
            return RouteComparison(
                False, "replay-partial", f"exploring: replay not faithful: {unfaithful}", len(seeds)
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
    if tolerated:
        # Agreement that needed the float-reordering tolerance is reported
        # under its own level, never as exact.
        named = ", ".join(f"`{name}`" for name in sorted(tolerated)[:5])
        return RouteComparison(
            True,
            "tolerance",
            f"{observed} agree on every trajectory ({level}), within {_TOLERANCE:g} on {named}",
            len(seeds),
        )
    return RouteComparison(True, level, f"{observed} agree on every trajectory", len(seeds))
