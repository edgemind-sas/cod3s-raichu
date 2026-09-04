"""Building a `pyraichu.muscadet` component from a declaration held in DATA.

A muscadet component is normally a subclass of :class:`pyraichu.muscadet.ObjFlow`
overriding ``add_flows``. That subclass is almost never *behaviour*: it declares
flows, capacities, measurement channels, rule sets, transfer pairs and failure
modes, and every one of those is a declaration a mapping can carry. What the
subclass really provides is a **place to write the declaration** and, less
visibly, the right ORDER to write it in.

This module is that place, for a caller whose declaration arrives as data: a
COD3S Platform export, a knowledge base held in YAML, a generated study. The
vocabulary it reads is **muscadet's own declaration spec**, section for section
and key for key, rather than a parallel one invented here. A declaration written
for muscadet is the declaration this module reads.

It owns two entry points:

- :func:`check_spec` validates a declaration and builds nothing;
- :func:`build_component` turns it into a live component.

**The order is the whole difficulty, and it is not guessable.**
:data:`DECLARATION_SECTIONS` writes it down once. It is neither alphabetical nor
arbitrary: a capacity, a failure mode and a rule all name flows, so the flows
come first; a rule guard may gate on the automaton a failure mode declares, so
the modes come before the rules; a transfer pair refuses a flow a rule already
carries, so the pairs come last. Declared in any other order, each of those
refusals is unreachable and the misspelling it exists to catch reaches the
engine instead.

That order is where this module departs from muscadet's own, deliberately:
muscadet declares its failure modes *after* ``set_flows()`` because their
effects clamp variables that do not exist until then, and this layer has no such
split. Here a failure mode is a declaration like any other, and a rule guard
resolves the automaton it names at declaration time, so the modes must precede
the rules.

**A class stays usable as a template.** ``cls`` names a registered component
class and ``params`` is its own declaration; the class declares its ports first
and the spec's sections are added on top.

Examples
--------
>>> from pyraichu.declare import build_component
>>> from pyraichu.muscadet import System
>>> system = System(name="HEATING")
>>> pump = build_component(system, {
...     "name": "PUMP",
...     "flows": [
...         {"cls": "FlowContinuousIn", "name": "elec"},
...         {"cls": "FlowContinuousOut", "name": "heat"},
...         {"cls": "FlowIn", "name": "call", "logic": "or"},
...         {"cls": "FlowOut", "name": "healthy", "var_prod_default": True},
...     ],
...     "rules": [{"name": "heat_pump", "rules": [
...         {"cond": ["call"], "cons": {"elec": 2.0}, "prod": {"heat": 7.0}},
...     ]}],
... })
>>> [flow.name for flow in pump.flows_continuous_out]
['heat']
"""

from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any

from . import muscadet as authoring

__all__ = [
    "COMPONENT_CLASSES",
    "COMPONENT_KEYS",
    "DECLARATION_SECTIONS",
    "FLOW_CLASSES",
    "PLAIN_SECTIONS",
    "ComponentSpecError",
    "build_component",
    "check_spec",
    "entry_call",
    "register_component_class",
]


class ComponentSpecError(ValueError):
    """A component declaration this layer refuses to build."""


#: The declaration sections, IN THE ORDER THEY MUST BE DECLARED. See the module
#: docstring for what each dependency is and why the order makes three refusals
#: reachable instead of dead.
DECLARATION_SECTIONS = (
    "measurements_in",
    "flows",
    "capacities",
    "failure_modes",
    "rules",
    "transfers",
)

#: Keys naming the component itself rather than one of its sections.
CONSTRUCTOR_KEYS = (
    "name",
    "cls",
    "params",
    "label",
    "description",
    "metadata",
)

#: Written by muscadet's read-back and read by nobody: the class a spec was read
#: back FROM. A spec is always expanded onto a bare component, so the original
#: class name would otherwise be lost, and a template picker wants to show it.
SOURCE_CLS_KEY = "source_cls"

#: Sections muscadet declares and this layer cannot carry, with the reason. They
#: are accepted while EMPTY, because muscadet's read-back writes every section
#: whether or not the component has one, and refused as soon as they declare
#: something.
UNCARRIED_SECTIONS = {
    "measurements_out": (
        "a measurement reading the component PUBLISHES. Here a capacity "
        "publishes its own level and `System.connect_measurement` wires it, so "
        "there is no republisher to declare: an instrument standing between a "
        "volume and a voter has no counterpart yet"
    ),
    "automata": (
        "a two-state automaton declared on the component. This layer DERIVES "
        "every automaton from the declaration that needs one (a failure mode, "
        "a capacity bound, a temporised or triggered output, a guarded rule "
        "set) and declares none of its own"
    ),
}

#: Component-level keys carrying no counterpart, refused as soon as they ask for
#: something. Same treatment as the sections above: a value that says nothing is
#: accepted, so a read-back is not refused for its decoration.
UNCARRIED_KEYS = {
    "create_default_out_automata": (
        "the ok/nok automaton pair muscadet derives on every discrete output. "
        "This layer derives no such pair, so an indicator naming one would be "
        "silently absent"
    ),
}

#: Every key a declaration may carry.
COMPONENT_KEYS = frozenset(
    CONSTRUCTOR_KEYS
    + (SOURCE_CLS_KEY,)
    + DECLARATION_SECTIONS
    + tuple(UNCARRIED_SECTIONS)
    + tuple(UNCARRIED_KEYS)
)

#: The component classes a declaration's ``cls`` may name. Seeded with the bare
#: component and extended through :func:`register_component_class`, so a
#: knowledge base of shipped classes resolves by name without this module
#: knowing about it. Resolution is EXPLICIT on purpose: scanning the subclass
#: tree would make two classes of the same name silently interchangeable, and
#: which of the two a declaration got would depend on import order.
COMPONENT_CLASSES: dict[str, type] = {"ObjFlow": authoring.ObjFlow}


def register_component_class(cls: type, name: str | None = None) -> type:
    """Register `cls` under `name` (its own name by default) so a
    declaration's ``cls`` can reach it.

    Returns the class, so it reads as a decorator.
    """
    if not (isinstance(cls, type) and issubclass(cls, authoring.ObjFlow)):
        raise ComponentSpecError(
            f"{cls!r} is not a component class: a declaration builds a "
            f"subclass of `{authoring.ObjFlow.__name__}`"
        )
    COMPONENT_CLASSES[name or cls.__name__] = cls
    return cls


# --- the vocabulary of one declaration entry --------------------------------


@dataclass(frozen=True)
class _Vocabulary:
    """What one section entry may carry, and what happens to the rest.

    Three classes of key, and the difference between the last two is the
    whole discipline of this module:

    - ``carried`` maps a muscadet declaration key onto the authoring
      keyword that honours it;
    - ``inert`` names a key this layer does not read but that says nothing
      at the value listed, which is muscadet's own default. It is accepted
      at that value and REFUSED at any other, so a read-back writing every
      field is not refused for its defaults while a declaration that
      actually asks for something still is;
    - ``uncarried`` names a key refused outright, with the reason.
    """

    carried: dict[str, str]
    inert: dict[str, Any] = field(default_factory=dict)
    uncarried: dict[str, str] = field(default_factory=dict)

    def accepted(self) -> list[str]:
        return sorted(set(self.carried) | set(self.inert))


#: The two Python extension points muscadet declares and this layer refuses by
#: name wherever they appear. Approximating either by the nearest declarable
#: shape would answer a different model without saying so.
_FUNCTION_KEYS = {
    "allocation_fun": (
        "an allocation split written as a Python function. Declare "
        "`allocation` as 'proportional', 'shares' or 'priority' instead"
    ),
    "combine_fun": (
        "a reading combination written as a Python function. muscadet "
        "declares no data form for it, so it cannot be carried"
    ),
}

#: What every flow declaration carries beside its own family's keys: fields
#: muscadet writes on every flow and that declare nothing here.
_FLOW_SHARED = {
    "component_authorized": None,
    "combine": None,
}

#: The refusal every flow vocabulary gives `combine` once it carries more
#: than its inert default, reused across all six rather than falling back
#: on the generic "accepted only at its default" message.
_FLOW_UNCARRIED = dict(
    _FUNCTION_KEYS,
    combine=(
        "how several producers of one flow combine into what it reads. A "
        "continuous flow is the sum of its connections and a discrete one "
        "votes through its declared `logic`; a combination policy belongs "
        "to a measurement channel instead. Declare one with "
        "`add_measurement_in` and combine its readings there"
    ),
)

_DISCRETE_IN = _Vocabulary(
    carried={"name": "name", "logic": "logic"},
    inert=dict(
        _FLOW_SHARED,
        var_type="bool",
        var_fed_default=None,
        var_in_default=False,
        var_available_in_default=True,
    ),
    uncarried=_FLOW_UNCARRIED,
)

_DISCRETE_OUT_KEYS = {
    "name": "name",
    "var_prod_default": "var_prod_default",
    "var_prod_cond": "var_prod_cond",
}

_DISCRETE_OUT_INERT = dict(
    _FLOW_SHARED,
    var_type="bool",
    var_fed_default=None,
    var_is_active_default=True,
    var_fed_available_out_init=True,
    var_fed_available_out_reset=True,
    var_prod_cond_inner_mode="or",
    var_prod_cond_negate=[],
    var_prod_cond_compare=[],
    negate=False,
)

_DISCRETE_OUT = _Vocabulary(
    carried=_DISCRETE_OUT_KEYS,
    inert=_DISCRETE_OUT_INERT,
    uncarried=_FLOW_UNCARRIED,
)

_DISCRETE_OUT_TEMPO = _Vocabulary(
    carried=dict(
        _DISCRETE_OUT_KEYS,
        occ_enable_flow="enable_time",
        occ_disable_flow="disable_time",
        init_enable="init_enable",
    ),
    inert=dict(
        _DISCRETE_OUT_INERT,
        state_enable_name="enabled",
        state_disable_name="disabled",
        state_enabling_name="enabling",
        state_disabling_name="disabling",
    ),
    uncarried=_FLOW_UNCARRIED,
)

_DISCRETE_OUT_ON_TRIGGER = _Vocabulary(
    carried=dict(
        _DISCRETE_OUT_KEYS,
        trigger_time_up="trigger_time_up",
        trigger_time_down="trigger_time_down",
        trigger_logic="trigger_logic",
    ),
    inert=_DISCRETE_OUT_INERT,
    uncarried=_FLOW_UNCARRIED,
)

_CONTINUOUS_IN = _Vocabulary(
    carried={
        "name": "name",
        "var_in_default": "var_in_default",
        # muscadet spells the declared demand of a pure consumer
        # `var_demand_default` on the INPUT; this layer spells the same
        # quantity `var_demand_in_default`.
        "var_demand_default": "var_demand_in_default",
        # The counterpart of the output's profile, in the same shape.
        "profile": "profile",
        # No muscadet counterpart: materialise the rate this flow carries
        # as an observable channel, for a controller to threshold.
        "publish_rate": "publish_rate",
    },
    inert=dict(_FLOW_SHARED, var_type="float"),
    uncarried=_FLOW_UNCARRIED,
)

_CONTINUOUS_OUT = _Vocabulary(
    carried={
        "name": "name",
        "var_fed_default": "var_fed_default",
        "allocation": "allocation",
        "allocation_shares": "allocation_shares",
        "allocation_priorities": "allocation_priorities",
        "profile": "profile",
        # No muscadet counterpart: a ceiling on what an output can
        # deliver per unit time is a property of the equipment that
        # neither muscadet nor the class it mirrors carries, and the
        # industrial corpus this layer is validated against declares one
        # on its electrolyser.
        "max_rate": "max_rate",
        # No muscadet counterpart either: see the input's own entry.
        "publish_rate": "publish_rate",
    },
    inert=dict(_FLOW_SHARED, var_type="float", var_demand_in_default=0.0),
    uncarried=dict(
        _FLOW_UNCARRIED,
        var_demand_in_default=(
            "the aggregated demand an output reads when no consumer is "
            "connected. This layer derives that read from the connections "
            "themselves, so the declared value would be dropped"
        ),
    ),
)

#: The flow classes a ``flows`` entry may name, canonical and 1.x spellings
#: alike, each with the authoring method it maps to and the keys it reads.
FLOW_CLASSES: dict[str, tuple[str, _Vocabulary]] = {
    "FlowDiscreteIn": ("add_flow_in", _DISCRETE_IN),
    "FlowIn": ("add_flow_in", _DISCRETE_IN),
    "FlowDiscreteOut": ("add_flow_out", _DISCRETE_OUT),
    "FlowOut": ("add_flow_out", _DISCRETE_OUT),
    "FlowDiscreteOutTempo": ("add_flow_out_tempo", _DISCRETE_OUT_TEMPO),
    "FlowOutTempo": ("add_flow_out_tempo", _DISCRETE_OUT_TEMPO),
    "FlowDiscreteOutOnTrigger": (
        "add_flow_out_on_trigger",
        _DISCRETE_OUT_ON_TRIGGER,
    ),
    "FlowOutOnTrigger": ("add_flow_out_on_trigger", _DISCRETE_OUT_ON_TRIGGER),
    "FlowContinuousIn": ("add_flow_continuous_in", _CONTINUOUS_IN),
    "FlowContinuousOut": ("add_flow_continuous_out", _CONTINUOUS_OUT),
}

#: The flow classes whose entries declare an INPUT, so a production condition
#: naming one resolves on the input side.
_INPUT_CLASSES = frozenset(
    {"FlowDiscreteIn", "FlowIn", "FlowContinuousIn"},
)

_MEASUREMENT_IN = _Vocabulary(
    carried={"name": "name", "flows": "flows"},
    inert={"level_default": 0.0, "fill_default": 0.0, "combine": None},
    uncarried=dict(
        _FUNCTION_KEYS,
        combine=(
            "how several readings of one channel reduce to one. A channel here "
            "observes a single publisher, and a declared combination that was "
            "dropped would read as a vote while the channel took a sum"
        ),
    ),
)

_CAPACITY = _Vocabulary(
    carried={
        "name": "name",
        "flow": "flow",
        "flows": "flows",
        "capacity": "capacity",
        "side": "side",
        "content_init": "content_init",
        "fill_rate": "fill_rate",
        # No muscadet counterpart: the fraction of the volume the content
        # must move back from a bound before the capacity leaves it, which
        # this engine locates rather than steps over.
        "hysteresis": "hysteresis",
    },
)

_RULE_SET = _Vocabulary(
    carried={
        "name": "name",
        "rules": "rules",
        # No muscadet counterpart: how two rule sets producing into one
        # contested output share its demand. muscadet leaves it open; here
        # its absence is refused rather than defaulted.
        "apportionment": "apportionment",
    },
)

_TRANSFER = _Vocabulary(
    carried={"name": "name", "flows": "flows", "equation": "equation"},
)

_FAILURE_MODE_SHARED = {
    "name": "name",
    "failure_cond": "failure_cond",
    "failure_effects": "failure_effects",
    "repair_effects": "repair_effects",
    # No muscadet counterpart: the discrete out-flows the mode gates.
    # muscadet reaches them through `failure_effects` patterns, which here
    # resolve against the continuous outputs only.
    "targets": "targets",
}

_FAILURE_MODE_INERT = {
    "failure_state": "occ",
    "repair_state": "rep",
    "repair_cond": True,
}

#: The two shapes a ``failure_modes`` entry may take. A standalone failure-mode
#: object is a component in its own right and is declared as one.
FAILURE_MODE_CLASSES: dict[str, tuple[str, _Vocabulary]] = {
    "delay": (
        "add_delay_failure_mode",
        _Vocabulary(
            carried=dict(
                _FAILURE_MODE_SHARED,
                failure_time="failure_time",
                repair_time="repair_time",
            ),
            inert=dict(
                _FAILURE_MODE_INERT, failure_param_name="ttf", repair_param_name="ttr"
            ),
        ),
    ),
    "exp": (
        "add_exp_failure_mode",
        _Vocabulary(
            carried=dict(
                _FAILURE_MODE_SHARED,
                failure_rate="failure_rate",
                repair_rate="repair_rate",
            ),
            inert=dict(
                _FAILURE_MODE_INERT,
                failure_param_name="lambda",
                repair_param_name="mu",
            ),
        ),
    ),
}

#: Sections whose entries map onto one authoring method with one vocabulary.
#: ``flows`` and ``failure_modes`` are absent: each dispatches on its entry's
#: ``cls`` and is expanded apart.
PLAIN_SECTIONS: dict[str, tuple[str, _Vocabulary]] = {
    "measurements_in": ("add_measurement_in", _MEASUREMENT_IN),
    "capacities": ("add_capacity", _CAPACITY),
    "rules": ("add_rule_set", _RULE_SET),
    "transfers": ("add_transfer", _TRANSFER),
}

#: The one transfer law and the one time profile a mapping can carry. Every
#: other shape of either family is a Python function.
_TRANSFER_CLASS = authoring._TRANSFER_CLASS
_PROFILE_CLASS = authoring._PROFILE_CLASS

#: Beyond this many conjunctions, a production condition converted out of
#: muscadet's conjunctive form is refused rather than expanded: the expansion is
#: exact but its size is the product of the clause widths, and an expression
#: that large is a modelling accident rather than a declaration.
_MAX_PROD_COND_GROUPS = 256


# --- refusals reachable from the mapping alone ------------------------------


def _refuse_callables(where: str, value: Any) -> None:
    """Refuse a Python callable anywhere in a declaration, naming its field.

    A callable builds a component and does not survive being written out, so a
    declaration holding one is a model that cannot be reproduced from its own
    data. muscadet's read-back refuses the same shapes; this refuses them one
    step earlier, before anything is built.
    """
    if callable(value):
        raise ComponentSpecError(
            f"{where} holds a Python {type(value).__name__}, which no mapping "
            f"can carry. Declare the equivalent shape instead: a named "
            f"allocation policy, a `{_TRANSFER_CLASS}` equation, a "
            f"`{_PROFILE_CLASS}` time profile, or keep this component a "
            f"subclass"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _refuse_callables(f"{where}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _refuse_callables(f"{where}[{index}]", item)


def _says_nothing(value: Any, default: Any) -> bool:
    """True when an inert key carries the value that declares nothing.

    ``None`` always says nothing: muscadet writes an unset field as ``None``
    and a read-back should not be refused for it.
    """
    if value is None:
        return True
    if isinstance(default, bool) or isinstance(value, bool):
        return value is default
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        return float(value) == float(default)
    return value == default


def _keywords(where: str, entry: dict, vocabulary: _Vocabulary) -> dict[str, Any]:
    """The authoring keywords one declaration entry expands to.

    Every key is classified: carried, inert at its declared value, or refused
    by name. Nothing is dropped in silence, which is the whole point of reading
    a declaration as data rather than handing it to a constructor.

    A key is classified BEFORE its value is walked for a Python callable, so
    the two keys muscadet declares as extension points keep their own reason
    rather than falling back on the generic refusal of a function.
    """
    keywords: dict[str, Any] = {}
    for key, value in entry.items():
        # `cls` names the shape of the entry, and the dispatch above has
        # already consumed it: a section that dispatches on nothing simply
        # ignores it, as muscadet's own read-back drops it.
        if key == "cls":
            continue
        if key in vocabulary.carried:
            _refuse_callables(f"{where}.{key}", value)
            keywords[vocabulary.carried[key]] = value
            continue
        # An inert key is looked at BEFORE its refusal, so a key that is both
        # (one this layer cannot carry, whose muscadet default says nothing) is
        # accepted at that default and refused with its own reason elsewhere.
        if key in vocabulary.inert and _says_nothing(value, vocabulary.inert[key]):
            continue
        if key in vocabulary.uncarried:
            raise ComponentSpecError(
                f"{where} declares `{key}`: {vocabulary.uncarried[key]}"
            )
        if key in vocabulary.inert:
            raise ComponentSpecError(
                f"{where} declares `{key}`={value!r}, which this layer does "
                f"not read. It is accepted only at its default "
                f"{vocabulary.inert[key]!r}, where it says nothing"
            )
        raise ComponentSpecError(
            f"{where} carries unknown declaration key `{key}`; it accepts "
            f"{', '.join(vocabulary.accepted())}"
        )

    return keywords


def _entries(spec: dict, section: str, name: str) -> list[dict]:
    """One declaration section, as a list of mappings.

    A single mapping is the short form of a one-entry section, as muscadet
    reads it.
    """
    entries = spec.get(section)
    if entries is None:
        return []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        raise ComponentSpecError(
            f"Component {name}: section '{section}' is a list of declarations, "
            f"got {type(entries).__name__}"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise ComponentSpecError(
                f"Component {name}: every entry of section '{section}' is a "
                f"mapping, got {type(entry).__name__}"
            )
    return list(entries)


# --- production conditions ---------------------------------------------------


def _prod_cond_operand(where: str, operand: Any, inputs: set[str]) -> str:
    """One production-condition operand, reduced to the flow name this layer
    reads.

    muscadet's operand vocabulary carries a negation and a threshold beside the
    name; a discrete production condition here reads flow names and nothing
    else, so those two are refused rather than dropped. ``port`` is honoured
    where it agrees with this layer's own resolution, which searches the input
    side first exactly as muscadet's does, and refused where it would select
    the other side of a name carried both ways.
    """
    if isinstance(operand, str):
        return operand
    if not isinstance(operand, dict):
        raise ComponentSpecError(
            f"{where} carries the production-condition operand {operand!r}; an "
            f"operand is a flow name or a mapping carrying `name`"
        )

    for key in ("negate", "op", "value"):
        if operand.get(key) is not None and operand.get(key) is not False:
            raise ComponentSpecError(
                f"{where} declares the production-condition operand key "
                f"`{key}`, which this layer does not carry: a discrete "
                f"production condition reads flow names, and a negation or a "
                f"threshold has no counterpart here"
            )

    unknown = sorted(set(operand) - {"name", "port", "negate", "op", "value"})
    if unknown:
        raise ComponentSpecError(
            f"{where} carries unknown production-condition operand keys "
            f"{unknown}; an operand carries `name` and `port`"
        )

    name = operand.get("name")
    if not isinstance(name, str):
        raise ComponentSpecError(
            f"{where} carries a production-condition operand without a `name`: "
            f"{operand!r}"
        )

    port = operand.get("port")
    if port == "out" and name in inputs:
        raise ComponentSpecError(
            f"{where} resolves the production-condition operand `{name}` to "
            f"the output side, while this component declares `{name}` as an "
            f"input too. This layer resolves the input side first and cannot "
            f"honour the other selection; rename one of the two flows"
        )
    if port not in (None, "in", "out"):
        raise ComponentSpecError(
            f"{where} declares the production-condition operand `{name}` on "
            f"port {port!r}, expected 'in' or 'out'"
        )
    return name


def _prod_cond(where: str, declared: Any, inputs: set[str]) -> list[list[str]]:
    """muscadet's production condition, converted to this layer's form.

    **The two conventions are not the same one, and reading either as the other
    inverts the condition.** muscadet normalises a production condition into
    CONJUNCTIVE normal form: every top-level element is a clause, its own
    operands OR-ed, and the clauses are AND-ed. This layer reads a list of
    groups the other way round, as the OR of conjunctions, which is the
    platform-export form. Passed through untouched, ``[["a", "b"], ["c"]]``
    would mean ``(a or b) and c`` on one side and ``(a and b) or c`` on the
    other, and nothing would say so.

    So the clauses are expanded: one conjunction per choice of a single operand
    from each clause, which is exact and, for a real condition, small.

    The flat form agrees by accident and is worth stating, because muscadet's
    own docstring gets it backwards: ``["a", "b"]`` is TWO clauses of one
    operand, hence ``a and b``, and that is what the layer underneath reads a
    flat list as too. Only the nested form has to be converted.
    """
    if not declared:
        return []
    if isinstance(declared, (str, dict)):
        declared = [declared]
    if not isinstance(declared, (list, tuple)):
        raise ComponentSpecError(
            f"{where} carries a production condition that is neither a name, "
            f"an operand nor a list: {declared!r}"
        )

    clauses: list[list[str]] = []
    for group in declared:
        operands = group if isinstance(group, (list, tuple)) else [group]
        clauses.append(
            [_prod_cond_operand(where, operand, inputs) for operand in operands]
        )

    width = 1
    for clause in clauses:
        width *= max(len(clause), 1)
    if width > _MAX_PROD_COND_GROUPS:
        raise ComponentSpecError(
            f"{where} carries a production condition of {len(clauses)} "
            f"conjoined clauses expanding to {width} disjunctions, past the "
            f"{_MAX_PROD_COND_GROUPS} this layer converts. Split the component "
            f"or state the condition through a rule guard"
        )

    return [list(choice) for choice in itertools.product(*clauses)]


# --- occurrence laws ---------------------------------------------------------


def _delay(where: str, key: str, declared: Any) -> float:
    """A temporisation law, reduced to the delay this layer carries.

    A temporised output here waits a fixed time; a random one has no
    counterpart, so it is refused by name rather than reduced to its mean.
    """
    if declared is None:
        return 0.0
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        return float(declared)
    if not isinstance(declared, dict):
        raise ComponentSpecError(
            f"{where} declares `{key}`={declared!r}, which is no occurrence "
            f"law: declare {{'cls': 'delay', 'time': t}}"
        )
    law = declared.get("cls", "delay")
    if law == "inst":
        return 0.0
    if law != "delay":
        raise ComponentSpecError(
            f"{where} declares `{key}` under the occurrence law '{law}'; a "
            f"temporised output here waits a fixed time, so only 'delay' and "
            f"'inst' are carried"
        )
    unknown = sorted(set(declared) - {"cls", "time"})
    if unknown:
        raise ComponentSpecError(
            f"{where} declares `{key}` carrying unknown keys {unknown}; a "
            f"delay law carries `cls` and `time`"
        )
    return float(declared.get("time", 0.0))


# --- the declared shapes a mapping can carry --------------------------------


def _check_declared_shape(where: str, key: str, declared: Any, expected: str) -> None:
    """Refuse a transfer equation or a time profile whose shape has no mapping
    form, naming the one that has.

    Both families follow the same pattern: a base class carrying a Python
    function, and one declarable subclass. The base is refused HERE rather than
    at construction, so a batch of declarations reports it without an engine.
    """
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise ComponentSpecError(
            f"{where} declares {declared!r} as `{key}`, which is not one: it is "
            f"the mapping {{'cls': '{expected}', ...}}"
        )
    shape = declared.get("cls")
    if shape is None:
        raise ComponentSpecError(
            f"{where} declares `{key}` with no `cls` key naming its shape; the "
            f"one declarable shape is `{expected}`"
        )
    if shape != expected:
        raise ComponentSpecError(
            f"{where} declares the `{key}` shape `{shape}`; the one declarable "
            f"shape is `{expected}`, every other shape of that family being a "
            f"Python function that no mapping can carry"
        )


# --- expansion ---------------------------------------------------------------


@dataclass(frozen=True)
class _Call:
    """One authoring call a declaration expands to."""

    section: str
    entry: str
    method: str
    keywords: dict[str, Any]


def entry_call(kind: str, entry: dict, *, where: str) -> tuple[str, dict[str, Any]]:
    """The authoring method and keyword arguments ONE declaration entry
    expands to, ready to be called on a component.

    `kind` names either a flow class (:data:`FLOW_CLASSES`, the entry's own
    ``cls``) or a section whose entries share one vocabulary
    (:data:`PLAIN_SECTIONS`). Every key of the entry is classified: carried
    onto an authoring keyword, inert at the value that declares nothing, or
    refused by name.

    This is the **one** place a declaration entry is read. `build_component`
    reaches it through the whole-component plan below, and the serialized
    plugin path (`pyraichu.plugins.muscadet`) reaches it entry by entry for
    the sections it carries: a key one of the two accepted and the other
    refused would be a vocabulary that says two things.

    What stays out of it is what only the whole component knows: a discrete
    production condition resolves its operands against the component's input
    names, so :func:`_flow_calls` converts it after this returns.
    """
    if kind in FLOW_CLASSES:
        method, vocabulary = FLOW_CLASSES[kind]
    elif kind in PLAIN_SECTIONS:
        method, vocabulary = PLAIN_SECTIONS[kind]
    else:
        raise ComponentSpecError(
            f"{where} names the declaration shape `{kind}`, which is none of "
            f"{', '.join(sorted(set(FLOW_CLASSES) | set(PLAIN_SECTIONS)))}"
        )

    keywords = _keywords(where, entry, vocabulary)
    # Named by the key the DECLARATION carries, not by the keyword it maps
    # onto: a refusal that names a key the caller never wrote is a refusal
    # the caller cannot act on.
    for declared, keyword in (
        ("occ_enable_flow", "enable_time"),
        ("occ_disable_flow", "disable_time"),
    ):
        if keyword in keywords:
            keywords[keyword] = _delay(where, declared, keywords[keyword])
    if "profile" in keywords:
        _check_declared_shape(where, "profile", keywords["profile"], _PROFILE_CLASS)
    if "equation" in keywords:
        _check_declared_shape(where, "equation", keywords["equation"], _TRANSFER_CLASS)
    return method, keywords


def _flow_calls(spec: dict, name: str) -> list[_Call]:
    """The ``flows`` section, expanded.

    The input names are collected over the WHOLE section before any entry is
    expanded, so a production condition naming a flow declared further down
    still resolves: a declaration is a set, not a sequence.
    """
    entries = _entries(spec, "flows", name)

    inputs = {
        entry.get("name")
        for entry in entries
        if entry.get("cls") in _INPUT_CLASSES and isinstance(entry.get("name"), str)
    }

    calls = []
    for index, entry in enumerate(entries):
        where = f"Component {name}: flow {entry.get('name', index)!r}"
        shape = entry.get("cls")
        if shape is None:
            raise ComponentSpecError(
                f"{where} carries no `cls` key naming its class; a flow is one "
                f"of {', '.join(sorted(FLOW_CLASSES))}"
            )
        if shape not in FLOW_CLASSES:
            raise ComponentSpecError(
                f"{where} names the flow class `{shape}`, which is none of "
                f"{', '.join(sorted(FLOW_CLASSES))}"
            )

        method, keywords = entry_call(shape, entry, where=where)

        if "var_prod_cond" in keywords:
            keywords["var_prod_cond"] = _prod_cond(
                where, keywords["var_prod_cond"], inputs
            )

        calls.append(_Call("flows", entry.get("name", index), method, keywords))

    return calls


def _failure_mode_calls(spec: dict, name: str) -> list[_Call]:
    """The ``failure_modes`` section, expanded."""
    calls = []
    for index, entry in enumerate(_entries(spec, "failure_modes", name)):
        where = f"Component {name}: failure mode {entry.get('name', index)!r}"
        kind = entry.get("cls")
        if kind not in FAILURE_MODE_CLASSES:
            raise ComponentSpecError(
                f"{where} has cls={kind!r}; a failure mode declared on a "
                f"component is one of {', '.join(sorted(FAILURE_MODE_CLASSES))}. "
                f"A standalone failure-mode object is a component of its own "
                f"and is declared as one"
            )
        method, vocabulary = FAILURE_MODE_CLASSES[kind]
        keywords = _keywords(where, entry, vocabulary)

        # muscadet's `failure_cond` defaults to True, meaning "nothing gates
        # it"; here the gate is the name of one local variable, so the default
        # reads as no gate and anything richer is refused rather than reduced.
        cond = keywords.get("failure_cond")
        if cond is True or cond is None:
            keywords.pop("failure_cond", None)
        elif not isinstance(cond, str):
            raise ComponentSpecError(
                f"{where} declares `failure_cond`={cond!r}; here the gate is "
                f"the name of one local variable of this component, and a "
                f"condition expression has no counterpart"
            )

        for key in ("failure_effects", "repair_effects"):
            if key in keywords:
                keywords[key] = _effects(where, key, keywords[key])

        calls.append(_Call("failure_modes", entry.get("name", index), method, keywords))

    return calls


def _effects(where: str, key: str, declared: Any) -> list[tuple[str, Any]]:
    """A mode's declared effects, normalised to the ``(pattern, value)`` pairs
    the authoring layer resolves.

    JSON has no tuple, so a round-tripped declaration carries two-element lists
    and they mean the same thing.
    """
    if declared is None:
        return []
    if not isinstance(declared, (list, tuple)):
        raise ComponentSpecError(
            f"{where} declares `{key}`={declared!r}; effects are a list of "
            f"(pattern, value) pairs"
        )
    pairs = []
    for effect in declared:
        if not isinstance(effect, (list, tuple)) or len(effect) != 2:
            raise ComponentSpecError(
                f"{where} declares the `{key}` entry {effect!r}; an effect is a "
                f"(pattern, value) pair"
            )
        pairs.append((effect[0], effect[1]))
    return pairs


def _plain_calls(spec: dict, name: str, section: str) -> list[_Call]:
    """One section whose entries map onto a single authoring method."""
    calls = []
    for index, entry in enumerate(_entries(spec, section, name)):
        label = entry.get("name", index)
        where = f"Component {name}: {section} entry {label!r}"
        method, keywords = entry_call(section, entry, where=where)
        calls.append(_Call(section, label, method, keywords))
    return calls


def _resolve_class(spec: dict, name: str, classes: dict[str, type]) -> type:
    """The component class a declaration names.

    ``cls`` is a registered class NAME, the form a declaration held in data
    carries, or the class itself, which a caller holding one may pass directly.
    """
    declared = spec.get("cls", "ObjFlow")
    if isinstance(declared, type):
        if not issubclass(declared, authoring.ObjFlow):
            raise ComponentSpecError(
                f"Component {name}: `cls` names {declared.__name__}, which is "
                f"no component class"
            )
        return declared
    if not isinstance(declared, str):
        raise ComponentSpecError(
            f"Component {name}: `cls` is a component class or the name of one, "
            f"got {type(declared).__name__}"
        )
    component_class = classes.get(declared)
    if component_class is None:
        raise ComponentSpecError(
            f"Component {name}: `cls` names the component class `{declared}`, "
            f"which is not registered; the registered classes are "
            f"{', '.join(sorted(classes))}. Register it with "
            f"`pyraichu.declare.register_component_class`, or pass the class "
            f"itself"
        )
    return component_class


def _check_params(spec: dict, name: str, component_class: type) -> dict[str, Any]:
    """The named class's own declaration, refused when the class reads none.

    A class whose ``add_flows`` takes no keyword would drop them all in
    silence, which is exactly the loss this module refuses everywhere else.
    """
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ComponentSpecError(
            f"Component {name}: 'params' is the declaration of a component "
            f"class, a mapping, got {type(params).__name__}"
        )
    if not params:
        return {}

    _refuse_callables(f"Component {name}: params", params)

    signature = inspect.signature(component_class.add_flows)
    reads = any(
        parameter.kind
        in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for key, parameter in signature.parameters.items()
        if key != "self"
    )
    if not reads:
        raise ComponentSpecError(
            f"Component {name}: 'params' names the declaration of a component "
            f"CLASS, and `{component_class.__name__}.add_flows` reads none, so "
            f"the keys {', '.join(sorted(params))} would be silently dropped. "
            f"Declare them in the sections instead, or name a class that reads "
            f"them"
        )

    # Bound against the signature rather than tried against the call, so a
    # parameter the class does not read is refused HERE, by name, instead of
    # reaching the caller as a bare TypeError with no declaration in it. The
    # placeholder stands in for `self`, which the build supplies.
    try:
        signature.bind(None, **params)
    except TypeError as error:
        raise ComponentSpecError(
            f"Component {name}: 'params' does not fit "
            f"`{component_class.__name__}.add_flows`: {error}"
        ) from error

    return dict(params)


def _plan(spec: Any, classes: dict[str, type]) -> tuple[str, type, dict, list[_Call]]:
    """Everything a declaration expands to, checkable from the mapping alone.

    The single expansion :func:`check_spec` and :func:`build_component` share,
    so validation and construction cannot drift: what validates is exactly what
    would be called.
    """
    if not isinstance(spec, dict):
        raise ComponentSpecError(
            f"A component declaration is a mapping, got {type(spec).__name__}"
        )

    name = spec.get("name")
    if not name or not isinstance(name, str):
        raise ComponentSpecError(f"Component declaration without a 'name': {spec!r}")

    unknown = sorted(set(spec) - COMPONENT_KEYS)
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        raise ComponentSpecError(
            f"Component {name}: unknown declaration key{plural} "
            f"{', '.join(repr(key) for key in unknown)}; it accepts "
            f"{', '.join(sorted(COMPONENT_KEYS))}"
        )

    for section, reason in UNCARRIED_SECTIONS.items():
        if spec.get(section):
            raise ComponentSpecError(
                f"Component {name}: section '{section}' declares {reason}"
            )
    for key, reason in UNCARRIED_KEYS.items():
        if spec.get(key):
            raise ComponentSpecError(f"Component {name}: `{key}` asks for {reason}")

    for key in ("label", "description", "metadata"):
        if key in spec:
            _refuse_callables(f"Component {name}: {key}", spec[key])

    component_class = _resolve_class(spec, name, classes)
    params = _check_params(spec, name, component_class)

    calls: list[_Call] = []
    for section in DECLARATION_SECTIONS:
        if section == "flows":
            calls += _flow_calls(spec, name)
        elif section == "failure_modes":
            calls += _failure_mode_calls(spec, name)
        else:
            calls += _plain_calls(spec, name, section)

    return name, component_class, params, calls


# --- the two entry points ----------------------------------------------------


def check_spec(spec: Any, classes: dict[str, type] | None = None) -> str:
    """Validate a declaration WITHOUT building anything, and return its name.

    Everything checkable from the mapping alone is checked here, before the
    component exists: a caller validating a batch of declarations should not
    have to raise an engine to find out that one of them is misspelled.

    The sections are checked in the order :data:`DECLARATION_SECTIONS` imposes
    rather than in the order the mapping happens to list them, so a declaration
    faulty in two places reports the fault the build would reach first.

    What is left to :func:`build_component` is what only the component can
    answer: that a rule names a declared flow, that a capacity holds one, that a
    conduit does not meter what a rule already carries.

    Parameters
    ----------
    spec : dict
        The declaration.
    classes : dict, optional
        The component classes ``cls`` may name, defaulting to
        :data:`COMPONENT_CLASSES`.

    Returns
    -------
    str
        The declared component name.

    Raises
    ------
    ComponentSpecError
        On any fault reachable from the mapping alone.
    """
    name, _, _, _ = _plan(spec, COMPONENT_CLASSES if classes is None else classes)
    return name


def build_component(
    system: authoring.System,
    spec: Any,
    classes: dict[str, type] | None = None,
) -> authoring.ObjFlow:
    """Build one component from a declaration held in data, and register it.

    The declaration is validated first, then expanded in the order
    :data:`DECLARATION_SECTIONS` imposes, whatever order the mapping listed its
    sections in. The named class declares its own ports first and the sections
    are added on top, so a class stays usable as a template.

    Parameters
    ----------
    system : pyraichu.muscadet.System
        The system the component is added to.
    spec : dict
        The declaration. ``name`` is required; ``cls`` defaults to the bare
        component class and ``params`` is the named class's own declaration.
    classes : dict, optional
        The component classes ``cls`` may name, defaulting to
        :data:`COMPONENT_CLASSES`.

    Returns
    -------
    pyraichu.muscadet.ObjFlow
        The built component, registered under its name and ready to connect.

    Raises
    ------
    ComponentSpecError
        For anything the declaration cannot carry, and for what only the build
        can answer: the message names the section and the entry, and carries
        the authoring layer's own diagnostic underneath.
    """
    name, component_class, params, calls = _plan(
        spec, COMPONENT_CLASSES if classes is None else classes
    )

    if name in system.comp:
        raise ComponentSpecError(
            f"Component {name}: system `{system.name}` already holds a "
            f"component of that name. A declaration builds a NEW component; "
            f"give this one a distinct 'name'"
        )

    # Built without running the constructor, so the class's own declaration can
    # be made WITH its parameters and the sections added on top of it. The
    # serialized plugin path builds a component the same way, and
    # `_init_declarations` exists precisely so neither has to know the list of
    # declaration lists.
    component = component_class.__new__(component_class)
    component.name = name
    component._init_declarations()

    # Decoration muscadet carries on the component itself. It reaches no
    # generated model: it is kept so a declaration survives a round trip
    # through a live component rather than losing what a platform export knows
    # about the instance.
    if spec.get("label") is not None:
        component.label = spec["label"]
    if spec.get("description") is not None:
        component.description = spec["description"]
    if spec.get("metadata"):
        component.metadata = dict(spec["metadata"])

    component.add_flows(**params)

    for call in calls:
        method = getattr(component, call.method)
        try:
            method(**call.keywords)
        except (ValueError, KeyError) as error:
            # The authoring layer's diagnostic names the flow, the coefficient
            # or the pattern at fault; what it cannot name is which declaration
            # entry asked for it, since it never saw one. A KeyError is what a
            # mapping missing a required inner key raises, and it reads as a
            # bare key name without its type.
            detail = (
                str(error)
                if isinstance(error, ValueError)
                else f"{type(error).__name__}: {error}"
            )
            raise ComponentSpecError(
                f"Component {name}: section '{call.section}', entry "
                f"{call.entry!r}: {detail}"
            ) from error

    system.comp[name] = component
    return component
