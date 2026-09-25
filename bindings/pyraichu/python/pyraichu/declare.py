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

It owns five entry points, over three scales:

- :func:`check_spec` validates a component declaration and builds nothing;
- :func:`build_component` turns it into a live component;
- :func:`check_system_spec` and :func:`build_system` do the same for a whole
  system, which is the scale that carries what no component knows: how they
  are wired. That document is ``muscadet.declare.system_spec``, the one an
  engine receives at muscadet's extension point;
- :func:`build_document` answers the model document the whole declaration
  describes, which is one scale further out than the system: it holds what is
  in ``components`` and is no part of the flow graph.

**A component declaration takes three shapes**, and states which under
:data:`COMPONENT_KIND_KEY`. Absent, or ``"flow"``, it is a component holding
flows, which is what every declaration written before the key existed means and
what the sections below describe. ``"two_state_mode"`` is a component of
``system.comp`` carrying no flow at all and holding a two-state automaton.
``"controller"`` is a ``muscadet.ObjCtrl``, a PEER of ``ObjFlow`` and not a
subclass of it: it carries no flow either, and what it holds is two sections,
the quantities it OBSERVES and the signals it PUBLISHES, each signal with the
emission grammar that computes it.

That second shape covers THREE families, and the kind names the two states
they share rather than the nature of the most common one. Two of them apply
their effects to components they NAME rather than own -- the ``cod3s.ObjFM``
façades, spelled ``failure_*`` / ``repair_*``, and ``cod3s.ObjMode2S`` itself,
spelled ``occ_*`` / ``not_occ_*``. Nothing on those components records the
mode, so skipping the entry would not lose decoration, it would lose the model
-- a compromise cascade would become a system where nothing ever fails. The
third is ``cod3s.ObjEvent``, which affects nothing at all: it is SELF-HOSTED,
names no target, and watches a condition over arbitrary components of the
system. The COD3S Platform synthesises one per study event and one per
indicator whose formula has more than one clause, so a document carrying one
is an ordinary document.

Skipping a controller would be the same loss and worse in one way: a controller
is WIRED, so a document that dropped it would keep the connections naming it
and rebuild into a system whose commanded equipment is never told anything.

The three shapes are read by the same functions and built by different ones.
Neither a standalone mode nor a controller becomes an ``ObjFlow``: each becomes
a ``pyraichu.plugins.muscadet`` object -- an ``ObjFM`` or an ``ObjEvent`` for
the mode, an ``ObjCtrl`` for the controller -- expanded ONCE THE FLOW
COMPONENTS EXIST, which is :func:`build_document`'s scale and not
:func:`build_system`'s. Those are also the objects a COD3S Platform study's
modes, events and controllers are translated into, so the two corpora share
one expansion instead of this reader carrying a second
and lesser one.

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

import math

import copy
import inspect
import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from . import muscadet as authoring
from .indicators import (
    GENERATED_INDICATORS,
    generated_indicators,
    merge_indicators,
)

__all__ = [
    "AVAILABILITY_SUFFIX",
    "COMPONENT_CLASSES",
    "COMPONENT_KEYS",
    "COMPONENT_KIND_CONTROLLER",
    "COMPONENT_KIND_FLOW",
    "COMPONENT_KIND_KEY",
    "COMPONENT_KIND_TWO_STATE_MODE",
    "COMPONENT_KINDS",
    "CONTROLLER_KEYS",
    "CONTROLLER_SECTIONS",
    "DECLARATION_SECTIONS",
    "DERIVED_OUT_AUTOMATA_KEY",
    "FLOW_CLASSES",
    "MODE_CLASSES",
    "MODE_LOGIC",
    "MODE_OPERATORS",
    "MUSCADET_QUANTITY_SUFFIX",
    "MUSCADET_SIGNAL_SUFFIX",
    "PLAIN_SECTIONS",
    "PRODUCTION_SUFFIX",
    "SYSTEM_SPEC_VERSION",
    "ComponentSpecError",
    "SystemSpecError",
    "build_component",
    "build_document",
    "build_system",
    "capacity_absent_variables",
    "capacity_content_variables",
    "check_controller_spec",
    "check_mode_spec",
    "check_spec",
    "check_system_spec",
    "component_kind",
    "controller_object",
    "controller_signal_variables",
    "derived_out_states",
    "entry_call",
    "event_automaton",
    "event_occurrence_state",
    "mode_object",
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
    "mixtures": (
        "a group of continuous inputs drawn TOGETHER at one volumetric rate "
        "(muscadet's `add_mixture_in`, R51): a machine displacing a VOLUME of "
        "whatever the volume it draws from holds, what leaves per constituent "
        "being fixed by the composition of that volume rather than declared. "
        "Every rate this layer carries is a rate PER FLOW -- a rule's `cons`, "
        "a capacity's `serve_rate`, a source's `rate` -- so a group of two "
        "constituents read here would become two independent demands, which is "
        "two degrees of freedom where the physics has one and is the very "
        "model R51 exists to refuse. Nothing stands in its place: a model that "
        "ventilates belongs on the reference engine until this layer carries "
        "the group"
    ),
}

#: Component-level keys carrying no counterpart, refused as soon as they ask for
#: something. Same treatment as the sections above: a value that says nothing is
#: accepted, so a read-back is not refused for its decoration.
#:
#: **Empty, and kept rather than removed.** It held
#: ``create_default_out_automata`` until the platform corpus was read back
#: through this layer: muscadet writes that key ONLY when it is ``True``
#: (``muscadet/declare.py``, "written only when it says something"), so the
#: "accepted while it says nothing" contract above never applied to it and
#: every platform export was refused. The refusal is now indexed on what the
#: absence PROVOKES rather than on the declaration -- see
#: :func:`derived_out_states` and the state indicator of
#: :mod:`pyraichu.muscadet_engine` -- and the mechanism stays here because a
#: component-level key without a counterpart is exactly what it is for.
UNCARRIED_KEYS: dict[str, str] = {}

#: The key a component declares the DERIVED ok/nok automaton pair under,
#: muscadet's own (``ObjFlow(create_default_out_automata=...)``).
DERIVED_OUT_AUTOMATA_KEY = "create_default_out_automata"

#: Key a component declaration states its SHAPE under, muscadet's own
#: (``muscadet.declare.COMPONENT_KIND_KEY``). Absent means
#: :data:`COMPONENT_KIND_FLOW`, which is what every declaration written before
#: the key existed means -- so no stored document has to be reread, and this
#: reader answers a document without it exactly as it always did.
COMPONENT_KIND_KEY = "kind"

#: A component holding flows: the shape the sections above describe, and the
#: default.
COMPONENT_KIND_FLOW = "flow"

#: A STANDALONE TWO-STATE mode: a component of ``system.comp`` carrying no flow
#: at all, holding a two-state automaton. The value names the STATE COUNT,
#: which is what says which constructor to call, and it covers the three
#: families :data:`MODE_CLASSES` reads: the ``cod3s.ObjFM`` façades,
#: ``cod3s.ObjMode2S`` itself, and ``cod3s.ObjEvent``. Only the first two apply
#: effects to components they name; an event observes and writes nothing. See
#: the standalone-mode section below.
COMPONENT_KIND_TWO_STATE_MODE = "two_state_mode"

#: A CONTROLLER: a ``muscadet.ObjCtrl``, which is a PEER of ``ObjFlow`` and not
#: a subclass of it. It carries no flow and no occurrence law, so it is neither
#: of the two shapes above: it OBSERVES quantities and PUBLISHES signals, under
#: the two sections :data:`CONTROLLER_SECTIONS` names. See the controller
#: section below.
COMPONENT_KIND_CONTROLLER = "controller"

COMPONENT_KINDS = (
    COMPONENT_KIND_FLOW,
    COMPONENT_KIND_TWO_STATE_MODE,
    COMPONENT_KIND_CONTROLLER,
)

#: Every key a FLOW declaration may carry.
COMPONENT_KEYS = frozenset(
    CONSTRUCTOR_KEYS
    + (SOURCE_CLS_KEY, COMPONENT_KIND_KEY, DERIVED_OUT_AUTOMATA_KEY)
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

    Four classes of key, and the difference between the middle two is the
    whole discipline of this module:

    - ``carried`` maps a muscadet declaration key onto the authoring
      keyword that honours it;
    - ``inert`` names a key this layer does not read but that says nothing
      at the value listed, which is muscadet's own default. It is accepted
      at that value and REFUSED at any other, so a read-back writing every
      field is not refused for its defaults while a declaration that
      actually asks for something still is;
    - ``uncarried`` names a key refused outright, with the reason;
    - ``runtime`` names a field that is not a declaration at all: an engine
      object muscadet binds while it builds and rebuilds at the next
      construction. It is dropped WHATEVER it holds, where an inert key is
      dropped only at the value that says nothing. The difference is that a
      handle never carries a decision, so there is no value of it a
      declaration could lose.
    """

    carried: dict[str, str]
    inert: dict[str, Any] = field(default_factory=dict)
    uncarried: dict[str, str] = field(default_factory=dict)
    runtime: frozenset = frozenset()

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

#: The connection filter that filters nothing: muscadet's own default, one
#: entry matching every backend class name. It is what its read-back writes on
#: a flow nobody restricted, so it is the value that says nothing here.
_ANY_COMPONENT_AUTHORIZED = [{"class_name_bkd": ".*"}]

#: What every flow declaration carries beside its own family's keys: fields
#: muscadet writes on every flow and that declare nothing here.
_FLOW_SHARED = {
    "component_authorized": _ANY_COMPONENT_AUTHORIZED,
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
    component_authorized=(
        "which component classes may connect to this flow. muscadet refuses a "
        "connection its filter excludes; this layer wires what the connection "
        "list says and checks no class, so a narrowed filter would be a "
        "refusal the model declares and never gets. Leave it at the default "
        "that filters nothing and keep the restriction on the muscadet side"
    ),
)

#: What a DISCRETE flow carries by default where it has fed nothing yet.
#: muscadet's read-back writes the field on every flow, and writes `False`
#: there, so `False` is the value that says nothing on both sides. Written as
#: a name rather than inline because the two discrete vocabularies below must
#: not drift apart: read back through one and refused through the other, one
#: muscadet component would be buildable and its neighbour not.
_DISCRETE_FED_DEFAULT = False

_DISCRETE_IN = _Vocabulary(
    carried={
        "name": "name",
        "logic": "logic",
        # What the input reads WHILE NOTHING FEEDS IT (muscadet
        # `FlowDiscreteIn.var_in_default`, default False). Carried rather
        # than inert because `True` is how a model grounds a chain at the
        # physical edge -- an always-fed boundary input -- and refusing it
        # refused the whole model rather than the one field.
        "var_in_default": "var_in_default",
    },
    inert=dict(
        _FLOW_SHARED,
        var_type="bool",
        var_fed_default=_DISCRETE_FED_DEFAULT,
        # The same thing on the AVAILABILITY channel, and the one value it
        # is ever declared at is the neutral element: out of connection
        # muscadet reads `fed = agg_in(...) AND agg_avail(..., True)`, so
        # the feed channel decides alone. Connected, this layer has no
        # availability port at all: a producer's `{flow}_fed_out` already
        # ANDs its own `{flow}_fed_available_out`, so fed implies
        # available per producer and every aggregate this layer writes is
        # monotone. `False` would say something, and is refused.
        var_available_in_default=True,
    ),
    uncarried=_FLOW_UNCARRIED,
)

#: How the two levels of a production condition combine, and muscadet's own
#: default (``FlowDiscreteOut.var_prod_cond_inner_mode``). Named because BOTH
#: values are carried and the pair is read in three places: the vocabulary
#: below, the conversion of :func:`_prod_cond`, and the refusal of a third
#: spelling.
PROD_COND_INNER_MODES = ("or", "and")
PROD_COND_INNER_MODE_DEFAULT = "or"

_DISCRETE_OUT_KEYS = {
    "name": "name",
    "var_prod_default": "var_prod_default",
    "var_prod_cond": "var_prod_cond",
    # NOT an authoring keyword: it selects the SHAPE the condition beside it
    # is written in, and `_flow_calls` consumes it while converting that
    # condition. Carried rather than inert because reading one shape for the
    # other inverts the condition without a word -- see :func:`_prod_cond`.
    "var_prod_cond_inner_mode": "var_prod_cond_inner_mode",
    # The availability gate's seed and its reset discipline: muscadet's two
    # knobs of a DORMANT service function. A platform export writes `False`
    # on every output of a standby channel, so refusing them at anything but
    # muscadet's default refused the whole corpus.
    "var_fed_available_out_init": "var_fed_available_out_init",
    "var_fed_available_out_reset": "var_fed_available_out_reset",
}

_DISCRETE_OUT_INERT = dict(
    _FLOW_SHARED,
    var_type="bool",
    var_fed_default=_DISCRETE_FED_DEFAULT,
    var_is_active_default=True,
    var_prod_cond_negate=[],
    var_prod_cond_compare=[],
    negate=False,
)

_DISCRETE_OUT = _Vocabulary(
    carried=_DISCRETE_OUT_KEYS,
    inert=_DISCRETE_OUT_INERT,
    uncarried=_FLOW_UNCARRIED,
)

#: The automaton state a dynamic discrete output BINDS once it is built:
#: muscadet keeps the state object on the flow (`FlowDiscreteOutTempo.
#: state_enable_bkd`, `FlowDiscreteOutOnTrigger.trigger_up`) and its read-back
#: writes it out, since neither name carries the `var_`/`sm_` prefix its
#: runtime-handle rule looks for. They are handles all the same: this layer
#: derives the automaton and names the state itself, so there is nothing in
#: one a declaration could ask for and nothing a reader could lose.
_DYNAMIC_OUT_RUNTIME = frozenset({"state_enable_bkd", "trigger_up"})

_DISCRETE_OUT_TEMPO = _Vocabulary(
    carried=dict(
        _DISCRETE_OUT_KEYS,
        occ_enable_flow="enable_law",
        occ_disable_flow="disable_law",
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
    runtime=_DYNAMIC_OUT_RUNTIME,
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
    runtime=_DYNAMIC_OUT_RUNTIME,
)

#: What a CONTINUOUS INPUT has been fed before anything flows. muscadet's
#: read-back writes the field on every flow and writes `0.0` there, and on an
#: input it is the seed of the MIRROR `{name}_fed_in` (`FlowModel`, muscadet
#: `flow.py`), which the sensitive method overwrites with what the connections
#: delivered. Nothing on this side reads that mirror: a continuous input is
#: read through its connections precisely because the mirror lags a step, so
#: the seed says nothing and `0.0` is accepted.
#:
#: **Not an alias of `var_in_default`**, which muscadet documents as the value
#: an input reads WHEN NOT CONNECTED and which this vocabulary carries just
#: above. Mapping one onto the other would start an unconnected input at a
#: value the declaration never put there.
_CONTINUOUS_IN_FED_DEFAULT = 0.0

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
        # as an observable channel, for a controller to threshold. Either
        # what crossed or what could have, `true` meaning the first.
        "publish_rate": "publish_rate",
    },
    inert=dict(
        _FLOW_SHARED,
        var_type="float",
        var_fed_default=_CONTINUOUS_IN_FED_DEFAULT,
    ),
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
    inert=dict(
        _FLOW_SHARED,
        var_type="float",
        var_demand_in_default=0.0,
        # How the operands of a production condition combine. This
        # vocabulary carries NO `var_prod_cond` -- what a continuous output
        # produces is derived from its rule sets, and a declared production
        # condition is refused by name as a key this flow class does not
        # know -- so the mode that would combine its operands says nothing at
        # muscadet's default and is refused above it. A DISCRETE output
        # CARRIES the same key, because it carries the condition the key
        # shapes; here there is no condition for it to shape.
        var_prod_cond_inner_mode=PROD_COND_INNER_MODE_DEFAULT,
    ),
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

#: The flow classes whose entries declare a CONTINUOUS flow, either way round.
#: They are the ones a rate channel can be opened on, and both directions
#: qualify: muscadet publishes the quantity crossing the wire and leaves which
#: side of it the publisher sits on to the publisher.
_CONTINUOUS_CLASSES = frozenset({"FlowContinuousIn", "FlowContinuousOut"})

#: The flow classes whose entries declare a DISCRETE OUTPUT. A failure mode
#: reaches one by **gating** it and a continuous one by **derating** it, and
#: the two are not spelled alike on this side: see :func:`_gate_effects`.
_DISCRETE_OUT_CLASSES = frozenset(
    {
        "FlowDiscreteOut",
        "FlowOut",
        "FlowDiscreteOutTempo",
        "FlowOutTempo",
        "FlowDiscreteOutOnTrigger",
        "FlowOutOnTrigger",
    },
)

#: How muscadet names the availability gate of a discrete output: the variable
#: a mode clamps to stop that output producing. It is the dominant 1.x
#: spelling of "this mode kills this output", the flow's own name being the
#: other one, and both are read here.
AVAILABILITY_SUFFIX = "_fed_available_out"

#: How muscadet names the production variable of a discrete output: the one a
#: mode writes to START that output, which is the other half of the same
#: vocabulary. Held in a variable rather than derived from the production
#: condition precisely so a mode has something to latch; see
#: :meth:`pyraichu.muscadet.ObjFlow._build_flows_out`.
PRODUCTION_SUFFIX = "_prod_available"

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
        # Whether the volume has a through-path at all, which `side`
        # cannot say: `both` and `out` share the side `out`, so a buffer
        # and a reservoir reach this reader indistinguishable without it.
        # muscadet writes it on EVERY capacity, at its default too, which
        # is why it is carried rather than inert: the one key here whose
        # presence says nothing about whether the model asks for
        # something.
        "transmits": "transmits",
        # The ceiling on what may leave the volume. Omitted at muscadet's
        # default, which is unbounded, and unbounded is what a volume on
        # an output means: it serves what is asked of it while it holds
        # something.
        "serve_rate": "serve_rate",
        # The DISCHARGE COMMAND, and the two parallel matrices muscadet
        # lifts a negation and a comparison out of its operands into. The
        # three are one declaration: the authoring layer folds the
        # matrices back onto the operand each cell is aligned with, so a
        # document spelling a threshold either way declares the same
        # command. muscadet's own read-back writes the operands alone,
        # having rebuilt them from the matrices (`_prod_cond_spec`); a
        # document written against its `Capacity` model carries the
        # matrices instead, and both arrive here.
        "serve_cond": "serve_cond",
        "serve_cond_negate": "serve_cond_negate",
        "serve_cond_compare": "serve_cond_compare",
        # How that command reads its two levels. Carried rather than
        # inert at muscadet's default, because a command declaring the
        # other one says something and would otherwise be read inverted.
        "serve_cond_inner_mode": "serve_cond_inner_mode",
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


# --- the second shape of a component: a STANDALONE failure mode -------------
#
# A ``system.comp`` is not only made of components holding flows. A mode
# declared with ``system.add_component(cls="ObjFMDelay", fm_name=...,
# targets=[...])`` is a component of its own: it carries no flow, it holds an
# occurrence law, and it applies its effects to components it NAMES rather than
# owns. Nothing on those components records it -- unlike a mode declared ON one,
# which lands in that component's ``failure_modes`` section -- so skipping the
# entry would not lose decoration, it would lose the model.
#
# **It does not land in the same place as a mode declared on a component.** A
# ``failure_modes`` entry becomes ``add_delay_failure_mode`` on the component
# that owns it, and the availability of that component's discrete outputs
# follows the mode's automaton. A standalone mode has no such owner: it reaches
# several components, its condition may read a variable or an automaton state of
# any component of the model, and its effects WRITE the attributes they name.
# That is the muscadet plugin's ``ObjFM`` object (:mod:`pyraichu.plugins.
# muscadet`), which is what a platform study's modes already expand to, so this
# reader translates a declaration into one rather than inventing a lesser third
# path. :func:`build_document` is where the two meet.
#
# **A third family shares the kind, and shares almost nothing else.** A
# ``cod3s.ObjEvent`` is also a two-state component of ``system.comp``, and it
# is where the reading above stops applying: it is SELF-HOSTED, so it names no
# target, writes no attribute and has no common-cause order, and its whole
# declaration is a condition over arbitrary components of the system with the
# comparison it is tested by. It expands to the plugin's ``ObjEvent`` object,
# which is what a platform study's events and its multi-clause indicators
# already become -- so here too the translation is shared rather than doubled.

#: The parameter variable each occurrence law is bound to, per direction, as
#: the cod3s mode classes name them. RAICHU bakes the value and has no
#: parameter variable, so the name is decoration -- but one that DISAGREES with
#: the law its class carries is a declaration contradicting itself, and it is
#: refused rather than ignored.
_MODE_PARAM_NAMES = {
    "exp": ("lambda", "mu"),
    "delay": ("ttf", "ttr"),
    "inst": ("gamma", "mu"),
}

#: The field each occurrence law states its value under, which is also what
#: the ENGINE names its parameter variable after: an ``ObjMode2S`` drawing
#: from an exponential names it ``occ_rate`` where the ``ObjFM`` façade names
#: it ``lambda``. Two defaults for one law, and a document carries whichever
#: its writer derived.
_MODE_LAW_FIELD = {"exp": "rate", "delay": "time", "inst": "prob"}

#: What a mode's naming templates are when nobody has renamed anything. A
#: common-cause automaton is named after them, and this layer names its own
#: (``fm__cc_1_2``, cod3s' own convention since 1.9), so a template asking for
#: something else renames an automaton an indicator may be watching.
_MODE_NAME_TEMPLATES = {
    "param_name_order_prefix": "__{order}_o_{order_max}",
    "trans_name_prefix": "__cc_{target_comb_u}",
}

#: Refusals shared by both mode vocabularies, each naming what it asks for.
_MODE_UNCARRIED = dict(
    param_name_order_prefix=(
        "how the per-order parameter variables are named. RAICHU bakes a law's "
        "value into the transition and declares no parameter variable, so "
        "there is no name to give"
    ),
    trans_name_prefix=(
        "the template a common-cause automaton is named after. This layer "
        "names them itself, cod3s' own way (`fm__cc_1_2`), so a template "
        "asking for another spelling renames automata that indicators and "
        "sequences reach by name"
    ),
    trans_name_prefix_fun=(
        "a naming template written as a Python function. See "
        "`trans_name_prefix`, and note that a mapping carries no function "
        "either way"
    ),
    step=(
        "a cod3s `step` object driving the mode. It is a live object, and this "
        "layer reads a declaration held in data"
    ),
)

#: Keys BOTH vocabularies spell alike, carried unchanged into the wire mapping
#: the plugin object is built from.
_MODE_SHARED_CARRIED = {
    "targets": "targets",
    "behaviour": "behaviour",
    "cond_inner_logic": "cond_inner_logic",
    "cond_outer_logic": "cond_outer_logic",
}

#: Inert on both: what muscadet writes on every mode and what says nothing
#: here at the value it writes.
_MODE_SHARED_INERT = dict(
    _MODE_NAME_TEMPLATES,
    trans_name_prefix_fun=None,
    step=None,
    drop_inactive_automata=True,
)

#: ``cod3s.ObjFM`` and its subclasses -- what a muscadet model writes, and what
#: ``system.add_component(cls="ObjFMDelay", ...)`` builds. The occurrence law
#: is carried by the CLASS here, so the declaration states only its parameters.
_FAILURE_MODE_MODE = _Vocabulary(
    carried=dict(
        _MODE_SHARED_CARRIED,
        fm_name="fm_name",
        target_name="target_name",
        failure_state="failure_state",
        failure_cond="failure_cond",
        failure_effects="failure_effects",
        failure_param_name="failure_param_name",
        failure_param="failure_param",
        repair_state="repair_state",
        repair_cond="repair_cond",
        repair_effects="repair_effects",
        # One-shot effects, written on the firing edge (a transition's
        # `effects`), where the two above are held while the state lasts.
        failure_effects_trans="failure_effects_trans",
        repair_effects_trans="repair_effects_trans",
        repair_param_name="repair_param_name",
        repair_param="repair_param",
    ),
    inert=dict(_MODE_SHARED_INERT),
    uncarried=dict(_MODE_UNCARRIED),
)

#: ``cod3s.ObjMode2S`` itself, the generic two-state engine the COD3S Platform
#: emits natively. Its occurrence law is DECLARED rather than carried by the
#: class, which is the one substantive difference between the two vocabularies.
_MODE_2S_MODE = _Vocabulary(
    carried=dict(
        _MODE_SHARED_CARRIED,
        mode_name="fm_name",
        target_name="target_name",
        occ_state="occ_state",
        occ_law="occ_law",
        occ_cond="occ_cond",
        occ_effects="occ_effects",
        occ_param_name="occ_param_name",
        occ_param="occ_param",
        occ_parked_state="occ_parked_state",
        not_occ_state="not_occ_state",
        not_occ_law="not_occ_law",
        not_occ_cond="not_occ_cond",
        not_occ_effects="not_occ_effects",
        # One-shot effects, written on the firing edge (a transition's
        # `effects`), where the two above are held while the state lasts.
        occ_effects_trans="occ_effects_trans",
        not_occ_effects_trans="not_occ_effects_trans",
        not_occ_param_name="not_occ_param_name",
        not_occ_param="not_occ_param",
    ),
    inert=dict(
        _MODE_SHARED_INERT,
        aut_name=None,
        not_occ_parked_state=None,
    ),
    uncarried=dict(
        _MODE_UNCARRIED,
        aut_name=(
            "the name of the mode's own automaton. This layer names it `fm`, "
            "cod3s' own name for a mode over targets, so renaming it here "
            "would leave every indicator and sequence that reaches it by name "
            "pointing at nothing"
        ),
        not_occ_parked_state=(
            "the micro-state a lost RETURN draw waits in. Only an on-demand "
            "(inst) return law parks, and this layer expands an on-demand "
            "occurrence only"
        ),
    ),
)

#: ``cod3s.ObjEvent``: the third family, and the one that affects nothing. A
#: second façade over the same engine, SELF-HOSTED -- one automaton on the
#: event itself, no target and no effect -- whose occurrence is a condition
#: over arbitrary components of the system, compared to a value.
#:
#: **Every constructor argument of it is carried**, which no other family
#: here can say, and each key is spelled the way ``cod3s.ObjEvent.__init__``
#: takes it (``cod3s/pycatshoo/component.py``): the two tempos become the delay
#: laws of its two transitions, the three naming keys the automaton and its
#: states, and the comparison pair the test the condition tree is wrapped in.
#: All of them have a counterpart in
#: :func:`pyraichu.plugins.muscadet._expand_objevent`.
#:
#: The one key here that is NOT an argument of it is
#: ``drop_inactive_automata``, which muscadet's vocabulary accepts on every
#: two-state component: an event hosts ONE automaton and has no common-cause
#: order, so there is nothing for the flag to drop. It is accepted at
#: muscadet's own default, where it says nothing, and refused with that reason
#: anywhere else -- the treatment every inert-and-uncarried key gets.
_EVENT_MODE = _Vocabulary(
    carried={
        # An event's name IS a constructor argument, where every other mode's
        # is derived from its targets.
        "name": "name",
        "cond": "cond",
        "cond_operator": "cond_operator",
        "cond_value": "cond_value",
        "tempo_occ": "tempo_occ",
        "tempo_not_occ": "tempo_not_occ",
        "event_aut_name": "event_aut_name",
        "occ_state_name": "occ_state_name",
        "not_occ_state_name": "not_occ_state_name",
        "inner_logic": "inner_logic",
        "outer_logic": "outer_logic",
    },
    inert={"drop_inactive_automata": True},
    uncarried={
        "drop_inactive_automata": (
            "whether a common-cause order whose law is inactive gets an "
            "automaton anyway. An event is self-hosted: it holds ONE "
            "automaton and has no common-cause order, so there is none to "
            "drop or to keep"
        )
    },
)

#: The six comparisons an event's ``cond_operator`` may name, muscadet's own
#: ``MODE_OPERATORS`` and the plugin's own ``_OPE``: the three layers spell
#: them alike, so nothing is translated here beyond checking the spelling is
#: one of them.
MODE_OPERATORS = ("==", "!=", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class _ModeClass:
    """One mode class a ``two_state_mode`` declaration's ``cls`` may name.

    ``cls`` is resolved by NAME here, where muscadet resolves it through the
    live subclass tree: this layer imports neither cod3s nor muscadet, so it
    knows the classes by the names their declarations carry. A class outside
    the table is refused by name rather than read in a vocabulary that may not
    fit it.
    """

    #: The vocabulary the class's own constructor takes.
    vocabulary: _Vocabulary

    #: ``(occurrence, return)`` occurrence laws the class carries, or
    #: ``(None, None)`` for a class whose laws are DECLARED (``ObjMode2S``).
    laws: tuple

    #: The wire ``cls`` the shared cod3s translation reads. Two names for one
    #: dialect: the deprecated ``muscadet.ObjFailureMode*`` are subclasses of
    #: the cod3s ones and are built from the same wire.
    wire: str

    #: Whether effects and conditions name FLOWS (the muscadet façade, whose
    #: patterns are regular expressions over the target's flow names) or
    #: VARIABLES (the cod3s classes, which name the target's attributes
    #: exactly). Both spellings are load-bearing in models that exist, and
    #: the class is what tells them apart.
    by_flow: bool = False

    #: The key the mode's OWN name is declared under, which is not the same as
    #: the key the document files it under: the three families spell it
    #: ``fm_name``, ``mode_name`` and ``name``. Declared rather than sniffed
    #: from the vocabulary, so a family added here has to say which it is.
    name_key: str = "fm_name"

    #: The two keys the inner and outer truth functions of a condition are
    #: declared under. The engine calls them ``cond_inner_logic`` /
    #: ``cond_outer_logic`` and ``ObjEvent`` calls them ``inner_logic`` /
    #: ``outer_logic``, and a key a constructor does not take is a key
    #: nothing builds.
    logic_keys: tuple = ("cond_inner_logic", "cond_outer_logic")

    #: Whether the mode hosts its OWN automaton and names no target. True for
    #: ``ObjEvent`` alone, and it carries three consequences at once: there is
    #: no target list to resolve a condition leaf against, no effect to apply,
    #: and no common-cause combination -- so no per-order parameter vector and
    #: nothing for ``drop_inactive_automata`` to drop.
    self_hosted: bool = False


#: The mode classes this reader knows, by the name a declaration carries.
#:
#: ``ObjFM`` and ``muscadet.ObjFailureMode`` are deliberately ABSENT: neither
#: carries an occurrence law (``set_occ_law_failure`` is defined by the three
#: flavours below and by none of the two bases), so a mode declared as one has
#: no law to build a transition from. ``ObjDegMode`` is absent for another
#: reason: it is a MULTI-state degradation mode, so it is not the shape this
#: kind names at all, and it gets the refusal :func:`check_mode_spec` gives.
MODE_CLASSES: dict[str, _ModeClass] = {
    "ObjFMExp": _ModeClass(_FAILURE_MODE_MODE, ("exp", "exp"), "ObjFMExp"),
    "ObjFMDelay": _ModeClass(_FAILURE_MODE_MODE, ("delay", "delay"), "ObjFMDelay"),
    "ObjFMInst": _ModeClass(_FAILURE_MODE_MODE, ("inst", "exp"), "ObjFMInst"),
    "ObjFailureModeExp": _ModeClass(
        _FAILURE_MODE_MODE, ("exp", "exp"), "ObjFMExp", by_flow=True
    ),
    "ObjFailureModeDelay": _ModeClass(
        _FAILURE_MODE_MODE, ("delay", "delay"), "ObjFMDelay", by_flow=True
    ),
    "ObjMode2S": _ModeClass(
        _MODE_2S_MODE, (None, None), "ObjMode2S", name_key="mode_name"
    ),
    # An event draws both its transitions from a DELAY law, built out of its
    # two tempos: that is what `cod3s.ObjEvent.__init__` hands its engine, and
    # what `_expand_objevent` emits.
    "ObjEvent": _ModeClass(
        _EVENT_MODE,
        ("delay", "delay"),
        "ObjEvent",
        name_key="name",
        logic_keys=("inner_logic", "outer_logic"),
        self_hosted=True,
    ),
}

#: Every key a standalone-mode declaration may carry, per vocabulary. The
#: component-level keys are the same as a flow component's, minus the sections
#: a mode has none of.
_MODE_COMPONENT_KEYS = frozenset(
    ("name", "cls", COMPONENT_KIND_KEY, "label", "description", "metadata")
)

#: What a refusal calls the entry it refuses, by whether its class is
#: self-hosted. An event is not a failure mode: a message calling it one would
#: send a modeller looking for the targets and the effects it has none of.
_MODE_FAMILY = {False: "Failure mode", True: "Event"}

#: The two truth functions a mode composes its condition groups with, carried
#: by NAME. A document cannot hold a callable, and the plugin reads the same
#: two names, so nothing is translated here beyond checking that the name is
#: one of them.
MODE_LOGIC = ("all", "any")

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
        # A runtime handle is dropped before it is classified: it is an engine
        # object rather than a declaration, so neither its presence nor its
        # value says anything a reader could honour or lose.
        if key in vocabulary.runtime:
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

#: What a production-condition operand may carry, and the whole of it. muscadet
#: writes back exactly these five keys and reads no others
#: (``muscadet/declare.py``, ``_prod_cond_spec``; ``muscadet/obj.py``,
#: ``apply_prod_cond``). ``release`` is absent on purpose: muscadet's operand
#: ignores it, and a band needs a location to hold between its two edges where
#: the production variable here is rewritten at every evaluation.
PROD_COND_OPERAND_KEYS = ("name", "port", "negate", "op", "value")

#: The comparison operators an operand may carry. The rule-guard vocabulary,
#: and the same list muscadet reaches from both directions
#: (``muscadet.rules._COMPARATORS``, read by ``cond_readers``): one comparison
#: syntax for a controller's guard and for a production condition.
PROD_COND_COMPARISONS = ("<", "<=", ">", ">=", "==", "!=")


def _prod_cond_operand(where: str, operand: Any, inputs: set[str]) -> str | dict:
    """One production-condition operand, in the form the layer underneath
    reads.

    An operand that says nothing beyond its flow name is reduced **to that
    name**, which is the historical form: every document this layer built
    before it read the other three keys still builds, character for character.
    ``port: "in"`` reduces too, and so does ``port: "out"`` on a name this
    component carries only on the output side, because both name the side the
    default resolution -- inputs first, exactly as muscadet's -- picks anyway.
    That is why a platform export is unmoved by this: muscadet writes ``port``
    on every flow operand it hands back, and on almost every one of them it
    states the default.

    What survives as a mapping is what changes the reading:

    - ``negate``, muscadet's ``var_prod_cond_negate`` written inline: the
      operand's guard is denied;
    - ``op`` and ``value``, muscadet's ``var_prod_cond_compare``: the operand
      compares what its name carries against a threshold instead of reading a
      boolean state;
    - ``port: "out"`` on a name this component declares as an input **too**.
      That one is a resolution rule rather than a key: the default resolves the
      input side first, and this is the single case where honouring the
      explicit selection and applying the default part company.

    The two that cannot be combined are refused rather than ordered: a
    comparison already yields a truth value, so it is denied by the opposite
    operator. muscadet refuses the pair at the same place and for the same
    reason (``muscadet.rules.check_operand_negation``).
    """
    if isinstance(operand, str):
        return operand
    if not isinstance(operand, dict):
        raise ComponentSpecError(
            f"{where} carries the production-condition operand {operand!r}; an "
            f"operand is a flow name or a mapping carrying `name`"
        )

    unknown = sorted(set(operand) - set(PROD_COND_OPERAND_KEYS))
    if unknown:
        raise ComponentSpecError(
            f"{where} carries unknown production-condition operand keys "
            f"{unknown}; an operand carries "
            f"{', '.join(f'`{key}`' for key in PROD_COND_OPERAND_KEYS)}"
        )

    name = operand.get("name")
    if not isinstance(name, str):
        raise ComponentSpecError(
            f"{where} carries a production-condition operand without a `name`: "
            f"{operand!r}"
        )

    port = operand.get("port")
    if port not in (None, "in", "out"):
        raise ComponentSpecError(
            f"{where} declares the production-condition operand `{name}` on "
            f"port {port!r}, expected 'in' or 'out'"
        )

    negated = bool(operand.get("negate", False))
    comparison = operand.get("op")
    threshold = operand.get("value")

    if comparison is not None:
        if comparison not in PROD_COND_COMPARISONS:
            raise ComponentSpecError(
                f"{where} compares the production-condition operand `{name}` "
                f"with `{comparison}`, expected one of "
                f"{', '.join(PROD_COND_COMPARISONS)}"
            )
        if threshold is None:
            raise ComponentSpecError(
                f"{where} compares the production-condition operand `{name}` "
                f"with `{comparison}` and no `value` to compare it against"
            )
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ComponentSpecError(
                f"{where} compares the production-condition operand `{name}` "
                f"against `value`={threshold!r}, which is no number"
            ) from None
        if negated:
            raise ComponentSpecError(
                f"{where} both negates and compares the production-condition "
                f"operand `{name}`; a comparison already yields a truth value, "
                f"so it is denied by the opposite operator rather than beside "
                f"it"
            )
    elif threshold is not None:
        raise ComponentSpecError(
            f"{where} declares `value`={threshold!r} on the production-"
            f"condition operand `{name}` and no `op`; a threshold is the "
            f"right-hand side of a comparison, so it needs the operator "
            f"reading it"
        )

    # The one port selection the default resolution would not have made. Every
    # other one is dropped, so the mapping below carries what it changes and
    # nothing else -- and so the historical form survives wherever it said the
    # same thing.
    ported = port == "out" and name in inputs
    if not negated and comparison is None and not ported:
        return name

    reduced: dict[str, Any] = {"name": name}
    if ported:
        reduced["port"] = "out"
    if negated:
        reduced["negate"] = True
    if comparison is not None:
        reduced["op"] = comparison
        reduced["value"] = threshold
    return reduced


def _prod_cond(
    where: str,
    declared: Any,
    inputs: set[str],
    inner_mode: str = PROD_COND_INNER_MODE_DEFAULT,
) -> list[list[str | dict]]:
    """muscadet's production condition, converted to this layer's form.

    **The two conventions are not the same one, and reading either as the other
    inverts the condition.** This layer reads a list of groups as the OR of
    conjunctions -- the platform-export form -- while what muscadet's list
    means is decided by ``var_prod_cond_inner_mode``, which swaps BOTH levels
    at once (``muscadet/flow.py``, ``prod_cond_holds``):

    - ``"or"``, muscadet's default, is ``all(any(...))``: every top-level
      element is a clause, its own operands OR-ed, and the clauses AND-ed.
      That is CONJUNCTIVE normal form, and it has to be converted;
    - ``"and"`` is ``any(all(...))``, which is ALREADY the disjunction of
      conjunctions this layer reads, so it is passed through untouched.

    Passed through untouched under ``"or"``, ``[["a", "b"], ["c"]]`` would
    mean ``(a or b) and c`` on one side and ``(a and b) or c`` on the other,
    and nothing would say so. Converted under ``"and"``, ``[["a"], ["b"]]``
    would mean ``a or b`` on one side and ``a and b`` on the other -- which is
    what a platform export hits, since it writes ``"and"`` on every discrete
    output it declares.

    The CNF conversion expands the clauses: one conjunction per choice of a
    single operand from each clause, which is exact and, for a real condition,
    small. The disjunctive form needs none of it, and no ceiling either: there
    is nothing to multiply out.

    The flat form agrees by accident UNDER ``"or"`` and is worth stating,
    because muscadet's own docstring gets it backwards: ``["a", "b"]`` is TWO
    clauses of one operand, hence ``a and b``, and that is what the layer
    underneath reads a flat list as too.
    """
    if inner_mode not in PROD_COND_INNER_MODES:
        raise ComponentSpecError(
            f"{where} declares `var_prod_cond_inner_mode`={inner_mode!r}; a "
            f"production condition combines its two levels one way or the "
            f"other, so it is "
            f"{' or '.join(repr(mode) for mode in PROD_COND_INNER_MODES)}"
        )
    if not declared:
        return []
    if isinstance(declared, (str, dict)):
        declared = [declared]
    if not isinstance(declared, (list, tuple)):
        raise ComponentSpecError(
            f"{where} carries a production condition that is neither a name, "
            f"an operand nor a list: {declared!r}"
        )

    clauses: list[list[str | dict]] = []
    for group in declared:
        operands = group if isinstance(group, (list, tuple)) else [group]
        # An empty group is the one malformed shape that runs to completion
        # without a word, and it does so DIFFERENTLY on the two readings. Under
        # `"and"` it reaches the layer underneath as an empty conjunction,
        # which holds; under `"or"` the expansion below multiplies it out and a
        # product over an empty clause is EMPTY, so the condition disappears
        # entirely -- no writer, the variable left on its declared default,
        # which is the dormant function and the exact opposite of the empty
        # disjunction the group states.
        #
        # muscadet refuses it too, with the same reasoning and on the family it
        # had left for later: `FlowContinuousOut.check_prod_cond_shape` says
        # "under inner mode 'or' the output would never produce, under 'and'
        # the condition would never bind: neither is a declaration", and scopes
        # itself to the continuous classes because "the discrete classes are
        # 1.x surface with the same laxity, and tightening them belongs to its
        # own change". This is the discrete half of that one refusal, so
        # relaxing it here would make the two families answer differently to
        # one malformed export.
        if not operands:
            raise ComponentSpecError(
                f"{where} carries an empty group in its production condition; "
                f"a group that reads nothing says nothing about when the "
                f"output produces, and the two readings of "
                f"`var_prod_cond_inner_mode` do not even agree on what it "
                f"would mean"
            )
        clauses.append(
            [_prod_cond_operand(where, operand, inputs) for operand in operands]
        )

    if inner_mode == "and":
        return clauses

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


#: cod3s class name -> the short law name this layer reads. Only the laws a
#: temporised output carries: any other keeps the name it was declared under,
#: so its refusal names what the caller wrote.
_COD3S_LAW_NAMES = {
    "DelayOccDistribution": "delay",
    "InstOccDistribution": "inst",
    "ExpOccDistribution": "exp",
}

#: Short law name -> the one parameter it carries beside `cls`.
_TEMPO_LAW_PARAMETERS = {"delay": "time", "exp": "rate"}


def _tempo_law(where: str, key: str, declared: Any) -> dict[str, Any]:
    """A temporisation law, in the engine's transition form.

    A fixed delay waits `time`; an exponential law draws the wait at `rate`,
    which is the reference engine's reading (measured 2026-09-24: the
    switch-on probability t hours after the condition holds is
    1 - exp(-rate * t)). An instantaneous law is a zero delay when it surely
    fires; any other law is refused by name rather than reduced to something
    it is not.
    """
    if declared is None:
        return {"distrib": "delay", "time": 0.0}
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        return {"distrib": "delay", "time": float(declared)}
    if not isinstance(declared, dict):
        raise ComponentSpecError(
            f"{where} declares `{key}`={declared!r}, which is no occurrence "
            f"law: declare {{'cls': 'delay', 'time': t}}"
        )
    law = declared.get("cls", "delay")
    # muscadet's read-back writes the cod3s object's class name, not the short
    # name: both spellings are one law.
    law = _COD3S_LAW_NAMES.get(law, law)
    if law == "inst":
        # A firing probability below one is a draw, not a delay.
        probs = declared.get("probs") or [1]
        if list(probs) != [1]:
            raise ComponentSpecError(
                f"{where} declares `{key}` as an instantaneous law with firing "
                f"probabilities `probs`={probs!r}; a temporised output here "
                f"fires surely, so only a sure firing is carried"
            )
        return {"distrib": "delay", "time": 0.0}
    parameter = _TEMPO_LAW_PARAMETERS.get(law)
    if parameter is None:
        raise ComponentSpecError(
            f"{where} declares `{key}` under the occurrence law '{law}'; a "
            f"temporised output here waits on 'delay', 'exp' or 'inst'"
        )
    unknown = sorted(set(declared) - {"cls", parameter})
    if unknown:
        raise ComponentSpecError(
            f"{where} declares `{key}` carrying unknown keys {unknown}; a "
            f"{law} law carries `cls` and `{parameter}`"
        )
    if law == "exp" and parameter not in declared:
        raise ComponentSpecError(f"{where} declares `{key}` as an exponential law with no `rate`")
    value = declared.get(parameter, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ComponentSpecError(
            f"{where} declares `{key}` with `{parameter}`={value!r}; it must be a finite non-negative number"
        )
    return {"distrib": law, parameter: float(value)}


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


# --- the class stamp muscadet writes inside a rule set -----------------------
#
# muscadet serializes every object it dumps with a `cls` key naming its class,
# and a rule set is dumped whole: the stamp lands on the SET's entry, where
# :func:`_keywords` already drops it, but also on each rule of the set and on
# each operand of a rule's guard, two levels no section vocabulary reaches.
#
# Dropping it is this module's business and not the authoring layer's. This is
# the reader of muscadet's vocabulary, so knowing that muscadet stamps a class
# name on what it serializes belongs here; pyraichu's own writing surface
# (`pyraichu.muscadet.ObjFlow.add_rule_set`) describes the rules IT accepts,
# and teaching it a key it never means would widen that surface for nothing.


def _rule_set_rules(declared: Any) -> Any:
    """The ``rules`` of a rule set, with muscadet's class stamp dropped from
    each rule and from each operand of its guard.

    Nothing else is validated and nothing else is dropped: what a rule may
    carry is :meth:`pyraichu.muscadet.ObjFlow._parse_rule`'s to say, and a
    shape this function does not recognise is handed on untouched so that the
    refusal comes from there, with its own message.
    """
    if not isinstance(declared, (list, tuple)):
        return declared
    return [_rule_without_stamp(rule) for rule in declared]


def _rule_without_stamp(declared: Any) -> Any:
    """One rule of a set, without the class stamp on it or on its guard."""
    if not isinstance(declared, dict):
        return declared
    rule = {key: value for key, value in declared.items() if key != "cls"}
    if "cond" in rule:
        rule["cond"] = _guard_without_stamp(rule["cond"])
    return rule


def _guard_without_stamp(declared: Any) -> Any:
    """A rule's guard, without the class stamp on its operands.

    The guard is a list of operands, or the single operand that is the short
    form of a one-operand guard, or the bare flow name that is the short form
    of an operand: only the mapping forms carry a stamp.
    """
    if isinstance(declared, dict):
        return {key: value for key, value in declared.items() if key != "cls"}
    if isinstance(declared, (list, tuple)):
        return [_guard_without_stamp(operand) for operand in declared]
    return declared


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
    names, so :func:`_flow_calls` converts it after this returns. It converts
    it under ``var_prod_cond_inner_mode``, which is carried here and is the
    one key of this table that names NO authoring keyword: it says which
    shape the condition beside it is written in, and is consumed with it.
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
        ("occ_enable_flow", "enable_law"),
        ("occ_disable_flow", "disable_law"),
    ):
        if keyword in keywords:
            keywords[keyword] = _tempo_law(where, declared, keywords[keyword])
    if "profile" in keywords:
        _check_declared_shape(where, "profile", keywords["profile"], _PROFILE_CLASS)
    if "equation" in keywords:
        _check_declared_shape(where, "equation", keywords["equation"], _TRANSFER_CLASS)
    if "rules" in keywords:
        keywords["rules"] = _rule_set_rules(keywords["rules"])
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

        # Not an authoring keyword: it says which SHAPE the condition beside
        # it was written in, so it is consumed here, with the condition. Read
        # whether or not one is declared, so a third spelling is refused on
        # the key rather than surviving on a flow that happens to carry no
        # condition and inverting the next one that does.
        inner_mode = keywords.pop(
            "var_prod_cond_inner_mode", PROD_COND_INNER_MODE_DEFAULT
        )
        converted = _prod_cond(
            where, keywords.get("var_prod_cond"), inputs, inner_mode
        )
        if "var_prod_cond" in keywords:
            keywords["var_prod_cond"] = converted

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

        _apply_gate_effects(where, spec, keywords)

        calls.append(_Call("failure_modes", entry.get("name", index), method, keywords))

    return calls


def _discrete_out_flows(spec: dict) -> list[str]:
    """The discrete outputs a component declares, in declaration order."""
    return [
        entry["name"]
        for entry in spec.get("flows") or []
        if isinstance(entry, dict)
        and entry.get("cls") in _DISCRETE_OUT_CLASSES
        and isinstance(entry.get("name"), str)
    ]


def _pattern_names(where: str, key: str, pattern: Any, candidate: str) -> bool:
    """Whether an effect pattern names ``candidate``, ANCHORED.

    muscadet's own rule (``muscadet.derating.match_flow_name``), and anchored
    for its reason: unanchored, ``"H2"`` would name ``H2O`` too and an effect
    meant for one output would reach its neighbour.
    """
    if not isinstance(pattern, str):
        return False
    try:
        return re.search(f"^{pattern}$", candidate) is not None
    except re.error as error:
        raise ComponentSpecError(
            f"{where} declares the `{key}` pattern {pattern!r}, which is not a "
            f"regular expression: {error}"
        ) from None


def _gate_effects(
    where: str,
    key: str,
    effects: list[tuple[str, Any]],
    discrete_outs: list[str],
    gating: bool,
) -> tuple[list[str], list[tuple[str, Any]]]:
    """Split a mode's effects into the discrete outputs they GATE and the rest.

    The one place the two layers spell the same statement differently, and
    therefore the one place the translation belongs. muscadet says "this mode
    kills this output" with an effect on the output's availability, reached by
    either of two spellings -- the flow's own name, or the
    ``{flow}_fed_available_out`` variable behind it. This layer says it with
    the mode's ``targets``, and keeps ``failure_effects`` for what a mode does
    to a CONTINUOUS output, which is derate it rather than kill it.

    Left untranslated, such an effect names no continuous output and the
    authoring layer refuses the whole component -- which is what a muscadet
    model reaching this layer used to do.

    ``gating`` is the value that means "gated" for the direction being read:
    ``False`` on the failure side, ``True`` on the repair side. A repair
    effect restoring availability is ABSORBED rather than carried: the gate
    has a per-step reset on both sides, so leaving the failing state restores
    the output with nothing declared. The opposite value in either direction
    is refused rather than reduced: a failure that makes an output available,
    or a repair that keeps it down, is a statement neither layer's gate makes.
    """
    gated: list[str] = []
    remaining: list[tuple[str, Any]] = []
    for pattern, value in effects:
        named = [
            flow
            for flow in discrete_outs
            if _pattern_names(where, key, pattern, flow)
            or _pattern_names(where, key, pattern, f"{flow}{AVAILABILITY_SUFFIX}")
        ]
        if not named:
            remaining.append((pattern, value))
            continue
        if bool(value) is not gating:
            raise ComponentSpecError(
                f"{where} declares `{key}` {(pattern, value)!r} on the "
                f"discrete output(s) {named}: the availability of a discrete "
                f"output is a gate, and on this side of the mode it can only "
                f"be {gating!r}. A mode returning an output degraded rather "
                f"than as-new is a continuous statement, declared as a rate"
            )
        gated += [flow for flow in named if flow not in gated]
    return gated, remaining


def _apply_gate_effects(where: str, spec: dict, keywords: dict[str, Any]) -> None:
    """Rewrite a mode's availability effects into its ``targets``, in place.

    Nothing is refused here for the ABSENCE of a gate: a mode naming neither
    an availability effect nor a target keeps the meaning this layer has
    always given it, every discrete output. That absence means the opposite in
    muscadet's own document, where a mode clamps what it names and nothing
    else, and the two readings are reconciled one scale up -- see
    :func:`_refuse_ungated_modes`, which the system-scale entry point runs and
    the component-scale one does not.
    """
    discrete_outs = _discrete_out_flows(spec)
    if not discrete_outs:
        return

    gated, keywords["failure_effects"] = _gate_effects(
        where,
        "failure_effects",
        keywords.get("failure_effects") or [],
        discrete_outs,
        gating=False,
    )
    _, keywords["repair_effects"] = _gate_effects(
        where,
        "repair_effects",
        keywords.get("repair_effects") or [],
        discrete_outs,
        gating=True,
    )
    for key in ("failure_effects", "repair_effects"):
        if not keywords[key]:
            del keywords[key]

    declared = list(keywords.get("targets") or [])
    targets = declared + [flow for flow in gated if flow not in declared]
    if targets:
        keywords["targets"] = targets


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

    kind = component_kind(spec, name)
    if kind != COMPONENT_KIND_FLOW:
        raise ComponentSpecError(
            f"Component {name}: {COMPONENT_KIND_KEY}={kind!r} declares a "
            f"standalone failure mode, which holds no flow and is therefore no "
            f"component this builds. It is expanded onto the components it "
            f"names, once they exist, which is what "
            f"`pyraichu.declare.build_document` does"
        )

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

    **All three shapes are validated here.** A declaration says which of
    :data:`COMPONENT_KINDS` it is under :data:`COMPONENT_KIND_KEY`; a standalone
    failure mode goes to :func:`check_mode_spec` and a controller to
    :func:`check_controller_spec`. It is the readers that read the three shapes
    and the builders that do not: this one answers "is this declaration well
    formed", which is a question a mode and a controller both have an answer to.

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
    kind = component_kind(spec)
    if kind == COMPONENT_KIND_TWO_STATE_MODE:
        return check_mode_spec(spec)
    if kind == COMPONENT_KIND_CONTROLLER:
        return check_controller_spec(spec)
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


# ---------------------------------------------------------------------------
# The second shape: reading a standalone failure mode
# ---------------------------------------------------------------------------


def component_kind(spec: Any, name: Any = None) -> str:
    """Which of :data:`COMPONENT_KINDS` a declaration states it is.

    An absent key is :data:`COMPONENT_KIND_FLOW`, which is what every document
    written before the key existed means: a reader that ignored it read exactly
    what this one reads for them.
    """
    if not isinstance(spec, dict):
        raise ComponentSpecError(
            f"A component declaration is a mapping, got {type(spec).__name__}"
        )
    kind = spec.get(COMPONENT_KIND_KEY, COMPONENT_KIND_FLOW)
    if kind not in COMPONENT_KINDS:
        raise ComponentSpecError(
            f"Component {name or spec.get('name')}: "
            f"{COMPONENT_KIND_KEY}={kind!r} is not one of {list(COMPONENT_KINDS)}"
        )
    return kind


def _mode_class(spec: dict, where: str) -> _ModeClass:
    """The :class:`_ModeClass` a declaration's ``cls`` names, or a refusal."""
    declared = spec.get("cls")
    mode_class = MODE_CLASSES.get(declared) if isinstance(declared, str) else None
    if mode_class is None:
        raise ComponentSpecError(
            f"{where}: cls={declared!r} is not a mode class this layer reads. "
            f"A two-state component names the class that carries its "
            f"occurrence, one of {', '.join(sorted(MODE_CLASSES))}. A class "
            f"carrying no law of its own (`ObjFM`, `ObjFailureMode`) has no "
            f"transition to build, and a MULTI-state degradation mode is not "
            f"the shape this kind names"
        )
    return mode_class


def _mode_order_vector(where: str, key: str, declared: Any) -> list:
    """A per-common-cause-order parameter vector, as one value per order.

    **A document has no tuples, and the engine reads one.** ``ObjMode2S``
    takes a TUPLE at an order to mean "the parameters of that order" and wraps
    anything else into a one-element one, so a document round-tripped through
    JSON carries ``[1.0, [0], [0]]`` where the mode was declared with one rate
    per order. A list at an order therefore MEANS what the tuple means, and
    the one parameter each law of this layer takes is read out of it.

    A vector of more than one parameter at an order is refused rather than
    truncated: none of the three laws here reads a second one, so a mode
    declaring one is a mode whose law is not the one its class carries.
    """
    if declared is None:
        return []
    if not isinstance(declared, (list, tuple)):
        declared = [declared]
    orders = []
    for index, entry in enumerate(declared):
        if isinstance(entry, (list, tuple)):
            if len(entry) != 1:
                raise ComponentSpecError(
                    f"{where} declares `{key}` order {index + 1} as "
                    f"{list(entry)!r}: an order carries the one parameter its "
                    f"law takes, and no law read here takes {len(entry)}"
                )
            entry = entry[0]
        orders.append(entry)
    return orders


def _mode_targets(spec: dict, vocabulary: _Vocabulary, where: str) -> list[str]:
    """The components a mode affects, refusing what has no counterpart."""
    targets = spec.get("targets")
    if targets is None and "targets" in vocabulary.accepted():
        raise ComponentSpecError(
            f"{where}: targets=None puts a mode in its SELF-HOSTED shape, one "
            f"automaton on the mode itself and no target at all. Here a mode "
            f"writes the attributes of the components it names, so a mode "
            f"naming none has nothing to write and no counterpart"
        )
    if (
        not isinstance(targets, (list, tuple))
        or not targets
        or not all(isinstance(target, str) for target in targets)
    ):
        raise ComponentSpecError(
            f"{where}: 'targets' is the non-empty list of component names the "
            f"mode affects, got {targets!r}"
        )
    return [str(target) for target in targets]


#: The three names an EVENT is built with when its declaration renames
#: nothing: its automaton, the state it is in once it has occurred, and the
#: state it is in until then. They are ``cod3s.ObjEvent``'s own defaults, and
#: :func:`pyraichu.plugins.muscadet._expand_objevent` BUILDS under them, which
#: is what makes the two layers spell an event alike. The builder holds its own
#: copy -- it is the thing that constructs them, this is the thing that reads a
#: declaration back -- so what the two share is the convention, not a constant.
#: Everything that READS an event resolves here.
_EVENT_AUTOMATON_NAME = "ev"
_EVENT_OCC_STATE = "occ"
_EVENT_NOT_OCC_STATE = "not_occ"


def _is_event(spec: Any) -> bool:
    """Whether a component declaration is an EVENT: self-hosted, so it names
    no target, writes no attribute, and holds one automaton on itself."""
    declared = spec.get("cls") if isinstance(spec, dict) else None
    mode_class = MODE_CLASSES.get(declared) if isinstance(declared, str) else None
    return mode_class is not None and mode_class.self_hosted


def event_automaton(spec: Any) -> tuple[str, set[str]] | None:
    """The automaton an EVENT declaration hosts, and the two states it holds.

    ``None`` for anything that is not an event, which is what makes this the
    one place the event's naming is written down: a condition leaf watching
    an event's state resolves through it, and so does an indicator declared
    on one (:mod:`pyraichu.muscadet_engine`).

    **The three names are the reason both can be honoured at all.** An event
    names its automaton ``event_aut_name`` and its states
    ``occ_state_name`` / ``not_occ_state_name``; ``cod3s.ObjEvent`` hands
    those three straight to its engine and
    :func:`pyraichu.plugins.muscadet._expand_objevent` builds them under the
    same three names. So the two layers spell an event's automaton and its
    states IDENTICALLY -- which is exactly what they do not do for a mode,
    whose automaton this layer names itself.
    """
    if not _is_event(spec):
        return None
    return (
        str(spec.get("event_aut_name") or _EVENT_AUTOMATON_NAME),
        {
            event_occurrence_state(spec),
            str(spec.get("not_occ_state_name") or _EVENT_NOT_OCC_STATE),
        },
    )


def event_occurrence_state(spec: Any) -> str | None:
    """The state an EVENT is in once it has OCCURRED, and ``None`` for
    anything that is not an event.

    :func:`event_automaton` says which two states an event holds and stops
    there, membership being all an indicator or a condition leaf needs. What
    needs this one is whatever has to reach the occurrence in particular, and
    a sequence TARGET is that: a trajectory ends at the state the feared event
    reaches, not at either of the two (:mod:`pyraichu.muscadet_engine`).

    Read off the declaration rather than assumed, ``cod3s``'s own helper
    hard-coding ``occ``: a modeller who renamed it would otherwise get a
    target on a state no automaton holds, and a campaign that stops at
    nothing.
    """
    if not _is_event(spec):
        return None
    return str(spec.get("occ_state_name") or _EVENT_OCC_STATE)


def derived_out_states(spec: Any) -> set[str]:
    """The STATES of the ok/nok pairs muscadet derives on the discrete outputs
    of one component declaration, and an empty set for every other one.

    ``create_default_out_automata`` asks muscadet for one two-state automaton
    per discrete output, named after the flow and holding ``{flow}_ok`` and
    ``{flow}_nok`` (``muscadet/obj.py``, ``set_flows``). This layer derives no
    such pair -- every automaton it builds comes from a declaration that needs
    one -- so what the key changes here is not the model but what can be
    OBSERVED on it, and that is where it is refused: an indicator naming one of
    these states is refused by name (:mod:`pyraichu.muscadet_engine`), and a
    document naming none loses nothing.

    Indexing the refusal on the observation rather than on the declaration is
    what lets a platform export through at all: muscadet writes the key only
    when it is ``True``, so every value it ever writes is one that asks for
    the pair, and the whole reference corpus carries it.

    Kept beside :func:`event_automaton` because the two answer one question
    between them -- which automata a component declaration is observable
    through -- and a reader of a state indicator asks both.
    """
    if not isinstance(spec, dict) or not spec.get(DERIVED_OUT_AUTOMATA_KEY):
        return set()
    states = set()
    for entry in spec.get("flows") or []:
        if not isinstance(entry, dict) or entry.get("cls") not in _DISCRETE_OUT_CLASSES:
            continue
        name = entry.get("name")
        if isinstance(name, str):
            states |= {f"{name}_ok", f"{name}_nok"}
    return states


def _mode_automaton(where: str, obj: str, watched: dict) -> str:
    """The automaton a reference to another two-state component's STATE names.

    An EVENT hosts its own, under the name it declares (:func:`event_automaton`).

    A mode over one target builds a single automaton, ``fm``. A common-cause
    mode builds one per combination (``fm__cc_1_2``), and a condition naming
    "the state of that mode" names no one of them, so it is refused rather
    than resolved to the first.
    """
    event = event_automaton(watched)
    if event is not None:
        return event[0]
    targets = len(watched.get("targets") or ["_"])
    if targets > 1:
        raise ComponentSpecError(
            f"{where} watches a state of the mode `{obj}`, which is a "
            f"common-cause mode over {targets} targets and therefore builds "
            f"one automaton per combination, not one. Name the combination's "
            f"automaton, or watch a variable the mode writes instead"
        )
    return "fm"


def _mode_leaf(
    where: str,
    leaf: Any,
    targets: list[str],
    modes: dict,
    volumes: dict | None = None,
) -> dict:
    """One condition leaf, as the plugin's condition tree names the same thing.

    Three rewrites, all of them the two layers naming one thing differently:

    - a leaf carrying no ``obj`` reads the mode's own target, which is what
      cod3s resolves it against. A mode over SEVERAL targets resolves it per
      combination, and one condition is shared by every combination here, so
      that case is refused rather than resolved to one of them. An EVENT has
      no target at all -- cod3s compiles its tree "with a system-wide
      resolution (no per-target ``obj_default``: events observe arbitrary
      components)" -- so there a leaf has to name what it watches, and the
      refusal says so rather than falling over an empty list;
    - a leaf whose ``obj`` is another two-state component and whose ``attr``
      names one of that component's STATES is a state reference. cod3s reaches
      a mode's states through the same ``attr`` key it reaches a variable
      with; here a state and a variable are two different references, and
      reading the first as the second would compare a state name to a boolean;
    - a leaf whose ``attr`` names the QUANTITY A VOLUME HOLDS, which muscadet
      spells ``{c}_qty`` and ``{c}_qty_{f}`` and this layer ``{c}_content``
      and ``{c}_content_{f}``. A mode armed on a low tank level is the most
      ordinary use a safety study makes of a volume, and the disagreement was
      already written down for the INDICATOR that names the same reading
      (:func:`capacity_content_variables`): a condition goes through that very
      function rather than through a second table of its own. Its other half
      answers here too, the variables muscadet creates and this layer has none
      of (:func:`capacity_absent_variables`), so a condition naming one is
      refused BY ITS NAME with what stands in its place.

    `volumes` is ``{component: (translated, absent)}`` for the components that
    hold a volume, keyed on the name the MODEL carries them under, as
    :func:`_mode_volumes` reads them off the document. Nothing rewrites on a
    component declaring no capacity, so a variable that happens to end in
    ``_qty`` elsewhere is left exactly as the document wrote it.
    """
    if not isinstance(leaf, dict):
        raise ComponentSpecError(
            f"{where}: a condition leaf is a mapping naming what it watches, "
            f"got {leaf!r}"
        )
    leaf = dict(leaf)
    if not leaf.get("obj"):
        if not targets:
            raise ComponentSpecError(
                f"{where} carries the leaf {leaf!r}, which names no `obj`. A "
                f"leaf with no object reads the mode's own target, and this "
                f"one names none: an event observes arbitrary components of "
                f"the system, so name the `obj` the leaf watches"
            )
        if len(targets) > 1:
            raise ComponentSpecError(
                f"{where} carries the leaf {leaf!r}, which names no `obj` and "
                f"therefore reads the mode's own target. This mode has "
                f"{len(targets)}, resolved one per common-cause combination, "
                f"and one condition is shared by all of them here: name the "
                f"`obj` the leaf watches"
            )
        leaf["obj"] = targets[0]

    held = (volumes or {}).get(leaf["obj"])
    if held is not None and isinstance(leaf.get("attr"), str):
        translated, absent = held
        unavailable = absent.get(leaf["attr"])
        if unavailable is not None:
            raise ComponentSpecError(
                f"{where} watches `{leaf['obj']}.{leaf['attr']}`, a capacity "
                f"variable muscadet creates and this layer has no attribute "
                f"for: {unavailable}"
            )
        if leaf["attr"] in translated:
            leaf["attr"] = translated[leaf["attr"]]

    watched = modes.get(leaf["obj"])
    if watched is not None and "attr" in leaf:
        states = _mode_state_names(watched)
        if leaf["attr"] not in states:
            family = (
                "an event" if event_automaton(watched) else "a standalone mode"
            )
            raise ComponentSpecError(
                f"{where} watches `{leaf['obj']}.{leaf['attr']}`, and "
                f"`{leaf['obj']}` is {family}: it holds no variable, only the "
                f"states {sorted(states)}"
            )
        leaf["automaton"] = _mode_automaton(where, leaf["obj"], watched)
        leaf["state"] = leaf.pop("attr")
    return leaf


def _mode_state_names(spec: dict) -> set[str]:
    """The two states a two-state component declaration builds its automaton with.

    Three families, three spellings, and an EVENT's is neither of the other
    two: it names them ``occ_state_name`` / ``not_occ_state_name`` and
    defaults the second to ``not_occ``, where a mode over targets defaults it
    to ``rep``. Read with the mode vocabulary alone, ``not_occ`` would be
    refused as a state the event has not and ``rep`` offered in its place.
    """
    event = event_automaton(spec)
    if event is not None:
        return event[1]
    return {
        str(spec.get("failure_state") or spec.get("occ_state") or "occ"),
        str(spec.get("repair_state") or spec.get("not_occ_state") or "rep"),
    }


def _mode_volumes(components: dict) -> dict[str, tuple[dict, dict]]:
    """The volume-holding components of a document, with the two readings a
    condition needs of each: what to rename, and what to refuse by name.

    Keyed on the name the MODEL carries the component under -- its own
    ``name``, not the key the document files it by -- because that is the name
    a condition leaf has to spell for anything downstream to resolve it, and it
    is what :func:`build_component` builds the component as.

    A component declaring no capacity is left out of the mapping entirely, so
    a lookup on it falls through and the document's own spelling stands.
    """
    volumes: dict[str, tuple[dict, dict]] = {}
    for name, entry in (components or {}).items():
        if not isinstance(entry, dict):
            continue
        translated = capacity_content_variables(entry)
        absent = capacity_absent_variables(entry)
        if translated or absent:
            volumes[str(entry.get("name") or name)] = (translated, absent)
    return volumes


def _mode_cond(
    where: str,
    key: str,
    declared: Any,
    targets: list[str],
    modes: dict,
    by_flow: bool,
    volumes: dict | None = None,
) -> Any:
    """A mode's condition, as the plugin's condition tree.

    A bare boolean passes through: ``True`` is "nothing gates it" and
    ``False`` is a direction that never fires, and the plugin reads both.
    Everything else is cod3s' structured tree, normalised to the OR-of-AND
    groups the plugin normalises to as well, with each leaf resolved.

    ``by_flow`` is the muscadet façade's dict shorthand: ``{"c1": True}``
    requires every named INPUT flow of every target to be fed with that value,
    which is one conjunction over the targets. It names a flow and never a
    variable, so no volume is read there: what it builds is a `{flow}_fed_in`
    reference, which both layers spell alike.
    """
    if declared is None or isinstance(declared, bool):
        return declared
    _refuse_callables(f"{where}.{key}", declared)

    if isinstance(declared, dict):
        if by_flow:
            return [
                [
                    {"obj": target, "attr": f"{flow}_fed_in", "value": value}
                    for flow, value in declared.items()
                    for target in targets
                ]
            ]
        groups = [[declared]]
    elif isinstance(declared, (list, tuple)):
        entries = list(declared)
        groups = (
            [entries]
            if entries and all(isinstance(entry, dict) for entry in entries)
            else [list(group) for group in entries]
        )
    else:
        raise ComponentSpecError(
            f"{where} declares `{key}`={declared!r}; a condition is a boolean, "
            f"a leaf, a list of leaves or a list of such groups"
        )

    return [
        [
            _mode_leaf(f"{where}: `{key}`", leaf, targets, modes, volumes)
            for leaf in group
        ]
        for group in groups
    ]


def _mode_flow_effects(
    where: str, key: str, declared: dict, targets: list[str], components: dict
) -> dict:
    """The muscadet façade's flow-pattern effects, as the variables they name.

    ``muscadet.ObjFailureMode`` names its target's FLOWS with a regular
    expression and the library resolves each match to what the flow offers a
    mode; ``cod3s.ObjFM`` names the variable exactly. Both spellings are
    load-bearing in models that exist, and the class is what tells them apart,
    so this is the one place the two are reconciled.

    What a discrete output offers is its availability gate, which this layer
    derives and a mode may hold down. A CONTINUOUS output offers a derating
    variable the mode itself owns, one per (automaton, output) pair; a mode
    declared ON a component reaches it here through `failure_effects`, and a
    standalone one has no such variable, so a pattern naming a continuous
    output is refused rather than silently applied to nothing.
    """
    resolved: dict[str, Any] = {}
    for target in targets:
        flows = (components.get(target) or {}).get("flows") or []
        discrete = [
            entry["name"]
            for entry in flows
            if isinstance(entry, dict)
            and entry.get("cls") in _DISCRETE_OUT_CLASSES
            and isinstance(entry.get("name"), str)
        ]
        continuous = [
            entry["name"]
            for entry in flows
            if isinstance(entry, dict)
            and entry.get("cls") == "FlowContinuousOut"
            and isinstance(entry.get("name"), str)
        ]
        for pattern, value in declared.items():
            named = [
                flow for flow in discrete if _pattern_names(where, key, pattern, flow)
            ]
            derated = [
                flow for flow in continuous if _pattern_names(where, key, pattern, flow)
            ]
            if derated:
                raise ComponentSpecError(
                    f"{where} declares the `{key}` pattern {pattern!r}, which "
                    f"names the continuous output(s) {derated} of `{target}`. "
                    f"A mode derates a continuous output through a variable it "
                    f"OWNS, one per automaton and output, and a standalone "
                    f"mode declares none here: declare the derating on the "
                    f"component's own `failure_modes` section instead"
                )
            if not named:
                raise ComponentSpecError(
                    f"{where} declares the `{key}` pattern {pattern!r}, which "
                    f"names no discrete output of `{target}`. Its outputs are "
                    f"{discrete or 'none'}"
                )
            for flow in named:
                variable = f"{flow}{AVAILABILITY_SUFFIX}"
                if resolved.setdefault(variable, value) != value:
                    raise ComponentSpecError(
                        f"{where} resolves `{key}` to two different values on "
                        f"`{variable}`; a mode applies ONE set of effects to "
                        f"every target it names"
                    )
    return resolved


def _mode_effects(
    where: str,
    key: str,
    declared: Any,
    targets: list[str],
    components: dict,
    by_flow: bool,
) -> dict:
    """A mode's effects, as ``{variable: value}`` on each of its targets."""
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise ComponentSpecError(
            f"{where} declares `{key}`={declared!r}; a standalone mode's "
            f"effects are a mapping of what it writes to the value it writes"
        )
    _refuse_callables(f"{where}.{key}", declared)
    if by_flow:
        return _mode_flow_effects(where, key, declared, targets, components)
    return {str(name): value for name, value in declared.items()}


def _mode_laws(spec: dict, mode_class: _ModeClass) -> tuple:
    """``(occurrence, return)`` law kinds a declaration draws from.

    Carried by the CLASS for the ``ObjFM`` façades and DECLARED for a plain
    ``ObjMode2S``, which is the one substantive difference between the two
    vocabularies: an ``ObjFMDelay`` IS a delay law, while an ``ObjMode2S``
    states which law it draws from.
    """
    if mode_class.laws[0] is not None:
        return mode_class.laws
    return tuple(
        (spec.get(key) or {}).get("cls") if isinstance(spec.get(key), dict) else None
        for key in ("occ_law", "not_occ_law")
    )


def _mode_param_names(spec: dict, laws: tuple, where: str) -> None:
    """Refuse a parameter-variable name disagreeing with the law it names.

    RAICHU bakes a law's value into the transition and declares no parameter
    variable, so the name carries nothing here -- but a declaration whose
    ``ObjFMDelay`` is bound to ``lambda`` contradicts itself, and a reader that
    ignored the name would build the delay the class says and not the rate the
    name says.

    Two spellings are accepted per direction, and both are somebody's default:
    the ``ObjFM`` façades name a parameter after the law (``ttf``, ``lambda``)
    and the engine names it after the direction and the law's own field
    (``occ_time``, ``occ_rate``). A document carries whichever its writer
    derived, and neither says anything more than the law already does.
    """
    for key, law, direction in (
        ("failure_param_name", laws[0], "occ"),
        ("occ_param_name", laws[0], "occ"),
        ("repair_param_name", laws[1], "not_occ"),
        ("not_occ_param_name", laws[1], "not_occ"),
    ):
        declared = spec.get(key)
        if declared is None or law not in _MODE_PARAM_NAMES:
            continue
        expected = {
            _MODE_PARAM_NAMES[law][direction != "occ"],
            f"{direction}_{_MODE_LAW_FIELD[law]}",
        }
        if len(list(declared)) != 1 or list(declared)[0] not in expected:
            raise ComponentSpecError(
                f"{where} declares {key}={declared!r} while its "
                f"{'occurrence' if direction == 'occ' else 'return'} is drawn "
                f"from a `{law}` law, whose parameter is named "
                f"{sorted(expected)}"
            )


def _mode_drop_inactive(spec: dict, laws: tuple, where: str) -> None:
    """Refuse ``drop_inactive_automata=False`` where it asks for something.

    muscadet writes the key on every common-cause mode whose orders were all
    built, which is the ORDINARY case, and writes ``False`` there because
    ``False`` rebuilds them all. Here an order with no active law builds no
    automaton and there is no spelling for building one that can never fire --
    so the value says nothing while every order carries an active law, which is
    exactly the case muscadet writes it in, and asks for something otherwise.
    """
    if spec.get("drop_inactive_automata") is not False:
        return
    for key, law in (
        ("failure_param", laws[0]),
        ("repair_param", laws[1]),
        ("occ_law", laws[0]),
        ("not_occ_law", laws[1]),
    ):
        declared = spec.get(key)
        if declared is None or law is None:
            continue
        values = (
            declared.get(_MODE_LAW_FIELD.get(law))
            if isinstance(declared, dict)
            else declared
        )
        for order, value in enumerate(values if isinstance(values, list) else [values]):
            value = value[0] if isinstance(value, (list, tuple)) and value else value
            if value is None or (law == "exp" and not (value or 0) > 0):
                raise ComponentSpecError(
                    f"{where} declares drop_inactive_automata=False while its "
                    f"`{key}` leaves common-cause order {order + 1} inactive "
                    f"({value!r}). Here an order with no active law builds no "
                    f"automaton, and there is no automaton that can never fire "
                    f"to build instead"
                )


def _mode_wire(spec: dict, mode_class: _ModeClass, where: str) -> dict:
    """The cod3s mode wire a declaration means, classified key by key.

    Two keys are dropped HERE rather than by :class:`_Vocabulary`, because
    what they say depends on the law the declaration draws from and an inert
    value is one the vocabulary knows in advance: the parameter variable names
    (:func:`_mode_param_names`) and the drop-inactive gate
    (:func:`_mode_drop_inactive`), each already checked against that law.

    Neither applies to a SELF-HOSTED event, which has no per-order parameter
    and no common-cause order: there the drop-inactive gate goes back to the
    vocabulary, which refuses it by what it asks for rather than dropping it
    after a check that would look at nothing.

    The event's own ``name`` is kept where every other family's is dropped:
    it is a constructor argument there, not the key the document files the
    component under, and the vocabulary is what says which.
    """
    laws = _mode_laws(spec, mode_class)
    _mode_param_names(spec, laws, where)
    # The component-level keys, minus the ones this vocabulary takes as
    # CONSTRUCTOR arguments: an event's `name` is one, and dropping it would
    # build an event with no name at all.
    dropped = set(_MODE_COMPONENT_KEYS) - set(mode_class.vocabulary.carried)
    dropped |= {
        "failure_param_name",
        "repair_param_name",
        "occ_param_name",
        "not_occ_param_name",
    }
    if not mode_class.self_hosted:
        _mode_drop_inactive(spec, laws, where)
        dropped.add("drop_inactive_automata")
    return _keywords(
        where,
        {key: value for key, value in spec.items() if key not in dropped},
        mode_class.vocabulary,
    )


def check_mode_spec(spec: Any, name: Any = None) -> str:
    """Validate a two-state component declaration, and return its name.

    The counterpart of :func:`check_spec` for the second shape a component
    declaration takes, and the same contract: everything checkable from the
    mapping ALONE, before anything is built. What needs the rest of the
    document -- that a target is declared, that an effect names a variable the
    target carries -- is :func:`build_document`'s, which is the scale that
    holds it.

    Raises
    ------
    ComponentSpecError
        On any fault reachable from the mapping alone.
    """
    if not isinstance(spec, dict):
        raise ComponentSpecError(
            f"A component declaration is a mapping, got {type(spec).__name__}"
        )
    declared_name = spec.get("name") or name
    mode_class = _mode_class(
        spec, f"Two-state component {declared_name or spec.get('cls')}"
    )
    where = f"{_MODE_FAMILY[mode_class.self_hosted]} {declared_name or spec.get('cls')}"
    vocabulary = mode_class.vocabulary

    accepted = set(vocabulary.accepted()) | _MODE_COMPONENT_KEYS
    unknown = sorted(set(spec) - accepted)
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        raise ComponentSpecError(
            f"{where}: unknown declaration key{plural} "
            f"{', '.join(repr(key) for key in unknown)}; a {spec['cls']} "
            f"declaration accepts {', '.join(sorted(accepted))}"
        )

    name_key = mode_class.name_key
    mode_name = spec.get(name_key)
    if not mode_name or not isinstance(mode_name, str):
        raise ComponentSpecError(
            f"{where}: {name_key!r} is what it is called, and it is a "
            f"non-empty string"
        )

    if mode_class.self_hosted:
        # An event has no target, so there is nothing to check there -- and
        # `cond` becomes what it is FOR: an event that watches nothing never
        # fires, which is a component declared for no reason rather than a
        # mode left ungated.
        if spec.get("cond") is None:
            raise ComponentSpecError(
                f"{where}: 'cond' is what the event watches, and an event "
                f"that watches nothing never fires"
            )
        if spec.get("cond_operator", "==") not in MODE_OPERATORS:
            raise ComponentSpecError(
                f"{where}: cond_operator={spec['cond_operator']!r} is not one "
                f"of {list(MODE_OPERATORS)}. A document carries the SPELLING "
                f"of a comparison, never the function it compiled to"
            )
    else:
        _mode_targets(spec, vocabulary, where)

    # The engine DERIVES the component's name from the two, and the document
    # files the mode under it: an entry whose key names something else is an
    # entry nothing in the document can reach. An event is self-hosted and its
    # name IS a constructor argument, so it carries no `target_name` and the
    # check is vacuous there.
    target_name = spec.get("target_name")
    if declared_name and target_name and f"{target_name}__{mode_name}" != declared_name:
        raise ComponentSpecError(
            f"{where}: the engine names this mode "
            f"{f'{target_name}__{mode_name}'!r}, from its 'target_name' and "
            f"its {name_key!r}, and the document files it under "
            f"{declared_name!r}. One of the two names something nothing else "
            f"can reach"
        )

    for key in mode_class.logic_keys:
        if key in spec and spec[key] not in MODE_LOGIC:
            raise ComponentSpecError(
                f"{where}: {key}={spec[key]!r} is not one of {list(MODE_LOGIC)}. "
                f"A document carries the NAME of a truth function, never the "
                f"function"
            )

    # Classified last, so a key refused for a reason of its own keeps that
    # reason: every check above is more specific than this one, which is what
    # the vocabulary answers about a key nobody else looked at.
    _mode_wire(spec, mode_class, where)

    if declared_name:
        return declared_name
    if not target_name:
        raise ComponentSpecError(
            f"{where}: it carries neither a 'name' nor a 'target_name', and "
            f"the name the document files a mode under is derived from the "
            f"second. Deriving one from the target LIST instead would invent a "
            f"name nothing else in the document reaches"
        )
    return f"{target_name}__{mode_name}"


def mode_object(spec: Any, components: dict | None = None, name: Any = None) -> dict:
    """The muscadet-plugin object a two-state component declaration means.

    The translation, in one place and reachable without building anything: a
    caller comparing what the two engines were handed reads this rather than a
    trajectory. What comes out is a ``pyraichu.plugins.muscadet`` ``ObjFM``,
    ``ObjFMInst`` or ``ObjEvent`` object, which is the same object a COD3S
    Platform study's modes and events are translated into -- one expansion for
    the two corpora rather than a second, lesser one written here.

    Parameters
    ----------
    spec : dict
        The declaration, ``kind`` = ``two_state_mode``.
    components : dict, optional
        The document's components, by name. What they answer is what the
        declaration alone cannot: which of the objects a condition watches is
        another two-state component, which flows a target holds, and which of
        them holds a VOLUME, whose level the two layers spell apart
        (:func:`_mode_volumes`). Given none, a condition on a tank level is
        left in muscadet's spelling and refused downstream, which is what a
        caller reading a mode out of its document gets.
    """
    from .importers.cod3s_platform import (
        TranslationError,
        event_object,
        failure_mode_object,
    )

    declared_name = check_mode_spec(spec, name)
    mode_class = _mode_class(spec, f"Two-state component {declared_name}")
    where = f"{_MODE_FAMILY[mode_class.self_hosted]} {declared_name}"
    components = components or {}
    targets = (
        [] if mode_class.self_hosted
        else _mode_targets(spec, mode_class.vocabulary, where)
    )

    modes = {
        entry_name: entry
        for entry_name, entry in components.items()
        if isinstance(entry, dict)
        and entry.get(COMPONENT_KIND_KEY) == COMPONENT_KIND_TWO_STATE_MODE
    }
    missing = [target for target in targets if components and target not in components]
    if missing:
        raise ComponentSpecError(
            f"{where}: it affects {missing}, which the document does not "
            f"declare. A mode names the components it reaches, and they are "
            f"components of the same system"
        )

    wire = _mode_wire(spec, mode_class, where)
    wire["cls"] = mode_class.wire
    # An event names none, and the key is not in its vocabulary: writing an
    # empty list there would put it in the shape a mode gating nothing has.
    if not mode_class.self_hosted:
        wire["targets"] = targets

    # The occurrence and return laws, one entry per common-cause order.
    for key in ("failure_param", "repair_param", "occ_param", "not_occ_param"):
        if key in wire:
            wire[key] = _mode_order_vector(where, key, wire[key])

    volumes = _mode_volumes(components)
    for key in ("failure_cond", "repair_cond", "occ_cond", "not_occ_cond", "cond"):
        if key in wire:
            wire[key] = _mode_cond(
                where, key, wire[key], targets, modes, mode_class.by_flow, volumes
            )

    for key in (
        "failure_effects",
        "repair_effects",
        "occ_effects",
        "not_occ_effects",
        "failure_effects_trans",
        "repair_effects_trans",
        "occ_effects_trans",
        "not_occ_effects_trans",
    ):
        if key in wire:
            wire[key] = _mode_effects(
                where, key, wire[key], targets, components, mode_class.by_flow
            )

    translate = event_object if mode_class.self_hosted else failure_mode_object
    try:
        obj = translate(wire)
    except (TranslationError, ValueError, KeyError) as error:
        raise ComponentSpecError(f"{where}: {error}") from error

    # The name the DOCUMENT files the component under, not the mode's own: it
    # is what a condition of another mode names, and what an indicator
    # reaches. For an event the two are the same string, its name being a
    # constructor argument, so this says nothing new and costs nothing.
    obj["name"] = declared_name
    for key in mode_class.logic_keys:
        if key in wire:
            obj[key] = wire[key]
    return obj


# ---------------------------------------------------------------------------
# The third shape: reading a CONTROLLER
# ---------------------------------------------------------------------------
#
# ``muscadet.ObjCtrl`` is a PEER of ``ObjFlow``, not a subclass of it: a flow
# transports a conserved quantity, a controller transports a reading or a
# signal, and nothing is allocated. It therefore has no ``add_flows`` and no
# ``set_flows()``, its two sections are passed to one constructor call, and it
# holds none of the ordering that makes :data:`DECLARATION_SECTIONS` what it is.
#
# Here the controller has a port of its own -- `pyraichu.plugins.controller` --
# and this module TRANSLATES a muscadet declaration into that port's object
# rather than reading the declaration twice. The two vocabularies say the same
# things and spell three of them differently, and every one of the three is a
# number a modeller tuned:
#
# - an observation input's SEED. muscadet writes one field per nature
#   (``level_default``, ``rate_default``, ``ratio_default``) and the port reads
#   one ``default``, the nature being already declared by ``kind``. Carried
#   through :data:`_CONTROL_IN_SEED`, so the field that says something for the
#   declared nature is the one read and the other three are inert;
# - a value output's SEED, which muscadet writes as ``level_default`` where the
#   port reads ``default``;
# - a republication's GAIN, which muscadet folds into the ``republish`` operator
#   it belongs to and leaves out of the output. An output NOTHING computes
#   carries the gain instead, under ``gain_default``, and there this layer has
#   no counterpart: an uncomputed output keeps the number it was created with
#   and holds no gain at all.
#
# ``emit`` passes VERBATIM: the emission grammar of the port is muscadet's own,
# operator for operator and field for field, so translating it would be a
# second spelling of one closed list and a place for the two to drift.

#: The sections a controller declaration carries, in the order they are
#: declared in: an output's grammar names an input, so the inputs come first.
CONTROLLER_SECTIONS = ("controls_in", "controls_out")

#: Every key a CONTROLLER declaration may carry. ``params`` is absent: a
#: controller is built by one constructor call, so there is no class template to
#: parameterise and no second declaration to add on top of one.
CONTROLLER_KEYS = frozenset(
    (
        "name",
        "cls",
        "label",
        "description",
        "metadata",
        COMPONENT_KIND_KEY,
        SOURCE_CLS_KEY,
    )
    + CONTROLLER_SECTIONS
)

#: The one class a controller declaration's ``cls`` may name. muscadet writes
#: the PEER class and keeps what it read under :data:`SOURCE_CLS_KEY`, because a
#: subclass declares its interfaces in its own constructor and would declare
#: them twice over the sections read back.
CONTROLLER_CLASS = "ObjCtrl"

#: Which of muscadet's four channel seeds declares the value an observation
#: input of each nature holds before its first sweep. The other three say
#: nothing at 0.0 and are refused above it: a seed that was dropped is a
#: reading that starts somewhere the document did not put it.
_CONTROL_IN_SEED = {
    "level": "level_default",
    "rate": "rate_default",
    "ratio": "ratio_default",
}

#: Every seed muscadet writes on an observation input, whatever its nature.
#: ``fill_default`` is in no entry of :data:`_CONTROL_IN_SEED`: a weighted fill
#: is a second reading of the volume a level comes from, and an observation
#: input reads ONE number.
_CONTROL_IN_SEEDS = ("level_default", "fill_default", "rate_default", "ratio_default")

#: What an observation input of each nature may carry. One vocabulary per
#: nature, derived from :data:`_CONTROL_IN_SEED` rather than written three
#: times: the seed of the declared nature is carried, the other three are inert
#: at muscadet's own default.
_CONTROL_IN = {
    kind: _Vocabulary(
        carried={
            "name": "name",
            "kind": "kind",
            "flows": "flows",
            "aggregate": "aggregate",
            seed: "default",
        },
        inert={other: 0.0 for other in _CONTROL_IN_SEEDS if other != seed},
    )
    for kind, seed in _CONTROL_IN_SEED.items()
}

#: What a BOOLEAN output may carry: muscadet's vocabulary is this layer's, key
#: for key.
_CONTROL_OUT_BOOL = _Vocabulary(
    carried={"name": "name", "kind": "kind", "default": "default", "emit": "emit"},
)

#: What a VALUE output may carry. muscadet declares one as a measurement
#: publication, so it writes the publication's own fields; here an output
#: publishes ONE number on ``{name}_level_out``, which is what lets a second
#: controller read it.
_CONTROL_OUT_VALUE = _Vocabulary(
    carried={
        "name": "name",
        "kind": "kind",
        "level_default": "default",
        "emit": "emit",
    },
    inert={
        "flows": [],
        "fill_default": 0.0,
        "ratio_default": 0.0,
        "gain_default": 1.0,
    },
    uncarried={
        "flows": (
            "the constituents a publication splits into, one published level "
            "each. A value output here publishes ONE number, which is what an "
            "observer reads and what makes a chain of controllers possible; a "
            "per-constituent publication has no counterpart. Declare a second "
            "output per constituent instead"
        ),
        "gain_default": (
            "the initial value of `{name}_level_gain` on an output NOTHING "
            "computes. Here an uncomputed output keeps the number it was "
            "created with and carries no gain at all, so a declared one would "
            "be dropped. A gain belongs to the `republish` that publishes "
            "through it, under its own `gain`"
        ),
    },
)

#: How an output of each nature is read, by the ``kind`` it declares.
_CONTROL_OUT = {"bool": _CONTROL_OUT_BOOL, "value": _CONTROL_OUT_VALUE}


def _controller_sections(spec: dict, name: str) -> tuple[list[dict], list[dict]]:
    """The two sections of a controller, translated entry by entry.

    Shared by :func:`check_controller_spec` and :func:`controller_object` so the
    refusal a check gives and the object a build hands the port come from one
    reading: a key accepted by one and dropped by the other is a declaration
    lost between two functions that agree.
    """
    from .plugins.controller import (
        CTRL_OUT_KINDS,
        MEASUREMENT_KINDS,
        _build_control_in,
        _build_control_out,
    )

    translated: dict[str, list[dict]] = {}
    for section, kinds, vocabularies, default_kind in (
        ("controls_in", MEASUREMENT_KINDS, _CONTROL_IN, "level"),
        ("controls_out", CTRL_OUT_KINDS, _CONTROL_OUT, "bool"),
    ):
        entries = []
        seen: set[str] = set()
        for index, entry in enumerate(_entries(spec, section, name)):
            iface = entry.get("name")
            if not iface or not isinstance(iface, str):
                raise ComponentSpecError(
                    f"Controller {name}: entry {index} of section "
                    f"'{section}' carries no 'name'. An interface name is what "
                    f"an output's grammar names its input by and what a "
                    f"connection reaches its message box through, so an "
                    f"unnamed one is reachable by nothing"
                )
            if iface in seen:
                raise ComponentSpecError(
                    f"Controller {name}: section '{section}' declares "
                    f"`{iface}` twice"
                )
            seen.add(iface)

            where = f"Controller {name}: {section} '{iface}'"
            kind = entry.get("kind", default_kind)
            if kind not in kinds:
                raise ComponentSpecError(
                    f"{where} declares kind={kind!r}, which is not one of "
                    f"{list(kinds)}"
                )
            entries.append(_keywords(where, entry, vocabularies[kind]))
        translated[section] = entries

    # Every translated entry, walked by the port's own doors. They are PURE --
    # they answer a dataclass and touch no engine -- so a batch of declarations
    # is sorted without raising a system, and the refusal is the SAME one either
    # way: what changes is only how early it arrives, and that it arrives as
    # this module's own rather than as the bare `ValueError` underneath. It is
    # what turns an aggregation this engine cannot compute, an unknown operator,
    # an inverted band, an empty combination and a `k` beside an `or` into
    # refusals reachable from the mapping alone.
    for section, door in (
        ("controls_in", _build_control_in),
        ("controls_out", _build_control_out),
    ):
        for entry in translated[section]:
            try:
                door(f"Controller {name}: {section}", entry)
            except ValueError as error:
                raise ComponentSpecError(str(error)) from error

    return translated["controls_in"], translated["controls_out"]


def check_controller_spec(spec: Any, name: Any = None) -> str:
    """Validate a controller declaration, and return its name.

    The counterpart of :func:`check_spec` for the third shape a component
    declaration takes, and the same contract: everything checkable from the
    mapping ALONE, before anything is built. What needs the rest of the
    document -- that a connection reaching an input names one this controller
    declares, that the publisher it reads holds the constituent -- is
    :func:`build_document`'s, which is the scale that holds it.

    Raises
    ------
    ComponentSpecError
        On any fault reachable from the mapping alone.
    """
    if not isinstance(spec, dict):
        raise ComponentSpecError(
            f"A component declaration is a mapping, got {type(spec).__name__}"
        )
    declared_name = spec.get("name") or name
    if not declared_name or not isinstance(declared_name, str):
        raise ComponentSpecError(
            f"Controller declaration without a 'name': {spec!r}"
        )

    unknown = sorted(set(spec) - CONTROLLER_KEYS)
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        raise ComponentSpecError(
            f"Controller {declared_name}: unknown declaration key{plural} "
            f"{', '.join(repr(key) for key in unknown)}; a controller "
            f"declaration accepts {', '.join(sorted(CONTROLLER_KEYS))}"
        )

    declared_class = spec.get("cls", CONTROLLER_CLASS)
    if declared_class != CONTROLLER_CLASS:
        raise ComponentSpecError(
            f"Controller {declared_name}: cls={declared_class!r} is not "
            f"{CONTROLLER_CLASS!r}. A controller declaration is always expanded "
            f"onto the peer class, a subclass declaring its own interfaces in "
            f"its constructor and so declaring them twice over the sections "
            f"read back; the class actually read travels under "
            f"{SOURCE_CLS_KEY!r}"
        )

    controls_in, controls_out = _controller_sections(spec, declared_name)
    if not controls_in and not controls_out:
        raise ComponentSpecError(
            f"Controller {declared_name}: it declares neither an observation "
            f"input nor a control output, which is an empty component. A "
            f"controller observes at least one quantity or publishes at least "
            f"one signal"
        )

    # Two interfaces on one message box. Not a duplicate NAME, which each
    # section already refuses: a value output called `x` publishes on
    # `x_level_out` and a BOOLEAN output called `x_level` publishes on the very
    # same box, so the two names differ and the box a connection reaches does
    # not. Left alone, one of the two would take every connection written for
    # the other, and nothing would say which.
    claimed: dict[str, str] = {}
    for entry, box in itertools.chain(
        ((entry, _controller_in_anchor(entry)[0]) for entry in controls_in),
        ((entry, _controller_out_box(entry)) for entry in controls_out),
    ):
        held = claimed.get(box)
        if held is not None:
            raise ComponentSpecError(
                f"Controller {declared_name}: interfaces `{held}` and "
                f"`{entry['name']}` both claim the message box `{box}`, so a "
                f"connection reaching it names neither one. Rename one of the "
                f"two"
            )
        claimed[box] = entry["name"]

    return declared_name


def controller_object(spec: Any, name: Any = None) -> dict:
    """The muscadet-plugin object a controller declaration means.

    The translation, in one place and reachable without building anything: a
    caller comparing what the two engines were handed reads this rather than a
    trajectory. What comes out is a ``pyraichu.plugins.muscadet`` ``ObjCtrl``
    object, which is the same object a COD3S Platform study's controllers are
    translated into -- one expansion for the two corpora rather than a second,
    lesser one written here.

    Parameters
    ----------
    spec : dict
        The declaration, ``kind`` = :data:`COMPONENT_KIND_CONTROLLER`.
    name : str, optional
        The name the DOCUMENT files the controller under, used when the entry
        itself carries none.

    Returns
    -------
    dict
        A ``{"type": "ObjCtrl", ...}`` object of :mod:`pyraichu.plugins.muscadet`.
    """
    declared_name = check_controller_spec(spec, name)
    controls_in, controls_out = _controller_sections(spec, declared_name)
    return {
        "type": "ObjCtrl",
        "name": declared_name,
        "controls_in": controls_in,
        "controls_out": controls_out,
    }


#: What muscadet names the VARIABLE behind a boolean controller output
#: (``muscadet.obj_ctrl.CtrlSignalOut.var_name``). The two layers agree on
#: everything else a controller exposes -- the message box ``{output}_out``,
#: the R44 endpoints, a value output's ``{output}_level`` -- and part on this
#: one alone: muscadet holds the signal in ``{output}_signal_out`` so that a
#: mode's unanchored regular expression has a name of its own to anchor on,
#: while this layer holds it in ``{output}`` and exports it on ``{output}_out``.
MUSCADET_SIGNAL_SUFFIX = "_signal_out"


def controller_signal_variables(spec: Any) -> dict[str, str]:
    """A controller's boolean outputs: muscadet's variable name, then this one's.

    The one spelling the two layers do not share on a controller, and the
    reason it needs naming at all: an indicator is written against the
    variable muscadet created, so a model observing ``high_signal_out`` is
    refused here as naming an attribute no component has. The refusal is loud,
    which is why this is a translation and not a second name -- the indicator
    keeps the name the document declared it under, and only what it POINTS AT
    is read in this layer's spelling.

    Derived from the port's own :class:`pyraichu.plugins.controller._ControlOut`
    rather than restated, so the day the attribute is named otherwise this
    follows rather than drifts. A value output is absent: both layers call it
    ``{output}_level``.

    Parameters
    ----------
    spec : dict
        A component declaration. Anything that is not a controller answers an
        empty mapping, so a caller sweeps a document without sorting it first.

    Returns
    -------
    dict
        ``{muscadet variable: this layer's attribute}``, empty for everything
        that is not a controller with a boolean output.
    """
    from .plugins.controller import CTRL_OUT_BOOL, _ControlOut

    if not isinstance(spec, dict):
        return {}
    if spec.get(COMPONENT_KIND_KEY) != COMPONENT_KIND_CONTROLLER:
        return {}

    found: dict[str, str] = {}
    for entry in spec.get("controls_out") or []:
        if not isinstance(entry, dict):
            continue
        output = entry.get("name")
        if not isinstance(output, str) or not output:
            continue
        if entry.get("kind", CTRL_OUT_BOOL) != CTRL_OUT_BOOL:
            continue
        attribute = _ControlOut(
            name=output, kind=CTRL_OUT_BOOL, default=False, node=None
        ).attribute
        found[f"{output}{MUSCADET_SIGNAL_SUFFIX}"] = attribute
    return found


#: What muscadet names the QUANTITY a capacity holds (``muscadet.capacity``):
#: ``{c}_qty`` for the whole volume and ``{c}_qty_{f}`` per constituent. This
#: layer holds the same quantity in ``{c}_content`` and ``{c}_content_{f}``
#: (:func:`pyraichu.muscadet._content_attribute`), and that is the only name of
#: a capacity the two spell apart: ``{c}_fill`` and ``{c}_fill_{f}`` are shared,
#: and so is ``{c}_ratio_{f}`` wherever both layers emit it.
MUSCADET_QUANTITY_SUFFIX = "_qty"

#: The capacity variables muscadet creates that this layer has NO attribute
#: for, by the suffix that names them, with what stands in their place. Read
#: with the constituent's name and the capacity's, which is why the entries are
#: templates rather than sentences: a refusal that names ``{c}_ratio_{f}`` and
#: then explains what replaces ``{f}`` is the whole point of refusing by name
#: rather than letting the load fail on "unknown attribute".
#:
#: Each was inventoried against a live pair of engines on 2026-09-15, on a
#: single-constituent volume and on a two-constituent one, and each is here for
#: a reason of its own rather than as a leftover:
#:
#: - ``inflow`` / ``outflow`` are muscadet's two hooks onto its allocation
#:   sweeps, written by the sweeps and read by the capacity's own equation.
#:   This layer integrates the content straight from the fed variables, which
#:   both layers name alike, so what replaces them is not a renaming but the
#:   pair the derivative is actually written over;
#: - ``ratio`` is absent on a SINGLE-constituent volume alone. muscadet emits
#:   it on every volume; here a volume publishes a ratio per constituent only
#:   when it holds more than one (:func:`pyraichu.muscadet._publishes_ratios`),
#:   the share of a volume in itself being identically one wherever it holds
#:   anything. On a volume holding several the name is SHARED and nothing here
#:   applies to it.
#:
#: ``serve_rate`` was a fourth entry and is one no longer: this layer publishes
#: the ceiling as a variable per held flow, under muscadet's own name for it
#: (:meth:`pyraichu.muscadet._Capacity.ceiling_of`), so that a failure mode has
#: something to clamp. The name is SHARED, an observation on it reaches the
#: attribute, and refusing it here would refuse a reading both engines answer.
_CAPACITY_ABSENT_VARIABLES = {
    "_inflow_": (
        "muscadet writes it from its allocation sweeps; this layer integrates "
        "the content from `{flow}_fed_in` minus `{flow}_fed_out`, which both "
        "layers name alike, so observe `{flow}_fed_in`"
    ),
    "_outflow_": (
        "muscadet writes it from its allocation sweeps; this layer integrates "
        "the content from `{flow}_fed_in` minus `{flow}_fed_out`, which both "
        "layers name alike, so observe `{flow}_fed_out`"
    ),
    "_ratio_": (
        "a volume holding a single constituent publishes no ratio here, that "
        "share being identically one wherever it holds anything; observe "
        "`{capacity}_content` for what it holds, or `{capacity}_fill` for how "
        "full it is. A volume holding SEVERAL constituents does publish "
        "`{capacity}_ratio_{flow}`, under that very name"
    ),
}


def _capacity_entries(spec: Any) -> list[tuple[str, list[str]]]:
    """The capacities of one component declaration, as ``(name, flows)``.

    The shared reading of the two functions below, and the reason a variable
    of a capacity is recognised by the capacity it belongs to rather than by
    its suffix: a component holding a flow variable that happens to end in
    ``_qty`` names no capacity and is left exactly as the document wrote it.

    ``flow`` is muscadet's single-flow short form of ``flows``, and a ``flows``
    entry is a name or a mapping carrying ``name`` and ``weight``
    (:meth:`pyraichu.muscadet.ObjFlow.add_capacity`). Anything it cannot read
    is skipped rather than refused: this answers an OBSERVATION, and the
    declaration itself is checked where it is built.
    """
    if not isinstance(spec, dict):
        return []
    if spec.get(COMPONENT_KIND_KEY, COMPONENT_KIND_FLOW) != COMPONENT_KIND_FLOW:
        return []

    found: list[tuple[str, list[str]]] = []
    for entry in spec.get("capacities") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        declared = entry.get("flows")
        if declared is None:
            declared = [] if entry.get("flow") is None else [entry.get("flow")]
        flows = []
        for held in declared if isinstance(declared, (list, tuple)) else []:
            held = held.get("name") if isinstance(held, dict) else held
            if isinstance(held, str) and held:
                flows.append(held)
        found.append((name, flows))
    return found


def capacity_content_variables(spec: Any) -> dict[str, str]:
    """A capacity's held quantity: muscadet's variable name, then this one's.

    The one spelling the two layers do not share on a capacity, and the
    observation that needs it most: the level of a tank is what a continuous
    study is written to watch. An indicator is named against the variable
    muscadet created, so a model observing ``reserve_qty`` is refused here as
    naming an attribute no component has -- loud, but refused, which is a model
    that runs on one engine and not the other.

    Derived from :func:`pyraichu.muscadet._content_attribute`, the function
    that CREATES the attribute, rather than restated, so the day it is named
    otherwise this follows rather than drifts. ``{c}_fill`` and ``{c}_fill_{f}``
    are absent because both layers spell them alike, and translating them would
    invent a disagreement.

    Parameters
    ----------
    spec : dict
        A component declaration. Anything that declares no capacity answers an
        empty mapping, so a caller sweeps a document without sorting it first.

    Returns
    -------
    dict
        ``{muscadet variable: this layer's attribute}``, empty for everything
        holding no volume.
    """
    found: dict[str, str] = {}
    for name, flows in _capacity_entries(spec):
        found[f"{name}{MUSCADET_QUANTITY_SUFFIX}"] = authoring._content_attribute(name)
        for flow in flows:
            found[f"{name}{MUSCADET_QUANTITY_SUFFIX}_{flow}"] = (
                authoring._content_attribute(name, flow)
            )
    return found


def capacity_absent_variables(spec: Any) -> dict[str, str]:
    """The capacity variables muscadet creates and this layer has none of, with
    what replaces each.

    The other half of :func:`capacity_content_variables`, and the half that
    exists so a refusal says something. Three of muscadet's capacity variables
    have no attribute of the same name here, and not one of them is a spelling
    disagreement: see :data:`_CAPACITY_ABSENT_VARIABLES` for what each is. An
    observation naming one is refused BY ITS NAME, saying what stands in its
    place, rather than reaching the engine and failing there on an attribute
    nobody can trace back to a declaration.

    ``{c}_ratio_{f}`` is conditional and the reason this reads the flows: a
    volume holding several constituents publishes it under that very name, so
    listing it there would refuse an observation this layer answers.

    Three and not four: ``{c}_serve_rate_{f}`` was here while the ceiling was
    inlined into the service expression, and left when it became a variable a
    failure mode can clamp. It is a SHARED name now, so an observation on it
    falls through to the attribute of that very name.

    Returns
    -------
    dict
        ``{muscadet variable: what replaces it}``, empty for everything holding
        no volume.
    """
    found: dict[str, str] = {}
    for name, flows in _capacity_entries(spec):
        ratios = authoring._publishes_ratios(flows)
        for suffix, replacement in _CAPACITY_ABSENT_VARIABLES.items():
            if suffix == "_ratio_" and ratios:
                continue
            for flow in flows:
                found[f"{name}{suffix}{flow}"] = replacement.format(
                    capacity=name, flow=flow
                )
    return found


# ---------------------------------------------------------------------------
# The SYSTEM scale: what a component declaration cannot carry
# ---------------------------------------------------------------------------

#: Version of the system declaration this module reads, muscadet's own
#: (``muscadet.declare.SYSTEM_SPEC_VERSION``). A document from a future MAJOR
#: is refused with its own number rather than half-built into a system nobody
#: can explain: the whole point of the document is that two engines read the
#: same thing, so a reader that guesses defeats it.
SYSTEM_SPEC_VERSION = "1.0.0"

#: The suffix pair of an ordinary flow connection, and the third channel a
#: discrete trigger flow offers. Ordered longest first, so ``f_trigger_in``
#: resolves to the trigger of ``f`` rather than to a flow called ``f_trigger``.
_IN_SUFFIXES = ("_trigger_in", "_in")

#: What a capacity publishes its level on, and what a measurement channel
#: reads it with. Recognising the pair is what tells a MEASUREMENT link from a
#: flow connection: both are message boxes ending in ``_out`` and ``_in``, and
#: read as a flow connection a level link would wire two flows that do not
#: exist.
_LEVEL_OUT_SUFFIX = "_level_out"
_LEVEL_IN_SUFFIX = "_level_in"

#: What a continuous flow publishes the rate it carries on (R38), and what a
#: ``kind="rate"`` observation input imports it with. The one box of a
#: controller input that is NOT ``{name}_level_in``.
_RATE_OUT_SUFFIX = "_rate_out"
_RATE_IN_SUFFIX = "_rate_in"


class SystemSpecError(ValueError):
    """A system declaration this layer refuses to build."""


def _connection_ends(entry: Any, index: int) -> tuple[str, str, str, str]:
    """One connection entry, read as its four strings."""
    if not isinstance(entry, dict):
        raise SystemSpecError(
            f"connection {index}: a connection is a mapping, got "
            f"{type(entry).__name__}"
        )
    missing = [
        key
        for key in ("source", "source_box", "target", "target_box")
        if not entry.get(key)
    ]
    if missing:
        raise SystemSpecError(f"connection {entry!r}: missing {missing}")
    return (
        str(entry["source"]),
        str(entry["source_box"]),
        str(entry["target"]),
        str(entry["target_box"]),
    )


def _controller_in_anchor(entry: dict) -> tuple[str, str, str | None]:
    """One observation input, as ``(message box, port, published alias)``.

    Three strings because the three do not agree, and muscadet's message box is
    why: a box carries SEVERAL variables -- the total level, the weighted fill,
    one level per constituent, the ratios -- so a single anchor pair names the
    box on both sides, a ratio importing on the very box its level comes on.
    RAICHU connects ATTRIBUTE to attribute, so the port names the ONE variable
    the input actually reads, and the third string is the alias the publisher
    exports that variable under: ``None`` when the anchor already names it.
    """
    name = entry["name"]
    kind = entry.get("kind") or "level"
    flows = list(entry.get("flows") or [])
    if kind == "rate":
        return f"{name}{_RATE_IN_SUFFIX}", f"{name}{_RATE_IN_SUFFIX}", None
    if kind == "ratio":
        alias = f"ratio_{flows[0]}"
        return f"{name}{_LEVEL_IN_SUFFIX}", f"{name}_{alias}_in", alias
    if flows:
        alias = f"level_{flows[0]}"
        return f"{name}{_LEVEL_IN_SUFFIX}", f"{name}_{alias}_in", alias
    return f"{name}{_LEVEL_IN_SUFFIX}", f"{name}{_LEVEL_IN_SUFFIX}", None


def _controller_out_box(entry: dict) -> str:
    """The message box one controller output publishes on.

    A boolean output exports one attribute on ``{name}_out``, which is the very
    box a discrete in-flow imports, so a controller drives a source with no
    adapter. A value output publishes a number on ``{name}_level_out``,
    indistinguishable from a capacity's own publication by whoever observes it,
    which is what makes a chain of controllers possible.
    """
    name = entry["name"]
    if (entry.get("kind") or "bool") == "bool":
        return f"{name}_out"
    return f"{name}{_LEVEL_OUT_SUFFIX}"


def _wire_controller_link(
    system: authoring.System,
    ends: tuple[str, str, str, str],
    controllers: dict,
) -> None:
    """Wire one connection at least one end of which is a controller.

    **By the RAW route, always**, which is what the README prescribes for a
    measurement link and what muscadet's own read-back now writes: a controller
    has only measurement links, it declares no flow, and a pair read as a flow
    connection would wire flows nobody declared -- on a controller, whose class
    holds no flow collection at all, it would not even reach that far.

    What the resolution does is name the ONE variable each end holds. muscadet's
    message box carries several and RAICHU connects attribute to attribute, so
    a level link into an input reading a constituent is a different pair from
    the anchor the document writes: see :func:`_controller_in_anchor`.
    """
    source, source_box, target, target_box = ends
    where = f"connection {source}.{source_box} -> {target}.{target_box}"

    publisher = controllers.get(source)
    observer = controllers.get(target)

    if publisher is not None:
        boxes = sorted(
            _controller_out_box(entry) for entry in publisher["controls_out"]
        )
        if source_box not in boxes:
            raise SystemSpecError(
                f"{where}: controller `{source}` publishes on "
                f"{boxes or 'no box at all'}, and not on `{source_box}`. A "
                f"boolean output exports `{{name}}_out` and a value output "
                f"publishes `{{name}}_level_out`"
            )

    target_port = target_box
    if observer is not None:
        anchors = {}
        for entry in observer["controls_in"]:
            box, port, alias = _controller_in_anchor(entry)
            anchors[box] = (entry, port, alias)
        if target_box not in anchors:
            raise SystemSpecError(
                f"{where}: controller `{target}` observes on "
                f"{sorted(anchors) or 'no box at all'}, and not on "
                f"`{target_box}`. An observation input imports a level or a "
                f"ratio on `{{name}}_level_in` and a rate on `{{name}}_rate_in`"
            )
        read, target_port, alias = anchors[target_box]
        if alias is not None:
            # A constituent's level and a constituent's share are published by
            # the VOLUME that holds it, each under an alias of its own inside
            # the box the anchor names. The publishing end has to name the same
            # variable, or the link would join the total to a share.
            if publisher is not None:
                raise SystemSpecError(
                    f"{where}: observation input `{read['name']}` reads the "
                    f"constituent {read.get('flows')}, and a controller "
                    f"publishes ONE number, which holds no constituent. Read "
                    f"the volume itself, or declare an input reading the whole"
                )
            if not source_box.endswith(_LEVEL_OUT_SUFFIX):
                raise SystemSpecError(
                    f"{where}: observation input `{read['name']}` reads the "
                    f"constituent {read.get('flows')}, which the volume holding "
                    f"it publishes on a `{{capacity}}_level_out` box, and "
                    f"`{source_box}` is not one"
                )
            source_box = f"{source_box[: -len(_LEVEL_OUT_SUFFIX)]}_{alias}_out"

    read_ports = [target_port]
    if (
        observer is None
        and publisher is not None
        and target_box.endswith(_LEVEL_IN_SUFFIX)
    ):
        # The other way round: a controller's number read by a FLOW component's
        # measurement channel. A channel reads a volume, so it materialises a
        # LEVEL and a weighted FILL, and muscadet's box carries both -- measured
        # on PyCATSHOO, an `ObjCtrl` value output publishing 7.5 leaves the
        # observer's level AND its fill at 7.5, both references connected. A
        # controller publishes no volume, so there is no second number to
        # publish: what makes the two sides agree is the ONE number reaching
        # BOTH references, which is what muscadet's single box does. Wired to
        # the level alone, the fill would read 0 where muscadet reads the level.
        channel = target_box[: -len(_LEVEL_IN_SUFFIX)]
        declared = next(
            (
                link
                for link in system.comp[target].measurements_in
                if link.name == channel
            ),
            None,
        )
        if declared is None:
            raise SystemSpecError(
                f"{where}: component `{target}` declares no measurement link "
                f"`{channel}`"
            )
        if declared.flows:
            # Refused where muscadet under-reads in silence: there the
            # constituent references are simply left unconnected and answer
            # their default, so a channel naming a substance reads zero of it
            # from an instrument that holds none.
            raise SystemSpecError(
                f"{where}: measurement link `{channel}` of `{target}` reads the "
                f"constituents {list(declared.flows)}, and a controller "
                f"publishes ONE number, which holds none. A constituent is a "
                f"substance a volume holds, so read the volume itself"
            )
        read_ports.append(f"{channel}_fill_in")

    for port in read_ports:
        system._connections.append(
            {
                "from": {"component": source, "port": source_box},
                "to": {"component": target, "port": port},
            }
        )


def _wire_connections(
    system: authoring.System, entries: list, controllers: dict | None = None
) -> None:
    """Wire every declared connection, refusing what has no counterpart.

    Three families of message-box pair reach this layer, and telling them
    apart is the whole of the work: they are all ``{something}_out`` joined to
    ``{something}_in``, and the wrong reading of a level link would connect
    two flows nobody declared.

    - ``{f}_out`` to ``{f}_in`` is an ordinary flow connection;
    - ``{f}_out`` to ``{f}_trigger_in`` is a trigger's third channel;
    - ``{c}_level_out`` to ``{ch}_level_in`` **anchors a measurement link**,
      which is not one connection but a family of them: the totals, one pair
      per constituent, and the ratios when the volume publishes them.
      :meth:`~pyraichu.muscadet.System.connect_measurement` emits that whole
      family from the pair alone, so the anchor is honoured and the rest of
      the family is then RECOGNISED rather than re-wired. Recognised against
      what the call actually emitted, never against a second guess at the
      naming convention: a document carrying a member the call does not emit
      is refused, and that is exactly the case where the two sides disagree
      about what the link is made of.

    A controller's links go to :func:`_wire_controller_link`, which reconciles
    the same box against the ports a controller holds.

    **The `flow` key is read by nothing here, and that is deliberate.**
    muscadet writes it on a pair that follows the flow convention AND really
    names a flow on both ends, and leaves it out otherwise, a measurement link
    included -- which is exactly the distinction the three suffix readings above
    make from the box names alone. A document carrying it and the same document
    without it therefore wire identically, so no stored document has to be
    reread and none is routed down the flow path for a name it happens to carry.

    `controllers` names the document's controllers, by the name the document
    files each under, as the plugin objects :func:`controller_object` answers.
    A connection reaching one goes to :func:`_wire_controller_link`, since a
    controller is no part of ``system.comp`` and the ports it holds are its
    own.
    """
    controllers = controllers or {}
    absorbed: set[tuple[str, str, str, str]] = set()
    deferred: list[tuple[str, str, str, str]] = []

    for index, entry in enumerate(entries or []):
        ends = _connection_ends(entry, index)
        source, source_box, target, target_box = ends
        for name in (source, target):
            if name not in system.comp and name not in controllers:
                raise SystemSpecError(
                    f"connection {entry!r}: `{name}` is not a declared "
                    f"component of system `{system.name}`"
                )

        if source in controllers or target in controllers:
            _wire_controller_link(system, ends, controllers)
            continue

        if source_box.endswith(_LEVEL_OUT_SUFFIX) and target_box.endswith(
            _LEVEL_IN_SUFFIX
        ):
            before = len(system._connections)
            capacity = source_box[: -len(_LEVEL_OUT_SUFFIX)]
            channel = target_box[: -len(_LEVEL_IN_SUFFIX)]
            try:
                system.connect_measurement(source, capacity, target, channel)
            except ValueError as error:
                raise SystemSpecError(f"connection {entry!r}: {error}") from error
            absorbed |= {
                (
                    source,
                    made["from"]["port"],
                    target,
                    made["to"]["port"],
                )
                for made in system._connections[before:]
            }
            continue

        deferred.append(ends)

    for source, source_box, target, target_box in deferred:
        if (source, source_box, target, target_box) in absorbed:
            # A member of a measurement link its anchor already emitted.
            continue
        if not source_box.endswith("_out"):
            raise SystemSpecError(
                f"connection {source}.{source_box} -> {target}.{target_box}: a "
                f"connection leaves an output box, whose name ends in '_out'"
            )
        flow_out = source_box[: -len("_out")]
        for suffix in _IN_SUFFIXES:
            if target_box.endswith(suffix):
                flow_in = target_box[: -len(suffix)]
                break
        else:
            raise SystemSpecError(
                f"connection {source}.{source_box} -> {target}.{target_box}: a "
                f"connection enters an input box, whose name ends in one of "
                f"{list(_IN_SUFFIXES)}"
            )
        if suffix == "_trigger_in":
            if flow_in != flow_out:
                raise SystemSpecError(
                    f"connection {source}.{source_box} -> {target}.{target_box}: "
                    f"a trigger is wired from the output of the SAME flow, and "
                    f"`{flow_out}` is not `{flow_in}`"
                )
            system.connect_trigger(source, target, flow_out)
        else:
            system.connect(source, flow_out, target, flow_in)


def check_system_spec(spec: Any) -> None:
    """Validate a system declaration, building nothing.

    Sorting a batch before paying the build, and the reason the version is
    checked HERE rather than deeper: a document from a major nobody reads is
    refused with its own number, at the door.

    Both shapes of component declaration are validated: a flow component and a
    standalone failure mode are entries of the same section, told apart by
    :data:`COMPONENT_KIND_KEY`.

    **The model level is an OPEN vocabulary**, unlike the component level:
    measured on 2026-09-17, a key nothing here knows is accepted by this
    function and dropped by :func:`build_document`, where an unknown key on a
    component is refused by name against :data:`COMPONENT_KEYS`. So
    :data:`~pyraichu.indicators.GENERATED_INDICATORS` needed no door opened to
    be read, only a reader; what it does need is that its VALUE be checked
    here, since a key read by truthiness alone would take the string
    ``"false"`` for an instruction to observe everything.
    """
    if not isinstance(spec, dict):
        raise SystemSpecError(
            f"a system declaration is a mapping, got {type(spec).__name__}"
        )
    generated_indicators(spec, refuse=SystemSpecError)
    version = spec.get("version")
    if version is None:
        raise SystemSpecError("a system declaration carries a 'version'")
    major = str(version).split(".")[0]
    if major != SYSTEM_SPEC_VERSION.split(".")[0]:
        raise SystemSpecError(
            f"system declaration version {version!r} is not readable here, "
            f"which reads {SYSTEM_SPEC_VERSION.split('.')[0]}.x"
        )
    components = spec.get("components")
    if not isinstance(components, dict):
        raise SystemSpecError(
            "'components' is a mapping of name to component declaration"
        )
    for name, component in components.items():
        check_spec({**component, "name": component.get("name", name)})
    for index, entry in enumerate(spec.get("connections") or []):
        source, _, target, _ = _connection_ends(entry, index)
        for side, named in (("source", source), ("target", target)):
            if named not in components:
                raise SystemSpecError(
                    f"connection {entry!r}: {side} {named!r} is not a declared "
                    f"component"
                )


def _split_kinds(spec: dict) -> tuple[dict, dict, dict]:
    """The document's components, split into the three shapes they take.

    Named ``(flows, controllers, modes)`` and returned in that order because it
    is also the order they have to be BUILT in, which is not a preference:

    - a controller is wired to the flow components and reads what they publish,
      so those exist first, exactly as a mode's targets do;
    - a mode resolves every effect it declares against every component it names,
      and a controller is one of them -- a mode moves a threshold or blinds a
      signal (R44) -- so the controllers come before the modes or a mode
      reaching one names an attribute the document does not yet hold.

    muscadet's own ``build_system`` orders the first two the same way, for the
    same reason.
    """
    components = spec.get("components")
    if not isinstance(components, dict):
        raise SystemSpecError(
            "'components' is a mapping of name to component declaration"
        )
    buckets: dict[str, dict] = {kind: {} for kind in COMPONENT_KINDS}
    for name, component in components.items():
        declared = {**component, "name": component.get("name", name)}
        buckets[component_kind(declared)][name] = declared
    return (
        buckets[COMPONENT_KIND_FLOW],
        buckets[COMPONENT_KIND_CONTROLLER],
        buckets[COMPONENT_KIND_TWO_STATE_MODE],
    )


def build_system(
    spec: Any,
    system: authoring.System | None = None,
    classes: dict[str, type] | None = None,
) -> authoring.System:
    """Build the FLOW SYSTEM a declaration held in data describes.

    The system-scale counterpart of :func:`build_component`, and the reader of
    ``muscadet.declare.system_spec``: what it adds over reading each component
    is the part no component knows, which is how they are wired.

    Components first, then the wiring: not a preference, a connection needs
    both its ends to exist.

    **A standalone failure mode is not part of what this builds**, and is
    refused rather than dropped. It holds no flow, it is wired to nothing, and
    it writes attributes of components the system has yet to build: it belongs
    to the DOCUMENT and not to the flow graph, which is why
    :func:`build_document` is the entry point that carries it. A caller
    reaching a system for its wiring gets the system; one reaching a model to
    run gets the document.

    What is deliberately NOT here is the observation: an indicator is named on
    the DOCUMENT, and this layer's authoring surface emits its own from the
    variables it generated. Reconciling the two is the business of whoever
    turns the built system into a model to run, which is
    :mod:`pyraichu.muscadet_engine`.

    Parameters
    ----------
    spec : dict
        A declaration produced by ``muscadet.declare.system_spec``, or written
        by hand.
    system : pyraichu.muscadet.System, optional
        An existing system to fill. Given one, the caller owns its naming.
    classes : dict, optional
        The component classes a declaration's ``cls`` may name.

    Returns
    -------
    pyraichu.muscadet.System

    Raises
    ------
    SystemSpecError
        For a document this layer cannot read, and for a wiring with no
        counterpart here.
    ComponentSpecError
        Through :func:`build_component`, for a component declaration.
    """
    check_system_spec(spec)
    flows, controllers, modes = _split_kinds(spec)
    if modes:
        raise SystemSpecError(
            f"the document declares the standalone failure mode(s) "
            f"{sorted(modes)}, which hold no flow and are no part of the "
            f"system's wiring: they are expanded onto the components they name, "
            f"once those exist. Build the whole document with "
            f"`pyraichu.declare.build_document`, or drop the modes to build "
            f"the flow system alone"
        )
    if controllers:
        raise SystemSpecError(
            f"the document declares the controller(s) {sorted(controllers)}, "
            f"which are PEERS of a flow component and not flow components: they "
            f"hold no flow, so this system has no place to put one, and their "
            f"wiring joins ports a flow component does not declare. Build the "
            f"whole document with `pyraichu.declare.build_document`, or drop "
            f"the controllers to build the flow system alone"
        )
    return _build_flow_system(flows, spec, system, classes)


def _wired_rate_channels(flows: dict, connections: Any) -> dict[str, set[str]]:
    """The continuous flows a connection asks to publish their rate on.

    **muscadet has no key to write here, and that is the whole reason this
    reading exists.** Its ``add_mb`` gives EVERY continuous flow a rate
    observation box, unconditionally, so a modeller wiring a sensor onto
    ``{f}_rate_out`` writes nothing anywhere -- there was never a choice to
    record. This layer publishes that channel only where ``publish_rate`` is
    declared, for the reason the key exists: a port and an equation on every
    flow of every model would buy the few an observer actually reads. A muscadet
    document therefore reached the engine naming a port nothing had created,
    and was refused on `connection endpoint ... does not exist`, which names
    neither the flow nor the key nor what to declare.

    The document does carry the information, though, one level up: it carries
    the CONNECTION. A flow whose rate a measurement link names is a flow
    somebody observes, which is exactly the question ``publish_rate`` asks. So
    the publication is derived from the wiring rather than guessed from a
    convention, and RAICHU still publishes fewer channels than muscadet: only
    the observed ones.

    The observer's end needs nothing derived: ``{name}_rate_in`` is a
    controller's own declared observation input (``kind="rate"``), and a
    controller that declares none simply has no such box.

    One ambiguity, and it is read rather than assumed: ``{f}_rate_out`` is also
    the ORDINARY output port of a flow literally called ``{f}_rate``. A
    component declaring one is left alone, so a document that always meant the
    plain connection keeps meaning it.

    Returns
    -------
    dict
        ``{component: {flow, ...}}``, empty for a document wiring no rate.
    """
    declared: dict[str, dict[str, set[str]]] = {}
    for key, entry in (flows or {}).items():
        if not isinstance(entry, dict):
            continue
        held: dict[str, set[str]] = {"continuous": set(), "any": set()}
        for flow in entry.get("flows") or []:
            if not isinstance(flow, dict) or not isinstance(flow.get("name"), str):
                continue
            held["any"].add(flow["name"])
            if flow.get("cls") in _CONTINUOUS_CLASSES:
                held["continuous"].add(flow["name"])
        declared[str(entry.get("name") or key)] = held

    wired: dict[str, set[str]] = {}
    for entry in connections or []:
        if not isinstance(entry, dict):
            continue
        source, box = entry.get("source"), entry.get("source_box")
        if not isinstance(source, str) or not isinstance(box, str):
            continue
        if not box.endswith(_RATE_OUT_SUFFIX):
            continue
        held = declared.get(source)
        if held is None:
            continue
        flow = box[: -len(_RATE_OUT_SUFFIX)]
        if flow not in held["continuous"] or f"{flow}_rate" in held["any"]:
            continue
        wired.setdefault(source, set()).add(flow)
    return wired


def _publish_wired_rates(declared: dict, wired: dict[str, set[str]]) -> dict:
    """One component declaration, with the rate channels its wiring asks for.

    What the channel CARRIES is the delivered quantity, never the capability:
    muscadet's own rate observation box exports ``var_fed``, the total the flow
    delivers, so a derived channel reads what the reference engine would have
    published there. ``publish_rate`` stays declarable and WINS when it is
    declared, which is how a model asks for the capability instead -- the
    quantity a regulator wants, and the one no document could ask for by
    wiring alone.

    Both directions are eligible and at most one is published: a component
    declaring the same name as a continuous input AND a continuous output
    would put two publishers on one port, so the OUTPUT is taken, being the
    end a measurement reads a delivery from.

    The declaration is copied rather than written into: it belongs to the
    caller's document, and a reader that edited it would leave a second build
    of the same document carrying a key the modeller never wrote.
    """
    names = wired.get(str(declared.get("name") or ""))
    if not names:
        return declared

    entries = list(declared.get("flows") or [])
    eligible: dict[str, list[int]] = {}
    for index, flow in enumerate(entries):
        if not isinstance(flow, dict) or flow.get("name") not in names:
            continue
        if flow.get("cls") in _CONTINUOUS_CLASSES:
            eligible.setdefault(flow["name"], []).append(index)

    published: set[int] = set()
    for indexes in eligible.values():
        # A declared `publish_rate` is the modeller's word and is left alone,
        # `false` included: it is how a model asks for the capability instead
        # of the delivery, and how it declines the channel outright. Read over
        # the whole name, both directions at once: one of the two publishing
        # already is what the second publisher would collide with.
        if any(entries[index].get("publish_rate") is not None for index in indexes):
            continue
        published.add(
            next(
                (
                    index
                    for index in indexes
                    if entries[index].get("cls") == "FlowContinuousOut"
                ),
                indexes[0],
            )
        )
    if not published:
        return declared
    return {
        **declared,
        "flows": [
            {**flow, "publish_rate": authoring.RATE_DELIVERED}
            if index in published
            else flow
            for index, flow in enumerate(entries)
        ],
    }


def _build_flow_system(
    flows: dict,
    spec: dict,
    system: authoring.System | None,
    classes: dict[str, type] | None,
    controllers: dict | None = None,
) -> authoring.System:
    """The flow components of a document, built and wired.

    `controllers` are the document's controller declarations, which this system
    does not build: they are named here because the WIRING is the system's, and
    a connection reaching a controller has to resolve against what that
    controller declares. See :func:`_wire_connections`.

    The system is built with the document's own answer on the generated
    indicator set, so :meth:`~pyraichu.muscadet.System.build_dict` writes the
    document this declaration describes and not the one a class-based author
    would have written. A system the CALLER supplied is left as the caller set
    it: it is the caller's system, and this reader fills it rather than
    configures it.
    """
    if system is None:
        system = authoring.System(
            name=spec.get("name") or "system",
            generated_indicators=generated_indicators(
                spec, refuse=SystemSpecError
            ),
        )

    wired = _wired_rate_channels(flows, spec.get("connections"))
    for declared in flows.values():
        _refuse_ungated_modes(declared)
        build_component(
            system, _publish_wired_rates(declared, wired), classes=classes
        )

    _wire_connections(system, spec.get("connections"), controllers or {})
    return system


def build_document(
    spec: Any,
    system: authoring.System | None = None,
    classes: dict[str, type] | None = None,
) -> dict[str, Any]:
    """The RAICHU model document a muscadet system declaration describes.

    The whole of the reading, at the scale the document has: the flow
    components, wired, plus the controllers and the standalone failure modes
    expanded onto them. Reachable without running anything, so a caller
    comparing what the two engines were handed reads this rather than a
    trajectory.

    **The order is the point**, and it runs in three steps rather than two.
    A standalone mode resolves every effect it declares against every component
    it names, and names a condition that may watch a variable or an automaton
    state of any of them, so the components are built first and the modes
    expanded onto the result. Declared the other way round, a mode would resolve
    its effects against components that do not exist yet. The controllers land
    BETWEEN the two, for the same reason read twice: they read what the flow
    components publish, and a mode reaches INTO one -- a threshold it moves, a
    signal it blinds (R44) -- so a mode expanded before them would name an
    attribute the document does not yet hold. muscadet's own ``build_system``
    orders the flow components first for the first of those reasons.

    Both are expanded through :mod:`pyraichu.plugins.muscadet`, whose ``ObjFM``
    and ``ObjCtrl`` objects are what a COD3S Platform study's modes and
    controllers already become: one expansion for the two corpora, rather than
    a second and lesser one written for this reader alone.

    Parameters
    ----------
    spec : dict
        A declaration produced by ``muscadet.declare.system_spec``.
    system : pyraichu.muscadet.System, optional
        An existing system to fill with the flow components.
    classes : dict, optional
        The component classes a declaration's ``cls`` may name.

    Returns
    -------
    dict
        The model document, under the format envelope when the system carries
        a construct that needs one, exactly as
        :meth:`pyraichu.muscadet.System.build_dict` answers it.

    Raises
    ------
    SystemSpecError, ComponentSpecError
        For anything the document declares and this layer cannot carry, named.
    """
    from . import seal
    from .plugins.muscadet import MuscadetPlugin, _carry_grafts

    check_system_spec(spec)
    flows, controllers, modes = _split_kinds(spec)
    # Translated BEFORE the wiring, because the wiring needs them: a connection
    # reaching a controller resolves against the ports that controller holds,
    # and those are the object's, not the declaration's.
    #
    # Keyed by the name the OBJECT carries, never by the document's key for it.
    # An entry filed under one name and declaring another builds a component
    # under the second, so a connection naming the first reaches nothing: keyed
    # this way it is refused as an undeclared component, which is the refusal
    # `build_component` already gives a flow entry for the same disagreement.
    built_controllers = [
        controller_object(entry, name=name) for name, entry in controllers.items()
    ]
    controller_objects = {obj["name"]: obj for obj in built_controllers}
    built = _build_flow_system(flows, spec, system, classes, controller_objects)
    if not modes and not controllers:
        return built.build_dict()

    # Pass 1: the flow components, so a controller has something to read and a
    # mode something to resolve its targets against. The evaluation order this
    # answers is thrown away: it closes over the components declared so far,
    # and neither the controllers nor the modes are among them yet.
    components, _ = built.generate()
    pristine = copy.deepcopy(components)
    flow_count = len(components)
    body: dict[str, Any] = {
        "name": built.name,
        "components": components,
        "connections": built._connections,
        "indicators": [],
        "targets": [],
        # Written here and not only at the end, because the plugin
        # finalisation below reads the model to decide the same thing: one
        # key, read by whichever writer gets there first.
        GENERATED_INDICATORS: built.generated_indicators,
    }

    # Every component a mode may NAME, the controllers included. A controller
    # is a component of the system like any other, and R44 makes three of its
    # endpoints -- a threshold, a publication's gain, a boolean output's
    # availability -- ordinary variables a `cod3s.ObjFM` writes by their exact
    # name. Left out of this mapping, a mode reaching one was refused as
    # affecting a component "which the document does not declare", which is
    # the blinded-instrument scenario refused at the door.
    declared = {**flows, **controllers, **modes}
    objects = list(built_controllers)
    objects += [
        mode_object(entry, declared, name=name) for name, entry in modes.items()
    ]
    plugin = MuscadetPlugin()
    for obj in objects:
        _expand_object(plugin, obj, body)

    # Checked once every mode has been expanded, and not one at a time: a mode
    # may watch a mode the document lists AFTER it, and its automaton does not
    # exist until that one is expanded.
    persistent_gates = _persistent_availability_gates(flows)
    reinitialized_gates = _availability_gates(flows) - persistent_gates
    for obj in objects:
        if obj["type"] != "ObjCtrl":
            _refuse_unreachable_references(obj, body)
            _refuse_a_latched_production_a_condition_also_writes(obj, body)
            _refuse_a_held_write_on_a_persistent_gate(obj, persistent_gates)
            _refuse_a_pulse_on_a_reinitialized_gate(obj, reinitialized_gates)

    # The two whole-model closures the plugin's own expansion runs, and for
    # reasons that hold here identically: an availability gate the document
    # asked to MEMORISE cannot also be written by a mode, and two modes
    # writing one attribute each undo the other's failure unless their
    # reinitialization effects are folded into one writer. Neither is
    # decidable one mode at a time.
    plugin.finalize_model(body, objects)

    # Pass 2: the continuous network, closed over the controller and mode
    # components. The sweep order must cover every declared step EXACTLY, and a
    # mode's effect function is one, as is each of a controller's readings and
    # publications, so an order derived before they existed refuses the model it
    # belongs to. `generate` takes them as `foreign` for that, and lists their
    # steps LAST, which is downstream of every level a controller can read.
    foreign = body["components"][flow_count:]
    rebuilt, order = built.generate(foreign=foreign)
    for placeholder, original, final in zip(components[:flow_count], pristine, rebuilt):
        # What a mode wrote INTO a target component during pass 1 -- the
        # mirror automaton an `external` mode grafts -- is not in the fresh
        # build, which answers the declaration alone. The difference between
        # the placeholder and its pristine copy IS the graft.
        _carry_grafts(placeholder, original, final)
    body["components"] = rebuilt + foreign
    # The flow components' own indicators FIRST, then what the objects
    # expanded above emitted, MERGED rather than concatenated, by the rule
    # every writer of an indicator goes through. The first of the two is
    # emitted only where the document asked for it
    # (:data:`~pyraichu.indicators.GENERATED_INDICATORS`); what an object
    # emits for the attributes it generated is its own, and is written
    # whatever the document answered on that key.
    #
    # Reachable, and not a precaution: an indicator's name is FLATTENED to
    # `{component}_{variable}`, so two different observations meet as soon as
    # one component's name ends where the other's variable begins. A tank
    # `TANK` whose capacity `level` publishes `level_content` is observed as
    # `TANK_level_content`, and so is the signal `content` of a controller
    # named `TANK_level` -- naming a controller after the level it watches
    # being the ordinary way to name one. Concatenated, the document came out
    # carrying that name twice and the engine refused it on `duplicate
    # indicator name`, naming the indicator and neither writer.
    body["indicators"] = merge_indicators(
        built.indicators(rebuilt) if built.generated_indicators else [],
        body["indicators"],
        collision=lambda name, emitted, already: SystemSpecError(
            f"indicator {name!r} is emitted on {emitted} by an object of the "
            f"document while the rebuilt flow components already observe "
            f"{already} under that name"
        ),
    )

    if order is None:
        return body
    body.update(authoring.model_level_keys(order))
    # Sealed the way `build_dict` seals: the required-feature list is derived
    # from the body, never composed, so it cannot lag what the body holds.
    return seal(body)


#: How a plugin object is named in a refusal, by its ``type``. The object is the
#: engine's vocabulary and the declaration is the modeller's, and it is the
#: declaration a modeller has to change: an ``ObjCtrl`` refusal that said
#: "ObjCtrl" would name a class the document never mentions.
_OBJECT_SUBJECTS = {"ObjCtrl": "Controller"}


def _object_where(obj: dict) -> str:
    """What a refusal calls one expanded object: an event is not a mode, and a
    controller is neither."""
    family = _OBJECT_SUBJECTS.get(
        obj.get("type"), _MODE_FAMILY[obj.get("type") == "ObjEvent"]
    )
    return f"{family} {obj.get('name')}"


def _refuse_unreachable_references(obj: dict, body: dict) -> None:
    """Refuse a mode naming an attribute or a state nothing in the model holds.

    A standalone mode WRITES the attributes its effects name and READS the ones
    its condition names, and the two layers do not generate the same ones on a
    component. Left to the engine, either is a model refused for an unknown
    attribute, in a message carrying neither the declaration nor the mode that
    asked for it -- and the mode is exactly what a modeller has to change.

    An EVENT only reads: it has no effect, and its one condition is ``cond``.
    It is checked the same way, and it is the case that needs it most -- an
    event observes arbitrary components, so nothing upstream has already
    checked that the one it names exists.
    """
    attributes = {
        component["name"]: {
            entry["name"] for entry in component.get("attributes") or []
        }
        for component in body.get("components") or []
    }
    states = {
        component["name"]: {
            (automaton["name"], state)
            for automaton in component.get("automata") or []
            for state in automaton.get("states") or []
        }
        for component in body.get("components") or []
    }
    where = _object_where(obj)

    for key in ("failure_effects", "repair_effects"):
        for variable in obj.get(key) or {}:
            for target in obj.get("targets") or []:
                held = attributes.get(target)
                if held is None or variable in held:
                    continue
                gates = sorted(a for a in held if a.endswith(AVAILABILITY_SUFFIX))
                raise ComponentSpecError(
                    f"{where} declares `{key}` on `{target}.{variable}`, which "
                    f"this layer's `{target}` does not carry. What it does "
                    f"carry that a mode writes is "
                    f"{gates or 'no availability gate'}"
                )

    for key in ("failure_cond", "repair_cond", "cond"):
        for leaf in _condition_leaves(obj.get(key)):
            named = leaf.get("obj")
            if named not in attributes:
                raise ComponentSpecError(
                    f"{where}: its `{key}` watches `{named}`, which the model "
                    f"does not hold"
                )
            if "attr" in leaf and leaf["attr"] not in attributes[named]:
                raise ComponentSpecError(
                    f"{where}: its `{key}` watches `{named}.{leaf['attr']}`, "
                    f"which this layer's `{named}` does not carry"
                )
            reference = (leaf.get("automaton"), leaf.get("state"))
            if "attr" not in leaf and reference not in states[named]:
                raise ComponentSpecError(
                    f"{where}: its `{key}` watches the state "
                    f"`{named}.{reference[0]}.{reference[1]}`, which the model "
                    f"does not hold"
                )


def _refuse_a_latched_production_a_condition_also_writes(obj: dict, body: dict) -> None:
    """Refuse a mode latching a production its own flow's condition rewrites.

    A mode writing ``{flow}_prod_available`` is how a DORMANT output is
    started, and it is carried exactly: with no production condition declared,
    nothing else writes the variable and the latch is the only writer there is.

    Declare a condition as WELL, and muscadet has two writers on one variable
    resolved by the order events happen in -- the production method fires when
    an operand changes, the mode's effect fires when the mode does, and
    whichever ran last stands. This engine has no such ordering: an effect is
    HELD, re-evaluated to a fixpoint for as long as the mode's state lasts,
    and the expansion completes it with the rest branch reinitialization would
    have restored. The two writers then contradict each other at every
    evaluation and the held one wins, so an output whose condition holds reads
    as unproduced until the mode fires -- measured against PyCATSHOO on a
    source-fed flow: the reference produces from t = 0, this engine from the
    mode's own date.

    Refused by name rather than answered differently, for the reason the
    persistent availability gate is (see
    :func:`~pyraichu.plugins.muscadet._refuse_a_held_write_on_a_persistent_gate`):
    a divergence a study would have no way of noticing is worse than a model
    it cannot run.
    """
    writers = {
        component.get("name"): {
            function.get("name")
            for function in component.get("sensitive_functions") or []
        }
        for component in body.get("components") or []
    }
    for key in ("failure_effects", "repair_effects"):
        for variable in obj.get(key) or {}:
            if not variable.endswith(PRODUCTION_SUFFIX):
                continue
            for target in obj.get("targets") or []:
                if f"update_{variable}" not in writers.get(target, set()):
                    continue
                flow = variable[: -len(PRODUCTION_SUFFIX)]
                raise ComponentSpecError(
                    f"Failure mode {obj.get('name')} declares `{key}` on "
                    f"`{target}.{variable}`, and `{flow}` also declares a "
                    f"production condition, which writes that same variable. "
                    f"muscadet resolves the two by the order events happen in; "
                    f"here an effect is held and re-evaluated, so it wins at "
                    f"every evaluation and the condition never shows. A "
                    f"DORMANT output a mode starts declares NO production "
                    f"condition (muscadet's `var_prod_cond`) and keeps its "
                    f"`var_prod_default`; drop the condition, or stop writing "
                    f"`{variable}`"
                )


def _availability_gates(flows: dict) -> set[tuple[str, str]]:
    """Every availability gate the flow declarations carry, persistent or
    reinitialized, as ``(component, attribute)`` pairs."""
    gates = set()
    for name, spec in flows.items():
        for entry in spec.get("flows") or []:
            if not isinstance(entry, dict) or entry.get("cls") not in _DISCRETE_OUT_CLASSES:
                continue
            flow = entry.get("name")
            if isinstance(flow, str):
                gates.add((str(spec.get("name") or name), flow + AVAILABILITY_SUFFIX))
    return gates


def _refuse_a_pulse_on_a_reinitialized_gate(obj: dict, gates: set[tuple[str, str]]) -> None:
    """Refuse a one-shot write on a gate the reference RESETS every step.

    The declaration route's half of
    :func:`~pyraichu.plugins.muscadet._refuse_a_pulse_on_a_reinitialized_gate`.
    On the reference engine the pulse is undone by the next step's
    reinitialization, so it is only seen inside one fixpoint; here an edge
    write stays until something else writes the attribute, so the same
    declaration would latch where the reference forgets."""
    for key in ("failure_effects_trans", "repair_effects_trans"):
        for variable in obj.get(key) or {}:
            for target in obj.get("targets") or []:
                if (target, variable) in gates:
                    raise ComponentSpecError(
                        f"Failure mode {obj.get('name')} writes `{target}.{variable}` "
                        f"once (`{key}`), and its flow reinitializes that gate every "
                        "step (muscadet's `var_fed_available_out_reset=True`): the "
                        "reference undoes the pulse at the next step, where this "
                        "engine would keep it. Declare the gate persistent "
                        "(`var_fed_available_out_reset=False`), which is what a "
                        "one-shot effect latches"
                    )


def _persistent_availability_gates(flows: dict) -> set[tuple[str, str]]:
    """The availability gates the flow declarations mark PERSISTENT, as the
    ``(component, attribute)`` pairs a mode's effect would name.

    ``var_fed_available_out_reset=False`` asks the gate to MEMORISE its last
    value instead of falling back on its seed. The control travels on every
    output port of a library that declares it -- the reference corpus carries
    27 of them -- while only a handful of ports are ever written by a mode, so
    what matters is not the control but the collision with a writer.
    """
    gates = set()
    for name, spec in flows.items():
        for entry in spec.get("flows") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("cls") not in _DISCRETE_OUT_CLASSES:
                continue
            if entry.get("var_fed_available_out_reset") is not False:
                continue
            flow = entry.get("name")
            if isinstance(flow, str):
                gates.add((str(spec.get("name") or name), flow + AVAILABILITY_SUFFIX))
    return gates


def _refuse_a_held_write_on_a_persistent_gate(
    obj: dict, gates: set[tuple[str, str]]
) -> None:
    """Refuse a standalone mode writing an availability gate declared
    persistent.

    The declaration route's half of
    :func:`~pyraichu.plugins.muscadet._refuse_a_held_write_on_a_persistent_gate`,
    which asks the same question of the platform's flatter vocabulary and
    cannot see a declaration's flows. A gate a COMPONENT's own failure modes
    derive is caught a layer lower, where the derivation is emitted
    (:meth:`pyraichu.muscadet.ObjFlow._build_flows_out`); this one catches the
    other writer a document has, the two-state mode declared as a component of
    its own.

    Why it is a refusal and not a divergence: this engine has no per-variable
    reset to switch off. A mode's effect is HELD, re-evaluated to a fixpoint
    while its state lasts, and completed with the rest branch reinitialization
    would have restored -- so the gate comes back up on repair, which is the
    opposite of what persistence was declared for, and nothing in the model
    would say so.
    """
    if not gates:
        return
    for key in ("failure_effects", "repair_effects"):
        for variable in obj.get(key) or {}:
            for target in obj.get("targets") or []:
                if (target, variable) not in gates:
                    continue
                raise ComponentSpecError(
                    f"Failure mode {obj.get('name')} declares `{key}` on "
                    f"`{target}.{variable}`, whose flow is declared "
                    f"persistent (muscadet's "
                    f"`var_fed_available_out_reset=False`). A held effect on "
                    f"a gate that is never reinitialized has no faithful "
                    f"expansion here: this engine restores the rest state the "
                    f"mode declares, so the gate would come back up on repair "
                    f"instead of latching. Declare a reinitialized gate, or "
                    f"stop writing `{variable}`"
                )


def _condition_leaves(cond: Any) -> list[dict]:
    """The leaves of a normalised condition, or none for a bare boolean."""
    if not isinstance(cond, list):
        return []
    return [leaf for group in cond for leaf in group if isinstance(leaf, dict)]


def _expand_object(plugin: Any, obj: dict, body: dict) -> None:
    """One controller or mode object, expanded into the document body in place."""
    try:
        components, connections, indicators = plugin.expand_object(obj, body)[:3]
    except (ValueError, KeyError) as error:
        detail = (
            str(error)
            if isinstance(error, ValueError)
            else f"{type(error).__name__}: {error}"
        )
        raise ComponentSpecError(f"{_object_where(obj)}: {detail}") from error
    body["components"] += components
    body["connections"] += connections
    # By the same rule the plugin expansion uses, and for the same reason: an
    # object emits indicators under the name its domain spells the observation
    # with, and a second writer of that observation is naming one indicator.
    #
    # Two OBJECTS meet here the way an object and a flow component meet above,
    # and for the same flattening: a controller `P` with a signal `q_r` and a
    # controller `P_q` with a signal `r` both emit `P_q_r`. Neither knows about
    # the other -- an object is expanded on its own -- so the encounter can
    # only be seen from here.
    merge_indicators(
        body.setdefault("indicators", []),
        indicators,
        collision=lambda name, emitted, already: ComponentSpecError(
            f"{_object_where(obj)}: the indicator {name!r} is emitted on "
            f"{emitted} while the document already observes {already} under "
            f"that name"
        ),
    )


def _refuse_ungated_modes(spec: dict) -> None:
    """Refuse a failure mode of ``spec`` that gates no discrete output.

    The one point where the two vocabularies read the same ABSENCE in
    opposite directions, and the reason it is caught here rather than in
    :func:`build_component`.

    A failure mode of this layer gates the availability of the component's
    discrete outputs, and an empty ``targets`` means EVERY one of them
    (`pyraichu.muscadet.ObjFlow._build_flows_out`). A failure mode of muscadet
    is a two-state automaton that clamps exactly the variables its effects
    name, so a mode naming none clamps nothing. Built as it stands, a muscadet
    mode that derates a continuous output -- or that only flips a state an
    indicator watches -- would silently kill every discrete output of its
    component.

    Refused rather than reinterpreted: this layer has no spelling for "gates
    none of them", and inventing one would change what a declaration written
    for this layer has always meant. The refusal names the outputs at stake so
    a modeller can say which ones the mode kills.

    Applied at the SYSTEM scale only, which is where the document is known to
    be muscadet's: :func:`build_component` reads a declaration that may as
    well have been written for this layer, and its meaning is left alone.
    """
    discrete_outs = _discrete_out_flows(spec)
    if not discrete_outs:
        return
    name = spec.get("name")
    for index, entry in enumerate(_entries(spec, "failure_modes", name)):
        if entry.get("targets"):
            continue
        where = f"Component {name}: failure mode {entry.get('name', index)!r}"
        gated, _ = _gate_effects(
            where,
            "failure_effects",
            _effects(where, "failure_effects", entry.get("failure_effects")),
            discrete_outs,
            gating=False,
        )
        if not gated:
            raise SystemSpecError(
                f"{where} gates none of the discrete outputs {discrete_outs} "
                f"of its component, and here an empty target list means ALL of "
                f"them. Declare which outputs the mode kills, on its "
                f"`failure_effects` or its `targets`"
            )
