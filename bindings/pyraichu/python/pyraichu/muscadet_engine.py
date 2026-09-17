"""RAICHU as an engine of muscadet: registration, and the run it receives.

muscadet is a modelling facade over more than one engine. It exposes an
extension point and imports no engine at all; an engine registers itself
there, and from that moment a modeller writes muscadet and chooses where the
run happens::

    system.simulate({"nb_runs": 1000, "schedule": [0, 10, 20]}, engine="raichu")

**The modeller never names pyraichu.** This module is advertised under the
``muscadet.engines`` entry point group of the ``pyraichu`` distribution, so an
installed pyraichu is a registered engine and nothing has to be imported to
select it. Installing the package IS the registration; :func:`register` is the
hook muscadet calls, and is also there for a caller who would rather register
explicitly.

What crosses the seam, and what does not
----------------------------------------
What arrives is the **system declaration**, ``muscadet.declare.system_spec``:
a document, checked by muscadet on the way out. Never the live system, which
would make the seam "share the reference engine's objects" instead of "share a
document".

The run parameters arrive BESIDE it, in cod3s's own vocabulary
(``nb_runs``, ``schedule``, ``seed``), because they configure a run and do not
describe a system. :func:`simulate` translates them into a RAICHU Monte-Carlo:
the schedule becomes the sample instants and its last instant the horizon.

Beside them again, and beside the document too, travels the one run keyword
muscadet spells itself: :data:`RUN_TARGETS`, the feared events this run stops
at. It is not a run PARAMETER -- ``_run_parameters`` refuses it inside the
parameters, and says so -- because the same system is run twice, free-cycling
for its availability figures and first-occurrence for its sequences, and a
section of the document would make those two runs two different systems.
muscadet checks the names against the declaration; what is left here is the
translation into what RAICHU stops at, an automaton and a state, which is the
half only an engine can spell.

`pyraichu.muscadet` is where the declaration lands
--------------------------------------------------
The authoring layer that mirrors muscadet's idioms is no longer a second
public interface competing with muscadet's own: it is the **internal
adapter**, reached through :func:`pyraichu.declare.build_document` and never
named by a modeller. Nothing new enters it; what an alignment adds lands in
the declaration reader (:mod:`pyraichu.declare`) or here.

What this module owns, and the layer below does not, is the OBSERVATION: an
indicator is named on the document, while the authoring layer emits its own
from the variables it generated. Reconciling the two is a property of the
model to run, not of the system, which is why it lives beside the run.

Kept import-light: importing pyraichu must not import muscadet, and does not.
`muscadet` is reached inside :func:`register` alone, which nobody calls before
muscadet is the thing doing the calling.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from . import MODEL_ENVELOPE_KEY, Model, load_model, model_body, monte_carlo, seal
from . import interactive as open_interactive
from .declare import (
    SystemSpecError,
    build_document,
    derived_out_states,
    event_automaton,
    event_occurrence_state,
)

__all__ = [
    "ENGINE_DESCRIPTION",
    "ENGINE_NAME",
    "RUN_TARGETS",
    "build_model",
    "isimu_start",
    "register",
    "simulate",
]

#: How a muscadet run selects this engine. It is the entry point's name too,
#: written once here so the two cannot drift: a distribution advertising one
#: name while the code registers another is an engine that is installed and
#: unreachable, and nothing says so.
ENGINE_NAME = "raichu"

#: Free text muscadet shows beside the name when a human lists what is
#: installed.
ENGINE_DESCRIPTION = (
    "RAICHU, a native Rust engine for hybrid (PDMP) simulation, through its "
    "pyraichu binding"
)

#: The one run keyword muscadet spells itself (``muscadet.engine.RUN_TARGETS``):
#: the events a run stops at, handed over BESIDE the document because a target
#: configures a run and does not describe a system. One system is run twice --
#: free-cycling for its availability figures, first-occurrence for its
#: sequences -- and a section of the declaration would make those two runs two
#: different systems.
#:
#: Written out rather than imported from muscadet, which importing pyraichu
#: must not do. The two spellings cannot drift in silence: a keyword muscadet
#: sent under another name would fall through to :func:`pyraichu.monte_carlo`
#: and be refused there, by name.
RUN_TARGETS = "targets"

#: Run-parameter keys that describe HOW the reference engine draws its
#: replicas, TRACES them or REPORTS them, rather than what is computed, and
#: that RAICHU answers its own way. Accepted and not read, because refusing
#: them would refuse every study cod3s writes: a cod3s parameter object writes
#: every field whether or not it was set, so a refusal on the PRESENCE of a
#: key is a refusal of the whole corpus.
#:
#: The four added for the platform corpus all describe the reference runner's
#: own post-processing (``cod3s.specs.study_yaml.SimulationConfig``): which
#: transitions PyCATSHOO monitors, which cycles the sequence analyser folds
#: away afterwards, and whether a mode that failed to build aborts the study.
#: None of them describes the system, and none has a RAICHU counterpart that
#: could answer them differently.
#:
#: Where a key changes a RESULT rather than a draw, it is not here: see
#: :data:`_DIVERGENT_PARAMETERS`.
_UNREAD_PARAMETERS = (
    "time_unit",
    "rng",
    "rng_bloc_size",
    "filter_objevent_in_sequences",
    "filter_objfm_in_sequences",
    "monitor_patterns",
    "strict_failure_modes",
)

#: Run-parameter keys that are accepted, not read, and **would change a
#: result** on a model that has something for them to govern. They are kept
#: apart from the silence above on purpose: what they ask for is not a knob
#: this engine lacks but a convention it holds differently, which is a
#: DIVERGENCE and belongs in ``muscadet.conformance``, the registry a reader
#: consults before choosing an engine. Stating it here as well would be a
#: second wording of one gap; refusing it would refuse every continuous study
#: the platform writes, and refusing it on a purely discrete one would refuse
#: a study it governs nothing of.
_DIVERGENT_PARAMETERS = {
    "pdmp_dt": (
        "the base integration step of the reference engine's PDMP solver. "
        "RAICHU integrates its own way -- it locates a crossing rather than "
        "sampling a fixed grid -- so the step has no counterpart to set. On a "
        "purely discrete model it governs nothing on either engine"
    ),
}

#: Run-parameter keys asking for something this engine does not produce. They
#: are refused as soon as they ask, and accepted while they say nothing: a
#: cod3s parameter object writes every field whether or not it was set.
_UNCARRIED_PARAMETERS = {
    "trace_level": (
        "a PyCATSHOO simulator trace. RAICHU records a causal journal instead, "
        "which is asked for on the run and read off the result"
    ),
    "trace_elements": (
        "the elements a PyCATSHOO trace follows; see `trace_level`"
    ),
}

#: What muscadet's declaration calls an indicator's kind, and what RAICHU
#: names the same observation.
_INDICATOR_SUBJECT = {
    "PycVarIndicator": "var",
    "PycAttrIndicator": "attr_name",
}

#: What ``PycAttrIndicator.attr_type`` reads when the attribute it names is a
#: STATE rather than a variable, and the key the state's name is then in.
#:
#: cod3s has TWO spellings of one observation and writes the second: a
#: ``PycSTIndicator`` carrying ``state``, and a ``PycAttrIndicator`` carrying
#: ``attr_type: "ST"`` with the state in ``attr_name``. Both resolve to the
#: same ``{component}.{state}`` expression on the reference engine
#: (``cod3s/pycatshoo/indicator.py``, ``get_expr``), so reading only the first
#: made a state indicator a variable indicator on a variable no component has.
_STATE_ATTR_TYPE = "ST"
_STATE_ATTR_NAME_KEY = "attr_name"

#: The kind naming a STATE rather than a variable, and the one component
#: family it is carried on.
#:
#: A state indicator names a state of an automaton, and the two layers do not
#: name automata or their states alike in general -- muscadet's failure mode
#: `m` sits in `m_occ` of automaton `{comp}_m`, RAICHU's in `nok` of automaton
#: `m` -- so the mapping would be a translation nobody has declared.
#:
#: **An EVENT is the case where it IS declared**, and the lift is indexed on
#: what justifies it rather than on the class: a `cod3s.ObjEvent` hands its
#: engine the three names it was declared with (`event_aut_name`,
#: `occ_state_name`, `not_occ_state_name`) and
#: `pyraichu.plugins.muscadet._expand_objevent` builds the automaton and its
#: two states under exactly those three. The two layers therefore spell the
#: reference identically, which is the only thing the refusal was ever about.
#: `pyraichu.declare.event_automaton` is where those names are read, so this
#: module holds no second copy of the convention.
_STATE_INDICATOR_KIND = "PycSTIndicator"


def register() -> None:
    """Register RAICHU at muscadet's extension point.

    The hook the ``muscadet.engines`` entry point resolves to, and the shape
    that group prescribes: no argument, and it registers under the entry
    point's own name. muscadet checks afterwards that the name actually
    appeared, so a hook that registered nothing is a reported absence rather
    than an engine that is installed and invisible.

    ``replace=True``: registering twice is this engine taking its own name
    over, which is a no-op of a state, and never two packages racing for one
    name. Without it, a caller that imports this module and then lets muscadet
    discover the entry point pays an ``EngineAlreadyRegisteredError`` for
    having been explicit.
    """
    import muscadet

    muscadet.register_engine(
        name=ENGINE_NAME,
        simulate=simulate,
        isimu_start=isimu_start,
        description=ENGINE_DESCRIPTION,
        replace=True,
    )


# ---------------------------------------------------------------------------
# The declaration, turned into a model to run
# ---------------------------------------------------------------------------


def _event_specs(components: Any) -> dict[str, Mapping[str, Any]]:
    """The EVENT declarations a document carries, by the name they answer to.

    The one reading of "which of these components is an event", so a state
    indicator, a condition leaf and a sequence target all resolve through
    :func:`pyraichu.declare.event_automaton` rather than through three tests
    that can drift.
    """
    if not isinstance(components, Mapping):
        return {}
    return {
        str(entry.get("name") or name): entry
        for name, entry in components.items()
        if event_automaton(entry) is not None
    }


def _events(components: Any) -> dict[str, tuple[str, set[str]]]:
    """The EVENTS a document declares, by name: their automaton and its states.

    Read through :func:`pyraichu.declare.event_automaton`, which is where the
    three names an event is built with are written down, so a state indicator
    and a condition leaf watching the same event resolve through one reading
    rather than two that can drift.
    """
    return {
        name: event_automaton(entry)
        for name, entry in _event_specs(components).items()
    }


def _derived_states(components: Any) -> dict[str, set[str]]:
    """The states of the ok/nok pairs muscadet DERIVES on the discrete outputs
    of each component, by component name.

    Read through :func:`pyraichu.declare.derived_out_states`, which is where
    the naming of that pair is written down. The declaration key itself is
    accepted -- muscadet writes it only when it asks for the pair, so refusing
    it refused every platform export -- and what is refused instead is an
    indicator that names one of these states, which is the one thing the
    absence of the pair takes away.
    """
    if not isinstance(components, Mapping):
        return {}
    found = {}
    for name, entry in components.items():
        states = derived_out_states(entry)
        if states:
            found[str(entry.get("name") or name)] = states
    return found


def _state_indicator(
    spec: Mapping[str, Any],
    events: Mapping[str, tuple[str, set[str]]],
    derived: Mapping[str, set[str]],
) -> dict[str, Any]:
    """One state indicator, which only an EVENT carries. See
    :data:`_STATE_INDICATOR_KIND` for why it is the one family it is.

    The state is read from ``state`` or from ``attr_name``, cod3s's two
    spellings of one observation (:data:`_STATE_ATTR_NAME_KEY`).
    """
    name = spec.get("name")
    component = str(spec.get("component") or "")
    state = str(spec.get("state") or spec.get(_STATE_ATTR_NAME_KEY) or "")
    if not (name and component and state):
        raise SystemSpecError(
            f"indicator {dict(spec)!r}: a declaration names its indicator, its "
            f"component and the state it observes"
        )
    event = events.get(component)
    if event is None:
        if state in derived.get(component, ()):
            raise SystemSpecError(
                f"indicator {name!r} observes the state {state!r} of "
                f"{component!r}, which is one of the ok/nok pair muscadet "
                f"DERIVES on a discrete output "
                f"(`create_default_out_automata`). This layer derives no such "
                f"pair -- every automaton it builds comes from a declaration "
                f"that needs one -- so the indicator would be silently "
                f"absent. Declare the failure mode the pair stands for, or "
                f"observe the flow's `_fed_out` variable instead"
            )
        raise SystemSpecError(
            f"indicator {name!r} observes the state {state!r} of "
            f"{component!r}, which the document does not declare as an EVENT. "
            f"A state indicator is carried on an event alone: it is the one "
            f"two-state component whose automaton and states muscadet and "
            f"RAICHU spell alike, where a failure mode's `m` sits in `m_occ` "
            f"of automaton `{{comp}}_m` here and in `nok` of automaton `m` "
            f"there, a mapping nobody has declared"
        )
    automaton, states = event
    if state not in states:
        raise SystemSpecError(
            f"indicator {name!r} observes the state {state!r} of the event "
            f"{component!r}, which holds {sorted(states)}"
        )
    return {
        "name": str(name),
        "target": "state",
        "component": component,
        "automaton": automaton,
        "state": state,
    }


def _indicator(
    spec: Mapping[str, Any],
    events: Mapping[str, tuple[str, set[str]]],
    derived: Mapping[str, set[str]],
) -> dict[str, Any]:
    """One declared indicator, as RAICHU names the same observation."""
    kind = spec.get("kind")
    name = spec.get("name")
    attr_type = str(spec.get("attr_type", "VAR"))
    if kind == _STATE_INDICATOR_KIND or (
        kind == "PycAttrIndicator" and attr_type == _STATE_ATTR_TYPE
    ):
        return _state_indicator(spec, events, derived)
    if kind not in _INDICATOR_SUBJECT:
        raise SystemSpecError(
            f"indicator {name!r} is a {kind!r}, which this engine does not "
            f"carry (it reads {sorted(_INDICATOR_SUBJECT) + [_STATE_INDICATOR_KIND]})"
        )
    if kind == "PycAttrIndicator" and attr_type != "VAR":
        raise SystemSpecError(
            f"indicator {name!r} observes attr_type={attr_type!r}, which is "
            f"neither a variable ('VAR') nor a state "
            f"('{_STATE_ATTR_TYPE}'): this engine carries those two"
        )
    subject = spec.get(_INDICATOR_SUBJECT[kind])
    component = spec.get("component")
    if not (name and component and subject):
        raise SystemSpecError(
            f"indicator {dict(spec)!r}: a declaration names its indicator, its "
            f"component and the variable it observes"
        )
    return {
        "name": str(name),
        "target": "attribute",
        "attr": {"component": str(component), "attribute": str(subject)},
    }


def _merge_indicators(
    body: dict[str, Any],
    declared: Iterable[Mapping],
    events: Mapping[str, tuple[str, set[str]]],
    derived: Mapping[str, set[str]],
) -> None:
    """Add the document's indicators to the generated model, in place.

    The authoring layer emits one indicator per observable variable it
    generated, named ``{component}_{variable}`` -- muscadet's own convention,
    which is why a declaration usually asks for something already there. What
    is added here is what it does NOT cover: an indicator the modeller renamed,
    one on a variable this layer generates without observing, or one on the
    state of an event, which no variable stands for.

    A declared name already taken by a DIFFERENT observation is refused rather
    than resolved: two indicators of one name is a result whose reader cannot
    tell which he got. Compared on the WHOLE entry rather than on its ``attr``,
    which a state indicator carries none of: an entry compared on a key it has
    not would read as equal to every other one of its kind.

    **The estimate comes back under the DECLARED name, and a consumer keyed on
    ``{component}_{attribute}`` will not always find it.** Worth stating here
    because the agreement between the two looks total and is not: a ``VAR``
    indicator's declared name IS ``{component}_{attribute}``, cod3s's own
    default pattern (``PycAttrIndicator.cls_validator``), so every one of them
    is found either way and the coincidence hides the rule. A STATE indicator
    stands for no variable, so it agrees with nothing: the platform's
    ``synthetic_event`` declares ``Rail_and_Detecteur_fed`` on the event
    ``_ind_eb45a35f``, and a consumer looking for ``_ind_eb45a35f_occ`` finds
    no estimate.

    Measured 2026-09-14 on that package, rendered by the platform's production
    runner (``scripts/cod3s-run-study-raichu``, ``_indicator_rows``): 30 of its
    40 rows, the ten of the state indicator skipped on a warning. The values
    themselves are right -- looked up by declared name the forty rows equal the
    golden -- so what is left is the LOOKUP, which belongs to whoever reads the
    estimates and not to this seam. Registering the estimate a second time
    under ``{component}_{attribute}`` would be this module inventing a name the
    document never declared, and two names for one observation is exactly what
    the refusal above exists to prevent.
    """
    existing = {
        entry["name"]: entry for entry in body.get("indicators") or [] if "name" in entry
    }

    def observation(entry: Mapping) -> dict:
        return {key: value for key, value in entry.items() if key != "name"}

    for spec in declared or []:
        wanted = _indicator(spec, events, derived)
        already = existing.get(wanted["name"])
        if already is None:
            body.setdefault("indicators", []).append(wanted)
            existing[wanted["name"]] = wanted
            continue
        if observation(already) != observation(wanted):
            raise SystemSpecError(
                f"indicator {wanted['name']!r} is declared on "
                f"{observation(wanted)} while the generated model already "
                f"observes {observation(already)} under that name"
            )


def _target_names(targets: Any) -> list[str]:
    """The event names a run declares as its sequence targets, read as a
    vocabulary and before any document is involved.

    muscadet reads them the same way on its side and hands over what it read
    (:data:`RUN_TARGETS`), so on the seam this is a re-reading of something
    already sound. It is not redundant, because :func:`build_model` is a
    public door of its own: the platform reaches for the MODEL, to give it to
    a second engine call the seam has no kind for, and a caller coming in that
    way has crossed no validation at all.

    **A bare string is refused rather than iterated.** ``"PANNE_OND"`` taken
    as a sequence is nine targets named ``P``, ``A``, ``N``..., so the mistake
    would come back as nine sentences that never mention it.
    """
    if targets is None:
        return []
    if isinstance(targets, str):
        raise SystemSpecError(
            f"`{RUN_TARGETS}`={targets!r} is one string where a run declares a "
            f"LIST of event names: pass [{targets!r}], a bare name being read "
            f"one target per letter"
        )
    if not isinstance(targets, (list, tuple)):
        raise SystemSpecError(
            f"`{RUN_TARGETS}` is the list of events a run stops at, got "
            f"{type(targets).__name__}"
        )
    names: list[str] = []
    for entry in targets:
        if not isinstance(entry, str) or not entry:
            raise SystemSpecError(
                f"`{RUN_TARGETS}` names each target by the name of its event, "
                f"got {entry!r}"
            )
        if entry not in names:
            names.append(entry)
    return names


def _targets(components: Any, names: Iterable[str]) -> list[dict[str, Any]]:
    """The model's ``targets`` section, from the event names a run declares.

    The translation muscadet leaves to the engine, and the only half of the
    vocabulary that is RAICHU's: muscadet names the EVENT, this names the
    automaton and the state whose activation ends and labels a trajectory.
    Both halves are needed because the two engines spell that state
    differently, and a name is the one thing they agree on.

    Resolved through :mod:`pyraichu.declare`, where an event's three names are
    written down: :func:`~pyraichu.declare.event_automaton` for the automaton
    and :func:`~pyraichu.declare.event_occurrence_state` for the state a
    trajectory ends at. Nothing about how an event is named is restated here,
    which is what keeps a renamed one honoured -- a target on a state no
    automaton holds is a campaign that stops at nothing.
    """
    declared = _event_specs(components)
    known = sorted(declared) or ["<none>"]
    entries = []
    for name in names:
        spec = declared.get(name)
        if spec is None:
            raise SystemSpecError(
                f"the run stops at {name!r}, which this document declares no "
                f"EVENT under: a sequence target ends a trajectory at a feared "
                f"event's occurrence, and there is nothing here to reach. The "
                f"events this document declares are {known}"
            )
        entries.append(
            {
                "name": name,
                "component": name,
                "automaton": event_automaton(spec)[0],
                "state": event_occurrence_state(spec),
            }
        )
    return entries


def build_model(
    spec: Mapping[str, Any], targets: Sequence[str] | None = None
) -> Model:
    """The RAICHU model a muscadet system declaration describes.

    The whole translation, in one place and reachable without running
    anything: a caller comparing what the two engines were handed reads this
    rather than a trajectory.

    ``build_document`` and not ``build_system``: a document's ``components``
    holds two shapes, the components carrying flows and the standalone failure
    modes, and only the first are part of the system's wiring. The second are
    expanded onto the first once they exist, which is the document's scale and
    not the system's.

    ``targets`` is the run's sequence targets (:data:`RUN_TARGETS`), named by
    their events, and it is taken HERE and not only on :func:`simulate`
    because the model is a thing a caller legitimately asks for: the platform
    builds it once and hands it to two engine calls, the Monte-Carlo campaign
    and :func:`pyraichu.analyse_sequences`, which the seam has no kind for. A
    door that existed on the run alone would leave that caller writing the
    ``targets`` section by hand, which is exactly the workaround this replaces.

    The section is written and nothing else: whether the trajectories actually
    STOP there is the run's business (``stop_at_targets``, see
    :func:`simulate`), and a model carrying targets is read by the sequence
    analysis either way.
    """
    document = build_document(dict(spec))
    components = spec.get("components")
    body = model_body(document)
    _merge_indicators(
        body,
        spec.get("indicators") or [],
        _events(components),
        _derived_states(components),
    )
    declared = _targets(components, _target_names(targets))
    if declared:
        body["targets"] = list(body.get("targets") or []) + declared
    # Re-sealed rather than returned as it stands: the required-feature list is
    # derived from the body, and the body has just been written to.
    if MODEL_ENVELOPE_KEY in document:
        document = seal(document)
    return load_model(json.dumps(document))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _as_mapping(params: Any) -> dict[str, Any]:
    """Run parameters as a plain mapping, whatever shape they arrived in.

    cod3s hands a pydantic ``PycMCSimulationParam`` or the dict a caller wrote;
    muscadet passes on whichever it received, untouched, which is the seam
    working as intended.
    """
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return dict(params)
    dump = getattr(params, "model_dump", None)
    if callable(dump):
        return dict(dump())
    raise SystemSpecError(
        f"run parameters are a mapping or a cod3s parameter object, got "
        f"{type(params).__name__}"
    )


def _instants(schedule: Any) -> list[float]:
    """The sample instants a schedule names, sorted.

    cod3s writes an instant either as a number or as a linear range; the range
    is expanded here rather than by importing cod3s, which this module must not
    do -- it would make an engine binding depend on the reference engine's
    stack to be installed at all.
    """
    if schedule is None:
        return []
    if isinstance(schedule, (int, float)) and not isinstance(schedule, bool):
        schedule = [schedule]
    instants: list[float] = []
    for entry in schedule:
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            instants.append(float(entry))
            continue
        fields = entry if isinstance(entry, Mapping) else getattr(entry, "__dict__", {})
        try:
            start = float(fields["start"])
            end = float(fields["end"])
            count = int(fields["nvalues"])
        except (KeyError, TypeError, ValueError):
            raise SystemSpecError(
                f"schedule entry {entry!r}: an instant is a number, or a range "
                f"declaring 'start', 'end' and 'nvalues'"
            ) from None
        if count <= 1:
            instants.append(end)
            continue
        step = (end - start) / (count - 1)
        instants += [start + step * index for index in range(count)]
    return sorted(instants)


def _run_parameters(params: Any) -> tuple[int, list[float], int]:
    """``(nb_runs, instants, seed)``, refusing what this engine cannot honour."""
    declared = _as_mapping(params)
    for key, reason in _UNCARRIED_PARAMETERS.items():
        if declared.get(key):
            raise SystemSpecError(f"run parameter `{key}` asks for {reason}")
    for key in (
        tuple(_UNCARRIED_PARAMETERS) + _UNREAD_PARAMETERS + tuple(_DIVERGENT_PARAMETERS)
    ):
        declared.pop(key, None)

    nb_runs = int(declared.pop("nb_runs", 1) or 1)
    instants = _instants(declared.pop("schedule", None))
    seed = declared.pop("seed", None)
    if declared:
        raise SystemSpecError(
            f"run parameters carry {sorted(declared)}, which this engine does "
            f"not read. A knob of the engine itself travels as a keyword of "
            f"the run, beside the parameters, not inside them"
        )
    if not instants:
        raise SystemSpecError(
            "a run declares its 'schedule': it is what says how long the "
            "trajectories are and where they are read"
        )
    return nb_runs, instants, 0 if seed is None else int(seed)


def simulate(spec: Mapping[str, Any], params: Any = None, **kwargs: Any):
    """Batch run of a muscadet system declaration on RAICHU.

    The runner muscadet calls for ``system.simulate(..., engine="raichu")``.
    The schedule's instants are the sample instants and its last one the
    horizon, which is what makes the two engines answer at the same dates.

    ``targets`` is :data:`RUN_TARGETS`, muscadet's own run keyword: the events
    this run stops at, named by their events and translated into the model's
    ``targets`` section by :func:`build_model`. It is read HERE rather than
    passed on, because :func:`pyraichu.monte_carlo` has no such argument and
    would refuse it by name.

    **``stop_at_targets`` is DERIVED from it, and an explicit one still
    wins.** A target that does not stop the trajectory is not a target: it is
    what the reference engine does without being asked (PyCATSHOO stops
    unconditionally on an ``addTarget``), and it is what the caller who
    declared a feared event means. The two RAICHU knobs stay separable
    underneath -- the model carries the targets, the run decides whether it
    latches -- so a study that really wants a free-cycling campaign over a
    model carrying targets says ``stop_at_targets=False`` and gets it.

    ``postpone_post_proc`` is accepted and does nothing: on the reference path
    it defers writing the indicator values back onto the live system, and here
    there is nothing to defer -- the estimates ARE the answer, handed back to
    the caller. Every other keyword travels through to
    :func:`pyraichu.monte_carlo`, so an engine knob (``threads``,
    ``quantiles``, ``rtol``...) reaches the engine that understands it and a
    misspelling is refused there, by name.

    Returns
    -------
    pyraichu.McEstimates
        RAICHU's own result. muscadet hands an engine's answer back untouched:
        the reference path writes its indicators onto the live system and
        returns nothing, this one returns the estimates.
    """
    kwargs.pop("postpone_post_proc", None)
    targets = _target_names(kwargs.pop(RUN_TARGETS, None))
    nb_runs, instants, seed = _run_parameters(params)
    kwargs.setdefault("stop_at_targets", bool(targets))
    return monte_carlo(
        build_model(spec, targets),
        nb_runs=nb_runs,
        t_max=instants[-1],
        samples=instants,
        seed=seed,
        **kwargs,
    )


def isimu_start(spec: Mapping[str, Any], params: Any = None, **kwargs: Any):
    """Interactive session over a muscadet system declaration, on RAICHU.

    The same declaration as :func:`simulate`, and deliberately the same build:
    a model that behaved differently one step at a time than in bulk is a
    divergence nothing would report.

    Run parameters are optional here, an interactive session having no
    replicas and no schedule; a horizon is honoured when the schedule declares
    one.

    ``targets`` (:data:`RUN_TARGETS`) is accepted, and muscadet hands it to
    both kinds of run on purpose: a keyword one entry point takes and the
    other dies on would make a demonstration and a campaign diverge on the
    very study they are meant to be two views of. It reaches the same
    :func:`build_model` as a batch run, so the session is opened over the same
    model -- and nothing stops the stepping, a session having no
    ``stop_at_targets`` to set. Whoever drives it decides what reaching a
    feared event means, which is the point of driving by hand.

    Returns
    -------
    pyraichu.Interactive
        RAICHU's own session object, handed back untouched. It is not
        ``cod3s``'s interactive engine and does not pretend to be: what a
        driver of both needs is a session it can step, and the two are stepped
        differently.
    """
    targets = _target_names(kwargs.pop(RUN_TARGETS, None))
    declared = _as_mapping(params)
    instants = _instants(declared.get("schedule"))
    if instants:
        kwargs.setdefault("t_max", instants[-1])
    return open_interactive(build_model(spec, targets), **kwargs)
