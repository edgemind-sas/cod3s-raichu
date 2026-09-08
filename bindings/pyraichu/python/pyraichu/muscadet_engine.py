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

`pyraichu.muscadet` is where the declaration lands
--------------------------------------------------
The authoring layer that mirrors muscadet's idioms is no longer a second
public interface competing with muscadet's own: it is the **internal
adapter**, reached through :func:`pyraichu.declare.build_system` and never
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
from typing import Any, Iterable, Mapping

from . import MODEL_ENVELOPE_KEY, Model, load_model, model_body, monte_carlo, seal
from . import interactive as open_interactive
from .declare import SystemSpecError, build_system

__all__ = [
    "ENGINE_DESCRIPTION",
    "ENGINE_NAME",
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

#: Run-parameter keys that describe HOW the reference engine draws its
#: replicas rather than what is computed, and that RAICHU answers its own way.
#: Accepted and not read, because refusing them would refuse every study cod3s
#: writes; where they change a result rather than a draw, the divergence is
#: muscadet's conformance registry to state, not this module's to hide.
_UNREAD_PARAMETERS = ("time_unit", "rng", "rng_bloc_size")

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
#: names the same observation. Only the variable kinds are here: a state
#: indicator names a state of an automaton, and the two layers do not name
#: automata or their states alike -- muscadet's failure mode `m` sits in
#: `m_occ` of automaton `{comp}_m`, RAICHU's in `nok` of automaton `m` -- so
#: the mapping is a translation nobody has declared and it is refused by name
#: rather than guessed.
_INDICATOR_SUBJECT = {
    "PycVarIndicator": "var",
    "PycAttrIndicator": "attr_name",
}


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


def _indicator(spec: Mapping[str, Any]) -> dict[str, Any]:
    """One declared indicator, as RAICHU names the same observation."""
    kind = spec.get("kind")
    name = spec.get("name")
    if kind not in _INDICATOR_SUBJECT:
        raise SystemSpecError(
            f"indicator {name!r} is a {kind!r}, which this engine does not "
            f"carry (it reads {sorted(_INDICATOR_SUBJECT)}). An indicator on "
            f"an automaton's state names a state muscadet and RAICHU spell "
            f"differently, and no mapping between the two is declared"
        )
    if kind == "PycAttrIndicator" and str(spec.get("attr_type", "VAR")) != "VAR":
        raise SystemSpecError(
            f"indicator {name!r} observes attr_type={spec.get('attr_type')!r}; "
            f"only a variable is carried, for the reason a state indicator is "
            f"refused"
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


def _merge_indicators(body: dict[str, Any], declared: Iterable[Mapping]) -> None:
    """Add the document's indicators to the generated model, in place.

    The authoring layer emits one indicator per observable variable it
    generated, named ``{component}_{variable}`` -- muscadet's own convention,
    which is why a declaration usually asks for something already there. What
    is added here is what it does NOT cover: an indicator the modeller renamed,
    or one on a variable this layer generates without observing.

    A declared name already taken by a DIFFERENT observation is refused rather
    than resolved: two indicators of one name is a result whose reader cannot
    tell which he got.
    """
    existing = {
        entry["name"]: entry for entry in body.get("indicators") or [] if "name" in entry
    }
    for spec in declared or []:
        wanted = _indicator(spec)
        already = existing.get(wanted["name"])
        if already is None:
            body.setdefault("indicators", []).append(wanted)
            existing[wanted["name"]] = wanted
            continue
        if already.get("target") != wanted["target"] or already.get("attr") != wanted[
            "attr"
        ]:
            raise SystemSpecError(
                f"indicator {wanted['name']!r} is declared on "
                f"{wanted['attr']} while the generated model already observes "
                f"{already.get('attr') or already} under that name"
            )


def build_model(spec: Mapping[str, Any]) -> Model:
    """The RAICHU model a muscadet system declaration describes.

    The whole translation, in one place and reachable without running
    anything: a caller comparing what the two engines were handed reads this
    rather than a trajectory.
    """
    system = build_system(dict(spec))
    document = system.build_dict()
    _merge_indicators(model_body(document), spec.get("indicators") or [])
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
    for key in tuple(_UNCARRIED_PARAMETERS) + _UNREAD_PARAMETERS:
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

    ``postpone_post_proc`` is accepted and does nothing: on the reference path
    it defers writing the indicator values back onto the live system, and here
    there is nothing to defer -- the estimates ARE the answer, handed back to
    the caller. Every other keyword travels through to
    :func:`pyraichu.monte_carlo`, so an engine knob (``threads``,
    ``quantiles``, ``rtol``, ``stop_at_targets``...) reaches the engine that
    understands it and a misspelling is refused there, by name.

    Returns
    -------
    pyraichu.McEstimates
        RAICHU's own result. muscadet hands an engine's answer back untouched:
        the reference path writes its indicators onto the live system and
        returns nothing, this one returns the estimates.
    """
    kwargs.pop("postpone_post_proc", None)
    nb_runs, instants, seed = _run_parameters(params)
    return monte_carlo(
        build_model(spec),
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

    Returns
    -------
    pyraichu.Interactive
        RAICHU's own session object, handed back untouched. It is not
        ``cod3s``'s interactive engine and does not pretend to be: what a
        driver of both needs is a session it can step, and the two are stepped
        differently.
    """
    declared = _as_mapping(params)
    instants = _instants(declared.get("schedule"))
    if instants:
        kwargs.setdefault("t_max", instants[-1])
    return open_interactive(build_model(spec), **kwargs)
