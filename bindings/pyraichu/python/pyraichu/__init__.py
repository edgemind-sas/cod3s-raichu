"""Python binding for the RAICHU hybrid (PDMP) simulation engine.

The heavy lifting happens in the Rust extension module
``pyraichu._pyraichu``; this package is the thin, typed, Pythonic wrapper
around it.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from ._pyraichu import (
    DEFAULT_CONFIDENCE,
    MODEL_ENVELOPE_KEY,
    MODEL_FORMAT_REVISION,
    FlowConfig,
    ModelError,
    SimulationError,
    __version__,
    analyse_raw_sequences_json,
    analyse_sequences_json,
    exploration_domain_json,
    exploration_minimal_sequences_json,
    explore_json,
    fault_tree_envelope_json,
    fault_tree_json,
    fault_tree_quantify_json,
    importance_json,
    monte_carlo_json,
    required_features,
    run_sequences_json,
    seal_model,
    simulate_json,
    switching_loops_json,
    cycle_groups_json,
    unfed_triggers_json,
    validate_exploration,
    validate_fault_tree_envelope,
    validate_fault_tree_structure,
    validate_model,
)
from ._pyraichu import (
    Interactive as _RawInteractive,
)

# The extension entry points the quantification module calls, kept as
# attributes of this package as they were before it moved out.
from ._pyraichu import quantify_json as quantify_json
from ._pyraichu import validate_quantification as validate_quantification
from .journal import AttributeChange, Cascade, JournalQuery, TransitionHistory
from .library import (
    check_library,
    derive_evaluation_order,
    expand_documents,
    load_system,
    run_study,
    standard_library,
)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "FAULT_TREE_FORMAT",
    "FAULT_TREE_MEASURES",
    "FAULT_TREE_STRUCTURE_FORMAT",
    "FAULT_TREE_STRUCTURE_VERSION",
    "FAULT_TREE_VERSION",
    "MAX_MISSION_TIMES",
    "MODEL_ENVELOPE_KEY",
    "MODEL_FORMAT_REVISION",
    "QUANTIFICATION_METHODS",
    "ReductionWarning",
    "AttributeChange",
    "BasicEventImportance",
    "BiasFamily",
    "Cascade",
    "ComponentImportance",
    "ConfidenceInterval",
    "CrossEntropyResult",
    "Cut",
    "EventImportance",
    "Event",
    "Exploration",
    "ExploredSequence",
    "Extremes",
    "FaultTree",
    "FaultTreeQuantification",
    "Fireable",
    "FlowConfig",
    "GroupImportance",
    "ImportanceAnalysis",
    "IndicatorEstimate",
    "Interactive",
    "JournalQuery",
    "McEstimates",
    "Model",
    "ModelError",
    "Observation",
    "OperatorAdvance",
    "PilotIteration",
    "Quantification",
    "SequenceCampaign",
    "SequenceCondition",
    "SimulationError",
    "SimulationResult",
    "SplittingBatch",
    "SplittingLevel",
    "SplittingResult",
    "Study",
    "TargetProbability",
    "TransitionHistory",
    "UnboundedRateError",
    "__version__",
    "analyse_raw_sequences",
    "analyse_sequences",
    "cycle_groups",
    "expand_model",
    "exploration_domain",
    "explore",
    "fault_tree",
    "importance",
    "interactive",
    "load_model",
    "load_system",
    "expand_documents",
    "check_library",
    "derive_evaluation_order",
    "run_study",
    "standard_library",
    "model_body",
    "monte_carlo",
    "quantify",
    "read_exploration",
    "read_fault_tree",
    "read_fault_tree_structure",
    "read_quantification",
    "required_features",
    "run_sequences",
    "seal",
    "seal_model",
    "simulate",
    "switching_loops",
    "unfed_triggers",
]


# The extension declares its exceptions in `_pyraichu`, a name pickle cannot
# import back, so an error crossing a process boundary (a campaign worker)
# could not be pickled. They are published here, under the package that
# re-exports them.
for _exported in (ModelError, SimulationError):
    _exported.__module__ = __name__
del _exported


class UnboundedRateError(SimulationError):
    """An integrated rate reached the magnitude the model reserves for "unbounded".

    Raised when a stock with no ceiling on what leaves it meets a demand with
    no ceiling on what it takes: the physics is an instantaneous transfer, which
    a rate cannot express, so the run is refused rather than answering a
    non-physical number. ``str()`` is the engine's message; the fields name what
    to bound.

    Attributes
    ----------
    variable : str
        The integrated variable, ``component.attribute``.
    time : float
        The simulation time of the evaluation.
    rate : float
        The right-hand side value the engine met.
    unbounded : float
        The magnitude the model declares as "unbounded".

    Raised by a Monte-Carlo campaign, it also carries ``replica``,
    ``nb_runs`` and ``seed``, like any :class:`SimulationError` a campaign
    raises (see :func:`monte_carlo`).
    """

    def __init__(
        self, message: str, variable: str, time: float, rate: float, unbounded: float
    ) -> None:
        super().__init__(message)
        self.variable = variable
        self.time = time
        self.rate = rate
        self.unbounded = unbounded

    def __reduce__(self):
        # The instance dictionary travels too: a campaign's replica failure
        # sets `replica`, `nb_runs` and `seed` on it, and a worker process
        # hands its error back pickled.
        return (
            type(self),
            (str(self), self.variable, self.time, self.rate, self.unbounded),
            self.__dict__,
        )


def _value_to_python(value: dict[str, Any]) -> bool | int | float:
    """Unwrap the engine's tagged value representation."""
    return value["value"]


@dataclass(frozen=True)
class Event:
    """A fired transition (discrete structure, validation level 1)."""

    time: float
    transition: str
    from_state: str
    to_state: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Event(t={self.time}, {self.transition}: "
            f"{self.from_state} → {self.to_state})"
        )


@dataclass(frozen=True)
class SimulationResult:
    """Full result of a deterministic simulation run."""

    events: list[Event]
    indicators: dict[str, list[tuple[float, bool | int | float]]]
    samples: dict[str, list[tuple[float, bool | int | float]]]
    journal: list[dict[str, Any]]
    provenance: dict[str, Any]
    final_time: float
    #: Counted work: machine-independent performance units of the run
    #: (explicit-equation passes, accepted/rejected solver steps,
    #: integration segments, flow sweeps, allocation capping passes,
    #: margin evaluations, immediate-guard scans). Unlike wall-clock these
    #: do not move with the machine, which is what makes a measurement
    #: reproducible by a third party.
    work: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SimulationResult({len(self.events)} events, "
            f"{len(self.indicators)} indicators, final_time={self.final_time})"
        )


@dataclass(frozen=True)
class Model:
    """A validated RAICHU model, held as its canonical JSON description."""

    json: str
    name: str = field(default="", compare=False)
    allow_fmu_import: bool = field(default=False, compare=False)
    base_dir: Path = field(default_factory=Path.cwd, compare=False)
    #: For a model expanded from a component type library: the origin of
    #: every element, by element path (see :mod:`pyraichu.library`).
    provenance: dict[str, Any] | None = field(default=None, compare=False)
    #: For a model expanded with a study: its run settings.
    simulation: dict[str, Any] | None = field(default=None, compare=False)
    #: Non-fatal observations of the expansion.
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Model({self.name!r})"


def model_body(document: dict[str, Any]) -> dict[str, Any]:
    """The model body of a document, in either accepted shape.

    A document is either the bare body (the whole existing corpus) or
    that body under the mandatory format envelope
    (``{"raichu_model": {...}, "model": {...}}``). Everything that reads
    a model's sections goes through here, so neither shape needs a
    special case anywhere else.
    """
    if MODEL_ENVELOPE_KEY in document:
        return document["model"]
    return document


def seal(
    document: dict[str, Any], extra_features: Iterable[str] = ()
) -> dict[str, Any]:
    """Wrap a model document in the format envelope when it needs one.

    The required-feature list is derived **in Rust** from the body's own
    content (:func:`required_features`), so an authoring layer never
    composes it by hand and it cannot lag what the body holds. A feature
    the document already declares is *kept*, never dropped: a name this
    engine does not implement must reach the engine to be refused there,
    and dropping it would restore exactly the silence the envelope
    exists to prevent. ``extra_features`` carries the declaration of a
    document whose body has since been rewritten (plugin expansion).

    A document using only baseline constructs and declaring nothing
    needs no envelope and its body is returned unchanged, which is why
    the existing corpus keeps its exact shape.
    """
    body = model_body(document)
    features = set(extra_features)
    features |= set(document.get(MODEL_ENVELOPE_KEY, {}).get("requires", []))
    features |= set(required_features(json.dumps(body)))
    if not features:
        return body
    return {
        MODEL_ENVELOPE_KEY: {
            "format": MODEL_FORMAT_REVISION,
            "requires": sorted(features),
        },
        "model": body,
    }


def expand_model(source: str | dict[str, Any]) -> dict[str, Any]:
    """Expand the ``"plugins"`` section of a model (if any) into core
    material and return the resulting core-schema **body**: the audit
    window on plugin translations (see :mod:`pyraichu.plugins`)."""
    from .plugins import expand_model as _expand

    model = json.loads(source) if isinstance(source, str) else source
    return _expand(model)


def load_model(
    source: str | Path | dict[str, Any],
    *,
    allow_fmu_import: bool = False,
    libraries: Iterable[Any] = (),
    study: Any = None,
) -> Model:
    """Load and validate a model from a JSON string or a dict, in either
    document shape (bare body, or body under the format envelope).

    A system document (``raichu_system``) is expanded from ``libraries``
    and the optional ``study`` (see :mod:`pyraichu.library`).

    Models carrying a ``"plugins"`` section (specialized object schemas:
    ObjFlow, ObjFM, ObjEvent, …) are expanded into the core schema
    first; use :func:`expand_model` to inspect the translation.
    Raises :class:`ModelError` with a precise, typed message when the
    model is invalid (never a crash).
    """
    base_dir = Path.cwd()
    if isinstance(source, Path) or (
        isinstance(source, str) and not source.lstrip().startswith("{")
    ):
        path = Path(source).resolve()
        base_dir = path.parent
        document = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        document = json.loads(source)
    else:
        document = source
    if "raichu_system" in document:
        return load_system(
            document,
            libraries,
            study,
            allow_fmu_import=allow_fmu_import,
            base_dir=base_dir,
        )
    if tuple(libraries) or study is not None:
        raise ModelError(
            "libraries and study apply to a component type library system "
            "(a `raichu_system` document); this document is not one"
        )
    # What the document declares survives the plugin expansion that
    # replaces its body: a declaration this engine cannot honour has to
    # reach the engine, which is where it is refused by name.
    declared = document.get(MODEL_ENVELOPE_KEY, {}).get("requires", [])
    body = model_body(document)
    if "plugins" in body:
        body = expand_model(document)
    units = body.get("fmu_units", [])
    if units and not allow_fmu_import:
        raise ModelError(
            f"FMU import for unit {units[0].get('name', '<unnamed>')!r} "
            "requires allow_fmu_import=True"
        )
    # Sealing is a no-op for a body using baseline constructs only and
    # declaring nothing, so the whole existing corpus keeps its exact
    # document shape.
    model_json = json.dumps(seal(body, declared))
    validate_model(model_json)
    name = body.get("name", "")
    return Model(
        json=model_json,
        name=name,
        allow_fmu_import=allow_fmu_import,
        base_dir=base_dir,
    )


def _series_dict(raw_series: list[dict[str, Any]]) -> dict[str, list[tuple[float, Any]]]:
    return {
        series["name"]: [(t, _value_to_python(v)) for t, v in series["points"]]
        for series in raw_series
    }


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bounds of one estimator, with the level and construction that
    produced them.

    ``method`` is ``"wilson"`` for an estimator that is a probability by
    declaration (a state indicator, or a boolean attribute), ``"normal"``
    for a general mean (cumulated sojourn, occurrence count), and
    ``"undefined"`` when fewer than two replicas made any interval
    impossible. ``level`` is the study parameter the run declared, kept
    here so a figure is never reported without the precision it is
    claimed at.

    ``constant_sample`` marks, instant by instant, where every replica
    returned the same value. No dispersion was observed there, so the
    bound is the one the *frequency* of a departure establishes,
    ``z**2 / (n + z**2)``, and not one built on an observed spread. The
    construction named by ``method`` is unchanged: a draw without
    dispersion is not a declaration of type.
    """

    level: float
    method: str
    low: list[float]
    high: list[float]
    constant_sample: list[bool]

    def half_width(self, index: int = 0) -> float:
        """Half the width of the interval at one schedule instant: the
        ``±`` a report prints next to the estimate."""
        return 0.5 * (self.high[index] - self.low[index])


def _interval(raw: dict[str, Any]) -> ConfidenceInterval:
    return ConfidenceInterval(
        level=raw["level"],
        method=raw["method"],
        low=raw["low"],
        high=raw["high"],
        constant_sample=raw["constant_sample"],
    )


@dataclass(frozen=True)
class Extremes:
    """The smallest and the largest value a measure took across the
    replicas, at each schedule instant (the ``min`` and ``max`` statistics)."""

    min: list[float]
    max: list[float]


@dataclass(frozen=True)
class IndicatorEstimate:
    """Monte-Carlo estimates of one indicator over the schedule."""

    name: str
    instants: list[float]
    mean: list[float]
    std: list[float]
    ci: ConfidenceInterval
    sojourn_mean: list[float]
    sojourn_std: list[float]
    sojourn_ci: ConfidenceInterval
    nb_occurrences_mean: list[float]
    nb_occurrences_std: list[float]
    nb_occurrences_ci: ConfidenceInterval
    #: Probability of having been active at least once by each instant (the
    #: RAMS "had value" measure): the first-entry distribution, which stays
    #: at 1 on a trajectory after the indicator falls back.
    reached_mean: list[float]
    reached_std: list[float]
    reached_ci: ConfidenceInterval
    extremes: Extremes
    sojourn_extremes: Extremes
    nb_occurrences_extremes: Extremes
    reached_extremes: Extremes
    quantiles: dict[float, list[float]]
    sojourn_quantiles: dict[float, list[float]]
    #: Mean number of departures from zero by each instant: moves of the
    #: value from exactly 0 to any non-zero value, sign ignored, the initial
    #: value never counted (the reference engine's ``nb_visits``, which a
    #: cod3s/muscadet ``nb-occurrences`` indicator is defined against).
    #: Unlike ``nb_occurrences_mean`` it does not count an active initial
    #: value, and it counts a move to a negative value.
    zero_departures_mean: list[float] = field(default_factory=list)
    zero_departures_std: list[float] = field(default_factory=list)
    zero_departures_ci: ConfidenceInterval = field(default_factory=lambda: _absent_interval())
    zero_departures_extremes: Extremes = field(default_factory=lambda: Extremes(min=[], max=[]))
    #: Probability of having been non-zero at least once by each instant,
    #: the initial value included (the reference engine's ``realized``,
    #: which a cod3s/muscadet ``had_value`` indicator is defined against).
    #: Unlike ``reached_mean`` it holds for a value that is only ever
    #: negative.
    nonzero_reached_mean: list[float] = field(default_factory=list)
    nonzero_reached_std: list[float] = field(default_factory=list)
    nonzero_reached_ci: ConfidenceInterval = field(default_factory=lambda: _absent_interval())
    nonzero_reached_extremes: Extremes = field(default_factory=lambda: Extremes(min=[], max=[]))


def _absent_interval() -> ConfidenceInterval:
    """The interval a result written before the departure measures existed
    reads back with: no instant and no construction (``"undefined"``)."""
    return ConfidenceInterval(level=0.95, method="undefined", low=[], high=[], constant_sample=[])


def _optional_interval(raw: dict[str, Any] | None) -> ConfidenceInterval:
    return _absent_interval() if raw is None else _interval(raw)


def _optional_extremes(raw: dict[str, Any] | None) -> Extremes:
    return Extremes(min=[], max=[]) if raw is None else Extremes(**raw)


@dataclass(frozen=True)
class McEstimates:
    """Monte-Carlo result: per-indicator estimates + provenance."""

    indicators: dict[str, IndicatorEstimate]
    nb_runs: int
    seed: int
    confidence: float
    engine_version: str
    fmu_units: list[dict[str, Any]] = field(default_factory=list)
    serial_fallback_unit: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"McEstimates({self.nb_runs} runs, seed={self.seed}, "
            f"{len(self.indicators)} indicators, "
            f"CI at {self.confidence:.0%})"
        )


def _mc_estimates(raw: dict[str, Any]) -> McEstimates:
    """The Python object of a Monte-Carlo result document."""
    indicators = {
        e["name"]: IndicatorEstimate(
            name=e["name"],
            instants=e["instants"],
            mean=e["mean"],
            std=e["std"],
            ci=_interval(e["ci"]),
            sojourn_mean=e["sojourn_mean"],
            sojourn_std=e["sojourn_std"],
            sojourn_ci=_interval(e["sojourn_ci"]),
            nb_occurrences_mean=e["nb_occurrences_mean"],
            nb_occurrences_std=e["nb_occurrences_std"],
            nb_occurrences_ci=_interval(e["nb_occurrences_ci"]),
            reached_mean=e["reached_mean"],
            reached_std=e["reached_std"],
            reached_ci=_interval(e["reached_ci"]),
            extremes=Extremes(**e["extremes"]),
            sojourn_extremes=Extremes(**e["sojourn_extremes"]),
            nb_occurrences_extremes=Extremes(**e["nb_occurrences_extremes"]),
            reached_extremes=Extremes(**e["reached_extremes"]),
            quantiles={s["q"]: s["values"] for s in e["quantiles"]},
            sojourn_quantiles={s["q"]: s["values"] for s in e["sojourn_quantiles"]},
            # Read with a fallback: a result document written before the
            # departure measures existed (a Monte-Carlo detail of an older
            # quantification envelope) carries none of them.
            zero_departures_mean=e.get("zero_departures_mean", []),
            zero_departures_std=e.get("zero_departures_std", []),
            zero_departures_ci=_optional_interval(e.get("zero_departures_ci")),
            zero_departures_extremes=_optional_extremes(e.get("zero_departures_extremes")),
            nonzero_reached_mean=e.get("nonzero_reached_mean", []),
            nonzero_reached_std=e.get("nonzero_reached_std", []),
            nonzero_reached_ci=_optional_interval(e.get("nonzero_reached_ci")),
            nonzero_reached_extremes=_optional_extremes(e.get("nonzero_reached_extremes")),
        )
        for e in raw["indicators"]
    }
    return McEstimates(
        indicators=indicators,
        nb_runs=raw["nb_runs"],
        seed=raw["seed"],
        confidence=raw["confidence"],
        engine_version=raw["engine_version"],
        fmu_units=raw.get("fmu_units", []),
        serial_fallback_unit=raw.get("serial_fallback_unit"),
    )


def monte_carlo(
    model: Model,
    nb_runs: int,
    t_max: float,
    samples: list[float],
    seed: int = 0,
    threads: int | None = None,
    quantiles: list[float] | None = None,
    confidence: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_step: float | None = None,
    tol_event: float | None = None,
    sub_samples: int | None = None,
    stop_at_targets: bool = False,
    flow: FlowConfig | None = None,
    event_resolution: float | None = None,
    require_parallel: bool = False,
) -> McEstimates:
    """Estimate indicator statistics over ``nb_runs`` replicas.

    Replica ``r`` uses RNG substream ``r`` of ``seed``; the reduction is
    index-ordered, so the result is byte-identical for any ``threads``
    value. ``quantiles`` (e.g. ``[0.25, 0.75]``) adds nearest-rank
    quantile series on both the sampled value and the cumulated
    sojourn. The GIL is released while the replicas run.

    A replica whose run fails refuses the whole campaign: an estimate over
    the other replicas would leave out the trajectory that failed, which
    is often one reaching the feared event. The error raised is the one
    the replica's own run raised (an :class:`UnboundedRateError` stays
    one), names the replica in its message, and carries ``replica``,
    ``nb_runs`` and ``seed`` as attributes: :func:`simulate` with the
    campaign's own horizon, samples, flow policy, solver keywords and
    ``stop_at_targets``, at ``seed=error.seed, rng_stream=error.replica``,
    replays it alone. A refusal raised before any replica runs (an
    invalid setting, an FMU permission) carries none of these
    attributes. The replica reported is the
    lowest-numbered one that failed, whatever ``threads`` is, so every
    replica numbered below it completed.

    ``confidence`` is the confidence level of the interval every
    estimator carries, a **study parameter** in ``(0, 1)``: state
    ``0.99`` and the whole result is reported at 99 %. Omitted, it is
    :data:`DEFAULT_CONFIDENCE` (0.95). The level applied comes back on
    :attr:`McEstimates.confidence` and on every
    :class:`ConfidenceInterval`, so a figure never travels without the
    precision it is claimed at. A level outside ``(0, 1)`` raises
    :class:`SimulationError` before the replicas run.

    ``rtol``/``atol``/``max_step``/``tol_event``/``sub_samples``
    override the ODE-backend parameters (engine defaults when omitted):
    the integration-effort knobs of the tolerance-parity experiments.

    ``event_resolution`` is the widest spacing a study accepts between two
    points of the crossing scan: a floor on resolution. A crossing that
    holds for less than the engine's own spacing (``max_step /
    sub_samples`` at most) is still bracketed when a finer one is asked;
    a coarser request changes nothing, bit for bit. A non-positive or
    non-finite value raises :class:`SimulationError`.

    ``stop_at_targets=True`` early-stops each trajectory at the first
    sequence target (feared event) and holds the frozen state through the
    remaining sample instants: the latch semantics of target-stopped
    studies (first-occurrence measures instead of free-cycling ones).

    ``flow`` is a :class:`FlowConfig` overriding the convergence policy
    of the continuous flow resolution for every replica (engine defaults
    when omitted).

    ``require_parallel=True`` refuses a model importing an FMU that
    allows only one instance per process, instead of running its
    replicas one at a time.
    """
    raw = json.loads(
        monte_carlo_json(
            model.json,
            nb_runs,
            t_max,
            samples,
            seed,
            threads,
            quantiles,
            confidence,
            rtol,
            atol,
            max_step,
            tol_event,
            sub_samples,
            stop_at_targets,
            flow,
            event_resolution,
            model.allow_fmu_import,
            str(model.base_dir),
            require_parallel,
        )
    )
    return _mc_estimates(raw)


class ReductionWarning(UserWarning):
    """The minimal-sequence reduction had to depart from its hypothesis.

    Cancelling a failure that was repaired before the feared event assumes
    the repaired component no longer contributes. Some trajectories proved
    otherwise: cancelling their cycles would leave the feared event with no
    recorded cause, so they were reduced with their cycles kept. Either a
    repaired failure had a lasting effect (a storage that drained during the
    outage, for example), or the feared event has a cause that no monitored
    transition records.
    The result stays correct; the warning says that the model holds such an
    effect. :func:`run_sequences` and :meth:`Exploration.minimal_sequences_with_report`
    return the full report.
    """


def _warn_reduction(reduction: dict[str, Any], full_report: str) -> None:
    """Emit a :class:`ReductionWarning` when the reduction refused a cancellation,
    or cancelled cycle groups without knowing whether it could; ``full_report``
    names where the caller finds the whole report."""
    count = reduction["contradicted"]
    if count:
        if reduction.get("cycle_groups") is None:
            cause = "Either a repaired failure has a lasting effect in this model"
        else:
            cause = (
                "Either a repaired failure has an effect the per-group decision does not see "
                "(a latch or a counter a sensitive function keeps; declare "
                "`cycle_cancellation: never` on its group)"
            )
        warnings.warn(
            f"{count} trajectories (weight {reduction['contradicted_weight']:g}) reached a feared "
            "event that cancelling their repaired failures would leave without a cause; they were "
            f"reduced with their failure/repair cycles kept. {cause}, or the feared event has a "
            f"cause that no monitored transition records: {full_report} returns the full "
            "reduction report.",
            ReductionWarning,
            stacklevel=3,
        )
    if reduction.get("cycle_groups_unknown"):
        warnings.warn(
            "this input was written before the per-group decisions were recorded, so every cycle "
            "group was cancelled, including any whose repaired failure has a lasting effect; "
            "re-run the campaign or the exploration with this engine to apply them.",
            ReductionWarning,
            stacklevel=3,
        )


def analyse_sequences(
    model: Model,
    nb_runs: int,
    t_max: float,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
) -> list[dict[str, Any]]:
    """Native minimal-sequence analysis: the RAMS output.

    Run ``nb_runs`` sequence-recording replicas (each early-stopping at the
    first feared-event target), then group → filter transient failure/repair
    cycles → greedily absorb super-sequences, returning the **minimal
    sequences** cod3s produces through its ``SequenceAnalyser``. Each entry is
    ``{events: [{obj, automaton, attr, time, kind}], end_cause, end_time,
    weight}`` (``weight`` = the number of trajectories that collapsed into
    it; ``automaton`` and ``kind`` as :class:`SequenceCampaign` describes
    them). The model must declare at least one target (an ``ObjEvent`` with
    ``"target": true``).

    ``flow`` is a :class:`FlowConfig` overriding the convergence policy
    of the continuous flow resolution for every replica (engine defaults
    when omitted).

    A trajectory that cancelling its repaired failures would leave without a
    cause keeps its cycles, and a :class:`ReductionWarning` says how many did.
    """
    result = json.loads(analyse_sequences_json(model.json, nb_runs, t_max, seed, threads, flow, True))
    _warn_reduction(result["reduction"], "run_sequences")
    return result["minimal"]


@dataclass(frozen=True)
class Observation:
    """A quantity a sequence campaign reads on every trajectory: the value of
    ``component.attribute`` at ``time``.

    The raw corpus carries it under ``name``, which a
    :class:`SequenceCondition` refers to. The value is the state the
    trajectory was in at that instant: after any event at that date, and the
    final state for a trajectory that stopped at a feared event earlier. An
    instant past the horizon reads the horizon. A boolean reads ``0`` or
    ``1``.
    """

    name: str
    component: str
    attribute: str
    time: float


#: The comparisons a :class:`SequenceCondition` accepts.
CONDITION_OPS = ("==", "!=", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class SequenceCondition:
    """Keep a trajectory when its observed value of ``observation`` compares
    to ``value`` as ``op`` says (one of ``==``, ``!=``, ``<``, ``<=``, ``>``,
    ``>=``). The comparison is on the recorded double, exactly.
    """

    observation: str
    op: str
    value: float

    def __post_init__(self) -> None:
        if self.op not in CONDITION_OPS:
            raise ValueError(f"{self.op!r} is not a comparison; use one of {', '.join(CONDITION_OPS)}")

    def _arg(self) -> tuple[str, str, float]:
        return (self.observation, self.op, float(self.value))


@dataclass
class SequenceCampaign:
    """A sequence campaign kept whole: its two reduced levels, and where its
    raw corpus was written.

    ``cleaned`` is every distinct path to each feared event, transient
    failure/repair cycles removed (cod3s's ``sequences_all.json`` level);
    ``minimal`` is what :func:`analyse_sequences` returns. Both are lists of
    ``{events: [{obj, automaton, attr, time, kind}], end_cause, end_time,
    weight}``. An event's ``automaton`` is the automaton of ``obj`` whose
    state ``attr`` it entered (``None`` when the source names none: a corpus
    written before 0.83.0); its ``kind`` is the role the model declares for
    the transition, ``"failure"``, ``"repair"`` or ``"observation"``, as the
    raw corpus records it. ``kind`` is ``None`` when the transition declares
    none, on a corpus written before 0.85.0, and on a cleaned event whose
    merged trajectories do not all declare the same kind. The reduction
    merges events that match on ``(obj, attr, automaton)``.

    ``raw_path`` is the ``raichu.sequences`` corpus, one line per trajectory
    (see the sequence-format reference), or ``None`` when none was asked for.
    ``header`` is that corpus's first line when it was read back by
    :func:`analyse_raw_sequences`, ``None`` otherwise.

    ``condition`` reports the :class:`SequenceCondition` the levels were
    reduced under, ``{observation, op, value, total_trajectories,
    kept_trajectories}``, or is ``None`` when every trajectory was reduced.
    The raw corpus always holds every trajectory.

    ``reduction`` reports what the reduction did beyond its two steps:
    ``contradicted`` trajectories reached a feared event that cancelling
    their repaired failures would leave without a cause, so they kept their
    cycles; ``contradicted_weight`` is their total weight and
    ``contradicted_indices`` the replica indices of the first hundred.
    """

    cleaned: list[dict[str, Any]]
    minimal: list[dict[str, Any]]
    raw_path: Path | None = None
    header: dict[str, Any] | None = None
    condition: dict[str, Any] | None = None
    reduction: dict[str, Any] | None = None


def run_sequences(
    model: Model,
    nb_runs: int,
    t_max: float,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
    raw_path: str | Path | None = None,
    observations: Iterable[Observation] = (),
    condition: SequenceCondition | None = None,
) -> SequenceCampaign:
    """A sequence campaign whose raw corpus is kept.

    Runs the campaign :func:`analyse_sequences` runs, with the same seed
    giving the same trajectories, and returns both reduced levels. When
    ``raw_path`` is given, every trajectory's raw sequence is written there in
    the ``raichu.sequences`` format (JSON Lines: a header line, then one line
    per replica in replica order), straight from the engine: a campaign of any
    size never becomes Python objects. :func:`analyse_raw_sequences` reads it
    back and recomputes the reduction.

    ``observations`` are read on every trajectory and written into the raw
    corpus; they do not change the trajectories. ``condition`` reduces only
    the trajectories it holds on, and names one of ``observations``.
    """
    path = None if raw_path is None else Path(raw_path)
    observed = [(o.name, o.component, o.attribute, float(o.time)) for o in observations]
    levels = json.loads(
        run_sequences_json(
            model.json,
            nb_runs,
            t_max,
            seed,
            threads,
            flow,
            None if path is None else str(path),
            observed or None,
            None if condition is None else condition._arg(),
        )
    )
    return SequenceCampaign(
        cleaned=levels["cleaned"],
        minimal=levels["minimal"],
        raw_path=path,
        condition=levels.get("condition"),
        reduction=levels["reduction"],
    )


def analyse_raw_sequences(raw_path: str | Path, condition: SequenceCondition | None = None) -> SequenceCampaign:
    """Read a ``raichu.sequences`` corpus and reduce it again.

    The reduction is the engine's own, so a corpus written by
    :func:`run_sequences` gives back exactly the levels that campaign
    returned. With ``condition``, only the trajectories it holds on are
    reduced; it must name an observation the corpus carries. Refuses another
    format and a version newer than this engine reads.
    """
    path = Path(raw_path)
    levels = json.loads(analyse_raw_sequences_json(str(path), None if condition is None else condition._arg()))
    return SequenceCampaign(
        cleaned=levels["cleaned"],
        minimal=levels["minimal"],
        raw_path=path,
        header=levels["header"],
        condition=levels.get("condition"),
        reduction=levels["reduction"],
    )


@dataclass(frozen=True)
class ExploredSequence:
    """One retained sequence of an exploration: an ordered path from the
    initial state to the target in which nothing else happens (not a
    minimal cut sequence, which is a reduction over many such paths).

    ``steps`` holds the fired transitions in firing order as indices into
    the result's shared step table (``Exploration.steps``), so a result
    stays proportional to the total number of fired transitions as small
    integers. ``transitions`` and ``events`` resolve them on demand:
    ``transitions`` lists every fired transition, each
    ``{transition, from, to}``; ``events`` lists the monitored-state
    entries among them, each ``{obj, attr, cycle_group}``, in the
    vocabulary of the Monte-Carlo sequence corpus minus the date (an exact
    exploration chooses the order of the jumps, not their dates).
    ``probability`` is the probability that a trajectory follows exactly
    this path and reaches the target by the horizon; ``error_bound``
    bounds its absolute error, and ``imprecise`` is ``True`` when that
    bound exceeds the declared relative precision.
    """

    steps: tuple[int, ...]
    end_cause: str
    probability: float
    error_bound: float
    imprecise: bool
    _table: tuple[dict[str, Any], ...] = field(repr=False, compare=False, default=())

    @property
    def transitions(self) -> list[dict[str, str]]:
        """Every fired transition, in firing order, each
        ``{transition, from, to}``."""
        return [
            {"transition": step["transition"], "from": step["from"], "to": step["to"]}
            for step in (self._table[i] for i in self.steps)
        ]

    @property
    def events(self) -> list[dict[str, Any]]:
        """The monitored-state entries, in firing order, each
        ``{obj, attr, cycle_group}``."""
        return [
            dict(event)
            for event in (self._table[i]["event"] for i in self.steps)
            if event is not None
        ]

    @classmethod
    def _from_dict(cls, raw: dict[str, Any], table: tuple[dict[str, Any], ...]) -> ExploredSequence:
        return cls(
            steps=tuple(raw["steps"]),
            end_cause=raw["end_cause"],
            probability=raw["probability"],
            error_bound=raw["error_bound"],
            imprecise=raw["imprecise"],
            _table=table,
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "steps": list(self.steps),
            "end_cause": self.end_cause,
            "probability": self.probability,
            "error_bound": self.error_bound,
            "imprecise": self.imprecise,
        }


#: Fields of an :class:`Exploration`, in the order of the
#: ``raichu.exploration`` format; ``steps`` and ``sequences`` are
#: converted separately.
_EXPLORATION_FIELDS = (
    "format",
    "version",
    "engine_version",
    "model",
    "algorithm",
    "target",
    "horizon",
    "cutoffs",
    "gap_tolerance",
    "precision",
    "steps",
    "sequences",
    "lower",
    "upper",
    "cutoff_tallies",
    "inconclusive",
    "expanded_nodes",
    "imprecise_sequences",
)


@dataclass(frozen=True)
class Exploration:
    """The result of a sequence-tree exploration (:func:`explore`), in its
    open format, ``raichu.exploration`` (version 1 for an exact result,
    version 2 for a discretised one).

    ``sequences`` are the retained :class:`ExploredSequence` s by
    decreasing probability; ``steps`` is the table they index, one entry
    per distinct fired transition and destination,
    ``{transition, from, to, event}`` with ``event`` ``None`` when the
    transition is not monitored. They are disjoint events, so ``lower``, the
    sum of their probabilities, is a lower bound on the probability of
    reaching ``target`` by ``horizon``; ``upper`` adds the mass every
    cut-off discarded, and ``cutoff_tallies`` says what each one pruned
    (``{name: {pruned_nodes, mass}}``). ``inconclusive`` is ``True`` when
    the relative gap ``(upper - lower) / upper`` exceeds
    ``gap_tolerance``. ``cutoffs`` and ``precision`` record the settings,
    ``algorithm`` the driver, ``expanded_nodes`` the work done, and
    ``imprecise_sequences`` how many probabilities are not guaranteed to
    the declared precision.

    ``discretisation`` is ``None`` for an exact result (and absent from its
    document). For a discretised one it is ``{level, refinement}``:
    ``level`` is the number of equal-mass cells behind the reported
    numbers, and ``refinement`` is ``None`` when no error estimate was
    made, otherwise ``{base_level, base_lower, base_upper,
    error_estimate, truncation_dominated}``. The probabilities and bounds
    of a discretised result are those of the discretised model;
    :attr:`error_estimate` is the estimate of the discretisation error.
    """

    format: str
    version: int
    engine_version: str
    model: str
    algorithm: str
    target: str
    horizon: float
    cutoffs: dict[str, Any]
    gap_tolerance: float
    precision: dict[str, Any]
    steps: tuple[dict[str, Any], ...]
    sequences: list[ExploredSequence]
    lower: float
    upper: float
    cutoff_tallies: dict[str, dict[str, Any]]
    inconclusive: bool
    expanded_nodes: int
    imprecise_sequences: int
    discretisation: dict[str, Any] | None = None
    cycle_groups: list[dict[str, Any]] | None = None

    @classmethod
    def _from_json(cls, text: str) -> Exploration:
        raw = json.loads(text)
        fields = {name: raw[name] for name in _EXPLORATION_FIELDS}
        table = tuple(raw["steps"])
        fields["steps"] = table
        fields["sequences"] = [ExploredSequence._from_dict(s, table) for s in raw["sequences"]]
        fields["discretisation"] = raw.get("discretisation")
        fields["cycle_groups"] = raw.get("cycle_groups")
        return cls(**fields)

    @property
    def relative_gap(self) -> float:
        """``(upper - lower) / upper``, 0 when ``upper`` is 0."""
        return (self.upper - self.lower) / self.upper if self.upper > 0 else 0.0

    @property
    def error_estimate(self) -> float | None:
        """The discretisation error estimate by refinement (a probability):
        ``None`` for an exact result, and for a discretised one whose
        refinement was switched off (no estimate was made). It is an
        estimate, not a bound."""
        if self.discretisation is None or self.discretisation["refinement"] is None:
            return None
        return self.discretisation["refinement"]["error_estimate"]

    def to_json(self, path: str | Path | None = None) -> str:
        """The result in the ``raichu.exploration`` format, as JSON text;
        also written to ``path`` when one is given. :func:`read_exploration`
        reads it back into an equal object."""
        document = {name: getattr(self, name) for name in _EXPLORATION_FIELDS}
        document["steps"] = list(self.steps)
        document["sequences"] = [s._to_dict() for s in self.sequences]
        if self.discretisation is not None:
            document["discretisation"] = self.discretisation
        if self.cycle_groups is not None:
            document["cycle_groups"] = self.cycle_groups
        text = json.dumps(document)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def minimal_sequences(self) -> list[dict[str, Any]]:
        """The minimal sequences of this result, through the engine's own
        reduction (group, filter failure/repair cycles, absorb
        super-sequences), as :func:`analyse_sequences` returns them.

        Each ``weight`` is a **probability** here, not a replica count: the
        retained sequences are disjoint, so the weights that collapse into
        one minimal sequence add up, and they total ``lower``. The dates
        are 0: the exact driver never moves the clock, and a discretised
        sequence merges cells fired at different instants, which is also
        why an exploration is not an input to date-based post-processing.

        A retained sequence that cancelling its repaired failures would leave
        without a cause keeps its cycles, and a :class:`ReductionWarning` says
        how many did; :meth:`minimal_sequences_with_report` returns the report.
        """
        minimal, reduction = self.minimal_sequences_with_report()
        _warn_reduction(reduction, "Exploration.minimal_sequences_with_report")
        return minimal

    def minimal_sequences_with_report(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """:meth:`minimal_sequences`, and the reduction report.

        The report is ``{contradicted, contradicted_weight,
        contradicted_indices, contradicted_share}``: the retained sequences
        whose cycles were kept because cancelling them would leave the
        feared event without a cause, their probability mass, the indices
        of the first hundred in :attr:`sequences`, and that mass as a share
        of ``lower``. No warning is emitted.
        """
        result = json.loads(exploration_minimal_sequences_json(self.to_json(), True))
        return result["minimal"], result["reduction"]


def explore(
    model: Model,
    target: str,
    horizon: float,
    *,
    algorithm: str = "exact",
    min_probability: float | None = None,
    max_length: int | None = None,
    max_failures: int | None = None,
    max_branches: int | None = None,
    gap_tolerance: float | None = None,
    rel_precision: float | None = None,
    max_terms: int | None = None,
    threads: int | None = None,
    level: int | None = None,
    refine: bool = True,
) -> Exploration:
    """Explore the sequence tree of ``model`` to the feared event
    ``target`` (a declared target), instead of drawing Monte-Carlo
    replicas.

    Returns every retained sequence reaching the target, with its
    probability at ``horizon`` (in the model's time unit), and guaranteed
    bounds on what the cut-offs left out. Every cut-off is optional:
    ``min_probability`` prunes a prefix whose probability of completing by
    the horizon falls below it, ``max_length`` bounds the fired
    transitions of a sequence, ``max_failures`` the fired transitions of
    declared kind ``failure``, and ``max_branches`` the expanded nodes.
    ``gap_tolerance`` (default 0.01) is the relative gap above which the
    result is flagged inconclusive; ``rel_precision`` and ``max_terms``
    override the numerical precision of the sequence probabilities.
    ``threads`` sets the worker count; the result does not depend on it.

    ``algorithm`` selects the driver:

    - ``"exact"`` (default): the Markov family (instantaneous branchings,
      exponential laws whose rate is constant between jumps, no continuous
      evolution), with every probability in closed form.
    - ``"discretised"``: every law the engine carries and continuous
      evolution. At each explored state the distribution of the next event
      (which armed transition fires first, and when) is cut into ``level``
      cells of equal probability (default 8). With ``refine`` (default)
      the run is repeated at ``2 x level``, which is the reported one, and
      the difference between the two is the error estimate
      (:attr:`Exploration.error_estimate`, an estimate, not a bound); with
      ``refine=False`` no estimate is made. ``max_branches`` defaults to
      1 000 000 expanded nodes per pass when omitted.

    ``level`` and ``refine=False`` apply to discretised exploration only,
    ``rel_precision`` and ``max_terms`` to exact exploration only; passing
    one to the other method raises. Raises :class:`SimulationError` for an
    unknown algorithm or an invalid setting (before anything runs), for a
    model outside the method's domain (see :func:`exploration_domain`
    for exact exploration), and, in exact exploration, when a law outside
    its domain becomes armed, naming the transition and the sequence that
    armed it.
    """
    return Exploration._from_json(
        explore_json(
            model.json,
            target,
            horizon,
            algorithm,
            min_probability,
            max_length,
            max_failures,
            max_branches,
            gap_tolerance,
            rel_precision,
            max_terms,
            threads,
            level,
            refine,
            model.allow_fmu_import,
            str(model.base_dir),
        )
    )


def read_exploration(source: str | Path) -> Exploration:
    """Read an exploration result written by :meth:`Exploration.to_json`.

    ``source`` is a path (a :class:`~pathlib.Path`, or a string that is
    not JSON text) or the JSON text itself. Raises
    :class:`SimulationError` for another format or a version newer than
    this engine reads.
    """
    if isinstance(source, Path) or not source.lstrip().startswith("{"):
        source = Path(source).read_text(encoding="utf-8")
    validate_exploration(source)
    return Exploration._from_json(source)


def exploration_domain(model: Model) -> list[dict[str, Any]]:
    """Screen ``model`` for the exact exploration domain without exploring
    it: every reason found statically that it is outside (an ODE, a
    watched transition its automaton can reach, an expression reading
    time), each ``{kind, ..., message}``. Empty when none is found; a law
    outside the domain (delay, Weibull, ...) is only found during a run,
    and only when it becomes armed.
    """
    return json.loads(exploration_domain_json(model.json))


#: The ``format`` of a fault-tree envelope (:meth:`FaultTree.envelope`).
FAULT_TREE_FORMAT = "raichu.fault_tree"
#: The envelope version this engine writes, and the highest it reads
#: (version 1 reads too, see :func:`read_fault_tree`).
FAULT_TREE_VERSION = 2
#: The independent structural output format and version (version 1 reads
#: too, see :func:`read_fault_tree_structure`).
FAULT_TREE_STRUCTURE_FORMAT = "raichu.fault_tree.structure"
FAULT_TREE_STRUCTURE_VERSION = 2
#: The most mission times one envelope quantifies.
MAX_MISSION_TIMES = 20
#: What a fault tree's probabilities measure: the probability without
#: repair (the default), or the unavailability counting the repairs.
FAULT_TREE_MEASURES = ("probability_without_repair", "unavailability")


@dataclass(frozen=True)
class FaultTree:
    """A fault tree generated by backward chaining (:func:`fault_tree`)."""

    #: The top gate, nested: ``{"node": "gate", "gate": "and" | "or" |
    #: "at_least", "k": ..., "id": ..., "origin": ..., "children": [...]}``
    #: or ``{"node": "basic", "event": index}``. Never a constant: a
    #: degenerate tree is refused. ``id`` is the gate's occurrence
    #: identifier, unique in the tree and stable from one generation to the
    #: next (``"top"``, then ``"G1"``, ``"G2"``, ... in preorder), the name
    #: the OpenPSA document gives it. ``origin``, when present, is what the
    #: gate stands for: ``{"kind": "attribute" | "state" | "transition",
    #: "component", "name", "negated"}``, ``negated`` meaning the gate holds
    #: when the attribute is false. An origin may repeat: an attribute read
    #: in two places is two gates.
    top: dict[str, Any]
    #: The basic events, each a transition's draw: ``name``, ``component``,
    #: ``automaton``, ``transition``, ``target``, its ``law`` with the
    #: law's parameters, and its ``repairs``: the transitions back from
    #: ``target`` to the state the event leaves, each ``{"transition",
    #: "law", "guarded"}`` (``law`` is ``None`` when no single law stands
    #: for it).
    basic_events: list[dict[str, Any]]
    #: The minimal cut sets, each a sorted list of basic-event names, by
    #: size then name; None when extraction was not requested.
    minimal_cut_sets: list[list[str]] | None
    #: The tree in the OpenPSA model-exchange format.
    open_psa: str
    #: Why the tree's probability may exceed the model's own probability
    #: without repair, one coded warning per transition and reason concerned
    #: (a draw timed from a state entered during the mission, a guard
    #: reading a state, a competing transition, a state left for another
    #: than the one it came from): ``{"kind": ..., <parameters>, "message":
    #: <English sentence>}``. Key on ``kind``, never on ``message``. Empty
    #: when the tree is the model's exact structure. Repairs are not warned
    #: here: what the measure makes of them is in :meth:`envelope`.
    warnings: list[dict[str, Any]] = field(default_factory=list)
    #: The measure the tree was generated for (one of
    #: :data:`FAULT_TREE_MEASURES`), named in :attr:`open_psa`; the default
    #: of :meth:`quantify` and :meth:`envelope`.
    measure: str = "probability_without_repair"
    #: What the tree was generated from, for :meth:`envelope`.
    _source: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    _structure: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def structure(self) -> dict[str, Any]:
        """Return the versioned structural output without quantification.

        Omitted cuts are ``None``, distinct from an empty collection.
        """
        if self._structure is None:
            raise SimulationError(
                "fault tree: no producer structural result for this tree"
            )
        return json.loads(json.dumps(self._structure))

    def quantify(
        self,
        *,
        mission_time: float | None = None,
        max_bdd_nodes: int = 10_000_000,
        cut_set_limit: int = 100_000,
        cut_sets: bool = True,
        engine: str = "auto",
        max_order: int | None = None,
        min_cut_probability: float = 0.0,
        max_cut_sets: int = 1_000_000,
        max_expansions: int = 100_000_000,
        measure: str | None = None,
    ) -> FaultTreeQuantification:
        """Quantify this tree; see :func:`quantify`. ``measure`` defaults to
        the tree's own (:attr:`measure`)."""
        return _quantify_fault_tree(
            self.open_psa,
            measure=measure,
            mission_time=mission_time,
            max_bdd_nodes=max_bdd_nodes,
            cut_set_limit=cut_set_limit,
            cut_sets=cut_sets,
            engine=engine,
            max_order=max_order,
            min_cut_probability=min_cut_probability,
            max_cut_sets=max_cut_sets,
            max_expansions=max_expansions,
        )

    def envelope(
        self,
        mission_times: Iterable[float],
        *,
        max_bdd_nodes: int = 10_000_000,
        cut_set_limit: int = 100_000,
        cut_sets: bool = True,
        engine: str = "auto",
        max_order: int | None = None,
        min_cut_probability: float = 0.0,
        max_cut_sets: int = 1_000_000,
        max_expansions: int = 100_000_000,
        measure: str | None = None,
    ) -> dict[str, Any]:
        """The ``raichu.fault_tree`` envelope of this tree, as a dictionary.

        The tree is quantified at each of ``mission_times`` (1 to
        :data:`MAX_MISSION_TIMES` instants, strictly increasing); the last
        one is the **horizon**, the only instant whose minimal cut sets and
        importance measures are computed. Every probability is in
        ``measure`` (default: the tree's own, :attr:`measure`), which the
        envelope names: ``"probability_without_repair"``, the top's
        probability without repair at its mission time, an upper bound of
        the probability that it has occurred on the model with its
        repairs; or ``"unavailability"``, the probability that it holds
        then, counting the repairs. The warnings that depend on the measure
        (``repair_ignored``, ``asymptotic_unavailability``, ...) are in
        ``generation.warnings``, once per basic event, after
        :attr:`warnings`; ``generation.exact`` says whether there are none.
        Every instant, and the horizon, carries ``gates``: the probability
        of every gate at that time, in ``measure``, each ``{"id",
        "probability", "reason"}``, the top's first. ``id`` is the gate's
        identifier in :attr:`top`; ``probability`` is exact, or ``None``
        with its ``reason``: ``"cut_sets"`` (the gate lies in a module
        quantified from its cut sets), ``"reads_cut_sets"`` (it reads one),
        ``"outgrew_bdd"`` or ``"not_reached"``. The other keywords are
        those of :meth:`quantify`. The schema is documented in the
        fault-tree format reference.
        """
        if self._source is None:
            raise SimulationError(
                "fault tree: this tree was not generated by pyraichu.fault_tree, "
                "so there is no model to quantify it against; use quantify()"
            )
        text = fault_tree_envelope_json(
            mission_times=[float(t) for t in mission_times],
            max_bdd_nodes=max_bdd_nodes,
            cut_set_limit=cut_set_limit,
            cut_sets=cut_sets,
            engine=engine,
            max_order=max_order,
            min_cut_probability=min_cut_probability,
            max_cut_sets=max_cut_sets,
            max_expansions=max_expansions,
            measure=self.measure if measure is None else measure,
            **self._source,
        )
        return json.loads(text)


def _fault_tree_document_text(document: str | Path | dict[str, Any]) -> str:
    if isinstance(document, dict):
        return json.dumps(document)
    if isinstance(document, Path) or not str(document).lstrip().startswith("{"):
        return Path(document).read_text(encoding="utf-8")
    return document


def _legacy_warnings(warnings: list[Any]) -> list[Any]:
    """A version-1 document's warnings, each plain sentence as the
    variant ``{"kind": "legacy", "message": <sentence>}``, the form the
    engine's own reader gives it."""
    return [
        {"kind": "legacy", "message": w} if isinstance(w, str) else w
        for w in warnings
    ]


def read_fault_tree(document: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Read a ``raichu.fault_tree`` envelope: a JSON text, the path of a
    file holding one, or an already parsed dictionary.

    The engine's own reader checks it first, refusing another ``format``
    or a ``version`` above :data:`FAULT_TREE_VERSION` with
    :class:`SimulationError`, so a document this engine does not know is
    refused rather than read with empty fields. A version-1 envelope
    reads: its warnings, plain sentences there, come back as
    ``{"kind": "legacy", "message": ...}``, and its instants carry no
    ``gates``.
    """
    text = _fault_tree_document_text(document)
    validate_fault_tree_envelope(text)
    envelope = json.loads(text)
    if envelope["version"] == 1:
        generation = envelope["generation"]
        generation["warnings"] = _legacy_warnings(generation["warnings"])
        for result in [*envelope["instants"], envelope["horizon"]]:
            result["warnings"] = _legacy_warnings(result["warnings"])
    return envelope


def read_fault_tree_structure(document: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Read a structural result, checking its producer format and version.

    Accept a JSON text, file path or dictionary. A version-1 structure
    reads: its warnings, plain sentences there, come back as
    ``{"kind": "legacy", "message": ...}``, and its gates carry no ``id``
    nor ``origin``.
    """
    text = _fault_tree_document_text(document)
    validate_fault_tree_structure(text)
    structure = json.loads(text)
    if structure["version"] == 1:
        tree = structure["tree"]
        tree["warnings"] = _legacy_warnings(tree["warnings"])
    return structure


def _profile_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    return {"kind": "float", "value": float(value)}


def fault_tree(
    model: Model,
    top: dict[str, Any] | None = None,
    *,
    targets: Iterable[str] | None = None,
    profile: dict[str, Any] | None = None,
    max_nodes: int | None = None,
    cut_set_limit: int = 100_000,
    cut_sets: bool = True,
    name: str = "fault_tree",
    measure: str = "probability_without_repair",
) -> FaultTree:
    """Generate the fault tree explaining why ``top``, or one of
    ``targets``, can become true.

    ``measure`` (one of :data:`FAULT_TREE_MEASURES`) is what the tree is
    meant to be quantified in: the default of :meth:`FaultTree.quantify`
    and :meth:`FaultTree.envelope`, named in its OpenPSA document. The
    generation itself does not depend on it: each basic event carries its
    repairs.

    ``cut_sets=False`` skips structural extraction and returns
    ``minimal_cut_sets=None``. Its budget is unused in that case.
    :meth:`FaultTree.structure` publishes the versioned structural result.

    Give exactly one of the two. ``top`` is an expression over the model's
    states and attributes, in the model's own expression vocabulary
    (``state_active``, ``attr``, ``bool``, ``cmp``, ``add``, ``if``,
    ``const``); a vote is ``sum(if(state, 1, 0)) >= k``. ``targets`` names
    targets the model declares (its ``targets`` section, the feared events
    of a study): the top is their disjunction, and an observer's
    transition into a target is crossed as its condition.

    The explanation is backward chaining: a state is entered by a
    transition whose draw is a basic event, from a source state that has
    to be reached, under a guard that has to hold. An attribute a
    sensitive function or an explicit equation computes is **unrolled**
    into what computes it, down to the states (a muscadet flow variable
    is one); an attribute nothing writes keeps its initial value, or the
    value ``profile`` gives it by qualified name (``"component.attribute"``).
    Negations are pushed down to the states, "not in ``ok``" reading as
    "in one of the other states". A loop back to a state being explained
    adds nothing, so a repair loop is handled rather than truncated: the
    repair is attached to its event instead, and counted only under the
    unavailability. :attr:`FaultTree.warnings` says where the tree may
    exceed the model's own probability.

    :class:`SimulationError` is raised, naming the cause, for a
    **degenerate** tree (a top that reduces to a constant once unrolled),
    a top that needs a state the model can leave to persist, a state read
    in arithmetic other than a vote, an attribute that cannot be unrolled
    (computed by an ODE, a transition's effect, an operator, a program,
    or by two writers), and an observer that waits before firing.
    """
    if (top is None) == (targets is None):
        raise SimulationError(
            "fault tree: give either a top expression or targets, not both and not neither"
        )
    source = {
        "model_json": model.json,
        "top_json": None if top is None else json.dumps(top),
        "targets_json": None if targets is None else json.dumps(list(targets)),
        "profile_json": None
        if profile is None
        else json.dumps([[k, _profile_value(v)] for k, v in profile.items()]),
        "max_nodes": max_nodes,
        "name": name,
    }
    raw = json.loads(
        fault_tree_json(
            source["model_json"],
            source["top_json"],
            source["profile_json"],
            max_nodes,
            cut_set_limit,
            name,
            source["targets_json"],
            cut_sets,
            measure,
        )
    )
    events = raw["tree"]["basic_events"]
    return FaultTree(
        top=raw["tree"]["top"],
        basic_events=events,
        minimal_cut_sets=None
        if raw["minimal_cut_sets"] is None
        else [
            sorted(events[i]["name"] for i in cut) for cut in raw["minimal_cut_sets"]
        ],
        open_psa=raw["open_psa"],
        warnings=raw["tree"]["warnings"],
        measure=measure,
        _source=source,
        _structure=raw["structure"],
    )


@dataclass(frozen=True)
class BasicEventImportance:
    """The importance of one basic event in a quantified fault tree.

    With ``P`` the top probability, ``p`` the event's, ``P1`` and ``P0``
    the top probability with the event certain and impossible. A ratio
    whose denominator is zero is ``None``: in particular the risk
    reduction worth of an event whose removal removes the whole risk
    (``P0 = 0``), which is infinite.
    """

    #: The basic event's name.
    name: str
    #: Its probability at the mission time.
    probability: float | None
    #: Birnbaum's marginal importance, ``P1 - P0``.
    birnbaum: float
    #: The critical importance factor, ``p (P1 - P0) / P``.
    criticality: float | None
    #: Fussell-Vesely in the PSA-code convention, ``(P - P0) / P``.
    fussell_vesely: float | None
    #: The diagnostic importance factor, ``p P1 / P``.
    diagnostic: float | None
    #: Risk achievement worth, ``P1 / P``.
    risk_achievement_worth: float | None
    #: Risk reduction worth, ``P / P0``.
    risk_reduction_worth: float | None


@dataclass(frozen=True)
class FaultTreeQuantification:
    """A fault tree quantified by :func:`quantify`."""

    #: The top-event probability at the mission time.
    probability: float
    #: The method that produced it: ``"bdd"`` (binary decision diagrams),
    #: ``"cut_sets"`` (cut sets under cutoffs), or ``"bdd+cut_sets"`` when
    #: the modules used both; ``provenance["modules"]`` says which.
    method: str
    #: Whether the probability is exact (no cutoff, no bound).
    exact: bool
    #: A guaranteed upper bound on the top-event probability: equal to it
    #: when exact; ``None`` when none can be certified.
    upper_bound: float | None
    #: The instant the basic events' laws were read at.
    mission_time: float | None
    #: Whether the top is monotone in every basic event.
    coherent: bool
    #: Every basic event's importance, in the tree's event order.
    importance: list[BasicEventImportance]
    #: The number of minimal cut sets (exact unless cutoffs removed some);
    #: ``None`` for a non-coherent tree or when they were not requested.
    cut_set_count: int | None
    #: Whether the cut sets are all of them.
    cut_sets_complete: bool
    #: The minimal cut sets, each a sorted list of basic-event names, by
    #: size then name; ``None`` when omitted, see :attr:`cut_sets_omitted`.
    minimal_cut_sets: list[list[str]] | None
    #: Why the minimal cut sets are not listed.
    cut_sets_omitted: str | None
    #: What deserves attention before trusting the number (a module that
    #: fell back to its cut sets, a cutoff that bit, a neglected mass larger
    #: than the estimate), coded per module: ``{"kind": ..., "module": index,
    #: <parameters>, "message": <English sentence>}``; empty for an exact
    #: result.
    warnings: list[dict[str, Any]]
    #: How the numbers were produced: the variable-ordering heuristic, and
    #: per module its variables in diagram order and its diagram size.
    provenance: dict[str, Any]
    #: What the probabilities measure (one of :data:`FAULT_TREE_MEASURES`).
    measure: str = "probability_without_repair"


def _quantify_fault_tree(
    tree: FaultTree | str | Path,
    *,
    top: str | None = None,
    mission_time: float | None = None,
    max_bdd_nodes: int = 10_000_000,
    cut_set_limit: int = 100_000,
    cut_sets: bool = True,
    engine: str = "auto",
    max_order: int | None = None,
    min_cut_probability: float = 0.0,
    max_cut_sets: int = 1_000_000,
    max_expansions: int = 100_000_000,
    measure: str | None = None,
) -> FaultTreeQuantification:
    """Quantify a fault tree with binary decision diagrams, falling back
    to cut sets under cutoffs for a module too large for them.

    ``tree`` is a :class:`FaultTree` generated by :func:`fault_tree`, an
    OpenPSA document as text, or the path of one. ``top`` names the top
    gate when the document has several gates no other references.

    Each basic event's probability is in ``measure``: under
    ``"probability_without_repair"``, its law's cumulative distribution at
    ``mission_time``; under ``"unavailability"``, a repairable event's
    probability of being down then, ``λ/(λ+µ) (1 - e^{-(λ+µ)t})`` for an
    exponential failure with exponential repairs and the asymptotic
    ``MTTR / (MTTF + MTTR)`` otherwise. Without ``measure``, the one the
    document names (a :class:`FaultTree`'s), else the probability without
    repair. ``mission_time`` is required as soon as one law depends on
    time. The result carries the exact top probability, every
    event's importance, and the minimal cut sets when the tree is
    coherent and they number at most ``cut_set_limit`` (their exact
    number is always given). A tree carrying a negation that survives
    simplification is quantified exactly too, but its minimal cut sets
    are not computed and the result says so. ``cut_sets=False`` skips
    their extraction, usually the costliest step, which neither the
    probability nor the importance measures need.

    ``engine`` chooses how each independent module is quantified:
    ``"auto"`` (default) builds its exact diagram and, when that outgrows
    ``max_bdd_nodes``, falls back to its cut sets under cutoffs;
    ``"exact"`` raises :class:`SimulationError` instead; ``"cut_sets"``
    uses cut sets for every module. The cut-set engine keeps the cut sets
    of order at most ``max_order``, of probability at least
    ``min_cut_probability``, and at most ``max_cut_sets`` of them (the
    most probable), expanding at most ``max_expansions`` partial sets. A
    module it quantifies is valued by the pivotal upper bound of what it
    kept, and the result's ``upper_bound`` adds an upper bound on
    everything the cutoffs neglected: a guaranteed bound on the top, for a
    coherent tree. It refuses a negated event.
    """
    if isinstance(tree, FaultTree):
        text = tree.open_psa
    elif isinstance(tree, Path) or not str(tree).lstrip().startswith("<"):
        text = Path(tree).read_text(encoding="utf-8")
    else:
        text = tree
    raw = json.loads(
        fault_tree_quantify_json(
            text,
            top,
            mission_time,
            max_bdd_nodes,
            cut_set_limit,
            cut_sets,
            engine,
            max_order,
            min_cut_probability,
            max_cut_sets,
            max_expansions,
            measure,
        )
    )
    names = raw["events"]
    result = raw["result"]
    cuts = result["minimal_cut_sets"]
    return FaultTreeQuantification(
        probability=result["probability"],
        method=result["method"],
        exact=result["exact"],
        upper_bound=result["upper_bound"],
        mission_time=result["mission_time"],
        coherent=result["coherent"],
        importance=[
            BasicEventImportance(
                name=names[entry["event"]],
                probability=entry["probability"],
                birnbaum=entry["birnbaum"],
                criticality=entry["criticality"],
                fussell_vesely=entry["fussell_vesely"],
                diagnostic=entry["diagnostic"],
                risk_achievement_worth=entry["risk_achievement_worth"],
                risk_reduction_worth=entry["risk_reduction_worth"],
            )
            for entry in result["importance"]
        ],
        cut_set_count=result["cut_set_count"],
        cut_sets_complete=result["cut_sets_complete"],
        minimal_cut_sets=None
        if cuts is None
        else sorted(
            (sorted(names[i] for i in cut) for cut in cuts),
            key=lambda cut: (len(cut), cut),
        ),
        cut_sets_omitted=result["cut_sets_omitted"],
        warnings=result["warnings"],
        provenance=result["provenance"],
        measure=raw["measure"],
    )


# The quantification entry point lives in its own module; it is imported
# here, after the names it builds on, and re-exported as part of this package.
from ._quantification import (
    QUANTIFICATION_METHODS,
    BiasFamily,
    CrossEntropyResult,
    PilotIteration,
    Quantification,
    SplittingBatch,
    SplittingLevel,
    SplittingResult,
    Study,
    TargetProbability,
    quantify,
    read_quantification,
)


@dataclass(frozen=True)
class Cut:
    """One minimal cut set of the feared event."""

    #: The basic events, as ``component.automaton.state`` qualified names.
    events: list[str]
    #: How many trajectories reached the feared event with this cut
    #: realized.
    weight: float


class _PivotalWorths:
    """The two worths every importance unit derives from its pivotal
    levels: shared by the mode, component and group levels."""

    q_system_failed: list[float]
    q_system_intact: list[float]

    def risk_achievement(self, analysis: ImportanceAnalysis) -> list[float]:
        """Risk-achievement worth, ``q_system_failed / q_cuts``.

        A ratio, so it is derived rather than reported: its denominator
        legitimately reaches zero (a campaign where the feared event never
        occurred), and that case reads better as an explicit ``inf`` here
        than as a ``null`` on the wire.
        """
        return [
            f / q if q > 0 else math.inf
            for f, q in zip(self.q_system_failed, analysis.q_cuts)
        ]

    def risk_reduction(self, analysis: ImportanceAnalysis) -> list[float]:
        """Risk-reduction worth, ``q_cuts / q_system_intact``."""
        return [
            q / i if i > 0 else math.inf
            for q, i in zip(analysis.q_cuts, self.q_system_intact)
        ]


@dataclass(frozen=True)
class EventImportance(_PivotalWorths):
    """The importance measures of one basic event, one failure mode.

    Every list is indexed like :attr:`ImportanceAnalysis.instants`.
    """

    #: The ``component.automaton.state`` qualified name.
    event: str
    #: The component whose automaton carries it.
    component: str
    #: The automaton, by its name inside the component.
    automaton: str
    #: The failure state.
    state: str
    #: ``q_i(t)``: probability that the state is active.
    unavailability: list[float]
    #: Probability that the system is *critical* for this mode: it fails
    #: if the mode occurs and holds if it does not.
    birnbaum: list[float]
    #: The share of the feared-event probability that passes through a cut
    #: containing this mode.
    fussell_vesely: list[float]
    #: ``birnbaum · unavailability / q_cuts``.
    criticality: list[float]
    #: The feared-event probability with this mode certainly occurred.
    q_system_failed: list[float]
    #: The feared-event probability with this mode made impossible.
    q_system_intact: list[float]


@dataclass(frozen=True)
class ComponentImportance(_PivotalWorths):
    """The importance measures of one component, every failure mode its
    automata carry taken as one unit.

    Every list is indexed like :attr:`ImportanceAnalysis.instants`.
    """

    component: str
    #: The failure states the cut structure attributes to it, as
    #: ``automaton.state`` names.
    events: list[str]
    #: ``q_i(t)``: probability that at least one of those states is active.
    unavailability: list[float]
    #: Probability that the system is *critical* for this component: it
    #: fails if the component fails and holds if the component holds.
    #: Computed for the component, not summed over its modes, which it is
    #: not.
    birnbaum: list[float]
    #: The share of the feared-event probability that passes through a cut
    #: containing this component.
    fussell_vesely: list[float]
    #: ``birnbaum · unavailability / q_cuts``.
    criticality: list[float]
    #: The feared-event probability with this component certainly failed.
    q_system_failed: list[float]
    #: The feared-event probability with this component made perfect.
    q_system_intact: list[float]


@dataclass(frozen=True)
class GroupImportance(_PivotalWorths):
    """The importance measures of one declared group of basic events, as
    one unit (see the ``groups`` argument of :func:`importance`).

    Every list is indexed like :attr:`ImportanceAnalysis.instants`.
    """

    group: str
    #: Its basic events, as ``component.automaton.state`` names.
    events: list[str]
    #: ``q_g(t)``: probability that at least one of them is active.
    unavailability: list[float]
    #: Probability that the system is critical for the group:
    #: ``P(Φ(D ∪ E_g)) − P(Φ(D \ E_g))``.
    birnbaum: list[float]
    #: The share of the feared-event probability that passes through a cut
    #: meeting the group.
    fussell_vesely: list[float]
    #: ``birnbaum · unavailability / q_cuts``.
    criticality: list[float]
    #: The feared-event probability with the whole group failed.
    q_system_failed: list[float]
    #: The feared-event probability with the whole group made perfect.
    q_system_intact: list[float]


@dataclass(frozen=True)
class ImportanceAnalysis:
    """Native importance measures of one feared event."""

    #: The feared event, as a ``component.automaton.state`` qualified name.
    target: str
    #: The instants every series is sampled at, ascending.
    instants: list[float]
    #: The feared-event probability as the trajectories recorded it.
    q_target: list[float]
    #: The same probability as the reconstructed cut structure has it. A
    #: gap with :attr:`q_target` says the cut corpus is incomplete: a path
    #: no replica walked cannot be in it.
    q_cuts: list[float]
    #: The minimal cut sets, heaviest first.
    cuts: list[Cut]
    #: Per-component measures, keyed by component name. Insertion order
    #: is the ranking, most important first (descending Fussell-Vesely at
    #: the last instant), so ``list(components)`` reads as the answer to
    #: "where do I invest".
    components: dict[str, ComponentImportance]
    #: Trajectory weight the analysis rests on (one per replica).
    nb_runs: float
    #: Engine version that produced it.
    engine_version: str
    #: Per-failure-mode measures, keyed by ``component.automaton.state``,
    #: for the basic events of the cuts, ranked like :attr:`components`.
    basic_events: dict[str, EventImportance] = field(default_factory=dict)
    #: Per-group measures, keyed by group name, one per declared group,
    #: ranked like :attr:`components`.
    groups: dict[str, GroupImportance] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ImportanceAnalysis(target={self.target!r}, "
            f"{len(self.cuts)} cuts, {len(self.basic_events)} basic events, "
            f"{len(self.components)} components, {len(self.groups)} groups)"
        )


def _basic_event_name(event: dict) -> str:
    """The ``component.automaton.state`` name of a wire basic event."""
    return f"{event['obj']}.{event['automaton']}.{event['attr']}"


#: A basic event as a caller may give it: a ``component.automaton.state``
#: name, or its three parts, as a tuple or as a mapping with the wire keys.
BasicEvent = Union[str, tuple[str, str, str], Mapping[str, str]]


def _basic_event_wire(event: BasicEvent, where: str) -> dict:
    """The wire form of a basic event.

    A name is split on its last two dots, which is right only when the
    automaton and the state carry none. A component or mode name may carry
    dots of its own (an external mode's automaton is named after its
    carrier), so a caller holding the three parts passes them as a
    triplet and nothing is split.
    """
    if isinstance(event, str):
        parts = event.rsplit(".", 2)
    elif isinstance(event, Mapping):
        parts = [event.get("obj"), event.get("automaton"), event.get("attr")]
    elif isinstance(event, (tuple, list)):
        parts = list(event)
    else:
        parts = []
    if len(parts) != 3 or not all(isinstance(part, str) and part for part in parts):
        raise ValueError(
            f"{where}: `{event!r}` is not a `component.automaton.state` basic event"
            " (a name, or a (component, automaton, state) triplet)"
        )
    return {"obj": parts[0], "automaton": parts[1], "attr": parts[2]}


_MEASURES = (
    "unavailability",
    "birnbaum",
    "fussell_vesely",
    "criticality",
    "q_system_failed",
    "q_system_intact",
)


def importance(
    model: Model,
    nb_runs: int,
    t_max: float,
    instants: list[float],
    target: str | None = None,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
    basic_events: str | list[BasicEvent] = "failures",
    groups: dict[str, list[BasicEvent]] | None = None,
) -> ImportanceAnalysis:
    """Native importance measures: which failure mode, which component
    and which group carries the risk.

    Runs one sequence-recording Monte-Carlo campaign and reduces it to the
    **Birnbaum**, **Fussell-Vesely** and **criticality** series at
    ``instants`` of every failure mode, every component and every declared
    group, together with the minimal cut sets they were computed on.

    The support is RAICHU's native sequence recording. Replayed up to the
    first feared event, each trajectory gives the failures active when it
    occurred, a cut; merged and reduced, the cuts are the minimal cut
    sets, hence the structure function. Left to run to the horizon, the
    same trajectories replay the state of every failure mode at any
    instant. The measures then follow from their definitions, with no
    independence assumed and no rare-event approximation: Birnbaum is the
    pivotal difference ``P(Φ(D ∪ E_i)) − P(Φ(D \\ E_i))``,
    Fussell-Vesely the share of system failures whose realized cut meets
    the unit.

    A **basic event** is one failure state of one automaton, named
    ``component.automaton.state``: two modes of one pump are two automata
    of the pump, hence two basic events, and the pump itself is the unit
    that gathers them. ``basic_events`` says which recorded states count:

    - ``"failures"`` (default): the states the model's failure transitions
      enter (``kind: "failure"``), so an observer, an intermediate feared
      event or the park of a lost draw is never a failure;
    - ``"monitored"``: every recorded state other than its automaton's
      initial one, for a model that declares no failure role;
    - a list of basic events, each a ``component.automaton.state`` name or,
      when a component or mode name carries a dot, its
      ``(component, automaton, state)`` triplet (a tuple, or a mapping with
      the keys ``obj``, ``automaton`` and ``attr``).

    ``groups`` maps a group name to its basic events, given the same way, each group being
    measured as one unit (a physical component whose modes live on other
    objects, a subsystem). An event may sit in several groups, which is
    how a common cause is attributed to each of its targets.

    ``target`` names the feared event among the model's declared targets;
    a model with exactly one may leave it out. ``flow`` is a
    :class:`FlowConfig` overriding the convergence policy of the
    continuous flow resolution for every replica.

    A campaign in which the feared event never occurred returns an empty
    analysis with ``q_target`` and ``q_cuts`` at zero, saying why.
    """
    if isinstance(basic_events, str):
        selection: str | list[dict] = basic_events
    else:
        selection = [_basic_event_wire(name, "basic_events") for name in basic_events]
    options = {
        "basic_events": selection,
        "groups": [
            {
                "name": name,
                "events": [
                    _basic_event_wire(event, f"group `{name}`") for event in events
                ],
            }
            for name, events in (groups or {}).items()
        ],
    }
    raw = json.loads(
        importance_json(
            model.json,
            nb_runs,
            t_max,
            instants,
            target,
            seed,
            threads,
            flow,
            json.dumps(options),
        )
    )

    def measures(entry: dict) -> dict:
        return {name: entry[name] for name in _MEASURES}

    analysis = ImportanceAnalysis(
        target=_basic_event_name(raw["target"]),
        instants=raw["instants"],
        q_target=raw["q_target"],
        q_cuts=raw["q_cuts"],
        cuts=[
            Cut(
                events=[_basic_event_name(e) for e in cut["events"]],
                weight=cut["weight"],
            )
            for cut in raw["cuts"]
        ],
        components={
            c["component"]: ComponentImportance(
                component=c["component"], events=c["events"], **measures(c)
            )
            for c in raw["components"]
        },
        nb_runs=raw["nb_runs"],
        engine_version=raw["engine_version"],
        basic_events={
            _basic_event_name(e["event"]): EventImportance(
                event=_basic_event_name(e["event"]),
                component=e["event"]["obj"],
                automaton=e["event"]["automaton"],
                state=e["event"]["attr"],
                **measures(e),
            )
            for e in raw["basic_events"]
        },
        groups={
            g["group"]: GroupImportance(
                group=g["group"],
                events=[_basic_event_name(e) for e in g["events"]],
                **measures(g),
            )
            for g in raw["groups"]
        },
    )
    return analysis


def switching_loops(model: Model) -> list[dict]:
    """Switching loops of ``model``, found without simulating it.

    A switching loop is a cycle of the dependency graph, automaton to
    variable to automaton: an automaton whose guard reads a quantity that
    its own decision moves. Each of its two states then produces the
    condition that justifies the other, so the mode has no fixpoint and
    what sets its period is the width of the narrowest threshold on the
    cycle rather than anything physical.

    A **warning and never a refusal**: the loop itself is legitimate, a
    thermostat is one, and so is every controlled tank. What makes one
    pathological is a switch with no band, so a loop is reported only
    when some automaton on it is entered and left at the same threshold,
    and a loop whose every switch has a band is silent.

    Each entry carries ``automata`` (the cycle), ``bandless`` (those of
    them that switch on a single threshold), ``through`` (the variables
    the cycle passes through) and a ready-phrased ``message``. The cure
    is a band on the threshold, or reading a quantity the rule does not
    move; the run-time counterpart is ``max_transition_firings``, which
    catches the same thing after the wait rather than before it.
    """
    return json.loads(switching_loops_json(model.json))


def cycle_groups(model: Model) -> list[dict]:
    """Whether the minimal-sequence reduction may cancel each cycle group
    of ``model``, decided without simulating it.

    The reduction drops a failure repaired before the feared event, on the
    hypothesis that a repaired component no longer contributes. That fails
    when the failure's effect is remembered: a storage drained during the
    outage stays low after the repair. A group is **kept** (its whole
    history stays in the sequences) for a target when its effect reaches
    that target through a quantity that remembers it: an ODE, an FMU
    output, a latch its own repair does not reset, a timer that survives an
    interruption, or a stateful automaton such as a band controller.

    Each entry carries ``component``, ``cycle_group``, ``automata``, and one
    decision per target, ``{target, cancellable, origin, witness,
    memory}``: ``origin`` is ``analysis`` or ``declared`` (a transition's
    ``cycle_cancellation`` overrides the analysis), ``witness`` the path
    from the group to the target and ``memory`` the quantity that remembers.
    ``messages`` phrases each kept target in one line. A sequence campaign
    reports the decisions it applied in ``SequenceCampaign.reduction``.
    """
    return json.loads(cycle_groups_json(model.json))


def unfed_triggers(model: Model) -> list[dict]:
    """Trigger inputs of ``model`` that nothing feeds, found without
    simulating it.

    An in port aggregates what reaches it, and an aggregation over
    nothing still answers: ``any`` and ``sum`` answer false and zero,
    ``all`` answers the vacuous truth. A guard reading a port no
    connection reaches is therefore **pinned** from the initial instant
    to the end of the run, and the automaton it drives settles into one
    state with no transition out of it. A cold standby delivers while its
    trigger input is *absent*, so a forgotten wire arms it at ``t = 0``
    and for ever: the run terminates normally, the journal says nothing,
    and the campaign returns an availability that is too good.

    A **warning and never a refusal**: such a model is valid, the
    semantics is settled and the engines agree on it. It is only, almost
    always, an oversight. What is reported is narrow for that reason: a
    guard is pinned only when it folds to a constant exactly, so a guard
    that also reads a clock or a moving attribute is silent, and the way
    to declare an always-on output on purpose is to declare no trigger at
    all, which is silent too.

    A guard rarely reads the port itself: an authoring layer puts a
    derived attribute in between, and a muscadet rule set thresholds
    ``P.E_capability_in`` rather than ``P.E_in``. The fold therefore
    **crosses a definition** to reach the port, on one condition: the
    attribute must have exactly one writer and that writer must be an
    explicit equation. An attribute an ODE integrates, a sensitive
    function assigns or two writers share moves, and the fold stops at
    it; an attribute *nothing* writes is the opposite case and is read as
    the constant it is.

    And the seal has to **flatter the result**. Crossing definitions
    reaches modes sealed the pessimistic way round -- a rule that
    consumes the flow nothing feeds cannot draw, so it produces zero,
    which the results already show -- and those stay silent. A mode is
    reported when some quantity reading it is nothing while the automaton
    is elsewhere and not nothing once it is pinned there.

    How far it reaches is decided by whoever writes the guards and the
    definitions behind them. On :mod:`pyraichu.muscadet`, two
    constructors put a port within reach: ``add_flow_out_on_trigger``,
    inside the guard, and ``add_rule_set`` on a **continuous** condition,
    through the equation of the input it thresholds. The same rule set on
    a boolean condition stays out of reach, ``{flow}_fed_in`` being
    assigned by a sensitive function.

    Out of reach, and with nothing there to report. A boolean in-flow
    answers for its own emptiness: muscadet gives it a declared
    out-of-connection value (``var_in_default``, false unless stated) and
    :mod:`pyraichu.muscadet` writes it into the aggregating expression, so
    an in-flow nothing feeds reads **unfed** rather than the vacuous truth
    of ``all``. A mode guarded on it is sealed shut for ever instead of
    armed for ever, the component starves, and a component producing
    nothing is what the results already show. That is the pessimistic
    direction this diagnostic deliberately keeps quiet on, so the frontier
    is one of reach and not a gap left to close.

    Each entry carries ``ports`` (the in ports to wire), ``automaton``
    and ``state`` (the mode their emptiness seals) so the finding can be
    checked rather than trusted, and a ready-phrased ``message``. The
    cure is to connect that input, or to drop the guard and state the
    mode unconditionally. The guard rather than the trigger: what the
    emptiness seals may be a rule condition, and a rule has no trigger
    port to take away. Its sibling is :func:`switching_loops`, which
    runs on the same schedule and answers a different question: a loop is
    a mode with no fixpoint, this is a mode with no choice.
    """
    return json.loads(unfed_triggers_json(model.json))


def simulate(
    model: Model,
    t_max: float = math.inf,
    journal: bool = False,
    confluence_check: bool = False,
    samples: list[float] | None = None,
    seed: int = 0,
    rng_stream: int = 0,
    flow: FlowConfig | None = None,
    max_transition_firings: int | None = None,
    max_flow_restarts: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_step: float | None = None,
    tol_event: float | None = None,
    sub_samples: int | None = None,
    event_resolution: float | None = None,
    stop_at_targets: bool = False,
) -> SimulationResult:
    """Run one simulation of ``model`` up to ``t_max``.

    ``samples`` is an ascending list of instants at which every
    indicator is recorded (dense output for continuous variables).
    ``seed``/``rng_stream`` drive the stochastic laws (ignored by
    deterministic models); the same pair replays bit-identically.
    ``flow`` is a :class:`FlowConfig` overriding the convergence policy
    of the continuous flow resolution (engine defaults when omitted).
    ``max_transition_firings`` caps how many times ONE transition may
    fire in this trajectory, and ``max_flow_restarts`` how many times the
    active set of the continuous flow network may change, beyond which
    the model is declared to be chattering and the run fails with a
    diagnosis naming the transition, or the edges that crossed last.
    Together they are what turns a limit cycle, where time advances by a
    little every turn and neither Zeno guard can see it, into a typed
    error rather than a run that never ends: the first catches a mode
    that flips, the second a network whose routing does. ``0`` disables
    either; omitted, the engine default applies.

    ``rtol``/``atol``/``max_step``/``tol_event``/``sub_samples``/
    ``event_resolution`` override the ODE-backend parameters, and
    ``stop_at_targets`` ends the run at the first feared event, exactly
    as :func:`monte_carlo` takes them: one replica of a campaign replays
    here with the campaign's own settings and the coordinates its
    failure carries (``seed=error.seed, rng_stream=error.replica``).

    The GIL is released while the Rust engine runs. Raises
    :class:`SimulationError` on typed engine failures (instantaneous
    loop, chattering, non-confluence when ``confluence_check`` is
    enabled, …).
    """
    raw = json.loads(
        simulate_json(
            model.json,
            t_max,
            journal,
            confluence_check,
            samples,
            seed,
            rng_stream,
            flow,
            max_transition_firings,
            max_flow_restarts,
            model.allow_fmu_import,
            str(model.base_dir),
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            tol_event=tol_event,
            sub_samples=sub_samples,
            event_resolution=event_resolution,
            stop_at_targets=stop_at_targets,
        )
    )
    events = [
        Event(
            time=e["time"],
            transition=e["transition"],
            from_state=e["from"],
            to_state=e["to"],
        )
        for e in raw["events"]
    ]
    return SimulationResult(
        events=events,
        indicators=_series_dict(raw["indicators"]),
        samples=_series_dict(raw["samples"]),
        journal=raw["journal"],
        provenance=raw["provenance"],
        final_time=raw["final_time"],
        work=raw.get("work", {}),
    )


@dataclass(frozen=True)
class Fireable:
    """An armed transition offered to interactive control."""

    index: int
    transition: str
    kind: str  # "delay" | "stochastic" | "inst" | "watched"
    # Firing date. None when the transition has none yet: a watched
    # boundary or a located condition not crossed yet (``fire`` refuses
    # it), or a deferred stochastic transition (dated by the deferred or
    # operator interfaces).
    date: float | None

    def __repr__(self) -> str:
        if self.date is not None:
            when = f"t={self.date}"
        elif self.kind == "watched":
            when = "boundary"
        elif self.kind == "stochastic":
            when = "deferred"
        else:
            when = "condition"
        return f"Fireable({self.transition} [{self.kind}] @ {when})"


@dataclass(frozen=True)
class OperatorAdvance:
    """Native outcome of a bounded operator-controlled advance.

    ``stop`` is ``target``, ``event``, ``choice`` or ``incomplete``.
    ``choice`` names the pending transition without selecting its branch.
    """

    requested_time: float
    reached_time: float
    stop: str
    events: list[Event]
    choice: str | None


class Interactive:
    """Step-by-step interactive simulation over a RAICHU model.

    With ``operator_control=True``, stochastic dates are not drawn and
    ``advance_operator_to`` stops before unresolved probabilistic choices.
    With ``deferred_draws=True``, stochastic transitions arm without a date
    and never fire by themselves, and an instantaneous transition with
    several random branches waits likewise instead of drawing one, keeping
    its place among its instant's events: :meth:`step` and
    :meth:`advance_to` stop before it, where a free run would draw it;
    :meth:`deferred` reports each with its law and age, and
    :meth:`fire_deferred_at` fires one at a chosen instant (with a chosen
    branch), which is how a trajectory drawn elsewhere is replayed.

    Drive the engine one event at a time under your own control, rather
    than running it to the horizon in one shot:

    - :meth:`fireable`: the currently-armed transitions (earliest first);
    - :meth:`fire`: fire a *chosen* one, optionally **forcing** its
      outcome branch with ``to=`` (bypassing the random draw, which is
      what makes stochastic mechanics reproducibly testable);
    - :meth:`step`: advance to the next scheduled event, as a plain run
      would;
    - :meth:`set_date`: reschedule an armed transition;
    - :meth:`snapshot` / :meth:`restore`: checkpoint and undo;
    - :meth:`reset`: back to ``t = 0``;
    - :attr:`time`, :meth:`attribute`, :meth:`state`, :meth:`history`,
      :meth:`journal`: inspection between events.

    Models carrying a ``"plugins"`` section are expanded and validated
    (as :func:`load_model` does). ``seed``/``rng_stream`` drive the
    stochastic laws; the same pair replays bit-identically. ``flow`` is a
    :class:`FlowConfig` overriding the convergence policy of the
    continuous flow resolution (engine defaults when omitted).
    """

    def __init__(
        self,
        model: Model | str | dict[str, Any],
        t_max: float = math.inf,
        journal: bool = False,
        confluence_check: bool = False,
        seed: int = 0,
        rng_stream: int = 0,
        flow: FlowConfig | None = None,
        operator_control: bool = False,
        deferred_draws: bool = False,
    ) -> None:
        if not isinstance(model, Model):
            model = load_model(model)
        self._model = model
        self._raw = _RawInteractive(
            model.json,
            t_max,
            journal,
            confluence_check,
            seed,
            rng_stream,
            flow,
            model.allow_fmu_import,
            str(model.base_dir),
            operator_control,
            deferred_draws,
        )

    @property
    def model(self) -> Model:
        """The validated model driving this session."""
        return self._model

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._raw.time

    def fireable(self) -> list[Fireable]:
        """The currently-armed transitions, earliest date first."""
        return [Fireable(**f) for f in json.loads(self._raw.fireable())]

    def fire(self, name: str, to: str | None = None) -> Event:
        """Fire the armed transition ``name`` (by qualified name).

        ``to`` **forces** the destination branch to that state name,
        bypassing the RNG / deterministic-branch resolution: the
        reproducible outcome control. Raises :class:`SimulationError` if
        the transition is not armed or ``to`` is not one of its branches.
        """
        return self._event(self._raw.fire(name, to))

    def step(self) -> Event | None:
        """Advance to the next scheduled event; ``None`` at the horizon."""
        raw = self._raw.step()
        return None if raw is None else self._event(raw)

    def advance_to(self, date: float) -> None:
        """Fire every event through ``date`` and leave the clock there."""
        self._raw.advance_to(date)

    def advance_operator_to(
        self, date: float, *, max_events: int = 10_000
    ) -> OperatorAdvance:
        """Stop at the first event or unresolved choice without drawing dates.

        Requires ``operator_control=True``. Numerical errors restore the
        preceding state; an ``incomplete`` outcome commits bounded progress.
        """
        raw = json.loads(self._raw.advance_operator_to(date, max_events))
        return OperatorAdvance(
            requested_time=raw["requested_time"],
            reached_time=raw["reached_time"],
            stop=raw["stop"],
            events=[self._event_from_dict(e) for e in raw["events"]],
            choice=raw["choice"],
        )

    def deferred(self) -> list[dict[str, Any]]:
        """The stochastic transitions armed without a date, ascending by index.

        Each is ``{index, transition, law, armed_at, age, paused, hazard}``:
        ``law`` is the compiled occurrence law in the model schema's spelling
        (``{"distrib": "exp", "rate": 0.5}``), ``age`` the time spent armed
        and running, ``hazard`` the cumulative hazard of a state-dependent
        rate (``None`` otherwise). Empty unless the session was opened with
        ``deferred_draws=True``.
        """
        return json.loads(self._raw.deferred())

    def fire_deferred_at(self, name: str, date: float, branch: int | None = None) -> Event:
        """Fire the deferred transition ``name`` at ``date``.

        The deterministic events due before ``date`` fire on the way;
        ``branch`` forces the destination of a multi-target law (by
        position in the transition's targets), which a held branch choice
        needs. Raises
        :class:`SimulationError` when ``name`` is not armed.
        """
        return self._event(self._raw.fire_deferred_at(name, date, branch))

    def set_input(
        self,
        qualified: str,
        value: bool | int | float,
        allowed_inputs: list[str],
    ) -> None:
        """Set an attribute explicitly named by the export input manifest."""
        self._raw.set_input(
            qualified, json.dumps(_profile_value(value)), allowed_inputs
        )

    def set_date(self, name: str, date: float) -> None:
        """Override an armed transition's firing date (``>=`` current time)."""
        self._raw.set_date(name, date)

    def reset(self) -> None:
        """Reset the session to its initial state (``t = 0``, fresh RNG)."""
        self._raw.reset()

    def attribute(self, qualified: str) -> bool | int | float | None:
        """Value of ``component.attribute``; ``None`` if unknown."""
        raw = self._raw.attribute(qualified)
        return None if raw is None else _value_to_python(json.loads(raw))

    def state(self, qualified: str) -> str | None:
        """Current state name of ``component.automaton``; ``None`` if unknown."""
        return self._raw.state(qualified)

    def history(self) -> list[Event]:
        """The events fired so far, chronological."""
        return [self._event_from_dict(e) for e in json.loads(self._raw.history())]

    def journal(self) -> list[dict[str, Any]]:
        """The causal journal recorded so far, as :func:`simulate` returns it.

        Requires a session opened with ``journal=True``; raises
        :class:`SimulationError` otherwise. After :meth:`restore`, it is
        the journal of the restored state, so branches fired from one
        snapshot each read their own history.
        """
        return json.loads(self._raw.journal())

    def snapshot(self) -> Any:
        """Capture the full state as an opaque checkpoint (for :meth:`restore`)."""
        return self._raw.snapshot()

    def restore(self, snapshot: Any) -> None:
        """Reinstate a checkpoint captured by :meth:`snapshot` (undo)."""
        self._raw.restore(snapshot)

    @staticmethod
    def _event_from_dict(e: dict[str, Any]) -> Event:
        return Event(
            time=e["time"],
            transition=e["transition"],
            from_state=e["from"],
            to_state=e["to"],
        )

    def _event(self, raw_json: str) -> Event:
        return self._event_from_dict(json.loads(raw_json))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Interactive({self._model.name!r}, t={self.time})"


def interactive(
    model: Model | str | dict[str, Any],
    t_max: float = math.inf,
    journal: bool = False,
    confluence_check: bool = False,
    seed: int = 0,
    rng_stream: int = 0,
    flow: FlowConfig | None = None,
    operator_control: bool = False,
    deferred_draws: bool = False,
) -> Interactive:
    """Open an :class:`Interactive` session over ``model`` (a
    :class:`Model`, a JSON string, or a dict; plugins are expanded and
    validated as :func:`load_model` does)."""
    return Interactive(
        model,
        t_max=t_max,
        journal=journal,
        confluence_check=confluence_check,
        seed=seed,
        rng_stream=rng_stream,
        flow=flow,
        operator_control=operator_control,
        deferred_draws=deferred_draws,
    )


#: Submodules that moved into :mod:`pyraichu.muscadet` and keep a relay at
#: their old path for one release. Reading one as an attribute of this
#: package imports the relay, which warns and returns the moved module.
_MOVED_SUBMODULES = frozenset({"declare", "muscadet_engine"})


def __getattr__(name: str) -> Any:
    if name in _MOVED_SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
