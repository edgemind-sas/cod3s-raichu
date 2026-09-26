"""Python binding for the RAICHU hybrid (PDMP) simulation engine.

The heavy lifting happens in the Rust extension module
``pyraichu._pyraichu``; this package is the thin, typed, Pythonic wrapper
around it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._pyraichu import (
    MODEL_ENVELOPE_KEY,
    MODEL_FORMAT_REVISION,
    FlowConfig,
    Interactive as _RawInteractive,
    ModelError,
    SimulationError,
    __version__,
    analyse_raw_sequences_json,
    analyse_sequences_json,
    exploration_domain_json,
    exploration_minimal_sequences_json,
    explore_json,
    run_sequences_json,
    monte_carlo_json,
    required_features,
    seal_model,
    simulate_json,
    switching_loops_json,
    validate_exploration,
    validate_model,
)
from .journal import Cascade, JournalQuery, TransitionHistory, AttributeChange

__all__ = [
    "Cascade",
    "Event",
    "Exploration",
    "ExploredSequence",
    "Extremes",
    "Fireable",
    "FlowConfig",
    "IndicatorEstimate",
    "Interactive",
    "JournalQuery",
    "MODEL_ENVELOPE_KEY",
    "MODEL_FORMAT_REVISION",
    "TransitionHistory",
    "AttributeChange",
    "McEstimates",
    "Model",
    "ModelError",
    "SimulationError",
    "SimulationResult",
    "__version__",
    "analyse_raw_sequences",
    "analyse_sequences",
    "expand_model",
    "exploration_domain",
    "explore",
    "run_sequences",
    "SequenceCampaign",
    "interactive",
    "load_model",
    "model_body",
    "monte_carlo",
    "read_exploration",
    "required_features",
    "seal",
    "seal_model",
    "simulate",
    "switching_loops",
]


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


def load_model(source: str | dict[str, Any]) -> Model:
    """Load and validate a model from a JSON string or a dict, in either
    document shape (bare body, or body under the format envelope).

    Models carrying a ``"plugins"`` section (specialized object schemas:
    ObjFlow, ObjFM, ObjEvent, …) are expanded into the core schema
    first; use :func:`expand_model` to inspect the translation.
    Raises :class:`ModelError` with a precise, typed message when the
    model is invalid (never a crash).
    """
    document = json.loads(source) if isinstance(source, str) else source
    # What the document declares survives the plugin expansion that
    # replaces its body: a declaration this engine cannot honour has to
    # reach the engine, which is where it is refused by name.
    declared = document.get(MODEL_ENVELOPE_KEY, {}).get("requires", [])
    body = model_body(document)
    if "plugins" in body:
        body = expand_model(document)
    # Sealing is a no-op for a body using baseline constructs only and
    # declaring nothing, so the whole existing corpus keeps its exact
    # document shape.
    model_json = json.dumps(seal(body, declared))
    validate_model(model_json)
    name = body.get("name", "")
    return Model(json=model_json, name=name)


def _series_dict(raw_series: list[dict[str, Any]]) -> dict[str, list[tuple[float, Any]]]:
    return {
        series["name"]: [(t, _value_to_python(v)) for t, v in series["points"]]
        for series in raw_series
    }


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
    sojourn_mean: list[float]
    sojourn_std: list[float]
    nb_occurrences_mean: list[float]
    nb_occurrences_std: list[float]
    #: Probability of having been active at least once by each instant (the
    #: RAMS "had value" measure): the first-entry distribution, which stays
    #: at 1 on a trajectory after the indicator falls back.
    reached_mean: list[float]
    reached_std: list[float]
    extremes: Extremes
    sojourn_extremes: Extremes
    nb_occurrences_extremes: Extremes
    reached_extremes: Extremes
    quantiles: dict[float, list[float]]
    sojourn_quantiles: dict[float, list[float]]


@dataclass(frozen=True)
class McEstimates:
    """Monte-Carlo result: per-indicator estimates + provenance."""

    indicators: dict[str, IndicatorEstimate]
    nb_runs: int
    seed: int
    engine_version: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"McEstimates({self.nb_runs} runs, seed={self.seed}, "
            f"{len(self.indicators)} indicators)"
        )


def monte_carlo(
    model: Model,
    nb_runs: int,
    t_max: float,
    samples: list[float],
    seed: int = 0,
    threads: int | None = None,
    quantiles: list[float] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_step: float | None = None,
    tol_event: float | None = None,
    sub_samples: int | None = None,
    stop_at_targets: bool = False,
    flow: FlowConfig | None = None,
) -> McEstimates:
    """Estimate indicator statistics over ``nb_runs`` replicas.

    Replica ``r`` uses RNG substream ``r`` of ``seed``; the reduction is
    index-ordered, so the result is byte-identical for any ``threads``
    value. ``quantiles`` (e.g. ``[0.25, 0.75]``) adds nearest-rank
    quantile series on both the sampled value and the cumulated
    sojourn. The GIL is released while the replicas run.

    ``rtol``/``atol``/``max_step``/``tol_event``/``sub_samples``
    override the ODE-backend parameters (engine defaults when omitted):
    the integration-effort knobs of the tolerance-parity experiments.

    ``stop_at_targets=True`` early-stops each trajectory at the first
    sequence target (feared event) and holds the frozen state through the
    remaining sample instants: the latch semantics of target-stopped
    studies (first-occurrence measures instead of free-cycling ones).

    ``flow`` is a :class:`FlowConfig` overriding the convergence policy
    of the continuous flow resolution for every replica (engine defaults
    when omitted).
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
            rtol,
            atol,
            max_step,
            tol_event,
            sub_samples,
            stop_at_targets,
            flow,
        )
    )
    indicators = {
        e["name"]: IndicatorEstimate(
            name=e["name"],
            instants=e["instants"],
            mean=e["mean"],
            std=e["std"],
            sojourn_mean=e["sojourn_mean"],
            sojourn_std=e["sojourn_std"],
            nb_occurrences_mean=e["nb_occurrences_mean"],
            nb_occurrences_std=e["nb_occurrences_std"],
            reached_mean=e["reached_mean"],
            reached_std=e["reached_std"],
            extremes=Extremes(**e["extremes"]),
            sojourn_extremes=Extremes(**e["sojourn_extremes"]),
            nb_occurrences_extremes=Extremes(**e["nb_occurrences_extremes"]),
            reached_extremes=Extremes(**e["reached_extremes"]),
            quantiles={s["q"]: s["values"] for s in e["quantiles"]},
            sojourn_quantiles={s["q"]: s["values"] for s in e["sojourn_quantiles"]},
        )
        for e in raw["indicators"]
    }
    return McEstimates(
        indicators=indicators,
        nb_runs=raw["nb_runs"],
        seed=raw["seed"],
        engine_version=raw["engine_version"],
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
    ``{events: [{obj, attr, time}], end_cause, end_time, weight}`` (``weight``
    = the number of trajectories that collapsed into it). The model must
    declare at least one target (an ``ObjEvent`` with ``"target": true``).

    ``flow`` is a :class:`FlowConfig` overriding the convergence policy
    of the continuous flow resolution for every replica (engine defaults
    when omitted).
    """
    return json.loads(
        analyse_sequences_json(model.json, nb_runs, t_max, seed, threads, flow)
    )


@dataclass(frozen=True)
class SequenceCampaign:
    """A sequence campaign kept whole: its two reduced levels, and where its
    raw corpus was written.

    ``cleaned`` is every distinct path to each feared event, transient
    failure/repair cycles removed (cod3s's ``sequences_all.json`` level);
    ``minimal`` is what :func:`analyse_sequences` returns. Both are lists of
    ``{events: [{obj, attr, time}], end_cause, end_time, weight}``.

    ``raw_path`` is the ``raichu.sequences`` corpus, one line per trajectory
    (see the sequence-format reference), or ``None`` when none was asked for.
    ``header`` is that corpus's first line when it was read back by
    :func:`analyse_raw_sequences`, ``None`` otherwise.
    """

    cleaned: list[dict[str, Any]]
    minimal: list[dict[str, Any]]
    raw_path: Path | None = None
    header: dict[str, Any] | None = None


def run_sequences(
    model: Model,
    nb_runs: int,
    t_max: float,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
    raw_path: str | Path | None = None,
) -> SequenceCampaign:
    """A sequence campaign whose raw corpus is kept.

    Runs the campaign :func:`analyse_sequences` runs, with the same seed
    giving the same trajectories, and returns both reduced levels. When
    ``raw_path`` is given, every trajectory's raw sequence is written there in
    the ``raichu.sequences`` format (JSON Lines: a header line, then one line
    per replica in replica order), straight from the engine: a campaign of any
    size never becomes Python objects. :func:`analyse_raw_sequences` reads it
    back and recomputes the reduction.
    """
    path = None if raw_path is None else Path(raw_path)
    levels = json.loads(
        run_sequences_json(
            model.json,
            nb_runs,
            t_max,
            seed,
            threads,
            flow,
            None if path is None else str(path),
        )
    )
    return SequenceCampaign(cleaned=levels["cleaned"], minimal=levels["minimal"], raw_path=path)


def analyse_raw_sequences(raw_path: str | Path) -> SequenceCampaign:
    """Read a ``raichu.sequences`` corpus and reduce it again.

    The reduction is the engine's own, so a corpus written by
    :func:`run_sequences` gives back exactly the levels that campaign
    returned. Refuses another format and a version newer than this engine
    reads.
    """
    path = Path(raw_path)
    levels = json.loads(analyse_raw_sequences_json(str(path)))
    return SequenceCampaign(
        cleaned=levels["cleaned"], minimal=levels["minimal"], raw_path=path, header=levels["header"]
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

    @classmethod
    def _from_json(cls, text: str) -> Exploration:
        raw = json.loads(text)
        fields = {name: raw[name] for name in _EXPLORATION_FIELDS}
        table = tuple(raw["steps"])
        fields["steps"] = table
        fields["sequences"] = [ExploredSequence._from_dict(s, table) for s in raw["sequences"]]
        fields["discretisation"] = raw.get("discretisation")
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
        """
        return json.loads(exploration_minimal_sequences_json(self.to_json()))


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

    ``level`` and ``refine=False`` apply to the discretised algorithm only,
    ``rel_precision`` and ``max_terms`` to the exact one only; passing one
    to the other algorithm raises. Raises :class:`SimulationError` for an
    unknown algorithm or an invalid setting (before anything runs), for a
    model outside the algorithm's domain (see :func:`exploration_domain`
    for the exact one), and, in the exact algorithm, when a law outside
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
    either; omitted, the engine default applies. The GIL is released
    while the Rust engine runs. Raises :class:`SimulationError` on typed
    engine failures (instantaneous loop, chattering, non-confluence when
    ``confluence_check`` is enabled, …).
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
    date: float | None  # firing date; None for an unlocated watched boundary

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        when = "boundary" if self.date is None else f"t={self.date}"
        return f"Fireable({self.transition} [{self.kind}] @ {when})"


class Interactive:
    """Step-by-step interactive simulation over a RAICHU model.

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
    - :attr:`time`, :meth:`attribute`, :meth:`state`, :meth:`history`:
      inspection between events.

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
    ) -> None:
        if not isinstance(model, Model):
            model = load_model(model)
        self._model = model
        self._raw = _RawInteractive(
            model.json, t_max, journal, confluence_check, seed, rng_stream, flow
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
    )
