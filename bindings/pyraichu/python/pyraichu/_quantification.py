"""The quantification contract's Python entry point: :class:`Study`,
:class:`Quantification` and :func:`quantify`.

Everything here is re-exported by :mod:`pyraichu`, which is where it is
documented and imported from (``pyraichu.quantify``, ``pyraichu.Study``, ...);
this module only keeps the package's ``__init__`` to a readable size.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import (
    Exploration,
    FaultTree,
    FaultTreeQuantification,
    McEstimates,
    Model,
    _mc_estimates,
    _quantify_fault_tree,
)
from ._pyraichu import SimulationError, quantify_json, validate_quantification

__all__ = [
    "QUANTIFICATION_METHODS",
    "BiasFamily",
    "CrossEntropyResult",
    "PilotIteration",
    "Quantification",
    "SplittingBatch",
    "SplittingLevel",
    "SplittingResult",
    "Study",
    "TargetProbability",
    "quantify",
    "read_quantification",
]


#: The quantification methods :func:`quantify` provides for a study, in the
#: order the documentation presents them.
QUANTIFICATION_METHODS: tuple[str, ...] = (
    "monte_carlo",
    "exact",
    "discretised",
    "cross_entropy",
    "splitting",
)


@dataclass(frozen=True)
class Study:
    """The question a quantification answers, stated once for every
    method: the probability that ``target`` is the first declared target
    (feared event) reached by ``horizon``.

    ``instants`` are the reporting instants of the Monte-Carlo detailed
    result (ascending, within ``[0, horizon]``; the horizon alone when
    omitted) and ``seed`` the master seed of the methods that draw at
    random (Monte-Carlo, cross-entropy and splitting): a method that ignores
    one does not record it. ``threads`` sets the worker
    count; no result depends on it.
    """

    target: str
    horizon: float
    instants: tuple[float, ...] | None = None
    seed: int = 0
    threads: int | None = None

    def _to_json(self) -> str:
        return json.dumps(
            {
                "target": self.target,
                "horizon": float(self.horizon),
                "instants": None
                if self.instants is None
                else [float(t) for t in self.instants],
                "seed": self.seed,
                "threads": self.threads,
            },
            default=_json_value,
        )


@dataclass(frozen=True)
class TargetProbability:
    """The probability of a study's target, with the uncertainty its
    method can state.

    ``kind`` is ``"confidence_interval"`` for Monte-Carlo simulation:
    ``estimate`` is ``reached / replicas`` and ``[low, high]`` its interval
    at ``level``, built by ``interval_method`` (``"wilson"``, the estimate
    being a proportion). It is ``"bounds"`` for the explorations:
    ``[low, high]`` are the guaranteed lower and upper bounds (on the
    discretised model for a discretised exploration), ``inconclusive``
    flags a gap above the declared tolerance, and ``error_estimate`` is the
    discretisation error estimate by refinement (an estimate, not a bound;
    ``None`` for an exact exploration and when no refinement was made).
    It is ``"weighted_estimate"`` for cross-entropy: ``estimate`` is the
    mean over ``replicas`` of each replica's likelihood ratio when it
    reached the target first, ``standard_error`` its standard error,
    ``[low, high]`` its interval at ``level`` (``interval_method``
    ``"weighted_normal"``, not clamped at zero), ``reached`` the unweighted
    hit count, ``effective_sample_size`` how many equally weighted hits the
    estimate is worth (near 1: one replica carries it),
    ``relative_error`` the standard error over the estimate, and
    ``inconclusive`` is true when that effective sample size is below the
    method's threshold: the estimate and its interval are then not to be
    trusted, however narrow the interval.
    ``"splitting_estimate"`` uses ``"batch_student"``: the mean and
    standard error of ``batches`` independent estimates. ``extinct_batches``
    counts zero estimates from extinction. Any extinction, a nonpositive
    lower bound or relative half-width above one makes it ``inconclusive``.
    Fields that do not apply to a kind are ``None``.
    """

    kind: str
    low: float
    high: float
    estimate: float | None = None
    reached: int | None = None
    replicas: int | None = None
    level: float | None = None
    interval_method: str | None = None
    inconclusive: bool | None = None
    error_estimate: float | None = None
    standard_error: float | None = None
    effective_sample_size: float | None = None
    relative_error: float | None = None
    batches: int | None = None
    extinct_batches: int | None = None

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> TargetProbability:
        if raw["kind"] == "confidence_interval":
            return cls(
                kind=raw["kind"],
                low=raw["low"],
                high=raw["high"],
                estimate=raw["estimate"],
                reached=raw["reached"],
                replicas=raw["replicas"],
                level=raw["level"],
                interval_method=raw["method"],
            )
        if raw["kind"] == "weighted_estimate":
            return cls(
                kind=raw["kind"],
                low=raw["low"],
                high=raw["high"],
                estimate=raw["estimate"],
                reached=raw["reached"],
                replicas=raw["replicas"],
                level=raw["level"],
                interval_method=raw["method"],
                standard_error=raw["standard_error"],
                effective_sample_size=raw["effective_sample_size"],
                relative_error=raw.get("relative_error"),
                inconclusive=raw["inconclusive"],
            )
        if raw["kind"] == "splitting_estimate":
            return cls(
                kind=raw["kind"],
                low=raw["low"],
                high=raw["high"],
                estimate=raw["estimate"],
                standard_error=raw["standard_error"],
                level=raw["level"],
                interval_method=raw["method"],
                batches=raw["batches"],
                extinct_batches=raw["extinct_batches"],
                inconclusive=raw["inconclusive"],
            )
        if raw["kind"] == "bounds":
            return cls(
                kind=raw["kind"],
                low=raw["lower"],
                high=raw["upper"],
                inconclusive=raw["inconclusive"],
                error_estimate=raw.get("error_estimate"),
            )
        raise SimulationError(
            f"probability kind {raw['kind']!r} is not known to this engine"
        )


@dataclass(frozen=True)
class BiasFamily:
    """A family of constant-rate exponential transitions sharing one bias
    factor in a cross-entropy campaign: its ``label``, its member
    ``transitions`` (qualified names), whether it is a ``repair`` family
    (started at 1, never escalated) and the ``factor`` the final campaign
    drew its rates with."""

    label: str
    transitions: tuple[str, ...]
    repair: bool
    factor: float


@dataclass(frozen=True)
class PilotIteration:
    """One pilot iteration of a cross-entropy fit: the ``factors`` it drew
    with (in family order), the ``hits`` among its replicas, whether it
    saw none and ``escalated`` the non-repair factors, and the
    ``effective_sample_size`` of its hit weights (a pilot below the
    method's threshold cannot confirm the fit)."""

    factors: tuple[float, ...]
    hits: int
    escalated: bool
    effective_sample_size: float


@dataclass(frozen=True)
class CrossEntropyResult:
    """The detailed result of a cross-entropy quantification: the
    ``target``, the ``reached`` and ``replicas`` counts of the final
    campaign, the ``families`` with their fitted factors, the pilot
    ``history`` and whether the fit ``converged`` before its iteration
    cap. The estimate and its diagnostics are the envelope's
    :class:`TargetProbability`."""

    target: str
    reached: int
    replicas: int
    families: tuple[BiasFamily, ...]
    history: tuple[PilotIteration, ...]
    converged: bool

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> CrossEntropyResult:
        return cls(
            target=raw["target"],
            reached=raw["reached"],
            replicas=raw["replicas"],
            families=tuple(
                BiasFamily(
                    label=f["label"],
                    transitions=tuple(f["transitions"]),
                    repair=f["repair"],
                    factor=f["factor"],
                )
                for f in raw["families"]
            ),
            history=tuple(
                PilotIteration(
                    factors=tuple(it["factors"]),
                    hits=it["hits"],
                    escalated=it["escalated"],
                    effective_sample_size=it["effective_sample_size"],
                )
                for it in raw["history"]
            ),
            converged=raw["converged"],
        )


@dataclass(frozen=True)
class SplittingLevel:
    """A selection level, the number killed and its exact survival fraction."""

    level: float
    killed: int
    survival_fraction: float


@dataclass(frozen=True)
class SplittingBatch:
    """An independent batch estimate, including zero when extinct."""

    estimate: float
    extinct: bool
    levels: tuple[SplittingLevel, ...]
    trajectories: int


@dataclass(frozen=True)
class SplittingResult:
    """Independent splitting batches in deterministic order.

    The envelope's probability carries the Student interval over these
    batches. Extinct batches count as zero; any extinction flags the estimate
    inconclusive. ``trajectories`` counts initial particles and restarts.
    """

    target: str
    seed: int
    batches: tuple[SplittingBatch, ...]
    extinct_batches: int
    estimate_inconclusive: bool
    trajectories: int

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> SplittingResult:
        return cls(
            target=raw["target"],
            seed=raw["seed"],
            batches=tuple(
                SplittingBatch(
                    estimate=b["estimate"],
                    extinct=b["extinct"],
                    levels=tuple(SplittingLevel(**level) for level in b["levels"]),
                    trajectories=b["trajectories"],
                )
                for b in raw["batches"]
            ),
            extinct_batches=raw["extinct_batches"],
            estimate_inconclusive=raw["estimate_inconclusive"],
            trajectories=raw["trajectories"],
        )


@dataclass(frozen=True)
class Quantification:
    """The answer to a :class:`Study` (:func:`quantify`), whatever the
    method: the ``raichu.quantification`` envelope, version 3 (versions 1 and 2
    envelopes read as well).

    ``method`` and ``settings`` are the method and the settings it
    applied, defaults resolved. The provenance follows: ``engine_version``,
    the ``model`` name and ``model_hash`` (``sha256:`` and the SHA-256 of
    the sealed model as canonical JSON, comparable between results of the
    same engine version), ``target`` and ``horizon``, and ``instants`` and
    ``seed`` only for a method that uses them (``None`` otherwise).
    ``probability`` is the probability that ``target`` is the first
    declared target reached by ``horizon``, with its uncertainty, and
    ``detail`` the method's own result, unchanged: :class:`McEstimates`
    for Monte-Carlo simulation, :class:`Exploration` for the explorations,
    :class:`CrossEntropyResult` for cross-entropy, :class:`SplittingResult`
    for splitting.
    """

    format: str
    version: int
    method: str
    settings: dict[str, Any]
    engine_version: str
    model: str
    model_hash: str
    target: str
    horizon: float
    instants: tuple[float, ...] | None
    seed: int | None
    probability: TargetProbability
    detail: McEstimates | Exploration | CrossEntropyResult | SplittingResult
    fmu_units: list[dict[str, Any]] = field(default_factory=list)
    _text: str = field(default="", init=False, compare=False, repr=False)

    @classmethod
    def _from_json(cls, text: str) -> Quantification:
        raw = json.loads(text)
        provenance = raw["provenance"]
        detail = raw["detail"]
        # One branch per detail kind, and no fallback: a kind this reader
        # does not know is refused rather than read as another.
        parsed: McEstimates | Exploration | CrossEntropyResult | SplittingResult
        if "monte_carlo" in detail:
            parsed = _mc_estimates(detail["monte_carlo"])
        elif "exploration" in detail:
            parsed = Exploration._from_json(json.dumps(detail["exploration"]))
        elif "cross_entropy" in detail:
            parsed = CrossEntropyResult._from_dict(detail["cross_entropy"])
        elif "splitting" in detail:
            parsed = SplittingResult._from_dict(detail["splitting"])
        else:
            raise SimulationError(
                f"detail kind {sorted(detail)!r} is not known to this engine"
            )
        instants = provenance.get("instants")
        quantification = cls(
            format=raw["format"],
            version=raw["version"],
            method=raw["method"]["name"],
            settings=raw["method"]["settings"],
            engine_version=provenance["engine_version"],
            model=provenance["model"],
            model_hash=provenance["model_hash"],
            target=provenance["target"],
            horizon=provenance["horizon"],
            instants=None if instants is None else tuple(instants),
            seed=provenance.get("seed"),
            probability=TargetProbability._from_dict(raw["probability"]),
            detail=parsed,
            fmu_units=provenance.get("fmu_units", []),
        )
        # The engine's own text, set only here: an object derived from this
        # one (``dataclasses.replace``) or built by hand does not carry it,
        # and cannot claim a document it was not read from.
        object.__setattr__(quantification, "_text", text)
        return quantification

    def to_json(self, path: str | Path | None = None) -> str:
        """The envelope as JSON text, exactly as the engine wrote it; also
        written to ``path`` when one is given. :func:`read_quantification`
        reads it back into an equal object.

        Only an object the engine produced, or :func:`read_quantification`
        read, holds that text: one built by hand or derived with
        ``dataclasses.replace`` raises :class:`SimulationError` rather than
        write a document that does not describe it."""
        if not self._text:
            raise SimulationError(
                "this Quantification was not produced by the engine or read from "
                "an envelope, so it holds no envelope text to write"
            )
        if path is not None:
            Path(path).write_text(self._text, encoding="utf-8")
        return self._text


def read_quantification(source: str | Path) -> Quantification:
    """Read a quantification envelope written by
    :meth:`Quantification.to_json`.

    ``source`` is a path (a :class:`~pathlib.Path`, or a string that is not
    JSON text) or the JSON text itself. Raises :class:`SimulationError` for
    another format, a version newer than this engine reads, or an envelope
    whose method, probability and detail do not belong together.
    """
    if isinstance(source, Path) or not source.lstrip().startswith("{"):
        source = Path(source).read_text(encoding="utf-8")
    validate_quantification(source)
    return Quantification._from_json(source)


def _json_value(value: Any) -> Any:
    """A numpy scalar or array (or anything with ``tolist``/``item``) as
    the plain value ``json.dumps`` writes, as the other entry points accept
    them through the extension."""
    for convert in ("tolist", "item"):
        if hasattr(value, convert):
            return getattr(value, convert)()
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def _quantify_study(
    model: Model, study: Study, method: str, settings: dict[str, Any],
    require_parallel: bool = False,
) -> Quantification:
    return Quantification._from_json(
        quantify_json(
            model.json,
            study._to_json(),
            method,
            json.dumps(settings, default=_json_value),
            model.allow_fmu_import,
            str(model.base_dir),
            require_parallel,
        )
    )


# The keywords of the fault-tree path, read off its own signature so the
# two cannot drift apart.
_FAULT_TREE_OPTIONS = frozenset(inspect.signature(_quantify_fault_tree).parameters) - {"tree"}


def quantify(
    tree: Model | FaultTree | str | Path,
    study: Study | None = None,
    *,
    method: str | None = None,
    **options: Any,
) -> Quantification | FaultTreeQuantification:
    """Quantify a study on a model, or a fault tree.

    **A study on a model**, ``quantify(model, study, method=...,
    **settings)``: the one entry point to RAICHU's five engines. ``study``
    is a :class:`Study` (the feared event, the horizon, and for the methods
    that draw at random the reporting instants and the seed); ``method`` is one of
    :data:`QUANTIFICATION_METHODS`:

    - ``"monte_carlo"``, Monte-Carlo simulation: ``nb_runs`` (required),
      ``confidence`` (default 0.95), ``quantiles``. One campaign stopped at
      the targets; the probability is the proportion of replicas whose
      first target reached is the study's, with a Wilson interval.
    - ``"exact"``, exact exploration (the Markov family):
      ``min_probability``, ``max_length``, ``max_failures``,
      ``max_branches``, ``gap_tolerance``, ``rel_precision``,
      ``max_terms``, as :func:`explore` takes them.
    - ``"discretised"``, discretised exploration: the same cut-offs,
      ``gap_tolerance``, ``level`` and ``refine``.
    - ``"cross_entropy"``, biased Monte-Carlo with factors fitted by
      cross-entropy, for feared events too rare for a plain campaign:
      ``nb_runs`` (required), ``pilot_runs``, ``max_iterations``,
      ``smoothing``, ``tolerance``, ``confidence``, ``initial_factor``,
      ``escalation_ratio``, ``factor_min``, ``factor_max``, ``fit``,
      ``min_effective_sample_size`` and ``families`` (qualified transition
      name to family label). The probability is a weighted estimate with its
      diagnostics and an ``inconclusive`` verdict; a campaign that never
      reaches the target raises :class:`SimulationError`.

    - ``"splitting"``, adaptive multilevel splitting: ``importance`` is
      required, as ``{"kind": "attribute", "name": "component.score"}`` for a
      declared numeric attribute, or ``{"kind": "cut_sets"}`` for the
      automatic score built from the target's minimal cut sets (optional
      ``max_cut_sets``, default 1000; beyond the cap the campaign is refused
      by name, and a model fault-tree generation refuses, a non-monotone
      state read for instance, refuses ``cut_sets`` quoting its reason:
      declare an attribute then). ``particles``, ``batches``,
      ``max_iterations`` use driver defaults; ``confidence`` defaults to 0.95
      and ``score_grid`` to no extra dates. Scores are read at completed
      instants and optional grid dates. The probability is a mean over
      independent batches with a Student interval; extinction is
      inconclusive, an iteration cap raises without an estimate. Every
      native law is supported through age-conditioned restarts; FMUs are
      refused.

    Returns a :class:`Quantification`. An unknown method, or a setting that
    belongs to another method, raises :class:`SimulationError` naming the
    valid ones before anything runs; so does an unknown target. The GIL is
    released while the engine runs.

    **A fault tree**, ``quantify(tree, top=..., mission_time=...,
    max_bdd_nodes=..., cut_set_limit=..., cut_sets=..., engine=...,
    max_order=..., min_cut_probability=..., max_cut_sets=...,
    max_expansions=...)``: ``tree`` is a :class:`FaultTree`, an OpenPSA
    document as text, or the path of one, quantified with binary decision
    diagrams, a module too large for them falling back to its cut sets
    under cutoffs with a guaranteed upper bound (``engine``, default
    ``"auto"``); see :class:`FaultTreeQuantification` for what comes back. The first argument
    decides which: a :class:`Model` is a study, anything else a fault tree.
    The first parameter keeps the name ``tree`` it had when this function
    quantified fault trees only, so a keyword call still works.
    """
    if isinstance(tree, Model):
        if not isinstance(study, Study):
            raise TypeError(
                "quantify(model, study, method=...) takes a pyraichu.Study as "
                f"its second argument, got {type(study).__name__}"
            )
        if method is None:
            raise SimulationError(
                "quantify(model, study) needs a method; the methods are "
                + ", ".join(f"`{name}`" for name in QUANTIFICATION_METHODS)
            )
        require_parallel = bool(options.pop("require_parallel", False))
        return _quantify_study(tree, study, method, options, require_parallel)
    if study is not None or method is not None:
        raise TypeError(
            "a study and a method apply to a pyraichu.Model; a fault tree is "
            "quantified with quantify(tree, top=..., mission_time=...)"
        )
    unknown = sorted(set(options) - _FAULT_TREE_OPTIONS)
    if unknown:
        raise TypeError(f"quantify() got an unexpected keyword argument {unknown[0]!r}")
    return _quantify_fault_tree(tree, **options)
