"""Python binding for the RAICHU hybrid (PDMP) simulation engine.

The heavy lifting happens in the Rust extension module
``pyraichu._pyraichu``; this package is the thin, typed, Pythonic wrapper
around it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ._pyraichu import (
    ModelError,
    SimulationError,
    __version__,
    monte_carlo_json,
    simulate_json,
    validate_model,
)
from .journal import Cascade, JournalQuery, TransitionHistory, AttributeChange

__all__ = [
    "Cascade",
    "Event",
    "IndicatorEstimate",
    "JournalQuery",
    "TransitionHistory",
    "AttributeChange",
    "McEstimates",
    "Model",
    "ModelError",
    "SimulationError",
    "SimulationResult",
    "__version__",
    "expand_model",
    "load_model",
    "monte_carlo",
    "simulate",
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


def expand_model(source: str | dict[str, Any]) -> dict[str, Any]:
    """Expand the ``"plugins"`` section of a model (if any) into core
    material and return the resulting core-schema dict — the audit
    window on plugin translations (see :mod:`pyraichu.plugins`)."""
    from .plugins import expand_model as _expand

    model = json.loads(source) if isinstance(source, str) else source
    return _expand(model)


def load_model(source: str | dict[str, Any]) -> Model:
    """Load and validate a model from a JSON string or a dict.

    Models carrying a ``"plugins"`` section (specialized object schemas
    — ObjFlow, ObjFM, ObjEvent, …) are expanded into the core schema
    first; use :func:`expand_model` to inspect the translation.
    Raises :class:`ModelError` with a precise, typed message when the
    model is invalid (never a crash).
    """
    model = json.loads(source) if isinstance(source, str) else source
    if "plugins" in model:
        model = expand_model(model)
    model_json = json.dumps(model)
    validate_model(model_json)
    name = model.get("name", "")
    return Model(json=model_json, name=name)


def _series_dict(raw_series: list[dict[str, Any]]) -> dict[str, list[tuple[float, Any]]]:
    return {
        series["name"]: [(t, _value_to_python(v)) for t, v in series["points"]]
        for series in raw_series
    }


@dataclass(frozen=True)
class IndicatorEstimate:
    """Monte-Carlo estimates of one indicator over the schedule."""

    name: str
    instants: list[float]
    mean: list[float]
    std: list[float]
    sojourn_mean: list[float]
    sojourn_std: list[float]
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


def simulate(
    model: Model,
    t_max: float = math.inf,
    journal: bool = False,
    confluence_check: bool = False,
    samples: list[float] | None = None,
    seed: int = 0,
    rng_stream: int = 0,
) -> SimulationResult:
    """Run one simulation of ``model`` up to ``t_max``.

    ``samples`` is an ascending list of instants at which every
    indicator is recorded (dense output for continuous variables).
    ``seed``/``rng_stream`` drive the stochastic laws (ignored by
    deterministic models); the same pair replays bit-identically.
    The GIL is released while the Rust engine runs. Raises
    :class:`SimulationError` on typed engine failures (instantaneous
    loop, non-confluence when ``confluence_check`` is enabled, …).
    """
    raw = json.loads(
        simulate_json(
            model.json, t_max, journal, confluence_check, samples, seed, rng_stream
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
    )
