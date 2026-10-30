"""Component type libraries: library, system and study documents.

A model can be written without code as three data documents: a library
of parameterised component types (``raichu_library``), a system that
instantiates and connects them (``raichu_system``), and a study that
activates the modes the types catalogue and says how to run
(``raichu_study``). The engine expands them into the flat core model;
this module reads the documents and hands them to it.

Documents are JSON. A path ending in ``.yaml`` or ``.yml`` is also read,
through PyYAML when it is installed; YAML reads unquoted ``on``, ``off``,
``yes`` and ``no`` as booleans, so quote a name spelled that way.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

from ._pyraichu import (
    ModelError,
    check_library_json,
    check_quantification_method_json,
    derive_evaluation_order_json,
    embedded_library_json,
    expand_documents_json,
    standard_library_json,
)

Document = Union[str, Path, dict[str, Any]]


def read_document(source: Document) -> dict[str, Any]:
    """A document as a dictionary: a dict is returned as is, a string
    starting with ``{`` is parsed as JSON, anything else is a path (JSON,
    or YAML for ``.yaml`` and ``.yml``)."""
    if isinstance(source, dict):
        return source
    if isinstance(source, str) and source.lstrip().startswith("{"):
        return json.loads(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError(
                f"{path} is YAML and PyYAML is not installed; install it or write the document as JSON"
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _text(document: Document) -> str:
    return json.dumps(read_document(document))


def _texts(documents: Iterable[Document]) -> list[str]:
    return [_text(d) for d in documents]


def expand_documents(
    system: Document,
    libraries: Iterable[Document] = (),
    study: Document | None = None,
) -> dict[str, Any]:
    """Expand a system (and optionally a study) into a flat core model.

    Returns ``{"model": <sealed model document>, "provenance": {...},
    "simulation": {...} or None, "warnings": [...]}``. The provenance maps
    every element path of the model (``"attribute:Srv.f_service_fed_out"``)
    to the document node it came from. Raises :class:`ModelError` naming
    the document, the JSON Pointer of the fault and what is wrong.
    """
    study_text = None if study is None else _text(study)
    raw = expand_documents_json(_text(system), _texts(libraries), study_text)
    return json.loads(raw)


def load_system(
    system: Document,
    libraries: Iterable[Document] = (),
    study: Document | None = None,
    *,
    allow_fmu_import: bool = False,
    base_dir: Path | None = None,
):
    """Expand and load a system, ready to simulate.

    The returned :class:`pyraichu.Model` carries the provenance of every
    element (``model.provenance``) and the study's run settings
    (``model.simulation``), which :func:`run_study` reads.

    A system declaring FMU units needs ``allow_fmu_import=True``, as a flat
    model does. Their relative paths resolve at run time against
    ``base_dir``: the system file's directory when ``system`` is a path,
    the working directory otherwise.
    """
    from . import Model, validate_model

    if base_dir is None:
        base_dir = Path.cwd()
        if isinstance(system, Path) or (
            isinstance(system, str) and not system.lstrip().startswith("{")
        ):
            base_dir = Path(system).resolve().parent
    expansion = expand_documents(system, libraries, study)
    body = expansion["model"].get("model", expansion["model"])
    units = body.get("fmu_units", [])
    if units and not allow_fmu_import:
        raise ModelError(
            f"FMU import for unit {units[0].get('name', '<unnamed>')!r} "
            "requires allow_fmu_import=True"
        )
    quantification = expansion.get("quantification")
    if quantification is not None:
        method = quantification["method"]
        try:
            check_quantification_method_json(method["name"], json.dumps(method.get("settings", {})))
        except ModelError as error:
            raise ModelError(f"study document, at `/study/quantification/method`: {error}") from None
    simulation = expansion["simulation"]
    if simulation is not None:
        _check_keywords(
            "/study/simulation/options",
            _RUN_ENTRY[simulation["method"]](),
            simulation.get("options", {}),
            _SIMULATION_OWNED,
            base_dir,
        )
    analyses = expansion.get("analyses")
    for name, params in (analyses or {}).items():
        _check_keywords(f"/study/analyses/{name}", _ANALYSIS_ENTRY[name](), params, {"model"}, base_dir)
    model_json = json.dumps(expansion["model"])
    validate_model(model_json)
    return Model(
        json=model_json,
        name=body.get("name", ""),
        provenance=expansion["provenance"],
        simulation=expansion["simulation"],
        quantification=quantification,
        analyses=analyses,
        warnings=tuple(expansion["warnings"]),
        allow_fmu_import=allow_fmu_import,
        base_dir=base_dir,
    )


#: The run entry point of each `simulation.method`, and of each analysis
#: a study may request, resolved lazily (they live in the package root).
_RUN_ENTRY = {
    "monte_carlo": lambda: _root().monte_carlo,
    "sequences": lambda: _root().run_sequences,
}
_ANALYSIS_ENTRY = {
    "fault_tree": lambda: _root().fault_tree,
    "importance": lambda: _root().importance,
    "sequence_analysis": lambda: _root().analyse_sequences,
}
#: Keywords a `simulation` section states itself, never as an option.
_SIMULATION_OWNED = {"model", "nb_runs", "t_max", "samples", "seed", "stop_at_targets"}


def _root():
    from . import __name__ as package
    import sys

    return sys.modules[package]


def _check_keywords(
    pointer: str, entry, given: dict[str, Any], owned: set[str], base_dir: Path
) -> None:
    """Refuse, by name, a keyword `entry` does not take, one the study
    states elsewhere, a required one left out, and an object-valued one
    (`flow`, `observations`, `condition`) whose fields do not build, so a
    misspelt setting fails when the study is loaded rather than when it
    runs."""
    parameters = inspect.signature(entry).parameters
    accepted = set(parameters) - owned
    unknown = sorted(set(given) - accepted)
    if unknown:
        raise ModelError(
            f"study document, at `{pointer}`: {', '.join(f'`{k}`' for k in unknown)} "
            f"is not a parameter of `{entry.__name__}` here (accepted: "
            f"{', '.join(sorted(accepted))})"
        )
    missing = sorted(
        name
        for name, parameter in parameters.items()
        if name not in owned
        and name not in given
        and parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    if missing:
        raise ModelError(
            f"study document, at `{pointer}`: `{entry.__name__}` needs "
            f"{', '.join(f'`{k}`' for k in missing)}"
        )
    try:
        _keywords(given, base_dir)
    except (TypeError, ValueError) as error:
        raise ModelError(f"study document, at `{pointer}`: {error}") from None


def _keywords(given: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Keyword arguments from a study entry, the objects the entry points
    take built from their JSON form: `flow` as a :class:`FlowConfig`,
    `observations` as :class:`Observation` items, `condition` as a
    :class:`SequenceCondition`, and a relative `raw_path` resolved against
    the study's base directory."""
    root = _root()
    out = dict(given)
    if isinstance(out.get("flow"), dict):
        out["flow"] = root.FlowConfig(**out["flow"])
    if isinstance(out.get("observations"), list):
        out["observations"] = [
            root.Observation(**o) if isinstance(o, dict) else o for o in out["observations"]
        ]
    if isinstance(out.get("condition"), dict):
        out["condition"] = root.SequenceCondition(**out["condition"])
    if isinstance(out.get("raw_path"), str) and not Path(out["raw_path"]).is_absolute():
        out["raw_path"] = str(base_dir / out["raw_path"])
    return out


def check_library(library: Document, libraries: Iterable[Document] = ()) -> list[dict[str, Any]]:
    """Check every type of a library without a system.

    Each type is expanded once with its defaults and once per value of
    each enum parameter. Returns one entry per type: ``{"type", "status"}``
    with status ``valid``, ``invalid`` (with ``errors``) or ``unchecked``
    (a required parameter has no default; ``parameters`` names them).
    """
    raw = check_library_json(_text(library), _texts(libraries))
    return json.loads(raw)


def derive_evaluation_order(model: Document) -> list[dict[str, str]] | None:
    """The evaluation order of a flat core model, derived from what each
    explicit step reads: a step comes after every step whose result it
    reads, independent steps keep declaration order, and the steps of a
    cycle stay together. ``None`` for a model with no explicit step.

    This is the order the expander gives every expanded model; it does not
    look at an ``evaluation_order`` the model already carries.
    """
    return json.loads(derive_evaluation_order_json(_text(model)))


def embedded_library(name: str) -> dict[str, Any]:
    """A library shipped with this engine, by name (``"std"``), as a
    library document; :class:`ModelError` naming the embedded libraries
    for an unknown name."""
    return json.loads(embedded_library_json(name))


def standard_library() -> dict[str, Any]:
    """The standard library ``std`` shipped with this engine, as a library
    document."""
    return json.loads(standard_library_json())


def schema(kind: str) -> dict[str, Any]:
    """The published JSON Schema (draft 2020-12) of a ``library``,
    ``system`` or ``study`` document, as shipped with this engine."""
    from importlib.resources import files

    if kind not in ("library", "system", "study"):
        raise ValueError(f"no schema {kind!r}: the documents are library, system and study")
    resource = files("pyraichu") / "schema" / f"raichu-{kind}.schema.json"
    return json.loads(resource.read_text(encoding="utf-8"))


def run_study(model, *, section: str | None = None, threads: int | None = None):
    """Run a model loaded from a study with the study's own settings.

    The sections a study may carry, each run by the entry point it names:

    - ``simulation``: ``method: monte_carlo`` returns the
      :func:`pyraichu.monte_carlo` estimates, sampled at the study's
      ``instants`` (the horizon alone when none are given), and
      ``method: sequences`` the :func:`pyraichu.run_sequences` campaign,
      each with the section's ``options`` as extra keywords;
    - ``quantification``: the :func:`pyraichu.quantify` envelope of its
      method;
    - ``fault_tree``, ``importance``, ``sequence_analysis`` (under
      ``analyses``): :func:`pyraichu.fault_tree`,
      :func:`pyraichu.importance` and :func:`pyraichu.analyse_sequences`
      with the parameters the study writes.

    A study carrying one section runs it; a study carrying several runs the
    one ``section`` names. ``threads``, when given, applies to the runs that
    take it and wins over a thread count the study states.
    """
    root = _root()
    available = [name for name in ("simulation", "quantification") if getattr(model, name, None)]
    available += list(getattr(model, "analyses", None) or {})
    if not available:
        raise ModelError(
            f"model {model.name!r} carries no run settings; load it with a study that has a "
            "`simulation`, a `quantification` or an `analyses` section"
        )
    if section is None:
        if len(available) > 1:
            raise ModelError(
                f"model {model.name!r} carries several sections ({', '.join(available)}); "
                "name the one to run with section=..."
            )
        section = available[0]
    if section not in available:
        raise ModelError(f"model {model.name!r} carries no `{section}` section")
    if section == "quantification":
        from ._quantification import Study, quantify

        q = model.quantification
        study = Study(
            target=q["target"],
            horizon=q["horizon"],
            instants=tuple(q["instants"]) if q.get("instants") is not None else None,
            seed=q.get("seed", 0),
            threads=threads,
        )
        return quantify(model, study, method=q["method"]["name"], **q["method"].get("settings", {}))
    if section != "simulation":
        params = _keywords(model.analyses[section], model.base_dir)
        entry = _ANALYSIS_ENTRY[section]()
        if threads is not None and "threads" in inspect.signature(entry).parameters:
            params["threads"] = threads
        return entry(model, **params)
    settings = model.simulation
    options = _keywords(settings.get("options", {}), model.base_dir)
    if threads is not None:
        options["threads"] = threads
    if settings["method"] == "sequences":
        return root.run_sequences(
            model,
            nb_runs=settings["nb_runs"],
            t_max=settings["horizon"],
            seed=settings.get("seed", 0),
            **options,
        )
    return root.monte_carlo(
        model,
        nb_runs=settings["nb_runs"],
        t_max=settings["horizon"],
        samples=settings.get("instants") or [settings["horizon"]],
        seed=settings.get("seed", 0),
        stop_at_targets=settings.get("stop_at_targets", False),
        **options,
    )
