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

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

from ._pyraichu import (
    ModelError,
    check_library_json,
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
):
    """Expand and load a system, ready to simulate.

    The returned :class:`pyraichu.Model` carries the provenance of every
    element (``model.provenance``) and the study's run settings
    (``model.simulation``), which :func:`run_study` reads.
    """
    from . import Model, validate_model

    expansion = expand_documents(system, libraries, study)
    model_json = json.dumps(expansion["model"])
    validate_model(model_json)
    body = expansion["model"].get("model", expansion["model"])
    return Model(
        json=model_json,
        name=body.get("name", ""),
        provenance=expansion["provenance"],
        simulation=expansion["simulation"],
        warnings=tuple(expansion["warnings"]),
    )


def check_library(library: Document, libraries: Iterable[Document] = ()) -> list[dict[str, Any]]:
    """Check every type of a library without a system.

    Each type is expanded once with its defaults and once per value of
    each enum parameter. Returns one entry per type: ``{"type", "status"}``
    with status ``valid``, ``invalid`` (with ``errors``) or ``unchecked``
    (a required parameter has no default; ``parameters`` names them).
    """
    raw = check_library_json(_text(library), _texts(libraries))
    return json.loads(raw)


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


def run_study(model, *, threads: int | None = None):
    """Run a model loaded from a study with the study's run settings.

    ``method: monte_carlo`` returns the :func:`pyraichu.monte_carlo`
    estimates, sampled at the study's ``instants`` (the horizon alone when
    none are given); ``method: sequences`` returns the
    :func:`pyraichu.run_sequences` campaign.
    """
    from . import monte_carlo, run_sequences

    settings = model.simulation
    if not settings:
        raise ModelError(
            f"model {model.name!r} carries no run settings; load it with a study that has a `simulation` section"
        )
    if settings["method"] == "sequences":
        return run_sequences(
            model,
            nb_runs=settings["nb_runs"],
            t_max=settings["horizon"],
            seed=settings.get("seed", 0),
            threads=threads,
        )
    return monte_carlo(
        model,
        nb_runs=settings["nb_runs"],
        t_max=settings["horizon"],
        samples=settings.get("instants") or [settings["horizon"]],
        seed=settings.get("seed", 0),
        threads=threads,
        stop_at_targets=settings.get("stop_at_targets", False),
    )
