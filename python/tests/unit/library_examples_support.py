"""Behaviour records and flat-model loaders for the example libraries.

Each published example is written twice: as the flat core model its page
or its generator builds, and as a component type library. A library talks
to its neighbours through ports where the flat model reads them by name,
so the two documents can never be equal. What must be equal is what they
do, and that is what a *record* captures: every event (time and
transition), how many times each transition fired, and every indicator at
fixed instants, for one seed.

Two records are compared exactly. A rename map translates the names of the
library side into the flat model's names first, for the one case where a
catalogued mode renames a transition.

The loaders rebuild the flat models the way the documentation does: a page
model by executing the page's non-skipped ``python`` blocks, a generated
model by importing ``examples/<name>/model.py`` under a module name of its
own (both generators are called ``model``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import textwrap
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pyraichu

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"
EXAMPLES = REPO / "examples"

_FENCE = re.compile(
    r"(?P<skip><!--\s*skip\s*-->\n)?(?P<indent>[ \t]*)```python\b[^\n]*\n(?P<body>.*?)\n(?P=indent)```",
    re.DOTALL,
)


def record(
    model,
    *,
    t_max: float,
    instants: Sequence[float],
    seed: int = 0,
    rename: Mapping[str, str] | None = None,
    **options,
) -> dict:
    """Run `model` once and return what it did.

    `model` is a loaded `pyraichu.Model`. `rename` maps a transition path of
    this model (``component.automaton.transition``) to the name it is
    compared under; paths it does not name are kept. `options` go to
    `pyraichu.simulate` (an integrator setting, `max_step`).
    """
    rename = rename or {}
    result = pyraichu.simulate(model, t_max=t_max, samples=list(instants), seed=seed, **options)
    events = [(e.time, rename.get(e.transition, e.transition)) for e in result.events]
    return {
        "seed": seed,
        "events": events,
        "firings": dict(sorted(Counter(name for _, name in events).items())),
        "samples": {
            name: [(time, value) for time, value in series]
            for name, series in sorted(result.samples.items())
        },
    }


def differences(expected: dict, observed: dict, *, limit: int = 10) -> list[str]:
    """Readable lines naming every disagreement, empty when the records agree.

    The comparison is exact: same seed, same events at the same dates, same
    firing counts, same indicator values at every instant.
    """
    lines: list[str] = []
    if expected["seed"] != observed["seed"]:
        lines.append(f"seed: expected {expected['seed']}, got {observed['seed']}")
    if expected["firings"] != observed["firings"]:
        keys = sorted(set(expected["firings"]) | set(observed["firings"]))
        for key in keys:
            e, o = expected["firings"].get(key, 0), observed["firings"].get(key, 0)
            if e != o:
                lines.append(f"firings of {key}: expected {e}, got {o}")
    for index, (e, o) in enumerate(zip(expected["events"], observed["events"])):
        if e != o:
            lines.append(f"first differing event #{index}: expected {e}, got {o}")
            break
    if len(expected["events"]) != len(observed["events"]):
        lines.append(
            f"event count: expected {len(expected['events'])}, "
            f"got {len(observed['events'])}"
        )
    missing = sorted(set(expected["samples"]) - set(observed["samples"]))
    extra = sorted(set(observed["samples"]) - set(expected["samples"]))
    if missing:
        lines.append(f"indicators missing: {missing}")
    if extra:
        lines.append(f"indicators not in the reference: {extra}")
    for name in sorted(set(expected["samples"]) & set(observed["samples"])):
        if len(expected["samples"][name]) != len(observed["samples"][name]):
            lines.append(
                f"{name}: expected {len(expected['samples'][name])} samples, "
                f"got {len(observed['samples'][name])}"
            )
        for e, o in zip(expected["samples"][name], observed["samples"][name]):
            if e != o:
                lines.append(f"{name} at t={e[0]}: expected {e[1]!r}, got {o[1]!r}")
                break
    return lines[:limit]


def page_namespace(page: str) -> dict:
    """Run a documentation page's executed ``python`` blocks; return their namespace.

    `page` is relative to ``docs/`` (``"examples/heated-room.md"``). The page
    runs in a temporary directory where ``examples/`` points at the
    repository's, as the documentation tests run it, so a block reads a
    committed library by the path a reader uses from a checkout root.
    """
    text = (DOCS / page).read_text()
    namespace: dict = {}
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "examples").symlink_to(EXAMPLES, target_is_directory=True)
        os.chdir(tmp)
        try:
            for match in _FENCE.finditer(text):
                if not match.group("skip"):
                    exec(compile(textwrap.dedent(match.group("body")), page, "exec"), namespace)  # noqa: S102
        finally:
            os.chdir(here)
    return namespace


def example_module(name: str, file: str = "model") -> ModuleType:
    """Import ``examples/<name>/<file>.py`` under a module name of its own."""
    module_name = f"example_{name}_{file}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = EXAMPLES / name / f"{file}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def library_model(name: str, system: str = "system.json", study: str | None = None):
    """Load ``examples/<name>/library/`` as a model: its library, a system, a study."""
    folder = EXAMPLES / name / "library"
    return pyraichu.load_system(
        folder / system,
        [folder / "library.json"],
        folder / study if study else None,
    )
