"""Execute every runnable code block in the documentation so the docs
never drift from the engine.

Policy:
- ```python fenced blocks on a documentation page are concatenated (in
  order) and executed in one shared namespace, so a later block may
  build on an earlier one. A block preceded by an ``<!-- skip -->``
  HTML comment is illustrative and not executed (use sparingly).
- ```json fenced blocks preceded by an ``<!-- model -->`` comment are
  loaded with ``pyraichu.load_model`` (must be valid models).

This harness depends only on ``pyraichu`` (no PyCATSHOO), so it is safe
to ship and to run in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"

PAGES = [DOCS / "index.md"] + sorted(
    p
    for sub in ("tutorial", "reference", "guides", "benchmarks", "pycatshoo")
    for p in (DOCS / sub).glob("*.md")
)

_FENCE = re.compile(
    r"(?P<skip><!--\s*(?P<marker>skip|model)\s*-->\n)?"
    r"```(?P<lang>python|json)\b[^\n]*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _blocks(text: str):
    for m in _FENCE.finditer(text):
        yield m.group("lang"), m.group("marker"), m.group("body")


def _page_id(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


@pytest.mark.parametrize("page", PAGES, ids=_page_id)
def test_doc_page_examples_run(page: Path):
    text = page.read_text()
    namespace: dict = {}
    ran = 0
    for lang, marker, body in _blocks(text):
        # Snippet-include directives (`--8<-- "file"`) are resolved by
        # mkdocs at build time; they are not runnable as written.
        if body.strip().startswith("--8<--"):
            continue
        if lang == "python":
            if marker == "skip":
                continue
            exec(compile(body, str(page), "exec"), namespace)  # noqa: S102
            ran += 1
        elif lang == "json" and marker == "model":
            import pyraichu

            pyraichu.load_model(body)
            ran += 1
    if not PAGES:  # pragma: no cover - guards an empty glob
        pytest.skip("no documentation pages yet")
    assert ran >= 0  # a page may legitimately have no runnable blocks
