"""Execute every runnable code block in the documentation so the docs
never drift from the engine.

Policy:
- ```python fenced blocks on a documentation page are concatenated (in
  order) and executed in one shared namespace, so a later block may
  build on an earlier one. A block preceded by an ``<!-- skip -->``
  HTML comment is illustrative and not executed (use sparingly).
- ```json fenced blocks preceded by an ``<!-- model -->`` comment are
  loaded with ``pyraichu.load_model`` (must be valid models).
- A page runs in a fresh temporary directory, so an example that writes
  a file (a raw corpus, an exploration) leaves nothing in the checkout.
  An ``examples`` entry there points at the repository's ``examples/``,
  so a page reads a committed library by the path a reader uses from a
  checkout root.

This harness depends only on ``pyraichu`` (no PyCATSHOO), so it is safe
to ship and to run in CI.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"

PAGES = [DOCS / "index.md"] + sorted(
    p
    for sub in ("tutorial", "engine", "examples", "reference", "guides", "benchmarks", "pycatshoo")
    for p in (DOCS / sub).glob("*.md")
)

# A fence closes at the indentation it opened with: a block nested in a
# tab or an admonition is indented, and a closing fence matched without
# its indentation would swallow everything up to the next block.
_FENCE = re.compile(
    r"(?P<skip><!--\s*(?P<marker>skip|model)\s*-->\n)?"
    r"(?P<indent>[ \t]*)```(?P<lang>python|json)\b[^\n]*\n(?P<body>.*?)\n(?P=indent)```",
    re.DOTALL,
)


def _blocks(text: str):
    for m in _FENCE.finditer(text):
        # A nested block runs as written at the top level.
        yield m.group("lang"), m.group("marker"), textwrap.dedent(m.group("body"))


def _page_id(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


@pytest.mark.parametrize("page", PAGES, ids=_page_id)
def test_doc_page_examples_run(page: Path, tmp_path: Path, monkeypatch):
    (tmp_path / "examples").symlink_to(REPO / "examples", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
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

            document = json.loads(body)
            has_fmu = bool(pyraichu.model_body(document).get("fmu_units"))
            pyraichu.load_model(document, allow_fmu_import=has_fmu)
            ran += 1
    if not PAGES:  # pragma: no cover - guards an empty glob
        pytest.skip("no documentation pages yet")
    assert ran >= 0  # a page may legitimately have no runnable blocks
