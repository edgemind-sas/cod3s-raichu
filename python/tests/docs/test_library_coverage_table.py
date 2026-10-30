"""Every key of the core model has a row in the library format's coverage
table, so a construct added to the model cannot go without a home in the
three documents unnoticed."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[3] / "docs" / "reference"


def keys_of(section: str) -> set[str]:
    """The keys of the first table under `## <section>` in the model reference."""
    text = (DOCS / "model-schema.md").read_text(encoding="utf-8")
    body = text.split(f"\n## {section}\n", 1)[1]
    first_table = []
    for line in body.splitlines():
        if line.startswith("|"):
            first_table.append(line)
        elif first_table:
            break
    return {m.group(1) for line in first_table if (m := re.match(r"\| `([a-z_]+)`", line))}


def coverage_table() -> str:
    text = (DOCS / "library-format.md").read_text(encoding="utf-8")
    return text.split("\n## Coverage of the core model\n", 1)[1].split("\n## ", 1)[0]


def test_every_model_key_has_a_home():
    table = coverage_table()
    missing = sorted(k for k in keys_of("Model") if f"`{k}`" not in table)
    assert not missing, missing


def test_every_component_collection_has_a_home():
    table = coverage_table()
    missing = sorted(k for k in keys_of("Component") - {"name"} if f"`{k}`" not in table)
    assert not missing, missing
