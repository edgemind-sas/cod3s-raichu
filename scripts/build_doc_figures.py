"""Regenerate every documentation figure.

Each ``docs/figures/<name>.py`` (files starting with ``_`` are helpers) is
run in turn; it writes its SVGs under ``docs/assets/figures/``. The figures
are committed, so the site build needs neither the engine nor matplotlib:
run this after a change that moves a number a figure shows.

    .venv/bin/python scripts/build_doc_figures.py            # all figures
    .venv/bin/python scripts/build_doc_figures.py tutorial   # names containing "tutorial"
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    scripts = sorted(
        p for p in FIGURES.glob("*.py")
        if not p.name.startswith("_") and (not wanted or any(w in p.stem for w in wanted))
    )
    if not scripts:
        print("no figure script matches", wanted, file=sys.stderr)
        return 1
    sys.path.insert(0, str(FIGURES))
    for script in scripts:
        print(f"-> {script.name}")
        runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
