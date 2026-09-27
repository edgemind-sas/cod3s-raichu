"""Shared chart style for the documentation figures.

Every figure is drawn twice, once per site colour scheme, and saved as SVG
with its text kept as text. A page references both files with
mkdocs-material's ``#only-light`` / ``#only-dark`` suffixes::

    ![Unavailability](../assets/figures/tutorial-unavailability-light.svg#only-light){ .figure }
    ![Unavailability](../assets/figures/tutorial-unavailability-dark.svg#only-dark){ .figure }

A figure script defines ``draw(fig, theme)`` and calls :func:`render`.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import pyraichu  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "assets" / "figures"

_FENCE = re.compile(
    r"(?P<skip><!--\s*skip\s*-->\n)?```python\b[^\n]*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def page_namespace(page: str) -> dict:
    """Run a page's executed ``python`` blocks and return their namespace.

    A figure reuses the model its page defines instead of restating it, so
    the page and its charts cannot drift apart. Blocks marked
    ``<!-- skip -->`` are left out, as the documentation tests leave them
    out; the page runs in a temporary directory, as there.
    """
    text = (DOCS / page).read_text()
    namespace: dict = {}
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for match in _FENCE.finditer(text):
                if not match.group("skip"):
                    exec(compile(match.group("body"), page, "exec"), namespace)  # noqa: S102
        finally:
            os.chdir(here)
    return namespace

NAVY = "#1f416d"
ORANGE = "#ef7b26"
GREY = "#c9d4e6"


@dataclass(frozen=True)
class Theme:
    """The colours one scheme draws with."""

    name: str
    text: str
    grid: str
    #: Series colours, in order: the first is the main quantity.
    series: tuple[str, ...]
    #: Fill of a confidence band around the first series.
    band: str


LIGHT = Theme(
    name="light",
    text="#333f48",
    grid="#d9dee5",
    series=(NAVY, ORANGE, "#5b8c5a", "#8a6bb0", "#7f7f7f"),
    band="#1f416d33",
)
DARK = Theme(
    name="dark",
    text="#dfe5ec",
    grid="#3a4450",
    series=("#8fb3e0", ORANGE, "#8fc58d", "#b89ddb", "#b0b0b0"),
    band="#8fb3e040",
)


def _rc(theme: Theme) -> dict:
    return {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "text.color": theme.text,
        "axes.labelcolor": theme.text,
        "axes.edgecolor": theme.text,
        "xtick.color": theme.text,
        "ytick.color": theme.text,
        "axes.grid": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=list(theme.series)),
        "legend.frameon": False,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    }


def render(
    name: str,
    draw: Callable[[plt.Figure, Theme], None],
    size: tuple[float, float] = (7.0, 3.8),
) -> list[Path]:
    """Draw ``draw`` once per theme and write ``<name>-<theme>.svg``.

    The engine version is written into the SVG metadata, so a figure says
    which engine produced its numbers.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in (LIGHT, DARK):
        with plt.rc_context(_rc(theme)):
            fig = plt.figure(figsize=size, layout="constrained")
            draw(fig, theme)
            path = OUT / f"{name}-{theme.name}.svg"
            fig.savefig(
                path,
                format="svg",
                metadata={
                    "Creator": f"RAICHU {pyraichu.__version__} documentation figures",
                    "Date": None,
                },
            )
            plt.close(fig)
            written.append(path)
    return written
