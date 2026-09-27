"""Schematic of the heated tank (Aldemir's benchmark), light and dark.

Hand-laid SVG rather than a plot: named groups, real text, editable in
Inkscape. Written as a script only so that the two colour variants share
one drawing.
"""

from __future__ import annotations

from _style import DOCS

OUT = DOCS / "assets" / "schematics"

THEMES = {
    "light": {"ink": "#333f48", "water": "#c9d4e6", "pump": "#1f416d",
              "valve": "#1f416d", "hot": "#ef7b26", "mark": "#7a8691"},
    "dark": {"ink": "#dfe5ec", "water": "#2f4b70", "pump": "#8fb3e0",
             "valve": "#8fb3e0", "hot": "#ef7b26", "mark": "#9aa6b2"},
}

# Tank geometry: 1 m of level is 22 px, the floor of the tank is at y = 360.
FLOOR, M = 360, 22.0


def y(level_m: float) -> float:
    return FLOOR - level_m * M


def svg(t: dict) -> str:
    ink, mark = t["ink"], t["mark"]
    marks = [(10, "overflow 10 m", t["hot"]), (8, "high set-point 8 m", mark),
             (6, "low set-point 6 m", mark), (4, "dry-out 4 m", t["hot"])]
    lines = []
    for level, label, colour in marks:
        dash = "" if colour == t["hot"] else ' stroke-dasharray="6 4"'
        lines.append(
            f'<line x1="300" y1="{y(level)}" x2="470" y2="{y(level)}" stroke="{colour}" '
            f'stroke-width="2"{dash}/>'
            f'<text x="480" y="{y(level) + 5}" font-size="16" fill="{colour}">{label}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440" font-family="Open Sans, DejaVu Sans, Arial, sans-serif">
<title>Heated tank: two inlet pumps, one outlet valve, a heat source</title>
<defs><marker id="arrow-{t['pump'][1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{t['pump']}"/></marker></defs>
<g id="tank">
  <rect x="300" y="{y(7)}" width="170" height="{7 * M}" fill="{t['water']}"/>
  <path d="M300,{y(11.5)} V{FLOOR} H470 V{y(11.5)}" fill="none" stroke="{ink}" stroke-width="4"/>
  <text x="385" y="{y(2.2)}" font-size="18" fill="{ink}" text-anchor="middle">h, θ</text>
</g>
<g id="levels">{''.join(lines)}</g>
<g id="pump1">
  <circle cx="120" cy="{y(9.5)}" r="26" fill="none" stroke="{t['pump']}" stroke-width="3"/>
  <text x="120" y="{y(9.5) + 6}" font-size="18" fill="{t['pump']}" text-anchor="middle">P1</text>
  <path d="M40,{y(9.5)} H94 M146,{y(9.5)} H292" stroke="{t['pump']}" stroke-width="3" fill="none" marker-end="url(#arrow-{t['pump'][1:]})"/>
</g>
<g id="pump2">
  <circle cx="120" cy="{y(6.5)}" r="26" fill="none" stroke="{t['pump']}" stroke-width="3"/>
  <text x="120" y="{y(6.5) + 6}" font-size="18" fill="{t['pump']}" text-anchor="middle">P2</text>
  <path d="M40,{y(6.5)} H94 M146,{y(6.5)} H292" stroke="{t['pump']}" stroke-width="3" fill="none" marker-end="url(#arrow-{t['pump'][1:]})"/>
</g>
<g id="inlet-label"><text x="40" y="{y(11)}" font-size="16" fill="{ink}">inflow 1.5 m/h each, 15 °C</text></g>
<g id="valve">
  <path d="M470,{FLOOR - 14} H520 M560,{FLOOR - 14} H640" stroke="{t['valve']}" stroke-width="3" fill="none" marker-end="url(#arrow-{t['pump'][1:]})"/>
  <path d="M520,{FLOOR - 28} L560,{FLOOR} V{FLOOR - 28} L520,{FLOOR} Z" fill="none" stroke="{t['valve']}" stroke-width="3"/>
  <text x="540" y="{FLOOR + 24}" font-size="18" fill="{t['valve']}" text-anchor="middle">V</text>
  <text x="600" y="{FLOOR + 24}" font-size="16" fill="{ink}">outflow</text>
  <text x="600" y="{FLOOR + 44}" font-size="16" fill="{ink}">1.5 m/h</text>
</g>
<g id="heater">
  <path d="M320,{FLOOR + 12} q10,-10 20,0 t20,0 t20,0 t20,0 t20,0 t20,0 t20,0" stroke="{t['hot']}" stroke-width="3" fill="none"/>
  <text x="250" y="{FLOOR + 30}" font-size="16" fill="{t['hot']}" text-anchor="end">heat source</text>
</g>
<g id="overheat"><text x="385" y="{y(12.4)}" font-size="16" fill="{t['hot']}" text-anchor="middle">overheat: θ ≥ 100 °C</text></g>
</svg>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (OUT / f"heated-tank-{name}.svg").write_text(svg(theme))


if __name__ == "__main__":
    main()
