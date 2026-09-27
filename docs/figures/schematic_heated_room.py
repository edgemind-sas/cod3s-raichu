"""Schematic of the heated room example, light and dark."""

from __future__ import annotations

from _style import DOCS

OUT = DOCS / "assets" / "schematics"

THEMES = {
    "light": {"ink": "#333f48", "wall": "#333f48", "air": "#e8edf4", "main": "#1f416d",
              "backup": "#5b8c5a", "hot": "#ef7b26", "cold": "#3a78b5"},
    "dark": {"ink": "#dfe5ec", "wall": "#dfe5ec", "air": "#26303c", "main": "#8fb3e0",
             "backup": "#8fc58d", "hot": "#ef7b26", "cold": "#8fb3e0"},
}


def heater(x: int, colour: str, name: str, band: str, ink: str) -> str:
    fins = "".join(
        f'<line x1="{x + 10 + 12 * k}" y1="236" x2="{x + 10 + 12 * k}" y2="282" '
        f'stroke="{colour}" stroke-width="4"/>' for k in range(7))
    return (f'<g id="{name}"><rect x="{x}" y="230" width="100" height="58" rx="6" fill="none" '
            f'stroke="{colour}" stroke-width="3"/>{fins}'
            f'<text x="{x + 50}" y="314" font-size="18" fill="{colour}" text-anchor="middle">{name}</text>'
            f'<text x="{x + 50}" y="338" font-size="15" fill="{ink}" text-anchor="middle">{band}</text></g>')


def svg(t: dict) -> str:
    ink = t["ink"]
    waves = "".join(
        f'<path d="M{600 + 22 * k},150 q8,-12 16,0 t16,0" stroke="{t["cold"]}" stroke-width="3" '
        f'fill="none"/>' for k in range(3))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 400" font-family="Open Sans, DejaVu Sans, Arial, sans-serif">
<title>Heated room: a main and a backup heater, heat lost to the outside</title>
<defs><marker id="loss" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{t['cold']}"/></marker></defs>
<g id="room">
  <rect x="60" y="60" width="440" height="300" fill="{t['air']}"/>
  <rect x="60" y="60" width="440" height="300" fill="none" stroke="{t['wall']}" stroke-width="6"/>
  <text x="280" y="120" font-size="22" fill="{ink}" text-anchor="middle">room T</text>
  <text x="280" y="150" font-size="16" fill="{t['hot']}" text-anchor="middle">frozen: T &lt; 10 °C</text>
</g>
{heater(120, t['main'], 'main', 'on &lt; 19 °C, off &gt; 21 °C', ink)}
{heater(330, t['backup'], 'backup', 'on &lt; 17 °C, off &gt; 20 °C', ink)}
<g id="losses">
  <path d="M506,270 H600" stroke="{t['cold']}" stroke-width="4" fill="none" marker-end="url(#loss)"/>
  <text x="580" y="300" font-size="16" fill="{t['cold']}" text-anchor="middle">losses</text>
  <text x="580" y="320" font-size="16" fill="{t['cold']}" text-anchor="middle">0.1 (T − T_out)</text>
</g>
<g id="outside">
  {waves}
  <text x="660" y="110" font-size="18" fill="{ink}" text-anchor="middle">outside</text>
  <text x="660" y="182" font-size="15" fill="{ink}" text-anchor="middle">T_out: −4 to 8 °C</text>
  <text x="660" y="200" font-size="15" fill="{ink}" text-anchor="middle">daily cycle</text>
</g>
</svg>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (OUT / f"heated-room-{name}.svg").write_text(svg(theme))


if __name__ == "__main__":
    main()
