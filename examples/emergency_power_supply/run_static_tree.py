"""Static fault tree of the benchmark, quantified by binary decision diagrams.

Usage: python run_static_tree.py

A BDMP is a fault tree whose leaves carry Markov processes and whose
triggers make some leaves wait until they are needed (Bouissou and Bon
2003). Dropping the triggers and the order constraints leaves the static
fault tree a classical tool would quantify: every gate keeps its inputs,
a priority-AND gate becomes an AND gate (the order of its inputs is lost),
an approximation OR gate stays the single leaf it is in the dynamic
model, failing at the sum of its leaves' rates (exact here: none of those
leaves feeds another gate), a leaf failing in function gets the law `1 - exp(-lambda t)` and a leaf
failing on demand the constant probability `gamma`. Every basic event is
then independent and never repaired, which is the non-repairable reading.

The tree is written as an OpenPSA document under `model/` (an adaptation
of the Figaro 0 file, not versioned) and quantified at the three mission
times of Bouissou, Khan, Katoen and Krcal (2020), Table 3. Writes
`results/static_tree.json`.
"""

from __future__ import annotations

import json
from xml.sax.saxutils import quoteattr

import common
import pyraichu
import translate

#: Mission times of the non-repairable comparison (2020, Table 3), in hours.
MISSION_TIMES = (100.0, 1000.0, 10000.0)
GATE_TYPES = {"and_gate": "and", "or_gate": "or", "then_gate": "and"}


def gate_inputs(obj) -> list[str]:
    """The inputs of a gate: its sons, or the two inputs of a priority-AND."""
    if obj.type == "then_gate":
        return obj.interface["first"] + obj.interface["second"]
    return obj.interface["sons"]


def open_psa() -> tuple[str, dict]:
    """The static tree as an OpenPSA document, and a count of its parts."""
    fmodel = translate.Parser(translate.tokenize(translate.figaro_source())).parse()
    constants = translate.Translation(fmodel).const
    top = fmodel.objects[translate.UNDESIRABLE_EVENT].interface["sons"][0]
    gates, events, seen = [], [], set()

    def ref(name: str) -> str:
        cls = fmodel.objects[name].type
        kind = "gate" if cls in GATE_TYPES else "basic-event"
        return f"<{kind} name={quoteattr(name)}/>"

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        obj = fmodel.objects[name]
        if obj.type in GATE_TYPES:
            inputs = gate_inputs(obj)
            gates.append(
                f"<define-gate name={quoteattr(name)}><{GATE_TYPES[obj.type]}>"
                + "".join(ref(i) for i in inputs)
                + f"</{GATE_TYPES[obj.type]}></define-gate>"
            )
            for i in inputs:
                visit(i)
        elif obj.type in ("f_leaf", "approx_or_gate"):
            rate = constants[
                (name, "agg_lambda" if obj.type == "approx_or_gate" else "lambda")
            ]
            events.append(
                f"<define-basic-event name={quoteattr(name)}><exponential>"
                f'<float value="{rate!r}"/><system-mission-time/>'
                "</exponential></define-basic-event>"
            )
        elif obj.type == "i_leaf":
            gamma = constants[(name, "gamma")]
            events.append(
                f"<define-basic-event name={quoteattr(name)}>"
                f'<float value="{gamma!r}"/></define-basic-event>'
            )
        else:
            raise ValueError(f"{name}: no static reading for a {obj.type}")

    visit(top)
    document = (
        '<?xml version="1.0"?><opsa-mef><define-fault-tree name="eps_static">'
        + "".join(gates)
        + "".join(events)
        + "</define-fault-tree></opsa-mef>\n"
    )
    counts = {"top": top, "gates": len(gates), "basic_events": len(events)}
    return document, counts


def main() -> None:
    document, counts = open_psa()
    translate.MODEL_DIR.mkdir(exist_ok=True)
    path = translate.MODEL_DIR / "eps_static_tree.xml"
    path.write_text(document)
    rows = []
    with common.timed("static_tree", **counts):
        for t in MISSION_TIMES:
            q = pyraichu.quantify(
                str(path), top=counts["top"], mission_time=t, cut_sets=True
            )
            rows.append(
                {
                    "mission_time": t,
                    "top_probability": q.probability,
                    "exact": q.exact,
                    "method": q.method,
                    "minimal_cut_sets": q.cut_set_count,
                }
            )
            print(
                f"{t:>7.0f} h  {q.probability:.4e}  {q.method}  {q.cut_set_count} cut sets"
            )
    result = {"format": "eps_example.static_tree", **counts, "rows": rows}
    (common.RESULTS / "static_tree.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
