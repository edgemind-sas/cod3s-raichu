"""The gas production benchmark as a component type library.

`model.py` writes the four cases as flat models, one component at a time,
each reading the others by name. This file writes the same plant as data
documents built from two reusable types:

- `ProductionUnit`: a unit that fails (exponential, faster while it runs
  at its maximum) and is repaired (lognormal). It exports whether it is up
  (`std.state_out`), reads whether the other unit is up, and in cases 1 to
  3 its repair refills the reservoir at once by writing the reservoir's
  level through its `reservoir_in` port;
- `BufferReservoir`: the reservoir, its level, whether it is empty, partly
  filled or full, and the delivered share of the demand.

One library serves the four cases: they differ by instance parameters
only, written in `system_case<N>.json`.

Nothing computes at expansion, so every derived constant (a net flow, a
delivered share) is computed here from `model.py`'s constants, exactly as
`model.py` computes it. Run this file to rewrite `library/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY_DIR = HERE / "library"


def _model():
    """`model.py` of this directory, under a name of its own."""
    name = "gas_production_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _model()


def param(name: str) -> dict:
    return {"$param": name}


def num(x) -> dict:
    return {"op": "const", "value": {"kind": "float", "value": x}}


def own(attribute: str) -> dict:
    return M.attr("$self", attribute)


def own_state(automaton: str, state: str) -> dict:
    return M.in_state("$self", automaton, state)


def any_in(port: str) -> dict:
    return {"op": "port_agg", "port": {"component": "$self", "port": port}, "agg": "any"}


def sum_in(port: str) -> dict:
    return {"op": "port_agg", "port": {"component": "$self", "port": port}, "agg": "sum"}


def number(name: str, description: str, default=None) -> dict:
    spec = {"name": name, "kind": "number", "description": description}
    if default is not None:
        spec["default"] = default
    return spec


def boolean(name: str, description: str, default=None) -> dict:
    spec = {"name": name, "kind": "bool", "description": description}
    if default is not None:
        spec["default"] = default
    return spec


def production_unit() -> dict:
    other_up = any_in("other_up_in")
    at_max = M.negate(other_up)
    # Case 4: both units also run at their maximum while they refill the
    # reservoir.
    at_max_filling = M.either(at_max, M.negate(any_in("reservoir_full_in")))
    refill = {
        "target": {"component": "$port", "attribute": "reservoir_in"},
        "value": M.if_(other_up, num(param("capacity")), sum_in("reservoir_in")),
    }
    return {
        "name": "ProductionUnit",
        "description": (
            "A production unit: up or down, failing at `failure_rate` and at"
            " `failure_rate_at_max` while it runs at its maximum, repaired after"
            " a lognormal duration. It exports whether it is up on `up_out`."
            " With `refill`, its repair refills the reservoir at once when the"
            " other unit is up."
        ),
        "parameters": [
            number("failure_rate", "Failure rate at partial load, in 1/h."),
            number("failure_rate_at_max", "Failure rate while running at its maximum, in 1/h."),
            number("repair_mu", "Mean of the logarithm of the repair duration (ln h)."),
            number("repair_sigma", "Standard deviation of the logarithm of the repair duration."),
            boolean("refill", "Whether a repair that brings both units up refills the reservoir at once (cases 1 to 3)."),
            boolean("at_max_while_filling", "Whether the unit also runs at its maximum while the reservoir is not full (case 4).", False),
            number("capacity", "Reservoir capacity, in m3: the level a refill restores.", M.CAPACITY),
        ],
        "uses": [
            {
                "fragment": "std.state_out",
                "with": {"name": "up", "automaton": "health", "states": ["up"], "init": True},
            }
        ],
        "body": {
            "ports": [
                {"name": "other_up_in", "dir": "in"},
                {"name": "reservoir_full_in", "dir": "in"},
                {"name": "reservoir_in", "dir": "in"},
            ],
            "automata": [
                {
                    "name": "health",
                    "states": ["up", "down"],
                    "init": "up",
                    "transitions": [
                        {
                            "name": "fail",
                            "source": "up",
                            "targets": ["down"],
                            "distrib": "exp",
                            "rate_expr": M.if_(
                                {
                                    "$choose": "at_max_while_filling",
                                    "cases": {"true": at_max_filling, "false": at_max},
                                },
                                num(param("failure_rate_at_max")),
                                num(param("failure_rate")),
                            ),
                            "kind": "failure",
                            "monitored": True,
                        },
                        {
                            "name": "repair",
                            "source": "down",
                            "targets": ["up"],
                            "distrib": "lognormal",
                            "mu": param("repair_mu"),
                            "sigma": param("repair_sigma"),
                            "kind": "repair",
                            "monitored": True,
                            "effects": {"$choose": "refill", "cases": {"true": [refill], "false": []}},
                        },
                    ],
                }
            ],
        },
    }


def buffer_reservoir() -> dict:
    up1, up2 = any_in("unit1_up_in"), any_in("unit2_up_in")
    level = own("level")
    empty, full = own_state("store", "empty"), own_state("store", "full")
    net = M.if_(
        up1,
        M.if_(up2, num(0.0), num(param("net_unit1_alone"))),
        M.if_(up2, num(param("net_unit2_alone")), num(param("net_none"))),
    )
    flow = {
        "$choose": "filling",
        "cases": {
            "true": M.if_(M.both(up1, up2), num(param("net_filling")), net),
            "false": net,
        },
    }
    rate = M.if_(
        empty,
        {"op": "max", "args": [flow, num(0.0)]},
        M.if_(full, {"op": "min", "args": [flow, num(0.0)]}, flow),
    )
    inst = {"distrib": "inst", "probs": []}
    return {
        "name": "BufferReservoir",
        "description": (
            "The reservoir covering the units' shortfall: its level follows the"
            " net flow, held at 0 while empty and at the capacity while full."
            " It reads which unit is up on `unit1_up_in` and `unit2_up_in`,"
            " publishes its level on `level_out` and whether it is full on"
            " `full_out`, and computes the delivered share of the demand."
        ),
        "parameters": [
            number("capacity", "Capacity, in m3.", M.CAPACITY),
            number("net_unit1_alone", "Net flow with U1 up alone, in m3/h.", M.PHI1_MAX - M.PHI_NOM),
            number("net_unit2_alone", "Net flow with U2 up alone, in m3/h.", M.PHI2_MAX - M.PHI_NOM),
            number("net_none", "Net flow with both units down, in m3/h.", -M.PHI_NOM),
            number("net_filling", "Net flow with both units up while it refills (case 4), in m3/h.", M.PHI1_MAX + M.PHI2_MAX - M.PHI_NOM),
            number("share_unit1", "Delivered share of the demand with U1 alone and the reservoir empty.", M.PHI1_MAX / M.PHI_NOM),
            number("share_unit2", "Delivered share of the demand with U2 alone and the reservoir empty.", M.PHI2_MAX / M.PHI_NOM),
            boolean("filling", "Whether both units up refill it at their maximum instead of keeping it as it is (case 4)."),
        ],
        "uses": [
            {
                "fragment": "std.state_out",
                "with": {"name": "full", "automaton": "store", "states": ["full"], "init": True},
            }
        ],
        "body": {
            "attributes": [
                {"name": "level", "kind": "float", "init": {"kind": "float", "value": param("capacity")}},
                {"name": "delivered", "kind": "float", "init": {"kind": "float", "value": 1.0}},
                {"name": "total_loss", "kind": "bool", "init": {"kind": "bool", "value": False}},
            ],
            "ports": [
                {"name": "unit1_up_in", "dir": "in"},
                {"name": "unit2_up_in", "dir": "in"},
                {"name": "level_out", "dir": "out", "attr": "level"},
            ],
            "automata": [
                {
                    "name": "store",
                    "states": ["full", "partial", "empty"],
                    "init": "full",
                    "transitions": [
                        {"name": "start_draining", "source": "full", "targets": ["partial"],
                         "guard": M.cmp("lt", flow, num(0.0)), **inst},
                        {"name": "run_dry", "source": "partial", "targets": ["empty"],
                         "guard": M.both(M.cmp("le", level, num(0.0)), M.cmp("lt", flow, num(0.0))),
                         "distrib": "watched"},
                        {"name": "fill_up", "source": "partial", "targets": ["full"],
                         "guard": M.both(M.cmp("ge", level, num(param("capacity"))), M.cmp("gt", flow, num(0.0))),
                         "distrib": "watched"},
                        {"name": "restart", "source": "empty", "targets": ["partial"],
                         "guard": M.either(M.cmp("gt", flow, num(0.0)), M.cmp("gt", level, num(0.0))),
                         **inst},
                    ],
                }
            ],
            "equations": [{"target": "level", "kind": "ode", "expr": rate}],
            "sensitive_functions": [
                {
                    "name": "production",
                    "effects": [
                        {
                            "target": {"component": "$self", "attribute": "delivered"},
                            "value": M.if_(
                                M.negate(empty),
                                num(1.0),
                                M.if_(up1, num(param("share_unit1")),
                                      M.if_(up2, num(param("share_unit2")), num(0.0))),
                            ),
                        },
                        {
                            "target": {"component": "$self", "attribute": "total_loss"},
                            "value": M.both(empty, M.negate(up1), M.negate(up2)),
                        },
                    ],
                }
            ],
        },
    }


def library() -> dict:
    return {
        "raichu_library": {"format": 1},
        "library": {
            "name": "gas_production",
            "version": "1.0.0",
            "imports": ["std"],
            "description": (
                "The gas production benchmark (Labeau and Dutuit 2004; Eymard and"
                " Mercier 2008) as a component type library: two production units"
                " and the buffer reservoir that covers their shortfall."
            ),
            "types": [production_unit(), buffer_reservoir()],
        },
    }


def system(case: int) -> dict:
    lam1, lam2, lam1_max, lam2_max = M.RATES[case]
    refill = case != 4

    def unit(name: str, lam: float, lam_max: float) -> dict:
        mu, sigma = M.REPAIR[name]
        return {
            "name": name,
            "type": "ProductionUnit",
            "parameters": {
                "failure_rate": lam,
                "failure_rate_at_max": lam_max,
                "repair_mu": mu,
                "repair_sigma": sigma,
                "refill": refill,
                "at_max_while_filling": not refill,
            },
        }

    return {
        "raichu_system": {"format": 1},
        "system": {
            "name": f"gas_production_case{case}",
            "library": {"name": "gas_production", "version": "1.0.0"},
            "instances": [
                unit("U1", lam1, lam1_max),
                unit("U2", lam2, lam2_max),
                {"name": "reservoir", "type": "BufferReservoir", "parameters": {"filling": not refill}},
            ],
            "connections": [
                {"from": "U1.up_out", "to": "U2.other_up_in"},
                {"from": "U2.up_out", "to": "U1.other_up_in"},
                {"from": "U1.up_out", "to": "reservoir.unit1_up_in"},
                {"from": "U2.up_out", "to": "reservoir.unit2_up_in"},
                {"from": "reservoir.level_out", "to": "U1.reservoir_in"},
                {"from": "reservoir.level_out", "to": "U2.reservoir_in"},
                {"from": "reservoir.full_out", "to": "U1.reservoir_full_in"},
                {"from": "reservoir.full_out", "to": "U2.reservoir_full_in"},
            ],
            "indicators": M.build(4)["indicators"],
        },
    }


def documents() -> dict[str, dict]:
    """Every document of `library/`, by file name."""
    out = {"library.json": library()}
    for case in M.CASES:
        out[f"system_case{case}.json"] = system(case)
    return out


def main() -> None:
    LIBRARY_DIR.mkdir(exist_ok=True)
    for name, document in documents().items():
        (LIBRARY_DIR / name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        print(name)


if __name__ == "__main__":
    main()
