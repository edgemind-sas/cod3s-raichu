"""The gas production benchmark as a RAICHU model, one per case.

Two production units, U1 (3200 m3/h at most) and U2 (5500 m3/h at most),
serve a constant demand of 7500 m3/h that neither meets alone, with a
reservoir of 220 000 m3 that supplies the shortfall while one or both
units are being repaired. Failures are exponential, repairs lognormal.
The case comes from Labeau and Dutuit (lambda-mu 14, 2004, hal-01570847);
the data and the four variants are those of Eymard and Mercier (Reliability
Engineering & System Safety 93, 2008, doi:10.1016/j.ress.2006.12.001,
hal-00693079, section 2.1, Tables 1, 2 and 7 to 12), both deposited on HAL
under CC BY 4.0. The model is written here from those published facts.

Cases (Eymard and Mercier, Table 1):

- 1: lambda1 = 1/20 000, lambda2 = 1/4000 per hour; the reservoir is full
  again as soon as both units are up.
- 2: the same, failure rates ten times higher.
- 3: case 2, and a unit running alone at its maximum fails ten times more
  often again (lambda').
- 4: case 3 without the instant refill: with both units up the reservoir
  refills at 3200 + 5500 - 7500 = 1200 m3/h, both units running at their
  maximum (rates lambda') until it is full.

Usage: python model.py  (writes model/gas_production_case<N>.json)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

#: Maximum rates of the units and nominal demand, in m3/h.
PHI1_MAX, PHI2_MAX, PHI_NOM = 3200.0, 5500.0, 7500.0
#: Reservoir capacity, in m3.
CAPACITY = 220_000.0
#: Year length used for the published frequencies (Eymard and Mercier, p. 5).
YEAR = 8766.0
#: Repair laws: lognormal, ln(duration) ~ N(mu, sigma^2), median exp(mu)
#: (Eymard and Mercier, Table 2; Labeau and Dutuit, Table 1).
REPAIR = {"U1": (0.23, 2.25), "U2": (0.50, 1.83)}
#: Failure rates per hour: (lambda1, lambda2, lambda'1, lambda'2), Table 1.
RATES = {
    1: (1 / 20_000, 1 / 4000, 1 / 20_000, 1 / 4000),
    2: (1 / 2000, 1 / 400, 1 / 2000, 1 / 400),
    3: (1 / 2000, 1 / 400, 1 / 200, 1 / 40),
    4: (1 / 2000, 1 / 400, 1 / 200, 1 / 40),
}
CASES = tuple(RATES)


def num(x: float) -> dict:
    return {"op": "const", "value": {"kind": "float", "value": float(x)}}


def flag(x: bool) -> dict:
    return {"op": "const", "value": {"kind": "bool", "value": x}}


def attr(component: str, name: str) -> dict:
    return {"op": "attr", "attr": {"component": component, "attribute": name}}


def in_state(component: str, automaton: str, state: str) -> dict:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


def both(*args: dict) -> dict:
    return {"op": "bool", "bool_op": "and", "args": list(args)}


def either(*args: dict) -> dict:
    return {"op": "bool", "bool_op": "or", "args": list(args)}


def negate(arg: dict) -> dict:
    return {"op": "bool", "bool_op": "not", "args": [arg]}


def cmp(op: str, lhs: dict, rhs: dict) -> dict:
    return {"op": "cmp", "cmp": op, "lhs": lhs, "rhs": rhs}


def if_(cond: dict, then: dict, otherwise: dict) -> dict:
    return {"op": "if", "cond": cond, "then": then, "otherwise": otherwise}


def up(unit: str) -> dict:
    return in_state(unit, "health", "up")


LEVEL = attr("reservoir", "level")
EMPTY = in_state("reservoir", "store", "empty")
FULL = in_state("reservoir", "store", "full")


def net_flow() -> dict:
    """dX3/dt before saturation, m3/h (Eymard and Mercier, Tables 7, 10):
    0 or +1200 (case 4) with both up, -4300 with U1 alone, -2000 with U2
    alone, -7500 with both down. Case 4's +1200 is applied by `flow`."""
    return if_(
        up("U1"),
        if_(up("U2"), num(0.0), num(PHI1_MAX - PHI_NOM)),
        if_(up("U2"), num(PHI2_MAX - PHI_NOM), num(-PHI_NOM)),
    )


def unit(name: str, other: str, case: int) -> dict:
    """A production unit: up/down, exponential failure, lognormal repair.

    The failure rate is lambda' while the unit runs at its maximum: alone
    (the other unit down), and in case 4 also while both refill the
    reservoir (both up, reservoir not full). The repair is not restarted
    when the other unit fails: its guard does not read the other unit."""
    lam, lam_max = (
        (RATES[case][0], RATES[case][2])
        if name == "U1"
        else (RATES[case][1], RATES[case][3])
    )
    at_max = negate(up(other))
    if case == 4:
        at_max = either(at_max, negate(FULL))
    mu, sigma = REPAIR[name]
    repair = {
        "name": "repair",
        "source": "down",
        "targets": ["up"],
        "distrib": "lognormal",
        "mu": mu,
        "sigma": sigma,
        "kind": "repair",
        "monitored": True,
    }
    if case != 4:
        # Cases 1-3: the reservoir is full again as soon as both are up
        # (Eymard and Mercier, Table 8: every jump into (1,1) sets X3 = R).
        repair["effects"] = [
            {
                "target": {"component": "reservoir", "attribute": "level"},
                "value": if_(up(other), num(CAPACITY), LEVEL),
            }
        ]
    return {
        "name": name,
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
                        "rate_expr": if_(at_max, num(lam_max), num(lam)),
                        "kind": "failure",
                        "monitored": True,
                    },
                    repair,
                ],
            }
        ],
    }


def reservoir(case: int) -> dict:
    """The reservoir: its level and whether it is empty, partly filled or
    full. The level follows the net flow, held at 0 while empty and at the
    capacity while full; reaching either bound is a watched transition."""
    flow = net_flow()
    if case == 4:
        flow = if_(both(up("U1"), up("U2")), num(PHI1_MAX + PHI2_MAX - PHI_NOM), flow)
    rate = if_(
        EMPTY,
        {"op": "max", "args": [flow, num(0.0)]},
        if_(FULL, {"op": "min", "args": [flow, num(0.0)]}, flow),
    )
    inst = {"distrib": "inst", "probs": []}
    return {
        "name": "reservoir",
        "attributes": [
            {
                "name": "level",
                "kind": "float",
                "init": {"kind": "float", "value": CAPACITY},
            },
            # Delivered production over nominal: 1 while the reservoir
            # covers the shortfall, the surviving unit's share once empty.
            {
                "name": "delivered",
                "kind": "float",
                "init": {"kind": "float", "value": 1.0},
            },
            # Total loss: both units down and the reservoir empty.
            {
                "name": "total_loss",
                "kind": "bool",
                "init": {"kind": "bool", "value": False},
            },
        ],
        "automata": [
            {
                "name": "store",
                "states": ["full", "partial", "empty"],
                "init": "full",
                "transitions": [
                    {
                        "name": "start_draining",
                        "source": "full",
                        "targets": ["partial"],
                        "guard": cmp("lt", flow, num(0.0)),
                        **inst,
                    },
                    # Empty only while draining: at zero with a filling
                    # flow (case 4) the reservoir is leaving `empty`.
                    {
                        "name": "run_dry",
                        "source": "partial",
                        "targets": ["empty"],
                        "guard": both(
                            cmp("le", LEVEL, num(0.0)), cmp("lt", flow, num(0.0))
                        ),
                        "distrib": "watched",
                    },
                    # Full only while filling: at the capacity with a
                    # draining flow the reservoir is leaving `full`.
                    {
                        "name": "fill_up",
                        "source": "partial",
                        "targets": ["full"],
                        "guard": both(
                            cmp("ge", LEVEL, num(CAPACITY)), cmp("gt", flow, num(0.0))
                        ),
                        "distrib": "watched",
                    },
                    # Out of empty when the flow turns positive (case 4) or
                    # when an instant refill restored the level (cases 1-3).
                    {
                        "name": "restart",
                        "source": "empty",
                        "targets": ["partial"],
                        "guard": either(
                            cmp("gt", flow, num(0.0)), cmp("gt", LEVEL, num(0.0))
                        ),
                        **inst,
                    },
                ],
            }
        ],
        "equations": [{"target": "level", "kind": "ode", "expr": rate}],
        "sensitive_functions": [
            {
                "name": "production",
                "effects": [
                    {
                        "target": {"component": "reservoir", "attribute": "delivered"},
                        "value": if_(
                            negate(EMPTY),
                            num(1.0),
                            if_(
                                up("U1"),
                                num(PHI1_MAX / PHI_NOM),
                                if_(up("U2"), num(PHI2_MAX / PHI_NOM), num(0.0)),
                            ),
                        ),
                    },
                    {
                        "target": {"component": "reservoir", "attribute": "total_loss"},
                        "value": both(EMPTY, negate(up("U1")), negate(up("U2"))),
                    },
                ],
            }
        ],
    }


def build(case: int) -> dict:
    """The model document of `case` (1 to 4)."""
    if case not in RATES:
        raise ValueError(f"case must be one of {CASES}, not {case!r}")
    body = {
        "name": f"gas_production_case{case}",
        "components": [unit("U1", "U2", case), unit("U2", "U1", case), reservoir(case)],
        "indicators": [
            # Time with the reservoir empty: 1 - A over the horizon.
            {
                "name": "empty",
                "target": "state",
                "component": "reservoir",
                "automaton": "store",
                "state": "empty",
            },
            {
                "name": "delivered",
                "target": "attribute",
                "attr": {"component": "reservoir", "attribute": "delivered"},
            },
            {
                "name": "total_loss",
                "target": "attribute",
                "attr": {"component": "reservoir", "attribute": "total_loss"},
            },
        ],
    }
    if case == 4:
        return body
    # The instant refill of cases 1-3 is an edge effect on the level, an
    # ODE target: a reset map, a construct the document declares.
    return {
        "raichu_model": {"format": 1, "requires": ["transition_effects"]},
        "model": body,
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for case in CASES:
        path = MODEL_DIR / f"gas_production_case{case}.json"
        path.write_text(json.dumps(build(case), indent=1) + "\n")
        print(path.name)


if __name__ == "__main__":
    main()


assert math.isclose(PHI1_MAX + PHI2_MAX - PHI_NOM, 1200.0)
