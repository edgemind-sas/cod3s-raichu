"""The APPRODYN feedwater test case as a component type library.

`model.py` writes each variant as a flat model whose components read one
another by name: the power profile reads the turbo-pumps' states, each
pump reads the others, and the trip logic reads everything. This file
writes the same plant as data documents built from reusable types, which
reach one another through ports only:

- `PowerCycle` publishes each set of power phases a neighbour reads (at
  power, one or two turbo-pumps required, the valves' takeover and
  service phases) with `std.state_out`;
- `ExtractionPump` (three instances) and `TurboPump` (two) publish whether
  they run or stand by, and read the counts and phases they need;
- `FeedwaterValve` (PD and GD) reads the phases it takes over and serves
  in, through the ports it is wired to;
- `Plant` holds the latched trip conditions, which the pumps write through
  a port (a common-cause loss, a turbo-pump refusing to start at 60 %),
  exactly as the flat model's transitions write `plant`;
- `TripLogic` reads the trip conditions and records the first trip.

The variants differ by instance parameters only: the ARE trip rule, the
common-cause failures (kept or dropped with `$when`), the perfect
turbo-pumps of the diagnostic variant. Every rate is computed here from
`model.py`'s constants by the same expression `model.py` uses, because
nothing computes at expansion. Run this file to rewrite `library/`.
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
    name = "approdyn_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _model()

#: Variants written as systems: the published ones and the diagnostic one.
SYSTEMS = {**M.VARIANTS, **M.DIAGNOSTIC}

#: Sets of power phases the neighbours read, by exported attribute.
PHASE_SETS = {
    "alive": [p for p in M.PHASES if p != "end"],
    "started": [p for p in M.PHASES if p not in ("end", "up0")],
    "at_power": list(M.AT_POWER),
    "two_tpa": list(M.TWO_TPA),
    "one_tpa": list(M.ONE_TPA),
    "forced60": ["forced60"],
    "forcing_failed": ["forcing_failed"],
    "pd_takes_over": ["start2"],
    "pd_in_service": ["up1", "switch15"],
    "gd_takes_over": ["switch15"],
    "gd_in_service": ["up2", "at60", "up3", "full", "forced60", "down"],
}


def param(name: str) -> dict:
    return {"$param": name}


def own_state(automaton: str, state: str) -> dict:
    return M.at("$self", automaton, state)


def any_in(port: str) -> dict:
    return {"op": "port_agg", "port": {"component": "$self", "port": port}, "agg": "any"}


def sum_in(port: str) -> dict:
    return {"op": "port_agg", "port": {"component": "$self", "port": port}, "agg": "sum"}


def in_ports(*names: str) -> list[dict]:
    return [{"name": name, "dir": "in"} for name in names]


def export(name: str, automaton: str, states: list[str], init: bool) -> dict:
    return {
        "fragment": "std.state_out",
        "with": {"name": name, "automaton": automaton, "states": states, "init": init},
    }


def number(name: str, description: str) -> dict:
    return {"name": name, "kind": "number", "description": description}


def boolean(name: str, description: str) -> dict:
    return {"name": name, "kind": "bool", "description": description}


def through(port: str, value: dict) -> dict:
    """An edge effect writing the attribute behind `port`."""
    return {"target": {"component": "$port", "attribute": port}, "value": value}


# ---- types --------------------------------------------------------------------


def power_cycle() -> dict:
    tpa_running = sum_in("tpa_running_in")
    calendar_done = any_in("plateau_over_in")
    lost_one = M.cmp("eq", tpa_running, M.num(1))
    transitions = [
        M.delay("reach_2pc", "up0", "start2", param("ramp_0_2")),
        M.inst("take_over_2pc", "start2", ["up1"], [],
               M.all_of(M.cmp("ge", tpa_running, M.num(1)), any_in("pd_open_in"))),
        M.delay("reach_15pc", "up1", "switch15", param("ramp_2_15")),
        M.inst("take_over_15pc", "switch15", ["up2"], [], any_in("gd_open_in")),
        M.delay("reach_60pc", "up2", "at60", param("ramp_15_60")),
        M.inst("second_tpa_running", "at60", ["up3"], [], M.cmp("eq", tpa_running, M.num(2))),
        M.delay("reach_100pc", "up3", "full", param("ramp_60_100")),
        M.inst("force_60pc_full", "full", ["forcing_failed", "forced60"], [param("forcing_pfd")], lost_one),
        M.inst("force_60pc_ramp", "up3", ["forcing_failed", "forced60"], [param("forcing_pfd")], lost_one),
        M.inst("back_to_full", "forced60", ["full"], [],
               M.all_of(M.cmp("eq", tpa_running, M.num(2)), M.negate(calendar_done))),
        M.inst("end_of_plateau", "full", ["down"], [], calendar_done),
        M.inst("end_of_plateau_forced", "forced60", ["down"], [], calendar_done),
        M.delay("shut_down", "down", "end", param("ramp_down")),
    ]
    return {
        "name": "PowerCycle",
        "description": (
            "The power profile of scenario 1: ramps, plateau, forced return to"
            " 60 % after the loss of a turbo-pump at power, ramp down. Its"
            " takeovers wait for a running turbo-pump and the valve that serves"
            " next. It publishes every set of phases a neighbour reads."
        ),
        "parameters": [
            number("ramp_0_2", "Rise from 0 to 2 %, in h."),
            number("ramp_2_15", "Rise from 2 to 15 %, in h."),
            number("ramp_15_60", "Rise from 15 to 60 %, in h."),
            number("ramp_60_100", "Rise from 60 to 100 %, in h."),
            number("ramp_down", "Ramp down, in h."),
            number("forcing_pfd", "Failure of the forced return to 60 %, per demand."),
        ],
        "uses": [
            export(name, "profile", states, "up0" in states)
            for name, states in PHASE_SETS.items()
        ],
        "body": {
            "ports": in_ports("tpa_running_in", "pd_open_in", "gd_open_in", "plateau_over_in"),
            "automata": [
                {"name": "profile", "states": list(M.PHASES), "init": "up0", "transitions": transitions}
            ],
        },
    }


def calendar() -> dict:
    return {
        "name": "Calendar",
        "description": "The end of the full-power plateau.",
        "parameters": [number("plateau_end", "Date the plateau ends, in h.")],
        "uses": [export("plateau_over", "plateau", ["over"], False)],
        "body": {
            "automata": [
                {
                    "name": "plateau",
                    "states": ["open", "over"],
                    "init": "open",
                    "transitions": [M.delay("plateau_ends", "open", "over", param("plateau_end"))],
                }
            ]
        },
    }


def steam_header() -> dict:
    return {
        "name": "SteamHeader",
        "description": "The VVP steam header: any failure trips the plant.",
        "parameters": [number("rate", "Failure rate, in 1/h.")],
        "uses": [export("failed", "state", ["failed"], False)],
        "body": {
            "automata": [
                {
                    "name": "state",
                    "states": ["ok", "failed"],
                    "init": "ok",
                    "transitions": [M.exp_t("fail", "ok", "failed", param("rate"), kind="failure")],
                }
            ]
        },
    }


def extraction_pump() -> dict:
    running_count = sum_in("running_in")
    demanded = M.all_of(
        M.cmp("lt", running_count, M.num(2)), M.negate(any_in("earlier_standby_in"))
    )
    repairs = list(M.CEX_REPAIR)
    return {
        "name": "ExtractionPump",
        "description": (
            "A CEX extraction pump, running or in standby: started on demand"
            " (it may refuse) when fewer than two run and no earlier pump stands"
            " by, repaired back to standby. A common-cause failure raises the"
            " plant's `cex_ccf` condition through `cex_ccf_in`."
        ),
        "parameters": [
            boolean("running", "Whether the pump runs at the start (else it stands by)."),
            number("fail_rate", "Running failure rate (72 h repair), in 1/h."),
            number("rupture_rate", "Running rupture rate (168 h repair), in 1/h."),
            number("pfd", "Refusal to start, per demand."),
            boolean("ccf", "Whether 5 % of the running failures are common-cause."),
            number("ccf_rate", "Common-cause failure rate, in 1/h."),
            *(number(f"repair_rate_{s}", f"Repair rate out of `{s}`, in 1/h.") for s in repairs),
        ],
        "uses": [
            export("running", "state", ["running"], param("running")),
            export("standby", "state", ["standby"], {"$choose": "running", "cases": {"true": False, "false": True}}),
        ],
        "body": {
            "ports": in_ports("running_in", "earlier_standby_in", "cex_ccf_in"),
            "automata": [
                {
                    "name": "state",
                    "states": ["running", "standby", *repairs],
                    "init": {"$choose": "running", "cases": {"true": "running", "false": "standby"}},
                    "transitions": [
                        M.exp_t("fail", "running", "rep72", param("fail_rate"), kind="failure"),
                        M.exp_t("rupture", "running", "rep168", param("rupture_rate"), kind="failure"),
                        M.inst("start", "standby", ["rep72", "running"], [param("pfd")], demanded),
                        {
                            "$when": "ccf",
                            "items": [
                                M.exp_t("common_cause", "running", "rep168", param("ccf_rate"),
                                        kind="failure", effects=[through("cex_ccf_in", M.true())])
                            ],
                        },
                        *(M.exp_t(f"repair_{s}", s, "standby", param(f"repair_rate_{s}"), kind="repair")
                          for s in repairs),
                    ],
                }
            ],
        },
    }


def turbo_pump() -> dict:
    running_count = sum_in("tpa_running_in")
    required = {
        "op": "if",
        "cond": any_in("two_tpa_in"),
        "then": M.num(2),
        "otherwise": {"op": "if", "cond": any_in("one_tpa_in"), "then": M.num(1), "otherwise": M.num(0)},
    }
    demanded = M.all_of(M.cmp("lt", running_count, required), M.negate(any_in("priority_standby_in")))
    refused_at_60 = through(
        "refused_at_60_in", M.all_of(any_in("forced60_in"), M.negate(own_state("state", "running")))
    )
    repairs = list(M.TPA_REPAIR)
    nominal = [
        M.inst("start", "standby", ["rep28", "rep2", "rep24", "running"],
               [param("p_ht"), param("p_t_short"), param("p_t_long")], demanded,
               effects=[refused_at_60]),
        M.exp_t("fail_long", "running", "rep24", param("fail_long_rate"), kind="failure"),
        M.exp_t("fail_short", "running", "rep2", param("fail_short_rate"), kind="failure"),
        M.exp_t("rupture", "running", "rep144", param("rupture_rate"), kind="failure"),
        {
            "$when": "ccf",
            "items": [
                M.exp_t("common_cause", "running", "rep24", param("ccf_rate"),
                        kind="failure", effects=[through("tpa_ccf_in", M.true())])
            ],
        },
        *(M.exp_t(f"repair_{s}", s, "standby", param(f"repair_rate_{s}"), kind="repair") for s in repairs),
    ]
    return {
        "name": "TurboPump",
        "description": (
            "A TPA turbo-pump, the series of its turbine and out-of-turbine"
            " parts: started on demand when fewer run than the power phase"
            " requires (it may refuse, into one of three repair states), failing"
            " while running, repaired back to standby. A refusal while forced at"
            " 60 % raises the plant's `tpa_refused_at_60` condition, a"
            " common-cause failure its `tpa_ccf` condition, both through ports."
            " `perfect` is the diagnostic turbo-pump that always starts and"
            " never fails."
        ),
        "parameters": [
            number("p_ht", "Start refused by the out-of-turbine part (28 h repair)."),
            number("p_t_short", "Start refused by the turbine, short repair (2 h)."),
            number("p_t_long", "Start refused by the turbine, long repair (24 h)."),
            number("fail_long_rate", "Running failure rate with a 24 h repair, in 1/h."),
            number("fail_short_rate", "Running failure rate with a 2 h repair, in 1/h."),
            number("rupture_rate", "Running rupture rate (144 h repair), in 1/h."),
            boolean("ccf", "Whether 5 % of the running failures are common-cause."),
            number("ccf_rate", "Common-cause failure rate, in 1/h."),
            boolean("perfect", "The diagnostic turbo-pump: always starts, never fails."),
            *(number(f"repair_rate_{s}", f"Repair rate out of `{s}`, in 1/h.") for s in repairs),
        ],
        "uses": [
            export("running", "state", ["running"], False),
            export("standby", "state", ["standby"], True),
        ],
        "body": {
            "ports": in_ports(
                "tpa_running_in", "two_tpa_in", "one_tpa_in", "forced60_in",
                "priority_standby_in", "refused_at_60_in", "tpa_ccf_in",
            ),
            "automata": [
                {
                    "name": "state",
                    "states": ["standby", "running", *repairs],
                    "init": "standby",
                    "transitions": [
                        {"$when": "perfect", "is": False, "items": nominal},
                        {"$when": "perfect", "items": [M.inst("start", "standby", ["running"], [], demanded)]},
                    ],
                }
            ],
        },
    }


def feedwater_valve() -> dict:
    return {
        "name": "FeedwaterValve",
        "description": (
            "An ARE regulating valve: opens when the power phase it takes over"
            " in begins (it may refuse, and is then repaired), and while in"
            " service fails into a trip at a rate set by the trip rule."
        ),
        "parameters": [
            number("p_refuse", "Refusal to open, per demand."),
            number("repair_rate", "Repair rate after a refusal, in 1/h."),
            number("trip_rate", "Rate of the failures in service that trip the plant, in 1/h."),
        ],
        "uses": [
            export("open", "valve", ["open"], False),
            export("tripped", "valve", ["tripped"], False),
        ],
        "body": {
            "ports": in_ports("takes_over_in", "in_service_in"),
            "automata": [
                {
                    "name": "valve",
                    "states": ["closed", "open", "refused", "tripped"],
                    "init": "closed",
                    "transitions": [
                        M.inst("open", "closed", ["refused", "open"], [param("p_refuse")], any_in("takes_over_in")),
                        M.exp_t("repair", "refused", "closed", param("repair_rate"), kind="repair"),
                        M.exp_t("fail_in_service", "open", "tripped", param("trip_rate"),
                                kind="failure", guard=any_in("in_service_in")),
                    ],
                }
            ],
        },
    }


FLAGS = ("cex_ccf", "tpa_ccf", "tpa_refused_at_60")


def plant() -> dict:
    return {
        "name": "Plant",
        "description": (
            "The latched trip conditions, written by the pumps through their"
            " ports and read by the trip logic."
        ),
        "body": {
            "attributes": [
                {"name": f, "kind": "bool", "init": {"kind": "bool", "value": False}} for f in FLAGS
            ],
            "ports": [{"name": f"{f}_out", "dir": "out", "attr": f} for f in FLAGS],
        },
    }


def trip_logic() -> dict:
    running = sum_in("cex_running_in")
    standby = sum_in("cex_standby_in")
    tpa_running = sum_in("tpa_running_in")
    tpa_standby = sum_in("tpa_standby_in")
    alive, started = any_in("alive_in"), any_in("started_in")
    cex_lost = M.any_of(
        any_in("cex_ccf_in"),
        M.all_of(M.cmp("lt", running, M.num(2)), M.cmp("eq", standby, M.num(0))),
    )
    tpa_lost = M.any_of(
        any_in("tpa_ccf_in"),
        any_in("tpa_refused_at_60_in"),
        any_in("forcing_failed_in"),
        M.all_of(any_in("at_power_in"), M.cmp("eq", tpa_running, M.num(0)), M.cmp("eq", tpa_standby, M.num(0))),
    )
    guards = {
        "vvp": M.all_of(alive, any_in("vvp_failed_in")),
        "cex": M.all_of(alive, cex_lost),
        "tpa": M.all_of(started, tpa_lost),
        "are": M.all_of(started, any_in("valve_tripped_in")),
    }
    return {
        "name": "TripLogic",
        "description": "The first trip and its cause; each cause is a target, so a history stops at the first trip.",
        "body": {
            "ports": in_ports(
                "alive_in", "started_in", "at_power_in", "forcing_failed_in", "vvp_failed_in",
                "cex_running_in", "cex_standby_in", "tpa_running_in", "tpa_standby_in",
                "valve_tripped_in", *(f"{f}_in" for f in FLAGS),
            ),
            "automata": [
                {
                    "name": "cause",
                    "states": ["none", *M.CAUSES],
                    "init": "none",
                    "transitions": [M.inst(f"trip_{c}", "none", [c], [], g) for c, g in guards.items()],
                }
            ],
        },
    }


def library() -> dict:
    return {
        "raichu_library": {"format": 1},
        "library": {
            "name": "approdyn",
            "version": "1.0.0",
            "imports": ["std"],
            "description": (
                "The APPRODYN feedwater test case (Aubry et al., 2012), discrete"
                " part, as a component type library."
            ),
            "types": [
                power_cycle(), calendar(), steam_header(), extraction_pump(),
                turbo_pump(), feedwater_valve(), plant(), trip_logic(),
            ],
        },
    }


# ---- systems ------------------------------------------------------------------


def system(variant: str) -> dict:
    rule, ccf = SYSTEMS[variant]
    perfect = variant in M.DIAGNOSTIC
    data = M.TpaData(ccf=ccf)
    share = M.CCF_SHARE if ccf else 0.0
    independent = (1 - share) * M.CEX_RATE

    def cex(name: str, running: bool) -> dict:
        return {
            "name": name,
            "type": "ExtractionPump",
            "parameters": {
                "running": running,
                "fail_rate": independent * 0.999,
                "rupture_rate": independent * 0.001,
                "pfd": M.CEX_PFD,
                "ccf": ccf,
                "ccf_rate": M.CCF_SHARE * M.CEX_RATE,
                **{f"repair_rate_{s}": 1 / t for s, t in M.CEX_REPAIR.items()},
            },
        }

    p_ht = data.out_of_turbine_pfd
    p_t = (1 - data.out_of_turbine_pfd) * data.turbine_pfd
    lam_ht, lam_t = data.out_of_turbine_rate, data.turbine_rate
    ind = 1 - share

    def tpa(name: str) -> dict:
        return {
            "name": name,
            "type": "TurboPump",
            "parameters": {
                "p_ht": p_ht,
                "p_t_short": p_t * 0.99,
                "p_t_long": p_t * 0.01,
                "fail_long_rate": ind * (lam_ht * 0.999 + lam_t * 0.75),
                "fail_short_rate": ind * lam_t * 0.25,
                "rupture_rate": ind * lam_ht * 0.001,
                "ccf": ccf,
                "ccf_rate": M.CCF_SHARE * (lam_ht + lam_t),
                "perfect": perfect,
                **{f"repair_rate_{s}": 1 / (t * data.repair_factor) for s, t in M.TPA_REPAIR.items()},
            },
        }

    def valve(name: str) -> dict:
        return {
            "name": name,
            "type": "FeedwaterValve",
            "parameters": {
                "p_refuse": M.ARE_PFD * M.ARE_REFUSE_OPEN,
                "repair_rate": 1 / M.ARE_REPAIR,
                "trip_rate": M.ARE_RATE * M.ARE_TRIP[rule],
            },
        }

    def wire(source: str, *targets: str) -> list[dict]:
        return [{"from": source, "to": target} for target in targets]

    cexes, tpas = ("CEX1", "CEX2", "CEX3"), ("TPA1", "TPA2")
    connections = [
        *wire("TPA1.running_out", "power.tpa_running_in"),
        *wire("TPA2.running_out", "power.tpa_running_in"),
        *wire("PD.open_out", "power.pd_open_in"),
        *wire("GD.open_out", "power.gd_open_in"),
        *wire("calendar.plateau_over_out", "power.plateau_over_in"),
    ]
    for pump in cexes:
        connections += wire(f"{pump}.running_out", *(f"{p}.running_in" for p in cexes))
    connections += wire("CEX1.standby_out", "CEX2.earlier_standby_in", "CEX3.earlier_standby_in")
    connections += wire("CEX2.standby_out", "CEX3.earlier_standby_in")
    connections += wire("plant.cex_ccf_out", *(f"{p}.cex_ccf_in" for p in cexes))
    for pump in tpas:
        connections += wire(f"{pump}.running_out", *(f"{p}.tpa_running_in" for p in tpas))
    for export_name in ("two_tpa", "one_tpa", "forced60"):
        connections += wire(f"power.{export_name}_out", *(f"{p}.{export_name}_in" for p in tpas))
    connections += wire("TPA1.standby_out", "TPA2.priority_standby_in")
    connections += wire("plant.tpa_refused_at_60_out", *(f"{p}.refused_at_60_in" for p in tpas))
    connections += wire("plant.tpa_ccf_out", *(f"{p}.tpa_ccf_in" for p in tpas))
    connections += wire("power.pd_takes_over_out", "PD.takes_over_in")
    connections += wire("power.pd_in_service_out", "PD.in_service_in")
    connections += wire("power.gd_takes_over_out", "GD.takes_over_in")
    connections += wire("power.gd_in_service_out", "GD.in_service_in")
    for export_name in ("alive", "started", "at_power", "forcing_failed"):
        connections += wire(f"power.{export_name}_out", f"trip.{export_name}_in")
    connections += wire("VVP.failed_out", "trip.vvp_failed_in")
    for pump in cexes:
        connections += wire(f"{pump}.running_out", "trip.cex_running_in")
        connections += wire(f"{pump}.standby_out", "trip.cex_standby_in")
    for pump in tpas:
        connections += wire(f"{pump}.running_out", "trip.tpa_running_in")
        connections += wire(f"{pump}.standby_out", "trip.tpa_standby_in")
    connections += wire("PD.tripped_out", "trip.valve_tripped_in")
    connections += wire("GD.tripped_out", "trip.valve_tripped_in")
    for flag in FLAGS:
        connections += wire(f"plant.{flag}_out", f"trip.{flag}_in")
    flat = M.build(rule, ccf, variant, perfect)["model"]
    return {
        "raichu_system": {"format": 1},
        "system": {
            "name": variant,
            "library": {"name": "approdyn", "version": "1.0.0"},
            "instances": [
                {
                    "name": "power",
                    "type": "PowerCycle",
                    "parameters": {
                        "ramp_0_2": M.RAMP_0_2,
                        "ramp_2_15": M.RAMP_2_15,
                        "ramp_15_60": M.RAMP_15_60,
                        "ramp_60_100": M.RAMP_60_100,
                        "ramp_down": M.RAMP_DOWN,
                        "forcing_pfd": data.forcing_pfd,
                    },
                },
                {"name": "calendar", "type": "Calendar", "parameters": {"plateau_end": M.PLATEAU_END}},
                {"name": "VVP", "type": "SteamHeader", "parameters": {"rate": M.VVP_RATE}},
                cex("CEX1", True),
                cex("CEX2", True),
                cex("CEX3", False),
                tpa("TPA1"),
                tpa("TPA2"),
                valve("PD"),
                valve("GD"),
                {"name": "plant", "type": "Plant"},
                {"name": "trip", "type": "TripLogic"},
            ],
            "connections": connections,
            "targets": flat["targets"],
            "indicators": flat["indicators"],
        },
    }


def documents() -> dict[str, dict]:
    """Every document of `library/`, by file name."""
    out = {"library.json": library()}
    for variant in SYSTEMS:
        out[f"system_{variant}.json"] = system(variant)
    return out


def main() -> None:
    LIBRARY_DIR.mkdir(exist_ok=True)
    for name, document in documents().items():
        (LIBRARY_DIR / name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        print(name)


if __name__ == "__main__":
    main()
