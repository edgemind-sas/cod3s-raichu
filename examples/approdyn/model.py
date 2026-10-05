"""The APPRODYN feedwater test case, discrete part, as a RAICHU model.

The case is the feedwater supply of one steam generator of a 900 MW
pressurised-water reactor, defined by the APPRODYN project (Aubry et al.,
final report, 2012, HAL hal-00740181, chapter 4). Its data and models are
"representative but not real", and the authors restrict them to use as a
test case: that is the only use made of them here.

What is modelled (report chapter 4):

- the VVP steam header: any failure trips the plant (leak or rupture,
  Table 4.5);
- three CEX extraction pumps, two running and one in standby (2-out-of-3):
  a running pump fails, the standby pump is started (it may refuse), the
  failed one is repaired and returns to standby; losing two pumps trips the
  plant, and 5 % of the running failures are common-cause losses of two
  pumps (Tables 4.4, 4.6, p. 41-44);
- two TPA turbo-pumps, each the series of a turbine part and an
  out-of-turbine part (Tables 4.4, 4.8, 4.10): one runs below 60 % power,
  both above; losing one above 60 % forces the power back to 60 %, a
  forcing that fails with probability 1e-3 (p. 49-50); a TPA lost with the
  other unavailable, or refusing to restart at 60 %, trips the plant; 5 %
  of the running failures are common-cause losses of both;
- two ARE regulating valves, the small-flow PD in service from 2 to 15 %
  power and the large-flow GD above (p. 54-55); each may refuse to open
  when it takes over, which holds the ramp until it is repaired.

What is NOT modelled: the steam generator level and its control (the
report gives neither the controller gains, nor the units, nor the steam
flow as a function of power), the sensors, the external perturbations,
repair failures, and the demands of the ramp down. See the documentation
page for the data choice made on each conflict of the report.

Scenario 1 (p. 30-31): rise from 0 to 2 % in 10 h, to 10 % in 10 h, to
100 % in 4 h; 18 months (12 960 h) at full power; 24 h ramp down. The
first trip ends the history.

Two trip rules for the ARE valves, `RULES`:

- "pdmp": any failure of the valve in service trips the plant, the
  counting rule of the published 4000-history Monte-Carlo (report p. 9:
  ARE failures forcing a drop to 2 % power are counted as trips);
- "tables": a failure trips the plant with the probability its mode and
  detection rate give (Tables 4.11 and 4.12): rupture 0.1 % x 1, internal
  leak 11 % x 0.1, external leak 1 % x 0.5, spurious manoeuvre 25 % x 0,
  blocking 63 % x 0.1 (its online repair fails 10 % of the time).

Common-cause failures (5 % of the running failures of the CEX pumps and of
the TPA, report p. 44 and p. 49) are an option, `ccf`: as specified, the
TPA common cause alone trips about six histories in ten over 18 months,
far above what the published Monte-Carlo attributes to the TPA (1.2 %);
the documentation page compares both readings.

Usage: python model.py  (writes model/<variant>.json, see VARIANTS)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

#: Failure rates per hour and failure probabilities on demand (Table 4.4).
VVP_RATE = 2.17e-5
CEX_RATE, CEX_PFD = 4.35e-5, 1.95e-3
TPA_HT_RATE, TPA_HT_PFD = 1.46e-4, 5.45e-4
TPA_T_RATE, TPA_T_PFD = 5.9e-4, 3.90e-3
ARE_RATE, ARE_PFD = 3.53e-5, 5.85e-3
#: Share of the running failures that are common-cause (p. 44, p. 49).
CCF_SHARE = 0.05
#: Failure of the forcing back to 60 % power, per demand (p. 50).
FORCING_PFD = 1e-3
#: Share of an ARE valve's demand failures that are a refusal to open
#: (Table 4.11); a refusal to close does not stop the takeover.
ARE_REFUSE_OPEN = 0.72
#: Probability that an ARE failure trips the plant, per rule (see above).
ARE_TRIP = {
    "pdmp": 1.0,
    "tables": 0.001 * 1.0 + 0.11 * 0.1 + 0.01 * 0.5 + 0.25 * 0.0 + 0.63 * 0.1,
}
RULES = tuple(ARE_TRIP)
#: Published variants: name -> (ARE trip rule, common-cause failures).
VARIANTS = {
    "approdyn_pdmp": ("pdmp", True),
    "approdyn_pdmp_noccf": ("pdmp", False),
    "approdyn_tables": ("tables", True),
}
#: Diagnostic variant: the published counting rule with perfect TPA, to
#: test whether the published Monte-Carlo's causes are those of a model
#: whose TPA hardly ever trip the plant.
DIAGNOSTIC = {"approdyn_pdmp_perfect_tpa": ("pdmp", True)}

#: Mean repair times in hours (Tables 4.6, 4.8, 4.10, 4.11).
CEX_REPAIR = {"rep72": 72.0, "rep168": 168.0}
TPA_REPAIR = {"rep2": 2.0, "rep24": 24.0, "rep28": 28.0, "rep144": 144.0}
ARE_REPAIR = 12.0

#: Scenario 1 (p. 30-31), in hours. The rise crosses 2 % at 10 h, 15 % at
#: 20 + 5 / 22.5 h, 60 % 2 h later and reaches 100 % 1.78 h later.
RAMP_0_2 = 10.0
RAMP_2_15 = 10.0 + 5.0 / 22.5
RAMP_15_60 = 45.0 / 22.5
RAMP_60_100 = 40.0 / 22.5
PLATEAU_END = 24.0 + 12_960.0
RAMP_DOWN = 24.0
#: Horizon of a history: the whole cycle.
HORIZON = PLATEAU_END + RAMP_DOWN

CAUSES = ("vvp", "cex", "tpa", "are")


# ---- expression helpers -----------------------------------------------------


def num(x: float) -> dict:
    return {"op": "const", "value": {"kind": "float", "value": float(x)}}


def false() -> dict:
    return {"op": "const", "value": {"kind": "bool", "value": False}}


def true() -> dict:
    return {"op": "const", "value": {"kind": "bool", "value": True}}


def attr(component: str, name: str) -> dict:
    return {"op": "attr", "attr": {"component": component, "attribute": name}}


def at(component: str, automaton: str, state: str) -> dict:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


def all_of(*args: dict) -> dict:
    return {"op": "bool", "bool_op": "and", "args": list(args)}


def any_of(*args: dict) -> dict:
    return {"op": "bool", "bool_op": "or", "args": list(args)}


def negate(arg: dict) -> dict:
    return {"op": "bool", "bool_op": "not", "args": [arg]}


def cmp(op: str, lhs: dict, rhs: dict) -> dict:
    return {"op": "cmp", "cmp": op, "lhs": lhs, "rhs": rhs}


def count(*conds: dict) -> dict:
    return {
        "op": "add",
        "args": [
            {"op": "if", "cond": c, "then": num(1), "otherwise": num(0)} for c in conds
        ],
    }


def phase(*names: str) -> dict:
    return any_of(*(at("power", "profile", n) for n in names))


def flag_effect(name: str, value: dict) -> dict:
    return {"target": {"component": "plant", "attribute": name}, "value": value}


def exp_t(name: str, source: str, target: str, rate: float, **extra) -> dict:
    return {
        "name": name,
        "source": source,
        "targets": [target],
        "distrib": "exp",
        "rate": rate,
        "monitored": True,
        **extra,
    }


def inst(
    name: str, source: str, targets: list[str], probs: list[float], guard: dict, **extra
) -> dict:
    return {
        "name": name,
        "source": source,
        "targets": targets,
        "distrib": "inst",
        "probs": probs,
        "guard": guard,
        "monitored": True,
        **extra,
    }


def delay(name: str, source: str, target: str, time: float) -> dict:
    return {
        "name": name,
        "source": source,
        "targets": [target],
        "distrib": "delay",
        "time": time,
    }


# ---- the power profile ------------------------------------------------------

#: Power phases, in the order of the cycle. `start2` waits for a running TPA
#: and an open PD valve at 2 %; `switch15` waits for the GD valve to open at
#: 15 %; `at60` waits for the second TPA; `forced60` is the forced return to
#: 60 % after the loss of a TPA at full power.
PHASES = (
    "up0",
    "start2",
    "up1",
    "switch15",
    "up2",
    "at60",
    "up3",
    "full",
    "forced60",
    "forcing_failed",
    "down",
    "end",
)
#: Phases at power, where a missing feedwater train trips the plant.
AT_POWER = ("up1", "switch15", "up2", "at60", "up3", "full", "forced60", "down")
#: TPA required per phase.
TWO_TPA = ("at60", "up3", "full", "forced60")
ONE_TPA = ("start2", "up1", "switch15", "up2", "down")


def tpa_running() -> dict:
    return count(at("TPA1", "state", "running"), at("TPA2", "state", "running"))


def tpa_standby() -> dict:
    return count(at("TPA1", "state", "standby"), at("TPA2", "state", "standby"))


def tpa_required() -> dict:
    return {
        "op": "if",
        "cond": phase(*TWO_TPA),
        "then": num(2),
        "otherwise": {
            "op": "if",
            "cond": phase(*ONE_TPA),
            "then": num(1),
            "otherwise": num(0),
        },
    }


def power() -> dict:
    calendar_done = at("calendar", "plateau", "over")
    lost_one = cmp("eq", tpa_running(), num(1))
    transitions = [
        delay("reach_2pc", "up0", "start2", RAMP_0_2),
        inst(
            "take_over_2pc",
            "start2",
            ["up1"],
            [],
            all_of(cmp("ge", tpa_running(), num(1)), at("PD", "valve", "open")),
        ),
        delay("reach_15pc", "up1", "switch15", RAMP_2_15),
        inst("take_over_15pc", "switch15", ["up2"], [], at("GD", "valve", "open")),
        delay("reach_60pc", "up2", "at60", RAMP_15_60),
        inst(
            "second_tpa_running", "at60", ["up3"], [], cmp("eq", tpa_running(), num(2))
        ),
        delay("reach_100pc", "up3", "full", RAMP_60_100),
        # A TPA lost at full power or on the last ramp: forced back to 60 %,
        # which fails with probability FORCING_PFD (p. 50).
        inst(
            "force_60pc_full",
            "full",
            ["forcing_failed", "forced60"],
            [FORCING_PFD],
            lost_one,
        ),
        inst(
            "force_60pc_ramp",
            "up3",
            ["forcing_failed", "forced60"],
            [FORCING_PFD],
            lost_one,
        ),
        inst(
            "back_to_full",
            "forced60",
            ["full"],
            [],
            all_of(cmp("eq", tpa_running(), num(2)), negate(calendar_done)),
        ),
        inst("end_of_plateau", "full", ["down"], [], calendar_done),
        inst("end_of_plateau_forced", "forced60", ["down"], [], calendar_done),
        delay("shut_down", "down", "end", RAMP_DOWN),
    ]
    return {
        "name": "power",
        "automata": [
            {
                "name": "profile",
                "states": list(PHASES),
                "init": "up0",
                "transitions": transitions,
            }
        ],
    }


def calendar() -> dict:
    return {
        "name": "calendar",
        "automata": [
            {
                "name": "plateau",
                "states": ["open", "over"],
                "init": "open",
                "transitions": [delay("plateau_ends", "open", "over", PLATEAU_END)],
            }
        ],
    }


# ---- components -------------------------------------------------------------


def vvp() -> dict:
    return {
        "name": "VVP",
        "automata": [
            {
                "name": "state",
                "states": ["ok", "failed"],
                "init": "ok",
                "transitions": [
                    exp_t("fail", "ok", "failed", VVP_RATE, kind="failure")
                ],
            }
        ],
    }


def cex_pump(name: str, running: bool, index: int, ccf: bool) -> dict:
    """A CEX pump: running or standby, repaired back to standby."""
    pumps = ("CEX1", "CEX2", "CEX3")
    running_count = count(*(at(p, "state", "running") for p in pumps))
    earlier_standby = [at(p, "state", "standby") for p in pumps[:index]]
    demanded = all_of(
        cmp("lt", running_count, num(2)), *(negate(c) for c in earlier_standby)
    )
    share = CCF_SHARE if ccf else 0.0
    independent = (1 - share) * CEX_RATE
    transitions = [
        exp_t("fail", "running", "rep72", independent * 0.999, kind="failure"),
        exp_t("rupture", "running", "rep168", independent * 0.001, kind="failure"),
        inst("start", "standby", ["rep72", "running"], [CEX_PFD], demanded),
    ]
    if ccf:
        # 5 % of the running failures are common-cause: two pumps lost.
        transitions.append(
            exp_t(
                "common_cause",
                "running",
                "rep168",
                CCF_SHARE * CEX_RATE,
                kind="failure",
                effects=[flag_effect("cex_ccf", true())],
            )
        )
    transitions += [
        exp_t(f"repair_{s}", s, "standby", 1 / t, kind="repair")
        for s, t in CEX_REPAIR.items()
    ]
    return {
        "name": name,
        "automata": [
            {
                "name": "state",
                "states": ["running", "standby", *CEX_REPAIR],
                "init": "running" if running else "standby",
                "transitions": transitions,
            }
        ],
    }


def tpa(name: str, other: str, ccf: bool, perfect: bool = False) -> dict:
    """A TPA turbo-pump: the series of its turbine part (T) and its
    out-of-turbine part (HT), each with its failure modes and repair times
    (Tables 4.8 and 4.10)."""
    demanded = cmp("lt", tpa_running(), tpa_required())
    if name == "TPA2":
        # TPA1 is started first; TPA2 when TPA1 is not in standby.
        demanded = all_of(demanded, negate(at("TPA1", "state", "standby")))
    # Start: HT refuses (repair 28 h), else T refuses (2 h for 99 %, 24 h).
    p_ht = TPA_HT_PFD
    p_t = (1 - TPA_HT_PFD) * TPA_T_PFD
    # A refusal while forced at 60 % trips the plant (p. 50). The effect is
    # evaluated after the draw, on every outcome: only a refusal (the pump
    # not running) raises the flag.
    refused_at_60 = [
        flag_effect(
            "tpa_refused_at_60",
            all_of(phase("forced60"), negate(at(name, "state", "running"))),
        )
    ]
    start = inst(
        "start",
        "standby",
        ["rep28", "rep2", "rep24", "running"],
        [p_ht, p_t * 0.99, p_t * 0.01],
        demanded,
        effects=refused_at_60,
    )
    ind = 1 - (CCF_SHARE if ccf else 0.0)
    transitions = [
        start,
        # HT running failure 99.9 % (24 h), rupture 0.1 % (144 h); T running
        # failure short 25 % (2 h), long 75 % (24 h).
        exp_t(
            "fail_long",
            "running",
            "rep24",
            ind * (TPA_HT_RATE * 0.999 + TPA_T_RATE * 0.75),
            kind="failure",
        ),
        exp_t("fail_short", "running", "rep2", ind * TPA_T_RATE * 0.25, kind="failure"),
        exp_t(
            "rupture", "running", "rep144", ind * TPA_HT_RATE * 0.001, kind="failure"
        ),
    ]
    if ccf:
        # 5 % of the running failures (modes I and II of the out-of-turbine
        # part, mode I of the turbine: p. 49) are common-cause, both TPA lost.
        transitions.append(
            exp_t(
                "common_cause",
                "running",
                "rep24",
                CCF_SHARE * (TPA_HT_RATE + TPA_T_RATE),
                kind="failure",
                effects=[flag_effect("tpa_ccf", true())],
            )
        )
    transitions += [
        exp_t(f"repair_{s}", s, "standby", 1 / t, kind="repair")
        for s, t in TPA_REPAIR.items()
    ]
    if perfect:
        # Diagnostic variant: the TPA always start and never fail.
        transitions = [inst("start", "standby", ["running"], [], demanded)]
    return {
        "name": name,
        "automata": [
            {
                "name": "state",
                "states": ["standby", "running", *TPA_REPAIR],
                "init": "standby",
                "transitions": transitions,
            }
        ],
    }


def are_valve(name: str, rule: str) -> dict:
    """An ARE valve: closed, open (in service) or under repair after a
    refusal to open. Failures in service trip the plant per `rule`."""
    if name == "PD":
        takes_over = phase("start2")
        in_service = phase("up1", "switch15")
    else:
        takes_over = phase("switch15")
        in_service = phase("up2", "at60", "up3", "full", "forced60", "down")
    p_refuse = ARE_PFD * ARE_REFUSE_OPEN
    return {
        "name": name,
        "automata": [
            {
                "name": "valve",
                "states": ["closed", "open", "refused", "tripped"],
                "init": "closed",
                "transitions": [
                    inst("open", "closed", ["refused", "open"], [p_refuse], takes_over),
                    exp_t("repair", "refused", "closed", 1 / ARE_REPAIR, kind="repair"),
                    exp_t(
                        "fail_in_service",
                        "open",
                        "tripped",
                        ARE_RATE * ARE_TRIP[rule],
                        kind="failure",
                        guard=in_service,
                    ),
                ],
            }
        ],
    }


def plant() -> dict:
    """Latched trip conditions written by the transitions that cause them."""
    flags = ("cex_ccf", "tpa_ccf", "tpa_refused_at_60")
    return {
        "name": "plant",
        "attributes": [
            {"name": f, "kind": "bool", "init": {"kind": "bool", "value": False}}
            for f in flags
        ],
    }


def trips() -> dict:
    """The first trip and its cause; each cause is a target, so a history
    stops at the first trip."""
    not_ended = negate(phase("end", "up0"))
    alive = negate(phase("end"))
    cex_pumps = ("CEX1", "CEX2", "CEX3")
    cex_lost = any_of(
        attr("plant", "cex_ccf"),
        all_of(
            cmp("lt", count(*(at(p, "state", "running") for p in cex_pumps)), num(2)),
            cmp("eq", count(*(at(p, "state", "standby") for p in cex_pumps)), num(0)),
        ),
    )
    tpa_lost = any_of(
        attr("plant", "tpa_ccf"),
        attr("plant", "tpa_refused_at_60"),
        phase("forcing_failed"),
        all_of(
            phase(*AT_POWER),
            cmp("eq", tpa_running(), num(0)),
            cmp("eq", tpa_standby(), num(0)),
        ),
    )
    are_lost = any_of(at("PD", "valve", "tripped"), at("GD", "valve", "tripped"))
    guards = {
        "vvp": all_of(alive, at("VVP", "state", "failed")),
        "cex": all_of(alive, cex_lost),
        "tpa": all_of(not_ended, tpa_lost),
        "are": all_of(not_ended, are_lost),
    }
    return {
        "name": "trip",
        "automata": [
            {
                "name": "cause",
                "states": ["none", *CAUSES],
                "init": "none",
                "transitions": [
                    inst(f"trip_{c}", "none", [c], [], g) for c, g in guards.items()
                ],
            }
        ],
    }


def build(
    rule: str = "pdmp",
    ccf: bool = True,
    name: str | None = None,
    perfect_tpa: bool = False,
) -> dict:
    """The model document for the ARE trip rule `rule`, with or without the
    common-cause failures."""
    if rule not in ARE_TRIP:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    body = {
        "name": name or f"approdyn_{rule}" + ("" if ccf else "_noccf"),
        "components": [
            power(),
            calendar(),
            vvp(),
            cex_pump("CEX1", True, 0, ccf),
            cex_pump("CEX2", True, 1, ccf),
            cex_pump("CEX3", False, 2, ccf),
            tpa("TPA1", "TPA2", ccf, perfect_tpa),
            tpa("TPA2", "TPA1", ccf, perfect_tpa),
            are_valve("PD", rule),
            are_valve("GD", rule),
            plant(),
            trips(),
        ],
        "targets": [
            {"name": f"trip_{c}", "component": "trip", "automaton": "cause", "state": c}
            for c in CAUSES
        ],
        "indicators": [
            {
                "name": f"trip_{c}",
                "target": "state",
                "component": "trip",
                "automaton": "cause",
                "state": c,
            }
            for c in CAUSES
        ],
    }
    return {
        "raichu_model": {"format": 1, "requires": ["transition_effects"]},
        "model": body,
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for name, (rule, ccf) in VARIANTS.items():
        path = MODEL_DIR / f"{name}.json"
        path.write_text(json.dumps(build(rule, ccf, name), indent=1) + "\n")
        print(path.name)
    for name, (rule, ccf) in DIAGNOSTIC.items():
        path = MODEL_DIR / f"{name}.json"
        path.write_text(
            json.dumps(build(rule, ccf, name, perfect_tpa=True), indent=1) + "\n"
        )
        print(path.name)


if __name__ == "__main__":
    main()
