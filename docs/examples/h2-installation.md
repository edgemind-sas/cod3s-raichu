# Hydrogen installation

A small solar hydrogen installation: photovoltaic panels power an
electrolyser, a battery carries it into the night, a storage buffers the
hydrogen, and two compressors deliver it to a user. The question is the
one a safety study asks of it: **how often does the user go without
hydrogen, and through which chains of failures?**

The installation is generic and its figures are illustrative. It is sized
so that it produces more than it delivers when nothing fails, which makes
every loss of supply the consequence of failures.

## Installation

![Hydrogen installation: PV panels feed the electrolyser and a battery; the electrolyser fills a storage; two compressors deliver to the user](../assets/schematics/h2-installation-light.svg#only-light){ .figure }
![Hydrogen installation: PV panels feed the electrolyser and a battery; the electrolyser fills a storage; two compressors deliver to the user](../assets/schematics/h2-installation-dark.svg#only-dark){ .figure }

| Component | Role | Name in the model |
|---|---|---|
| PV panels and inverter | electricity, following the sun | `PV` |
| Battery | stores the daytime surplus, powers the electrolyser at night | `BATT` |
| Electrolyser | turns electricity into hydrogen | `ELEC` |
| Storage | buffers hydrogen between production and use | `STORE` |
| Compressors A and B | deliver hydrogen to the user, redundant | `COMP_A`, `COMP_B` |
| User | draws hydrogen at a constant rate | `USER` |

## Hypotheses

Time is in hours, electricity in kW and kWh, hydrogen in kg.

| Quantity | Value |
|---|---|
| PV power | 150 · max(0, sin(2π (t − 6) / 24)) kW: zero at night, peak at noon |
| Battery capacity | 300 kWh, 150 kWh at t = 0 |
| Electrolyser | draws 50 kW and makes 1 kg/h while running |
| Storage capacity | 150 kg, 120 kg at t = 0 |
| User demand | 0.5 kg/h, i.e. 12 kg per day |

| Control rule | Condition |
|---|---|
| Electrolyser starts | storage below 140 kg, and PV ≥ 60 kW or battery ≥ 100 kWh |
| Electrolyser stops | storage at 150 kg, or PV < 50 kW with battery ≤ 10 kWh |
| User loses supply | storage ≤ 1 kg, or both compressors failed |
| User is supplied again | storage ≥ 10 kg and one compressor working |

The start and stop thresholds form **bands** (60 and 50 kW, 100 and 10 kWh,
140 and 150 kg, 1 and 10 kg): a rule entered at one value and left at
another cannot switch back and forth on the quantity it moves itself.

| Component | Failure | Repair |
|---|---|---|
| PV inverter | exponential, MTTF 6 000 h | exponential, MTTR 120 h |
| Battery | exponential, MTTF 5 000 h | exponential, MTTR 168 h |
| Electrolyser | exponential, MTTF 8 000 h | exponential, MTTR 240 h (spare parts) |
| Compressor A, B | exponential, MTTF 3 000 h each | exponential, MTTR 48 h each |

Deliberate simplifications: water is always available; the battery and the
electrolyser have no efficiency losses beyond the 50 kWh per kg; the PV
curve is the same every day (no weather, no seasons); the control reads
the true quantities (no sensor failure); the storage and the user never
fail.

## Model

The model is written in the core schema, a few helpers keeping the
expressions short. Every component is a box with ports, and every arrow of
the schematic is a connection.

```python
import math
import pyraichu

P_PEAK, BATT_CAP, BATT_INIT = 150.0, 300.0, 150.0
ELEC_LOAD, ELEC_H2 = 50.0, 1.0
START_PV, STOP_PV, START_RES, STOP_RES = 60.0, 50.0, 100.0, 10.0
STORE_CAP, STORE_INIT, FULL_RESUME = 150.0, 120.0, 140.0
DEMAND, TRIP, RESTART = 0.5, 1.0, 10.0
LAWS = {  # (MTTF, MTTR) in hours
    "PV": (6000.0, 120.0), "BATT": (5000.0, 168.0), "ELEC": (8000.0, 240.0),
    "COMP_A": (3000.0, 48.0), "COMP_B": (3000.0, 48.0),
}

def c(v): return {"op": "const", "value": {"kind": "float", "value": float(v)}}
def a(comp, attr): return {"op": "attr", "attr": {"component": comp, "attribute": attr}}
def p(comp, port): return {"op": "port_agg", "port": {"component": comp, "port": port}, "agg": "sum"}
def s(comp, aut, st): return {"op": "state_active", "state": {"component": comp, "automaton": aut, "state": st}}
def cmp(op, lhs, rhs): return {"op": "cmp", "cmp": op, "lhs": lhs, "rhs": rhs}
def AND(*x): return {"op": "bool", "bool_op": "and", "args": list(x)}
def OR(*x): return {"op": "bool", "bool_op": "or", "args": list(x)}
def IF(k, then, otherwise): return {"op": "if", "cond": k, "then": then, "otherwise": otherwise}
def fvar(name, v): return {"name": name, "kind": "float", "init": {"kind": "float", "value": float(v)}}
def ok(comp): return s(comp, "health", "ok")
```

Each component that can fail carries the same `health` automaton, whose
transitions are `monitored` so that sequence analysis records them. The
compressors also give their pair a `cycle_group`, which is explained with
the sequences below.

```python
#: Failures that act at once: a failure repaired before the feared event
#: did not cause it.
IMMEDIATE = {"COMP_A", "COMP_B"}

def health(name):
    mttf, mttr = LAWS[name]
    fail = {"name": "fail", "source": "ok", "targets": ["failed"], "distrib": "exp",
            "rate": 1 / mttf, "kind": "failure", "monitored": True}
    repair = {"name": "repair", "source": "failed", "targets": ["ok"], "distrib": "exp",
              "rate": 1 / mttr, "kind": "repair", "monitored": True}
    if name in IMMEDIATE:
        fail["cycle_group"] = repair["cycle_group"] = name
    return {"name": "health", "states": ["ok", "failed"], "init": "ok",
            "transitions": [fail, repair]}
```

The PV power is an explicit equation of time, zero while the inverter is
down:

```python
sun = {"op": "sin", "arg": {"op": "mul", "args": [
    c(2 * math.pi / 24.0), {"op": "sub", "lhs": {"op": "time"}, "rhs": c(6.0)}]}}
pv = {
    "name": "PV",
    "attributes": [fvar("power", 0.0)],
    "ports": [{"name": "power_out", "dir": "out", "attr": "power"}],
    "automata": [health("PV")],
    "equations": [{"target": "power", "kind": "explicit", "expr": IF(
        ok("PV"), {"op": "mul", "args": [c(P_PEAK), {"op": "max", "args": [c(0.0), sun]}]},
        c(0.0))}],
}
```

The electrolyser's control is an automaton whose two transitions are
**watched**: the engine locates the instant a threshold is crossed on the
continuous quantities rather than checking it at the next event.

```python
running = AND(s("ELEC", "mode", "running"), ok("ELEC"))
elec = {
    "name": "ELEC",
    "attributes": [fvar("load", 0.0), fvar("h2", 0.0)],
    "ports": [{"name": "pv_in", "dir": "in"}, {"name": "reserve_in", "dir": "in"},
              {"name": "level_in", "dir": "in"},
              {"name": "load_out", "dir": "out", "attr": "load"},
              {"name": "h2_out", "dir": "out", "attr": "h2"}],
    "automata": [
        {"name": "mode", "states": ["stopped", "running"], "init": "stopped",
         "transitions": [
            {"name": "start", "source": "stopped", "targets": ["running"],
             "distrib": "watched", "guard": AND(
                 cmp("lt", p("ELEC", "level_in"), c(FULL_RESUME)),
                 OR(cmp("ge", p("ELEC", "pv_in"), c(START_PV)),
                    cmp("ge", p("ELEC", "reserve_in"), c(START_RES))))},
            {"name": "stop", "source": "running", "targets": ["stopped"],
             "distrib": "watched", "guard": OR(
                 cmp("ge", p("ELEC", "level_in"), c(STORE_CAP)),
                 AND(cmp("lt", p("ELEC", "pv_in"), c(STOP_PV)),
                     cmp("le", p("ELEC", "reserve_in"), c(STOP_RES))))}]},
        health("ELEC")],
    "equations": [
        {"target": "load", "kind": "explicit", "expr": IF(running, c(ELEC_LOAD), c(0.0))},
        {"target": "h2", "kind": "explicit", "expr": IF(running, c(ELEC_H2), c(0.0))}],
}
```

The battery integrates the PV surplus minus the electrolyser's draw. Being
full is a state of its own rather than a clamp inside the equation: a
threshold written into a right-hand side makes the solver hunt for a
discontinuity it is not told about, whereas a state change is an event it
locates.

```python
net = {"op": "sub", "lhs": p("BATT", "pv_in"), "rhs": p("BATT", "load_in")}
soc = a("BATT", "soc")
batt = {
    "name": "BATT",
    "attributes": [fvar("soc", BATT_INIT), fvar("reserve", BATT_INIT)],
    "ports": [{"name": "pv_in", "dir": "in"}, {"name": "load_in", "dir": "in"},
              {"name": "reserve_out", "dir": "out", "attr": "reserve"}],
    "automata": [
        {"name": "charge", "states": ["charging", "full"], "init": "charging",
         "transitions": [
            {"name": "full", "source": "charging", "targets": ["full"], "distrib": "watched",
             "guard": AND(cmp("ge", soc, c(BATT_CAP)), cmp("gt", net, c(0.0)))},
            {"name": "discharge", "source": "full", "targets": ["charging"],
             "distrib": "watched", "guard": cmp("lt", net, c(0.0))}]},
        health("BATT")],
    "equations": [
        {"target": "soc", "kind": "ode", "expr": IF(ok("BATT"), IF(
            s("BATT", "charge", "charging"), net, {"op": "min", "args": [net, c(0.0)]}),
            c(0.0))},
        {"target": "reserve", "kind": "explicit", "expr": IF(ok("BATT"), soc, c(0.0))}],
}
```

The storage integrates production minus delivery; the compressors publish
whether they work; the user trips and recovers on the storage level and on
the compressors.

```python
store = {
    "name": "STORE",
    "attributes": [fvar("level", STORE_INIT)],
    "ports": [{"name": "h2_in", "dir": "in"}, {"name": "draw_in", "dir": "in"},
              {"name": "level_out", "dir": "out", "attr": "level"}],
    "equations": [{"target": "level", "kind": "ode", "expr": {
        "op": "sub", "lhs": p("STORE", "h2_in"), "rhs": p("STORE", "draw_in")}}],
}

def compressor(name):
    return {
        "name": name,
        "attributes": [fvar("up", 1.0)],
        "ports": [{"name": "up_out", "dir": "out", "attr": "up"}],
        "automata": [health(name)],
        "equations": [{"target": "up", "kind": "explicit", "expr": IF(ok(name), c(1.0), c(0.0))}],
    }

user = {
    "name": "USER",
    "attributes": [fvar("draw", DEMAND)],
    "ports": [{"name": "level_in", "dir": "in"}, {"name": "comp_in", "dir": "in"},
              {"name": "draw_out", "dir": "out", "attr": "draw"}],
    "automata": [{"name": "supply", "states": ["supplied", "unsupplied"], "init": "supplied",
        "transitions": [
            {"name": "trip", "source": "supplied", "targets": ["unsupplied"],
             "distrib": "watched", "guard": OR(
                 cmp("le", p("USER", "level_in"), c(TRIP)),
                 cmp("lt", p("USER", "comp_in"), c(0.5)))},
            {"name": "restart", "source": "unsupplied", "targets": ["supplied"],
             "distrib": "watched", "guard": AND(
                 cmp("ge", p("USER", "level_in"), c(RESTART)),
                 cmp("ge", p("USER", "comp_in"), c(0.5)))}]}],
    "equations": [{"target": "draw", "kind": "explicit",
                   "expr": IF(s("USER", "supply", "supplied"), c(DEMAND), c(0.0))}],
}
```

The wiring follows the schematic, and the user's loss of supply is both an
indicator and the **target** of sequence analysis:

```python
def wire(src, out, dst, inp):
    return {"from": {"component": src, "port": out}, "to": {"component": dst, "port": inp}}

model = pyraichu.load_model({
    "name": "h2_installation",
    "components": [pv, batt, elec, store, compressor("COMP_A"), compressor("COMP_B"), user],
    "connections": [
        wire("PV", "power_out", "ELEC", "pv_in"), wire("PV", "power_out", "BATT", "pv_in"),
        wire("ELEC", "load_out", "BATT", "load_in"), wire("BATT", "reserve_out", "ELEC", "reserve_in"),
        wire("ELEC", "h2_out", "STORE", "h2_in"), wire("USER", "draw_out", "STORE", "draw_in"),
        wire("STORE", "level_out", "USER", "level_in"), wire("STORE", "level_out", "ELEC", "level_in"),
        wire("COMP_A", "up_out", "USER", "comp_in"), wire("COMP_B", "up_out", "USER", "comp_in"),
    ],
    "indicators": [
        {"name": "unsupplied", "target": "state",
         "component": "USER", "automaton": "supply", "state": "unsupplied"},
        {"name": "storage", "target": "attribute",
         "attr": {"component": "STORE", "attribute": "level"}},
        {"name": "battery", "target": "attribute",
         "attr": {"component": "BATT", "attribute": "soc"}},
        {"name": "pv", "target": "attribute",
         "attr": {"component": "PV", "attribute": "power"}},
        {"name": "electrolyser", "target": "attribute",
         "attr": {"component": "ELEC", "attribute": "load"}},
    ],
    "targets": [{"name": "unsupplied", "component": "USER",
                 "automaton": "supply", "state": "unsupplied"}],
})
```

A first quick look, one week without asking for anything else:

```python
week = pyraichu.simulate(model, t_max=24 * 7, seed=3,
                         samples=[24.0 * d for d in range(8)])
print("storage at each midnight:", [round(v, 1) for _, v in week.samples["storage"]])
```

## Trajectory around a loss of supply

One trajectory, chosen because the user loses supply in it, over the days
around that event:

![One trajectory: PV power and electrolyser draw, battery charge, storage level, with the electrolyser failure, its repair and the loss of supply marked](../assets/figures/example-h2-trajectory-light.svg#only-light){ .figure }
![One trajectory: PV power and electrolyser draw, battery charge, storage level, with the electrolyser failure, its repair and the loss of supply marked](../assets/figures/example-h2-trajectory-dark.svg#only-dark){ .figure }

## Unavailability over one year

`monte_carlo` estimates, at each instant, the probability that the user is
without hydrogen (`mean`: the installation keeps running and repairing),
and the probability that the user **has lost supply at least once** by
that instant (`reached_mean`), each with its confidence interval. The figure script runs 1 000 replicas over
8 760 h; the same call on a quarter and 50 replicas runs in seconds:

```python
months = [730.0, 1460.0, 2190.0]
est = pyraichu.monte_carlo(model, nb_runs=50, t_max=2190.0, samples=months, seed=1)
lost = est.indicators["unsupplied"]
print("P(lost supply at least once) by month 1, 2, 3:",
      [round(v, 2) for v in lost.reached_mean])
```

![Probability of being unsupplied at each instant and of having been unsupplied at least once, with 95 % confidence bands, over one year](../assets/figures/example-h2-unavailability-light.svg#only-light){ .figure }
![Probability of being unsupplied at each instant and of having been unsupplied at least once, with 95 % confidence bands, over one year](../assets/figures/example-h2-unavailability-dark.svg#only-dark){ .figure }

## Minimal sequences

`analyse_sequences` runs a campaign in which each trajectory stops at the
first loss of supply, and reduces the recorded failures to the **minimal
sequences** that lead to it:

<!-- skip -->
```python
sequences = pyraichu.analyse_sequences(model, nb_runs=1000, t_max=8760.0, seed=1)
```

![Minimal sequences leading to the loss of supply, by share of the 1 000 trajectories](../assets/figures/example-h2-sequences-light.svg#only-light){ .figure }
![Minimal sequences leading to the loss of supply, by share of the 1 000 trajectories](../assets/figures/example-h2-sequences-dark.svg#only-dark){ .figure }

**A `cycle_group` on the compressors only.** Sequence reduction
removes a failure that was repaired before the feared event, on the
hypothesis that a repaired component no longer contributes. That holds
for a compressor, whose loss acts at once. It does not hold for the
production chain, because the storage delays its effect. In 6 of the 421
trajectories that lose supply, every failure had been repaired by then:
the PV inverter comes back at 15:30, the electrolyser runs until 17:55 on
the afternoon sun with an empty battery, and the storage, down to 2.8 kg,
runs dry at 23:50. Declared on the production chain, the hypothesis
reduces those trajectories to an **empty** sequence, and an empty
sequence that reaches the event is contained in every other one: all 421
trajectories then collapse onto it, and the analysis answers that no
failure is needed. The `cycle_group` is therefore a modelling statement,
made per component.

## Reading the results

**The trajectory** shows the nominal rhythm first: the electrolyser runs
from the morning, carries on into the evening on the battery, and the
storage oscillates just under 150 kg. The electrolyser then fails, at
t = 3 982 h. Production stops, the battery fills and stays full with
nothing drawing on it, and the storage empties at the user's 0.5 kg/h:
143 kg last 286 h, and the user loses supply at t = 4 269 h, twelve days
after the failure. The repair comes 356 h after the failure, longer than
the 240 h mean but not unusual for an exponential law, and the user is
supplied again once the storage is back at 10 kg.

**The unavailability** (1 000 replicas, one year) sits around 1.1 % at any
given instant, and the band shows how loosely 1 000 replicas pin a
probability of that size down. The probability of having lost supply at
least once grows almost linearly to 0.421 at one year, within
[0.391, 0.452]: a loss of supply is short and rare at any given moment,
but likely over a year.

**The minimal sequences** account for the same 421 trajectories out of
1 000: the electrolyser alone in 303 of them, the PV inverter alone in 91,
and the two compressors in 27 (16 with A first, 11 with B first; a
sequence is ordered). Both production failures lead to a loss of supply
only because their repair outlasts the storage, which is where the
storage size and the spare-parts delay act. The battery never appears on
its own: without it the electrolyser runs in daylight only, producing
about 9.1 kg a day against 12 kg drawn, so the storage covers the
2.9 kg/day deficit for about fifty days, far beyond the battery's 168 h
mean repair.

## Reproducing the figures

The charts come from `docs/figures/example_h2_installation.py`, which runs
the code of this page and then the campaigns above:

```bash
.venv/bin/python scripts/build_doc_figures.py example_h2
```
