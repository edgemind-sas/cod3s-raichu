# How the engine works

These pages follow RAICHU step by step on one small installation, from what
happens the instant something breaks to how a continuous quantity decides
the next event. Everything the diagrams show was recorded by running the
engine on the model below: nothing is drawn by hand, and a test checks that
the recordings still match the engine.

- [Fixpoint and race of dates](fixpoint-and-race.md): what the engine does
  when an event happens, and how it picks the next one.
- [Boundaries](boundaries.md): what changes when a level evolves
  continuously between events.

## The installation

Water is pumped into a tank. A **main pump** `P1` normally does the work. A
**backup pump** `P2` stays in standby and is started only when `P1` stops
delivering; starting may fail. The **tank** is fed as long as one pump
delivers.

| Component | Attributes | States | Failure mode |
|---|---|---|---|
| `P1` | `delivers` | `working`, `failed` | fails while working, then gets repaired |
| `P2` | `solicited`, `delivers` | `standby`, `running`, `failed` | fails to start on demand (10 %), or fails while running |
| `Tank` | `fed` | (none in the first chapter) | (none) |

Each component has one **sensitive function** that recomputes all of its
attributes from what it reads: `P1` from its own state, `P2` from `P1`'s
`delivers` and its own state, the tank from both pumps.

## Vocabulary

| Term | Meaning on these pages |
|---|---|
| event | a transition firing: a state change of one automaton |
| sensitive function | a function that recomputes some attributes; it runs again whenever something it reads changes |
| iteration of the fixpoint | one application of one sensitive function: the engine applies the pending functions one at a time, lowest index first, until none is left |
| stable state | the state reached when no function is pending any more |
| armed transition | a transition whose source state is active and whose guard holds: it is waiting for its date |
| firing date | the date an armed stochastic transition will fire, sampled when it is armed |
| watched transition | a transition whose guard reads a continuous quantity: it fires the instant the guard turns true |

## The model

The model is plain data, built by a few helpers:

```python
import pyraichu

FAIL_P1, REPAIR_P1 = 0.02, 0.1   # 1/h, a teaching scale rather than a plant's
FAIL_P2, REPAIR_P2 = 0.05, 0.1   # 1/h
START_FAILS = 0.1                # probability that P2 fails to start on demand


def const(value):
    kind = "bool" if isinstance(value, bool) else "float"
    return {"op": "const", "value": {"kind": kind, "value": value}}


def attr(component, name):
    return {"op": "attr", "attr": {"component": component, "attribute": name}}


def in_state(component, automaton, state):
    return {"op": "state_active",
            "state": {"component": component, "automaton": automaton, "state": state}}


def flag(name, value):
    return {"name": name, "kind": "bool", "init": {"kind": "bool", "value": value}}


def write(component, name, value):
    return {"target": {"component": component, "attribute": name}, "value": value}


def either(*args):
    return {"op": "bool", "bool_op": "or", "args": list(args)}


def negate(arg):
    return {"op": "bool", "bool_op": "not", "args": [arg]}


def exp(name, source, target, rate):
    return {"name": name, "source": source, "targets": [target],
            "distrib": "exp", "rate": rate}


main_pump = {
    "name": "P1",
    "attributes": [flag("delivers", True)],
    "automata": [{"name": "mode", "states": ["working", "failed"], "init": "working",
                  "transitions": [exp("fail", "working", "failed", FAIL_P1),
                                  exp("repair", "failed", "working", REPAIR_P1)]}],
    "sensitive_functions": [{"name": "update_P1", "effects": [
        write("P1", "delivers", in_state("P1", "mode", "working"))]}],
}

backup_pump = {
    "name": "P2",
    "attributes": [flag("solicited", False), flag("delivers", False)],
    "automata": [{"name": "mode", "states": ["standby", "running", "failed"],
                  "init": "standby", "transitions": [
        # Started when solicited: runs (90 %) or fails on demand (10 %).
        {"name": "start", "source": "standby", "targets": ["running", "failed"],
         "distrib": "inst", "probs": [1.0 - START_FAILS], "guard": attr("P2", "solicited")},
        {"name": "release", "source": "running", "targets": ["standby"],
         "distrib": "inst", "probs": [], "guard": negate(attr("P2", "solicited"))},
        exp("fail", "running", "failed", FAIL_P2),
        exp("repair", "failed", "standby", REPAIR_P2),
    ]}],
    "sensitive_functions": [{"name": "update_P2", "effects": [
        write("P2", "solicited", negate(attr("P1", "delivers"))),
        write("P2", "delivers", in_state("P2", "mode", "running"))]}],
}

tank = {
    "name": "Tank",
    "attributes": [flag("fed", True)],
    "sensitive_functions": [{"name": "update_Tank", "effects": [
        write("Tank", "fed", either(attr("P1", "delivers"), attr("P2", "delivers")))]}],
}

discrete_model = pyraichu.load_model({
    "name": "explainer_discrete",
    "components": [main_pump, backup_pump, tank],
})
```

The [boundaries](boundaries.md) chapter gives the tank a level. The pumps
are unchanged; the tank gains a `level` governed by a differential equation,
an `inflow` its function computes, and a `regime` automaton whose
transitions are **watched**: they fire the instant the level crosses a
threshold.

```python
CONSUMPTION = 1.0        # m/h drawn from the tank
Q_MAIN, Q_BACKUP = 2.0, 0.5   # m/h delivered by P1 and by P2
HIGH, HIGH_RESET = 8.0, 7.0   # m: P1 throttled above HIGH, released below HIGH_RESET
LOW, LOW_RESET = 2.0, 3.0     # m: alarm below LOW (consumers cut off), cleared above LOW_RESET
LEVEL_0 = 5.0                 # m


def cmp(op, lhs, rhs):
    return {"op": "cmp", "cmp": op, "lhs": lhs, "rhs": rhs}


def watched(name, source, target, guard):
    return {"name": name, "source": source, "targets": [target],
            "guard": guard, "distrib": "watched"}


def pick(cond, then, otherwise):
    return {"op": "if", "cond": cond, "then": then, "otherwise": otherwise}


level = attr("Tank", "level")
full = in_state("Tank", "regime", "full")

tank_with_level = {
    "name": "Tank",
    "attributes": [flag("fed", True),
                   {"name": "inflow", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                   {"name": "level", "kind": "float", "init": {"kind": "float", "value": LEVEL_0}}],
    "automata": [{"name": "regime", "states": ["normal", "full", "alarm"], "init": "normal",
                  "transitions": [
        watched("reach_high", "normal", "full", cmp("ge", level, const(HIGH))),
        watched("leave_high", "full", "normal", cmp("le", level, const(HIGH_RESET))),
        watched("reach_low", "normal", "alarm", cmp("le", level, const(LOW))),
        watched("leave_low", "alarm", "normal", cmp("ge", level, const(LOW_RESET))),
    ]}],
    "sensitive_functions": [{"name": "update_Tank", "effects": [
        write("Tank", "fed", either(attr("P1", "delivers"), attr("P2", "delivers"))),
        # Above HIGH the main pump is throttled to the consumption: the level holds.
        write("Tank", "inflow", pick(attr("P1", "delivers"),
                                     pick(full, const(CONSUMPTION), const(Q_MAIN)),
                                     pick(attr("P2", "delivers"), const(Q_BACKUP), const(0.0)))),
    ]}],
    # In alarm the consumers are cut off, so the level cannot fall below LOW.
    "equations": [{"target": "level", "kind": "ode",
                   "expr": {"op": "sub", "lhs": attr("Tank", "inflow"),
                            "rhs": pick(in_state("Tank", "regime", "alarm"),
                                        const(0.0), const(CONSUMPTION))}}],
}

continuous_model = pyraichu.load_model({
    "name": "explainer_continuous",
    "components": [main_pump, backup_pump, tank_with_level],
    "indicators": [{"name": "level", "target": "attribute",
                    "attr": {"component": "Tank", "attribute": "level"}}],
})
```
