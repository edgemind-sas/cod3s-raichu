# Component type libraries

This guide builds a small cyber-physical system from a library of component
types, then runs a study that chains three compromise modes. Everything is
data: the library, the system and the study are three JSON documents, and
the engine expands them into the flat model it simulates. The full format is
in the [component type library format reference](../reference/library-format.md).

The example is the STRA cyber case of the IMdR P23-4 project: a power supply
feeds a server, the server feeds an industrial process, and an attacker
escalates privileges on the server, starts a dormant service function, and
through it inhibits the process.

## The library

A type is built from fragments of the standard library `std` and declares
the modes it can suffer. The server's service output is dormant: it has no
production condition and starts unproduced, so only a mode can start it.

```python
import pyraichu

def attr(component, attribute):
    return {"op": "attr", "attr": {"component": component, "attribute": attribute}}

def is_(component, attribute, value):
    return {"op": "cmp", "cmp": "eq", "lhs": attr(component, attribute),
            "rhs": {"op": "const", "value": {"kind": "bool", "value": value}}}

library = {
    "raichu_library": {"format": 1},
    "library": {
        "name": "cyber", "version": "1.0.0", "imports": ["std"],
        "types": [
            {"name": "PowerSupply",
             "uses": [{"fragment": "std.bool_source_out", "with": {"name": "power"}}]},
            {"name": "Server",
             "uses": [
                 {"fragment": "std.bool_flow_in", "with": {"name": "power"}},
                 {"fragment": "std.bool_flow_out",
                  "with": {"name": "orange", "prod_cond": [["power_fed_in"]]}},
                 {"fragment": "std.bool_source_out",
                  "with": {"name": "service", "prod_init": False}}],
             "modes": [
                 {"name": "escalation", "nature": "compromise"},
                 {"name": "service_exploit", "nature": "compromise",
                  "effects": [{"attribute": "service_prod_available", "value": True}]}]},
            {"name": "Process",
             "uses": [
                 {"fragment": "std.bool_flow_in", "with": {"name": "orange"}},
                 {"fragment": "std.bool_flow_in", "with": {"name": "service", "logic": "or"}},
                 {"fragment": "std.bool_flow_out",
                  "with": {"name": "F1", "prod_cond": [["orange_fed_in"]]}}],
             "modes": [
                 {"name": "inhibition", "nature": "compromise",
                  "condition": is_("$self", "service_fed_in", True),
                  "effects": [{"attribute": "F1_fed_available_out", "value": False}]}]},
        ],
    },
}
```

`std.bool_flow_out` makes `<name>_fed_available_out` targetable with the rule
`any_cuts`, so the process's `inhibition` may cut `F1`;
`std.bool_source_out` makes the dormant `service_prod_available` targetable
with `any_activates`, so `service_exploit` may start it. The process's
catalogue says *when* `inhibition` can happen (once the service flow
arrives), never *how fast*: the law belongs to the study.

A library can be checked on its own, before any system uses it:

```python
assert [entry["status"] for entry in pyraichu.check_library(library)] == ["valid"] * 3
```

## The system

The system instantiates the types by name and connects their ports, written
`Instance.port`:

```python
system = {
    "raichu_system": {"format": 1},
    "system": {
        "name": "plant",
        "library": {"name": "cyber", "version": "1.0.0"},
        "instances": [
            {"name": "Alim", "type": "PowerSupply"},
            {"name": "Srv", "type": "Server"},
            {"name": "Proc", "type": "Process"}],
        "connections": [
            {"from": "Alim.power_out", "to": "Srv.power_in"},
            {"from": "Srv.orange_out", "to": "Proc.orange_in"},
            {"from": "Srv.service_out", "to": "Proc.service_in"}],
        "indicators": [
            {"name": "F1", "target": "attribute", "attr": {"component": "Proc", "attribute": "F1_fed_out"}}],
    },
}
```

## The study

The study activates the catalogued modes with their laws, and chains them:
`service_exploit` is possible only once `escalation` has occurred, which the
`mode_active` leaf states by the mode's public name.

```python
study = {
    "raichu_study": {"format": 1},
    "study": {
        "name": "attack",
        "system": "plant",
        "modes": [
            {"mode": "escalation", "instances": ["Srv"],
             "occurrence": {"distrib": "delay", "time": 10}},
            {"mode": "service_exploit", "instances": ["Srv"],
             "occurrence": {"distrib": "delay", "time": 5},
             "condition": {"op": "mode_active", "mode": "Srv.escalation"}},
            {"mode": "inhibition", "type": "Process",
             "occurrence": {"distrib": "delay", "time": 8}}],
        "feared_events": [
            {"name": "F1_lost", "target": True, "tempo_occ": 0.01,
             "condition": is_("Proc", "F1_fed_out", False)}],
        "simulation": {"method": "monte_carlo", "nb_runs": 1, "horizon": 30.0,
                       "instants": [14.0, 16.0, 22.0, 24.0]},
    },
}
```

## Running it

```python
model = pyraichu.load_system(system, [library], study)
result = pyraichu.simulate(model, t_max=30.0)
assert [(e.transition, e.time) for e in result.events] == [
    ("Srv.escalation.occ", 10.0),
    ("Srv.service_exploit.occ", 15.0),
    ("Proc.inhibition.occ", 23.0),
    ("F1_lost.ev.occ", 23.01),
]
assert result.indicators["F1"] == [(0.0, True), (23.0, False)]
```

Each mode is an automaton named after it inside its instance, so the events
read `Srv.service_exploit.occ`: the name the study wrote, not a generated
one. The study's run settings travel with the model:

```python
estimates = pyraichu.run_study(model)
```

## Where an element came from

Every element of the expanded model is traced back to the document node
that produced it:

```python
origin = model.provenance["attribute:Proc.F1_fed_available_out"]
assert origin["fragment"] == "std.bool_flow_out"
assert origin["type"] == "Process"
origin = model.provenance["automaton:Srv.service_exploit"]
assert (origin["document"], origin["pointer"], origin["nature"]) == ("study", "/study/modes/1", "compromise")
```

A refused document says which document, where, and why:

```python
bad = {"raichu_study": {"format": 1}, "study": {**study["study"], "modes": [
    {"mode": "service_exploit", "instances": ["Srv"],
     "occurrence": {"distrib": "delay", "time": 5},
     "condition": {"op": "mode_active", "mode": "Srv.escalation"}}]}}
try:
    pyraichu.load_system(system, [library], bad)
except pyraichu.ModelError as error:
    assert "`Srv.escalation`, which the study does not activate" in str(error)
```

## Documents on disk

`load_system` also takes paths. JSON files are read as is; `.yaml` and
`.yml` files are read through PyYAML when it is installed. YAML reads an
unquoted `on`, `off`, `yes` or `no` as a boolean, so quote a name spelled
that way.

## Continuous types

The continuous fragments of `std` build a conservative flow network out of
component-local declarations: every output publishes, on each of its
connections, what it can deliver and what it hands over, every input what
it asks and the share it accepts, and a distribution operator splits the
output over the claims. A type assembles them the way the boolean types
above assemble theirs. Here a source of 2 per hour fills a tank of
capacity 10 that nobody draws from:

```python
tank_library = {
    "raichu_library": {"format": 1},
    "library": {
        "name": "tanks", "version": "1.0.0", "imports": ["std"],
        "types": [
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "std.continuous_source", "with": {"name": "Q", "rate": 2.0}}]},
            {"name": "Tank", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "Q", "demand": 2.0}},
                {"fragment": "std.volume_flow",
                 "with": {"volume": "tank", "flow": "Q", "capacity": 10.0, "demand": 2.0}},
                {"fragment": "std.volume_outlet",
                 "with": {"volume": "tank", "flow": "Q", "rate": 2.0, "empty_automaton": "tank_bounds"}},
                {"fragment": "std.continuous_flow_out",
                 "with": {"name": "Q", "held": True, "empty_automaton": "tank_bounds"}},
                {"fragment": "std.volume_total",
                 "with": {"volume": "tank", "flows": ["Q"], "capacity": 10.0}}]},
        ],
    },
}
tank_system = {
    "raichu_system": {"format": 1},
    "system": {
        "name": "filling", "library": {"name": "tanks", "version": "1.0.0"},
        "instances": [{"name": "S", "type": "Source"}, {"name": "T", "type": "Tank"}],
        "connections": [{"from": "S.Q_out", "to": "T.Q_in"}],
        "indicators": [{"name": "content", "target": "attribute",
                        "attr": {"component": "T", "attribute": "tank_content"}}],
    },
}
tank_model = pyraichu.load_system(tank_system, [tank_library])
run = pyraichu.simulate(tank_model, t_max=8.0, samples=[0.0, 2.0, 4.0, 6.0, 8.0])
assert [round(v, 9) for _, v in run.samples["content"]] == [0.0, 4.0, 8.0, 10.0, 10.0]
full = [e.time for e in run.events if e.transition == "T.tank_bounds.tank_reach_full"]
assert abs(full[0] - 5.0) < 1e-9
```

The engine derives the order in which the network's equations are swept
from what each one reads, so a library author never writes one; the
per-connection quantities are [port equations](../reference/model-schema.md#port-equation)
of the fragments. The [format reference](../reference/library-format.md#the-standard-library)
lists every continuous fragment.

## Published examples written as libraries

Six worked examples carry a library of their own, under
`examples/<name>/library/`, each shown on its page and proved to behave
exactly as the page's flat model:

| Example | Types | What the library shows |
|---|---|---|
| [Two-out-of-three pumps](../examples/two-out-of-three.md) | `Pump`, `Voter` | a failure mode on the type with its laws in the study, k-out-of-n with `std.bool_flow_in` |
| [Heated room](../examples/heated-room.md) | `Heater`, `Room` | two instances of one type differing by parameters, a neighbour's temperature read through a port |
| [Heated tank](../examples/heated-tank.md) | `Pump`, `Valve`, `Tank` | two types built on one library-local fragment, a failure rate reading a port |
| [Hydrogen installation](../examples/h2-installation.md) | six equipment types | a shared `health` fragment, explicit equations ordered by the expander |
| [Gas production](../examples/gas-production.md) | `ProductionUnit`, `BufferReservoir` | four cases as four systems over one library, a state published with `std.state_out`, a repair writing a neighbour through a port |
| [APPRODYN](../examples/approdyn-feedwater.md) | eight types | phases published as wiring, latched conditions written through ports, variants kept or dropped with `$when` |

Two of them are generated from the flat model's own constants
(`examples/<name>/library.py`), because nothing computes at expansion and
their rates are products of published shares.
