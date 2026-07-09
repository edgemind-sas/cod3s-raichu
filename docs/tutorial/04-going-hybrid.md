# 4. Going hybrid: continuous dynamics

So far the state changed only in discrete jumps. Many systems also
**evolve continuously** between jumps — a temperature, a level, a
current — and the jumps and the continuous flow feed back on each other.
This is what RAICHU is built for. We model a **heated room**: a
thermostat switches a heater on and off at temperature thresholds, the
room temperature follows an ODE, and the heater can fail.

Two new ingredients:

- **equations** — a continuous attribute governed by `dV/dt = …`
  (`ode`) or `V = …` (`explicit`);
- **watched transitions** — a *guarded* transition whose guard involves a
  continuous attribute. Because that attribute drifts between events, the
  engine must **watch the trajectory** and fire the transition exactly
  when the guard flips — the crossing is located by root-finding, not by
  stepping. You declare this intent explicitly (`"distrib": "watched"`): the
  engine cannot guess, from a comparison alone, whether you mean "fire at
  the crossing" or merely "be eligible while true".

## The room: a continuous attribute

The room owns a `float` `temperature` and an **ODE**: it warms with the
power it receives and loses heat to the outside at rate `0.1`. The
incoming power is read from an in-port with a `sum` aggregation (several
heaters could feed it).

```python
import pyraichu

room = {
    "name": "Room",
    "attributes": [{"name": "temperature", "kind": "float",
                   "init": {"kind": "float", "value": 17.0}}],
    "ports": [{"name": "power_in", "dir": "in"},
              {"name": "temp_out", "dir": "out", "attr": "temperature"}],
    "equations": [{
        "target": "temperature", "kind": "ode",
        "expr": {"op": "sub",
            "lhs": {"op": "port_agg",
                    "port": {"component": "Room", "port": "power_in"}, "agg": "sum"},
            "rhs": {"op": "mul", "args": [
                {"op": "const", "value": {"kind": "float", "value": 0.1}},
                {"op": "sub",
                 "lhs": {"op": "attr",
                         "attr": {"component": "Room", "attribute": "temperature"}},
                 "rhs": {"op": "const", "value": {"kind": "float", "value": 13.0}}},
            ]}},
    }],
}
```

## The heater: watched transitions and a failure

The heater's thermostat is two **watched transitions**: turn `ON` when
the room falls below 15, `OFF` when it climbs above 20. A watched guard
is a comparison on a continuous quantity — the engine treats it as a
boundary and root-finds the crossing, so the switch happens at exactly
the right temperature, not at the nearest time step.

The heater also has a `health` automaton that can fail (exponentially)
and be repaired. Its delivered `power` is `5` only when it is both `ON`
and `OK` — a sensitive function with an `if` expression couples the
discrete state to the continuous input of the room.

```python
def threshold(op, value):
    return {"op": "cmp", "cmp": op,
            "lhs": {"op": "port_agg",
                    "port": {"component": "H", "port": "temp_in"}, "agg": "sum"},
            "rhs": {"op": "const", "value": {"kind": "float", "value": value}}}

heater = {
    "name": "H",
    "attributes": [{"name": "power", "kind": "float",
                   "init": {"kind": "float", "value": 5.0}}],
    "ports": [{"name": "temp_in", "dir": "in"},
              {"name": "power_out", "dir": "out", "attr": "power"}],
    "automata": [
        {"name": "functional", "states": ["ON", "OFF"], "init": "ON",
         "transitions": [
            {"name": "ON_to_OFF", "source": "ON", "targets": ["OFF"],
             "distrib": "watched", "guard": threshold("gt", 20.0)},
            {"name": "OFF_to_ON", "source": "OFF", "targets": ["ON"],
             "distrib": "watched", "guard": threshold("lt", 15.0)},
         ]},
        {"name": "health", "states": ["OK", "KO"], "init": "OK",
         "transitions": [
            {"name": "fail", "source": "OK", "targets": ["KO"],
             "distrib": "exp", "rate": 0.01},
            {"name": "repair", "source": "KO", "targets": ["OK"],
             "distrib": "exp", "rate": 0.2},
         ]},
    ],
    "sensitive_functions": [{
        "name": "update_power",
        "effects": [{
            "target": {"component": "H", "attribute": "power"},
            "value": {"op": "if",
                "cond": {"op": "bool", "bool_op": "and", "args": [
                    {"op": "state_active",
                     "state": {"component": "H", "automaton": "functional",
                               "state": "ON"}},
                    {"op": "state_active",
                     "state": {"component": "H", "automaton": "health",
                               "state": "OK"}}]},
                "then": {"op": "const", "value": {"kind": "float", "value": 5.0}},
                "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.0}}},
        }],
    }],
}
```

## Wiring and running

Power flows heater → room; temperature flows room → heater (so the
thermostat can watch it):

```python
model = pyraichu.load_model({
    "name": "heated_room",
    "components": [heater, room],
    "connections": [
        {"from": {"component": "H", "port": "power_out"},
         "to": {"component": "Room", "port": "power_in"}},
        {"from": {"component": "Room", "port": "temp_out"},
         "to": {"component": "H", "port": "temp_in"}},
    ],
    "indicators": [{"name": "temp", "target": "attribute",
                    "attr": {"component": "Room", "attribute": "temperature"}}],
})
```

A single trajectory shows the thermostat holding the room inside its
band (the dips and climbs are the heater cycling, and the occasional
failures):

```python
run = pyraichu.simulate(model, t_max=100.0, seed=1,
                        samples=[10.0 * k for k in range(11)])
print("temperature:", [round(v, 1) for _, v in run.samples["temp"]])
```

And a Monte-Carlo run gives the expected room temperature over time,
now *including* the effect of random heater failures on the continuous
dynamics:

```python
est = pyraichu.monte_carlo(
    model, nb_runs=2000, t_max=100.0,
    samples=[10.0 * k for k in range(11)], seed=1,
)
print("mean temperature:", [round(v, 1) for v in est.indicators["temp"].mean])
```

!!! note "What the engine did"
    Between jumps it integrated the temperature ODE with an adaptive
    Dormand–Prince method and **located each thermostat crossing** by
    root-finding on the guard, not by stepping. When the heater failed,
    the discrete jump changed the ODE's right-hand side, and integration
    resumed from the exact failure instant. The numerical tolerances are
    explicit and tunable — see
    [Numerical tuning](../guides/numerical-tuning.md).

## Where to go next

You now have the whole modelling vocabulary: components, attributes,
ports, automata, distributions, sensitive functions, equations and watched
transitions. From here:

- the [Model schema reference](../reference/model-schema.md) documents
  every field and operator;
- the [advanced guides](../guides/reproducibility.md) cover
  reproducibility, numerical tuning, the causal journal, the muscadet
  authoring layer and parallelism;
- the [benchmarks](../benchmarks/cross-validation.md) show RAICHU
  measured against an established C++ engine.
