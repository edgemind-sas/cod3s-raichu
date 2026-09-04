# Numerical tuning

When a model has continuous dynamics, the ODE integrator's effort is
**explicit and adjustable**: not a hidden constant. This lets you trade
accuracy for speed deliberately, and record exactly what you ran (the
tolerances are part of the [provenance](reproducibility.md)).

## The knobs

`monte_carlo` (and the engine behind `simulate`) accept:

| keyword | meaning | default |
|---|---|---|
| `rtol` | relative step-error tolerance | `1e-9` |
| `atol` | absolute step-error tolerance | `1e-12` |
| `max_step` | hard cap on the step size (missed-crossing safety net) | `0.1` |
| `tol_event` | time tolerance of the boundary-crossing bisection | `1e-10` |
| `sub_samples` | dense interior points scanned per step for guard crossings | `16` |

The defaults are **deliberately conservative**: they locate events to
`1e-10` and scan 16 interior points per step, buying far more accuracy
than most studies need. Relaxing them can speed a hybrid Monte-Carlo run
by an order of magnitude at an accuracy that is still excellent.

## A fast profile

Consider a deterministic thermostat: a heater cycles on a room whose
temperature follows an ODE, switching at watched thresholds (the
[hybrid tutorial](../tutorial/04-going-hybrid.md) builds this model).

<!-- skip -->
```python
# ... build `model` as in the hybrid tutorial (heater + room) ...
```

```python
import pyraichu

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
    "automata": [{"name": "functional", "states": ["ON", "OFF"], "init": "ON",
        "transitions": [
            {"name": "off", "source": "ON", "targets": ["OFF"],
             "distrib": "watched", "guard": threshold("gt", 20.0)},
            {"name": "on", "source": "OFF", "targets": ["ON"],
             "distrib": "watched", "guard": threshold("lt", 15.0)}]}],
    "sensitive_functions": [{"name": "p", "effects": [{
        "target": {"component": "H", "attribute": "power"},
        "value": {"op": "if",
            "cond": {"op": "state_active",
                     "state": {"component": "H", "automaton": "functional",
                               "state": "ON"}},
            "then": {"op": "const", "value": {"kind": "float", "value": 5.0}},
            "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.0}}}}]}],
}
room = {
    "name": "Room",
    "attributes": [{"name": "temperature", "kind": "float",
                   "init": {"kind": "float", "value": 17.0}}],
    "ports": [{"name": "power_in", "dir": "in"},
              {"name": "temp_out", "dir": "out", "attr": "temperature"}],
    "equations": [{"target": "temperature", "kind": "ode", "expr": {"op": "sub",
        "lhs": {"op": "port_agg",
                "port": {"component": "Room", "port": "power_in"}, "agg": "sum"},
        "rhs": {"op": "mul", "args": [
            {"op": "const", "value": {"kind": "float", "value": 0.1}},
            {"op": "sub",
             "lhs": {"op": "attr",
                     "attr": {"component": "Room", "attribute": "temperature"}},
             "rhs": {"op": "const", "value": {"kind": "float", "value": 13.0}}}]}}}],
}
model = pyraichu.load_model({
    "name": "thermostat", "components": [heater, room],
    "connections": [
        {"from": {"component": "H", "port": "power_out"},
         "to": {"component": "Room", "port": "power_in"}},
        {"from": {"component": "Room", "port": "temp_out"},
         "to": {"component": "H", "port": "temp_in"}}],
    "indicators": [{"name": "temp", "target": "attribute",
                    "attr": {"component": "Room", "attribute": "temperature"}}],
})

samples = [10.0 * k for k in range(11)]
default = pyraichu.monte_carlo(model, nb_runs=1, t_max=100.0, samples=samples, seed=1)
fast = pyraichu.monte_carlo(model, nb_runs=1, t_max=100.0, samples=samples, seed=1,
                            rtol=1e-6, atol=1e-9, tol_event=1e-6,
                            max_step=1.0, sub_samples=8)

gap = max(abs(a - b) for a, b in
          zip(default.indicators["temp"].mean, fast.indicators["temp"].mean))
print(f"fast vs default: max temperature gap = {gap:.1e}")
```

The `fast` profile tracks the conservative default to ~10⁻⁵ °C while
doing far less integration work. On this model the
[accuracy-cost parity benchmark](../benchmarks/accuracy-cost-parity.md)
measures that ~10⁻⁵ setting running an order of magnitude faster than
the default, and the default, in turn, delivers 3-4 orders of magnitude
more accuracy than a typical study can use.

## The flow-resolution policy

A model carrying a **distribution operator** (a supply split across
several consumers) resolves its network to a fixpoint before each
segment. That resolution has its own policy, adjustable through one
object rather than four more keywords, and accepted by `simulate`,
`monte_carlo`, `analyse_sequences` and `Interactive` under a single
`flow=`:

| knob | meaning | default |
|---|---|---|
| `sweep_budget` | sweeps the numeric level may spend once the saturation pattern has settled | `64` |
| `active_set_budget` | sweeps the combinatorial level may spend; `None` derives it from the compiled network | `None` |
| `relaxation` | under-relaxation weight latched on a detected two-cycle (`1.0` = no damping) | `0.5` |
| `tolerance` | per-edge convergence tolerance, and the dead band of every active-set margin | `1e-9` |

Take a supply of 100 units feeding one consumer that **backs off as it
is served**: it asks for `6 - 1.5x` where `x` is what it currently
receives. That is an over-correcting regulator, and it is what a network
whose demand depends on its own supply looks like at its most awkward.

```python
import pyraichu

served = {"op": "attr",
          "attr": {"component": "supply", "attribute": "out__alloc__ea"}}
network = pyraichu.load_model({
    "name": "backing_off",
    "components": [
        {"name": "supply",
         "attributes": [{"name": "capacity", "kind": "float",
                         "init": {"kind": "float", "value": 100.0}}],
         "ports": [{"name": "out", "dir": "out", "attr": "capacity",
                    "channels": [{"name": "demand"}, {"name": "alloc"}]}],
         "equations": [{"target": "out__demand__ea", "kind": "explicit",
             "expr": {"op": "sub",
                 "lhs": {"op": "const", "value": {"kind": "float", "value": 6.0}},
                 "rhs": {"op": "mul", "args": [
                     {"op": "const", "value": {"kind": "float", "value": 1.5}},
                     served]}}}],
         "allocations": [{"name": "split", "port": "out", "demand": "demand",
             "allocated": "alloc", "policy": "proportional",
             "available": {"op": "attr",
                 "attr": {"component": "supply", "attribute": "capacity"}}}]},
        {"name": "a",
         "attributes": [{"name": "got", "kind": "float",
                         "init": {"kind": "float", "value": 0.0}}],
         "ports": [{"name": "input", "dir": "in"}],
         "equations": [{"target": "got", "kind": "explicit",
             "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                      "port": {"component": "a", "port": "input"}}}]},
    ],
    "connections": [{"name": "ea",
                     "from": {"component": "supply", "port": "out"},
                     "to": {"component": "a", "port": "input"}}],
    "indicators": [{"name": "a_got", "target": "attribute",
                    "attr": {"component": "a", "attribute": "got"}}],
})

policy = pyraichu.FlowConfig()
print(policy)

# Every knob left unset keeps the engine default, so these two runs are
# the same run, down to the counted work.
default = pyraichu.simulate(network, t_max=1.0)
spelled_out = pyraichu.simulate(network, t_max=1.0, flow=pyraichu.FlowConfig())
assert default.work == spelled_out.work
print(f"served: {default.indicators['a_got'][-1][1]:.3f}")
```

Undamped, that regulator alternates forever between asking for 6 and
asking for nothing: its linearised multiplier is `-1.5`. The engine
detects the alternation and damps it at `relaxation = 0.5`, which
contracts onto `6/2.5 = 2.4`. Turn the damping off and the same model is
**refused**, with a diagnostic naming the flow that was oscillating:

```python
def refusal(knobs):
    try:
        pyraichu.simulate(network, t_max=1.0, flow=knobs)
    except pyraichu.SimulationError as failure:
        return str(failure)
    return ""

for knobs in (pyraichu.FlowConfig(relaxation=1.0),
              pyraichu.FlowConfig(sweep_budget=2)):
    message = refusal(knobs)
    assert message, f"{knobs} settled: the knob never reached the engine"
    print(f"{knobs}\n  refused: {message}")
```

Reach for these knobs in two situations, and rarely otherwise. A network
the engine refuses may settle under a larger `sweep_budget`, or under a
`tolerance` loosened to the accuracy the study actually needs. And a
network you suspect of settling on the wrong answer can be squeezed:
tightening the budgets turns a slow, silent creep into a diagnostic that
names the edges still moving.

Tightening `tolerance` below the event-location tolerance (`tol_event`,
`1e-10` by default) is the one setting to avoid: the resolution then
promises the flows more precisely than a boundary crossing can be
located, and a freshly resolved network can re-cross its own boundary on
the spot.

## When a run never ends

A simulation that grinds is worse than one that fails: the trajectory
reads correctly at every sample instant while it happens, so the only
symptom is the wall clock.

Two Zeno guards have always caught a loop that does **not** advance time
(`WatchedLoop` for transitions, `FlowChattering` for the flow active
set). Neither can see a **limit cycle**, where the clock does move, by a
little, every turn. Two budgets close that:

| Knob | Caps | Default |
|---|---|---|
| `max_transition_firings` | how many times ONE transition may fire in a trajectory | 100 000 |
| `max_flow_restarts` | how many segments the flow network may restart in a trajectory | 100 000 |

Both fail with the culprit named and the **mean simulated step** between
its turns. That number is the diagnosis: microseconds between firings of
a bound automaton is a numerical scale, and no model of a plant means it.

```
transition `T.buffer_bounds.buffer_reach_full` fired 100000 times
between t=4.000000000000002 and t=4.799998091533782
(average step 8.0e-6): the model is chattering, not evolving.
```

Set either to `0` to lift it: a genuinely fast-switching model is a
legitimate thing to want, and a cap that could not be raised would be a
limit on what may be modelled rather than a diagnostic.

The commonest cause is not a numerical setting at all. A volume that
declares no pass-through demand asks for nothing once full, falls below
its bound by the hysteresis width, claims its fill rate again and
refills: declare `var_demand_default` on its inlet and the cycle
disappears. The second commonest is a guard comparing a **rate**, whose
value depends on the decision the guard makes.

## Choosing a setting

- **Keep the defaults** for correctness-critical work, small models, or
  when you are unsure: they are safe.
- **Relax `rtol`/`tol_event` and raise `max_step`** for large hybrid
  Monte-Carlo campaigns where the ODE is smooth and a `1e-5`-level
  accuracy is ample.
- Always record the setting: it rides in the run's provenance, so a
  result is never ambiguous about how it was computed.
