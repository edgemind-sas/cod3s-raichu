# Numerical tuning

When a model has continuous dynamics, the ODE integrator's effort is
**explicit and adjustable** — not a hidden constant. This lets you trade
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
[accuracy–cost parity benchmark](../benchmarks/accuracy-cost-parity.md)
measures that ~10⁻⁵ setting running an order of magnitude faster than
the default — and the default, in turn, delivers 3–4 orders of magnitude
more accuracy than a typical study can use.

## Choosing a setting

- **Keep the defaults** for correctness-critical work, small models, or
  when you are unsure — they are safe.
- **Relax `rtol`/`tol_event` and raise `max_step`** for large hybrid
  Monte-Carlo campaigns where the ODE is smooth and a `1e-5`-level
  accuracy is ample.
- Always record the setting: it rides in the run's provenance, so a
  result is never ambiguous about how it was computed.
