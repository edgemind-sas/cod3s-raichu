# Migration guide (from PyCATSHOO)

This guide migrates two models from the PyCATSHOO Python API to RAICHU,
construct by construct. A RAICHU model is **data** — the Python `dict`
you build *is* the native JSON the engine consumes, with no hidden layer
— so "the Python authoring" and "the generated JSON" are the same thing.
We use the [concept mapping](concept-mapping.md) as the dictionary.

Two threads, increasing in coupling:

- **Thread A** — a discrete master/slave system (message boxes,
  references, sensitive methods, conditions).
- **Thread B** — a hybrid heated room (adds the PDMP manager and watched
  transitions).

## Thread A — discrete master/slave

A slave unit delivers only while the master is failed. In PyCATSHOO you
subclass `CComponent`, add attributes, an automaton, a message box, a
reference and a sensitive method:

<!-- skip -->
```python
# PyCATSHOO
class Master(Pyc.CComponent):
    def __init__(self, name):
        Pyc.CComponent.__init__(self, name)
        self.aut = self.addAutomaton("health")
        self.ok = self.addState("health", "OK", 1)
        self.ko = self.addState("health", "KO", 0)
        self.setInitState("OK")
        t = self.ok.addTransition("fail")
        t.setDistLaw(Pyc.IDistLaw.newLaw(self, Pyc.TLawType.expo, 0.05))
        t.addTarget(self.ko, Pyc.TTransType.fault)
        # … repair …
        mb = self.addMessageBox("toSlave")
        mb.addExport(self.ko, "request")     # export the KO state
```

The same model in RAICHU. The **message box → ports**, the exported
state becomes an **out-port on a boolean attribute** (kept in sync by a
sensitive function), and the automaton is plain data:

```python
import pyraichu

master = {
    "name": "Master",
    "attributes": [{"name": "ko", "kind": "bool",
                   "init": {"kind": "bool", "value": False}}],
    "ports": [{"name": "ko_out", "dir": "out", "attr": "ko"}],
    "automata": [{"name": "health", "states": ["OK", "KO"], "init": "OK",
        "transitions": [
            {"name": "fail", "source": "OK", "targets": ["KO"],
             "distrib": "exp", "rate": 0.05},
            {"name": "repair", "source": "KO", "targets": ["OK"],
             "distrib": "exp", "rate": 0.1}]}],
    "sensitive_functions": [{"name": "update_ko", "effects": [{
        "target": {"component": "Master", "attribute": "ko"},
        "value": {"op": "state_active",
                  "state": {"component": "Master", "automaton": "health",
                            "state": "KO"}}}]}],
}
```

On the slave side, a PyCATSHOO **reference** read with `orValue`, and a
**condition method**, become an **in-port** read with `port_agg` and a
transition **guard**:

<!-- skip -->
```python
# PyCATSHOO
class Slave(Pyc.CComponent):
    def __init__(self, name):
        ...
        self.request = self.addReference("request")
        mb = self.addMessageBox("toMaster")
        mb.addImport(self.request, "request")
        t = self.off.addTransition("start")
        t.setCondition(self.requested)          # a Python condition method
    def requested(self):
        return self.request.orValue(False)      # any connected source true
```

```python
slave = {
    "name": "Slave",
    "attributes": [{"name": "power", "kind": "float",
                   "init": {"kind": "float", "value": 0.0}}],
    "ports": [{"name": "request_in", "dir": "in"}],
    "automata": [{"name": "functional", "states": ["OFF", "ON"], "init": "OFF",
        "transitions": [
            {"name": "start", "source": "OFF", "targets": ["ON"],
             "distrib": "inst", "probs": [],
             "guard": {"op": "port_agg",
                       "port": {"component": "Slave", "port": "request_in"},
                       "agg": "any"}},
            {"name": "stop", "source": "ON", "targets": ["OFF"],
             "distrib": "inst", "probs": [],
             "guard": {"op": "bool", "bool_op": "not", "args": [
                 {"op": "port_agg",
                  "port": {"component": "Slave", "port": "request_in"},
                  "agg": "any"}]}}]}],
    "sensitive_functions": [{"name": "update_power", "effects": [{
        "target": {"component": "Slave", "attribute": "power"},
        "value": {"op": "if",
            "cond": {"op": "state_active",
                     "state": {"component": "Slave", "automaton": "functional",
                               "state": "ON"}},
            "then": {"op": "const", "value": {"kind": "float", "value": 5.0}},
            "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.0}}}}]}],
}
```

Finally the `connect` call becomes a **connection**, and the model runs:

```python
model = pyraichu.load_model({
    "name": "master_slave",
    "components": [master, slave],
    "connections": [{"from": {"component": "Master", "port": "ko_out"},
                     "to": {"component": "Slave", "port": "request_in"}}],
    "indicators": [{"name": "slave_on", "target": "state",
                    "component": "Slave", "automaton": "functional", "state": "ON"}],
})

est = pyraichu.monte_carlo(model, nb_runs=2000, t_max=100.0,
                           samples=[10.0 * k for k in range(11)], seed=1)
print("slave running:", [round(v, 3) for v in est.indicators["slave_on"].mean])
```

The slave runs ~33 % of the time — exactly the master's unavailability
(0.05/0.15), since it is on precisely when the master is down.

**Construct map used here:**

| PyCATSHOO | RAICHU |
|---|---|
| `addMessageBox` + `addExport` | out-port with `var` |
| `addReference` + `addImport` | in-port |
| `reference.orValue(...)` | `port_agg … "agg": "any"` |
| condition method | transition `guard` (expression) |
| sensitive method | sensitive function (`effects`) |
| `system.connect(...)` | a `connections` entry |

## Thread B — the hybrid heated room

The hybrid case adds two PyCATSHOO constructs: the **PDMP manager**
(ODE) and the **boundary checker / watched transition**.

<!-- skip -->
```python
# PyCATSHOO
pdmp = self.addPDMPManager("mgr")
pdmp.addODEVariable(self.temperature)
pdmp.addEquationMethod("dT", self)          # a Python method computing dT/dt
def dT(self):
    self.temperature.setDvdtODE(self.power.sumValue(0)
                                - 0.1 * (self.temperature.value() - 13))
# thermostat as a watched transition:
pdmp.addWatchedTransition(self.on_to_off)   # guarded by a boundary condition
```

In RAICHU the PDMP manager disappears: a component simply declares an
**`ode` equation**, and the thermostat is a **`watched` transition**
whose guard is a boundary comparison. Both are data:

```python
room = {
    "name": "Room",
    "attributes": [{"name": "temperature", "kind": "float",
                   "init": {"kind": "float", "value": 17.0}}],
    "ports": [{"name": "power_in", "dir": "in"},
              {"name": "temp_out", "dir": "out", "attr": "temperature"}],
    "equations": [{"target": "temperature", "kind": "ode",
        "expr": {"op": "sub",
            "lhs": {"op": "port_agg",
                    "port": {"component": "Room", "port": "power_in"}, "agg": "sum"},
            "rhs": {"op": "mul", "args": [
                {"op": "const", "value": {"kind": "float", "value": 0.1}},
                {"op": "sub",
                 "lhs": {"op": "attr",
                         "attr": {"component": "Room", "attribute": "temperature"}},
                 "rhs": {"op": "const", "value": {"kind": "float", "value": 13.0}}}]}}}],
}

def threshold(op, value):        # a boundary on the incoming temperature
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

hybrid = pyraichu.load_model({
    "name": "heated_room", "components": [heater, room],
    "connections": [
        {"from": {"component": "H", "port": "power_out"},
         "to": {"component": "Room", "port": "power_in"}},
        {"from": {"component": "Room", "port": "temp_out"},
         "to": {"component": "H", "port": "temp_in"}}],
    "indicators": [{"name": "temp", "target": "attribute",
                    "attr": {"component": "Room", "attribute": "temperature"}}],
})

run = pyraichu.simulate(hybrid, t_max=100.0, seed=1,
                        samples=[20.0 * k for k in range(6)])
print("temperature:", [round(v, 1) for _, v in run.samples["temp"]])
```

The thermostat holds the room in its 15–20 band.

**Construct map used here:**

| PyCATSHOO | RAICHU |
|---|---|
| `addPDMPManager` | *(none — declare equations directly)* |
| `addODEVariable` + `setDvdtODE` | `equations: [{kind: "ode", …}]` |
| `sumValue` on a power reference | `port_agg … "agg": "sum"` |
| boundary checker / `addWatchedTransition` | `"distrib": "watched"` + a `guard` comparison |
| `setDtCond` | `tol_event` (and the other [tolerances](../guides/numerical-tuning.md)) |

## What changes, beyond syntax

- **No callbacks in the loop.** PyCATSHOO's condition and equation
  *methods* are Python, called by the engine during the run. RAICHU's
  guards and equations are **expression trees** evaluated natively — the
  reason the [hybrid benchmark](../benchmarks/performance.md) is so much
  faster.
- **Build-time validation.** An unknown state or a malformed distribution is a
  precise `ModelError` at `load_model`, not a crash mid-run.
- **The model is inspectable data.** `model.json` is the whole model;
  diff it, store it, generate it.

For the full modelling vocabulary, start at the
[tutorial](../tutorial/01-first-model.md); for every field, the
[schema reference](../reference/model-schema.md).
