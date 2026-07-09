# muscadet authoring layer & plugins

The [tutorial](../tutorial/02-connecting-components.md) authors models as
explicit data — the native, fully general form. For **flow /
reliability-network** models (sources, lines, loads, failure modes,
redundancy), that is more verbose than it needs to be. The
`pyraichu.muscadet` layer is a thin, higher-level builder that generates
the same native models from a few lines of Python.

## The builder

You subclass `ObjFlow` and declare *flows*: a **flow in** aggregates its
suppliers, a **flow out** is produced when its condition holds and the
component has not failed. `add_exp_failure_mode` / `add_delay_failure_mode`
attach failure/repair behaviour. A `System` wires flows by name and runs
the model:

```python
import pyraichu.muscadet as mu

class Source(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_out(name="power", var_prod_default=True)

class Line(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name="power")
        self.add_flow_out(name="power", var_prod_cond=["power"])

class Load(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name="power")          # default "or" = redundant

system = mu.System("grid")
for cls, name in [(Source, "S"), (Line, "L1"), (Line, "L2"), (Load, "D")]:
    system.add_component(cls, name)

for line in ("L1", "L2"):
    system.comp[line].add_exp_failure_mode(
        name="fault", failure_rate=0.02, repair_rate=0.1,
        failure_cond="power_fed_out")

system.connect("S", "power", "L1", "power")
system.connect("S", "power", "L2", "power")
system.connect("L1", "power", "D", "power")
system.connect("L2", "power", "D", "power")

estimates = system.monte_carlo(
    nb_runs=2000, t_max=100.0, samples=[10.0 * k for k in range(11)], seed=1)
print("load powered:", [round(v, 3) for v in
                        estimates.indicators["D_power_fed_in"].mean])
```

The two redundant lines keep the load powered ~97 % of the time. The
builder created, under the hood, exactly the kind of ports, sensitive
functions and automata the tutorial wrote by hand — you can inspect them
with `system.build_dict()`. An `add_flow_in(name=…, logic="and")` (or an
integer *k* for k-out-of-n) changes the aggregation; `add_flow_out_tempo`
and `add_flow_out_on_trigger` add delayed and inhibition-driven flows.

## Plugins: the same objects as data

The same high-level objects can be expressed as **pure JSON**, in a
`"plugins"` section of a model, and expanded to the core schema by
`load_model` (or inspected with `expand_model`). This suits
config-driven or language-agnostic authoring:

<!-- model -->
```json
{
  "name": "plugin_demo",
  "plugins": {
    "muscadet": {
      "objects": [
        {"type": "ObjFlow", "name": "S",
         "flows_out": [{"name": "ok", "var_prod_default": true}]},
        {"type": "ObjFlow", "name": "B",
         "flows_in": [{"name": "ok"}],
         "flows_out": [{"name": "ok", "var_prod_cond": ["ok"]}],
         "failure_modes": [{"name": "fault", "distrib": {"distrib": "exp"},
                            "failure": 0.02, "repair": 0.1,
                            "failure_cond": "ok_fed_out"}]}
      ]
    }
  },
  "connections": [
    {"from": {"component": "S", "port": "ok_out"},
     "to": {"component": "B", "port": "ok_in"}}
  ]
}
```

`expand_model(spec)` returns the plugin-free core model, so the
translation is always auditable — nothing the plugin does is hidden from
you. Beyond `ObjFlow`, the plugin system provides **`ObjFM`** (failure
modes with common-cause orders) and **`ObjEvent`** (monitored events with
delays), each expanding deterministically to core components.

## Which to use

- **Native data** (the tutorial) — full generality: any automaton, ODE
  or watched transition.
- **muscadet builder** — flow/reliability networks in a fraction of the
  code.
- **Plugins** — the same high-level objects when your model comes from a
  file or another tool.

They interoperate: a plugin section and hand-written components can coexist
in the same model.
