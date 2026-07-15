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
you.

## The object catalogue

Beyond `ObjFlow`, the plugin system provides four object families, each
expanding deterministically to core components:

**`ObjFM`** — a failure mode over one or several target components, with
**common-cause orders**: per-order law lists generate one automaton per
target combination of each active order (`fm__cc_1_2`, …), every
combination drawing independently. An active failure with an inactive
(`null`) repair is a **non-repairable** mode — the failure state is
absorbing. Three behaviours:

- `internal` (default) — the mode writes the targets' attributes
  directly (held at the failure value while any impacting combination is
  failed, the initial value otherwise);
- `external` — a mutual lock: a control attribute drives a mirror
  automaton grafted into each target; a combination can only (re)fail
  once its targets are repaired, and vice-versa;
- `external_rep_indep` — a trigger model: the mode resets instantly and
  each target latches the failure until it repairs on its own law.

**`ObjFMInst`** — failure *on solicitation*: one Bernoulli draw per
demand front (probability `gamma` per common-cause order), exponential
repair; the anti-Zeno re-arm guarantees one draw per front.

**`ObjEvent`** — a monitored event over a condition tree, with
occurrence/clearance tempos (a repair during the tempo cancels the
pending occurrence). Flagged `"target": true`, it becomes a feared event
for [sequence analysis](sequence-analysis.md).

**`ObjLogicGate`** — an automaton-free combinational gate over condition
leaves: `or`, `and`, or k-of-n voting, recomputed edge-triggered on any
input change; several `out_elements` broadcast the same result.

```json
{"type": "ObjLogicGate", "name": "vote", "kind": "k", "k": 2,
 "cond": [[{"obj": "A", "attr": "ok"}], [{"obj": "B", "attr": "ok"}],
          [{"obj": "C", "attr": "ok"}]],
 "out_elements": ["ok"]}
```

Models exported from a COD3S platform instance translate directly into
these objects — see [Importing platform studies](platform-import.md).

## Which to use

- **Native data** (the tutorial) — full generality: any automaton, ODE
  or watched transition.
- **muscadet builder** — flow/reliability networks in a fraction of the
  code.
- **Plugins** — the same high-level objects when your model comes from a
  file or another tool.

They interoperate: a plugin section and hand-written components can coexist
in the same model.
