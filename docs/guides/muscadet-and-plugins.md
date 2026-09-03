# muscadet authoring layer & plugins

The [tutorial](../tutorial/02-connecting-components.md) authors models as
explicit data: the native, fully general form. For **flow /
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
functions and automata the tutorial wrote by hand: you can inspect them
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
translation is always auditable: nothing the plugin does is hidden from
you.

## The object catalogue

### `ObjFlow`

The plugin peer of the builder's component, and the only object that
carries a **conserved quantity**. Its boolean sections (`flows_in`,
`flows_out`, `failure_modes`) read the flat vocabulary shown above.

Its **continuous** sections read muscadet's own declaration vocabulary,
key for key: the vocabulary `pyraichu.declare` reads, so a key one entry
point accepts and the other refuses does not exist.

| Section | Declares |
|---|---|
| `flows_continuous_in` | a real-valued input: `var_in_default` (what it reads unconnected), `var_demand_default` (what a pure consumer asks for) |
| `flows_continuous_out` | a real-valued output: `var_fed_default`, a `profile` (a declared function of time), and the `allocation` policy splitting a shortage (`proportional`, `shares`, `priority`) |
| `capacities` | a volume over one or more held flows: `capacity`, `content_init`, `fill_rate`, `side`, `hysteresis` |
| `measurements_in` | the reading side of a measurement link: a channel observing a published level, carrying no quantity |
| `rules` | an ordered set of transformation rules (`cond` / `cons` / `prod`), running at the scale its scarcest input and least demanded output allow |
| `transfers` | a transfer pair: a quantity moved because a gradient drives it, under a `ConductiveTransfer` equation |

<!-- model -->
```json
{
  "name": "continuous_demo",
  "connections": [
    {"from": {"component": "WELL", "port": "water_out"},
     "to": {"component": "TANK", "port": "water_in"}},
    {"from": {"component": "TANK", "port": "water_out"},
     "to": {"component": "TOWN", "port": "water_in"}},
    {"from": {"component": "TANK", "port": "water_out"},
     "to": {"component": "FARM", "port": "water_in"}}
  ],
  "plugins": {
    "muscadet": {
      "objects": [
        {"type": "ObjFlow", "name": "WELL",
         "flows_continuous_out": [
           {"name": "water", "var_fed_default": 6.0,
            "profile": {"cls": "SinusoidalProfile",
                        "amplitude": 0.4, "period": 24.0, "offset": 0.6}}]},
        {"type": "ObjFlow", "name": "TANK",
         "flows_continuous_in": [{"name": "water"}],
         "flows_continuous_out": [
           {"name": "water", "var_fed_default": 5.0,
            "allocation": "shares",
            "allocation_shares": {"TOWN": 0.8, "FARM": 0.2}}],
         "capacities": [
           {"name": "vol", "flow": "water", "capacity": 200.0,
            "content_init": {"water": 80.0}, "fill_rate": 1.0}]},
        {"type": "ObjFlow", "name": "TOWN",
         "flows_continuous_in": [
           {"name": "water", "var_demand_default": 4.0}]},
        {"type": "ObjFlow", "name": "FARM",
         "flows_continuous_in": [
           {"name": "water", "var_demand_default": 2.0}]}
      ]
    }
  }
}
```

A whole continuous model is therefore writable as data, controllers
included, and the document it expands to is **the same one** the builder
writes for the same model: the plugin hands the declarations to a
`System` and calls the generation the builder calls.

#### The model-level pass

The continuous constructs are not component-local, and that is why they
are expanded in two steps. What a producer publishes to one consumer is
what remains once the *other* consumers are accounted for; an allocation
operator splits over the connections it serves; the sweep order runs
along the flow graph. None of that is knowable while an object is
expanded on its own, before the objects after it exist.

So a plugin may implement an optional `finalize_model(model, specs)`,
called once, after every object of every plugin has been expanded, with
the whole model and its own object list. The muscadet plugin emits the
connection-dependent material there.

One consequence is worth knowing: the continuous network **derives** the
`evaluation_order`, and closes it over every explicit equation and every
allocation the model declares, controllers included. A model that both
declares continuous flows and asserts an `evaluation_order` of its own
has two authorities on one sweep, and is refused rather than silently
overridden.

### The other object families

Beyond `ObjFlow`, the plugin system provides five object families, each
expanding deterministically to core components:

**`ObjFM`**: a failure mode over one or several target components, with
**common-cause orders**: per-order law lists generate one automaton per
target combination of each active order (`fm__cc_1_2`, …), every
combination drawing independently. An active failure with an inactive
(`null`) repair is a **non-repairable** mode: the failure state is
absorbing. Three behaviours:

- `internal` (default): the mode writes the targets' attributes
  directly (held at the failure value while any impacting combination is
  failed, the initial value otherwise);
- `external`: a mutual lock: a control attribute drives a mirror
  automaton grafted into each target; a combination can only (re)fail
  once its targets are repaired, and vice-versa;
- `external_rep_indep`: a trigger model: the mode resets instantly and
  each target latches the failure until it repairs on its own law.

**`ObjFMInst`**: failure *on solicitation*: one Bernoulli draw per
demand front (probability `gamma` per common-cause order), exponential
repair; the anti-Zeno re-arm guarantees one draw per front.

**`ObjEvent`**: a monitored event over a condition tree, with
occurrence/clearance tempos (a repair during the tempo cancels the
pending occurrence). Flagged `"target": true`, it becomes a feared event
for [sequence analysis](sequence-analysis.md).

**`ObjLogicGate`**: an automaton-free combinational gate over condition
leaves: `or`, `and`, or k-of-n voting, recomputed edge-triggered on any
input change; several `out_elements` broadcast the same result.

```json
{"type": "ObjLogicGate", "name": "vote", "kind": "k", "k": 2,
 "cond": [[{"obj": "A", "attr": "ok"}], [{"obj": "B", "attr": "ok"}],
          [{"obj": "C", "attr": "ok"}]],
 "out_elements": ["ok"]}
```

**`ObjCtrl`**: a **controller**, the peer of `ObjFlow` that carries a
reading or a signal instead of a conserved quantity. It declares
observation inputs (`controls_in`, a capacity level, a delivered rate or
a constituent's share, optionally reduced over several publishers by
`sum`, `mean` or `median`) and control outputs (`controls_out`, a boolean
signal or a published number), and each output's value is composed from a
**closed grammar** of four operators: `compare` (a reading against a
threshold), `band` (two thresholds and a direction: a hysteresis band),
`combine` (`and` / `or` / `not` / k-of-n) and `republish` (a reading,
times a gain).

Every threshold compiles to a **watched** two-state automaton, so a
crossing is located by root-finding rather than noticed at the next
discrete event. `band` is the operator that carries memory: a comparison
switches back the instant its condition stops holding, so a montage gated
on one chatters around a single level, while a band holds between its two
edges. That is what makes a two-threshold regulation expressible.

```json
{"type": "ObjCtrl", "name": "LOW",
 "controls_in": [{"name": "tank", "kind": "level"}],
 "controls_out": [{"name": "run", "kind": "bool",
                   "emit": {"op": "band", "input": "tank",
                            "direction": "below",
                            "activate": 6.0, "release": 8.0}}]}
```

Wired to a capacity's `tank_level_out` and to a pump's `run_in`, that is
the heated-tank regulation: the pumps run below 6 and stop above 8. Every
number the grammar carries is an attribute of the model
(`run_activate`, `run_release`, `{output}_level_gain`,
`{output}_forced`, `{output}_forced_value`,
`{output}_signal_available`), so an instance can be tuned away from its
declaration, an indicator can name a threshold, and an `ObjFM` can move
one or blind an output.

Models exported from a COD3S platform instance translate directly into
these objects: see [Importing platform studies](platform-import.md).

## Which to use

- **Native data** (the tutorial): full generality: any automaton, ODE
  or watched transition.
- **muscadet builder**: flow/reliability networks in a fraction of the
  code.
- **Plugins**: the same high-level objects when your model comes from a
  file or another tool. Every construct the builder offers is declarable
  here, continuous flows included, so a whole model is data.

They interoperate: a plugin section and hand-written components can coexist
in the same model. The one boundary is the continuous network, which is
resolved over the components the plugin declares: a continuous connection
crossing into a hand-written component is refused, naming it, because the
quantity it carries would be accounted for nowhere.
