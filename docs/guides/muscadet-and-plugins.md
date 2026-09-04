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
| `flows_continuous_in` | a real-valued input: `var_in_default` (what it reads unconnected), `var_demand_default` (what a pure consumer asks for), a `profile` scaling that demand over time |
| `flows_continuous_out` | a real-valued output: `var_fed_default`, a `max_rate` ceiling, a `profile` (a declared function of time), and the `allocation` policy splitting a shortage (`proportional`, `shares`, `priority`) |
| `capacities` | a volume over one or more held flows: `capacity`, `content_init`, `fill_rate`, `side`, `hysteresis`. A volume holding more than one flow also publishes each constituent's `ratio`, its fraction of the mixture |
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

#### Three quantities a volume publishes per constituent

| Published | Divided by | Answers |
|---|---|---|
| `{cap}_level_{flow}` | nothing | how much of it is held |
| `{cap}_fill_{flow}` | the declared volume | how full the vessel is |
| `{cap}_ratio_{flow}` | what the vessel holds | what the **mixture** is |

The third is the one a flammability threshold is written on: two per cent
of hydrogen in a room means two per cent of what the room holds, not two
per cent of the room. A controller cannot compute it, its output grammar
being closed at four operators with no arithmetic among them, so a
fraction a controller can threshold is one the volume publishes.

Only a volume holding **more than one** constituent publishes ratios: a
single-flow volume's ratio is identically one wherever it holds anything.
An **empty** volume reads 0 on every ratio, nothing being no fraction of
nothing.

#### A ceiling on what an output can deliver

`max_rate` says the equipment cannot make more than that per unit time,
whatever it is fed and whatever is asked of it. It has no muscadet
counterpart and it is not a failure-mode cap: a cap is a **fraction** of
nominal owned by the mode that declares it, this is an **absolute
quantity** that stands for the whole run. The two compose by minimum, as
two ceilings do.

On an output a rule set produces, the ceiling bounds the **scale the rule
runs at** as well as the quantity delivered, which is the half that
matters: bounded only at the output, the component would go on drawing
the full quantity from its suppliers while delivering the lesser quantity
its ceiling allows, and the difference would vanish inside it. An
electrolyser capped at 0.4 therefore asks its water and its power supply
for what 0.4 of hydrogen needs, and no more.

A rule's outputs are correlated, so a ceiling on one holds the others
down in proportion rather than letting them produce a surplus with
nowhere to go.

#### What a failure mode does to a continuous output

A `failure_modes` entry may carry `failure_effects` and `repair_effects`,
which name continuous outputs by regular expression and say what the mode
does to them. There are two things it can do, and they are not the same:

| Spelling | Meaning | Simultaneous effects compose by |
|---|---|---|
| `0.8`, or `{"cap": 0.8}` | a **cap**: what the mode LEAVES of the output | **minimum** |
| `{"tap": 0.2}` | a **tap**: what fraction it TAKES OFF the output | **sum** |
| `{"tap": 0.2, "to": "vent"}` | a routed tap: what it takes is delivered on another output of the component | **sum** |

The distinction is not cosmetic, it is the arithmetic. Two caps of 0.9
and 0.8 leave **0.8**: a cap is a constraint, and the binding one wins,
which is what a derated capacity means. Two taps of 0.1 and 0.2 leave
**0.7**: taps are parallel draws on one stream, so what leaves by one
does not pass the other. Folding taps by minimum would leave 0.8 and make
the second leak free; folding caps by sum would invent a limit neither
component has.

Nothing in the number distinguishes the two, which is why the spelling
carries it. A bare number is a cap, the 1.x reading, so an existing
declaration means exactly what it always did.

A **routed** tap is what closes the mass balance: the fraction is moved
to another continuous output of the same component instead of vanishing,
and both sides divide one published quantity, `{flow}_pre_tap_*`, what
the stream carried before its taps drew on it. Connect that output to
wherever the fraction goes next. Without `to`, the fraction leaves the
system, which is a loss to the environment and a legitimate model.

```json
{"type": "ObjFlow", "name": "STACK",
 "flows_continuous_out": [{"name": "H2", "var_fed_default": 10.0},
                          {"name": "H2_vent"}],
 "failure_modes": [
   {"name": "membrane_leak", "distrib": "delay",
    "failure": 100.0, "repair": 1e9,
    "failure_effects": {"H2": {"tap": 0.1, "to": "H2_vent"}}}]}
```

The demand channel is **not** divided by the tap rate: a plant that does
not know about its leak produces what it was asked for and delivers less,
and the shortfall is the consumer's. Making up the difference is
regulation, declared as a controller.

Four refusals keep a tap conserving, all at build time: a route naming
something that is not a continuous output of the component, a route that
loops, taps that could together take more than the stream carries, and a
receiving output that also produces something of its own.

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
