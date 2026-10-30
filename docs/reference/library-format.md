# Component type library format

A model can be written without code as three data documents:

| Document | Envelope | Holds |
|---|---|---|
| Library | `raichu_library` | parameterised component types, built from core constructs and named fragments, each with the catalogue of its failure and compromise modes |
| System | `raichu_system` | instances of library types, by name, and the connections between their ports |
| Study | `raichu_study` | which catalogued modes run, on which instances, under which laws; conditions across instances; feared events, indicators and run settings |

The engine expands the three into the flat core model of the
[model schema reference](model-schema.md), which it then validates,
compiles and simulates exactly as it does a model written directly. The
expanded model records the library, its version and the documents it came
from, and every element of it is traced back to the document node that
produced it.

Each document is a JSON object holding its envelope and its body, and
nothing else:

```json
{"raichu_library": {"format": 1}, "library": {"name": "plant", "version": "1.0.0"}}
```

Every object of the three documents refuses a field it does not know, and
the refusal names the field with its JSON Pointer. This holds inside type
bodies too: a transition spelling `gaurd` for `guard` is refused, where the
core reader would drop it silently. Every object that names something also
accepts a free `metadata` value (an editor's layout, an author's comments),
which the expander never reads, and a `description` text.

The JSON Schemas of the three documents are published beside this page
(`reference/schema/raichu-library.schema.json`,
`raichu-system.schema.json`, `raichu-study.schema.json`, draft 2020-12),
derived from the reader's own definitions; `pyraichu.library.schema("study")`
returns the copy shipped with the installed engine.

## Names

Instance, type, fragment, mode, feared-event and library names may hold any
character except `.`, `$` and control characters. Spaces and accents are
fine (`Algo Feu`, `Véhicule`). The `.` is reserved because a mode and a port
are named publicly as `Instance.mode` and `Instance.port`; the `$` is
reserved for the templating grammar.

## Library

| Field | Type | Meaning |
|---|---|---|
| `name` | string | library name; it prefixes its fragments when another library imports them (`std.bool_flow_in`) |
| `version` | string | matched exactly by the systems that use the library |
| `imports` | list of strings | libraries whose fragments this one applies (`["std"]`) |
| `fragments` | list | named, parameterised pieces of behaviour |
| `types` | list | component types |

### Types and fragments

| Field | On | Meaning |
|---|---|---|
| `name` | both | unique in the library |
| `parameters` | both | declared parameters (below) |
| `uses` | both | fragment applications: `{"fragment": "std.bool_flow_in", "with": {"name": "power"}}` |
| `body` | both | core constructs, templated: `attributes`, `ports`, `interfaces`, `automata`, `sensitive_functions`, `equations`, `allocations`, in the shapes of the model schema |
| `targetable` | both | attributes a mode may target, each with its combination rule |
| `modes` | type | the mode catalogue |

A type's component is the union of its own body and the bodies of every
fragment it applies (fragments may apply fragments). A name two of them
declare (an attribute, a port, an automaton, a function, the target of an
equation) is refused with both origins; a fragment applying itself, even
through others, is refused with the chain. There is no override.

A type body refers to its own instance as `$self`, and only to it: a
neighbour is reached through a port.

### Parameters

| Field | Meaning |
|---|---|
| `name` | parameter name |
| `kind` | `number`, `integer`, `bool`, `string`, `law`, `list` or `enum` |
| `default` | default value; a parameter without one is required |
| `values` | the allowed values of an `enum` |
| `items` | the item kind of a `list`: `number`, `integer`, `bool`, `string`, `string_list` or `object` |

A `law` is a core law object (`{"distrib": "exp", "rate": 0.1}`). A list of
`object` items is inserted as written where `{"$param": ...}` substitutes it,
and the core reader then checks each item at the place it lands: this is
how a split's shares, keyed by consumer, reach a distribution operator
(`[{"to": {"component": "B1", "port": "Elec_in"}, "value": 0.1}]`). An instance
giving a parameter its type does not declare, omitting a required one,
giving a value of the wrong kind, or choosing an `enum` value outside its
list, is refused naming the instance and the parameter.

### Templating

Inside a body, a `uses` value, a targetable entry or a catalogued mode,
four forms are resolved against the instance's parameter values:

| Form | Effect |
|---|---|
| `{"$param": "capacity"}` | the parameter's value, anywhere a value goes |
| `"${flow}_fed_out"` | interpolates a string, number or boolean parameter, or a `$for_each` variable, into a string; an integral number reads without a fraction (`2`, not `2.0`) |
| `{"$for_each": "flow_types", "as": "f", "items": [...]}` | as an item of an array: repeats `items` once per element of a list (or of a loop variable that is itself a list) |
| `{"$law": "failure", "name": "fail", "source": "ok", "targets": ["ko"]}` | the keys of a `law` parameter (`distrib`, `rate`, ...) written onto the object beside its other keys, the way a transition carries its law; a key both write is refused |
| `{"$choose": "method", "cases": {"mean": ..., "median": ...}}` | the case named by an `enum`, `string` or `bool` parameter (`"true"`, `"false"`) |
| `{"$when": "ccf", "items": [...]}`, `{"$when": "rule", "is": ["printed", "tables"], "items": [...]}` | as an item of an array: keeps `items` in place when the parameter's value is one of `is` (a value or a list), drops them otherwise; without `is`, the parameter is a `bool` and `items` are kept when it is true. `$choose` replaces a whole value; `$when` keeps or drops one item among its siblings (an optional transition). A value in `is` that an `enum` or `bool` parameter never takes is refused, and the items are expanded even when dropped, so a misspelling or an error in them is reported on every instance |

The string `"$self"` names the instance, and `"$self.x"` reads
`"<instance>.x"`. Nothing else computes at expansion: parameters reduce to
literals, and arithmetic stays in runtime expressions.

A body reaches its neighbours through its ports only. A
[port equation](model-schema.md#port-equation) is the one place a body may
also read the component `$edge`: the other channels of the connection the
equation is evaluated for. A transition's edge effect may also target the component
`$port`, which [writes through an in port](model-schema.md#writing-through-a-port)
to the neighbours connected to it: that is how a body acts on a
neighbour.

### Mode catalogue

A catalogued mode declares what it does, never its law:

| Field | Meaning |
|---|---|
| `name` | mode name, unique on the type |
| `nature` | `failure` or `compromise`; it serves selection and reporting and changes no semantics |
| `effects` | `[{"attribute": "COCOFE_fed_available_out", "value": false}]`: the value held on a targetable attribute while the mode is occurred (a plain JSON boolean or number) |
| `edge_effects` | `[{"attribute": "detected_fed_available_out", "value": true, "on": "occurrence"}]`: a value written once on a targetable attribute when the mode occurs (`occurrence`) or is repaired (`repair`), then kept (below) |
| `condition` | the shape of its activation condition: a core expression over the instance (`$self`), which may also use `mode_active` (below) |

An edge effect is a one-shot write, the way a gate that memorises (a
detection, an alarm) is modelled: the attribute keeps the written value
until another edge writes it, and nothing restores it on the other edge. It
is written on the transition that carries the mode's effects: the mode's
own `occ` or `rep`, its effect step's when the activation has an
`effect_step`, and for a common-cause group the edges of every combination
covering the member (each member's step when the group has one). On an
on-demand draw the write lands only when the draw enters `occ`, never when
it parks. Refused by name: an attribute that is not targetable, a value the
attribute's kind cannot take, an attribute a held effect or the type's own
behaviour also writes (an attribute written on an edge is written by edges
only), a repair write on a mode with no repair law, and writes on both
edges of one attribute by overlapping combinations of a group without an
effect step (one combination's repair would undo what another, still
occurred, wrote). Several modes may write one attribute on their edges.

### Combination rules

Every attribute a mode may target is listed under `targetable` with the rule
that combines the effects of several active modes:

| Rule | Attribute | Value while at least one targeting mode is occurred | Otherwise |
|---|---|---|---|
| `any_cuts` | bool | `false` | declared initial value |
| `any_activates` | bool | `true` | declared initial value |
| `min` | int, float | smallest value held by an occurred mode | declared initial value |
| `max` | int, float | largest value held by an occurred mode | declared initial value |

The nominal value takes no part in `min` or `max`. A rule may be templated
(`{"$choose": "fed_available_init", "cases": {"true": "any_cuts", "false":
"any_activates"}}`). A mode targeting an attribute with no rule is refused,
naming the attribute, the type and the mode; so is a rule that does not fit
the attribute's kind, an effect value the rule cannot hold (`true` under
`any_cuts`), and an attribute the type's own behaviour also writes: a
targeted attribute is written by its modes only. Effects are released on
repair.

## The standard library

`std` is written in this format and shipped with the engine; its version is
the engine's, and the expanded model records it. A library imports it with
`"imports": ["std"]`.

| Fragment | Parameters | Builds |
|---|---|---|
| `bool_flow_in` | `name`, `logic` (`and`, `or`, `k`), `k`, `default` | in port `<name>_in`, attribute `<name>_fed_in` true while the connected sources feed it under `logic`; an input nothing feeds reads `default` |
| `bool_flow_out` | `name`, `prod_cond`, `fed_available_init`, `negate` | out port `<name>_out` exporting `<name>_fed_out`, true while the production condition holds and `<name>_fed_available_out` is true; with `negate`, the negation of that delivery |
| `bool_source_out` | `name`, `prod_init`, `fed_available_init`, `negate` | the same output with no production condition: a constant source, or a dormant function when `prod_init` is false |
| `bool_flow_out_tempo` | `name`, `prod_cond`, `conditioned`, `prod_init`, `fed_available_init`, `init_enable`, `enable_law` (`delay`, `exp`), `enable_value`, `disable_law`, `disable_value` | a temporised output (muscadet `FlowOutTempo`): `<name>_fed_out` is true while the automaton `<name>_tempo` is `enabled` and `<name>_fed_available_out` holds; the transitions `<name>_enable` and `<name>_disable` wait on their law once the production condition holds or fails, and restart when it changes again first. `conditioned: false` leaves `<name>_prod_available` at `prod_init` |
| `bool_flow_out_trigger` | `name`, `prod_cond`, `conditioned`, `prod_init`, `fed_available_init`, `logic` (`and`, `or`, `k`), `k`, `time_up`, `time_down`, `armed` | a triggered output (muscadet `FlowOutOnTrigger`): `<name>_fed_out` is true while the automaton `<name>_trigger` is `up`, the production holds and `<name>_fed_available_out` holds. The trigger inhibits: `<name>_trigger_up` waits `time_up` once the aggregate over the in port `<name>_trigger_in` is false, `<name>_trigger_down` waits `time_down` once it is true (`and` over an unconnected port is false). `armed: true` starts the automaton `up`, which the declaration route does when nothing feeds the trigger and `time_up` is zero |
| `logic_gate` | `name`, `logic`, `k`, `default` | a boolean input re-exported on the out port `<name>_out` |
| `state_out` | `name`, `automaton`, `states`, `init` | whether `automaton` is in one of `states`, as the boolean `<name>` on the out port `<name>_out`: how a neighbour reads a state (`any`, `all`, or `sum` to count) |
| `bool_not` | `of`, `init` | the attribute `<of>_not`, the negation of the boolean attribute `<of>` of the instance: how a production condition, which reads attributes only, reads a flow being absent (`[["ctrl_fed_in_not"]]`) |

The continuous fragments build a conservative flow network out of
component-local declarations: each output publishes, on every connection,
what it can deliver and what it hands over, each input publishes what it
asks and the share it accepts, and a distribution operator splits the
output over the claims ([port equations](model-schema.md#port-equation)
write the per-connection quantities). They are the shapes of the muscadet
plugin's continuous objects, and each is characterised against it.

| Fragment | Parameters | Builds |
|---|---|---|
| `continuous_flow_out` | `name`, `available` (`capability`, `produced`), `allocation` (`proportional`, `shares`), `shares`, `held`, `empty_automaton` | out port `<name>_out` with its channels, the totals `<name>_demand_out` and `<name>_fed_out`, the claims and the distribution operator `<name>_alloc`; `held` makes a volume that has run out pass on what arrives |
| `continuous_flow_in` | `name`, `demand` | in port `<name>_in`, `<name>_capability_in`, `<name>_fed_in`, and the demand and shares it publishes |
| `surplus_release` | `name` | the input hands back, pro rata of offers, what two suppliers or more offer beyond its demand, a supplier that has run out being accepted first |
| `continuous_source` | `name`, `rate`, `profiled` | the capability of a source output, times `<name>_out_profile` when profiled |
| `continuous_sink` | `name`, `demand`, `profiled` | the demand of a consumer input, times `<name>_in_profile` when profiled |
| `time_profile` | `target`, `amplitude`, `phase_shift`, `period`, `offset`, `value_min`, `initial` | a sinusoidal profile written into `target` |
| `rate_publisher` | `name`, `of` (`capability`, `delivered`) | out port `<name>_rate_out` exporting `<name>_rate` |
| `volume_flow` | `volume`, `flow`, `capacity`, `content_init`, `demand`, `fill_rate`, `outlet` | one flow held in a volume: its content (an ODE), its fill fraction, and the demand of the input |
| `volume_outlet` | `volume`, `flow`, `rate`, `empty_automaton` | the capability of an output leaving a volume, bounded by what arrives while it is empty |
| `volume_stock` | `volume`, `flow`, `capacity`, `initial` | automaton `<volume>_stock_<flow>` (`empty`, `stocked`) for one flow of a multi-flow volume |
| `volume_total` | `volume`, `flows`, `capacity`, `initial`, `ratio_flows` | total content and fill, the bounds automaton `<volume>_bounds` (`empty`, `partial`, `full`), and each listed flow's share |
| `rule_set` | `name`, `active`, `idle`, `consumes`, `produces`, `capped`, `conditions`, `threshold` (`none`, `band`), `threshold_flow`, `activate`, `release` | automaton `<name>_mode` running the producing rule while its boolean inputs are fed (and its band holds), and the three scales `<name>_capability_scale`, `<name>_scale`, `<name>_draw_scale` |
| `rule_consumes` | `rule_set`, `active`, `flow`, `rate` | the demand of a consumed input |
| `rule_produces` | `rule_set`, `active`, `flow`, `rate`, `capped`, `max_rate`, `tapped` | the capability and production of an output |
| `tap` | `flow`, `outlets` | leaks taken from an output into the outlets, each the fraction `<outlet>_pct`, targetable with `max` |
| `aggregate_measure` | `name` | attribute `<name>`, the sum over the in port `<name>_in` |
| `threshold_compare` | `name`, `input`, `operator`, `threshold` | boolean output `<name>` true while `input` compares to the threshold |
| `threshold_band` | `name`, `input`, `direction` (`below`, `above`), `activate`, `release` | boolean output `<name>` with a hysteresis band |

The muscadet plugin's mixture serving and ring tearing need a pass over the
whole connection graph and have no fragment: a plant that needs them stays
on the plugin route.

The production condition `prod_cond` is a disjunction of conjunctions,
stated explicitly: a list of groups, each a list of attribute names of the
instance; the output is produced while every attribute of at least one
group is true (`[["a_fed_in", "b_fed_in"], ["c_fed_in"]]` reads
`(a and b) or c`). `<name>_fed_available_out` is targetable with `any_cuts`
when it starts true and `any_activates` when it starts false; the
production of a `bool_source_out` is targetable the same way, which is how
a compromise mode starts a dormant service function.

## System

| Field | Meaning |
|---|---|
| `name` | system name |
| `library` | `{"name": ..., "version": ...}`: the library instantiated, at exactly that version |
| `instances` | `[{"name": "B1", "type": "Battery", "parameters": {"capacity": 100}}]` |
| `connections` | `[{"from": "Srv.f_service_out", "to": "Proc.f_service_in"}]`, from an out port to an in port, with an optional edge `name` |
| `indicators`, `targets` | core indicators and sequence targets, referencing instances by name |
| `unbounded_rate` | the model's [unbounded rate](model-schema.md#unbounded-rate), optional; see below |
| `interface_connections` | `[{"from": "Pump.hydraulic", "to": "Valve.hydraulic"}]`, one [interface connection](model-schema.md#interface-connection) per entry, `Instance.interface` on each side, with an optional `name` |
| `programs` | [mixed-integer programs](model-schema.md#program) in the core form |
| `fmu_units` | [FMU units](model-schema.md#fmu-unit) in the core form |

Each instance becomes a component named after it, so a program or an FMU
unit written in the core form names instances where the core names
components (`{"component": "Pump", "attribute": "flow"}`). These three
constructs bind several instances, which is why a system declares them and
no type body can. A relative FMU `path` resolves when the model runs,
against the system file's directory (the working directory for a system
given as a string or a dictionary), and loading a system that declares an
FMU unit needs `allow_fmu_import=True`, as a flat model does.

A system with an instance applying `std.continuous_flow_out`,
`std.continuous_flow_in` or `std.volume_flow` is a continuous network, and its model declares
`unbounded_rate` at 1e30, the magnitude the standard fragments write for a
ceiling nobody set (a volume's `serve_rate`), as the muscadet plugin does on
every system with a continuous flow. The engine then refuses by name any
integrated rate reaching it, instead of draining a stock at that magnitude.
Such a system may state `unbounded_rate` only at that value; any other
system declares none unless it states one.

## Study

| Field | Meaning |
|---|---|
| `name` | study name; the expanded model is named `<system>/<study>` |
| `system` | the system the study applies to |
| `modes` | activations of catalogued modes |
| `one_off_modes` | modes declared on one instance only: `instance`, `name`, `nature`, `effects`, `condition`, laws, `repair_condition`, `effect_step` |
| `feared_events` | `name`, `condition`, `tempo_occ`, `tempo_not_occ` (a delay, or a timed law object), `target` |
| `common_cause_groups` | `name`, `mode`, `instances`, `orders` (`order`, `occurrence`, `repair`), `effect_step` |
| `indicators`, `targets` | added to the system's; a name the system already uses is refused |
| `simulation` | `method` (`monte_carlo` or `sequences`), `nb_runs`, `horizon`, `instants`, `seed`, `stop_at_targets`, `options` |
| `quantification` | `target`, `horizon`, `instants`, `seed`, `method` (`{"name": ..., "settings": {...}}`); see below |

`simulation.options` carries the other keywords of the run the method
names, with the names of `pyraichu.monte_carlo` or `pyraichu.run_sequences`:
`confidence`, `quantiles`, `threads`, the solver tolerances (`rtol`,
`atol`, `max_step`, `tol_event`, `sub_samples`, `event_resolution`), and
`flow` as an object of [`FlowConfig`](../guides/numerical-tuning.md)
fields; a sequence campaign also takes `observations` (a list of
`Observation` objects: `name`, `component`, `attribute`, `time`),
`condition` (a `SequenceCondition` object: `observation`, `op`, `value`) and
`raw_path`, resolved against the study's directory when relative. A keyword
the run does not take, one the section states itself (`seed`, `nb_runs`),
a required one left out, or an object whose fields do not build is refused
by name when the study is loaded. A `threads` given to `run_study` wins over
the one a study states.

`analyses` requests analyses of the study, each written with the
parameters of the function that runs it, the model aside:

| Key | Run by |
|---|---|
| `fault_tree` | `pyraichu.fault_tree` (`targets`, `top`, `profile`, `max_nodes`, `cut_set_limit`, `cut_sets`, `name`, `measure`) |
| `importance` | `pyraichu.importance` (`nb_runs`, `t_max`, `instants`, `target`, `seed`, `threads`, `flow`, `basic_events`, `groups`) |
| `sequence_analysis` | `pyraichu.analyse_sequences` (`nb_runs`, `t_max`, `seed`, `threads`, `flow`) |

An analysis the format does not list is refused when the study is read, a
parameter the function does not take when it is loaded.
`pyraichu.run_study(model, section="fault_tree")` runs one; a study
carrying a single section runs it without naming it.

A `quantification` states one quantification of the study, as
[`pyraichu.quantify`](../guides/quantification.md) takes it: the
probability that `target` is the first declared target reached by
`horizon`. `target` must name a target of the system or the study.
`method.name` is one of `monte_carlo`, `exact`, `discretised`,
`cross_entropy` and `splitting`, and `method.settings` carries that
method's settings, written and checked as the quantification envelope
records them: a setting of another method is refused, by name, when the
study is loaded. `pyraichu.run_study` returns the quantification envelope.
A study may carry `simulation`, `quantification` or both; with both,
`run_study(model, section=...)` names the one to run.

```json
{"quantification": {"target": "down", "horizon": 8760, "seed": 1,
  "method": {"name": "cross_entropy", "settings": {"nb_runs": 100000}}}}
```

An activation names a catalogued `mode`, then either `instances` (a list of
instance names) or `type` (every instance of that type), an `occurrence`
law (required; a timed law, on demand: `{"distrib": "inst", "probs": [p]}`,
or `{"distrib": "watched"}`), a `repair` law (absent: the mode stays occurred
once it has occurred; a timed law, `watched`, or certain:
`{"distrib": "inst", "probs": []}`), an `on_interruption` policy (default `reset`), an optional
`condition` and an optional `repair_condition`. A repair condition holds
the repair back while it is false: it guards the repair transition only,
never the occurrence nor the re-arm of an on-demand mode, and it needs a
`repair` law. The `on_interruption` policy applies to both transitions, so
a repair whose condition falls is interrupted under that policy.
`effect_step` (`"zero_delay"` or `"instant"`) carries the mode's effects one
step behind its state, through an automaton `<mode>__effect` whose
transitions follow the mode's `occ` state with a zero delay or an
instantaneous law of probability one. The step carries the mode's
sequence events and its `failure` and `repair` kinds, in the mode's cycle
group (`Srv.mdc_a__effect.occ`); the mode's own automaton records none. The
mode then occurs again only with that step at rest and is repaired only
once the step has taken hold. This is how a PyCATSHOO external mode
behaves (its effects ride on a mirror automaton one zero-delay step behind
the mode): a pulse repaired at once is still seen for that step by an
on-demand draw it solicits. `mode_active` keeps reading the mode's own
state. An activation on named instances wins over an activation on
their type; two activations of one mode on one instance at the same level
are refused. A mode the type does not catalogue is refused, naming the
catalogue.

A mode has two states (`rep` and `occ`, plus `parked` for an on-demand
draw). A `watched` law fires when the continuous boundary its condition
states is crossed, so a watched occurrence needs a condition (catalogue or
activation) and a watched repair a `repair_condition`. A watched law also
fires on entry when its condition already holds, so a repaired mode whose
watched occurrence condition still holds occurs again at once. A certain
repair is taken when its `repair_condition` is found true at a discrete
event, and needs one; a repair condition on a continuous attribute is
refused for it (`watched` repairs on the boundary). A law with
several branches has no second destination in a mode and is refused: a
type-body automaton expresses it.

A common-cause group fails several instances together with a mode
catalogued on each member's type (a mode with a catalogue condition is
refused). Each declared `order` `k` builds one automaton per combination of
`k` members, in a component named after the group; a combination is named
after its members' 1-based positions in `instances` (`cc_1_2`), fails and
repairs on its order's laws, and holds every member it covers with the
catalogued effects. An order left out builds nothing. With `effect_step`,
each member gets an automaton `<group>__effect` following the combinations
that cover it one step behind, and a combination occurs only with its
members' steps at rest and is repaired only once they have all taken hold:
the behaviour of a PyCATSHOO external common-cause mode. The members' steps
then carry the sequence events and kinds, in the group's cycle group, so a
combination of two members records two events (`P1.Group__effect.occ`,
`P2.Group__effect.occ`), as the declaration route's mirrors do. Without
`effect_step`, the combinations carry them (`Group.cc_1_2.occ`). A member may
not also have an individual activation of the same mode, and a study target
or a state indicator naming `Member.mode` for a group-held mode is refused:
the combinations live in the group's component. `mode_active` on a member
reads true while any combination covering it is occurred, whichever group
holds it.

A condition is a core expression over instance attributes, plus the leaf

```json
{"op": "mode_active", "mode": "Srv.mdc_a"}
```

true while that mode is occurred. A condition naming a mode the study does
not activate is refused. The guard of a mode is its catalogue condition
AND its activation condition.

## What the expansion produces

- Each activated mode is an automaton named after it inside its instance,
  with states `rep` (initial) and `occ`, transitions named after the state
  they enter (`occ`, `rep`), monitored, of kind `failure` and `repair`,
  with the mode as cycle group. A sequence event reads `Srv.mdc_b.occ`
  (`Srv.mdc_b__effect.occ` when the activation has an `effect_step`).
- An on-demand mode (occurrence `inst` with probability `p`) adds a state
  `parked`: on each rising edge of its condition the transition `occ`
  draws `occ` with probability `p` or `parked` otherwise (a start
  refused, or not, when solicited), and the transition `parked` returns to
  `rep` once the condition falls. Only the entry into `occ` is a sequence
  event; parking is what stops a lost draw from being drawn again within
  one solicitation.
- Each targeted attribute gets one sensitive function `modes__<attribute>`
  holding the combined value. An attribute written by edge effects gets
  none: its writes are the `effects` of the transitions that carry them.
- Each feared event is a component named after it, with an automaton `ev`
  (`not_occ`, `occ`) of observation transitions delayed by `tempo_occ` and
  `tempo_not_occ`, each a fixed delay or drawn from the timed law written
  there; `target: true` adds a sequence target.
- The model's `source` names the library and its version, the system, the
  study and the engine version that supplied `std`.
- A model with explicit equations gets an `evaluation_order` derived from
  what each step reads: a step follows every step whose result it reads,
  independent steps keep their declaration order, and the steps of a cycle
  stay together, each after what it reads but for the reads that close the
  cycle. `pyraichu.derive_evaluation_order` returns the same order for any
  flat model.

The provenance maps every element path to its origin:

| Key | Element |
|---|---|
| `component:C` | component (an instance or a feared event) |
| `attribute:C.a`, `port:C.p`, `interface:C.i` | attribute, port, interface |
| `automaton:C.a`, `transition:C.a.t` | automaton, transition |
| `function:C.f`, `equation:C.a`, `allocation:C.n` | sensitive function, equation targeting `a`, allocation |
| `connection:E`, `indicator:N`, `target:N` | connection (by name, or `from->to`), indicator, target |

An origin holds the document, the JSON Pointer of the node, and the
instance, type, fragment and mode (with its nature) the element belongs to.
An error the core raises on the expanded model is restated with the origin
of the element it names.
