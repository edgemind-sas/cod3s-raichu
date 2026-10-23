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
| `items` | the item kind of a `list`: `number`, `integer`, `bool`, `string` or `string_list` |

A `law` is a core law object (`{"distrib": "exp", "rate": 0.1}`). An instance
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
| `{"$choose": "method", "cases": {"mean": ..., "median": ...}}` | the case named by an `enum`, `string` or `bool` parameter (`"true"`, `"false"`) |

The string `"$self"` names the instance, and `"$self.x"` reads
`"<instance>.x"`. Nothing else computes at expansion: parameters reduce to
literals, and arithmetic stays in runtime expressions.

### Mode catalogue

A catalogued mode declares what it does, never its law:

| Field | Meaning |
|---|---|
| `name` | mode name, unique on the type |
| `nature` | `failure` or `compromise`; it serves selection and reporting and changes no semantics |
| `effects` | `[{"attribute": "COCOFE_fed_available_out", "value": false}]`: the value held on a targetable attribute while the mode is occurred (a plain JSON boolean or number) |
| `condition` | the shape of its activation condition: a core expression over the instance (`$self`), which may also use `mode_active` (below) |

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
| `bool_flow_out` | `name`, `prod_cond`, `fed_available_init` | out port `<name>_out` exporting `<name>_fed_out`, true while the production condition holds and `<name>_fed_available_out` is true |
| `bool_source_out` | `name`, `prod_init`, `fed_available_init` | the same output with no production condition: a constant source, or a dormant function when `prod_init` is false |
| `logic_gate` | `name`, `logic`, `k`, `default` | a boolean input re-exported on the out port `<name>_out` |

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

Each instance becomes a component named after it.

## Study

| Field | Meaning |
|---|---|
| `name` | study name; the expanded model is named `<system>/<study>` |
| `system` | the system the study applies to |
| `modes` | activations of catalogued modes |
| `one_off_modes` | modes declared on one instance only: `instance`, `name`, `nature`, `effects`, `condition`, laws |
| `feared_events` | `name`, `condition`, `tempo_occ`, `tempo_not_occ`, `target` |
| `indicators`, `targets` | added to the system's; a name the system already uses is refused |
| `simulation` | `method` (`monte_carlo` or `sequences`), `nb_runs`, `horizon`, `instants`, `seed`, `stop_at_targets` |

An activation names a catalogued `mode`, then either `instances` (a list of
instance names) or `type` (every instance of that type), an `occurrence`
law (required), a `repair` law (absent: the mode stays occurred once it has
occurred), an `on_interruption` policy (default `reset`) and an optional
`condition`. An activation on named instances wins over an activation on
their type; two activations of one mode on one instance at the same level
are refused. A mode the type does not catalogue is refused, naming the
catalogue.

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
  with the mode as cycle group. A sequence event reads `Srv.mdc_b.occ`.
- Each targeted attribute gets one sensitive function `modes__<attribute>`
  holding the combined value.
- Each feared event is a component named after it, with an automaton `ev`
  (`not_occ`, `occ`) of observation transitions delayed by `tempo_occ` and
  `tempo_not_occ`; `target: true` adds a sequence target.
- The model's `source` names the library and its version, the system, the
  study and the engine version that supplied `std`.

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
