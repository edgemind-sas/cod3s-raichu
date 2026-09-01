# Model schema reference

A RAICHU model is a JSON object (or an equivalent Python `dict`) passed
to `pyraichu.load_model`. This page documents every field, distribution and
expression operator. Types are the JSON types; a *ref* is a small object
that names something elsewhere in the model.

A complete, minimal model that uses most sections:

<!-- model -->
```json
{
  "name": "example",
  "components": [
    {
      "name": "C",
      "attributes": [
        {"name": "load", "kind": "float", "init": {"kind": "float", "value": 1.0}}
      ],
      "ports": [{"name": "out", "dir": "out", "attr": "load"}],
      "automata": [
        {
          "name": "health", "states": ["ok", "ko"], "init": "ok",
          "transitions": [
            {"name": "fail", "source": "ok", "targets": ["ko"],
             "distrib": "exp", "rate": 0.01}
          ]
        }
      ]
    }
  ],
  "connections": [],
  "indicators": [
    {"name": "C_ko", "target": "state",
     "component": "C", "automaton": "health", "state": "ko"}
  ]
}
```

## Model

| key | type | required | meaning |
|---|---|---|---|
| `name` | string | yes | model name (carried into provenance) |
| `components` | array of [Component](#component) | yes | the system's parts |
| `connections` | array of [Connection](#connection) | no (default `[]`) | out-port → in-port wiring |
| `indicators` | array of [Indicator](#indicator) | no (default `[]`) | what the engine measures |
| `targets` | array of [Target](#target) | no (default `[]`) | feared-event states for [sequence analysis](../guides/sequence-analysis.md) |
| `evaluation_order` | array of VarRef | no (default: declaration order) | sweep order of the explicit equations, see [Evaluation order](#evaluation-order) |

### Connection

`{ "from": PortRef, "to": PortRef, "name": string }` where a **PortRef**
is `{ "component": string, "port": string }`. `from` must name an
out-port, `to` an in-port.

`name` is optional and names the **edge**, nothing else: it only feeds
the naming of the per-connection attributes materialised for the source
port's [channels](#port). Absent, the edge is named after its
destination, which is unambiguous unless two connections join the same
pair of ports; naming at least one of those is then required, and the
model is refused otherwise.

## Component

Only `name` is required; every collection defaults to empty.

| key | type | meaning |
|---|---|---|
| `name` | string | component name (unique in the model) |
| `attributes` | array of [Attribute](#attribute) | intrinsic typed state |
| `ports` | array of [Port](#port) | connection points |
| `interfaces` | array of [Interface](#interface) | named groups of ports |
| `automata` | array of [Automaton](#automaton) | state machines |
| `sensitive_functions` | array of [SensitiveFunction](#sensitivefunction) | declarative effects |
| `equations` | array of [Equation](#equation) | continuous dynamics |
| `allocations` | array of [Allocation](#allocation) | conservative distribution operators |

### Attribute

`{ "name": string, "kind": "bool"|"int"|"float", "init": Value }`.
A **Value** is `{ "kind": "bool"|"int"|"float", "value": <literal> }`.

### Port

| key | type | meaning |
|---|---|---|
| `name` | string | port name (unique in the component) |
| `dir` | `"in"` \| `"out"` | direction |
| `attr` | string | **out-ports only**: the attribute this port exports |
| `channels` | array of [Channel](#channel) | **out-ports only**, optional (default `[]`): per-connection quantities |

An in-port omits `attr`; it aggregates whatever is connected to it (read
with the [`port_agg`](#expressions) operator).

### Channel

`{ "name": string, "init": float }` (`init` optional, default `0`),
declared on an **out** port. An in port never declares channels: it reads
them, and a channel list on an in port is refused.

An out port exports **one** attribute, so every in port connected to it
reads the same number. A channel is the opposite affordance: one quantity
**per connection**, so a producer can hand a different share to each
consumer over the same port (a conservative flow).

The channel is declared once, here; the **compiler materialises** one
float attribute per (connection, channel) on the producing component,
named

```
<out port>__<channel>__<edge>
```

where `<edge>` is the connection's `name` when it has one and
`<destination component>__<destination port>` otherwise. So a channel
`share` on port `out`, over an unnamed connection to `consumer.input`,
materialises `out__share__consumer__input` on the producing component.

Those attributes are **ordinary float attributes**: equations and
sensitive functions write them, indicators observe them, the causal
journal records them under that name, and the snapshot carries them. They
are not declared in `attributes`, and a declared attribute that collides
with a materialised name is refused at build time, naming both.

A consumer reads a channel through `port_agg` with a `channel` selector
(see [Expressions](#expressions)). Without a selector, `port_agg` keeps
reading the exported `attr`: the producer's total stays visible.

### Interface

`{ "name": string, "ports": [string, …] }`: a named bundle of the
component's ports, for connecting several at once.

### Automaton

`{ "name": string, "states": [string, …], "init": string,
"transitions": [Transition, …] }`. State names are scoped **to the
automaton**. `init` must be one of `states`.

### Transition

| key | type | meaning |
|---|---|---|
| `name` | string | transition name |
| `source` | string | source state (in the same automaton) |
| `targets` | array of string | destination state(s) |
| `guard` | [Expr](#expressions) | optional; must hold for the transition to be eligible |
| `on_interruption` | `"reset"` \| `"resume"` \| `"continue"` | optional (default `reset`); see [below](#interruption-policy) |
| `monitored` | bool | optional (default `false`); firing is recorded in the trajectory's [sequence](../guides/sequence-analysis.md) |
| `cycle_group` | string | optional; failure/repair partners share it so transient cycles cancel in the sequence pipeline (paired per component) |
| `distrib` + params | - | the occurrence distribution, flattened onto the transition (see [Distributions](#distributions)) |

### Target

`{ "name": string, "component": string, "automaton": string, "state":
string }`: a **feared event**: when the named state activates, a
sequence-recording trajectory records `name` as its end cause and stops
(after completing the current instant). Ignored unless sequence
recording / `stop_at_targets` is enabled.

## Distributions

A transition has one of **two natures**, distinguished by where its
randomness lives:

- **Timed**: the firing *date* is drawn (or fixed) and the transition
  has a single effective destination. It is either **deterministic**
  (`delay`) or **stochastic** (`exp`, `weibull`, `gamma`, `lognormal`,
  `uniform`, `empirical`).
- **Instantaneous**: the transition fires at the instant its guard
  holds; the randomness is in the **choice of destination** among its
  targets (`inst`, with `probs`).

Both natures are encoded through the `distrib` key, with the distribution's parameters
on the same transition object.

### Timed distributions (the firing date)

| `distrib` | parameters | notes |
|---|---|---|
| `delay` | `time`: number | fixed deterministic duration |
| `exp` | `rate`: number **or** `rate_expr`: [Expr](#expressions) | exponential; `rate_expr` is a state-dependent rate |
| `weibull` | `shape`, `scale`: number | |
| `lognormal` | `mu`, `sigma`: number | |
| `gamma` | `shape`, `scale`: number | |
| `uniform` | `low`, `high`: number | |
| `empirical` | `points`: array of `[t, F(t)]` | measured CDF (time, cumulative probability) |

### Instantaneous distribution (the destination branch)

| `distrib` | parameters | notes |
|---|---|---|
| `inst` | `probs`: array of number | fires when the guard holds; `probs` are the destination probabilities, `len(probs) = len(targets) − 1` (the complement is reconstructed) |

### `watched`: a guard on continuous attributes

`"distrib": "watched"` is **not a third nature**. It marks a *guarded*
transition whose guard involves continuously-evolving (ODE-driven)
attributes: the engine must **monitor the continuous trajectory** and
fire the transition exactly when the boundary is crossed (located by
root-finding), rather than re-checking the guard only at discrete events.

It is declared explicitly because that intent **cannot be inferred from
the guard alone**: the same comparison could instead gate a timed
transition's eligibility. A watched transition takes no distribution parameters
and requires a `guard` containing an ordering comparison
(`lt`/`le`/`gt`/`ge`).

### Interruption policy

`on_interruption` governs a running countdown whose guard becomes false:

| value | behaviour |
|---|---|
| `reset` (default) | the elapsed countdown is cancelled and redrawn when the guard holds again |
| `resume` | the countdown pauses and resumes where it left off |
| `continue` | the countdown never stops, guard or not |

## SensitiveFunction

`{ "name": string, "effects": [Assignment, …] }`. An **Assignment** is
`{ "target": VarRef, "value": Expr }`, where **VarRef** is
`{ "component": string, "attribute": string }`. The engine derives *when*
to run a sensitive function from the attributes and states its
expressions read: there is no manual trigger list, and no callback runs
during numerical integration.

## Equation

`{ "target": string, "kind": "ode"|"explicit", "expr": Expr }`. The
`target` is a local `float` attribute; `ode` means `d(target)/dt = expr`,
`explicit` means `target = expr`.

## Allocation

The **conservative distribution operator**: it reads one available
quantity and one demand per outgoing connection, and writes one allocated
quantity per outgoing connection, under a declared policy. Nothing to do
with the [occurrence distributions](#distributions) of a transition,
which say *when* something fires.

| key | type | meaning |
|---|---|---|
| `name` | string | operator name, unique in the component; the name that designates this step in the [evaluation order](#evaluation-order) |
| `port` | string | the **out** port whose connections receive the quantity |
| `available` | [Expr](#expressions) | the quantity to distribute (negative distributes nothing) |
| `demand` | string | [channel](#channel) of `port` carrying what each consumer asks for (read) |
| `allocated` | string | channel of `port` receiving what each consumer gets (written) |
| `policy` | `"proportional"` \| `"shares"` \| `"priority"` | how a shortage is split |

A [sensitive function](#sensitivefunction) effect writes **one** target,
so it cannot express a split: the share handed to one consumer depends on
what every other consumer asked for. The operator therefore writes the
whole vector at once, as one step of the explicit sweep, and is evaluated
at every evaluation point like an equation. That placement is what lets a
[watched](#watched-a-guard-on-continuous-attributes) guard reading an
allocated quantity be *located* at its crossing instant rather than
noticed at the next discrete date.

### Policies

| `policy` | parameters | rule |
|---|---|---|
| `proportional` | none | each consumer receives `available x demand / Σ demands`, capped at its own demand |
| `shares` | `shares`: array of ConsumerParam | each receives `available x share`, capped at its own demand |
| `priority` | `priorities`: array of ConsumerParam | consumers are served in full, in ascending rank, until the quantity runs out |

A **ConsumerParam** is `{ "to": PortRef, "value": number }`: the value
that applies to the connection ending at `to`. Keying by destination
rather than by position means inserting or reordering a connection cannot
re-attach a share to a different consumer. A keyed policy must cover every
connection of the port exactly once, and `shares` must sum to 1 over
them; both are refused at build time, naming the component and the flow.

### What the operator guarantees

- **Conservation.** No consumer receives more than it asked for, and the
  quantities handed out never exceed what was available. A consumer that
  asks for less than its share does not absorb the surplus: a capping loop
  fixes it at its demand and redistributes the rest, in at most one pass
  per consumer.
- **Order independence.** Each share is a function of its own demand and
  of the totals, never of a position in the sweep, so equal demands
  receive equal quantities. The two real ties, equal demands under
  `proportional` and equal ranks under `priority`, break by **connection
  declaration index**: a property of the model file, never of the
  engine's evaluation order or of a hash order.
- **One writer.** Nothing else may write an allocated quantity, neither an
  equation nor a sensitive function nor a second operator: refused at
  build time, because two writers mean the last one silently wins.

Negative demands and a negative available quantity are read as zero (a
level crossing zero mid-segment lands a few ulps below it, and that is
rounding, not a negative demand); a non-finite one is an error naming the
operator.

### How a network of operators is resolved

One pass of the sweep is not the answer when what a consumer asks for
depends on what it was given. The engine therefore **resolves** the
network at every discrete epoch (initialization, after a fired
transition) and again at every located active-set crossing, in two
stages:

1. **The active set, settled to exact equality.** Which consumers are
   saturated, and which branch of each minimum and each conditional the
   sweep takes, is a finite combinatorial question. It is settled first,
   because settling it turns most of the problem from asymptotic into
   finite.
2. **The flows, settled to a tolerance.** Once the active set repeats,
   the sweep is iterated until no quantity moves by more than the
   per-edge flow tolerance, `1e-9` relative above unit scale and absolute
   below it.

The iteration **descends**, and it starts from a **cold state** in which
every allocated quantity is zero. Each consumer then sizes itself as
though it held nothing, so the first pass over-estimates every delivery;
the sequence that follows is non-increasing and bounded below. Iterating
up from zero deliveries carries no such argument. The cold state is
recomputed at each resolution rather than carried from the previous one,
so nothing about the search lives outside the attribute vector and a
restored snapshot replays exactly.

A model that declares no operator has no active set to settle and skips
all of this: it runs the same single ordered pass it ran before the
resolution existed.

### The active set inside an integration segment

Only the *search* is done at the boundary. The resolved network is
evaluated by the ordinary explicit pass at **every solver stage**, so a
flow moves with the state and a [watched](#watched-a-guard-on-continuous-attributes)
guard reading one is located at its crossing instant.

The frozen active set is itself watched. Every operator contributes one
**active-set margin** per outgoing connection, monitored exactly like a
watched guard: the segment ends the moment a consumer would become
saturated (or stop being), the network is resolved again from that state,
and integration continues. With `journal` enabled each of those crossings
appears as an `active_set_crossed` record naming the operator, the edge,
and the two saturation classes.

Each margin carries a **dead band** equal to the flow tolerance. Without
it a network resolved *on* a boundary would re-cross it at once and
chatter there; with it, a residual smaller than what the resolution
itself promises is not treated as a crossing. That is why the band is the
flow tolerance and not the (ten times smaller) event-location tolerance.

A **minimum** gets no margin of its own. A limiting reagent written as a
minimum over inputs keeps a kink rather than a jump, which the integrator
handles unaided, and one watched guard per input pair would add a
quadratic population for accuracy the kink already provides. Which branch
a minimum takes still enters the resolution's stopping test, where it is
free.

### `priority` and surplus return: refused together

A consumer **returns surplus** when the demand it publishes on an edge
depends, through the explicit sweep, on the quantity that same operator
allocated it. That shape is legal and useful: it is what lets a consumer
limited elsewhere hand back what it cannot use, within the same
resolution. It is also the one cycle the
[algebraic-loop refusal](#allocation) deliberately allows.

Combined with `priority` it is **refused at build time**, naming the
component, the operator, the demand channel and the allocated channel it
reaches. The descending resolution needs the ordered pass to
over-estimate every delivery; nobody has shown that for a strict priority
order, where a consumer whose demand shrinks because it was served can
move the point at which the supply runs out and *raise* a later
consumer's delivery. Rather than promise a resolution and then report it
as non-convergent, the engine refuses the composition. Use `proportional`
or `shares`, whose weighted split keeps the over-estimate, or cut the
dependency.

`priority` on its own, and surplus return on its own, are both accepted.

<!-- model -->
```json
{
  "raichu_model": {"format": 1, "requires": ["allocation"]},
  "model": {
    "name": "shortage",
    "components": [
      {
        "name": "supply",
        "attributes": [
          {"name": "available", "kind": "float", "init": {"kind": "float", "value": 5.0}}
        ],
        "ports": [
          {"name": "out", "dir": "out", "attr": "available",
           "channels": [{"name": "demand"}, {"name": "alloc"}]}
        ],
        "equations": [
          {"target": "out__demand__a", "kind": "explicit",
           "expr": {"op": "const", "value": {"kind": "float", "value": 6.0}}},
          {"target": "out__demand__b", "kind": "explicit",
           "expr": {"op": "const", "value": {"kind": "float", "value": 4.0}}}
        ],
        "allocations": [
          {"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
           "available": {"op": "attr",
             "attr": {"component": "supply", "attribute": "available"}},
           "policy": "proportional"}
        ]
      },
      {
        "name": "a",
        "attributes": [
          {"name": "got", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ],
        "ports": [{"name": "input", "dir": "in"}],
        "equations": [
          {"target": "got", "kind": "explicit",
           "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {"component": "a", "port": "input"}}}
        ]
      },
      {
        "name": "b",
        "attributes": [
          {"name": "got", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ],
        "ports": [{"name": "input", "dir": "in"}],
        "equations": [
          {"target": "got", "kind": "explicit",
           "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {"component": "b", "port": "input"}}}
        ]
      }
    ],
    "connections": [
      {"name": "a", "from": {"component": "supply", "port": "out"},
       "to": {"component": "a", "port": "input"}},
      {"name": "b", "from": {"component": "supply", "port": "out"},
       "to": {"component": "b", "port": "input"}}
    ]
  }
}
```

Five units against demands of 6 and 4: `a` receives 3 and `b` receives 2.

## Evaluation order

The explicit equations are swept **once** per evaluation point, in a
single pass. An equation that reads an attribute the same pass has not
reached yet therefore reads what the *previous* evaluation point left
there. The order is part of the answer, not an implementation detail.

A **step** of the sweep is either an explicit equation, designated by its
target attribute, or an [allocation](#allocation), designated by its own
name. An operator writes many attributes, so it is named rather than
targeted; a component may not give one name to both, which is refused.

By default the order is **positional**: components in declaration order,
and inside each, the explicit equations in declaration order followed by
the distribution operators. Every model written without the field keeps
exactly that order.

`evaluation_order` overrides it with an explicit list of
`{ "component": string, "attribute": string }`, and must cover the
declared steps **exactly**: one entry each, no omission, no repetition,
and nothing that is neither an explicit equation nor an operator (an ODE
target is carried by the integrator, so it is never listed). A partial
order is refused rather than completed, naming what is missing, unknown
or repeated.

A model carrying the field must be sealed in the
[format envelope](#document-format-and-the-feature-envelope) and declare
the `evaluation_order` feature:

<!-- model -->
```json
{
  "raichu_model": {"format": 1, "requires": ["evaluation_order"]},
  "model": {
    "name": "ordered",
    "components": [
      {
        "name": "C",
        "attributes": [
          {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}},
          {"name": "y", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ],
        "equations": [
          {"target": "y", "kind": "explicit",
           "expr": {"op": "add", "args": [
             {"op": "attr", "attr": {"component": "C", "attribute": "x"}},
             {"op": "const", "value": {"kind": "float", "value": 1.0}}]}},
          {"target": "x", "kind": "explicit",
           "expr": {"op": "const", "value": {"kind": "float", "value": 5.0}}}
        ]
      }
    ],
    "evaluation_order": [
      {"component": "C", "attribute": "x"},
      {"component": "C", "attribute": "y"}
    ]
  }
}
```

Declared as above, `y` is 6 at every evaluation point. Without the field,
the positional sweep computes `y` before `x` and it is 1 at the first
one.

## Document format and the feature envelope

A model document comes in one of two shapes.

**Bare body.** The model object itself, as everywhere else on this page.
This is what the whole existing corpus uses, and it stays readable. A
bare body declares no feature, so it may only use **baseline**
constructs: everything the format could express before the feature
registry opened.

**Envelope.** A mandatory header plus the body underneath:

```json
{
  "raichu_model": {"format": 1, "requires": ["evaluation_order"]},
  "model": {"name": "…", "components": []}
}
```

| key | type | meaning |
|---|---|---|
| `format` | integer | revision of the envelope itself (currently `1`) |
| `requires` | array of string | feature names the document requires |

### Why a wrapper rather than a field

The schema accepts unknown fields. An optional `requires` field placed
next to `name` and `components` would therefore be **invisible** to an
engine that predates it: that engine would parse the document, ignore
the field *and the construct it announces*, and return a different
number instead of a refusal. Silence is the failure to avoid.

The envelope instead **displaces** the fields a legacy reader requires:
the body moves under `model`, so a reader expecting `name` and
`components` at the top level fails on a missing field rather than
succeeding on a misread. That refusal works even against engines
released before the envelope existed. Readers from this version on know
the shape, so their refusal is the precise, named one:

- a `requires` entry this engine does not implement → refused, naming
  the feature and listing the implemented ones;
- a body using a non-baseline construct the document does not declare →
  refused, naming the construct.

### The list is derived, not trusted

`requires` is never taken on trust. The reader derives the truth from
the parsed body and refuses a document whose declaration does not cover
it, so a hand-written model cannot use a construct and stay silent about
it. On the writing side the list is composed by the engine:
`pyraichu.required_features(body_json)` returns it and `pyraichu.seal`
wraps a body with it, which is how an authoring layer (and the plugin
expansion, through the model-level key `evaluation_order`) emits a
sealed document without ever writing the list by hand.

### Feature registry

| feature | construct |
|---|---|
| `evaluation_order` | model-level [evaluation order](#evaluation-order) |
| `allocation` | component-level [allocations](#allocation) |

The registry names **serialized constructs**, not engine behaviour, so a
change in how an existing construct is *interpreted* does not add a
feature: it makes the construct that carries it mean something new. The
[network resolution](#how-a-network-of-operators-is-resolved), the
active-set margins and the `priority`/surplus-return refusal all arrive
with `allocations` and are therefore covered by `allocation`. No engine
that implements `allocation` implements it without them: the two landed
in the same release, which is what keeps the one feature name honest.

## Indicator

`{ "name": string, "target": "attribute"|"state", … }`:

- `target: "attribute"` → `"attr": VarRef`
- `target: "state"` → `"component"`, `"automaton"`, `"state"` (strings)

Estimators are computed by `monte_carlo`: mean, standard deviation,
nearest-rank quantiles, and the cumulated **sojourn** (time-integral) of
the observed value.

## Expressions

Every expression is an object with an `"op"` tag. The 16 operators:

| `op` | fields | value |
|---|---|---|
| `const` | `value`: Value | a literal |
| `attr` | `attr`: VarRef | an attribute's current value |
| `state_active` | `state`: StateRef | `true` iff the automaton is in that state |
| `port_agg` | `port`: PortRef, `agg`: AggOp, `channel`: string (optional) | aggregate an in-port's connected values; with `channel`, the [per-connection](#channel) quantities instead of the exported attributes |
| `cmp` | `cmp`: CmpOp, `lhs`, `rhs`: Expr | a comparison → bool |
| `bool` | `bool_op`: `and`\|`or`\|`not`, `args`: array of Expr | boolean combination (`not` takes exactly one) |
| `add` | `args`: array of Expr | sum |
| `sub` | `lhs`, `rhs`: Expr | difference |
| `mul` | `args`: array of Expr | product |
| `div` | `lhs`, `rhs`: Expr | quotient |
| `min` / `max` | `args`: array of Expr | extremum |
| `if` | `cond`, `then`, `otherwise`: Expr | conditional |
| `sin` | `arg`: Expr | sine |
| `exp` | `arg`: Expr | exponential |
| `time` | *(none)* | the current simulation time |

Enumerations:

- **AggOp** (`agg`): `sum`, `count`, `all`, `any`, `mean`, `median`
- **CmpOp** (`cmp`): `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- **StateRef**: `{ "component", "automaton", "state" }`

## Simulation configuration

Two entry points consume a model (see the tutorial for usage):

`simulate(model, t_max, seed=0, rng_stream=0, samples=None,
journal=False, confluence_check=False, flow=None)`: one trajectory;
returns events, indicator series, dense `samples`, optional `journal`,
and `provenance`.

`monte_carlo(model, nb_runs, t_max, samples, seed=0, threads=None,
quantiles=None, rtol=None, atol=None, max_step=None, tol_event=None,
sub_samples=None, stop_at_targets=False, flow=None)`: parallel replicas;
returns per-indicator estimates. Replica *r* uses RNG substream *r*; the
reduction is index-ordered, so results are byte-identical for any
`threads`. The `rtol` / `atol` / `max_step` / `tol_event` /
`sub_samples` keywords set the ODE integration effort
([Numerical tuning](../guides/numerical-tuning.md)).

`flow` takes a `FlowConfig(sweep_budget=None, active_set_budget=None,
relaxation=None, tolerance=None)`: the convergence policy of the
continuous flow resolution, as one object rather than four more
keywords. Every knob left unset keeps the engine default, so omitting
`flow` and passing `FlowConfig()` are the same run
([Numerical tuning](../guides/numerical-tuning.md)).
