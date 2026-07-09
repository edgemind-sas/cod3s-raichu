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

### Connection

`{ "from": PortRef, "to": PortRef }` where a **PortRef** is
`{ "component": string, "port": string }`. `from` must name an
out-port, `to` an in-port.

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

### Attribute

`{ "name": string, "kind": "bool"|"int"|"float", "init": Value }`.
A **Value** is `{ "kind": "bool"|"int"|"float", "value": <literal> }`.

### Port

| key | type | meaning |
|---|---|---|
| `name` | string | port name (unique in the component) |
| `dir` | `"in"` \| `"out"` | direction |
| `attr` | string | **out-ports only**: the attribute this port exports |

An in-port omits `attr`; it aggregates whatever is connected to it (read
with the [`port_agg`](#expressions) operator).

### Interface

`{ "name": string, "ports": [string, …] }` — a named bundle of the
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
| `distrib` + params | — | the occurrence distribution, flattened onto the transition (see [Distributions](#distributions)) |

## Distributions

A transition has one of **two natures**, distinguished by where its
randomness lives:

- **Timed** — the firing *date* is drawn (or fixed) and the transition
  has a single effective destination. It is either **deterministic**
  (`delay`) or **stochastic** (`exp`, `weibull`, `gamma`, `lognormal`,
  `uniform`, `empirical`).
- **Instantaneous** — the transition fires at the instant its guard
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

### `watched` — a guard on continuous attributes

`"distrib": "watched"` is **not a third nature**. It marks a *guarded*
transition whose guard involves continuously-evolving (ODE-driven)
attributes: the engine must **monitor the continuous trajectory** and
fire the transition exactly when the boundary is crossed (located by
root-finding), rather than re-checking the guard only at discrete events.

It is declared explicitly because that intent **cannot be inferred from
the guard alone** — the same comparison could instead gate a timed
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
expressions read — there is no manual trigger list, and no callback runs
during numerical integration.

## Equation

`{ "target": string, "kind": "ode"|"explicit", "expr": Expr }`. The
`target` is a local `float` attribute; `ode` means `d(target)/dt = expr`,
`explicit` means `target = expr`.

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
| `port_agg` | `port`: PortRef, `agg`: AggOp | aggregate an in-port's connected values |
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
journal=False, confluence_check=False)` — one trajectory; returns
events, indicator series, dense `samples`, optional `journal`, and
`provenance`.

`monte_carlo(model, nb_runs, t_max, samples, seed=0, threads=None,
quantiles=None, rtol=None, atol=None, max_step=None, tol_event=None,
sub_samples=None)` — parallel replicas; returns per-indicator estimates.
Replica *r* uses RNG substream *r*; the reduction is index-ordered, so
results are byte-identical for any `threads`. The `rtol` / `atol` /
`max_step` / `tol_event` / `sub_samples` keywords set the ODE
integration effort ([Numerical tuning](../guides/numerical-tuning.md)).
