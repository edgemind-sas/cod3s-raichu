# 2. Connecting components

Real systems are made of parts that influence each other. In RAICHU a
component exposes **ports** and you wire an **out-port** to an
**in-port** with a **connection**. A component reads its inputs through
**port aggregations** and reacts through **sensitive functions** —
declarative rules that recompute an attribute whenever something it
depends on changes.

We will model a **redundant supply**: a source feeds two parallel
blocks, and a target is supplied as long as *at least one* block
delivers.

## Ports and attributes

An **out-port** publishes one attribute of its component. An **in-port**
receives whatever out-ports are connected to it — possibly several. A
attribute carries a typed value with an initial value (`bool`, `int` or
`float`):

```python
import pyraichu

# The source is always up; it exports a boolean on its out-port.
source = {
    "name": "S",
    "attributes": [{"name": "up", "kind": "bool",
                   "init": {"kind": "bool", "value": True}}],
    "ports": [{"name": "out", "dir": "out", "attr": "up"}],
}
```

## Reading inputs: port aggregation and sensitive functions

Each block has an internal `health` automaton (as in chapter 1) and a
boolean `delivering`. A block delivers when it is **fed** *and*
**working**. "Fed" means *any* connected source is true — a **port
aggregation** with the `any` operator over the in-port. The rule that
keeps `delivering` up to date is a **sensitive function**: you declare
the effect (an assignment whose value is an expression), and the engine
figures out *when* to run it from the attributes and states that
expression reads — no manual wiring, and no callback runs during the
numerical hot loop.

```python
def block(name):
    return {
        "name": name,
        "attributes": [{"name": "delivering", "kind": "bool",
                       "init": {"kind": "bool", "value": False}}],
        "ports": [{"name": "in", "dir": "in"},
                  {"name": "out", "dir": "out", "attr": "delivering"}],
        "automata": [{
            "name": "health", "states": ["working", "failed"], "init": "working",
            "transitions": [
                {"name": "fail", "source": "working", "targets": ["failed"],
                 "distrib": "exp", "rate": 0.02},
                {"name": "repair", "source": "failed", "targets": ["working"],
                 "distrib": "exp", "rate": 0.1},
            ],
        }],
        "sensitive_functions": [{
            "name": "update_delivering",
            "effects": [{
                "target": {"component": name, "attribute": "delivering"},
                "value": {"op": "bool", "bool_op": "and", "args": [
                    {"op": "port_agg",
                     "port": {"component": name, "port": "in"}, "agg": "any"},
                    {"op": "state_active",
                     "state": {"component": name, "automaton": "health",
                               "state": "working"}},
                ]},
            }],
        }],
    }
```

The `port_agg` operator turns an in-port's connected values into one
value. The aggregators are `any`, `all`, `count`, `sum`, `mean` and
`median` — enough to express OR-redundancy, k-out-of-n voting, load
sums and averages.

## The target and the connections

The target is supplied when *any* incoming block delivers:

```python
target = {
    "name": "T",
    "attributes": [{"name": "supplied", "kind": "bool",
                   "init": {"kind": "bool", "value": False}}],
    "ports": [{"name": "in", "dir": "in"}],
    "sensitive_functions": [{
        "name": "update_supplied",
        "effects": [{
            "target": {"component": "T", "attribute": "supplied"},
            "value": {"op": "port_agg",
                      "port": {"component": "T", "port": "in"}, "agg": "any"},
        }],
    }],
}

model = pyraichu.load_model({
    "name": "redundant_supply",
    "components": [source, block("B1"), block("B2"), target],
    "connections": [
        {"from": {"component": "S", "port": "out"},
         "to": {"component": "B1", "port": "in"}},
        {"from": {"component": "S", "port": "out"},
         "to": {"component": "B2", "port": "in"}},
        {"from": {"component": "B1", "port": "out"},
         "to": {"component": "T", "port": "in"}},
        {"from": {"component": "B2", "port": "out"},
         "to": {"component": "T", "port": "in"}},
    ],
    "indicators": [
        {"name": "T_supplied", "target": "attribute",
         "attr": {"component": "T", "attribute": "supplied"}},
    ],
})
```

## Does redundancy help?

```python
estimates = pyraichu.monte_carlo(
    model, nb_runs=2000, t_max=100.0,
    samples=[10.0 * k for k in range(11)], seed=1,
)
availability = estimates.indicators["T_supplied"].mean
print("T supplied:", [round(v, 3) for v in availability])
```

The target stays supplied ~97 % of the time. Each block is unavailable
about λ/(λ+μ) = 0.02/0.12 ≈ 0.167 of the time; with two independent
blocks in parallel the target is down only when **both** fail at once,
≈ 0.167² ≈ 0.028 — so ≈ 0.972 availability, which is what the estimate
shows. Redundancy turned a 17 % outage into a 3 % one.

!!! tip "You do not have to write this by hand"
    For flow/reliability networks like this one, the
    [muscadet authoring layer](../guides/muscadet-and-plugins.md) builds
    the same native model from a few lines of Python
    (`add_flow_in` / `add_flow_out` / failure modes). This chapter shows
    what it generates underneath.

→ [3. Stochastic behaviour and Monte-Carlo](03-stochastic-and-monte-carlo.md)
