# Mixed-integer optimisation

A model-level program selects decision attributes at each discrete fixpoint.
The engine solves it at initialization and whenever an input attribute or
state changes. It writes the proven optimum, a feasibility flag, and optionally
the objective value. No explicit solve call is needed.

This two-source dispatch meets demand 80 at the least variable cost. The cheap
source can supply 60 at unit cost 1; the expensive source can supply 50 at
unit cost 2. The optimum is 60 and 20, costing 100.

```python
import pyraichu


def ref(name):
    return {"component": "plant", "attribute": name}


def read(name):
    return {"op": "attr", "attr": ref(name)}


def number(value):
    return {"op": "const", "value": {"kind": "float", "value": value}}


model = pyraichu.load_model({
    "name": "least_cost_dispatch",
    "components": [{"name": "plant", "attributes": [
        {"name": "cheap", "kind": "float", "init": {"kind": "float", "value": 0.0}},
        {"name": "expensive", "kind": "float", "init": {"kind": "float", "value": 0.0}},
        {"name": "demand", "kind": "float", "init": {"kind": "float", "value": 80.0}},
        {"name": "feasible", "kind": "bool", "init": {"kind": "bool", "value": False}},
        {"name": "cost", "kind": "float", "init": {"kind": "float", "value": 0.0}},
    ]}],
    "programs": [{
        "name": "dispatch",
        "variables": [
            {"attribute": ref("cheap"), "lower": number(0), "upper": number(60),
             "on_infeasible": {"kind": "float", "value": 0.0}},
            {"attribute": ref("expensive"), "lower": number(0), "upper": number(50),
             "on_infeasible": {"kind": "float", "value": 0.0}},
        ],
        "sense": "minimize",
        "objective": {"op": "add", "args": [
            read("cheap"), {"op": "mul", "args": [number(2), read("expensive")]},
        ]},
        "constraints": [{"name": "supply", "expr": {"op": "add", "args": [
            read("cheap"), read("expensive")
        ]}, "lower": read("demand")}],
        "feasible": ref("feasible"),
        "objective_value": ref("cost"),
    }],
    "indicators": [
        {"name": name, "target": "attribute", "attr": ref(name)}
        for name in ("cheap", "expensive", "feasible", "cost")
    ],
})

result = pyraichu.simulate(model, t_max=1.0, journal=True)
assert result.indicators["cheap"][0] == (0.0, 60.0)
assert result.indicators["expensive"][0] == (0.0, 20.0)
assert result.indicators["feasible"][0] == (0.0, True)
assert result.indicators["cost"][0] == (0.0, 100.0)
assert [r["status"] for r in result.journal if r["record"] == "program_solved"] == ["optimal"]
```

To model a shortfall instead of marking an unmet demand infeasible, add a
nonnegative shortfall decision and put it in the supply constraint. Give it a
large penalty in the objective. When the cheap source is unavailable, a
penalty above both source costs dispatches 50 from the expensive source and
30 shortfall, while feasibility remains true.

```python
import json

body = json.loads(model.json)["model"]
body["components"][0]["attributes"].append({
    "name": "shortfall", "kind": "float",
    "init": {"kind": "float", "value": 0.0},
})
program = body["programs"][0]
program["variables"][0]["upper"] = number(0)
program["variables"].append({
    "attribute": ref("shortfall"), "lower": number(0),
    "on_infeasible": {"kind": "float", "value": 0.0},
})
program["objective"]["args"].append({
    "op": "mul", "args": [number(10), read("shortfall")],
})
program["constraints"][0]["expr"]["args"].append(read("shortfall"))
body["indicators"].append({
    "name": "shortfall", "target": "attribute", "attr": ref("shortfall"),
})
degraded = pyraichu.simulate(pyraichu.load_model(body), t_max=1.0)
assert degraded.indicators["cheap"][0] == (0.0, 0.0)
assert degraded.indicators["expensive"][0] == (0.0, 50.0)
assert degraded.indicators["shortfall"][0] == (0.0, 30.0)
assert degraded.indicators["feasible"][0] == (0.0, True)
```

Decision kinds come from attributes: `float` is continuous, `int` is integer,
and `bool` is binary. Every decision declares an infeasible fallback. An
infeasible solve writes those fallbacks, sets `feasible` to false, and sets an
`objective_value` output to 0. An unbounded program, node-limit stop, or
solver fault ends the trajectory with an error naming the program and date.

The objective and constraints must be affine in the decisions. Coefficients
and bounds may depend on discrete inputs, including state activity and
sensitive-function outputs. Programs cannot read the simulation clock, ODE
or explicit-equation outputs, or allocated channels. Such inputs need a
separate integration design because they change outside the discrete
fixpoint or are resolved after it.

Equal-cost optima use lexicographic minimisation of decisions in declaration
order after the primary objective. `tie_break: none` disables that guarantee;
results list these programs in `provenance.non_unique_programs`. A declared
`tie_break: {"objectives": [{"sense": "minimize", "expr": ...}]}` applies
secondary objectives before the decision-order rule. The HiGHS backend uses
one thread, a fixed seed and a deterministic node cap. Replay is bit-identical
on the same platform and engine build; continuous decision values may differ
in their last bits across platforms. Exact numeric inputs are cached per
worker, and journal solve records expose `cached`.
