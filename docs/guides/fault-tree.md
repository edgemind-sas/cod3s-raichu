# Fault trees

`pyraichu.fault_tree` generates the fault tree explaining why a **top
expression** over the model's states can become true, and hands it back as a
tree, as its minimal cut sets, and as an OpenPSA model-exchange document a
static tool reads.

```python
import pyraichu

def unit(name, guard=None):
    fail = {"name": "fail", "source": "ok", "targets": ["nok"],
            "distrib": "exp", "rate": 1e-3}
    if guard is not None:
        fail["guard"] = guard
    return {"name": name, "automata": [{"name": "health", "states": ["ok", "nok"],
                                        "init": "ok", "transitions": [fail]}]}

def nok(component):
    return {"op": "state_active",
            "state": {"component": component, "automaton": "health", "state": "nok"}}

# B fails only once A has; C fails on its own.
model = pyraichu.load_model({"name": "plant", "components": [
    unit("A"), unit("B", guard=nok("A")), unit("C")]})

top = {"op": "bool", "bool_op": "or",
       "args": [nok("B"), {"op": "bool", "bool_op": "and", "args": [nok("A"), nok("C")]}]}
tree = pyraichu.fault_tree(model, top)
assert tree.minimal_cut_sets == [["A.health.fail", "B.health.fail"],
                                 ["A.health.fail", "C.health.fail"]]
xml = tree.open_psa        # the OpenPSA document a static tool reads
```

## The method, and its hypothesis

The tree is built by **backward chaining**, the method of the reference
engine's own generator, under the same hypothesis, which is the whole content
of the feature:

- every **attribute keeps its initial value**, or the value a `profile` gives
  it by qualified name (`profile={"B.x": 10.0}`). A guard reading only
  attributes is therefore a constant: the transition is always possible, or
  never;
- only **states move**. A state is explained as being the initial one, OR as
  entered by one of the transitions into it; a transition is explained as its
  **basic event** (the draw of its law) AND its source state being reached AND
  its guard holding.

A loop back to a state already being explained further up the same branch
adds no new way in, so a repair loop is handled rather than truncated: `nok`
is reached from `ok`, `ok` is initial, and the repair back to `ok` never
appears in the tree.

The gates are simplified as they are built: constants propagated,
single-child gates collapsed, nested gates of one connective merged. A vote is
written `sum(if(state, 1, 0)) >= k` and becomes an `atleast` gate.

## What it is not

A tree is not the minimal sequences of a simulation (sequence analysis), which
let attributes move and order the events. The two answer different questions.
On a model built through the muscadet layer, a guard reads flow variables that
functions compute from states; frozen, those variables are constants, exactly
as on the reference engine, whose knowledge bases state a separate textual
condition for fault-tree generation for this reason. Write the top expression
over the states you want explained.

## What is refused

The method explains a value becoming **true** through states being entered,
so every expression it crosses has to be monotone in the states. A state read
under a negation, or inside arithmetic that is not a vote, is refused with the
expression named, and so is a transition whose rate reads a state.

## The file

`open_psa` is an OpenPSA MEF document: the gates under `define-fault-tree`,
and in `model-data` one `define-basic-event` per transition draw. An
exponential law is written `exponential` over the system mission time, a
Weibull law `Weibull`, an on-demand branch a probability; a law OpenPSA has no
time-to-failure expression for (delay, log-normal, gamma, uniform, empirical)
is carried as attributes of its event.
