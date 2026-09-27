# Interactive simulation

`simulate` runs one trajectory to the horizon in one call. An **interactive
session** runs the same engine one event at a time, under your control: you
see which transitions are armed, fire the one you choose, force the branch it
takes, move a firing date, and undo. It is the tool for checking a mechanism by
hand ("what happens if the pump fails to start, then gets repaired?") and for
testing a stochastic model deterministically.

```python
import pyraichu

model = pyraichu.load_model({"name": "pump", "components": [{
    "name": "P",
    "automata": [{"name": "mode", "states": ["standby", "running", "failed"],
                  "init": "standby", "transitions": [
        {"name": "start", "source": "standby", "targets": ["running", "failed"],
         "distrib": "inst", "probs": [0.99]},
        {"name": "fail", "source": "running", "targets": ["failed"],
         "distrib": "exp", "rate": 1e-3},
        {"name": "repair", "source": "failed", "targets": ["standby"],
         "distrib": "delay", "time": 10.0}]}]}]})

session = pyraichu.interactive(model, t_max=1000.0, seed=1)
```

`pyraichu.interactive(model, t_max=inf, journal=False, confluence_check=False,
seed=0, rng_stream=0, flow=None)` opens a `pyraichu.Interactive` session. The
model is a `Model`, a JSON string or a dict; a `"plugins"` section is expanded
and validated as `load_model` does. The keywords mean what they mean for
`simulate`, and the same `seed` / `rng_stream` pair replays bit-identically.

## Seeing what can fire

`fireable()` lists the armed transitions, earliest first, each a `Fireable`
with its `index`, qualified `transition` name, `kind` (`"delay"`,
`"stochastic"`, `"inst"` or `"watched"`) and firing `date` (`None` for a
watched boundary not located yet):

```python
[armed] = session.fireable()
assert (armed.transition, armed.kind, armed.date) == ("P.mode.start", "inst", 0.0)
```

A stochastic transition shows the date its law drew, so the list is what a
plain run would do next.

## Firing, and forcing the outcome

`fire(name, to=None)` fires the armed transition `name` and returns the
`Event` (`time`, `transition`, `from_state`, `to_state`). With `to`, the
destination is **forced** to that state instead of being drawn: this is what
makes a 1 % branch reachable on demand.

```python
checkpoint = session.snapshot()

event = session.fire("P.mode.start", to="failed")
assert (event.from_state, event.to_state) == ("standby", "failed")
assert session.state("P.mode") == "failed"
assert [f.transition for f in session.fireable()] == ["P.mode.repair"]
```

Firing a transition that is not armed, or forcing a state that is not one of
its targets, raises `SimulationError`.

`step()` advances to the next scheduled event instead, exactly as a plain run
would, and returns `None` at the horizon. `set_date(name, date)` moves an armed
transition's firing date (to the current time or later), which pins a
stochastic date to a value of your choosing:

```python
session.restore(checkpoint)          # undo the forced failure
assert session.state("P.mode") == "standby"

session.fire("P.mode.start", to="running")
session.set_date("P.mode.fail", 250.0)
event = session.step()
assert (event.transition, event.time) == ("P.mode.fail", 250.0)
assert session.time == 250.0
```

## Inspecting, undoing, restarting

Between events the session answers:

| call | answer |
|---|---|
| `time` | the current simulation time |
| `state("component.automaton")` | the current state name, `None` if unknown |
| `attribute("component.attribute")` | the current value, `None` if unknown |
| `history()` | the events fired so far, chronological |

```python
assert [e.transition for e in session.history()] == ["P.mode.start", "P.mode.fail"]
```

`snapshot()` captures the whole trajectory state (time, states, attributes,
armed transitions, generator position) as an opaque checkpoint, and
`restore(checkpoint)` reinstates it: an undo that can be taken as many times
as wanted, and from which the session continues as if nothing had happened
since. `reset()` goes back to `t = 0` with a fresh generator:

```python
session.reset()
assert session.time == 0.0 and session.history() == []
```

## Through the muscadet route

A muscadet system opens the same session with
`system.isimu_start(engine="raichu")`: see
[Running a muscadet model](muscadet-engine.md). What comes back is a
`pyraichu.Interactive`, stepped as described here.
