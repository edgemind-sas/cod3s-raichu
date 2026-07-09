# Causal journal

Debugging a discrete-event model often comes down to one question: *why
did that (not) happen?* RAICHU can record a structured, queryable
**causal journal** of a run — every scheduled, fired and dropped
transition, every attribute write with its cause, and the consequence
chain of each event. It is **zero-cost when off** and enabled per run
with `journal=True`:

<!-- skip -->
```python
result = pyraichu.simulate(model, t_max=15.0, journal=True)
query = pyraichu.JournalQuery(result.journal)
```

Below, three questions on two small models.

## "Why didn't it fire?"

A controller requests a worker's job for 4 time units; the job needs 6.
Because the job's guard (`requesting`) turns false at *t = 4*, its
countdown is dropped and it never finishes.

```python
import pyraichu

model = pyraichu.load_model({
    "name": "guarded_job",
    "components": [
        {"name": "C", "automata": [{
            "name": "req", "states": ["on", "off"], "init": "on",
            "transitions": [{"name": "stop", "source": "on", "targets": ["off"],
                             "distrib": "delay", "time": 4.0}]}]},
        {"name": "W", "automata": [{
            "name": "job", "states": ["idle", "done"], "init": "idle",
            "transitions": [{"name": "finish", "source": "idle",
                             "targets": ["done"], "distrib": "delay", "time": 6.0,
                             "guard": {"op": "state_active",
                                       "state": {"component": "C",
                                                 "automaton": "req",
                                                 "state": "on"}}}]}]},
    ],
})

result = pyraichu.simulate(model, t_max=20.0, journal=True)
query = pyraichu.JournalQuery(result.journal)

print(query.why_not_fired("W.job.finish"))
```

`why_not_fired` returns a `TransitionHistory` (with `fired`,
`schedules`, `drops`, `still_pending`) and a human-readable
`explanation`: it was scheduled at *t = 0* for *t = 6*, then **dropped
at t = 4** when its guard turned false. No guesswork.

## "What followed this event?"

`cascade_after` returns the full consequence chain of an event — the
sensitive functions it ran, the attributes it changed, and the
transitions it (re)scheduled or dropped:

```python
print(query.cascade_after("C.req.stop"))
```

Here it reports that firing `C.req.stop` (on → off) is what dropped the
worker's `finish` countdown — linking cause to effect directly.

## "Who changed this attribute?"

`who_changed` lists every write to an attribute, each with the sensitive
function that wrote it and the event that triggered it:

```python
lamp = pyraichu.load_model({
    "name": "lamp",
    "components": [{
        "name": "Lamp",
        "attributes": [{"name": "light", "kind": "bool",
                       "init": {"kind": "bool", "value": False}}],
        "automata": [{"name": "sw", "states": ["off", "on"], "init": "off",
            "transitions": [{"name": "flip", "source": "off", "targets": ["on"],
                             "distrib": "delay", "time": 3.0}]}],
        "sensitive_functions": [{"name": "update_light", "effects": [{
            "target": {"component": "Lamp", "attribute": "light"},
            "value": {"op": "state_active",
                      "state": {"component": "Lamp", "automaton": "sw",
                                "state": "on"}}}]}],
    }],
})

lamp_run = pyraichu.simulate(lamp, t_max=10.0, journal=True)
for change in pyraichu.JournalQuery(lamp_run.journal).who_changed("Lamp.light"):
    print(f"t={change.time}: {change.attribute} "
          f"{change.old} -> {change.new}  (by {change.cause}, "
          f"triggered by {change.trigger})")
```

The light turned on at *t = 3*, written by `Lamp.update_light`, triggered
by the switch flipping — a complete audit trail.

## When to use it

Turn the journal on while building or debugging a model — it explains
dropped transitions, unexpected attribute values and effect cascades far
faster than re-reading the model. Turn it off for production
Monte-Carlo runs, where it adds nothing to the estimates and you want
maximum throughput.
