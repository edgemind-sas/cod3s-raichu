# Running a muscadet model on RAICHU

[muscadet](https://github.com/edgemind-sas/muscadet) is a modelling façade for
reliability and flow networks. Since it grew an **engine extension point**, it
knows no engine but its own reference one: an engine registers itself, and a
run says where it happens.

`pyraichu` registers there. A modeller writes muscadet and picks RAICHU:

<!-- skip -->
```python
import muscadet
import muscadet.kb.rbd            # Source / Block / Target

system = muscadet.System(name="rbd")
system.add_component(name="S", cls="Source")
system.add_component(name="B", cls="Block")
system.add_component(name="T", cls="Target")
system.connect_flow(source="S", target="B", flow_name="is_ok")
system.connect_flow(source="B", target="T", flow_name="is_ok")
system.comp["B"].add_exp_failure_mode(
    name="failure", failure_rate=0.1, repair_rate=0.5,
    failure_effects=[("is_ok_fed_available_out", False)])
system.add_indicator_var(component="^T$", var="^is_ok_fed_in$", stats=["mean"])

params = {"nb_runs": 10_000, "schedule": [0, 5, 10, 20], "seed": 42}

system.simulate(params)                    # PyCATSHOO, the reference path
estimates = system.simulate(params, engine="raichu")   # RAICHU
```

**Nothing imports `pyraichu`.** The choice of engine is a string. Installing
the package is the registration: the wheel advertises itself under the
`muscadet.engines` entry point group, muscadet discovers it on first need, and
a third engine would be added the same way without a line of muscadet
changing.

Registering explicitly stays available for a caller who wants it, and is what
an application does when it pins its engines rather than discovering them:

<!-- skip -->
```python
from pyraichu.muscadet_engine import register
register()
```

## What crosses, and what comes back

What the engine receives is the **system declaration**
(`muscadet.declare.system_spec`): a document, checked by muscadet on the way
out, never the live system. The run parameters travel beside it, in cod3s's
own vocabulary, because they configure a run rather than describe a system:
the schedule's instants become the sample instants and its last one the
horizon, which is what makes the two engines answer at the same dates.

The answer is the engine's own, handed back untouched. The reference path
writes its indicator values onto the live system and returns nothing; RAICHU
returns `McEstimates`, so a study reads the result rather than the system:

<!-- skip -->
```python
target = estimates.indicators["T_is_ok_fed_in"]
list(zip(target.instants, target.mean))
```

An interactive session takes the same declaration and the same build
(`system.isimu_start(engine="raichu")`), which is deliberate: a model that
behaved differently one step at a time than in bulk would be a divergence
nothing reports. It returns a `pyraichu.Interactive` session, stepped
RAICHU's way and not cod3s's.

## Where the declaration lands

`pyraichu.muscadet` used to be a second authoring interface, mirroring
muscadet's idioms over RAICHU. It is now the **internal adapter**: the
declaration is read by `pyraichu.declare.build_system`, built through that
layer, and a modeller never names it. Nothing new enters it; an alignment
between the two vocabularies lands in the declaration reader or in
`pyraichu.muscadet_engine`, never in the mirror.

Two places in the vocabulary needed reconciling, and both are worth knowing
because they are what a modeller meets:

- **a failure mode's grip on a discrete output.** muscadet says "this mode
  kills this output" with an effect on the output's availability, spelled
  either as the flow's name or as its `{flow}_fed_available_out` variable.
  This layer says it with the mode's `targets` and keeps its effects for what
  a mode does to a *continuous* output, which is derate it. Both spellings are
  read, and a repair effect restoring availability is absorbed: the gate
  returns on its own when the mode leaves its failing state;
- **a mode that gates nothing.** Here an empty target list means *every*
  discrete output of the component, so a muscadet mode naming none of them --
  one that only derates a continuous output, or only flips a state an
  indicator watches -- is **refused by name** rather than built into a mode
  that silently kills outputs the declaration leaves alone.

What has no counterpart is refused, never approximated: an indicator on an
automaton's state (the two layers do not name states alike), a PyCATSHOO
trace, a run parameter this engine does not read. A knob of the engine itself
travels as a keyword of the run, beside the parameters:

<!-- skip -->
```python
system.simulate(params, engine="raichu", threads=8, quantiles=[0.05, 0.95])
```

Where an engine does what muscadet defines but does it *otherwise*, the
statement belongs to muscadet's own conformance registry rather than here:
`python -m muscadet.conformance raichu`.

## What replaces `cod3s.ComponentInstance.to_bkd_raichu`

`cod3s` carried a second backend on its component specification: a
`class_name_bkd["raichu"]` naming `pyraichu.muscadet.System` and the component
classes behind it, so that `spec.to_bkd("raichu")` built a RAICHU system
directly.

**It disappears rather than re-points.** Its whole purpose was to give cod3s a
second target, and it did so by making cod3s name a RAICHU class path: every
specification carried two class names per component, and adding a third engine
would have meant a third. That is exactly the coupling the extension point
removes. The route now is one target and one choice:

<!-- skip -->
```python
system = spec.to_bkd("pycatshoo")        # ONE build, muscadet's
system.simulate(params, engine="raichu") # the engine is a run parameter
```

The consequences, stated so nobody looks for the old path:

- a specification declares its components once, with no engine in it. The
  `raichu` entries of `class_name_bkd` become dead weight and their removal is
  a cod3s change, not a RAICHU one;
- a model written for the second backend keeps working while `to_bkd_raichu`
  exists, and buys nothing: the same model reaches the same engine through the
  declaration, and reaches it *checked*, muscadet validating the document on
  the way out;
- the cross-validation that covered the second backend is superseded by the
  one covering the route that survives: one muscadet model, both engines, a
  live PyCATSHOO oracle rather than a recorded trajectory. See
  [Cross-validation](../benchmarks/cross-validation.md).
