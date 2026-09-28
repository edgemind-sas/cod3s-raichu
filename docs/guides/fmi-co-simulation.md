# FMI co-simulation import

Importing an FMU executes native code supplied by that archive in the RAICHU
process. There is no sandbox. Use an FMU only when you trust its source, and
grant execution explicitly with `allow_fmu_import=True` when loading the model.

RAICHU accepts FMI 2.0 and FMI 3.0 co-simulation units. The FMU retains its
own solver. RAICHU advances its own automata and equations, exchanging values
with the unit at declared communication points. Model exchange is outside this
interface. Supply a finite simulation horizon: a unit with regular
communication points cannot run to an infinite horizon.

## Declare a unit

The model body has an `fmu_units` array. Each entry names an `.fmu` archive,
its communication step in model time units, input bindings from model
attributes to FMU variables, output bindings in the reverse direction, and
optional parameter start values. For example:

```json
{
  "fmu_units": [
    {
      "name": "plant",
      "path": "fmus/plant.fmu",
      "step": 0.1,
      "inputs": [
        {"attribute": {"component": "controller", "attribute": "command"}, "variable": "u"}
      ],
      "outputs": [
        {"attribute": {"component": "plant", "attribute": "temperature"}, "variable": "y"}
      ],
      "parameters": [
        {"variable": "gain", "value": {"kind": "float", "value": 2.0}}
      ]
    }
  ]
}
```

This fragment belongs inside a complete model body. Seal that body with
`pyraichu.seal(...)` or `Model::to_json()` so its format envelope declares
the `fmi` feature. The declaration lets older readers refuse the model; it
does not grant permission to run native code. Relative FMU paths resolve
against the model document's directory when loading a file, or the current
working directory for an in-memory document.

## Communication semantics

A unit with step `h` has regular points at `t0 + k × h`, using an integer grid
index. RAICHU holds its outputs between points. At a point, model transitions
due at that date fire first; the unit then advances with the inputs sampled at
the previous point. RAICHU writes all due units' outputs together, settles
sensitive functions and watched guards, then samples inputs for the next
interval. Several units therefore use Jacobi coupling: changing their order
in the document does not change the result.

If a model event changes an input between regular points, a unit that declares
variable communication step size receives an additional point at the event
date. Fixed-step units keep their previously sampled input until the next
regular point. An additional point never shifts the regular grid. A watched
threshold on an FMU output can fire only when that output changes at a point;
its date resolution is the communication step, not RAICHU's ODE event
tolerance.

Choose `h` from the fastest change that matters to the study. For example, if
a controller can change its command every 0.5 time units and a crossing date
must be reported within 0.1 time units, start with `h = 0.1`, then rerun at
`h = 0.05` and compare the reported crossings and study estimates. A smaller
step costs more FMU calls; convergence of the quantity of interest is the
evidence that the chosen step is adequate.

The engine records each imported unit's FMI version, generating tool and
version, instantiation token, and SHA-256 hash of the archive in result
provenance. The hash identifies the bytes used even if the path changes.

Monte-Carlo replicas use separate FMU instances. If a unit forbids multiple
instances in one process, an ordinary campaign runs serially and reports the
unit responsible; a campaign that explicitly requires parallel execution is
refused. Discretised sequence exploration needs the unit's state serialization
capability, because each branch must restore the same native state. Exact
exploration and structural fault-tree generation do not accept opaque FMUs.
