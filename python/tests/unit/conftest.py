"""Shared helpers of the unit suite.

This directory holds the tests that exercise RAICHU **on its own**: the
engine through its Python binding, the `pyraichu.muscadet` authoring
layer, `pyraichu.declare` and the plugins. They need no PyCATSHOO oracle,
no recorded reference trajectory and no client corpus, so a failure here
reads as an internal regression and nothing else. The cross-validation
suite lives apart, under `python/tests/validation/`, where a failure
reads as "RAICHU diverges from PyCATSHOO".

The helpers below are generic over a `SimulationResult` and are the
single definition for both suites: the validation conftest loads this
file rather than restating them.
"""

TOL = 1e-9

#: Numerical slack for a quantity or a date tied to a located crossing: a
#: capacity bound, a guard on a continuous quantity, a rule identity
#: change. The event-location tolerance is `1e-10`, so this is four
#: orders of magnitude of margin: enough to absorb it, still far too
#: tight for a crossing noticed only at the next discrete step.
CROSSING_TOL = 1e-6


def at_zero(result, indicator: str) -> float:
    """The value of `indicator` at the initial instant."""
    series = result.indicators[indicator]
    assert series, f"indicator `{indicator}` recorded nothing"
    return series[0][1]


def sampled(result, indicator: str, instant: float) -> float:
    """The value of `indicator` at one of the requested sample dates."""
    for time, value in result.samples[indicator]:
        if time == instant:
            return value
    raise AssertionError(f"`{indicator}` was not sampled at t={instant}")


def fired_at(result, transition: str) -> float:
    """The date `transition` fired, refusing a run where it never did."""
    for event in result.events:
        # Events name a transition by its component and automaton path.
        if event.transition.rsplit(".", 1)[-1] == transition:
            return event.time
    raise AssertionError(
        f"transition `{transition}` never fired; events: {result.events}"
    )


def settled(series):
    """An indicator series reduced to the value each instant settled on.

    The fixpoint propagation records every pass over an attribute, so one
    instant may carry several entries; only the last is the value that
    instant settled on, which is the one a PyCATSHOO indicator reports.
    Reading `result.samples` instead would report the DECLARED initial
    value at t = 0, because the transitions that fire at t = 0 land after
    the initialisation flush, while their `compute_signal_out` is a start
    method and so their t = 0 record is post-initialisation."""
    kept: list[tuple[float, object]] = []
    for time, value in series:
        if kept and kept[-1][0] == time:
            kept[-1] = (time, value)
        else:
            kept.append((time, value))
    return kept


def held_at(series, instant: float):
    """The value a settled series holds at `instant`: the last entry
    dated at or before it."""
    value = series[0][1]
    for time, entry in series:
        if time > instant:
            break
        value = entry
    return value
