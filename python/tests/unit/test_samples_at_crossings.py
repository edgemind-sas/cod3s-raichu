"""Dense samples requested at the very instant of a watched crossing.

Asking for a sample exactly at the date a watched transition fires must
record it, once, in time order, and must not shift the samples after it.
"""

import pyraichu


def _num(x):
    return {"op": "const", "value": {"kind": "float", "value": x}}


def _level():
    return {"op": "attr", "attr": {"component": "T", "attribute": "level"}}


def _model():
    """The level rises at 1 m/h from 0; a watched transition fires at 2 m,
    which stops the filling."""
    filling = {"op": "state_active",
               "state": {"component": "T", "automaton": "mode", "state": "filling"}}
    return pyraichu.load_model({
        "name": "crossing_samples",
        "components": [{
            "name": "T",
            "attributes": [{"name": "level", "kind": "float",
                            "init": {"kind": "float", "value": 0.0}}],
            "automata": [{"name": "mode", "states": ["filling", "full"], "init": "filling",
                          "transitions": [{"name": "stop", "source": "filling", "targets": ["full"],
                                           "distrib": "watched",
                                           "guard": {"op": "cmp", "cmp": "ge",
                                                     "lhs": _level(), "rhs": _num(2.0)}}]}],
            "equations": [{"target": "level", "kind": "ode",
                           "expr": {"op": "if", "cond": filling, "then": _num(1.0),
                                    "otherwise": _num(0.0)}}],
        }],
        "indicators": [{"name": "level", "target": "attribute",
                        "attr": {"component": "T", "attribute": "level"}},
                       {"name": "filling", "target": "state", "component": "T",
                        "automaton": "mode", "state": "filling"}],
    })


def test_a_sample_at_the_crossing_is_recorded_once_and_in_order():
    model = _model()
    crossing = pyraichu.simulate(model, t_max=5.0).events[0].time
    instants = sorted({0.5 * k for k in range(11)} | {crossing})
    samples = pyraichu.simulate(model, t_max=5.0, samples=instants).samples["level"]
    assert [t for t, _ in samples] == instants
    value_at = dict(samples)
    assert abs(value_at[crossing] - 2.0) < 1e-6
    assert all(abs(value_at[t] - min(t, 2.0)) < 1e-6 for t in instants)


def test_a_sample_at_the_crossing_reads_the_state_before_the_event():
    """As a sample at a dated event's instant does (model schema, edge
    effects): the transition firing at that instant is not yet seen."""
    model = _model()
    crossing = pyraichu.simulate(model, t_max=5.0).events[0].time
    run = pyraichu.simulate(model, t_max=5.0, samples=[1.0, crossing, 3.0])
    assert run.samples["filling"] == [(1.0, 1.0), (crossing, 1.0), (3.0, 0.0)]
