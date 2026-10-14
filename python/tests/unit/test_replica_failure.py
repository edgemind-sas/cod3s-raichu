"""A failing replica refuses its campaign, and says which one it was (#160).

A Monte-Carlo campaign whose replica raised used to surface the replica's
error bare: nothing said which replica it was, so it could not be replayed
alone. The campaign is still refused, because the replicas that fail are the
ones whose trajectory reached the failing state and an estimate over the
others would be biased; but the error now names the replica, carries the
coordinates that replay it with :func:`pyraichu.simulate`, and keeps the class
of what the replica raised.

The model mirrors ``crates/raichu-montecarlo/tests/replica_failure.rs``: a
switch failing at an exponential date of rate 1 turns a demand into one with
no fixed point, so about half the replicas of a horizon of 0.7 fail.
"""

import pickle

import pyraichu
import pytest

SWITCHED = r"""
{
  "name": "switched",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "capacity", "kind": "float",
         "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "capacity",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "if",
           "cond": {"op": "state_active",
                    "state": {"component": "switch", "automaton": "life", "state": "ko"}},
           "then": {"op": "if",
             "cond": {"op": "cmp", "cmp": "gt",
               "lhs": {"op": "attr", "attr": {"component": "supply",
                                              "attribute": "out__alloc__ea"}},
               "rhs": {"op": "const", "value": {"kind": "float", "value": 3.0}}},
             "then": {"op": "const", "value": {"kind": "float", "value": 1.0}},
             "otherwise": {"op": "const", "value": {"kind": "float", "value": 9.0}}},
           "otherwise": {"op": "const", "value": {"kind": "float", "value": 1.0}}}}
      ],
      "allocations": [
        {"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
         "available": {"op": "attr",
           "attr": {"component": "supply", "attribute": "capacity"}},
         "policy": "proportional"}
      ]
    },
    {
      "name": "a",
      "attributes": [
        {"name": "got", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "got", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                  "port": {"component": "a", "port": "input"}}}
      ]
    },
    {
      "name": "switch",
      "ports": [], "attributes": [],
      "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
        "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"],
                         "distrib": "exp", "rate": 1.0}]}]
    }
  ],
  "connections": [
    {"name": "ea", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}}
  ],
  "targets": [{"name": "switched", "component": "switch", "automaton": "life", "state": "ko"}],
  "indicators": [
    {"name": "got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}}
  ]
}
"""

SEED = 11
NB_RUNS = 64
T_MAX = 0.7


def switched():
    return pyraichu.load_model(SWITCHED)


def first_failing(model):
    for replica in range(NB_RUNS):
        try:
            pyraichu.simulate(model, T_MAX, seed=SEED, rng_stream=replica)
        except pyraichu.SimulationError:
            return replica
    raise AssertionError("no replica fails: this test exercises nothing")


def test_a_failing_replica_is_named_and_replays_alone():
    model = switched()
    expected = first_failing(model)
    with pytest.raises(pyraichu.SimulationError) as caught:
        pyraichu.monte_carlo(model, NB_RUNS, T_MAX, [T_MAX], seed=SEED)
    error = caught.value
    assert (error.replica, error.nb_runs, error.seed) == (expected, NB_RUNS, SEED)
    assert f"replica {expected} of {NB_RUNS} failed" in str(error)
    with pytest.raises(pyraichu.SimulationError) as replayed:
        pyraichu.simulate(
            model, T_MAX, samples=[T_MAX], seed=error.seed, rng_stream=error.replica
        )
    assert str(replayed.value) in str(error)


def test_the_reported_replica_does_not_depend_on_the_thread_count():
    model = switched()
    reported = set()
    for threads in (1, 2, 8):
        with pytest.raises(pyraichu.SimulationError) as caught:
            pyraichu.monte_carlo(
                model, NB_RUNS, T_MAX, [T_MAX], seed=SEED, threads=threads
            )
        reported.add(caught.value.replica)
    assert len(reported) == 1, reported


def test_the_coordinates_survive_a_pickle_round_trip():
    """A worker process hands its error back pickled."""
    with pytest.raises(pyraichu.SimulationError) as caught:
        pyraichu.monte_carlo(switched(), NB_RUNS, T_MAX, [T_MAX], seed=SEED)
    copy = pickle.loads(pickle.dumps(caught.value))
    assert (copy.replica, copy.nb_runs, copy.seed, str(copy)) == (
        caught.value.replica,
        NB_RUNS,
        SEED,
        str(caught.value),
    )


# A tank draining at a constant rate, and a valve failing at an exponential
# date: enough to see both the solver settings and the target early-stop.
DRAINING = r"""
{
  "name": "draining",
  "components": [
    {
      "name": "tank",
      "attributes": [
        {"name": "level", "kind": "float", "init": {"kind": "float", "value": 10.0}}
      ],
      "equations": [
        {"target": "level", "kind": "ode",
         "expr": {"op": "const", "value": {"kind": "float", "value": -1.0}}}
      ]
    },
    {
      "name": "valve",
      "ports": [], "attributes": [],
      "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
        "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"],
                         "distrib": "exp", "rate": 1.0}]}]
    }
  ],
  "targets": [{"name": "stuck", "component": "valve", "automaton": "life", "state": "ko"}]
}
"""


def test_simulate_takes_the_campaign_solver_settings():
    """A replica replays under the campaign's own settings, not the defaults."""
    model = pyraichu.load_model(DRAINING)
    default = pyraichu.simulate(model, 5.0, seed=SEED)
    capped = pyraichu.simulate(model, 5.0, seed=SEED, max_step=0.01)
    assert (
        capped.work["solver_steps_accepted"] > default.work["solver_steps_accepted"]
    )
    stopped = pyraichu.simulate(model, 50.0, seed=SEED, stop_at_targets=True)
    assert [e.transition for e in stopped.events] == ["valve.life.fail"]
    assert stopped.final_time == stopped.events[0].time < 50.0


def test_a_refusal_before_any_replica_carries_no_coordinates():
    """Nothing ran, so there is nothing to replay: the guide checks for it."""
    with pytest.raises(pyraichu.SimulationError) as caught:
        pyraichu.monte_carlo(switched(), NB_RUNS, T_MAX, [T_MAX], confidence=2.0)
    assert not hasattr(caught.value, "replica")
