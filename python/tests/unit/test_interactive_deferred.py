"""Deferred draws in an interactive session.

A session opened with ``deferred_draws=True`` arms every stochastic
transition without drawing its date: it never fires one by itself, reports
each armed one with its law and the age it has accumulated, and fires one at
the instant the caller chooses. Replaying the dates another run drew is what
this is for, and reading the law back is what tells two routes to the same
model apart when they would fire the same transition at different rates.
"""

from __future__ import annotations

import pyraichu
import pytest


def _pump(rate: float = 0.5) -> dict:
    return {
        "name": "pump",
        "components": [
            {
                "name": "pump",
                "attributes": [],
                "automata": [
                    {
                        "name": "health",
                        "states": ["ok", "failed"],
                        "init": "ok",
                        "transitions": [
                            {"name": "fail", "source": "ok", "targets": ["failed"],
                             "distrib": "exp", "rate": rate},
                            {"name": "repair", "source": "failed", "targets": ["ok"],
                             "distrib": "delay", "time": 1.0},
                        ],
                    }
                ],
            }
        ],
    }


def test_an_armed_transition_is_reported_with_its_law_and_age():
    session = pyraichu.interactive(_pump(), t_max=10.0, deferred_draws=True)
    session.advance_to(2.0)
    (armed,) = session.deferred()
    assert armed["transition"] == "pump.health.fail"
    assert armed["law"] == {"distrib": "exp", "rate": 0.5}
    assert armed["armed_at"] == 0.0
    assert armed["age"] == 2.0
    assert armed["paused"] is False


def test_a_deferred_transition_never_fires_by_itself():
    session = pyraichu.interactive(_pump(), t_max=10.0, deferred_draws=True)
    session.advance_to(10.0)
    assert session.history() == []
    assert session.state("pump.health") == "ok"


def test_the_caller_fires_it_at_the_date_it_chooses():
    session = pyraichu.interactive(_pump(), t_max=10.0, deferred_draws=True)
    event = session.fire_deferred_at("pump.health.fail", 3.25)
    assert (event.time, event.transition) == (3.25, "pump.health.fail")
    assert session.state("pump.health") == "failed"
    # The deterministic repair then fires on its own, a second later, and
    # the failure is armed again from that instant.
    session.advance_to(5.0)
    assert [(e.time, e.transition) for e in session.history()] == [
        (3.25, "pump.health.fail"),
        (4.25, "pump.health.repair"),
    ]
    (armed,) = session.deferred()
    assert armed["armed_at"] == 4.25


def test_firing_a_transition_that_is_not_armed_is_refused():
    session = pyraichu.interactive(_pump(), t_max=10.0, deferred_draws=True)
    with pytest.raises(pyraichu.SimulationError):
        session.fire_deferred_at("pump.health.repair", 1.0)


def test_a_drawn_session_has_nothing_deferred():
    session = pyraichu.interactive(_pump(), t_max=10.0)
    assert session.deferred() == []


def test_deferred_draws_and_operator_control_exclude_each_other():
    with pytest.raises(ValueError, match="deferred_draws"):
        pyraichu.interactive(_pump(), t_max=10.0, deferred_draws=True, operator_control=True)


def _on_demand() -> dict:
    """A start drawn when a demand rises at 1: running (0.9) or refused."""
    return {
        "name": "on_demand",
        "components": [
            {
                "name": "B",
                "attributes": [],
                "automata": [
                    {"name": "demand", "states": ["idle", "asked"], "init": "idle",
                     "transitions": [{"name": "ask", "source": "idle", "targets": ["asked"],
                                      "distrib": "delay", "time": 1.0}]},
                    {"name": "start", "states": ["ready", "running", "refused"], "init": "ready",
                     "transitions": [{"name": "draw", "source": "ready",
                                      "targets": ["running", "refused"],
                                      "distrib": "inst", "probs": [0.9],
                                      "guard": {"op": "state_active", "state": {
                                          "component": "B", "automaton": "demand", "state": "asked"}}}]},
                ],
            }
        ],
    }


def test_a_branch_choice_waits_and_takes_the_branch_the_caller_forces():
    session = pyraichu.interactive(_on_demand(), t_max=5.0, deferred_draws=True)
    session.advance_to(3.0)
    (armed,) = session.deferred()
    assert (armed["transition"], armed["armed_at"]) == ("B.start.draw", 1.0)
    assert armed["law"]["distrib"] == "inst"
    assert session.state("B.start") == "ready"
    event = session.fire_deferred_at("B.start.draw", 3.0, branch=1)
    assert (event.time, event.to_state) == (3.0, "refused")
