"""Solver parameters outside their domain are refused by name, up front.

`simulate` and `monte_carlo` take the ODE-backend keywords (`rtol`,
`atol`, `max_step`, `tol_event`, `event_resolution`). Only
`event_resolution` used to be checked: a NaN `max_step` was dropped by the
`min` it entered (no step cap at all, silently), a zero one surfaced as a
step-size underflow that read as model stiffness, and a negative tolerance
made the error control meaningless.
"""

import math

import pytest

import pyraichu


def ramp() -> pyraichu.Model:
    return pyraichu.load_model(
        {
            "name": "ramp",
            "components": [
                {
                    "name": "c",
                    "attributes": [
                        {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
                    ],
                    "equations": [
                        {
                            "target": "x",
                            "kind": "ode",
                            "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}},
                        }
                    ],
                }
            ],
        }
    )


REFUSED = [
    ("rtol", 0.0),
    ("rtol", -1e-9),
    ("rtol", math.nan),
    ("atol", -1e-12),
    ("atol", math.inf),
    ("max_step", math.nan),
    ("max_step", 0.0),
    ("max_step", -0.1),
    ("tol_event", 0.0),
    ("tol_event", math.nan),
]


@pytest.mark.parametrize(("name", "value"), REFUSED)
def test_simulate_refuses_a_solver_parameter_outside_its_domain(name, value):
    with pytest.raises(pyraichu.SimulationError, match=f"`{name}`"):
        pyraichu.simulate(ramp(), t_max=1.0, **{name: value})


@pytest.mark.parametrize(("name", "value"), REFUSED)
def test_monte_carlo_refuses_a_solver_parameter_outside_its_domain(name, value):
    with pytest.raises(pyraichu.SimulationError, match=f"`{name}`"):
        pyraichu.monte_carlo(ramp(), nb_runs=2, t_max=1.0, samples=[1.0], **{name: value})


def test_an_unbounded_step_cap_and_a_zero_absolute_tolerance_run():
    result = pyraichu.simulate(ramp(), t_max=1.0, max_step=math.inf, atol=0.0)
    assert result.final_time == 1.0
