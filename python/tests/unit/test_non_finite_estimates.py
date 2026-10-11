"""A Monte-Carlo estimate that is not finite is refused, never written.

JSON has no spelling for a NaN or an infinity: written, it came out as
``null``, which the Monte-Carlo result reader handed on as ``None`` (an
infinity indistinguishable from a NaN) and the quantification envelope
reader refused outright. Both writers now refuse instead, naming the
indicator, the measure and the instant.
"""

import pyraichu
import pytest


def _entry_setting_x_to(numerator, denominator):
    """A state entered at a date drawn uniformly on [1, 3]; on entry, the
    attribute ``x`` becomes ``numerator / denominator``. Entering is the
    study's target."""
    constant = lambda v: {"op": "const", "value": {"kind": "float", "value": v}}
    return pyraichu.load_model(
        pyraichu.seal(
            {
                "name": "finite",
                "components": [
                    {
                        "name": "c",
                        "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
                        "ports": [],
                        "automata": [
                            {
                                "name": "aut",
                                "states": ["off", "on"],
                                "init": "off",
                                "transitions": [
                                    {
                                        "name": "enter",
                                        "source": "off",
                                        "targets": ["on"],
                                        "distrib": "uniform",
                                        "low": 1.0,
                                        "high": 3.0,
                                        "effects": [
                                            {
                                                "target": {"component": "c", "attribute": "x"},
                                                "value": {"op": "div", "lhs": constant(numerator), "rhs": constant(denominator)},
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "indicators": [{"name": "x", "target": "attribute", "attr": {"component": "c", "attribute": "x"}}],
                "targets": [{"name": "entered", "component": "c", "automaton": "aut", "state": "on"}],
            }
        )
    )


@pytest.mark.parametrize("numerator", [0.0, 1.0, -1.0])
def test_a_monte_carlo_result_with_a_non_finite_estimate_is_refused(numerator):
    model = _entry_setting_x_to(numerator, 0.0)
    with pytest.raises(pyraichu.SimulationError, match=r"indicator `x`: its `mean` at t = 2 is"):
        pyraichu.monte_carlo(model, nb_runs=200, t_max=4.0, samples=[0.5, 2.0, 4.0])


@pytest.mark.parametrize("numerator", [0.0, 1.0, -1.0])
def test_a_quantification_with_a_non_finite_estimate_is_refused(numerator):
    model = _entry_setting_x_to(numerator, 0.0)
    study = pyraichu.Study("entered", 4.0, instants=[0.5, 2.0])
    with pytest.raises(pyraichu.SimulationError, match=r"indicator `x`: its `mean` at t = 2 is"):
        pyraichu.quantify(model, study, method="monte_carlo", nb_runs=200)


def test_finite_estimates_are_written_and_read_back():
    model = _entry_setting_x_to(1.0, 4.0)
    estimates = pyraichu.monte_carlo(model, nb_runs=200, t_max=4.0, samples=[0.5, 2.0, 4.0])
    assert estimates.indicators["x"].extremes.max[2] == 0.25
    q = pyraichu.quantify(model, pyraichu.Study("entered", 4.0, instants=[0.5, 2.0]), method="monte_carlo", nb_runs=200)
    assert pyraichu.read_quantification(q.to_json()).to_json() == q.to_json()
