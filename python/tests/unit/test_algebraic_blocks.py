"""Linear explicit cycles through the public Python loading and simulation API."""

import pyraichu
import pytest


def constant(value):
    return {"op": "const", "value": {"kind": "float", "value": value}}


def attribute(name):
    return {"op": "attr", "attr": {"component": "c", "attribute": name}}


def model_with(expression):
    return {
        "name": "linear_cycle",
        "components": [
            {
                "name": "c",
                "attributes": [
                    {
                        "name": "x",
                        "kind": "float",
                        "init": {"kind": "float", "value": 0.0},
                    }
                ],
                "equations": [{"target": "x", "kind": "explicit", "expr": expression}],
            }
        ],
        "indicators": [
            {
                "name": "x",
                "target": "attribute",
                "attr": {"component": "c", "attribute": "x"},
            }
        ],
    }


def test_a_linear_cycle_loads_and_solves_without_a_python_solve_call():
    model = pyraichu.load_model(
        model_with(
            {
                "op": "add",
                "args": [
                    {"op": "mul", "args": [constant(0.5), attribute("x")]},
                    constant(1.0),
                ],
            }
        )
    )
    result = pyraichu.simulate(model, t_max=1.0, samples=[0.0, 1.0])
    assert result.samples["x"] == [(0.0, 2.0), (1.0, 2.0)]


@pytest.mark.parametrize(
    "expression, diagnostic",
    [
        (attribute("x"), "singular"),
        ({"op": "min", "args": [attribute("x"), constant(1.0)]}, "min"),
    ],
)
def test_singular_and_nonlinear_cycles_remain_named_refusals(expression, diagnostic):
    with pytest.raises(pyraichu.ModelError) as raised:
        pyraichu.load_model(model_with(expression))
    assert "c.x" in str(raised.value)
    assert diagnostic in str(raised.value)
