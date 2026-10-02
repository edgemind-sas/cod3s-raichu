"""Interface connections through the binding: one declaration joins two
interfaces, paired by port name, in both directions."""

import json

import pyraichu
import pytest


def float_attribute(name, value):
    return {"name": name, "kind": "float", "init": {"kind": "float", "value": value}}


def received(target, component, port):
    return {
        "target": target,
        "kind": "explicit",
        "expr": {
            "op": "port_agg",
            "agg": "sum",
            "port": {"component": component, "port": port},
        },
    }


def two_way(wiring):
    body = {
        "name": "two_way",
        "components": [
            {
                "name": "A",
                "attributes": [float_attribute("a", 3.0), float_attribute("from_b", 0.0)],
                "ports": [
                    {"name": "p", "dir": "out", "attr": "a"},
                    {"name": "q", "dir": "in"},
                ],
                "interfaces": [{"name": "bus", "ports": ["p", "q"]}],
                "equations": [received("from_b", "A", "q")],
            },
            {
                "name": "B",
                "attributes": [float_attribute("b", 5.0), float_attribute("from_a", 0.0)],
                "ports": [
                    {"name": "q", "dir": "out", "attr": "b"},
                    {"name": "p", "dir": "in"},
                ],
                "interfaces": [{"name": "bus", "ports": ["q", "p"]}],
                "equations": [received("from_a", "B", "p")],
            },
        ],
        "indicators": [
            {
                "name": "a_received",
                "target": "attribute",
                "attr": {"component": "A", "attribute": "from_b"},
            },
            {
                "name": "b_received",
                "target": "attribute",
                "attr": {"component": "B", "attribute": "from_a"},
            },
        ],
    }
    body.update(wiring)
    return json.dumps(body)


BY_INTERFACE = {
    "interface_connections": [
        {
            "from": {"component": "A", "interface": "bus"},
            "to": {"component": "B", "interface": "bus"},
        }
    ]
}

BY_PORTS = {
    "connections": [
        {"from": {"component": "A", "port": "p"}, "to": {"component": "B", "port": "p"}},
        {"from": {"component": "B", "port": "q"}, "to": {"component": "A", "port": "q"}},
    ]
}


def final(result, indicator):
    return result.samples[indicator][-1][1]


def test_one_declaration_carries_both_directions():
    result = pyraichu.simulate(
        pyraichu.load_model(two_way(BY_INTERFACE)), t_max=1.0, samples=[1.0]
    )
    assert final(result, "a_received") == 5.0
    assert final(result, "b_received") == 3.0


def test_it_answers_what_the_port_connections_answer():
    grouped = pyraichu.simulate(
        pyraichu.load_model(two_way(BY_INTERFACE)), t_max=1.0, samples=[0.0, 1.0]
    )
    explicit = pyraichu.simulate(
        pyraichu.load_model(two_way(BY_PORTS)), t_max=1.0, samples=[0.0, 1.0]
    )
    assert grouped.samples == explicit.samples


def test_an_unpaired_port_is_refused_when_the_model_is_built():
    wiring = json.loads(json.dumps(BY_INTERFACE))
    wiring["interface_connections"][0]["to"]["interface"] = "missing"
    with pytest.raises(pyraichu.ModelError, match="B.missing"):
        pyraichu.load_model(two_way(wiring))
