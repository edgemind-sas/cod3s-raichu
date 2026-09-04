"""Regression guards for the muscadet-plugin fixes from the xhigh code
review: the ObjFlow law dispatch must honour the migrated `distrib` key,
ObjFMInst must fail fast on malformed specs rather than expand to a
silent no-op, and a delay-law repair must build a real repair transition.
"""

import pytest

import pyraichu


def _objflow_exp_spec():
    return {
        "name": "x",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "S",
                        "flows_out": [{"name": "f", "var_prod_default": True}],
                        "failure_modes": [
                            {
                                "name": "fm",
                                "distrib": "exp",
                                "failure": 0.01,
                                "repair": 0.1,
                                "failure_cond": "f_fed_out",
                            }
                        ],
                    }
                ]
            }
        },
        "components": [],
        "connections": [],
        "indicators": [],
    }


def test_objflow_exp_failure_mode_is_not_silently_a_delay():
    """The failure-mode dispatch must read the migrated `distrib` key: an
    `exp` law reaches the exponential branch (the rate is a rate, not a
    delay time)."""
    expanded = pyraichu.expand_model(_objflow_exp_spec())
    source = next(c for c in expanded["components"] if c["name"] == "S")
    laws = [t.get("distrib") for a in source["automata"] for t in a["transitions"]]
    assert laws and all(law == "exp" for law in laws)


def _inst_spec(**over):
    base = {
        "type": "ObjFMInst",
        "name": "m",
        "targets": ["E"],
        "failure": {"distrib": "inst", "prob": 0.3},
        "failure_cond": True,
        "failure_effects": {"failed": True},
    }
    base.update(over)
    return {
        "name": "z",
        "plugins": {"muscadet": {"objects": [base]}},
        "components": [
            {
                "name": "E",
                "attributes": [
                    {"name": "failed", "kind": "bool", "init": {"kind": "bool", "value": False}}
                ],
                "ports": [],
                "interfaces": [],
                "automata": [],
                "sensitive_functions": [],
                "equations": [],
            }
        ],
        "connections": [],
        "indicators": [],
    }


def test_objfm_inst_delay_repair_builds_a_repair_transition():
    """A delay-law repair must generate the occ->rep transition (not be
    silently dropped because it carries no `rate`)."""
    expanded = pyraichu.expand_model(_inst_spec(repair={"distrib": "delay", "time": 3}))
    miss = next(c for c in expanded["components"] if c["name"] == "m")
    repair = next(t for t in miss["automata"][0]["transitions"] if t["name"] == "rep")
    assert repair["distrib"] == "delay" and repair["time"] == 3


@pytest.mark.parametrize(
    "over",
    [
        {"targets": []},  # empty targets, old code rejected any count != 1
        {"failure": None, "failure_effects": {}},  # missing failure param
        {"failure": {"distrib": "inst"}},  # dict failure missing prob/gamma
    ],
)
def test_objfm_inst_malformed_spec_fails_fast(over):
    """Malformed ObjFMInst specs must raise at build, not expand to a
    do-nothing (silently over-optimistic) failure mode."""
    spec = _inst_spec(**over)
    if over.get("failure", "keep") is None:
        del spec["plugins"]["muscadet"]["objects"][0]["failure"]
    with pytest.raises((ValueError, pyraichu.ModelError)):
        pyraichu.expand_model(spec)
