"""One study, three engines, one envelope, from Python.

`pyraichu.quantify(model, study, method=...)` hands the same `Study` to
Monte-Carlo simulation, exact exploration or discretised exploration and
returns a `Quantification`: the method as applied, the provenance, the
probability that the target is the first declared target reached by the
horizon with its uncertainty, and the engine's own result unchanged. These
tests pin the Python surface on a model with a closed form; the Rust suite
of `raichu-quantify` pins the contract.
"""

import json
import math

import pyraichu
import pytest
from pyraichu.plugins import expand_model

RATE_A = 0.1
RATE_B = 0.03
HORIZON = 10.0


def _objfm(name, target, rate):
    return {
        "type": "ObjFM",
        "name": name,
        "targets": [target],
        "failure": [{"law": "exp", "rate": rate}],
        "repair": [{"law": "exp", "rate": 0.5}],
        "failure_effects": {"flow": False},
    }


def _pair(rate_b=RATE_B):
    """Two non-repairable components A and B, one ObjFM each, feared
    event: both down. `P = (1 - e^{-a t}) (1 - e^{-b t})`."""
    document = {
        "name": "pair",
        "plugins": {
            "muscadet": {
                "objects": [
                    _objfm("fa", "A", RATE_A),
                    _objfm("fb", "B", rate_b),
                    {
                        "type": "ObjEvent",
                        "name": "system_down",
                        "target": True,
                        "cond": [
                            [
                                {
                                    "obj": "A",
                                    "attr": "flow",
                                    "ope": "==",
                                    "value": False,
                                },
                                {
                                    "obj": "B",
                                    "attr": "flow",
                                    "ope": "==",
                                    "value": False,
                                },
                            ]
                        ],
                    },
                ]
            }
        },
        "components": [
            {
                "name": n,
                "attributes": [
                    {
                        "name": "flow",
                        "kind": "bool",
                        "init": {"kind": "bool", "value": True},
                    }
                ],
            }
            for n in ("A", "B")
        ],
    }
    body = expand_model(document)
    for component in body["components"]:
        for automaton in component.get("automata", []):
            automaton["transitions"] = [
                t for t in automaton["transitions"] if t.get("kind") != "repair"
            ]
    return pyraichu.load_model(body)


def _closed_form(rate_b=RATE_B):
    return (1.0 - math.exp(-RATE_A * HORIZON)) * (1.0 - math.exp(-rate_b * HORIZON))


STUDY = pyraichu.Study("system_down", HORIZON, instants=(2.5, 5.0, HORIZON), seed=11)


def test_the_three_methods_agree_with_the_closed_form():
    model = _pair()
    closed = _closed_form()

    mc = pyraichu.quantify(
        model, STUDY, method="monte_carlo", nb_runs=4000, confidence=0.99
    )
    assert isinstance(mc, pyraichu.Quantification)
    assert mc.method == "monte_carlo"
    p = mc.probability
    assert p.kind == "confidence_interval"
    assert p.interval_method == "wilson"
    assert p.level == 0.99
    assert p.replicas == 4000
    assert p.estimate == p.reached / 4000
    assert p.low <= closed <= p.high

    exact = pyraichu.quantify(model, STUDY, method="exact")
    assert exact.probability.kind == "bounds"
    assert exact.probability.low == pytest.approx(closed, rel=1e-12)
    assert exact.probability.high == pytest.approx(closed, rel=1e-12)
    assert exact.probability.error_estimate is None
    assert exact.probability.inconclusive is False
    assert isinstance(exact.detail, pyraichu.Exploration)

    disc = pyraichu.quantify(model, STUDY, method="discretised")
    d = disc.probability
    assert d.error_estimate is not None
    assert abs(d.low - exact.probability.low) <= d.error_estimate
    assert abs(d.high - exact.probability.high) <= d.error_estimate
    assert p.low <= exact.probability.low <= exact.probability.high <= p.high


def test_the_monte_carlo_detail_is_the_stop_at_targets_campaign():
    model = _pair()
    q = pyraichu.quantify(
        model, STUDY, method="monte_carlo", nb_runs=300, quantiles=[0.5]
    )
    direct = pyraichu.monte_carlo(
        model,
        300,
        HORIZON,
        list(STUDY.instants),
        seed=STUDY.seed,
        quantiles=[0.5],
        stop_at_targets=True,
    )
    assert isinstance(q.detail, pyraichu.McEstimates)
    assert q.detail == direct


def test_the_exploration_detail_is_the_explorer_result():
    model = _pair()
    q = pyraichu.quantify(model, STUDY, method="discretised", level=4)
    direct = pyraichu.explore(
        model, "system_down", HORIZON, algorithm="discretised", level=4
    )
    assert q.detail == direct
    assert q.settings["level"] == 4


def test_the_provenance_records_what_the_method_uses():
    model = _pair()
    mc = pyraichu.quantify(model, STUDY, method="monte_carlo", nb_runs=50)
    assert mc.seed == 11
    assert mc.instants == (2.5, 5.0, HORIZON)
    assert mc.engine_version == pyraichu.__version__
    assert mc.model == "pair"
    assert mc.target == "system_down"
    assert mc.horizon == HORIZON
    assert mc.model_hash.startswith("sha256:") and len(mc.model_hash) == 71

    exact = pyraichu.quantify(model, STUDY, method="exact")
    assert exact.seed is None and exact.instants is None
    other_seed = pyraichu.Study("system_down", HORIZON, seed=99, threads=1)
    assert (
        pyraichu.quantify(model, other_seed, method="exact").to_json()
        == exact.to_json()
    )
    assert exact.model_hash == mc.model_hash
    assert (
        pyraichu.quantify(_pair(rate_b=0.04), STUDY, method="exact").model_hash
        != exact.model_hash
    )


def test_the_envelope_reads_back_equal(tmp_path):
    model = _pair()
    for method, settings in [
        ("monte_carlo", {"nb_runs": 200}),
        ("exact", {}),
        ("discretised", {"level": 4}),
    ]:
        q = pyraichu.quantify(model, STUDY, method=method, **settings)
        text = q.to_json()
        assert json.loads(text)["format"] == "raichu.quantification"
        back = pyraichu.read_quantification(text)
        assert back == q
        assert back.to_json() == text
        path = tmp_path / f"{method}.json"
        q.to_json(path)
        assert pyraichu.read_quantification(path) == q
        assert pyraichu.read_quantification(str(path)) == q


def test_the_reader_refuses_a_future_version():
    q = pyraichu.quantify(_pair(), STUDY, method="exact")
    document = json.loads(q.to_json())
    document["version"] = 2
    with pytest.raises(pyraichu.SimulationError, match="version"):
        pyraichu.read_quantification(json.dumps(document))


def test_an_invalid_method_name_is_refused_with_the_valid_ones():
    with pytest.raises(pyraichu.SimulationError) as raised:
        pyraichu.quantify(_pair(), STUDY, method="bootstrap")
    message = str(raised.value)
    for name in pyraichu.QUANTIFICATION_METHODS:
        assert f"`{name}`" in message
    with pytest.raises(
        pyraichu.SimulationError, match="`monte_carlo`, `exact`, `discretised`"
    ):
        pyraichu.quantify(_pair(), STUDY)


def test_settings_of_another_method_are_refused():
    with pytest.raises(
        pyraichu.SimulationError, match="`level` does not apply to the `exact` method"
    ):
        pyraichu.quantify(_pair(), STUDY, method="exact", level=4)
    with pytest.raises(
        pyraichu.SimulationError, match="applies to `exact`, `discretised`"
    ):
        pyraichu.quantify(
            _pair(), STUDY, method="monte_carlo", nb_runs=10, max_length=3
        )
    with pytest.raises(pyraichu.SimulationError, match="nb_runs"):
        pyraichu.quantify(_pair(), STUDY, method="monte_carlo")


def test_an_unknown_target_names_the_declared_ones():
    with pytest.raises(pyraichu.SimulationError, match="`system_down`"):
        pyraichu.quantify(_pair(), pyraichu.Study("nope", HORIZON), method="exact")


def _unit(name, rate):
    return {
        "name": name,
        "attributes": [],
        "automata": [
            {
                "name": "health",
                "states": ["ok", "nok"],
                "init": "ok",
                "transitions": [
                    {
                        "name": "fail",
                        "source": "ok",
                        "targets": ["nok"],
                        "distrib": "exp",
                        "rate": rate,
                    }
                ],
            }
        ],
    }


def _nok(name):
    return {
        "op": "state_active",
        "state": {"component": name, "automaton": "health", "state": "nok"},
    }


def test_a_fault_tree_is_still_quantified_by_the_same_name():
    model = pyraichu.load_model(
        {"name": "plant", "components": [_unit("A", 0.1), _unit("B", 0.03)]}
    )
    tree = pyraichu.fault_tree(
        model, {"op": "bool", "bool_op": "and", "args": [_nok("A"), _nok("B")]}
    )
    result = pyraichu.quantify(tree, mission_time=HORIZON)
    assert isinstance(result, pyraichu.FaultTreeQuantification)
    assert result.probability == pytest.approx(_closed_form(), rel=1e-12)
    assert tree.quantify(mission_time=HORIZON) == result
    assert pyraichu.quantify(tree=tree, mission_time=HORIZON) == result
    with pytest.raises(TypeError, match="apply to a pyraichu.Model"):
        pyraichu.quantify(tree, STUDY)
    with pytest.raises(TypeError, match="apply to a pyraichu.Model"):
        pyraichu.quantify(tree, method="exact")
    with pytest.raises(TypeError, match="nb_runs"):
        pyraichu.quantify(tree, nb_runs=10)
    with pytest.raises(TypeError, match="Study"):
        pyraichu.quantify(model, "system_down", method="exact")


def test_an_envelope_not_written_by_the_engine_is_not_written_out():
    import dataclasses

    q = pyraichu.quantify(_pair(), STUDY, method="exact")
    derived = dataclasses.replace(q, horizon=99.0)
    with pytest.raises(pyraichu.SimulationError, match="not produced by the engine"):
        derived.to_json()


class _Scalar:
    """Stands in for a numpy scalar: a value with ``item``."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_array_library_scalars_are_accepted_as_settings():
    study = pyraichu.Study("system_down", HORIZON, seed=_Scalar(11))
    q = pyraichu.quantify(_pair(), study, method="monte_carlo", nb_runs=_Scalar(50))
    assert q.seed == 11
    assert q.probability.replicas == 50
