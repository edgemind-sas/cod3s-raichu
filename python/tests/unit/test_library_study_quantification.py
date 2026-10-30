"""A study document chooses any quantification method.

The `quantification` section of a study states one quantification, as
:func:`pyraichu.quantify` takes it: the feared event (`target`), the
`horizon`, the reporting `instants` and `seed`, and the `method` with its
settings, written `{"name": ..., "settings": {...}}` as the quantification
envelope records it. Each test runs the study and asks the same question
directly of :func:`pyraichu.quantify` on the expanded model.
"""

from __future__ import annotations

import math

import pyraichu
import pytest

LIBRARY_REF = {"name": "units", "version": "1.0.0"}
RATE = 1.0e-3
HORIZON = 100.0


def library():
    return {"raichu_library": {"format": 1}, "library": {**LIBRARY_REF, "types": [{
        "name": "Unit",
        "body": {
            "automata": [{
                "name": "health", "states": ["ok", "ko"], "init": "ok",
                "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"],
                                 "distrib": "exp", "rate": RATE}],
            }],
        },
    }]}}


def system():
    return {"raichu_system": {"format": 1}, "system": {
        "name": "single", "library": LIBRARY_REF,
        "instances": [{"name": "U", "type": "Unit"}],
        "targets": [{"name": "down", "component": "U", "automaton": "health", "state": "ko"}],
    }}


def study(name, settings=None, **extra):
    section = {"target": "down", "horizon": HORIZON,
               "method": {"name": name, "settings": settings or {}}, **extra}
    return {"raichu_study": {"format": 1}, "study": {
        "name": "q", "system": "single", "quantification": section}}


def run(name, settings=None, **extra):
    model = pyraichu.load_model(system(), libraries=[library()], study=study(name, settings, **extra))
    return model, pyraichu.run_study(model)


def direct(model, name, settings=None, **extra):
    question = pyraichu.Study(target="down", horizon=HORIZON, **extra)
    return pyraichu.quantify(model, question, method=name, **(settings or {}))


def test_an_existing_simulation_study_runs_as_before():
    old = {"raichu_study": {"format": 1}, "study": {
        "name": "mc", "system": "single",
        "simulation": {"method": "monte_carlo", "nb_runs": 200, "horizon": HORIZON, "seed": 3}}}
    model = pyraichu.load_model(system(), libraries=[library()], study=old)
    flat = pyraichu.monte_carlo(model, nb_runs=200, t_max=HORIZON, samples=[HORIZON], seed=3)
    assert pyraichu.run_study(model).indicators == flat.indicators
    assert model.quantification is None


def test_an_exact_quantification_meets_the_closed_form():
    model, answer = run("exact")
    expected = 1.0 - math.exp(-RATE * HORIZON)
    assert answer.probability.low == pytest.approx(expected, rel=1e-12)
    assert answer.probability.high == pytest.approx(expected, rel=1e-12)
    assert answer.method == "exact"


@pytest.mark.parametrize("name, settings, extra", [
    ("monte_carlo", {"nb_runs": 2000, "confidence": 0.9}, {"seed": 7}),
    ("monte_carlo", {"nb_runs": 500}, {"seed": 1, "instants": (25.0, 50.0, 100.0)}),
    ("discretised", {"level": 2}, {}),
    ("cross_entropy", {"nb_runs": 2000, "pilot_runs": 500}, {"seed": 5}),
    ("splitting", {"particles": 100, "batches": 4, "importance": {"kind": "cut_sets"}}, {"seed": 2}),
])
def test_each_method_answers_as_quantify_does(name, settings, extra):
    written = {k: list(v) if isinstance(v, tuple) else v for k, v in extra.items()}
    model, answer = run(name, settings, **written)
    assert answer.to_json() == direct(model, name, settings, **extra).to_json()


def test_a_quantification_naming_no_declared_target_is_refused_with_its_name():
    doc = study("exact")
    doc["study"]["quantification"]["target"] = "nowhere"
    with pytest.raises(pyraichu.ModelError, match="nowhere"):
        pyraichu.load_model(system(), libraries=[library()], study=doc)


def test_an_unknown_method_is_refused_when_the_study_is_loaded():
    with pytest.raises(pyraichu.ModelError, match="annealing"):
        pyraichu.load_model(system(), libraries=[library()], study=study("annealing"))


def test_a_setting_of_another_method_is_refused_when_the_study_is_loaded():
    with pytest.raises(pyraichu.ModelError, match="level"):
        pyraichu.load_model(system(), libraries=[library()], study=study("exact", {"level": 2}))


def test_the_schema_names_every_method_the_engine_provides():
    names = pyraichu.library.schema("study")["$defs"]["QuantificationMethodName"]["oneOf"]
    assert tuple(n["const"] for n in names) == pyraichu.QUANTIFICATION_METHODS


def test_a_study_with_both_sections_asks_which_to_run():
    doc = study("exact")
    doc["study"]["simulation"] = {"method": "monte_carlo", "nb_runs": 10, "horizon": HORIZON}
    model = pyraichu.load_model(system(), libraries=[library()], study=doc)
    with pytest.raises(pyraichu.ModelError, match="section"):
        pyraichu.run_study(model)
    assert pyraichu.run_study(model, section="quantification").method == "exact"
    assert pyraichu.run_study(model, section="simulation").nb_runs == 10


def test_a_target_declared_by_the_study_can_be_quantified():
    doc = study("exact")
    doc["study"]["targets"] = [{"name": "lost", "component": "U", "automaton": "health", "state": "ko"}]
    doc["study"]["quantification"]["target"] = "lost"
    bare = system()
    del bare["system"]["targets"]  # the only target is the study's
    model = pyraichu.load_model(bare, libraries=[library()], study=doc)
    assert pyraichu.run_study(model).probability.high == pytest.approx(1.0 - math.exp(-RATE * HORIZON))


def test_a_section_the_study_does_not_carry_is_refused_by_name():
    model, _ = run("exact")
    with pytest.raises(pyraichu.ModelError, match="no `simulation` section"):
        pyraichu.run_study(model, section="simulation")


def test_a_model_loaded_without_a_study_has_nothing_to_run():
    model = pyraichu.load_model(system(), libraries=[library()])
    with pytest.raises(pyraichu.ModelError, match="no run settings"):
        pyraichu.run_study(model)


def test_a_method_refusal_names_the_study_node():
    with pytest.raises(pyraichu.ModelError, match="/study/quantification/method"):
        pyraichu.load_model(system(), libraries=[library()], study=study("exact", {"level": 2}))
