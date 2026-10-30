"""A study document requests analyses and run settings.

A study's `simulation` takes the extra keywords of the run it names
(`options`: confidence, quantiles, thread count, solver tolerances, flow
policy), and its `analyses` section requests a fault tree, importance
measures or a minimal-sequence analysis. Each entry is written with the
parameter names of the Python function that runs it, checked against that
function when the study is loaded, so the study and the API share one
vocabulary. Each test runs the study and asks the same question directly.
"""

from __future__ import annotations

import pyraichu
import pytest

LIBRARY_REF = {"name": "pair", "version": "1.0.0"}
HORIZON = 1000.0


def library():
    return {"raichu_library": {"format": 1}, "library": {**LIBRARY_REF, "types": [{
        "name": "Unit",
        "parameters": [{"name": "rate", "kind": "number"}],
        "body": {"attributes": [{"name": "load", "kind": "float", "init": {"kind": "float", "value": 1.0}}],
                 "automata": [{
            "name": "health", "states": ["ok", "ko"], "init": "ok",
            "transitions": [
                {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp",
                 "rate": {"$param": "rate"}, "kind": "failure", "monitored": True},
                {"name": "repair", "source": "ko", "targets": ["ok"], "distrib": "exp",
                 "rate": 0.1, "kind": "repair", "monitored": True},
            ],
        }]},
    }]}}


def system():
    return {"raichu_system": {"format": 1}, "system": {
        "name": "pair", "library": LIBRARY_REF,
        "instances": [{"name": "A", "type": "Unit", "parameters": {"rate": 0.002}},
                      {"name": "B", "type": "Unit", "parameters": {"rate": 0.001}}],
        "indicators": [{"name": "A_ko", "target": "state",
                        "component": "A", "automaton": "health", "state": "ko"}],
        "targets": [{"name": "down", "component": "A", "automaton": "health", "state": "ko"}],
    }}


def load(**sections):
    study = {"raichu_study": {"format": 1}, "study": {"name": "s", "system": "pair", **sections}}
    return pyraichu.load_model(system(), libraries=[library()], study=study)


SIMULATION = {"method": "monte_carlo", "nb_runs": 300, "horizon": HORIZON, "seed": 4,
              "instants": [500.0, HORIZON]}


def test_simulation_options_reach_the_campaign():
    options = {"confidence": 0.9, "quantiles": [0.5], "rtol": 1e-8, "flow": {"tolerance": 1e-10}}
    model = load(simulation={**SIMULATION, "options": options})
    direct = pyraichu.monte_carlo(
        model, nb_runs=300, t_max=HORIZON, samples=[500.0, HORIZON], seed=4,
        confidence=0.9, quantiles=[0.5], rtol=1e-8, flow=pyraichu.FlowConfig(tolerance=1e-10))
    answer = pyraichu.run_study(model)
    assert answer.confidence == 0.9
    assert answer.indicators == direct.indicators


def test_an_unknown_simulation_option_is_refused_when_the_study_is_loaded():
    with pytest.raises(pyraichu.ModelError, match="rtoll"):
        load(simulation={**SIMULATION, "options": {"rtoll": 1e-8}})


def test_a_setting_the_simulation_states_itself_is_refused_as_an_option():
    with pytest.raises(pyraichu.ModelError, match="seed"):
        load(simulation={**SIMULATION, "options": {"seed": 3}})


def test_a_fault_tree_requested_by_the_study_is_the_one_the_api_builds():
    model = load(analyses={"fault_tree": {"targets": ["down"], "cut_set_limit": 1000}})
    tree = pyraichu.run_study(model, section="fault_tree")
    direct = pyraichu.fault_tree(model, targets=["down"], cut_set_limit=1000)
    assert tree.minimal_cut_sets == direct.minimal_cut_sets
    assert tree.open_psa == direct.open_psa


def test_importance_measures_requested_by_the_study_are_the_api_ones():
    params = {"nb_runs": 200, "t_max": HORIZON, "instants": [HORIZON], "target": "down", "seed": 2}
    model = load(analyses={"importance": params})
    answer = pyraichu.run_study(model, section="importance")
    direct = pyraichu.importance(model, **params)
    assert answer.q_target == direct.q_target
    assert answer.basic_events.keys() == direct.basic_events.keys()


def test_a_sequence_analysis_requested_by_the_study_is_the_api_one():
    params = {"nb_runs": 100, "t_max": HORIZON, "seed": 6}
    model = load(analyses={"sequence_analysis": params})
    assert pyraichu.run_study(model, section="sequence_analysis") == pyraichu.analyse_sequences(model, **params)


def test_an_unknown_analysis_parameter_is_refused_when_the_study_is_loaded():
    with pytest.raises(pyraichu.ModelError, match="cut_set_limt"):
        load(analyses={"fault_tree": {"cut_set_limt": 10}})


def test_an_unknown_analysis_is_refused_when_the_study_is_read():
    with pytest.raises(pyraichu.ModelError, match="fault_trees"):
        load(analyses={"fault_trees": {}})


def test_a_study_with_one_analysis_runs_it_without_naming_it():
    model = load(analyses={"fault_tree": {"targets": ["down"]}})
    assert pyraichu.run_study(model).minimal_cut_sets is not None


def test_a_study_with_several_sections_names_the_one_to_run():
    model = load(simulation=SIMULATION, analyses={"fault_tree": {"targets": ["down"]}})
    with pytest.raises(pyraichu.ModelError, match="fault_tree"):
        pyraichu.run_study(model)
    assert pyraichu.run_study(model, section="simulation").nb_runs == 300


def test_a_sequence_campaign_takes_observations_and_a_condition_in_json():
    observation = {"name": "a_late", "component": "A", "attribute": "load", "time": 0.0}
    sequences = {"method": "sequences", "nb_runs": 50, "horizon": HORIZON, "seed": 3,
                 "options": {"observations": [observation],
                             "condition": {"observation": "a_late", "op": ">=", "value": 0.0}}}
    model = load(simulation=sequences)
    direct = pyraichu.run_sequences(
        model, nb_runs=50, t_max=HORIZON, seed=3,
        observations=[pyraichu.Observation(**observation)],
        condition=pyraichu.SequenceCondition(observation="a_late", op=">=", value=0.0))
    answer = pyraichu.run_study(model)
    assert answer.minimal == direct.minimal
    assert answer.condition == direct.condition is not None


def test_a_misspelt_flow_field_is_refused_when_the_study_is_loaded():
    with pytest.raises(pyraichu.ModelError, match="/study/simulation/options.*toleranse"):
        load(simulation={**SIMULATION, "options": {"flow": {"toleranse": 1e-9}}})


def test_an_analysis_missing_a_required_parameter_is_refused_when_the_study_is_loaded():
    with pytest.raises(pyraichu.ModelError, match="/study/analyses/importance.*instants"):
        load(analyses={"importance": {"nb_runs": 10, "t_max": HORIZON}})


def test_the_thread_count_given_to_run_study_wins_over_the_study(monkeypatch):
    seen = {}
    real = pyraichu.monte_carlo

    def spy(model, **kwargs):
        seen["threads"] = kwargs.get("threads")
        return real(model, **kwargs)

    model = load(simulation={**SIMULATION, "options": {"threads": 1}})
    monkeypatch.setattr(pyraichu, "monte_carlo", spy)
    pyraichu.run_study(model, threads=2)
    assert seen["threads"] == 2
