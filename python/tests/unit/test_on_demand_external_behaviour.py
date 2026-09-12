"""An on-demand failure mode under an external behaviour.

A mode whose occurrence is a Bernoulli draw on solicitation (`inst`) and
whose effects reach its targets from outside (`external`,
`external_rep_indep`) used to be refused at import: "an on-demand (inst)
occurrence is only expanded with the `internal` behaviour". The refusal
was not an engine limit. There are two readers of an on-demand
occurrence, and the mode was handed to the one with no behaviour
machinery: `ObjFMInst` is the dedicated on-demand expander, `internal`
only, while `ObjFM` is the unified one and reads the whole 3x3 law
matrix, the `inst` cell included, under all three behaviours.

A behaviour says how effects reach the targets, an occurrence law says
what fires the mode: the two are orthogonal, so the fix is a routing one.
What is pinned here:

- an external on-demand mode routes to the unified expander and keeps its
  law, its behaviour and the parked-state name a study pins, in the native
  `cls: ObjMode2S` wire and in the legacy `cls: ObjFMInst` one;
- `internal` deliberately keeps its own expander: the two build the same
  states and edges but not the same monitoring, so moving it would change
  what every already-validated on-demand study reports;
- the mode expands and runs, with the control attribute and the mirror
  automaton an external behaviour means;
- the one combination that stays refused, by name: an on-demand law in the
  RETURN direction under `external_rep_indep`, where that law is each
  target's own repair and has no solicitation to draw on.
"""

import pyraichu
import pytest
from pyraichu.importers import translate_study
from pyraichu.plugins import expand_model

#: The solicitation the draw fires on: a demand clock that rises at 2.0
#: and stays high.
DEMAND = [[{"obj": "GEN", "automaton": "sig", "state": "high", "ope": "==", "value": True}]]


def _flag(name):
    """A target component carrying the one attribute the mode writes."""
    return {
        "name": name,
        "attributes": [
            {"name": "failed", "kind": "bool", "init": {"kind": "bool", "value": False}}
        ],
        "ports": [],
        "interfaces": [],
        "equations": [],
        "sensitive_functions": [],
        "automata": [],
    }


def _clock(name, automaton, rises_at):
    """A two-state clock that rises once and never falls back within any
    horizon these tests use."""
    return {
        "name": name,
        "attributes": [],
        "ports": [],
        "interfaces": [],
        "equations": [],
        "sensitive_functions": [],
        "automata": [
            {
                "name": automaton,
                "states": ["low", "high"],
                "init": "low",
                "transitions": [
                    {
                        "name": "rise",
                        "source": "low",
                        "targets": ["high"],
                        "on_interruption": "reset",
                        "distrib": "delay",
                        "time": rises_at,
                    },
                    {
                        "name": "fall",
                        "source": "high",
                        "targets": ["low"],
                        "on_interruption": "reset",
                        "distrib": "delay",
                        "time": 1e6,
                    },
                ],
            }
        ],
    }


def _topology(targets):
    return {
        "name": "on_demand_external",
        "plugins": {"muscadet": {"objects": []}},
        "components": [_flag(name) for name in targets] + [_clock("GEN", "sig", 2.0)],
        "connections": [],
        "indicators": [],
    }


def _study(failure_mode):
    return {
        "name": "on_demand_external",
        "failure_modes": [failure_mode],
        "events": [
            {
                "cls": "ObjEvent",
                "name": "ER",
                "cond": [[{"obj": "E1", "attr": "failed", "ope": "==", "value": True}]],
                "tempo_occ": 0.0,
                "tempo_not_occ": 0.0,
                "enabled": True,
            }
        ],
        "targets": [{"name": "ER", "enabled": True}],
        "indicators": [
            {
                "component": "^ER$",
                "attr_name": "^occ$",
                "attr_type": "ST",
                "stats": ["mean"],
                "measure": "sojourn-time",
                "enabled": True,
            }
        ],
        "simulation": {"nb_runs": 10, "schedule": [{"instant": 10.0}]},
    }


def _spec(failure_mode):
    """The plugin spec the importer emits for one failure mode."""
    objects, _indicators, _simulation, _measures = translate_study(_study(failure_mode))
    return next(o for o in objects if o["type"] != "ObjEvent")


def _model(failure_mode):
    objects, indicators, _simulation, _measures = translate_study(_study(failure_mode))
    model = _topology(failure_mode["targets"])
    model["plugins"]["muscadet"]["objects"].extend(objects)
    model["indicators"].extend(indicators)
    return model


def _native_on_demand():
    """One on-demand mode in the native wire the production translator
    emits: an `inst` occurrence law, a timed return, and the parked-state
    name the study pins."""
    return {
        "cls": "ObjMode2S",
        "fm_name": "miss",
        "targets": ["E1"],
        "behaviour": "internal",
        "occ_state": "occ",
        "not_occ_state": "rep",
        "occ_parked_state": "not_occ",
        "occ_law": {"cls": "inst", "prob": [0.3]},
        "not_occ_law": {"cls": "delay", "time": [4.0]},
        "occ_param_name": ["gamma"],
        "not_occ_param_name": ["ttr"],
        "failure_cond": DEMAND,
        "occ_effects": {"failed": True},
        "enabled": True,
    }


def _legacy_on_demand():
    """The same mode under the historical vocabulary: scalar gammas on the
    occurrence face, an exponential rate on the return one."""
    return {
        "cls": "ObjFMInst",
        "fm_name": "miss",
        "targets": ["E1"],
        "behaviour": "internal",
        "failure_param": [0.3],
        "repair_param": [0.1],
        "failure_cond": DEMAND,
        "failure_effects": {"failed": True},
        "enabled": True,
    }


EXTERNAL = ["external", "external_rep_indep"]


# --- routing ------------------------------------------------------------------


@pytest.mark.parametrize("behaviour", EXTERNAL)
def test_a_native_external_on_demand_mode_routes_to_the_unified_expander(behaviour):
    fm = _native_on_demand()
    fm["behaviour"] = behaviour
    spec = _spec(fm)
    assert spec["type"] == "ObjFM"
    assert spec["behaviour"] == behaviour
    assert spec["failure"] == [{"law": "inst", "prob": 0.3}]
    # The parked-state name the study pinned survives the routing.
    assert spec["absorb_state"] == "not_occ"


@pytest.mark.parametrize("behaviour", EXTERNAL)
def test_the_legacy_dialect_is_routed_too_with_its_numbers_spelled_out(behaviour):
    """The legacy dialect writes bare numbers where the unified expander
    reads law dicts: the translation spells them out, so one mode written
    in either dialect is one mode."""
    fm = _legacy_on_demand()
    fm["behaviour"] = behaviour
    spec = _spec(fm)
    assert spec["type"] == "ObjFM"
    assert spec["failure"] == [{"law": "inst", "prob": 0.3}]
    assert spec["repair"] == [{"law": "exp", "rate": 0.1}]


def test_the_legacy_dialect_keeps_the_inactive_return_convention():
    """A return rate of 0 is the inactive marker (the failure state stays
    absorbing), and it must stay one through the translation: read as an
    exponential of rate 0 the mode would repair never, which is a
    different model from having no return edge at all."""
    fm = _legacy_on_demand()
    fm["behaviour"] = "external"
    fm["repair_param"] = [0.0]
    assert _spec(fm)["repair"] == [None]


def test_an_internal_on_demand_mode_keeps_its_own_expander():
    """The routing moves the external behaviours and nothing else."""
    assert _spec(_native_on_demand())["type"] == "ObjFMInst"
    assert _spec(_legacy_on_demand())["type"] == "ObjFMInst"


@pytest.mark.parametrize("behaviour", EXTERNAL)
def test_a_timed_occurrence_is_untouched_by_the_routing(behaviour):
    fm = _native_on_demand()
    fm["behaviour"] = behaviour
    fm["occ_law"] = {"cls": "exp", "rate": [0.1]}
    fm["occ_param_name"] = ["lambda"]
    spec = _spec(fm)
    assert spec["type"] == "ObjFM"
    assert spec["failure"] == [{"law": "exp", "rate": 0.1}]
    assert "absorb_state" not in spec


# --- expansion and run --------------------------------------------------------


@pytest.mark.parametrize("behaviour", EXTERNAL)
def test_an_external_on_demand_mode_expands_to_the_draw_and_the_mirror(behaviour):
    fm = _native_on_demand()
    fm["behaviour"] = behaviour
    expanded = expand_model(_model(fm))
    mode = next(c for c in expanded["components"] if c["name"] == "miss")
    # The draw parks in the state the study named, beside the resting one.
    assert mode["automata"][0]["states"] == ["rep", "occ", "not_occ"]
    # An external behaviour drives the target from outside: a control
    # attribute on the mode, a mirror automaton grafted into the target.
    assert [a["name"] for a in mode["attributes"]] == ["ctrl_miss_E1"]
    target = next(c for c in expanded["components"] if c["name"] == "E1")
    assert [a["name"] for a in target["automata"]] == ["miss"]


def test_the_parked_state_defaults_beside_the_failure_state():
    """Left unsaid, the parked state is `not_<failure_state>`, which is
    what the dedicated on-demand expander defaults to as well: the two
    dialects of one mode keep naming the same state."""
    fm = _native_on_demand()
    fm["behaviour"] = "external"
    del fm["occ_parked_state"]
    expanded = expand_model(_model(fm))
    mode = next(c for c in expanded["components"] if c["name"] == "miss")
    assert mode["automata"][0]["states"] == ["rep", "occ", "not_occ"]


@pytest.mark.parametrize("behaviour", EXTERNAL)
def test_an_external_on_demand_mode_runs(behaviour):
    """The point of the routing is that a study of this shape produces
    results: a spec that translates but builds a model nothing can run
    would satisfy half of it and none of the rest."""
    fm = _native_on_demand()
    fm["behaviour"] = behaviour
    # A draw that always loses its bet keeps the run deterministic: the
    # demand rises at 2.0, the mode fails there, and the feared event
    # follows on the target.
    fm["occ_law"] = {"cls": "inst", "prob": [1.0]}
    result = pyraichu.simulate(pyraichu.load_model(_model(fm)), t_max=3.0)
    occurrences = [(t, v) for t, v in result.indicators["ER_occ"] if v == 1.0]
    assert occurrences and occurrences[0][0] == pytest.approx(2.0)


# --- the refusal that stays ---------------------------------------------------


def test_an_on_demand_return_law_under_rep_indep_is_refused_by_name():
    """An on-demand OCCURRENCE composes with this behaviour; an on-demand
    RETURN does not. Under `external_rep_indep` the order-1 return law is
    each target's own repair, a timed edge with no solicitation to draw
    on, so an `inst` law there has nothing to be drawn by. Refused by
    name rather than approximated into a delay."""
    spec = {
        "type": "ObjFM",
        "name": "miss",
        "targets": ["E1"],
        "behaviour": "external_rep_indep",
        "failure": [{"law": "inst", "prob": 0.3}],
        "repair": [{"law": "inst", "prob": 0.5}],
        "failure_effects": {"failed": True},
        "failure_cond": DEMAND,
    }
    model = _topology(["E1"])
    model["plugins"]["muscadet"]["objects"].append(spec)
    with pytest.raises(ValueError, match="miss.*on-demand.*solicitation"):
        expand_model(model)
