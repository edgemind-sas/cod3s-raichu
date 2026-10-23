"""STRA witness `cyber_3comp`, written as a component type library (U8).

The STRA cyber example (muscadet ``examples/isimu/cyber_3comp``) chains
three compromise modes: ``mdc_a`` escalates privileges on the server,
``mdc_b`` (possible once ``mdc_a`` has occurred) starts the server's
dormant service function, and ``mdc_proc`` (possible once the service flow
reaches the process) inhibits the process's two functions. The study is
deterministic, so the contract is exact:

- the recorded PyCATSHOO run (``STRA/models/cyber_3comp_study/sequences.xml``)
  fires ``mdc_a`` at 10, ``mdc_b`` at 15, ``mdc_proc`` at 23 and both feared
  events at 23.01, which the library route reproduces under public names;
- the platform's declaration route, which builds the same plant from the
  shapes the COD3S platform sends, answers the same values at every instant.

Oracle-free: both routes run on RAICHU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyraichu
import pyraichu.muscadet.declare as declare

FIXTURES = Path(__file__).parent / "library_fixtures" / "cyber_3comp"

#: Instants that fall between events, so both routes read settled values.
INSTANTS = [0.0, 5.0, 9.0, 11.0, 14.0, 16.0, 20.0, 22.0, 24.0, 30.0]

#: (component, attribute) observed on both routes.
OBSERVED = [
    ("Srv", "f_service_fed_out"),
    ("Proc", "f_service_fed_in"),
    ("Proc", "F1_fed_out"),
    ("Proc", "F2_fed_out"),
    ("Proc", "F1_fed_available_out"),
    ("Srv", "f_orange_fed_out"),
]


def library_model():
    return pyraichu.load_system(
        FIXTURES / "system.json", [FIXTURES / "library.json"], FIXTURES / "study.json"
    )


def test_the_chain_fires_at_the_recorded_dates_under_public_names():
    result = pyraichu.simulate(library_model(), t_max=30.0)
    assert [(e.transition, e.time) for e in result.events] == [
        ("Srv.mdc_a.occ", 10.0),
        ("Srv.mdc_b.occ", 15.0),
        ("Proc.mdc_proc.occ", 23.0),
        ("process_F1_lost.ev.occ", 23.01),
        ("process_F2_lost.ev.occ", 23.01),
    ]


def test_the_service_starts_at_15_and_the_functions_stop_at_23():
    result = pyraichu.simulate(library_model(), t_max=30.0)
    assert result.indicators["service"] == [(0.0, False), (15.0, True)]
    assert result.indicators["F1"] == [(0.0, True), (23.0, False)]
    assert result.indicators["F2"] == [(0.0, True), (23.0, False)]


def test_the_study_carries_the_recorded_run_settings_and_targets():
    model = library_model()
    assert model.simulation["horizon"] == 30.0
    assert model.simulation["seed"] == 42
    body = json.loads(model.json)
    body = body.get("model", body)
    assert [t["name"] for t in body["targets"]] == ["process_F1_lost", "process_F2_lost"]


# --- the platform's declaration route, for comparison ---------------------


def flow_in(name: str, logic: str) -> dict:
    return {
        "cls": "FlowIn",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_in_default": False,
        "var_available_in_default": True,
        "logic": logic,
    }


def flow_out(name: str, prod_default: bool, prod_cond: list | None = None) -> dict:
    spec = {
        "cls": "FlowOut",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": True,
        "var_prod_cond_inner_mode": "or",
        "var_prod_default": prod_default,
        "negate": False,
    }
    if prod_cond is not None:
        spec["var_prod_cond"] = prod_cond
    return spec


def flow_object(name: str, flows: list) -> dict:
    return {"name": name, "cls": "ObjFlow", "flows": flows, "capacities": [], "failure_modes": []}


def delay_mode(target: str, mode: str, ttf: float, effects: dict, cond=None) -> dict:
    spec = {
        "name": f"{target}__{mode}",
        "kind": "two_state_mode",
        "cls": "ObjFMDelay",
        "fm_name": mode,
        "targets": [target],
        "target_name": target,
        "failure_effects": effects,
        "failure_param_name": ["ttf"],
        "failure_param": [ttf],
        "repair_cond": False,
        "repair_param_name": ["ttr"],
        "repair_param": [1e9],
    }
    if cond is not None:
        spec["failure_cond"] = cond
    return spec


def connection(source: str, target: str, flow: str) -> dict:
    return {"source": source, "source_box": f"{flow}_out", "target": target, "target_box": f"{flow}_in"}


def declared_document() -> dict:
    """The plant as the platform declares it, with muscadet's spellings."""
    components = [
        flow_object("Alim", [flow_out("power", True)]),
        flow_object(
            "Srv",
            [
                flow_in("power", "and"),
                flow_out("f_orange", True, [["power"]]),
                flow_out("f_blue", True, [["power"]]),
                flow_out("f_service", False),
            ],
        ),
        flow_object(
            "Proc",
            [
                flow_in("f_orange", "and"),
                flow_in("f_blue", "and"),
                flow_in("f_service", "or"),
                flow_out("F1", True, [["f_orange"]]),
                flow_out("F2", True, [["f_blue"]]),
            ],
        ),
        delay_mode("Srv", "mdc_a", 10, {}),
        delay_mode(
            "Srv",
            "mdc_b",
            5,
            {"f_service_prod_available": True},
            cond=[[{"attr": "occ", "obj": "Srv__mdc_a", "value": True}]],
        ),
        delay_mode(
            "Proc",
            "mdc_proc",
            8,
            {"F1_fed_available_out": False, "F2_fed_available_out": False},
            cond=[[{"attr": "f_service_fed_in", "value": True}]],
        ),
    ]
    return {
        "version": "1.0.0",
        "name": "cyber_3comp",
        "components": {c["name"]: c for c in components},
        "connections": [
            connection("Alim", "Srv", "power"),
            connection("Srv", "Proc", "f_orange"),
            connection("Srv", "Proc", "f_blue"),
            connection("Srv", "Proc", "f_service"),
        ],
    }


def observed_series(model) -> dict:
    session = pyraichu.interactive(model, t_max=30.0)
    values = {}
    for instant in INSTANTS:
        session.advance_to(instant)
        for component, attribute in OBSERVED:
            values[(instant, component, attribute)] = session.attribute(f"{component}.{attribute}")
    return values


def test_the_declaration_route_answers_the_same_values_at_every_instant():
    declared = pyraichu.load_model(json.dumps(declare.build_document(declared_document())))
    library_values = observed_series(library_model())
    assert None not in library_values.values()
    # The values move: the service starts and the functions stop.
    assert library_values[(14.0, "Srv", "f_service_fed_out")] is False
    assert library_values[(16.0, "Srv", "f_service_fed_out")] is True
    assert library_values[(22.0, "Proc", "F1_fed_out")] is True
    assert library_values[(24.0, "Proc", "F1_fed_out")] is False
    assert library_values == observed_series(declared)
