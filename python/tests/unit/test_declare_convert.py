"""The muscadet declaration converter (`pyraichu.muscadet.convert`).

A declaration converts into a library, a system and a study, one type per
knowledge-base class, or is refused by name, per component. The converted
documents are then compared with the declaration route itself: exactly for a
deterministic study, by replaying drawn trajectories in deferred mode for a
stochastic one, where every armed transition must agree on its law after each
firing, so a wrongly converted rate is caught even though forced dates would
fire it anyway.

Oracle-free: both routes run on RAICHU.
"""

from __future__ import annotations

import copy
import json

import pytest
from pyraichu.muscadet.convert import compare_routes, convert
from test_library_cyber_3comp import declared_document

# --- declaration shapes, as the platform writes them ----------------------


def flow_in(name: str, logic="or", default=False) -> dict:
    return {
        "cls": "FlowIn",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_in_default": default,
        "var_available_in_default": True,
        "logic": logic,
    }


def flow_out(name: str, cond=None, produced=False, available=True, negate=False) -> dict:
    spec = {
        "cls": "FlowOut",
        "name": name,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_fed_available_out_init": available,
        "var_fed_available_out_reset": True,
        "var_prod_cond_inner_mode": "and",
        "var_prod_default": produced,
        "negate": negate,
    }
    if cond is not None:
        spec["var_prod_cond"] = cond
    return spec


def component(name: str, flows: list, class_name: str | None = None, source_cls="ObjFlow") -> dict:
    spec = {"name": name, "cls": "ObjFlow", "source_cls": source_cls, "flows": flows}
    if class_name is not None:
        spec["metadata"] = {"class_name": class_name}
    return spec


def exp_mode(
    mode: str,
    targets: list[str],
    effects: dict,
    rate=1e-3,
    repair=0.1,
    cls="ObjFMExp",
    cond=None,
    group=None,
) -> dict:
    """A mode as the platform files it: under `<group>__<mode>`, the group
    being the target's name for a single target."""
    group = group or targets[0]
    spec = {
        "name": f"{group}__{mode}",
        "kind": "two_state_mode",
        "cls": cls,
        "fm_name": mode,
        "targets": targets,
        "target_name": group,
        "behaviour": "external",
        "failure_effects": effects,
        "failure_param_name": ["gamma" if cls == "ObjFMInst" else "lambda"],
        "failure_param": rate if isinstance(rate, list) else [rate],
        "repair_param_name": ["mu"],
        "repair_param": [repair],
    }
    if cond is not None:
        spec["failure_cond"] = cond
    return spec


def event(name: str, cond: list) -> dict:
    return {"name": name, "kind": "two_state_mode", "cls": "ObjEvent", "cond": cond}


def link(source: str, target: str, flow: str) -> dict:
    return {"source": source, "source_box": f"{flow}_out", "target": target, "target_box": f"{flow}_in"}


def document(components: list, connections: list) -> dict:
    return {
        "version": "1.0.0",
        "name": "plant",
        "components": {c["name"]: c for c in components},
        "connections": connections,
    }


def two_pumps(rate_b=2e-3) -> dict:
    """A source feeding two pumps of one class in parallel, each with its own
    failure mode and repair, and an event watching the delivery."""
    pump = lambda name: component(
        name, [flow_in("water"), flow_out("water", [["water"]])], class_name="Pump"
    )
    return document(
        [
            component("tank", [flow_out("water", produced=True)], class_name="Tank"),
            pump("P1"),
            pump("P2"),
            component("sink", [flow_in("water", logic="or")], class_name="Sink"),
            exp_mode("fail", ["P1"], {"water_fed_available_out": False}, rate=1e-3),
            exp_mode("fail", ["P2"], {"water_fed_available_out": False}, rate=rate_b),
            event("dry", [[{"obj": "sink", "attr": "water_fed_in", "ope": "==", "value": False}]]),
        ],
        [link("tank", "P1", "water"), link("tank", "P2", "water"),
         link("P1", "sink", "water"), link("P2", "sink", "water")],
    )


def with_mode(doc: dict, replaced: str, spec: dict) -> dict:
    """`doc` with the mode `replaced` swapped for `spec`."""
    del doc["components"][replaced]
    doc["components"][spec["name"]] = spec
    return doc


def types_of(conversion) -> dict:
    return {t["name"]: t for t in conversion.library["library"]["types"]}


# --- U12: flow components and types ---------------------------------------


def test_instances_of_one_class_share_a_type_and_differing_values_become_parameters():
    doc = document(
        [
            component("S1", [flow_out("power", produced=True, available=True)], class_name="Supply"),
            component("S2", [flow_out("power", produced=True, available=False)], class_name="Supply"),
        ],
        [],
    )
    conversion = convert(doc)
    assert conversion.converted, conversion.refusals
    (supply,) = types_of(conversion).values()
    assert supply["name"] == "Supply"
    assert supply["parameters"] == [{"name": "power_fed_available_init", "kind": "bool"}]
    instances = conversion.system["system"]["instances"]
    assert [i["parameters"] for i in instances] == [
        {"power_fed_available_init": True},
        {"power_fed_available_init": False},
    ]


def test_a_class_whose_members_declare_different_flows_splits_into_variant_types():
    doc = document(
        [
            component("A", [flow_out("power", produced=True)], class_name="Supply"),
            component("B", [flow_out("power", produced=True), flow_out("data", produced=True)],
                      class_name="Supply"),
        ],
        [],
    )
    conversion = convert(doc)
    assert sorted(types_of(conversion)) == ["Supply", "Supply_v2"]
    assert any("`Supply` splits into 2 variant types" in note for note in conversion.notes)


def test_a_python_class_names_the_type_and_a_bare_component_gets_its_own():
    doc = document(
        [
            component("H1", [flow_out("heat", produced=True)], source_cls="Heater"),
            component("H2", [flow_out("heat", produced=True)], source_cls="Heater"),
            component("lone", [flow_in("heat")]),
        ],
        [link("H1", "lone", "heat")],
    )
    conversion = convert(doc)
    assert sorted(types_of(conversion)) == ["Heater", "lone"]
    # A Python class may hold behaviour its declaration does not say.
    assert conversion.unverified == ["H1", "H2"]


def test_a_continuous_flow_is_refused_naming_its_class_and_component():
    continuous = {"cls": "FlowContinuousIn", "name": "gas", "var_in_default": 0.0}
    doc = document([component("tank", [continuous]), component("ok", [flow_in("power")])], [])
    conversion = convert(doc)
    assert not conversion.converted
    assert conversion.library is None
    (refusal,) = conversion.refusals
    assert refusal.component == "tank"
    assert refusal.construct == "flow class `FlowContinuousIn`"


def test_a_negated_output_is_refused_by_name():
    doc = document([component("inv", [flow_in("x"), flow_out("y", [["x"]], negate=True)])], [])
    (refusal,) = convert(doc).refusals
    assert (refusal.component, refusal.construct) == ("inv", "negated output (`negate`)")


def test_an_operand_naming_an_output_reads_that_output():
    doc = document(
        [component("C", [flow_out("a", produced=True), flow_out("b", [["a"]])])], []
    )
    uses = types_of(convert(doc))["C"]["uses"]
    assert uses[1]["with"]["prod_cond"] == [["a_fed_out"]]


# --- U13: modes, controllers and gates -------------------------------------


def test_cyber_3comp_converts_and_matches_the_declaration_route_exactly():
    declaration = declared_document()
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    assert conversion.rename["Srv__mdc_b.fm.failure"] == "Srv.mdc_b.occ"
    comparison = compare_routes(declaration, conversion, t_max=30.0)
    assert comparison.identical, comparison.detail
    assert comparison.level == "exact"


def test_a_common_cause_order_is_refused_naming_the_order():
    doc = with_mode(two_pumps(), "P1__fail", exp_mode(
        "fail", ["P1", "P2"], {"water_fed_available_out": False}, rate=[1e-3, 5e-4], group="pumps"
    ))
    del doc["components"]["P2__fail"]
    (refusal,) = convert(doc).refusals
    assert refusal.construct == "common-cause order 2"


def test_a_multi_target_mode_with_independent_orders_only_converts_per_target():
    doc = with_mode(two_pumps(), "P1__fail", exp_mode(
        "fail", ["P1", "P2"], {"water_fed_available_out": False}, rate=[1e-3, None], group="pumps"
    ))
    del doc["components"]["P2__fail"]
    conversion = convert(doc)
    assert conversion.converted, conversion.refusals
    assert [a["instances"] for a in conversion.study["study"]["modes"]] == [["P1"], ["P2"]]
    assert conversion.rename["pumps__fail.fm__cc_2.occ__cc_2"] == "P2.fail.occ"
    comparison = compare_routes(doc, conversion, t_max=5000.0, seeds=range(4))
    assert comparison.identical, comparison.detail


def standby_plant(prob=0.3) -> dict:
    """A command feeding a backup set that starts only when the command is
    lost: its start fails on demand with probability `prob`, and a failed
    start is repaired."""
    elec = flow_out("elec", [[{"name": "ctrl", "port": "in", "negate": True}]])
    return document(
        [
            component("cmd", [flow_out("ctrl", produced=True)], class_name="Command"),
            component("backup", [flow_in("ctrl"), elec], class_name="Backup"),
            component("load", [flow_in("elec")], class_name="Load"),
            exp_mode("lost", ["cmd"], {"ctrl_fed_available_out": False}, rate=2e-2, repair=0.5),
            exp_mode(
                "refusal", ["backup"], {"elec_fed_available_out": False}, cls="ObjFMInst",
                rate=prob, repair=0.1,
                cond=[[{"attr": "ctrl_fed_in", "ope": "==", "value": False}]],
            ),
            event("blackout", [[{"obj": "load", "attr": "elec_fed_in", "ope": "==", "value": False}]]),
        ],
        [link("cmd", "backup", "ctrl"), link("backup", "load", "elec")],
    )


def test_an_on_demand_mode_converts_and_matches_by_replay():
    declaration = standby_plant()
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    (activation,) = [a for a in conversion.study["study"]["modes"] if a["mode"] == "refusal"]
    assert activation["occurrence"] == {"distrib": "inst", "probs": [0.3]}
    assert conversion.rename["backup__refusal.fm.occ"] == "backup.refusal.occ"
    assert conversion.rename["backup__refusal.fm.not_occ"] == "backup.refusal.parked"
    comparison = compare_routes(declaration, conversion, t_max=2000.0, seeds=range(6))
    assert comparison.identical, comparison.detail
    assert comparison.level == "replay"


def test_a_wrong_on_demand_probability_is_reported_divergent():
    declaration = standby_plant()
    wrong = copy.deepcopy(convert(declaration))
    for activation in wrong.study["study"]["modes"]:
        if activation["mode"] == "refusal":
            activation["occurrence"]["probs"] = [0.31]
    comparison = compare_routes(declaration, wrong, t_max=2000.0, seeds=range(2))
    assert not comparison.identical
    assert "backup.refusal.occ" in comparison.detail


def test_an_on_demand_repair_law_is_refused_by_name():
    declaration = standby_plant()
    spec = declaration["components"]["backup__refusal"]
    spec["cls"] = "ObjMode2S"
    for key in ("failure_effects", "failure_param_name", "failure_param", "repair_param_name",
                "repair_param", "failure_cond", "fm_name"):
        spec.pop(key, None)
    spec.update({
        "mode_name": "refusal",
        "occ_law": {"cls": "exp", "rate": [1e-2]},
        "occ_effects": {"elec_fed_available_out": False},
        "occ_param_name": ["lambda"], "occ_param": [1e-2],
        "not_occ_state": "rep",
        "not_occ_law": {"cls": "inst", "prob": [0.5]},
        "not_occ_param_name": ["not_occ_prob"], "not_occ_param": [0.5],
    })
    refusals = convert(declaration).refusals
    assert ("backup__refusal", "on-demand repair law (`inst`)") in [
        (r.component, r.construct) for r in refusals
    ]


def test_a_controller_is_refused_by_name():
    doc = two_pumps()
    doc["components"]["ctrl"] = {"name": "ctrl", "kind": "controller", "cls": "ObjCtrl"}
    (refusal,) = convert(doc).refusals
    assert (refusal.component, refusal.construct) == ("ctrl", "controller")


def gate(cond: list) -> dict:
    return {
        "name": "vote",
        "kind": "logic_gate",
        "cls": "ObjLogicGate",
        "source_cls": "ObjLogicGate",
        "logic_kind": "k",
        "k": 2,
        "cond": cond,
        "out_elements": ["order"],
        "metadata": {"class_name": "logic_kn"},
    }


def voting_plant(cond: list) -> dict:
    sensor = lambda name: component(name, [flow_out("alarm", produced=True)], class_name="Sensor")
    return document(
        [
            sensor("S1"), sensor("S2"), sensor("S3"),
            gate(cond),
            component("trip", [flow_in("order")], class_name="Trip"),
            exp_mode("lost", ["S1"], {"alarm_fed_available_out": False}, rate=0.01),
            exp_mode("lost", ["S2"], {"alarm_fed_available_out": False}, rate=0.02),
        ],
        [link("vote", "trip", "order")],
    )


def test_a_gate_reading_outputs_converts_to_connections_and_matches():
    cond = [[{"obj": s, "attr": "alarm_fed_out", "value": True}] for s in ("S1", "S2", "S3")]
    declaration = voting_plant(cond)
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    feeds = [c for c in conversion.system["system"]["connections"] if c["to"] == "vote.order_in"]
    assert [c["from"] for c in feeds] == ["S1.alarm_out", "S2.alarm_out", "S3.alarm_out"]
    comparison = compare_routes(declaration, conversion, t_max=300.0, seeds=range(4))
    assert comparison.identical, comparison.detail


def test_a_gate_reading_another_attribute_by_name_is_refused():
    cond = [[{"obj": s, "attr": "alarm_fed_available_out", "value": True}] for s in ("S1", "S2")]
    (refusal,) = convert(voting_plant(cond)).refusals
    assert (refusal.component, refusal.construct) == ("vote", "gate leaf read by name")


# --- U14: comparison levels and completeness --------------------------------


def test_a_stochastic_study_is_identical_by_replay():
    declaration = two_pumps()
    comparison = compare_routes(declaration, convert(declaration, targets=["dry"]),
                                t_max=5000.0, seeds=range(6))
    assert comparison.identical, comparison.detail
    assert comparison.level == "replay"
    assert comparison.trajectories == 6


def test_a_wrongly_converted_rate_is_reported_divergent_naming_the_transition():
    declaration = two_pumps()
    conversion = convert(declaration)
    wrong = copy.deepcopy(conversion)
    for activation in wrong.study["study"]["modes"]:
        if activation["instances"] == ["P2"]:
            activation["occurrence"]["rate"] *= 1.0 + 1e-9
    comparison = compare_routes(declaration, wrong, t_max=5000.0, seeds=range(2))
    assert not comparison.identical
    assert comparison.level == "replay"
    assert "`P2.fail.occ`" in comparison.detail


def test_the_statistical_comparison_is_labelled_statistical():
    declaration = two_pumps()
    comparison = compare_routes(
        declaration, convert(declaration), t_max=2000.0, seeds=range(30),
        method="statistical", instants=[500.0, 2000.0],
    )
    assert comparison.identical, comparison.detail
    assert comparison.level == "statistical"


def test_a_feared_event_marked_as_target_keeps_its_name():
    conversion = convert(two_pumps(), targets=["dry"], schedule=[100.0, 1000.0], nb_runs=50)
    (feared,) = conversion.study["study"]["feared_events"]
    assert (feared["name"], feared["target"]) == ("dry", True)
    assert conversion.rename["dry.ev.occ"] == "dry.ev.occ"
    assert conversion.study["study"]["simulation"]["horizon"] == 1000.0


def test_an_unaudited_class_is_flagged_even_when_both_routes_agree():
    declaration = two_pumps()
    declaration["components"]["P1"]["source_cls"] = "PumpWithPythonBehaviour"
    conversion = convert(declaration)
    assert conversion.unverified == ["P1"]
    assert compare_routes(declaration, conversion, t_max=1000.0, seeds=range(2)).identical


def test_loading_a_refused_conversion_names_the_refusals():
    doc = document([component("inv", [flow_in("x"), flow_out("y", [["x"]], negate=True)])], [])
    with pytest.raises(ValueError, match="negated output"):
        convert(doc).load()


def own_failure_mode(name: str, flow: str, law="exp", failure=0.02, repair=0.5) -> dict:
    """A mode declared in the component's own `failure_modes` section, as
    muscadet's `system_spec` writes it."""
    times = {"exp": ("failure_rate", "repair_rate"), "delay": ("failure_time", "repair_time")}[law]
    return {
        "cls": law,
        "name": name,
        "failure_cond": f"{flow}_fed_out",
        "failure_effects": [[f"{flow}_fed_available_out", False]],
        "failure_param_name": "lambda" if law == "exp" else "ttf",
        times[0]: failure,
        "failure_state": "occ",
        "repair_cond": True,
        "repair_effects": [],
        "repair_param_name": "mu" if law == "exp" else "ttr",
        times[1]: repair,
        "repair_state": "rep",
    }


def test_a_mode_of_the_component_own_section_converts_and_matches():
    source = component("S", [flow_out("power", produced=True)], source_cls="Source")
    source["failure_modes"] = [own_failure_mode("wear", "power")]
    block = component("B", [flow_in("power"), flow_out("power", [["power"]])], source_cls="Block")
    block["failure_modes"] = [own_failure_mode("trip", "power", law="delay", failure=7.0, repair=2.0)]
    declaration = document(
        [source, block, component("T", [flow_in("power")], source_cls="Target")],
        [link("S", "B", "power"), link("B", "T", "power")],
    )
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    assert types_of(conversion)["Block"]["modes"] == [
        {"name": "trip", "nature": "failure",
         "effects": [{"attribute": "power_fed_available_out", "value": False}]}
    ]
    assert conversion.rename["S.wear.failure"] == "S.wear.occ"
    # Audited classes (green on the muscadet bench): nothing left unverified.
    assert conversion.unverified == []
    comparison = compare_routes(declaration, conversion, t_max=200.0, seeds=range(5))
    assert comparison.identical, comparison.detail


def test_the_statistical_comparison_flags_a_rate_fifty_times_too_high():
    declaration = two_pumps(rate_b=2e-3)
    wrong = copy.deepcopy(convert(declaration))
    for activation in wrong.study["study"]["modes"]:
        activation["occurrence"]["rate"] *= 50.0
    comparison = compare_routes(
        declaration, wrong, t_max=2000.0, seeds=range(60),
        method="statistical", instants=[500.0, 2000.0],
    )
    assert not comparison.identical, comparison.detail
    assert comparison.level == "statistical"


def test_a_mode_the_draws_never_fire_is_forced_so_its_effects_are_compared():
    # P2 fails so rarely that no drawn trajectory fires it: its effect is
    # observed only because the comparison forces it once.
    declaration = two_pumps(rate_b=1e-12)
    wrong = copy.deepcopy(convert(declaration))
    for type_doc in wrong.library["library"]["types"]:
        for mode in type_doc.get("modes", []):
            mode.pop("effects", None)
    comparison = compare_routes(declaration, wrong, t_max=300.0, seeds=range(3))
    assert not comparison.identical
    assert comparison.level == "replay"
    assert "forcing" in comparison.detail


def test_a_mode_that_is_never_armed_leaves_the_replay_partial():
    declaration = two_pumps(rate_b=1e-12)
    # Gate P2's failure on a condition that never holds: it cannot be forced.
    declaration["components"]["P2__fail"]["failure_cond"] = [
        [{"attr": "water_fed_in", "ope": "==", "value": False}]
    ]
    comparison = compare_routes(declaration, convert(declaration), t_max=300.0, seeds=range(3))
    assert not comparison.identical
    assert comparison.level == "replay-partial"
    assert comparison.unexercised == ("P2__fail.fm.failure", "P2__fail.fm.repair")


def test_a_statistical_comparison_with_nothing_in_common_is_not_identical_and_needs_runs():
    declaration = two_pumps()
    conversion = convert(declaration)
    with pytest.raises(ValueError, match="at least 30 runs"):
        compare_routes(declaration, conversion, t_max=100.0, seeds=range(10), method="statistical")


def test_an_own_failure_mode_with_no_target_is_refused():
    source = component("S", [flow_out("power", produced=True)], source_cls="Source")
    untargeted = own_failure_mode("wear", "power")
    untargeted["failure_effects"] = []
    source["failure_modes"] = [untargeted]
    (refusal,) = convert(document([source], [])).refusals
    assert (refusal.component, refusal.construct) == ("S", "failure mode gating no output")


def test_a_condition_on_a_repair_state_named_not_occ_keeps_its_meaning():
    declaration = two_pumps(rate_b=5e-3)
    declaration["components"]["P2__fail"]["repair_state"] = "not_occ"
    declaration["components"]["up"] = event(
        "up", [[{"obj": "P2__fail", "attr": "not_occ", "ope": "==", "value": True}]]
    )
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    feared = {e["name"]: e for e in conversion.study["study"]["feared_events"]}
    assert '"state": "rep"' in json.dumps(feared["up"]["condition"])
    comparison = compare_routes(declaration, conversion, t_max=2000.0, seeds=range(4))
    assert comparison.identical, comparison.detail


# --- temporised outputs (FlowOutTempo) -------------------------------------


def tempo_out(name: str, cond=None, enable=None, disable=None, init_enable=False, produced=False) -> dict:
    spec = flow_out(name, cond, produced=produced)
    spec["cls"] = "FlowOutTempo"
    spec["init_enable"] = init_enable
    if enable is not None:
        spec["occ_enable_flow"] = enable
    if disable is not None:
        spec["occ_disable_flow"] = disable
    return spec


def backup_plant(enable, disable, rate=1e-2) -> dict:
    """A source feeding a backup whose output switches on and off with
    delays, the source failing and being repaired."""
    return document(
        [
            component("src", [flow_out("power", produced=True)], class_name="Source"),
            component(
                "backup",
                [flow_in("power"), tempo_out("relief", [["power"]], enable=enable, disable=disable)],
                class_name="Backup",
            ),
            component("load", [flow_in("relief")], class_name="Load"),
            exp_mode("lost", ["src"], {"power_fed_available_out": False}, rate=rate, repair=0.2),
            event("starved", [[{"obj": "load", "attr": "relief_fed_in", "ope": "==", "value": False}]]),
        ],
        [link("src", "backup", "power"), link("backup", "load", "relief")],
    )


def test_a_temporised_output_converts_to_the_tempo_fragment():
    declaration = backup_plant({"cls": "DelayOccDistribution", "time": 3.0}, {"cls": "DelayOccDistribution", "time": 0.5})
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    (use,) = [u for u in types_of(conversion)["Backup"]["uses"] if u["fragment"] == "std.bool_flow_out_tempo"]
    assert use["with"] == {
        "name": "relief",
        "prod_cond": [["power_fed_in"]],
        "fed_available_init": True,
        "init_enable": False,
        "enable_law": "delay",
        "enable_value": 3.0,
        "disable_law": "delay",
        "disable_value": 0.5,
    }
    comparison = compare_routes(declaration, conversion, t_max=2000.0, seeds=range(6))
    assert comparison.identical, comparison.detail
    assert comparison.level == "replay"


def test_a_temporised_output_drawing_its_waits_matches_by_replay():
    declaration = backup_plant({"cls": "ExpOccDistribution", "rate": 0.5}, {"cls": "ExpOccDistribution", "rate": 2.0})
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    comparison = compare_routes(declaration, conversion, t_max=2000.0, seeds=range(6))
    assert comparison.identical, comparison.detail


def test_a_wrong_enable_delay_is_reported_divergent():
    declaration = backup_plant({"cls": "DelayOccDistribution", "time": 3.0}, {"cls": "DelayOccDistribution", "time": 0.5})
    wrong = copy.deepcopy(convert(declaration))
    for use in types_of(wrong)["Backup"]["uses"]:
        if use["fragment"] == "std.bool_flow_out_tempo":
            use["with"]["enable_value"] = 3.5
    assert not compare_routes(declaration, wrong, t_max=2000.0, seeds=range(4)).identical


def test_every_blocking_construct_of_a_component_is_named():
    triggered = {"cls": "FlowOutOnTrigger", "name": "relay"}
    inverter = component("inv", [flow_in("x"), flow_out("y", [["x"]], negate=True), triggered])
    refusals = convert(document([inverter], [])).refusals
    assert [(r.component, r.construct) for r in refusals] == [
        ("inv", "negated output (`negate`)"),
        ("inv", "flow class `FlowOutOnTrigger`"),
    ]


def test_a_negated_operand_reads_a_std_negation_and_matches():
    # A backup produces only while its command does NOT arrive.
    negated = flow_out("elec", [[{"name": "ctrl", "port": "in", "negate": True}]])
    declaration = document(
        [
            component("cmd", [flow_out("ctrl", produced=True)], class_name="Command"),
            component("backup", [flow_in("ctrl"), negated], class_name="Backup"),
            component("load", [flow_in("elec")], class_name="Load"),
            exp_mode("lost", ["cmd"], {"ctrl_fed_available_out": False}, rate=1e-2, repair=0.3),
        ],
        [link("cmd", "backup", "ctrl"), link("backup", "load", "elec")],
    )
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    uses = types_of(conversion)["Backup"]["uses"]
    assert {"fragment": "std.bool_not", "with": {"of": "ctrl_fed_in"}} in uses
    assert uses[1]["with"]["prod_cond"] == [["ctrl_fed_in_not"]]
    comparison = compare_routes(declaration, conversion, t_max=1000.0, seeds=range(6))
    assert comparison.identical, comparison.detail


def test_a_compared_operand_is_still_refused():
    compared = flow_out("elec", [[{"name": "ctrl", "op": ">=", "value": 1.0}]])
    (refusal,) = convert(document([component("b", [flow_in("ctrl"), compared])], [])).refusals
    assert refusal.construct == "structured production operand"


def test_a_mode_on_a_component_with_a_negated_operand_converts():
    negated = flow_out("elec", [[{"name": "ctrl", "port": "in", "negate": True}]])
    declaration = document(
        [
            component("cmd", [flow_out("ctrl", produced=True)], class_name="Command"),
            component("backup", [flow_in("ctrl"), negated], class_name="Backup"),
            exp_mode("trip", ["backup"], {"elec_fed_available_out": False}, rate=1e-2),
        ],
        [link("cmd", "backup", "ctrl")],
    )
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    assert compare_routes(declaration, conversion, t_max=500.0, seeds=range(3)).identical


def test_an_unconditioned_temporised_output_starting_enabled_matches():
    # No production condition: the production keeps its default, and the
    # output starts enabled with a disable wait that never ends.
    plain = tempo_out("relief", produced=True, init_enable=True,
                      disable={"cls": "ExpOccDistribution", "rate": 0.0})
    declaration = document(
        [
            component("backup", [plain], class_name="Backup"),
            component("load", [flow_in("relief")], class_name="Load"),
            exp_mode("trip", ["backup"], {"relief_fed_available_out": False}, rate=1e-2),
        ],
        [link("backup", "load", "relief")],
    )
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    (use,) = types_of(conversion)["Backup"]["uses"]
    assert (use["with"]["conditioned"], use["with"]["init_enable"], use["with"]["disable_law"]) == (
        False,
        True,
        "never",
    )
    assert compare_routes(declaration, conversion, t_max=500.0, seeds=range(4)).identical


def test_a_temporised_output_reading_a_negated_input_matches():
    relief = tempo_out(
        "relief",
        [[{"name": "power", "port": "in", "negate": True}]],
        enable={"cls": "DelayOccDistribution", "time": 2.0},
    )
    declaration = document(
        [
            component("src", [flow_out("power", produced=True)], class_name="Source"),
            component("backup", [flow_in("power"), relief], class_name="Backup"),
            component("load", [flow_in("relief")], class_name="Load"),
            exp_mode("lost", ["src"], {"power_fed_available_out": False}, rate=1e-2, repair=0.2),
        ],
        [link("src", "backup", "power"), link("backup", "load", "relief")],
    )
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    assert compare_routes(declaration, conversion, t_max=2000.0, seeds=range(6)).identical


def test_an_internal_on_demand_mode_converts_and_matches():
    # An ObjFMInst read with its default (internal) behaviour carries its
    # laws as bare numbers and names its repair after the state it enters.
    declaration = standby_plant()
    del declaration["components"]["backup__refusal"]["behaviour"]
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    assert conversion.rename["backup__refusal.fm.rep"] == "backup.refusal.rep"
    comparison = compare_routes(declaration, conversion, t_max=2000.0, seeds=range(6))
    assert comparison.identical, comparison.detail
