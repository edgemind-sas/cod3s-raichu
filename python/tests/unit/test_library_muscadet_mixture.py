"""Mixture groups written with `muscadet` fragments.

A machine drawing a mixture moves one volume per unit of time of whatever
the volume it draws holds: each input of the group asks the rate times the
composition its supplier publishes (`mixture_in`, `continuous_flow_out` with
`available: mixture`, `volume_pooled_flow` with `drawn`). Several groups on
one volume are served pro rata of their composed requests.

The agreement with the plugin route, event for event and sample for sample,
is `test_the_fragment_agrees_with_its_plugin_counterpart` on the mixture
benches of `test_library_std_continuous.py`. This file checks that those
benches show the behaviour they are named after, that the comparison would
see a fragment missing, and that the plugin route reads mixture groups.
"""

from __future__ import annotations

import copy

import pytest

import pyraichu
from test_library_std_continuous import BENCHES, TYPES, library_model, plugin_model, run


def samples(bench, instants):
    objects, instances, connections, modes, _ = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    library = library_model(bench, instances, connections, modes, plugin["indicators"])
    result = pyraichu.simulate(pyraichu.load_model(library), t_max=max(instants), samples=instants)
    return {name: dict(series) for name, series in result.samples.items()}


def test_one_group_draws_its_rate_at_the_composition():
    """Nothing feeds the room: the shares stay at 1 in 100 and the room
    loses exactly the rate."""
    instants = [1.0, 5.0, 10.0]
    s = samples("mixture_group", instants)
    for t in instants:
        h, a = s["ROOM_room_content_H2"][t], s["ROOM_room_content_AIR"][t]
        assert h + a == pytest.approx(100.0 - 5.0 * t, rel=1e-12), t
        assert s["FAN_H2_fed_in"][t] == pytest.approx(0.05, rel=1e-12), t
        assert s["FAN_AIR_fed_in"][t] == pytest.approx(4.95, rel=1e-12), t


def test_what_arrives_enters_the_composition():
    """Hydrogen arrives at 0.5 and leaves at the room's share times 5, not
    at the rate it arrives."""
    instants = [1.0, 5.0, 10.0]
    s = samples("mixture_group_fed", instants)
    for t in instants:
        h, a = s["ROOM_room_content_H2"][t], s["ROOM_room_content_AIR"][t]
        assert h + a == pytest.approx(50.0 - 0.5 * t, rel=1e-9), t
        assert s["FAN_H2_fed_in"][t] == pytest.approx(5.0 * h / (h + a), rel=1e-9), t
        assert s["FAN_H2_fed_in"][t] != pytest.approx(0.5), t


def test_two_groups_add_their_draws():
    instants = [1.0, 5.0, 10.0]
    s = samples("two_mixture_groups", instants)
    for t in instants:
        h, a = s["ROOM_room_content_H2"][t], s["ROOM_room_content_AIR"][t]
        assert h + a == pytest.approx(100.0 - 4.0 * t, rel=1e-12), t
        assert s["FAN_AIR_fed_in"][t] == pytest.approx(2.97, rel=1e-12), t
        assert s["DRAFT_AIR_fed_in"][t] == pytest.approx(0.99, rel=1e-12), t


def test_groups_behind_a_ceiling_keep_their_pro_rata():
    """Requests of 100 and 1 behind an air outlet of 2: the air goes 100 to
    1, the claims scaled to what the outlet delivers rather than truncated
    at it."""
    s = samples("two_mixture_groups_under_a_ceiling", [1.0])
    fan, draft = s["FAN_AIR_fed_in"][1.0], s["DRAFT_AIR_fed_in"][1.0]
    assert fan + draft == pytest.approx(2.0, rel=1e-12)
    assert fan / draft == pytest.approx(100.0, rel=1e-9)


def test_a_commanded_group_draws_nothing_until_it_runs():
    s = samples("commanded_mixture_group", [1.0, 19.0])
    assert s["FAN_H2_fed_in"][1.0] == 0.0
    assert s["FAN_H2_fed_in"][19.0] > 0.0


@pytest.mark.parametrize(
    "bench, type_name, change",
    [
        # The outputs hand out the pooled draw instead of what the group asks.
        ("mixture_group", "DrawnRoom", "undrawn"),
        # The group always runs.
        ("commanded_mixture_group", "CommandedExtractor", "uncommanded"),
    ],
)
def test_the_comparison_sees_a_fragment_option_missing(monkeypatch, bench, type_name, change):
    import test_library_std_continuous as benches

    types = copy.deepcopy(TYPES)
    ty = next(t for t in types if t["name"] == type_name)
    for u in ty["uses"]:
        if change == "undrawn":
            if u["fragment"] == "muscadet.volume_pooled_flow":
                u["with"].pop("drawn")
            if u["fragment"] == "muscadet.continuous_flow_out":
                u["with"]["available"] = "pooled"
        elif u["fragment"] == "muscadet.mixture_in":
            u["with"].pop("run")
    monkeypatch.setattr(benches, "TYPES", types)
    objects, instances, connections, modes, t_max = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    library = library_model(bench, instances, connections, modes, plugin["indicators"])
    _, plugin_samples = run(plugin, t_max)
    _, library_samples = run(library, t_max)
    assert any(library_samples[name] != plugin_samples[name] for name in library_samples)


def test_the_plugin_route_draws_a_mixture_group():
    """The plugin reads `mixtures` as the declaration route does, and a
    volume it draws is generated once: its pooled draw is not carried a
    second time as if another object had grafted it."""
    objects, _, connections, _, t_max = BENCHES["mixture_group"]
    plugin = plugin_model("mixture_group", objects, connections)
    room = next(c for c in plugin["components"] if c["name"] == "ROOM")
    targets = [e["target"] for e in room["equations"]]
    assert targets.count("H2_pooled_out") == 1
    _, s = run(plugin, t_max)
    assert s["FAN_H2_fed_in"][-1] == pytest.approx(0.05, rel=1e-12)


def test_a_rule_consuming_a_flow_its_group_draws_is_refused():
    """The plugin refuses it because an input has one demand; the expander
    refuses it as two fragments writing that demand, naming both."""
    def mu(fragment, **with_):
        return {"fragment": f"muscadet.{fragment}", "with": with_}

    library = {"raichu_library": {"format": 1}, "library": {
        "name": "b", "version": "1.0.0", "imports": ["muscadet"], "types": [
            {"name": "Machine", "uses": [
                mu("continuous_flow_in", name="H2"), mu("continuous_flow_in", name="AIR"),
                mu("mixture_in", name="extraction", flows=["H2", "AIR"], flow_rate=5.0),
                mu("rule_set", name="r", active="run", idle="none", consumes=["H2"],
                   produces=["heat"], always_on=True),
                mu("rule_consumes", rule_set="r", active="run", flow="H2", rate=1.0, always_on=True),
                mu("rule_produces", rule_set="r", active="run", flow="heat", rate=1.0, always_on=True),
                mu("continuous_flow_out", name="heat", available="produced")]}]}}
    system = {"raichu_system": {"format": 1}, "system": {
        "name": "s", "library": {"name": "b", "version": "1.0.0"},
        "instances": [{"name": "M", "type": "Machine"}]}}
    with pytest.raises(pyraichu.ModelError, match="H2_demand_in"):
        pyraichu.expand_documents(system, [library])


def test_an_empty_constituent_is_drawn_at_its_share():
    s = samples("mixture_group_one_constituent_empty", [1.0, 5.0])
    for t in (1.0, 5.0):
        assert s["FAN_H2_fed_in"][t] == 0.0, t
        assert s["FAN_AIR_fed_in"][t] == pytest.approx(5.0, rel=1e-12), t


def test_the_plugin_route_observes_what_a_rule_consumes():
    """The plugin route builds each component's first pass before the
    network is generated, so the inputs it publishes are those the network
    settled: a rule fed by a pooled volume publishes what it consumes, as
    the class-based route does."""
    objects, _, connections, _, _ = BENCHES["pooled_draw_feeds_a_rule"]
    names = {i["name"] for i in plugin_model("pooled_draw_feeds_a_rule", objects, connections)["indicators"]}
    assert {"MIX_H2_consumed_in", "MIX_AIR_consumed_in"} <= names
