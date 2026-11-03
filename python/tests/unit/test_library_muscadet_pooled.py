"""Pooled serving out of a mixed volume, written with `muscadet` fragments.

A volume holding several flows downstream of the transfer does not serve
each output from its own stock. It pools what its consumers ask beyond what
transits it and hands each held flow a share of that excess in proportion
to its raw content (`volume_pooled`, `volume_pooled_flow`, and
`continuous_flow_out` with `available: pooled`). A rule fed by such a volume
runs at the draw its scarcest arrival allows and hands the rest back
(`rule_consumes` writing `consumed_in`, `surplus_release` reading the
`pooled` channel).

The agreement with the plugin route, event for event and sample for
sample, is `test_the_fragment_agrees_with_its_plugin_counterpart` on the
`pooled_*` benches of `test_library_std_continuous.py`. This file checks
that those benches show the behaviour they are named after, and that the
comparison would see the fragments missing.
"""

from __future__ import annotations

import copy

import pytest

import pyraichu
from test_library_std_continuous import BENCHES, library_model, plugin_model, run, TYPES


def samples(bench, instants):
    objects, instances, connections, modes, _ = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    library = library_model(bench, instances, connections, modes, plugin["indicators"])
    result = pyraichu.simulate(pyraichu.load_model(library), t_max=max(instants), samples=instants)
    return {name: dict(series) for name, series in result.samples.items()}


INSTANTS = [1.0, 5.0, 10.0, 20.0]


def test_two_consumers_share_the_excess_in_proportion_to_the_contents():
    """Nothing transits the room, so the excess is the whole request, 5,
    and each constituent leaves at 5 times its share of the content."""
    s = samples("pooled_draw", INSTANTS)
    for t in INSTANTS:
        h, a = s["ROOM_room_content_H2"][t], s["ROOM_room_content_AIR"][t]
        assert s["CH2_H2_fed_in"][t] == pytest.approx(5.0 * h / (h + a), rel=1e-9), t
        assert s["CAIR_AIR_fed_in"][t] == pytest.approx(min(3.0, 5.0 * a / (h + a)), rel=1e-9), t


def test_a_constituent_that_has_run_out_serves_only_what_arrives():
    """Until the air runs out too, at t = 50/3."""
    instants = [1.0, 5.0, 10.0, 15.0]
    s = samples("pooled_draw_one_constituent_empty", instants)
    for t in instants:
        assert s["CH2_H2_fed_in"][t] == 0.0, t
        assert s["CAIR_AIR_fed_in"][t] == pytest.approx(3.0), t


def test_a_rule_fed_by_the_pooled_draw_hands_back_what_it_does_not_consume():
    """The rule draws what the hydrogen allows, 20 h/(h+a) of each input,
    and the room loses exactly that of both constituents."""
    s = samples("pooled_draw_feeds_a_rule", [1.0, 5.0])
    for t in (1.0, 5.0):
        h, a = s["ROOM_room_content_H2"][t], s["ROOM_room_content_AIR"][t]
        drawn = 20.0 * h / (h + a)
        for flow in ("H2", "AIR"):
            assert s[f"MIX_{flow}_fed_in"][t] == pytest.approx(drawn, rel=1e-9), (t, flow)
            assert s[f"ROOM_{flow}_fed_out"][t] == pytest.approx(drawn, rel=1e-9), (t, flow)
        assert s["SINK_mix_fed_in"][t] == pytest.approx(2.0 * drawn, rel=1e-9), t
        assert (1.0 - h) == pytest.approx(99.0 - a, abs=1e-9), t


def test_a_pooled_and_a_plain_supplier_hand_the_rule_no_more_than_it_consumes():
    s = samples("pooled_and_plain_suppliers", [1.0, 5.0])
    for t in (1.0, 5.0):
        received = s["MIX_H2_fed_in"][t]
        assert received == pytest.approx(s["ROOM_H2_fed_out"][t] + s["S_H2_fed_out"][t], rel=1e-9), t
        assert received == pytest.approx(s["MIX_AIR_fed_in"][t], rel=1e-9), t


@pytest.mark.parametrize(
    "bench, type_name, change",
    [
        # Each output served from its own stock instead of the pooled draw.
        ("pooled_draw", "MixedRoom", "unpooled"),
        # The rule keeps the air it does not consume.
        ("pooled_draw_feeds_a_rule", "Mixer", "no_release"),
    ],
)
def test_the_comparison_sees_the_fragments_missing(monkeypatch, bench, type_name, change):
    import test_library_std_continuous as benches

    types = copy.deepcopy(TYPES)
    ty = next(t for t in types if t["name"] == type_name)
    if change == "unpooled":
        ty["uses"] = [
            {**u, "with": {k: v for k, v in u["with"].items() if k != "available"}}
            if u["fragment"] == "muscadet.continuous_flow_out"
            else u
            for u in ty["uses"]
            if u["fragment"] not in ("muscadet.volume_pooled", "muscadet.volume_pooled_flow")
        ]
    else:
        ty["uses"] = [u for u in ty["uses"] if u["fragment"] != "muscadet.surplus_release"]
    monkeypatch.setattr(benches, "TYPES", types)
    objects, instances, connections, modes, t_max = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    # The attributes the dropped fragments wrote are no longer there to read.
    kept = [i for i in plugin["indicators"] if not i["name"].endswith(("_beyond", "_pooled_out"))]
    library = library_model(bench, instances, connections, modes, kept)
    _, plugin_samples = run(plugin, t_max)
    _, library_samples = run(library, t_max)
    assert any(library_samples[name] != plugin_samples[name] for name in library_samples)


def test_a_release_on_an_input_no_rule_consumes_still_expands():
    """`surplus_release` reads what a rule consumes, which the input
    declares at 0 when no rule writes it: an input released between two
    suppliers with no rule behind it expands as it did before."""
    library = {"raichu_library": {"format": 1}, "library": {
        "name": "b", "version": "1.0.0", "imports": ["muscadet"], "types": [
            {"name": "Src", "uses": [
                {"fragment": "muscadet.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "muscadet.continuous_source", "with": {"name": "Q", "rate": 2.0}}]},
            {"name": "Sink", "uses": [
                {"fragment": "muscadet.continuous_flow_in", "with": {"name": "Q", "demand": 1.0}},
                {"fragment": "muscadet.surplus_release", "with": {"name": "Q"}},
                {"fragment": "muscadet.continuous_sink", "with": {"name": "Q", "demand": 1.0}}]}]}}
    system = {"raichu_system": {"format": 1}, "system": {
        "name": "s", "library": {"name": "b", "version": "1.0.0"},
        "instances": [{"name": "A", "type": "Src"}, {"name": "B", "type": "Src"},
                      {"name": "C", "type": "Sink"}],
        "connections": [{"from": "A.Q_out", "to": "C.Q_in"}, {"from": "B.Q_out", "to": "C.Q_in"}],
        "indicators": [{"name": "C_Q_fed_in", "target": "attribute",
                        "attr": {"component": "C", "attribute": "Q_fed_in"}}]}}
    model = pyraichu.expand_documents(system, [library])["model"]
    result = pyraichu.simulate(pyraichu.load_model(model), t_max=1.0, samples=[1.0])
    assert dict(result.samples["C_Q_fed_in"])[1.0] == pytest.approx(1.0)
