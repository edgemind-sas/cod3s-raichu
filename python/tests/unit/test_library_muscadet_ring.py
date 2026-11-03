"""Recirculation rings and transmitting volumes, written with `muscadet`
fragments.

A fan drawing air out of a room and blowing it back closes a ring of
instantaneous values through the room: the fan's rule is bounded by the
demand on its return output, which carries the fan's own demand for room
air through the room's transit. A library tears it by hand, listing the
return output in the `torn` parameter of the rule set making it; the
plugin route detects the same tear on its own. Left untorn, the ring is
refused, and the expander names the instances and rule sets it runs
through and the parameter that tears it.

The agreement with the plugin route is
`test_the_fragment_agrees_with_its_plugin_counterpart` on the
`ventilation_ring` and `transmitting_buffer` benches of
`test_library_std_continuous.py`; this file checks the behaviour those
benches are named after and the refusals around them.
"""

from __future__ import annotations

import copy

import pytest

import pyraichu
import test_library_std_continuous as benches
from test_library_std_continuous import BENCHES, TYPES, library_model, plugin_model, run, ventilation


def models(bench, types=None, instances=None):
    objects, bench_instances, connections, modes, t_max = BENCHES[bench]
    plugin = plugin_model(bench, objects, connections)
    if types is not None:
        saved = list(benches.TYPES)
        benches.TYPES[:] = types
    try:
        library = library_model(bench, instances or bench_instances, connections, modes, plugin["indicators"])
    finally:
        if types is not None:
            benches.TYPES[:] = saved
    return plugin, library, t_max


def with_ventilation(**options):
    types = [t for t in copy.deepcopy(TYPES) if t["name"] != "Ventilation"]
    return types + [ventilation("Ventilation", **options)]


def test_the_torn_ring_holds_the_room_and_exhausts_the_leak():
    """The fan exhausts the whole leak and tops the exhaust up with air,
    blows the air it drew back, and the room's air neither fills nor
    drains."""
    plugin, library, t_max = models("ventilation_ring")
    _, s = run(library, t_max)
    assert all(v == pytest.approx(90.0, rel=1e-12) for v in s["LOCAL_room_content_AIR"])
    assert s["ATMOSPHERE_exhaust_fed_in"][-1] == pytest.approx(50.0, rel=1e-12)
    assert s["VENTILATION_H2_leak_fed_in"][-1] == pytest.approx(0.05, rel=1e-12)
    assert s["VENTILATION_AIR_fed_in"][-1] == pytest.approx(49.95, rel=1e-12)


def test_the_order_does_not_depend_on_the_instance_declaration_order():
    plugin, library, t_max = models("ventilation_ring")
    _, other, _ = models("ventilation_ring", instances=list(reversed(BENCHES["ventilation_ring"][1])))
    assert run(other, t_max) == run(library, t_max)


def test_an_untorn_ring_is_refused_naming_its_instances_and_the_parameter():
    with pytest.raises(pyraichu.ModelError) as refused:
        models("ventilation_ring", types=with_ventilation(torn=[]))
    message = str(refused.value)
    for name in ("`LOCAL`", "`VENTILATION`", "`VENTILATION.renewal`", "`torn`", "muscadet.rule_set"):
        assert name in message, name
    # Off the ring: the leak only feeds it, the extraction only draws it.
    for name in ("`LEAK`", "`VENTILATION.extraction`"):
        assert name not in message, name


def test_tearing_an_output_off_the_ring_keeps_the_refusal():
    """Negative control: the extraction's exhaust is on no ring, so tearing
    it leaves the ring through the renewal standing."""
    with pytest.raises(pyraichu.ModelError, match="torn"):
        models("ventilation_ring", types=with_ventilation(torn=[], extraction_torn=["exhaust"]))


def test_a_tear_beside_the_ring_is_accepted():
    """Torn as well, the extraction's exhaust no longer bounds the
    extraction: the author's statement, accepted without a search for
    rings. The leak still bounds the extraction here, so nothing moves."""
    plugin, library, t_max = models("ventilation_ring", types=with_ventilation(torn=["AIR"], extraction_torn=["exhaust"]))
    scale = next(
        e for c in pyraichu.model_body(library)["components"] if c["name"] == "VENTILATION"
        for e in c["equations"] if e["target"] == "extraction_scale"
    )
    assert "exhaust_demand_out" not in str(scale["expr"])
    assert run(library, t_max) == run(plugin, t_max)


def test_a_torn_output_the_rule_set_does_not_make_is_refused():
    with pytest.raises(pyraichu.ModelError, match="AIRR"):
        models("ventilation_ring", types=with_ventilation(torn=["AIRR"]))


def test_a_transmitting_buffer_a_millionth_off_is_divergent():
    """Negative control of the integrator-level comparison: the spread it
    admits is far below what a capacity one millionth off moves."""
    objects, instances, connections, modes, t_max = BENCHES["transmitting_buffer"]
    altered = copy.deepcopy(instances)
    altered[1]["parameters"]["capacity"] *= 1.0 + 1e-6
    BENCHES["transmitting_buffer_off"] = (objects, altered, connections, modes, t_max)
    benches.INTEGRATOR_BENCHES.add("transmitting_buffer_off")
    try:
        with pytest.raises(AssertionError):
            benches.test_the_fragment_agrees_with_its_plugin_counterpart("transmitting_buffer_off")
    finally:
        del BENCHES["transmitting_buffer_off"]
        benches.INTEGRATOR_BENCHES.discard("transmitting_buffer_off")


def test_a_transmitting_buffer_passes_on_what_crosses_it_once_empty():
    plugin, library, t_max = models("transmitting_buffer")
    events, s = run(library, t_max)
    assert any(name.endswith("reach_empty") for _, name in events)
    assert any(name.endswith("leave_empty") for _, name in events)


def test_a_cascade_after_a_pooled_extraction_asks_the_renewal_the_rest():
    """The extraction asks the whole exhaust demand, 50, but makes only
    what the pooled room hands it; the renewal is asked what is left, so
    the exhaust is served whole."""
    plugin, library, t_max = models("ventilation_ring_cascade")
    _, s = run(library, t_max)
    assert s["ATMOSPHERE_exhaust_fed_in"][-1] == pytest.approx(50.0, rel=1e-9)
    assert s["VENTILATION_AIR_fed_in"][-1] > 40.0


def test_the_comparison_sees_the_expected_scale_missing():
    types = copy.deepcopy(TYPES)
    ty = next(t for t in types if t["name"] == "CascadedVentilation")
    ty["uses"] = [u for u in ty["uses"] if u["fragment"] != "muscadet.rule_expected"]
    for u in ty["uses"]:
        u["with"].pop("expected", None)
    plugin, library, t_max = models("ventilation_ring_cascade", types=types)
    assert run(library, t_max) != run(plugin, t_max)
