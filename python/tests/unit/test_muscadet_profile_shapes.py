"""The time profile shapes beyond the sinusoid: a ramped square wave
(`TrapezoidProfile`), a drawn periodic curve (`PiecewiseLinearProfile`), an
upper clamp on every shape, and a smoothed noise drawn per replica.

The declaration muscadet writes for them is read on both routes: the
authoring layer, which turns each shape into an expression of the clock,
and the converter, which writes `std.time_profile` uses. The reference
declaration under `profile_fixtures/` is muscadet's own, copied unchanged:
it carries one case of each kind (square, drawn, noise, input profile,
per-instance profile), and every case must convert and simulate.

A square wave is drawn as four points per period, `[0, low]`,
`[ramp, high]`, `[ramp + plateau, high]`, `[2 ramp + plateau, low]`, low
after that: the same periodic piecewise-linear signal as a drawn curve, so
both shapes share one engine node.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled
from pyraichu.muscadet.convert import compare_routes, convert, declaration_model

REFERENCE = Path(__file__).resolve().parent / "profile_fixtures" / "profile_declarations_reference.json"

#: Close enough for two evaluations of one closed form written with a
#: different operation order.
CLOSED_FORM_TOL = 1e-9


def reference() -> dict:
    # The declaration writes an unbounded clamp as `Infinity`, which
    # Python's json reads as `math.inf`.
    return json.loads(REFERENCE.read_text())


def square(low, high, period, high_duration, ramp_duration, phase_shift=0.0):
    def at(t: float) -> float:
        u = (t - phase_shift) % period
        if u < ramp_duration:
            return low + (high - low) * u / ramp_duration
        if u < ramp_duration + high_duration:
            return high
        if u < 2 * ramp_duration + high_duration:
            return high + (low - high) * (u - ramp_duration - high_duration) / ramp_duration
        return low

    return at


def drawn(points, period, phase_shift=0.0):
    def at(t: float) -> float:
        u = (t - phase_shift) % period
        closed = [*points, [period, points[0][1]]]
        for (x0, y0), (x1, y1) in zip(closed, closed[1:]):
            if x0 <= u < x1:
                return y0 + (y1 - y0) * (u - x0) / (x1 - x0)
        raise AssertionError(u)

    return at


def sine(amplitude, period, phase_shift, offset):
    return lambda t: amplitude * math.sin(2 * math.pi * (t - phase_shift) / period) + offset


def series(result, indicator: str, instants) -> list[float]:
    return [sampled(result, indicator, t) for t in instants]


INSTANTS = [i * 0.75 for i in range(65)]


# --- the reference declaration --------------------------------------------


def test_every_case_of_the_reference_declaration_converts_and_both_routes_agree():
    declaration = reference()
    conversion = convert(declaration)
    assert conversion.converted, conversion.refusals
    comparison = compare_routes(declaration, conversion, t_max=48.0)
    assert comparison.identical, comparison.detail


def test_every_case_of_the_reference_declaration_simulates_its_closed_form():
    result = pyraichu.simulate(declaration_model(reference()), t_max=48.0, samples=INSTANTS)
    expected = {
        "G1_P_out_profile": sine(0.5, 24.0, 0.0, 1.0),
        "G2_P_out_profile": square(0.2, 1.5, 24.0, 11.0, 0.5, 6.0),
        "D1_P_out_profile": lambda t: min(1.5, drawn([[0.0, 0.5], [8.0, 1.2], [16.0, 0.8]], 24.0)(t)),
        "L1_P_in_profile": lambda t: max(0.0, sine(0.4, 24.0, -3.0, 1.0)(t)),
    }
    for indicator, curve in expected.items():
        for t in INSTANTS:
            assert abs(sampled(result, indicator, t) - curve(t)) < CLOSED_FORM_TOL, (indicator, t)
    # The pure consumer's demand is its declared demand times its profile.
    for t in INSTANTS:
        assert abs(sampled(result, "L1_P_demand_in", t) - 1.5 * expected["L1_P_in_profile"](t)) < CROSSING_TOL


def test_the_reference_noise_key_is_the_flow_key_this_layer_computes():
    """The key muscadet writes is the FNV-1a 64-bit hash of
    `<component>/<flow>`; this layer computes the same one for a noisy
    profile declared without a key."""
    noisy = reference()["components"]["N1"]["flows"][0]["profile"]
    assert mu.flow_noise_key("N1", "P") == noisy["noise_key"]


def test_one_class_carrying_two_shapes_splits_into_two_types():
    """`Gen` is a sine on G1 and, by its per-instance profile, a square on
    G2: the shape is structure, so the class splits."""
    conversion = convert(reference())
    by_name = {i["name"]: i["type"] for i in conversion.system["system"]["instances"]}
    assert by_name["G1"] != by_name["G2"]


# --- square wave -------------------------------------------------------------


SQUARE = {
    "cls": "TrapezoidProfile",
    "low": 2.0,
    "high": 8.0,
    "period": 24.0,
    "high_duration": 8.0,
    "ramp_duration": 5.0 / 60.0,
    "phase_shift": 8.0,
    "value_min": 0.0,
    "value_max": math.inf,
}


def source(profile: dict | None, rate: float = 1.0) -> type:
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="w", var_fed_default=rate, profile=profile)

    return Source


def sink(profile: dict | None = None, demand: float = 100.0) -> type:
    class Sink(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w", var_demand_in_default=demand, profile=profile)

    return Sink


def line(producer: type, consumer: type) -> mu.System:
    system = mu.System("line")
    system.add_component(producer, "S")
    system.add_component(consumer, "T")
    system.connect("S", "w", "T", "w")
    return system


def test_a_square_wave_is_half_way_up_in_the_middle_of_its_ramp():
    """Low 2, high 8, rising at 8 h over a 5-minute ramp: 5 at 8 h 02 min 30 s."""
    instant = 8.0 + 2.5 / 60.0
    result = line(source(SQUARE), sink()).simulate(t_max=24.0, samples=[0.0, 4.0, instant, 12.0, 20.0])
    assert abs(sampled(result, "S_w_out_profile", instant) - 5.0) < CLOSED_FORM_TOL
    assert sampled(result, "S_w_out_profile", 12.0) == pytest.approx(8.0, abs=CLOSED_FORM_TOL)
    assert sampled(result, "S_w_out_profile", 4.0) == pytest.approx(2.0, abs=CLOSED_FORM_TOL)
    assert sampled(result, "S_w_out_profile", 20.0) == pytest.approx(2.0, abs=CLOSED_FORM_TOL)


@pytest.mark.parametrize(
    "shape",
    [
        SQUARE,
        # No plateau: the two middle points would share an instant.
        {**SQUARE, "high_duration": 0.0},
        # The plateau and its ramps fill the period: the fall ends on the
        # next period's first point.
        {**SQUARE, "high_duration": 24.0 - 2 * 5.0 / 60.0},
        {**SQUARE, "value_max": 6.0, "value_min": 3.0},
    ],
)
def test_a_square_wave_follows_its_closed_form(shape):
    result = line(source(shape), sink()).simulate(t_max=48.0, samples=INSTANTS)
    curve = square(*(shape[k] for k in ("low", "high", "period", "high_duration", "ramp_duration", "phase_shift")))
    for t in INSTANTS:
        expected = min(shape["value_max"], max(shape["value_min"], curve(t)))
        assert abs(sampled(result, "S_w_out_profile", t) - expected) < CLOSED_FORM_TOL, t


@pytest.mark.parametrize(
    "edit,message",
    [
        ({"ramp_duration": 0.0}, "ramp_duration"),
        ({"high_duration": 23.9}, "period"),
        ({"low": -1.0}, "low"),
        ({"name": "day"}, "unknown keys"),
    ],
)
def test_a_square_wave_that_is_not_one_is_refused_where_it_is_declared(edit, message):
    with pytest.raises(ValueError, match=message):
        source({**SQUARE, **edit})("S")


# --- drawn curve -------------------------------------------------------------


DRAWN = {
    "cls": "PiecewiseLinearProfile",
    "points": [[0.0, 0.5], [8.0, 1.2], [16.0, 0.8]],
    "period": 24.0,
    "phase_shift": 3.0,
    "value_min": 0.0,
    "value_max": 1.0,
}


def test_a_drawn_curve_follows_its_points_and_its_clamp():
    result = line(source(DRAWN), sink()).simulate(t_max=48.0, samples=INSTANTS)
    curve = drawn(DRAWN["points"], 24.0, 3.0)
    for t in INSTANTS:
        assert abs(sampled(result, "S_w_out_profile", t) - min(1.0, curve(t))) < CLOSED_FORM_TOL, t


@pytest.mark.parametrize(
    "points,message",
    [
        ([[1.0, 0.5], [8.0, 1.2]], "instant 0"),
        ([[0.0, 0.5], [8.0, 1.2], [8.0, 0.8]], "increasing"),
        ([[0.0, 0.5], [24.0, 1.2]], "period"),
        ([[0.0, 0.5], [8.0, -1.0]], "negative"),
        ([[0.0, 0.5]], "two points"),
    ],
)
def test_a_drawn_curve_that_is_not_one_is_refused_where_it_is_declared(points, message):
    with pytest.raises(ValueError, match=message):
        source({**DRAWN, "points": points})("S")


# --- noise ---------------------------------------------------------------------


NOISY = {**DRAWN, "value_max": math.inf, "noise_amplitude": 0.2, "noise_time_scale": 2.0}


def noisy_pair() -> mu.System:
    """Two instances of one class carrying the same noisy profile, declared
    without a key: each gets the key of its own flow."""
    system = mu.System("noisy")
    system.add_component(source(NOISY), "A")
    system.add_component(source(NOISY), "B")
    system.add_component(sink(), "T")
    system.add_component(sink(), "U")
    system.connect("A", "w", "T", "w")
    system.connect("B", "w", "U", "w")
    return system


def test_a_noise_differs_between_replicas_and_is_reproducible_by_seed():
    system = noisy_pair()
    first = series(system.simulate(t_max=48.0, samples=INSTANTS, seed=1), "A_w_out_profile", INSTANTS)
    again = series(system.simulate(t_max=48.0, samples=INSTANTS, seed=1), "A_w_out_profile", INSTANTS)
    other = series(system.simulate(t_max=48.0, samples=INSTANTS, seed=2), "A_w_out_profile", INSTANTS)
    assert first == again
    assert first != other
    # Null at its first node: every replica starts on the curve.
    assert first[0] == other[0] == pytest.approx(drawn(DRAWN["points"], 24.0, 3.0)(0.0), abs=CLOSED_FORM_TOL)


def test_two_instances_of_one_class_carry_independent_noises():
    result = noisy_pair().simulate(t_max=48.0, samples=INSTANTS, seed=1)
    assert series(result, "A_w_out_profile", INSTANTS) != series(result, "B_w_out_profile", INSTANTS)


def test_a_noise_keeps_its_declared_key():
    system = mu.System("keyed")
    system.add_component(source({**NOISY, "noise_key": 7}), "A")
    system.add_component(source({**NOISY, "noise_key": 7}), "B")
    system.add_component(sink(), "T")
    system.add_component(sink(), "U")
    system.connect("A", "w", "T", "w")
    system.connect("B", "w", "U", "w")
    result = system.simulate(t_max=48.0, samples=INSTANTS, seed=1)
    assert series(result, "A_w_out_profile", INSTANTS) == series(result, "B_w_out_profile", INSTANTS)


@pytest.mark.parametrize(
    "edit,message",
    [
        ({"noise_time_scale": None}, "noise_time_scale"),
        ({"noise_time_scale": 0.0}, "noise_time_scale"),
        ({"noise_amplitude": -0.1}, "noise_amplitude"),
        ({"noise_amplitude": 0.0}, "without noise"),
        ({"noise_key": -1}, "noise_key"),
        ({"noise_key": True}, "noise_key"),
    ],
)
def test_a_noise_that_is_not_one_is_refused_where_it_is_declared(edit, message):
    profile = {k: v for k, v in {**NOISY, **edit}.items() if v is not None}
    with pytest.raises(ValueError, match=message):
        source(profile)("S")
