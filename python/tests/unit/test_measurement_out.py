"""An instrument that republishes what it reads (muscadet's
``add_measurement_out``, R37).

A capacity publishes its own level. Redundancy is not several observations
of one tank, which are identical and reject nothing: it is several
INSTRUMENTS, each able to fail on its own, standing between the tank and
whoever votes on them. An instrument is a component, so a component has to
be able to publish a reading, under the very aliases a capacity uses, so
an observer cannot tell the two apart.

What an instrument publishes is its source's reading times
``{name}_level_gain``, the endpoint a failure mode clamps to make it lie:
0 is a dead instrument, 5 a wild one. The source is a capacity or a
measurement channel of the same component; without one, the published
level is a plain variable nothing refreshes.
"""

from __future__ import annotations

import math

import pytest

import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled

HOURS = [1.0, 2.5, 4.0]


class Rain(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="w", var_fed_default=1.0)


class Cistern(mu.ObjFlow):
    """A volume filling at one a unit of time from empty: its level IS the
    date, and its fill is the level over the capacity."""

    def add_flows(self):
        self.add_flow_continuous_in(name="w")
        self.add_capacity(name="vol", flow="w", capacity=100.0, fill_rate=1.0)


def instrument(source="vol", reads="vol", flows=None, gain=1.0, **extra):
    class Instrument(mu.ObjFlow):
        def add_flows(self):
            if reads is not None:
                self.add_measurement_in(name=reads, flows=flows)
            self.add_measurement_out(
                name="reading",
                source=source,
                flows=flows,
                gain_default=gain,
                **extra,
            )

    return Instrument


def observer(flows=None):
    class Observer(mu.ObjFlow):
        def add_flows(self):
            self.add_measurement_in(name="reading", flows=flows)

    return Observer


def a_chain(instrument_cls, flows=None) -> mu.System:
    system = mu.System("republished")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.add_component(instrument_cls, "I")
    system.add_component(observer(flows), "O")
    system.connect("P", "w", "T", "w")
    system.connect_measurement("T", "vol", "I")
    system.connect_measurement("I", "reading", "O")
    return system


def test_an_observer_reads_the_tank_through_the_instrument():
    result = a_chain(instrument()).simulate(t_max=5.0, samples=HOURS)
    for hour in HOURS:
        assert abs(sampled(result, "O_reading_level", hour) - hour) < CROSSING_TOL
        assert (
            abs(sampled(result, "O_reading_fill", hour) - hour / 100.0) < CROSSING_TOL
        )


def test_the_observer_reads_this_evaluation_s_publication():
    """Swept in the order tank, instrument's reading, its publication,
    observer: the observer's value equals the tank's to the last bit, which
    a sweep one evaluation late would not give on a moving level."""
    result = a_chain(instrument()).simulate(t_max=5.0, samples=HOURS)
    for hour in HOURS:
        assert sampled(result, "O_reading_level", hour) == sampled(
            result, "T_vol_content", hour
        )


def test_the_gain_scales_every_reading():
    result = a_chain(instrument(gain=0.5, flows=["w"]), flows=["w"]).simulate(
        t_max=5.0, samples=HOURS
    )
    for hour in HOURS:
        assert abs(sampled(result, "O_reading_level", hour) - 0.5 * hour) < CROSSING_TOL
        assert (
            abs(sampled(result, "O_reading_level_w", hour) - 0.5 * hour) < CROSSING_TOL
        )


def test_a_dead_instrument_reads_zero_while_the_tank_fills():
    result = a_chain(instrument(gain=0.0)).simulate(t_max=5.0, samples=HOURS)
    assert [sampled(result, "O_reading_level", hour) for hour in HOURS] == [0.0] * 3
    assert sampled(result, "T_vol_content", HOURS[-1]) > 3.0


def test_a_constituent_is_republished_with_its_share():
    result = a_chain(instrument(flows=["w"]), flows=["w"]).simulate(
        t_max=5.0, samples=HOURS
    )
    for hour in HOURS:
        assert abs(sampled(result, "O_reading_level_w", hour) - hour) < CROSSING_TOL
        assert (
            abs(sampled(result, "O_reading_fill_w", hour) - hour / 100.0) < CROSSING_TOL
        )


def test_an_instrument_holding_the_capacity_publishes_it_directly():
    class Tank(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_capacity(name="vol", flow="w", capacity=100.0, fill_rate=1.0)
            self.add_measurement_out(name="reading", source="vol")

    system = mu.System("self_published")
    system.add_component(Rain, "P")
    system.add_component(Tank, "T")
    system.add_component(observer(), "O")
    system.connect("P", "w", "T", "w")
    system.connect_measurement("T", "reading", "O")
    result = system.simulate(t_max=5.0, samples=HOURS)
    for hour in HOURS:
        assert abs(sampled(result, "O_reading_level", hour) - hour) < CROSSING_TOL


def test_two_instruments_in_a_row_are_swept_in_a_row():
    class Relay(mu.ObjFlow):
        def add_flows(self):
            self.add_measurement_in(name="reading")
            self.add_measurement_out(name="relayed", source="reading")

    class Last(mu.ObjFlow):
        def add_flows(self):
            self.add_measurement_in(name="relayed")

    system = mu.System("relayed")
    system.add_component(Rain, "P")
    system.add_component(Cistern, "T")
    system.add_component(instrument(), "I")
    system.add_component(Relay, "R")
    system.add_component(Last, "O")
    system.connect("P", "w", "T", "w")
    system.connect_measurement("T", "vol", "I")
    system.connect_measurement("I", "reading", "R")
    system.connect_measurement("R", "relayed", "O")
    result = system.simulate(t_max=5.0, samples=HOURS)
    for hour in HOURS:
        assert sampled(result, "O_relayed_level", hour) == sampled(
            result, "T_vol_content", hour
        )


def test_an_instrument_with_no_source_publishes_its_declared_level():
    system = mu.System("unsourced")
    system.add_component(instrument(source=None, reads=None, level_default=3.0), "I")
    system.add_component(observer(), "O")
    system.connect_measurement("I", "reading", "O")
    result = system.simulate(t_max=5.0, samples=HOURS)
    assert [sampled(result, "O_reading_level", hour) for hour in HOURS] == [3.0] * 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"source": "nothing"}, "names no capacity"),
        ({"flows": ["w", "w"]}, "more than once"),
    ],
)
def test_a_malformed_publication_is_refused_where_it_is_written(kwargs, message):
    with pytest.raises(ValueError, match=message):
        a_chain(instrument(**kwargs)).build_dict()


def test_a_continuous_output_as_source_is_refused_naming_the_rate_channel():
    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="q", var_fed_default=1.0)
            self.add_measurement_out(name="reading", source="q")

    system = mu.System("rate_source")
    system.add_component(Pump, "S")
    with pytest.raises(ValueError, match="publish_rate"):
        system.build_dict()


def test_a_name_a_capacity_already_publishes_is_refused():
    class Tank(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="w")
            self.add_capacity(name="vol", flow="w", capacity=100.0, fill_rate=1.0)
            self.add_measurement_out(name="vol", source="vol")

    system = mu.System("clash")
    with pytest.raises(ValueError, match="capacity"):
        system.add_component(Tank, "T")
        system.build_dict()


def test_the_level_is_its_own_closed_form_not_an_approximation():
    """The date the tank reaches 2.5 is 2.5; the instrument adds no lag."""
    result = a_chain(instrument()).simulate(t_max=5.0, samples=[2.5])
    assert math.isclose(sampled(result, "O_reading_level", 2.5), 2.5, abs_tol=1e-9)
