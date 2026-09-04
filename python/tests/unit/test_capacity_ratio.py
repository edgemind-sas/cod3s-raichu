"""What fraction of a volume one of its constituents is.

A volume already published two quantities per constituent: the raw
`level` it holds, and the `fill` that content occupies of the declared
volume. Neither is the fraction of the **mixture**, and the fraction of
the mixture is what a flammability threshold is written on: two per cent
of hydrogen in a room means two per cent of what the room holds, not two
per cent of the room.

A controller cannot compute it: its output grammar is closed at four
operators and none of them is arithmetic, which is deliberate. So a
fraction a controller can threshold is one the volume publishes, and this
is that publication.

- `{cap}_ratio_{flow}` is `content_{flow} / content_total`, published per
  constituent by a volume holding **more than one**. A volume holding a
  single constituent publishes none: its ratio is identically one
  wherever it holds anything, so it would carry no information and every
  single-flow volume in every model would pay a variable for it.
- An **empty** volume reads 0 on every ratio. Nothing is no fraction of
  nothing, and the alternative is a division the sweep would have to
  answer with a non-number.
- The ratios are swept after the total they divide by and before any
  observer reads them, which is asserted on the emitted order and not
  only on a value: read one sweep late, a ratio carries the previous
  instant's mixture and every number still looks plausible.
"""

import pytest

import pyraichu
import pyraichu.muscadet as mu
from conftest import CROSSING_TOL, sampled


def room(content: dict[str, float], flows=("H2", "O2")) -> type:
    class Room(mu.ObjFlow):
        def add_flows(self):
            for flow in flows:
                self.add_flow_continuous_in(
                    name=flow, var_demand_in_default=1.0 if flow == "H2" else 0.0
                )
            self.add_capacity(
                name="room",
                flows=list(flows),
                capacity=100.0,
                content_init=dict(content),
                fill_rate=1.0e6,
            )

    return Room


class Supply(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="H2", var_fed_default=1.0)


class Observer(mu.ObjFlow):
    def add_flows(self):
        self.add_measurement_in(name="room", flows=["H2", "O2"])


def watched_room(content: dict[str, float]) -> mu.System:
    system = mu.System("mixture")
    system.add_component(Supply, "S")
    system.add_component(room(content), "R")
    system.add_component(Observer, "W")
    system.connect("S", "H2", "R", "H2")
    system.connect_measurement("R", "room", "W")
    return system


def test_a_ratio_is_the_fraction_of_the_mixture():
    """Ninety of oxygen and a hydrogen inflow of one: at t = 5 the room
    holds 5 and 90, and the hydrogen ratio is 5/95 and not 5/100.

    The distinction is the whole point. `fill` divides by the declared
    volume and answers how full the room is; the ratio divides by what
    the room holds and answers what the mixture is."""
    result = watched_room({"H2": 0.0, "O2": 90.0}).simulate(
        t_max=10.0, samples=[0.0, 5.0, 10.0]
    )
    for instant, expected in ((0.0, 0.0), (5.0, 5 / 95), (10.0, 10 / 100)):
        assert abs(sampled(result, "R_room_ratio_H2", instant) - expected) < (
            CROSSING_TOL
        ), instant
    # And the fill, on the same run, answers the other question.
    assert abs(sampled(result, "R_room_fill_H2", 5.0) - 5 / 100) < CROSSING_TOL


def test_the_ratios_of_a_volume_sum_to_one():
    """A conservation identity, and the reason there is no total ratio:
    it would be the constant 1."""
    result = watched_room({"H2": 0.0, "O2": 90.0}).simulate(
        t_max=10.0, samples=[2.0, 6.0, 9.0]
    )
    for instant in (2.0, 6.0, 9.0):
        total = sampled(result, "R_room_ratio_H2", instant) + sampled(
            result, "R_room_ratio_O2", instant
        )
        assert abs(total - 1.0) < CROSSING_TOL, instant


def test_an_empty_volume_reads_zero_on_every_ratio():
    """Nothing is no fraction of nothing. The convention is asserted
    because the alternative is a division by zero the sweep would have to
    answer with a non-number, which would then travel."""
    system = mu.System("empty")
    system.add_component(room({"H2": 0.0, "O2": 0.0}), "R")
    system.add_component(Observer, "W")
    system.connect_measurement("R", "room", "W")
    result = system.simulate(t_max=1.0, samples=[0.0, 1.0])
    for flow in ("H2", "O2"):
        for instant in (0.0, 1.0):
            assert sampled(result, f"R_room_ratio_{flow}", instant) == 0.0


def test_the_observer_reads_the_same_number_the_volume_published():
    result = watched_room({"H2": 0.0, "O2": 90.0}).simulate(
        t_max=10.0, samples=[0.0, 5.0, 10.0]
    )
    for instant in (0.0, 5.0, 10.0):
        for flow in ("H2", "O2"):
            assert sampled(result, f"W_room_ratio_{flow}", instant) == sampled(
                result, f"R_room_ratio_{flow}", instant
            ), (flow, instant)


def test_the_ratios_are_swept_after_the_total_and_before_their_observer():
    """Asserted on the emitted order, not on a value: read one sweep
    late, a ratio carries the previous instant's mixture and every number
    still looks plausible. That is exactly what happened while this was
    being built."""
    order = [
        f"{entry['component']}.{entry['attribute']}"
        for entry in watched_room({"H2": 0.0, "O2": 90.0})
        .build_dict()["model"]["evaluation_order"]
    ]
    total = order.index("R.room_content")
    published = order.index("R.room_ratio_H2")
    read = order.index("W.room_ratio_H2")
    assert total < published < read, order[: read + 1]


def test_a_single_constituent_volume_publishes_no_ratio():
    """Its ratio is identically one wherever it holds anything, so it
    would carry no information, and every single-flow volume in every
    model would pay a variable and an equation for it."""
    system = mu.System("single")
    system.add_component(room({"H2": 5.0}, flows=("H2",)), "R")
    document = system.build_dict()["model"]["components"][0]
    names = {variable["name"] for variable in document["attributes"]}
    assert "room_fill_H2" in names
    assert not [name for name in names if "_ratio_" in name]


def test_a_controller_thresholds_a_published_ratio():
    """The reason the publication exists: a controller reads a ratio the
    way it reads a level, and can compare it to a threshold it could not
    have computed."""
    spec = {
        "name": "vented_room",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFlow",
                        "name": "S",
                        "flows_continuous_out": [
                            {"name": "H2", "var_fed_default": 1.0}
                        ],
                    },
                    {
                        "type": "ObjFlow",
                        "name": "R",
                        "flows_continuous_in": [
                            {"name": "H2", "var_demand_default": 1.0},
                            {"name": "O2"},
                        ],
                        "capacities": [
                            {
                                "name": "room",
                                "flows": ["H2", "O2"],
                                "capacity": 100.0,
                                "content_init": {"H2": 0.0, "O2": 90.0},
                                "fill_rate": 1.0e6,
                            }
                        ],
                    },
                    {
                        "type": "ObjCtrl",
                        "name": "CTRL",
                        "controls_in": [
                            {"name": "room", "kind": "ratio", "flows": ["H2"]}
                        ],
                        "controls_out": [
                            {
                                "name": "vent",
                                "kind": "bool",
                                "emit": {
                                    "op": "compare",
                                    "input": "room",
                                    "operator": ">",
                                    "threshold": 0.05,
                                },
                            }
                        ],
                    },
                ]
            }
        },
        "components": [],
        "connections": [
            {
                "from": {"component": "S", "port": "H2_out"},
                "to": {"component": "R", "port": "H2_in"},
            },
            {
                "from": {"component": "R", "port": "room_ratio_H2_out"},
                "to": {"component": "CTRL", "port": "room_ratio_H2_in"},
            },
        ],
        "indicators": [],
    }
    body = pyraichu.expand_model(spec)
    result = pyraichu.simulate(
        pyraichu.load_model(body), t_max=12.0, samples=[1.0, 11.0]
    )
    # 5 % of the mixture is reached when h/(h+90) = 0.05, at h = 90/19,
    # which the inflow of 1 reaches at t = 4.7368.
    assert abs(90.0 / 19.0 - 4.736842105263158) < 1e-12
    assert sampled(result, "CTRL_vent", 1.0) is False
    assert sampled(result, "CTRL_vent", 11.0) is True
