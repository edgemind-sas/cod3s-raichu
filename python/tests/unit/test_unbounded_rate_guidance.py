"""A stock drained without bound is refused by name, in the modeller's words.

A volume holding a stock, with no ceiling on what leaves it (``serve_rate``
left at its unbounded default), feeding something that takes without bound (a
room with an infinite ``fill_rate``) has an instantaneous transfer for physics,
which no rate expresses. Since 0.38.0 the engine refuses the rate by name
rather than integrating the magnitude the model reserves for "unbounded".

What this file pins is what the modeller is told, and that the advice works:

- the refusal is a typed :class:`pyraichu.UnboundedRateError` carrying the
  variable, on every route a muscadet model runs through;
- the muscadet layer names the capacity and the key to declare
  (``serve_rate``), and keeps the engine's own message in front of it;
- with a finite ``serve_rate`` the same model runs, and drains at that rate.

The numbers of the last point were measured on the reference engine
(muscadet 5.7.0 + PyCATSHOO 1.3.8.0, 2026-09-28) on this exact model, and both
engines agree on them: they are also closed forms, a stock of 5 fed at 0.01
and drained at ``serve_rate``.
"""

import math

import pyraichu
import pyraichu.muscadet as mu
import pytest
from pyraichu.muscadet import engine as muscadet_engine

STOCK = 5.0
INFLOW = 0.01
ROOM_INIT = 2.0
#: The substring the platform's refusal pin reads (backend
#: `test_reference_simulations.py`): the engine's message stays in front.
ENGINE_WORDING = 'reserves for "unbounded"'


def stocked_membrane(serve_rate=None):
    """Source -> membrane holding a stock beside a 1:1 rule -> unbounded room."""

    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2m", var_fed_default=INFLOW)

    class Membrane(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2m")
            self.add_flow_continuous_out(name="H2")
            ceiling = {} if serve_rate is None else {"serve_rate": serve_rate}
            self.add_capacity(
                name="membrane",
                flows=[{"name": "H2m", "weight": 1.0}],
                capacity=100.0,
                content_init={"H2m": STOCK},
                fill_rate=math.inf,
                side="in",
                **ceiling,
            )
            self.add_rule_set(
                name="release", rules=[{"cons": {"H2m": 1.0}, "prod": {"H2": 1.0}}]
            )

    class Room(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_capacity(
                name="room",
                flow="H2",
                capacity=100.0,
                content_init={"H2": ROOM_INIT},
                fill_rate=math.inf,
            )

    system = mu.System("stocked_membrane")
    system.add_component(Source, "SRC")
    system.add_component(Membrane, "MEM")
    system.add_component(Room, "ROOM")
    system.connect("SRC", "H2m", "MEM", "H2m")
    system.connect("MEM", "H2", "ROOM", "H2")
    return system


def assert_explained(error):
    """The typed refusal, naming the variable, the capacity and the key."""
    assert isinstance(error, pyraichu.SimulationError)
    assert error.variable == "MEM.membrane_content_H2m"
    assert error.time == 0.0
    assert abs(error.rate) >= error.unbounded
    message = str(error)
    assert ENGINE_WORDING in message, message
    assert "capacity `membrane` of `MEM`" in message, message
    assert "Declare a finite `serve_rate` on capacity `membrane`" in message, message


def test_a_single_trajectory_is_refused_with_the_capacity_to_bound():
    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        stocked_membrane().simulate(t_max=1.0, samples=[1.0])
    assert_explained(caught.value)


def test_a_campaign_is_refused_the_same_way():
    """The class survives the campaign's replica wrapping, and so do the
    replay coordinates, through a pickle round trip as well."""
    import pickle

    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        stocked_membrane().monte_carlo(nb_runs=2, t_max=1.0, samples=[1.0])
    assert_explained(caught.value)
    assert (caught.value.replica, caught.value.nb_runs) == (0, 2)
    copy = pickle.loads(pickle.dumps(caught.value))
    assert (copy.variable, copy.replica, copy.nb_runs, copy.seed) == (
        caught.value.variable,
        0,
        2,
        caught.value.seed,
    )


def test_the_platform_route_is_refused_the_same_way(monkeypatch):
    """`engine.simulate` is what muscadet calls for ``engine="raichu"``. Only
    the build is substituted, so the run and its refusal are the real ones."""
    model = stocked_membrane().build_model()
    monkeypatch.setattr(muscadet_engine, "build_model", lambda spec, targets: model)
    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        muscadet_engine.simulate({}, {"nb_runs": 1, "schedule": [1.0], "seed": 1})
    assert_explained(caught.value)


def test_a_variable_that_is_no_capacity_content_keeps_the_engine_message():
    """The explanation names a volume only when the variable is one's content,
    read from the model: anything else is handed back untouched."""
    model = stocked_membrane().build_model()
    error = pyraichu.UnboundedRateError(
        "engine says", "SRC.H2m_fed_out", 0.0, 1e30, 1e30
    )
    assert mu.explain_unbounded_rate(error, model) is error


def test_the_typed_error_survives_a_pickle_round_trip():
    """A Monte-Carlo worker process hands its error back pickled."""
    import pickle

    error = pyraichu.UnboundedRateError("engine says", "MEM.x", 1.5, -1e30, 1e30)
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.variable, copy.time, copy.rate, copy.unbounded) == (
        "engine says",
        "MEM.x",
        1.5,
        -1e30,
        1e30,
    )


@pytest.mark.parametrize("serve_rate", [0.5, 2.0])
def test_a_declared_serve_rate_drains_the_stock_at_that_rate(serve_rate):
    """The advice works: the stock leaves at `serve_rate` until it is empty,
    then passes its inflow on; the room receives exactly what leaves."""
    instants = [0.5, 1.0, 2.0, 4.0, 8.0]
    result = stocked_membrane(serve_rate).simulate(t_max=8.0, samples=instants)
    empty_at = STOCK / (serve_rate - INFLOW)
    for (t, membrane), (_, room) in zip(
        result.samples["MEM_membrane_content_H2m"],
        result.samples["ROOM_room_content_H2"],
    ):
        if t < empty_at:
            expected_membrane = STOCK - (serve_rate - INFLOW) * t
            expected_room = ROOM_INIT + serve_rate * t
        else:
            expected_membrane = 0.0
            expected_room = ROOM_INIT + STOCK + INFLOW * t
        assert membrane == pytest.approx(expected_membrane, abs=1e-6), t
        assert room == pytest.approx(expected_room, abs=1e-6), t


def leaking_room(serve_rate=None):
    """A leak into a room holding H2 in air, extracted by a fan whose exhaust
    nothing reads (issue #158, reduced from a platform study).

    The room's volume holds a mixture and serves each constituent at its
    share, so with ``serve_rate`` unbounded the fan's draw on H2 is the
    stand-in scaled by the H2 share (2 % here): under the magnitude the
    engine checks, and far beyond what the step controller can follow. The
    run used to end on "step size underflow ... too stiff"."""

    class Leak(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2", var_fed_default=1.0)

    class Room(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_flow_continuous_out(name="H2")
            self.add_flow_continuous_out(name="AIR")
            ceiling = {} if serve_rate is None else {"serve_rate": serve_rate}
            self.add_capacity(
                name="local",
                flows=[{"name": "H2", "weight": 1.0}, {"name": "AIR", "weight": 1.0}],
                capacity=1000.0,
                side="out",
                content_init={"H2": 2.0, "AIR": 100.0},
                fill_rate=1.0,
                transmits=True,
                **ceiling,
            )

    class Fan(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_flow_continuous_out(name="exhaust")
            self.add_rule_set(
                name="extraction",
                rules=[{"cons": {"H2": 51.0}, "prod": {"exhaust": 51.0}}],
            )

    system = mu.System("leaking_room")
    system.add_component(Leak, "LEAK")
    system.add_component(Room, "LOCAL")
    system.add_component(Fan, "VENTILATION")
    system.connect("LEAK", "H2", "LOCAL", "H2")
    system.connect("LOCAL", "H2", "VENTILATION", "H2")
    return system


def test_a_draw_diluted_by_a_mixture_share_is_refused_rather_than_called_stiff():
    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        leaking_room().simulate(t_max=1.0, samples=[1.0])
    error = caught.value
    assert error.variable == "LOCAL.local_content_H2"
    assert error.time == 0.0
    assert 0.0 < abs(error.rate) < error.unbounded
    message = str(error)
    assert ENGINE_WORDING in message, message
    assert "capacity `local` of `LOCAL`" in message, message
    assert "whose outputs nothing reads" in message, message


def test_the_diluted_draw_runs_once_the_volume_states_its_rate():
    """The advice works: with a finite ceiling the H2 stock drains at its
    share of it. Over the first instants the share is about 2/102, so the
    stock, fed at 1, moves at 1 - 51 x 2/102 = 0, within a percent."""
    result = leaking_room(serve_rate=51.0).simulate(t_max=0.01, samples=[0.01])
    (h2,) = [
        points[-1][1]
        for name, points in result.samples.items()
        if name.endswith("local_content_H2")
    ] or [None]
    assert h2 is not None and abs(h2 - 2.0) < 0.02, h2


def test_a_mixture_volume_feeding_an_unbounded_room_is_refused_too():
    """The diluted draw where the stand-in is the room's fill claim (an
    infinite ``fill_rate``), written inline rather than held by an
    attribute: the same refusal, on the same variable."""

    class Leak(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="H2", var_fed_default=1.0)

    class Local(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_flow_continuous_out(name="H2")
            self.add_flow_continuous_out(name="AIR")
            self.add_capacity(
                name="local",
                flows=[{"name": "H2", "weight": 1.0}, {"name": "AIR", "weight": 1.0}],
                capacity=1000.0,
                side="out",
                content_init={"H2": 2.0, "AIR": 100.0},
                fill_rate=1.0,
                transmits=True,
            )

    class Room(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_capacity(
                name="room",
                flow="H2",
                capacity=100.0,
                content_init={"H2": 0.0},
                fill_rate=math.inf,
            )

    system = mu.System("mixture_into_room")
    system.add_component(Leak, "LEAK")
    system.add_component(Local, "LOCAL")
    system.add_component(Room, "ROOM")
    system.connect("LEAK", "H2", "LOCAL", "H2")
    system.connect("LOCAL", "H2", "ROOM", "H2")
    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        system.simulate(t_max=1.0, samples=[1.0])
    assert caught.value.variable == "LOCAL.local_content_H2"
    assert "capacity `local` of `LOCAL`" in str(caught.value)
