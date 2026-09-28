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
    with pytest.raises(pyraichu.UnboundedRateError) as caught:
        stocked_membrane().monte_carlo(nb_runs=2, t_max=1.0, samples=[1.0])
    assert_explained(caught.value)


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
