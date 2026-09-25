"""A volume holding several flows serves its stock as a mixture.

The H2 plant's ventilated room is the shape: a room of 100 holding one
unit of hydrogen in 99 of air, and consumers drawing on each constituent
through an output of its own. The reference (muscadet 5.6.0,
`draw_from_capacity`) does not serve each output from its own stock while
that stock lasts. It pools what the consumers ask BEYOND what transits,
and hands each held flow its share of that excess in proportion to its
raw content:

    out_f = min(req_f, transit_f + beyond * m_f / sum m),
    beyond = sum over held g of max(req_g - transit_g, 0),

for as long as `f` is stocked (empty, it passes on what arrives, as any
volume does). Weights are ignored: the split reads raw contents. A
single-flow volume and a volume upstream of the transfer (``side="in"``)
do not mix.

A rule fed by such a volume then receives less of one input than it
asked for, and produces from what arrives: it runs at the draw the
scarcer arrival allows and hands back the surplus of the others, so the
room loses exactly what the rule consumes.

Every number here is a closed form of the model the reference was
measured on, cross-checked against the reference's own trajectories
(agreement 1e-10 on the contents), and quoted where a value is pinned.
"""

import math

import pyraichu.muscadet as mu
from conftest import sampled
from test_rule_input_surplus import _forward_reads

H2_0, AIR_0 = 1.0, 99.0
INSTANTS = (1.0, 2.0, 3.0, 4.0, 5.0)
#: The contents are integrated by the engine and compared to an RK4
#: reference of the closed-form right-hand side: far above both errors.
CONTENT_TOL = 1e-6


def room_class(side="out", held=("H2", "AIR"), weights=None, volume=100.0):
    """A transiting room over `held`, starting with 1 of hydrogen and 99
    of air. Its inputs are declared and fed only where a test connects
    them, so an unconnected one transits nothing."""

    class Room(mu.ObjFlow):
        def add_flows(self):
            for flow in held:
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room",
                flows=[
                    {"name": flow, "weight": (weights or {}).get(flow, 1.0)}
                    for flow in held
                ],
                capacity=volume,
                content_init={flow: {"H2": H2_0, "AIR": AIR_0}[flow] for flow in held},
                fill_rate=0.0,
                transmits=True,
                side=side,
            )

    return Room


def consumer(flow: str, demand: float):
    class Consumer(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name=flow, var_demand_in_default=demand)

    return Consumer


def source(flow: str, rate: float):
    class Source(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name=flow, var_fed_default=rate)

    return Source


def plain_consumers(asks: dict[str, float], leak: float | None = None, **room):
    """ROOM served to one plain consumer per asked flow, with an optional
    leak of hydrogen transiting the room."""
    system = mu.System("capacity_mixture")
    if leak is not None:
        system.add_component(source("H2", leak), "LEAK")
    system.add_component(room_class(**room), "ROOM")
    for flow, demand in asks.items():
        system.add_component(consumer(flow, demand), f"C{flow}")
        system.connect("ROOM", flow, f"C{flow}", flow)
    if leak is not None:
        system.connect("LEAK", "H2", "ROOM", "H2")
    return system


def mixing_rule():
    """ROOM feeding one rule consuming 10 of hydrogen and 10 of air into
    20 of mixture, which a sink asks for whole."""

    class Mix(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="mix")
            self.add_rule_set(
                name="r",
                rules=[{"cons": {"H2": 10.0, "AIR": 10.0}, "prod": {"mix": 20.0}}],
            )

    system = mu.System("capacity_mixture_rule")
    system.add_component(room_class(), "ROOM")
    system.add_component(Mix, "MIX")
    system.add_component(consumer("mix", 20.0), "SINK")
    for flow in ("H2", "AIR"):
        system.connect("ROOM", flow, "MIX", flow)
    system.connect("MIX", "mix", "SINK", "mix")
    return system


def served(req, transit, h, a):
    """muscadet 5.6.0 `draw_from_capacity`, the non-mixture branch."""
    beyond = sum(max(req[k] - transit[k], 0.0) for k in req)
    total = h + a
    share = {"H2": h / total, "AIR": a / total}
    return {k: min(req[k], transit[k] + beyond * share[k]) for k in req}


def rk4(rhs, y, t1, n=20000):
    step, t = t1 / n, 0.0
    for _ in range(n):
        k1 = rhs(t, y)
        k2 = rhs(t + step / 2, [a + step / 2 * b for a, b in zip(y, k1)])
        k3 = rhs(t + step / 2, [a + step / 2 * b for a, b in zip(y, k2)])
        k4 = rhs(t + step, [a + step * b for a, b in zip(y, k3)])
        y = [
            a + step / 6 * (b + 2 * c + 2 * d + e)
            for a, b, c, d, e in zip(y, k1, k2, k3, k4)
        ]
        t += step
    return y


def pooled_contents(req, transit, t):
    """The two contents at `t` under the pooled draw."""

    def rhs(_, y):
        out = served(req, transit, y[0], y[1])
        return [transit["H2"] - out["H2"], transit["AIR"] - out["AIR"]]

    return rk4(rhs, [H2_0, AIR_0], t)


def assert_pooled(result, req, transit):
    for instant in INSTANTS:
        h, a = pooled_contents(req, transit, instant)
        assert abs(sampled(result, "ROOM_room_content_H2", instant) - h) < CONTENT_TOL
        assert abs(sampled(result, "ROOM_room_content_AIR", instant) - a) < CONTENT_TOL
        # What each consumer receives is what the room lets go of: the
        # draw at the SETTLED contents, the same instant.
        h = sampled(result, "ROOM_room_content_H2", instant)
        a = sampled(result, "ROOM_room_content_AIR", instant)
        out = served(req, transit, h, a)
        for flow in req:
            if f"C{flow}_{flow}_fed_in" in result.samples:
                fed = sampled(result, f"C{flow}_{flow}_fed_in", instant)
                assert abs(fed - out[flow]) < 1e-9, (flow, instant, fed, out[flow])
                assert (
                    abs(sampled(result, f"ROOM_{flow}_fed_out", instant) - fed) < 1e-9
                )


# --- U39a: the pooled draw -------------------------------------------------


def test_two_consumers_share_the_excess_in_proportion_to_the_contents():
    """Hydrogen and air each asked 10: the excess of 20 is split 1 to 99,
    so the hydrogen consumer gets 0.2 and the air one its whole 10.
    Reference at t = 1 and t = 5: hydrogen 0.8098196 and 0.2474811."""
    req, transit = {"H2": 10.0, "AIR": 10.0}, {"H2": 0.0, "AIR": 0.0}
    result = plain_consumers(req).simulate(t_max=5.0, samples=list(INSTANTS))
    assert_pooled(result, req, transit)
    assert (
        abs(sampled(result, "ROOM_room_content_H2", 1.0) - 0.8098196494483848)
        < CONTENT_TOL
    )
    assert (
        abs(sampled(result, "ROOM_room_content_H2", 5.0) - 0.24748106037863826)
        < CONTENT_TOL
    )
    assert abs(sampled(result, "ROOM_room_content_AIR", 5.0) - 49.0) < CONTENT_TOL


def test_a_small_air_request_still_pools_with_the_hydrogen_one():
    """Hydrogen 10, air 0.5: the excess is 10.5, and the hydrogen share
    of it is 0.105 at the start. Reference at t = 5: 0.5869052."""
    req, transit = {"H2": 10.0, "AIR": 0.5}, {"H2": 0.0, "AIR": 0.0}
    result = plain_consumers(req).simulate(t_max=5.0, samples=list(INSTANTS))
    assert_pooled(result, req, transit)
    assert (
        abs(sampled(result, "ROOM_room_content_H2", 5.0) - 0.5869052277042746)
        < CONTENT_TOL
    )


def test_hydrogen_alone_follows_its_closed_form():
    """Only hydrogen is asked: it leaves at 10 h/(h+a) with the air held
    at 99, whose solution is h - h0 + a ln(h/h0) = -10 t."""
    req = {"H2": 10.0}
    result = plain_consumers(req).simulate(t_max=5.0, samples=list(INSTANTS))
    for instant in INSTANTS:
        h = sampled(result, "ROOM_room_content_H2", instant)
        assert abs(h - H2_0 + AIR_0 * math.log(h / H2_0) + 10.0 * instant) < 1e-5
        assert abs(sampled(result, "ROOM_room_content_AIR", instant) - AIR_0) < 1e-12
        assert (
            abs(sampled(result, "CH2_H2_fed_in", instant) - 10.0 * h / (h + AIR_0))
            < 1e-9
        )
    assert (
        abs(sampled(result, "ROOM_room_content_H2", 5.0) - 0.6058823107809423)
        < CONTENT_TOL
    )


def test_what_transits_is_passed_on_before_the_excess_is_shared():
    """A leak of 0.06 of hydrogen into the room: the transit crosses
    whole, and the remaining 9.94 is pooled, 0.1594 at the start.
    Reference at t = 5: 0.6077099."""
    req = {"H2": 10.0}
    result = plain_consumers(req, leak=0.06).simulate(t_max=5.0, samples=list(INSTANTS))
    for instant in INSTANTS:
        h, _ = pooled_contents(
            {"H2": 10.0, "AIR": 0.0}, {"H2": 0.06, "AIR": 0.0}, instant
        )
        assert abs(sampled(result, "ROOM_room_content_H2", instant) - h) < CONTENT_TOL
    h = sampled(result, "ROOM_room_content_H2", 1.0)
    assert (
        abs(sampled(result, "CH2_H2_fed_in", 1.0) - (0.06 + 9.94 * h / (h + AIR_0)))
        < 1e-9
    )
    assert (
        abs(sampled(result, "ROOM_room_content_H2", 5.0) - 0.6077098838100133)
        < CONTENT_TOL
    )


def test_weights_change_the_fill_and_not_the_split():
    """Hydrogen weighing 2 in a room of 200: the same trajectory as with
    weights of 1, the split reading raw contents."""
    plain = plain_consumers({"H2": 10.0}).simulate(t_max=5.0, samples=list(INSTANTS))
    weighed = plain_consumers({"H2": 10.0}, weights={"H2": 2.0}, volume=200.0).simulate(
        t_max=5.0, samples=list(INSTANTS)
    )
    for instant in INSTANTS:
        assert (
            abs(
                sampled(plain, "ROOM_room_content_H2", instant)
                - sampled(weighed, "ROOM_room_content_H2", instant)
            )
            < 1e-9
        )


def test_a_single_flow_volume_serves_from_its_own_stock():
    """Nothing to mix: the hydrogen consumer gets its whole 10 until the
    unit of stock is gone, at t = 0.1."""
    result = plain_consumers({"H2": 10.0}, held=("H2",)).simulate(
        t_max=0.2, samples=[0.05]
    )
    assert abs(sampled(result, "CH2_H2_fed_in", 0.05) - 10.0) < 1e-9
    assert abs(sampled(result, "ROOM_room_content_H2", 0.05) - 0.5) < CONTENT_TOL


def test_a_volume_upstream_of_the_transfer_does_not_mix():
    """``side="in"``: the reference serves each held flow from its own
    stock (hydrogen 10 until it runs out), and so does this layer."""
    result = plain_consumers({"H2": 10.0}, side="in").simulate(
        t_max=0.2, samples=[0.05]
    )
    assert abs(sampled(result, "CH2_H2_fed_in", 0.05) - 10.0) < 1e-9
    assert abs(sampled(result, "ROOM_room_content_AIR", 0.05) - AIR_0) < 1e-12


def test_the_pooled_draw_reads_nothing_a_later_sweep_settles():
    """The excess reads the demands and the transits, both settled before
    the room's allocation in the sweep order, and the rule's draw reads
    the room's allocation, settled before the rule's own visit.

    A transiting volume already reads two quantities across bands, and
    both are older than this: its capability passes on what arrived
    once it is empty, and its demand, once it is full, asks for what
    leaves. Those are the capability and demand of the volume itself,
    read one evaluation late by design; nothing of the production band
    may join them."""
    for system in (
        plain_consumers({"H2": 10.0, "AIR": 10.0}),
        plain_consumers({"H2": 10.0}, leak=0.06),
        mixing_rule(),
    ):
        late = [
            (target, read)
            for target, read in _forward_reads(system.build_dict())
            if not target[1].endswith(("_capability_out", "_demand_in"))
        ]
        assert late == []


# --- U39b: a rule fed by the mixture ----------------------------------------


def test_a_rule_fed_by_the_mixture_produces_from_the_scarcer_arrival():
    """One rule asking 10 of each: the room hands it 0.2 of hydrogen and
    10 of air, the rule runs at the draw the hydrogen allows (0.02), and
    the 9.8 of air it does not consume stays in the room. Both contents
    then fall at 20 h/(h+a). Reference at t = 1 and t = 5: hydrogen
    0.8184230 and 0.3651482, air 98.8184230 and 98.3651482."""
    result = mixing_rule().simulate(t_max=5.0, samples=list(INSTANTS))

    def rhs(_, y):
        drawn = 20.0 * y[0] / (y[0] + y[1])
        return [-drawn, -drawn]

    for instant in INSTANTS:
        h_ref, a_ref = rk4(rhs, [H2_0, AIR_0], instant)
        h = sampled(result, "ROOM_room_content_H2", instant)
        a = sampled(result, "ROOM_room_content_AIR", instant)
        assert abs(h - h_ref) < CONTENT_TOL
        assert abs(a - a_ref) < CONTENT_TOL
        drawn = 20.0 * h / (h + a)
        for flow in ("H2", "AIR"):
            assert abs(sampled(result, f"MIX_{flow}_fed_in", instant) - drawn) < 1e-9
            assert abs(sampled(result, f"ROOM_{flow}_fed_out", instant) - drawn) < 1e-9
        # Conservation: the mixture is what the two arrivals make.
        assert abs(sampled(result, "SINK_mix_fed_in", instant) - 2.0 * drawn) < 1e-9
        assert abs((H2_0 - h) - (AIR_0 - a)) < 1e-9
    assert (
        abs(sampled(result, "ROOM_room_content_H2", 1.0) - 0.8184229849574184)
        < CONTENT_TOL
    )
    assert (
        abs(sampled(result, "ROOM_room_content_AIR", 5.0) - 98.36514819149393)
        < CONTENT_TOL
    )


# --- review: shapes the first version did not cover --------------------


def ventilation_rules():
    """Two rule sets both exhausting into one outlet, one extracting the
    room's hydrogen and one its air: the showcase's ventilation."""

    class Fan(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_flow_continuous_in(name="AIR")
            self.add_flow_continuous_out(name="exhaust")
            self.add_rule_set(
                name="extraction", rules=[{"cons": {"H2": 25.0}, "prod": {"exhaust": 25.0}}]
            )
            self.add_rule_set(
                name="renewal", rules=[{"cons": {"AIR": 25.0}, "prod": {"exhaust": 25.0}}]
            )

    return Fan


def test_a_stock_only_mixture_feeds_a_cascade():
    """A room whose hydrogen has no way in (it is held, never fed)
    feeds the two cascaded rule sets. No hydrogen transits it, so what the
    cascade expects the extraction to receive is its share of the pooled
    request alone; the model used to be refused for reading an arrival
    the room does not have."""

    class Store(mu.ObjFlow):
        def add_flows(self):
            # Air has a way in; hydrogen is held and never arrives.
            self.add_flow_continuous_in(name="AIR")
            for flow in ("H2", "AIR"):
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room", flows=["H2", "AIR"], capacity=100.0,
                content_init={"H2": H2_0, "AIR": AIR_0}, side="out",
                transmits=True,
            )

    system = mu.System("capacity_mixture_store")
    system.add_component(Store, "ROOM")
    system.add_component(ventilation_rules(), "FAN")
    system.add_component(consumer("exhaust", 50.0), "OUT")
    for flow in ("H2", "AIR"):
        system.connect("ROOM", flow, "FAN", flow)
    system.connect("FAN", "exhaust", "OUT", "exhaust")
    result = system.simulate(t_max=1.0, samples=[0.5])
    # Held but never fed, the hydrogen output keeps its declared rate of
    # zero, so the extraction draws nothing and the renewal asks the
    # whole 50 of air. A mixture asked for one constituent hands that
    # constituent's share only, as the reference does: 50 x a/(h + a).
    h = sampled(result, "ROOM_room_content_H2", 0.5)
    a = sampled(result, "ROOM_room_content_AIR", 0.5)
    assert abs(sampled(result, "FAN_AIR_fed_in", 0.5) - 50.0 * a / (h + a)) < 1e-6
    assert abs(
        sampled(result, "OUT_exhaust_fed_in", 0.5) - sampled(result, "FAN_AIR_fed_in", 0.5)
    ) < 1e-9


def test_two_rule_sets_on_one_mixed_input_never_draw_more_than_arrives():
    """Two rule sets of one component both consume the room's hydrogen.
    Each is bounded by the share of the need that arrived, so together
    they consume exactly what the room handed over, never more."""

    class TwoUsers(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="H2")
            self.add_flow_continuous_out(name="p")
            self.add_flow_continuous_out(name="q")
            self.add_rule_set(name="one", rules=[{"cons": {"H2": 10.0}, "prod": {"p": 1.0}}])
            self.add_rule_set(name="two", rules=[{"cons": {"H2": 5.0}, "prod": {"q": 1.0}}])

    system = mu.System("capacity_mixture_two_users")
    system.add_component(room_class(), "ROOM")
    system.add_component(TwoUsers, "USE")
    system.add_component(consumer("p", 1.0), "P")
    system.add_component(consumer("q", 1.0), "Q")
    system.connect("ROOM", "H2", "USE", "H2")
    system.connect("USE", "p", "P", "p")
    system.connect("USE", "q", "Q", "q")
    result = system.simulate(t_max=1.0, samples=[0.5])
    arrived = sampled(result, "USE_H2_fed_in", 0.5)
    made_p = sampled(result, "P_p_fed_in", 0.5)
    made_q = sampled(result, "Q_q_fed_in", 0.5)
    assert arrived < 15.0 * 0.05  # a trace share of the 15 asked
    assert abs(10.0 * made_p + 5.0 * made_q - arrived) < 1e-9
    assert abs(made_p / made_q - 1.0) < 1e-9  # both sets scaled alike


def test_a_room_whose_own_rules_carry_a_held_flow_does_not_mix():
    """A rule of the room's own component carries its air, so the volume
    is served by those rules and does not pool: the hydrogen consumer
    gets what it asks from the hydrogen stock, as before."""

    class RoomWithRule(mu.ObjFlow):
        def add_flows(self):
            for flow in ("H2", "AIR"):
                self.add_flow_continuous_out(name=flow)
            self.add_flow_continuous_out(name="stale")
            self.add_capacity(
                name="room", flows=["H2", "AIR"], capacity=100.0,
                content_init={"H2": H2_0, "AIR": AIR_0}, side="out",
            )
            self.add_rule_set(name="age", rules=[{"cons": {}, "prod": {"AIR": 1.0}}])

    system = mu.System("capacity_mixture_excluded")
    system.add_component(RoomWithRule, "ROOM")
    system.add_component(consumer("H2", 0.5), "CH2")
    system.connect("ROOM", "H2", "CH2", "H2")
    body = system.build_dict()["model"]
    room = next(c for c in body["components"] if c["name"] == "ROOM")
    assert not any(e["target"].endswith("_pooled_out") for e in room["equations"])


def test_three_constituents_share_the_excess_in_proportion():
    """Three held flows, 1 of hydrogen, 1 of carbon monoxide and 98 of
    air; only hydrogen is asked, 10: it receives 10 x 1/100."""

    class Room3(mu.ObjFlow):
        def add_flows(self):
            for flow in ("H2", "CO", "AIR"):
                self.add_flow_continuous_in(name=flow)
                self.add_flow_continuous_out(name=flow)
            self.add_capacity(
                name="room", flows=["H2", "CO", "AIR"], capacity=100.0,
                content_init={"H2": 1.0, "CO": 1.0, "AIR": 98.0}, side="out",
                transmits=True,
            )

    system = mu.System("capacity_mixture_three")
    system.add_component(Room3, "ROOM")
    system.add_component(consumer("H2", 10.0), "CH2")
    system.connect("ROOM", "H2", "CH2", "H2")
    result = system.simulate(t_max=0.001, samples=[0.0005])
    # 1e-5: the hydrogen share itself drains by about 10 x 0.0005 x 1 %
    # before the sample.
    assert abs(sampled(result, "CH2_H2_fed_in", 0.0005) - 0.1) < 1e-5
