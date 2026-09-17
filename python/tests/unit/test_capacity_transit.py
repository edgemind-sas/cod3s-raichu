"""What a volume passes on, and where that is declared.

muscadet documents a `CapacityContinuous` at its default `ports="both"` as
a **buffer**: "what the volume does not hold back it transfers unchanged".
That behaviour lived in the Python of the KB class and in no key of the
exported declaration, so a tank read back here delivered nothing at all to
its downstream: the volume's own capability was read off the out-flow's
declared rate, which the exporting component writes as zero because it
produces nothing of its own, and a ceiling of zero leaves zero however
full the volume is.

Two halves close it, and this module pins both:

- **the volume is what its output delivers**, in place of that rate. It
  serves whatever is asked while it holds something, capped by
  `serve_rate` and by nothing else, and once empty only what currently
  crosses it. That is muscadet's `Capacity.serve_limit`, term for term;
- **a volume that transits asks upstream for what is asked of it**, so
  the shortage is smoothed rather than swallowed, and `fill_rate` is what
  it claims for itself on top of that.

`transmits` is where the through-path is DECLARED, rather than guessed
from the shape of the document. It has to be a key, because the two
shapes it separates are indistinguishable otherwise: muscadet maps both
`ports="both"` and `ports="out"` onto `side="out"`, so a buffer and a
reservoir arrive here identical but for the presence of an input flow of
the same name -- a convention the document never states.
"""

import math

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu
from conftest import TOL, sampled

#: The running example, in the units the ticket states it in: a source of
#: two, a volume of a hundred holding ten, a consumer asking for one.
SOURCE_RATE = 2.0
VOLUME = 100.0
HELD = 10.0
DEMAND = 1.0


class Source(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="q", var_fed_default=SOURCE_RATE)


class Sink(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_in(name="q", var_demand_in_default=DEMAND)


def tank_class(**capacity):
    """A both-sided volume over `q`, declaring no production of its own.

    The shape a muscadet `CapacityContinuous` exports: an input and an
    output of the same name, and an out-flow rate left at zero because
    the component makes nothing.
    """
    settings = dict(name="tank", flow="q", capacity=VOLUME, side="out")
    settings.update(capacity)

    class Tank(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="q")
            self.add_flow_continuous_out(name="q")
            self.add_capacity(**settings)

    return Tank


def tank_system(**capacity) -> mu.System:
    """Source, volume, consumer, in that order.

    `transmits=True` unless a case says otherwise, because that is what
    makes this volume the muscadet buffer it stands in for. It is not the
    default of `add_capacity`, and deliberately so: see
    `test_a_volume_says_whether_it_transits_rather_than_being_assumed_to`.
    """
    settings = dict(content_init={"q": HELD}, transmits=True)
    settings.update(capacity)
    system = mu.System("tank_transit")
    system.add_component(Source, "SRC")
    system.add_component(tank_class(**settings), "CAP")
    system.add_component(Sink, "SINK")
    system.connect("SRC", "q", "CAP", "q")
    system.connect("CAP", "q", "SINK", "q")
    return system


# --- 1. the volume delivers what it does not hold back ------------------


@pytest.mark.parametrize("fill_rate", [0.0, 0.5, 2.0, 1e9, None])
def test_a_transiting_volume_delivers_to_its_downstream(fill_rate):
    """The consumer receives what it asked for, whatever the volume
    claims for itself.

    Parametrised over the fill rate because that is what the measurement
    on the two engines was: the rate decides how fast the volume rises
    and decides nothing at all about what crosses it. `None` is the
    declared spelling of an unbounded claim.
    """
    result = tank_system(fill_rate=fill_rate).simulate(
        t_max=20.0, samples=[0.0, 5.0, 10.0]
    )
    for instant in (0.0, 5.0, 10.0):
        assert abs(sampled(result, "SINK_q_fed_in", instant) - DEMAND) < TOL, instant


def test_the_volume_is_what_the_output_delivers_and_not_its_declared_rate():
    """The heart of it: the component declares an out-flow rate of zero,
    because it produces nothing, and still serves from its stock.

    Read off the published capability rather than off the delivery, so
    the test says the volume REPLACED the rate rather than that the
    consumer happened to ask for little."""
    result = tank_system(fill_rate=0.0).simulate(t_max=5.0, samples=[1.0])
    assert sampled(result, "CAP_q_capability_out", 1.0) > DEMAND
    assert abs(sampled(result, "CAP_q_fed_out", 1.0) - DEMAND) < TOL


# --- 2. and the fill rate keeps its meaning -----------------------------


@pytest.mark.parametrize("fill_rate", [0.0, 0.5])
def test_the_fill_rate_is_what_the_volume_takes_beyond_the_draw(fill_rate):
    """R36: over and above what the downstream draws, the volume
    accumulates at its declared rate and at no other.

    The source can deliver two and the consumer asks for one, so there is
    room for a claim of up to one; both rates here sit under it, which is
    what makes the claim and not the supply the thing being measured.
    """
    result = tank_system(fill_rate=fill_rate).simulate(
        t_max=20.0, samples=[0.0, 5.0, 10.0]
    )
    for instant in (0.0, 5.0, 10.0):
        entering = sampled(result, "CAP_q_fed_in", instant)
        leaving = sampled(result, "CAP_q_fed_out", instant)
        assert abs(entering - (DEMAND + fill_rate)) < TOL, instant
        assert abs(entering - leaving - fill_rate) < TOL, instant
    # And the content is the claim integrated, from where it started.
    assert abs(sampled(result, "CAP_tank_content", 10.0) - (HELD + 10.0 * fill_rate)) < (
        TOL
    )


def test_a_claim_larger_than_the_supply_takes_what_there_is():
    """The claim is a ceiling on what the volume asks for, not a promise
    that it gets it: asking for three where two are made leaves one to
    accumulate once the consumer has drawn its own."""
    result = tank_system(fill_rate=2.0).simulate(t_max=20.0, samples=[10.0])
    assert abs(sampled(result, "CAP_q_fed_in", 10.0) - SOURCE_RATE) < TOL
    assert abs(sampled(result, "SINK_q_fed_in", 10.0) - DEMAND) < TOL
    held = sampled(result, "CAP_tank_content", 10.0)
    assert abs(held - (HELD + 10.0 * (SOURCE_RATE - DEMAND))) < TOL, held


def test_the_claim_stops_at_the_bound_and_the_transit_does_not():
    """"Until it is full": at the bound the volume asks only for what
    leaves it, so the content sits there while the consumer keeps being
    served in full."""
    system = tank_system(fill_rate=SOURCE_RATE, content_init={"q": VOLUME - 5.0})
    result = system.simulate(t_max=30.0, samples=[10.0, 20.0, 30.0])
    for instant in (10.0, 20.0, 30.0):
        held = sampled(result, "CAP_tank_content", instant)
        assert VOLUME - mu.DEFAULT_HYSTERESIS * VOLUME <= held <= VOLUME + 1e-6, instant
        # Full, and still passing the draw on: what enters equals what
        # leaves, and the consumer is served.
        assert abs(sampled(result, "SINK_q_fed_in", instant) - DEMAND) < TOL, instant
        assert (
            abs(
                sampled(result, "CAP_q_fed_in", instant)
                - sampled(result, "CAP_q_fed_out", instant)
            )
            < TOL
        ), instant


# --- 3. and nothing else moved ------------------------------------------


def test_an_output_with_no_rule_and_no_volume_still_produces_its_rate():
    """The rule ticket `ce40fc97` pinned stays pinned: a continuous
    output with no rule set and no capacity produces `var_fed_default`,
    which is what lets an unruled source deliver its seed.

    The transit above is declared of a VOLUME, so it reaches no flow that
    has none."""

    class Seeded(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_out(name="q", var_fed_default=3.0)

    system = mu.System("seeded")
    system.add_component(Seeded, "S")
    system.add_component(Sink, "L")
    system.connect("S", "q", "L", "q")
    result = system.simulate(t_max=5.0, samples=[1.0])
    assert abs(sampled(result, "S_q_capability_out", 1.0) - 3.0) < TOL
    assert abs(sampled(result, "L_q_fed_in", 1.0) - DEMAND) < TOL


def test_a_declared_rate_still_bounds_a_volume_that_has_one():
    """A rate that SAYS something is a ceiling the volume respects.

    Only a rate of zero is replaced, and a zero rate is the absence of a
    ceiling rather than a ceiling at nothing: the product is identically
    zero, so reading it as a bound would leave the output delivering
    nothing however full the volume was. A tank rated at half the demand
    delivers half."""

    class Rated(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="q")
            self.add_flow_continuous_out(name="q", var_fed_default=0.5)
            self.add_capacity(
                name="tank", flow="q", capacity=VOLUME, content_init={"q": HELD}
            )

    system = mu.System("rated")
    system.add_component(Source, "SRC")
    system.add_component(Rated, "CAP")
    system.add_component(Sink, "SINK")
    system.connect("SRC", "q", "CAP", "q")
    system.connect("CAP", "q", "SINK", "q")
    result = system.simulate(t_max=5.0, samples=[1.0])
    assert abs(sampled(result, "SINK_q_fed_in", 1.0) - 0.5) < TOL


def test_a_serve_rate_is_the_ceiling_a_volume_declares_for_itself():
    """And the way to bound a volume that produces nothing: the key
    muscadet spells the same way, defaulting to unbounded on both."""
    result = tank_system(fill_rate=0.0, serve_rate=0.25).simulate(
        t_max=5.0, samples=[1.0]
    )
    assert abs(sampled(result, "SINK_q_fed_in", 1.0) - 0.25) < TOL
    assert abs(sampled(result, "CAP_q_capability_out", 1.0) - 0.25) < TOL


# --- 4. a volume that declares no through-path --------------------------


def test_a_volume_with_no_through_path_serves_only_what_it_holds():
    """`transmits=False` is a pure store: emptied, it delivers nothing,
    where a buffer would still pass on what crosses it.

    Both volumes start empty and both are fed, so the only difference
    between the two runs is the key."""
    empty = dict(content_init={"q": 0.0}, fill_rate=0.0)
    stores = tank_system(transmits=False, **empty).simulate(t_max=5.0, samples=[2.0])
    buffers = tank_system(transmits=True, **empty).simulate(t_max=5.0, samples=[2.0])

    assert abs(sampled(stores, "SINK_q_fed_in", 2.0)) < TOL
    assert abs(sampled(buffers, "SINK_q_fed_in", 2.0) - DEMAND) < TOL


def test_a_transiting_buffer_cannot_be_under_declared():
    """The chatter `test_capacity_bound_chatter` diagnoses has no way in
    once the volume reads its own outlet.

    That module's buffer declares `var_demand_in_default=0`, asks for
    nothing at its bound, drops below it and refills, turning at the
    scale of the hysteresis. A TRANSITING volume carries the downstream
    demand upstream by itself, so what enters equals what leaves at the
    bound and there is nothing to chatter on: the same model, with the
    same missing declaration, settles."""
    system = tank_system(
        transmits=True,
        fill_rate=SOURCE_RATE,
        content_init={"q": VOLUME - 1.0},
    )
    result = system.simulate(t_max=50.0, samples=[50.0], max_transition_firings=10)
    assert abs(sampled(result, "SINK_q_fed_in", 50.0) - DEMAND) < TOL


# --- 5. the key is declared, and the reader carries it ------------------


def capacity_spec(**extra) -> dict:
    """A muscadet-shaped capacity document, as `system_spec` writes one."""
    capacity = {
        "name": "tank",
        "capacity": VOLUME,
        "content_init": {"q": HELD},
        "fill_rate": 0.0,
        "side": "out",
        "flows": [{"name": "q", "weight": 1.0, "side": "out", "cls": "CapacityFlow"}],
    }
    capacity.update(extra)
    return {
        "name": "CAP",
        "cls": "ObjFlow",
        "flows": [
            {
                "name": "q",
                "cls": "FlowContinuousIn",
                "var_type": "float",
                "var_fed_default": 0.0,
                "var_in_default": 0.0,
            },
            {
                "name": "q",
                "cls": "FlowContinuousOut",
                "var_fed_default": 0.0,
                "var_demand_in_default": 0.0,
                "allocation": "proportional",
                "var_prod_cond_inner_mode": "or",
            },
        ],
        "capacities": [capacity],
        "rules": [],
        "transfers": [],
        "automata": [],
        "failure_modes": [],
    }


@pytest.mark.parametrize("transmits", [True, False])
def test_the_reader_carries_the_declared_through_path(transmits):
    """muscadet writes `transmits` on EVERY exported capacity, at its
    default too, so a reader that did not know the key refused every
    continuous model carrying a volume."""
    declare.check_spec(capacity_spec(transmits=transmits))
    built = declare.build_component(mu.System("read"), capacity_spec(transmits=transmits))
    assert built.capacities[0].transmits is transmits


def test_the_reader_carries_a_declared_serve_rate():
    """Omitted at muscadet's default, which is unbounded, and unbounded
    is what a volume on an output means."""
    unbounded = declare.build_component(mu.System("read"), capacity_spec())
    assert unbounded.capacities[0].serve_rate == math.inf
    built = declare.build_component(mu.System("read"), capacity_spec(serve_rate=4.0))
    assert built.capacities[0].serve_rate == 4.0


def test_a_declared_serve_rate_is_a_variable_and_not_a_constant():
    """And it is PUBLISHED, one per held flow, under muscadet's own name
    for it: a ceiling folded into the capability leaves a failure mode
    written to throttle the discharge with nothing to clamp.

    What a mode does with it is `test_capacity_serve_cond`'s; what is
    pinned here is that the attribute exists and starts where the
    declaration put it."""
    built = declare.build_component(mu.System("read"), capacity_spec(serve_rate=4.0))
    component = built._build(set(), set())
    ceilings = {
        entry["name"]: entry["init"]["value"]
        for entry in component["attributes"]
        if "serve_rate" in entry["name"]
    }
    assert ceilings == {"tank_serve_rate_q": 4.0}


# --- 6. and the declaration is checked ----------------------------------


def test_a_through_path_that_is_not_a_predicate_is_refused():
    """It says WHETHER, not how much: the rate is `serve_rate`, and a
    number here would be a quantity read as a flag."""
    with pytest.raises(ValueError) as raised:
        tank_class(transmits=1.0)("CAP").add_flows()
    assert "`transmits` as a boolean" in str(raised.value)


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_a_negative_or_undefined_serve_rate_is_refused(bad):
    with pytest.raises(ValueError) as raised:
        tank_class(serve_rate=bad)("CAP").add_flows()
    assert "serve rate that is positive" in str(raised.value)


# --- 7. and the transit is never assumed --------------------------------


def test_a_volume_says_whether_it_transits_rather_than_being_assumed_to():
    """`transmits` defaults to FALSE here, where muscadet's own field
    defaults to True, and the difference is the whole safety of this
    change.

    muscadet's solver never reads the field: it takes the through-path
    from the wiring, so its default moves no model of its own. Here the
    field DECIDES, so a default of True reinterprets every volume ever
    written against this engine. Measured on the cross-validation corpus,
    it moved five documents, 172 of the 254 series of the plant among
    them, none of which any recorded reference compares.

    So a volume silent on the key is the store it has always been, and
    the same volume declaring the key is the buffer. Same geometry, same
    numbers, one word of difference."""
    silent = tank_system(transmits=False, fill_rate=0.0)
    speaking = tank_system(transmits=True, fill_rate=0.0)
    quiet = silent.simulate(t_max=5.0, samples=[1.0])
    heard = speaking.simulate(t_max=5.0, samples=[1.0])

    # Silent: the out-flow rate of zero is a ceiling, so nothing crosses,
    # and the inlet asks only for what it declares -- which is nothing.
    assert abs(sampled(quiet, "SINK_q_fed_in", 1.0)) < TOL
    assert abs(sampled(quiet, "CAP_q_fed_in", 1.0)) < TOL
    # Speaking: the volume is the source, and it draws what it passes on.
    assert abs(sampled(heard, "SINK_q_fed_in", 1.0) - DEMAND) < TOL
    assert abs(sampled(heard, "CAP_q_fed_in", 1.0) - DEMAND) < TOL


def test_a_volume_with_no_way_in_is_not_a_buffer_however_it_is_declared():
    """A through-path needs two ports, so a volume the component gives no
    inlet cannot transit whatever the key says.

    Three documents of the cross-validation corpus hold one -- the
    `and_method` fixture and the two median-sensor ones -- and reading
    them as buffers published an unbounded capability on an output that
    had published none. The shape is checked beside the key for that
    reason: `transmits` says the path is USED, the flows say it EXISTS,
    and a volume replaces its output's rate only when both agree."""

    class Reservoir(mu.ObjFlow):
        def add_flows(self):
            # No inlet, and no rate: what an unfed source of the corpus
            # looks like.
            self.add_flow_continuous_out(name="q")
            self.add_capacity(
                name="store",
                flow="q",
                capacity=VOLUME,
                content_init={"q": HELD},
                transmits=True,
            )

    system = mu.System("no_way_in")
    system.add_component(Reservoir, "R")
    system.add_component(Sink, "L")
    system.connect("R", "q", "L", "q")
    result = system.simulate(t_max=5.0, samples=[1.0])

    assert abs(sampled(result, "R_q_capability_out", 1.0)) < TOL
    assert abs(sampled(result, "L_q_fed_in", 1.0)) < TOL
