"""raichu-muscadet authoring layer (M3): muscadet ergonomics → RAICHU.

Ports the `rbd_02` example end-to-end through `pyraichu.muscadet`
(the roadmap M3 goal-condition item 3): same declarative authoring
style as `muscadet/examples/rbd_02/rbd_02.py`, RAICHU engine underneath.
The generated model must reproduce the PyCATSHOO-validated trajectory
(cross-checked against the live oracle by `test_oracle_rbd_02`).
"""

import math

import pyraichu.muscadet as mu

# PyCATSHOO-validated rbd_02 event dates (see test_oracle_rbd_02 and
# the M0 implementation notes: simultaneous pair at t=22).
RBD_02_EVENT_DATES = [4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 16.0, 18.0, 19.0, 22.0, 22.0, 24.0]


class Source(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_out(name="is_ok", var_prod_default=True)


class Block(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name="is_ok")
        self.add_flow_out(name="is_ok", var_prod_cond=["is_ok"])


class Target(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_in(name="is_ok", logic="and")


def build_rbd(delay: bool) -> mu.System:
    system = mu.System("rbd_02_via_layer")
    system.add_component(Source, "S")
    system.add_component(Block, "B1")
    system.add_component(Block, "B2")
    system.add_component(Target, "T")
    if delay:
        system.comp["B1"].add_delay_failure_mode(
            name="failure_deterministic",
            failure_time=4,
            repair_time=2,
            failure_cond="is_ok_fed_out",
        )
        system.comp["B2"].add_delay_failure_mode(
            name="failure_deterministic",
            failure_time=8,
            repair_time=3,
            failure_cond="is_ok_fed_out",
        )
    else:
        system.comp["B1"].add_exp_failure_mode(
            name="failure", failure_rate=0.25, repair_rate=0.5
        )
        system.comp["B2"].add_exp_failure_mode(
            name="failure", failure_rate=0.125, repair_rate=1.0 / 3.0
        )
    system.connect("S", "is_ok", "B1", "is_ok")
    system.connect("S", "is_ok", "B2", "is_ok")
    system.connect("B1", "is_ok", "T", "is_ok")
    system.connect("B2", "is_ok", "T", "is_ok")
    return system


def test_layer_reproduces_the_validated_rbd_02_trajectory():
    result = build_rbd(delay=True).simulate(t_max=24.0, confluence_check=True)
    assert [e.time for e in result.events] == RBD_02_EVENT_DATES
    fed = result.indicators["T_is_ok_fed_in"]
    assert fed == [
        (0.0, True),
        (4.0, False),
        (6.0, True),
        (8.0, False),
        (12.0, True),
        (16.0, False),
        (18.0, True),
        (19.0, False),
        (24.0, True),
    ]


def test_auto_connect_matches_explicit_connections():
    explicit = build_rbd(delay=True).simulate(t_max=24.0)
    auto = mu.System("rbd_02_auto")
    auto.add_component(Source, "S")
    auto.add_component(Block, "B1")
    auto.add_component(Block, "B2")
    auto.add_component(Target, "T")
    auto.comp["B1"].add_delay_failure_mode(
        name="failure_deterministic",
        failure_time=4,
        repair_time=2,
        failure_cond="is_ok_fed_out",
    )
    auto.comp["B2"].add_delay_failure_mode(
        name="failure_deterministic",
        failure_time=8,
        repair_time=3,
        failure_cond="is_ok_fed_out",
    )
    auto.auto_connect("S", "B1")
    auto.auto_connect("S", "B2")
    auto.auto_connect("B1", "T")
    auto.auto_connect("B2", "T")
    assert [e.time for e in auto.simulate(t_max=24.0).events] == [
        e.time for e in explicit.events
    ]


def test_layer_monte_carlo_matches_closed_form():
    """Exponential mode through the layer: B1 availability at t follows
    A(t) = μ/(λ+μ) + λ/(λ+μ)·e^{−(λ+μ)t} (single repairable block)."""
    system = build_rbd(delay=False)
    estimates = system.monte_carlo(
        nb_runs=4000, t_max=10.0, samples=[2.0, 5.0, 10.0], seed=9
    )
    est = estimates.indicators["B1_is_ok_fed_out"]
    lam, mu_ = 0.25, 0.5
    for k, t in enumerate(est.instants):
        analytic = mu_ / (lam + mu_) + lam / (lam + mu_) * math.exp(-(lam + mu_) * t)
        se = est.std[k] / math.sqrt(4000)
        assert abs(est.mean[k] - analytic) < 4.0 * max(se, 1e-6), (
            f"A({t}): {est.mean[k]} vs {analytic}"
        )


def test_k_out_of_n_logic():
    """2-out-of-3 target: fed while at least two blocks feed it."""

    class Target2oo3(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="is_ok", logic=2)

    system = mu.System("koon")
    system.add_component(Source, "S")
    for name in ("B1", "B2", "B3"):
        system.add_component(Block, name)
        system.connect("S", "is_ok", name, "is_ok")
        system.connect(name, "is_ok", "T", "is_ok")
    system.add_component(Target2oo3, "T")
    system.comp["B1"].add_delay_failure_mode(
        name="fm", failure_time=5, repair_time=100
    )
    system.comp["B2"].add_delay_failure_mode(
        name="fm", failure_time=9, repair_time=100
    )
    result = system.simulate(t_max=20.0)
    fed = result.indicators["T_is_ok_fed_in"]
    # 3 feeders at t=0 → ok; 2 feeders after t=5 → still ok (k=2);
    # 1 feeder after t=9 → lost.
    assert fed == [(0.0, True), (9.0, False)]
