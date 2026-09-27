"""Confidence intervals through the Python surface.

The engine-side proof lives in `crates/raichu-montecarlo/tests/confidence.rs`,
where the **coverage** of the interval is measured against the closed
form of an analytic case over two hundred campaigns. What is checked
here is what a study actually touches: that the level is a parameter of
the call and not a constant of the library, that it comes back with the
result, and that the bounds a report prints are the ones the closed form
gives.

The probe is a single `ok → nok` exponential transition, so the exact
firing probability is `P(T ≤ t) = 1 − e^{−λt}` and the estimated
probability at the horizon has a known truth to be bracketed. Driving
its rate to either extreme makes the sample constant, which is the case
an interval must not answer with a point.
"""

import math

import pytest

import pyraichu

RATE = 0.05
HORIZON = 10.0
#: Exact firing probability at `HORIZON`, from the closed form.
TRUTH = 1.0 - math.exp(-RATE * HORIZON)

#: Published two-sided normal deviates, so the closed forms below are
#: independent of the engine's own quantile (itself checked against the
#: AS 241 reference values on the Rust side).
DEVIATE = {
    0.80: 1.2815515655446004,
    0.95: 1.9599639845400545,
    0.99: 2.5758293035489004,
}


@pytest.fixture(scope="module")
def probe() -> pyraichu.Model:
    return pyraichu.load_model(
        {
            "name": "ci_probe",
            "components": [
                {
                    "name": "C",
                    "automata": [
                        {
                            "name": "aut",
                            "states": ["ok", "nok"],
                            "init": "ok",
                            "transitions": [
                                {
                                    "name": "fire",
                                    "source": "ok",
                                    "targets": ["nok"],
                                    "distrib": "exp",
                                    "rate": RATE,
                                }
                            ],
                        }
                    ],
                }
            ],
            "connections": [],
            "indicators": [
                {
                    "name": "fired",
                    "target": "state",
                    "component": "C",
                    "automaton": "aut",
                    "state": "nok",
                }
            ],
        }
    )


def estimate(probe, nb_runs: int = 2000, seed: int = 17, **kwargs):
    estimates = pyraichu.monte_carlo(
        probe, nb_runs=nb_runs, t_max=HORIZON, samples=[HORIZON], seed=seed, **kwargs
    )
    return estimates, estimates.indicators["fired"]


def wilson(p: float, n: int, level: float) -> tuple[float, float]:
    """Wilson score bounds, written out here so the engine is compared
    against the formula and not against itself."""
    z = DEVIATE[level]
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return max(center - half, 0.0), min(center + half, 1.0)


def test_an_indicator_carries_its_interval(probe):
    estimates, indicator = estimate(probe)

    for interval in (indicator.ci, indicator.sojourn_ci, indicator.nb_occurrences_ci):
        assert isinstance(interval, pyraichu.ConfidenceInterval)
        assert len(interval.low) == len(indicator.instants)
        assert len(interval.high) == len(indicator.instants)
    assert indicator.ci.low[0] <= indicator.mean[0] <= indicator.ci.high[0]


def test_the_default_level_is_the_engines_and_not_a_second_copy(probe):
    """A default written twice drifts. `pyraichu.DEFAULT_CONFIDENCE` is
    the engine's own constant, and a run that states nothing uses it."""
    estimates, indicator = estimate(probe)
    assert pyraichu.DEFAULT_CONFIDENCE == 0.95
    assert estimates.confidence == pyraichu.DEFAULT_CONFIDENCE
    assert indicator.ci.level == pyraichu.DEFAULT_CONFIDENCE


def test_the_level_is_a_parameter_of_the_study(probe):
    """Three levels on the same campaign: the bounds move with the level,
    and the result reports which one it answered at."""
    widths = {}
    for level in (0.80, 0.95, 0.99):
        estimates, indicator = estimate(probe, confidence=level)
        assert estimates.confidence == level
        assert indicator.ci.level == level
        widths[level] = indicator.ci.half_width()
    assert widths[0.80] < widths[0.95] < widths[0.99]


def test_the_bounds_are_the_ones_the_closed_form_gives(probe):
    """A state indicator is a probability by declaration, so its interval
    is the Wilson one: checked term by term against the formula, and it
    brackets the exact probability of the analytic case."""
    nb_runs, level = 2000, 0.95
    _, indicator = estimate(probe, nb_runs=nb_runs, confidence=level)

    assert indicator.ci.method == "wilson"
    low, high = wilson(indicator.mean[0], nb_runs, level)
    assert indicator.ci.low[0] == pytest.approx(low, abs=1e-12)
    assert indicator.ci.high[0] == pytest.approx(high, abs=1e-12)
    assert indicator.ci.low[0] <= TRUTH <= indicator.ci.high[0]


def test_the_sojourn_and_the_occurrences_get_a_normal_interval(probe):
    """Neither a cumulated sojourn nor an occurrence count is a
    probability, so neither gets a binomial interval."""
    nb_runs, level = 2000, 0.95
    _, indicator = estimate(probe, nb_runs=nb_runs, confidence=level)
    z = DEVIATE[level]

    for interval, mean, std in (
        (indicator.sojourn_ci, indicator.sojourn_mean, indicator.sojourn_std),
        (
            indicator.nb_occurrences_ci,
            indicator.nb_occurrences_mean,
            indicator.nb_occurrences_std,
        ),
    ):
        assert interval.method == "normal"
        assert not any(interval.constant_sample), "this campaign did disperse"
        half = z * std[0] / math.sqrt(nb_runs)
        assert interval.low[0] == pytest.approx(mean[0] - half, abs=1e-12)
        assert interval.high[0] == pytest.approx(mean[0] + half, abs=1e-12)


def test_more_replicas_narrow_the_interval(probe):
    """The complaint the interval answers: a point estimate cannot tell a
    small campaign from a large one. A hundredfold in replicas divides
    the half-width by ten."""
    _, small = estimate(probe, nb_runs=200, seed=3)
    _, large = estimate(probe, nb_runs=20_000, seed=3)

    assert small.mean[0] == pytest.approx(TRUTH, abs=0.06)
    assert large.mean[0] == pytest.approx(TRUTH, abs=0.01)
    assert small.ci.half_width() / large.ci.half_width() == pytest.approx(10.0, rel=0.1)


def test_an_impossible_level_is_refused(probe):
    for level in (0.0, 1.0, 2.0, -1.0):
        with pytest.raises(pyraichu.SimulationError, match="confidence"):
            estimate(probe, nb_runs=10, confidence=level)


# ---------------------------------------------------------------------
# A constant sample: the interval that must not close on a point
# ---------------------------------------------------------------------

#: Rate small enough that no replica fires by `HORIZON`, and rate large
#: enough that all of them do: at either extreme the sample is constant
#: and every observed dispersion is zero.
IMPOSSIBLE = 1e-12
CERTAIN = 1e6


def flat_probe(rate: float) -> pyraichu.Model:
    """The same single-transition probe, driven to one of the two ends."""
    return pyraichu.load_model(
        {
            "name": "ci_flat",
            "components": [
                {
                    "name": "C",
                    "automata": [
                        {
                            "name": "aut",
                            "states": ["ok", "nok"],
                            "init": "ok",
                            "transitions": [
                                {
                                    "name": "fire",
                                    "source": "ok",
                                    "targets": ["nok"],
                                    "distrib": "exp",
                                    "rate": rate,
                                }
                            ],
                        }
                    ],
                }
            ],
            "indicators": [
                {
                    "name": "fired",
                    "target": "state",
                    "component": "C",
                    "automaton": "aut",
                    "state": "nok",
                }
            ],
        }
    )


def frequency_bound(n: int, level: float) -> float:
    """Upper bound on the frequency of an outcome no draw showed, the
    rule of three in exact form, written out here so the engine is
    compared against the formula and not against itself."""
    z2 = DEVIATE[level] ** 2
    return z2 / (n + z2)


def test_an_event_no_replica_reached_is_not_reported_as_impossible():
    """The three estimators are constant at zero, so the two normal
    half-widths are zero and the intervals used to close on [0, 0]. At
    t = 1 the elapsed time is 1, so the sojourn's own scale is 1 and the
    three closed forms land on the same number."""
    nb_runs, level = 500, 0.95
    estimates = pyraichu.monte_carlo(
        flat_probe(IMPOSSIBLE),
        nb_runs=nb_runs,
        t_max=1.0,
        samples=[1.0],
        seed=1,
        confidence=level,
    )
    indicator = estimates.indicators["fired"]
    epsilon = frequency_bound(nb_runs, level)

    assert indicator.mean[0] == 0.0
    assert indicator.std[0] == 0.0
    assert indicator.nb_occurrences_std[0] == 0.0
    for interval in (indicator.ci, indicator.sojourn_ci, indicator.nb_occurrences_ci):
        assert interval.constant_sample == [True]
        assert interval.low[0] == 0.0
        assert interval.high[0] == pytest.approx(epsilon, abs=1e-12)
        assert interval.high[0] > 0.0
    # The construction is unchanged: a draw without dispersion is not a
    # declaration of type.
    assert indicator.ci.method == "wilson"
    assert indicator.sojourn_ci.method == "normal"
    assert indicator.nb_occurrences_ci.method == "normal"


def test_an_event_every_replica_reached_keeps_a_bound_below_its_mean():
    """The case the real campaign showed: `1.0000 +/- 0.0000` on the
    occurrence count, reported as [1.000000, 1.000000]. The bound below
    it is 1 - z**2/(n + z**2), which is Wilson's own closed form at
    p = 1."""
    nb_runs, level = 500, 0.95
    estimates = pyraichu.monte_carlo(
        flat_probe(CERTAIN),
        nb_runs=nb_runs,
        t_max=1.0,
        samples=[1.0],
        seed=1,
        confidence=level,
    )
    indicator = estimates.indicators["fired"]
    epsilon = frequency_bound(nb_runs, level)

    assert indicator.mean[0] == 1.0
    assert indicator.nb_occurrences_mean[0] == 1.0
    assert indicator.nb_occurrences_std[0] == 0.0

    occurrences = indicator.nb_occurrences_ci
    assert occurrences.constant_sample == [True]
    assert occurrences.low[0] == pytest.approx(1.0 - epsilon, abs=1e-12)
    assert occurrences.low[0] < indicator.nb_occurrences_mean[0]
    assert indicator.ci.low[0] == pytest.approx(1.0 - epsilon, abs=1e-12)
    assert indicator.ci.low[0] < indicator.mean[0]


def test_no_interval_closes_on_a_point_above_two_replicas():
    """The property itself, on the two degenerate ends, an ordinary
    campaign, and the smallest campaign that gets an interval at all."""
    samples = [0.5, 1.0, 4.0]
    for label, rate, nb_runs in (
        ("no replica reached it", IMPOSSIBLE, 500),
        ("every replica reached it", CERTAIN, 500),
        ("an ordinary campaign", 0.3, 500),
        ("two replicas", IMPOSSIBLE, 2),
    ):
        estimates = pyraichu.monte_carlo(
            flat_probe(rate), nb_runs=nb_runs, t_max=4.0, samples=samples, seed=5
        )
        indicator = estimates.indicators["fired"]
        for name in ("ci", "sojourn_ci", "nb_occurrences_ci"):
            interval = getattr(indicator, name)
            for k, instant in enumerate(samples):
                assert interval.high[k] > interval.low[k], (
                    f"{label}: {name} closed on the point {interval.low[k]} "
                    f"at t={instant} over {nb_runs} replicas"
                )


def test_the_sojourn_bound_is_the_time_the_campaign_leaves_open():
    """A sojourn is a time, so what it charges a departure is a time:
    what the replicas in epsilon could have spent in the state by t.
    An occurrence count, a pure number, does not scale with the
    horizon."""
    nb_runs, level = 500, 0.95
    samples = [1.0, 10.0, 100.0]
    estimates = pyraichu.monte_carlo(
        flat_probe(IMPOSSIBLE),
        nb_runs=nb_runs,
        t_max=100.0,
        samples=samples,
        seed=7,
        confidence=level,
    )
    indicator = estimates.indicators["fired"]
    epsilon = frequency_bound(nb_runs, level)

    for k, instant in enumerate(samples):
        assert indicator.sojourn_ci.low[k] == 0.0
        assert indicator.sojourn_ci.high[k] == pytest.approx(
            epsilon * instant, rel=1e-12
        )
        assert indicator.nb_occurrences_ci.high[k] == pytest.approx(epsilon, abs=1e-12)
