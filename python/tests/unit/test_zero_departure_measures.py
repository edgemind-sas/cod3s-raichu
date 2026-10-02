"""The zero-departure pair crosses into Python as ``zero_departures_*`` and
``nonzero_reached_*``.

They are the reference engine's ``nb_visits`` and ``realized`` computations
(measured on PyCATSHOO 1.3.8.0, 2026-10-02): a count of the moves from
exactly 0 to any non-zero value, the initial value never counted, and 1 once
the value has been non-zero, the initial value included. The Rust suite pins
them on every measured level sequence; this pins the estimate the binding
hands a launcher, and the muscadet route's choice of series per measure.
"""

import pyraichu
from pyraichu.muscadet.engine import MEASURE_EXTREMES, MEASURE_SERIES

#: The sample instants of the reference measurement.
INSTANTS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0]


def _float(x: float) -> dict:
    return {"op": "const", "value": {"kind": "float", "value": float(x)}}


def stepping_model(levels: list[float]) -> dict:
    """A float attribute ``c.v`` holding ``levels[i]`` on ``[i, i + 1)``, the
    last level held to the horizon: one automaton state per level, unit
    delays between them, and one sensitive function writing the level of the
    active state. Indicator ``v`` observes the attribute."""
    level = _float(levels[-1])
    for i in reversed(range(len(levels) - 1)):
        level = {
            "op": "if",
            "cond": {"op": "state_active", "state": {"component": "c", "automaton": "aut", "state": f"s{i}"}},
            "then": _float(levels[i]),
            "otherwise": level,
        }
    return {
        "name": "stepping",
        "components": [
            {
                "name": "c",
                "attributes": [{"name": "v", "kind": "float", "init": {"kind": "float", "value": float(levels[0])}}],
                "ports": [],
                "automata": [
                    {
                        "name": "aut",
                        "states": [f"s{i}" for i in range(len(levels))],
                        "init": "s0",
                        "transitions": [
                            {"name": f"t{i}", "source": f"s{i}", "targets": [f"s{i + 1}"], "distrib": "delay", "time": 1.0}
                            for i in range(len(levels) - 1)
                        ],
                    }
                ],
                "sensitive_functions": [
                    {"name": "upd", "effects": [{"target": {"component": "c", "attribute": "v"}, "value": level}]}
                ],
            }
        ],
        "indicators": [{"name": "v", "target": "attribute", "attr": {"component": "c", "attribute": "v"}}],
    }


def estimate(levels: list[float]) -> pyraichu.IndicatorEstimate:
    model = pyraichu.load_model(stepping_model(levels))
    return pyraichu.monte_carlo(model, nb_runs=1, t_max=INSTANTS[-1], samples=INSTANTS).indicators["v"]


def test_a_nonzero_initial_value_is_reached_but_is_not_a_departure():
    # Measured on the reference engine: nb_visits [0, 0, 1, 1, 1, 1],
    # realized [1, 1, 1, 1, 1, 1].
    indicator = estimate([1.0, 0.0, -1.0, 1.0, 0.0])
    assert indicator.zero_departures_mean == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert indicator.nonzero_reached_mean == [1.0] * 6
    # RAICHU's own pair is unchanged: the initial value is the first
    # occurrence and -1 -> 1 the second rising edge.
    assert indicator.nb_occurrences_mean == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert indicator.reached_mean == [1.0] * 6
    assert indicator.zero_departures_std == [0.0] * 6
    assert indicator.zero_departures_extremes.max == indicator.zero_departures_mean
    assert indicator.nonzero_reached_extremes.min == [1.0] * 6
    # A proportion: the same Wilson interval as `reached` on the same draws.
    assert indicator.nonzero_reached_ci == indicator.reached_ci


def test_a_departure_to_a_negative_value_counts():
    # Measured: nb_visits [0, 1, 1, 2, 2, 2], realized [0, 1, 1, 1, 1, 1].
    indicator = estimate([0.0, -3.0, 0.0, 2.0, 0.0])
    assert indicator.zero_departures_mean == [0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert indicator.nonzero_reached_mean == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert indicator.nb_occurrences_mean == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_the_muscadet_route_reads_the_reference_computations():
    # cod3s defines `nb-occurrences` as `nb_visits` and `had_value` as
    # `realized`: the route reads the pair that reproduces them.
    assert MEASURE_SERIES["nb-occurrences"] == ("zero_departures_mean", "zero_departures_std")
    assert MEASURE_SERIES["had_value"] == ("nonzero_reached_mean", "nonzero_reached_std")
    assert MEASURE_SERIES["sojourn-time"] == ("sojourn_mean", "sojourn_std")
    assert MEASURE_SERIES["value"] == ("mean", "std")
    assert MEASURE_EXTREMES["nb-occurrences"] == "zero_departures_extremes"
    assert MEASURE_EXTREMES["had_value"] == "nonzero_reached_extremes"
    fields = pyraichu.IndicatorEstimate.__dataclass_fields__
    for pair in MEASURE_SERIES.values():
        assert set(pair) <= set(fields)
    assert set(MEASURE_EXTREMES.values()) <= set(fields)
    assert set(MEASURE_EXTREMES) == set(MEASURE_SERIES)


def test_a_result_document_without_the_pair_still_reads():
    from pyraichu import _mc_estimates

    raw = pyraichu.monte_carlo(
        pyraichu.load_model(stepping_model([0.0, 2.0])), nb_runs=1, t_max=2.0, samples=[1.5]
    )
    document = {
        "indicators": [
            {
                "name": "v",
                "instants": [1.5],
                "mean": raw.indicators["v"].mean,
                "std": [0.0],
                "ci": {"level": 0.95, "method": "undefined", "low": [2.0], "high": [2.0], "constant_sample": [False]},
                "sojourn_mean": [1.0],
                "sojourn_std": [0.0],
                "sojourn_ci": {"level": 0.95, "method": "undefined", "low": [1.0], "high": [1.0], "constant_sample": [False]},
                "nb_occurrences_mean": [1.0],
                "nb_occurrences_std": [0.0],
                "nb_occurrences_ci": {"level": 0.95, "method": "undefined", "low": [1.0], "high": [1.0], "constant_sample": [False]},
                "reached_mean": [1.0],
                "reached_std": [0.0],
                "reached_ci": {"level": 0.95, "method": "undefined", "low": [1.0], "high": [1.0], "constant_sample": [False]},
                "extremes": {"min": [2.0], "max": [2.0]},
                "sojourn_extremes": {"min": [1.0], "max": [1.0]},
                "nb_occurrences_extremes": {"min": [1.0], "max": [1.0]},
                "reached_extremes": {"min": [1.0], "max": [1.0]},
                "quantiles": [],
                "sojourn_quantiles": [],
            }
        ],
        "nb_runs": 1,
        "seed": 0,
        "confidence": 0.95,
        "engine_version": "0.0.0",
    }
    indicator = _mc_estimates(document).indicators["v"]
    assert indicator.zero_departures_mean == []
    assert indicator.nonzero_reached_ci.low == []
    assert indicator.nb_occurrences_mean == [1.0]
