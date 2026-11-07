"""The ``P<N>`` statistics of every measure cross into Python.

A study asks ``P5`` and ``P95`` of an indicator measure. The launcher reads
the order off the statistic (``percentile_order``), asks it of the campaign
(``quantiles=``) and reads the series in the field ``MEASURE_QUANTILES``
names for the measure. The Rust suite pins the rank-error contract; this
pins what the binding hands a launcher.
"""

import dataclasses
import json

import pytest

import pyraichu
from pyraichu.muscadet.engine import MEASURE_QUANTILES, MEASURE_SERIES, percentile_order

REPAIRABLE = {
    "name": "percentiles",
    "components": [
        {
            "name": "c",
            "attributes": [],
            "ports": [],
            "automata": [
                {
                    "name": "aut",
                    "states": ["ok", "nok"],
                    "init": "ok",
                    "transitions": [
                        {"name": "fail", "source": "ok", "targets": ["nok"], "distrib": "exp", "rate": 0.1},
                        {"name": "repair", "source": "nok", "targets": ["ok"], "distrib": "exp", "rate": 0.5},
                    ],
                }
            ],
        }
    ],
    "indicators": [{"name": "down", "target": "state", "component": "c", "automaton": "aut", "state": "nok"}],
}

QUANTILE_FIELDS = (
    "quantiles",
    "sojourn_quantiles",
    "nb_occurrences_quantiles",
    "reached_quantiles",
    "zero_departures_quantiles",
    "nonzero_reached_quantiles",
)


def campaign(nb_runs=1_500, **kwargs):
    return pyraichu.monte_carlo(
        pyraichu.load_model(REPAIRABLE), nb_runs=nb_runs, t_max=40.0, samples=[5.0, 40.0], seed=3, **kwargs
    )


@pytest.mark.parametrize(
    ("stat", "order"),
    [("P0", 0.0), ("P5", 0.05), ("P25", 0.25), ("P50", 0.5), ("P95", 0.95), ("P99", 0.99), ("P100", 1.0)],
)
def test_a_percentile_statistic_names_its_order(stat, order):
    assert percentile_order(stat) == order


@pytest.mark.parametrize("stat", ["mean", "stddev", "min", "max", "P101", "P", "p5", "P5.5", "qle5", " P5"])
def test_any_other_statistic_is_not_a_percentile(stat):
    assert percentile_order(stat) is None


def test_every_study_measure_maps_to_a_quantile_field_of_the_estimate():
    assert set(MEASURE_QUANTILES) == set(MEASURE_SERIES)
    # Same pairing as the mean series: `<measure>_mean` reads `<measure>_quantiles`.
    for measure, (mean_field, _) in MEASURE_SERIES.items():
        expected = "quantiles" if mean_field == "mean" else mean_field.removesuffix("_mean") + "_quantiles"
        assert MEASURE_QUANTILES[measure] == expected, measure
    fields = {f.name for f in dataclasses.fields(pyraichu.IndicatorEstimate)}
    assert set(MEASURE_QUANTILES.values()) <= fields


def test_p5_and_p95_are_served_on_every_measure():
    orders = [percentile_order("P5"), percentile_order("P95")]
    estimates = campaign(quantiles=orders)
    assert estimates.quantile_rank_error == pyraichu.DEFAULT_QUANTILE_RANK_ERROR
    indicator = estimates.indicators["down"]
    for name in QUANTILE_FIELDS:
        series = getattr(indicator, name)
        assert sorted(series) == orders, name
        assert all(len(values) == 2 for values in series.values()), name
        assert all(low <= high for low, high in zip(series[0.05], series[0.95])), name
    # The P5-P95 band of the sojourn time is a band, not a point.
    sojourn = indicator.sojourn_quantiles
    assert sojourn[0.05][1] < sojourn[0.95][1]


def test_below_one_over_the_rank_error_the_percentiles_are_the_exact_ones():
    # 1/ε is 1 000 replicas at the default rank error.
    orders = [0.05, 0.5, 0.95]
    summarised = campaign(nb_runs=999, quantiles=orders).indicators["down"]
    exact = campaign(nb_runs=999, quantiles=orders, quantile_rank_error=0.0)
    assert exact.quantile_rank_error == 0.0
    for name in QUANTILE_FIELDS:
        assert getattr(summarised, name) == getattr(exact.indicators["down"], name), name


@pytest.mark.parametrize("rank_error", [-0.1, 0.6])
def test_a_rank_error_outside_its_range_is_refused_by_name(rank_error):
    with pytest.raises(pyraichu.SimulationError, match="quantile_rank_error"):
        campaign(quantiles=[0.5], quantile_rank_error=rank_error)


def test_a_result_written_before_the_measure_percentiles_reads_back():
    document = json.loads(
        pyraichu.monte_carlo_json(json.dumps(REPAIRABLE), 10, 40.0, [40.0], quantiles=[0.5])
    )
    document.pop("quantile_rank_error")
    for indicator in document["indicators"]:
        for name in QUANTILE_FIELDS[2:]:
            indicator.pop(name)
    read = pyraichu._mc_estimates(document)
    assert read.quantile_rank_error == 0.0
    assert read.indicators["down"].zero_departures_quantiles == {}
    assert read.indicators["down"].quantiles[0.5] == document["indicators"][0]["quantiles"][0]["values"]
