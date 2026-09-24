"""A declared threshold reaches the engine, or is refused by its name.

An indicator of a study may carry a THRESHOLD -- cod3s's ``operator`` /
``value_test`` pair -- and then it does not observe its attribute: it
observes the truth of ``attr <operator> value_test``. The difference is
one of quantity, not of spelling, and it shows on the ``sojourn-time``
measure: the sojourn of a raw attribute is the time-integral of its value
(an area), the sojourn of a threshold is the time the condition held (a
duration, bounded by the horizon).

Until the pair was read, the seam dropped it without a word. The study
still printed ``operator`` and ``value_test`` in the columns of its own
result file, so the row announced a predicate that the number on the same
row did not apply.

**The measured case, and the one the first test reproduces.** The H2
showcase, 60 h, one replica, one launch package handed to both engines:
``hydrogen_available`` (``T_H2_LP.lp_tank_qty_H2``, ``>``, ``0.0``,
``sojourn-time``) came back **60.00 h** from the reference engine and
**1199.10 h** from RAICHU. 1199.10 = 19.985 x 60, the tank's content
integrated over the horizon: twenty times the run itself, which is the
thing that makes the defect recognisable without an oracle.

Every figure below is a **closed form** of a deterministic trajectory --
a constant held over a horizon, or four two-hour outages at declared
dates -- and never a recorded run: a recording would pass just as happily
on the wrong quantity, which is what it did for as long as the threshold
was dropped.
"""

from __future__ import annotations

import json

import pytest

import pyraichu
import pyraichu.muscadet_engine as engine
from pyraichu.declare import SystemSpecError
from test_declare_capacity import a_tank
from test_muscadet_engine import FLOW, an_event, rbd_declaration

# --- the running example, at unit scale --------------------------------
#
# A tank holding hydrogen and receiving none: its content is the constant
# the showcase's own tank held, so the area and the duration are both a
# product of two numbers written here.

#: What the showcase's low-pressure tank holds, all run long.
CONTENT = 19.985

#: The showcase's horizon.
HORIZON = 60.0


def h2_shaped(indicators):
    """The showcase's shape, reduced to the one component that carries the
    observation: a volume over one continuous constituent, fed by nothing,
    so the level is a declared constant and the trajectory has no event."""
    tank = a_tank(name="T_H2_LP", capacity="lp_tank", flows=("H2",))
    tank["capacities"][0]["content_init"] = {"H2": CONTENT}
    return {
        "version": "1.0.0",
        "name": "h2_shaped",
        # The declared indicators and nothing else: the generated set
        # would observe every variable of the tank, which says nothing
        # more here and makes the assertions read off an index.
        "generated_indicators": False,
        "components": {"T_H2_LP": tank},
        "connections": [],
        "indicators": list(indicators),
    }


def attr_indicator(name, measure, **extra):
    entry = {
        "name": name,
        "kind": "PycAttrIndicator",
        "attr_type": "VAR",
        "component": "T_H2_LP",
        "attr_name": "lp_tank_qty_H2",
        "measure": measure,
        "stats": ["mean"],
    }
    entry.update(extra)
    return entry


def run(spec, schedule):
    return engine.simulate(
        spec, {"nb_runs": 1, "schedule": list(schedule), "seed": 42}
    )


def body_of(spec):
    return pyraichu.model_body(json.loads(engine.build_model(spec).json))


# --- 1. the threshold reaches the model --------------------------------


def test_a_declared_threshold_becomes_a_predicate_target():
    """What the seam used to write, and what it writes now. The attribute
    is still translated to the spelling this layer holds a volume under
    (`_qty` there, `_content` here): a threshold adds to that reading, it
    does not replace it."""
    spec = h2_shaped(
        [attr_indicator("hydrogen_available", "sojourn-time", operator=">", value_test=0.0)]
    )
    assert body_of(spec)["indicators"] == [
        {
            "name": "hydrogen_available",
            "target": "predicate",
            "attr": {"component": "T_H2_LP", "attribute": "lp_tank_content_H2"},
            "cmp": "gt",
            "value": {"kind": "float", "value": 0.0},
        }
    ]


def test_an_indicator_with_no_tested_value_stays_an_observation_of_the_value():
    """`value_test` decides, never `operator`. cod3s defaults `operator` to
    `"=="` and writes it on every indicator whether or not a predicate was
    asked for; its own registration branches on `value_test is None`
    (`PycAttrIndicator.create_bkd`). Reading the presence of `operator`
    would turn every plain observation of the corpus into a threshold on
    `== None`, which is the opposite mistake and just as silent."""
    spec = h2_shaped([attr_indicator("hydrogen_level", "value", operator="==")])
    assert body_of(spec)["indicators"] == [
        {
            "name": "hydrogen_level",
            "target": "attribute",
            "attr": {"component": "T_H2_LP", "attribute": "lp_tank_content_H2"},
        }
    ]


# --- 2. the number the study reads -------------------------------------


def test_the_showcase_figure_is_the_horizon_and_not_twenty_times_it():
    """The measurement of 2026-09-19, reproduced and inverted.

    Both indicators observe the same tank; the closed forms are the
    horizon (the condition held throughout) and CONTENT x horizon (the
    content integrated over it). The second is the number that was
    reported under the first's question."""
    spec = h2_shaped(
        [
            attr_indicator(
                "hydrogen_available", "sojourn-time", operator=">", value_test=0.0
            ),
            attr_indicator("hydrogen_level", "value"),
        ]
    )
    estimates = run(spec, [0, 5, 10, HORIZON])
    available = estimates.indicators["hydrogen_available"]
    level = estimates.indicators["hydrogen_level"]

    assert available.sojourn_mean[-1] == pytest.approx(HORIZON)
    assert level.sojourn_mean[-1] == pytest.approx(CONTENT * HORIZON)
    # 1199.10 h against a 60 h horizon: the impossibility that makes the
    # defect recognisable with no reference engine to compare against.
    assert level.sojourn_mean[-1] > HORIZON


def test_a_threshold_sojourn_never_outlasts_the_run():
    spec = h2_shaped(
        [attr_indicator("hydrogen_available", "sojourn-time", operator=">", value_test=0.0)]
    )
    estimates = run(spec, [0, 5, 10, 20, 40, HORIZON])
    available = estimates.indicators["hydrogen_available"]
    for instant, sojourn in zip(available.instants, available.sojourn_mean):
        assert sojourn <= instant + 1e-9


def test_the_threshold_that_never_holds_accumulates_nothing():
    """The control on the one above: same attribute, same measure, a
    threshold the constant never satisfies. A dropped threshold would give
    both of them the same number, which is how the defect stayed invisible
    on every study whose condition happened to hold throughout."""
    spec = h2_shaped(
        [
            attr_indicator("above", "sojourn-time", operator=">", value_test=CONTENT),
            attr_indicator("at_least", "sojourn-time", operator=">=", value_test=CONTENT),
        ]
    )
    estimates = run(spec, [0, HORIZON])
    assert estimates.indicators["above"].sojourn_mean[-1] == pytest.approx(0.0)
    assert estimates.indicators["at_least"].sojourn_mean[-1] == pytest.approx(HORIZON)


# --- 3. the negated boolean, on a trajectory that moves ----------------
#
# The showcase's threshold holds all run long, so it proves the quantity
# and not the tracking. The RBD of `test_muscadet_engine` moves: B1 fails
# at 4 and every 6 thereafter, repaired 2 later, and the target is fed
# through an `and` of two blocks. So `T.is_ok_fed_in` is false over
# [4, 6), [10, 12), [16, 18), [22, 24) -- four two-hour outages, and every
# figure below is read off those dates.


def rbd(indicators):
    spec = rbd_declaration()
    spec["generated_indicators"] = False
    spec["indicators"] = list(indicators)
    return spec


def var_indicator(name, measure, **extra):
    entry = {
        "name": name,
        "kind": "PycVarIndicator",
        "component": "T",
        "var": f"{FLOW}_fed_in",
        "measure": measure,
        "stats": ["mean"],
    }
    entry.update(extra)
    return entry


def test_a_negated_boolean_is_measured_on_the_negation_and_not_the_complement():
    """`== False` on a boolean is the observation of its negation, and the
    one a study asking for an UNAVAILABILITY writes. Dropped, it answered
    on the variable itself, so the number that came back was the
    availability: not a small error but the complement, and plausible."""
    estimates = run(
        rbd(
            [
                var_indicator("fed", "sojourn-time"),
                var_indicator("starved", "sojourn-time", operator="==", value_test=False),
            ]
        ),
        [0, 5, 10, 25],
    )
    fed = estimates.indicators["fed"]
    starved = estimates.indicators["starved"]
    # Four outages of 2 h by t = 25, one of them still only 1 h old at
    # t = 5 and one complete at t = 10.
    assert list(starved.sojourn_mean) == pytest.approx([0.0, 1.0, 2.0, 8.0])
    assert list(fed.sojourn_mean) == pytest.approx([0.0, 4.0, 8.0, 17.0])
    # The two partition the run, which is what says the second is the
    # negation of the first rather than another reading of it.
    for instant, up, down in zip(fed.instants, fed.sojourn_mean, starved.sojourn_mean):
        assert up + down == pytest.approx(instant)


def test_a_negated_boolean_counts_the_outages_and_not_the_recoveries():
    estimates = run(
        rbd([var_indicator("starved", "nb-occurrences", operator="==", value_test=False)]),
        [0, 25],
    )
    # Entries into the condition: the four failures, never the repairs.
    assert list(estimates.indicators["starved"].nb_occurrences_mean) == pytest.approx(
        [0.0, 4.0]
    )


def test_the_mean_of_a_negated_boolean_is_the_unavailability():
    estimates = run(
        rbd([var_indicator("starved", "value", operator="==", value_test=False)]),
        [0, 5, 7, 25],
    )
    # Down at 5 (inside [4, 6)), up at 7 and at 25 (outside every outage).
    assert list(estimates.indicators["starved"].mean) == pytest.approx(
        [0.0, 1.0, 0.0, 0.0]
    )


# --- 4. what is refused, and by name -----------------------------------


def test_an_operator_this_engine_has_no_comparison_for_is_refused_by_name():
    """The other half of the contract: a threshold that cannot be honoured
    is refused rather than dropped. Dropped, the indicator answers on the
    value while the result file keeps printing the operator."""
    spec = h2_shaped(
        [attr_indicator("weird", "value", operator="in", value_test=[1, 2])]
    )
    with pytest.raises(SystemSpecError) as raised:
        engine.build_model(spec)
    message = str(raised.value)
    assert "'weird'" in message and "'in'" in message
    # The message lists what IS carried, in the DOCUMENT's own spelling
    # rather than the model's, so a modeller reads the remedy off the
    # refusal instead of off this module.
    assert "'>'" in message and "'=='" in message


def test_a_tested_value_of_a_kind_no_attribute_holds_is_refused_by_name():
    spec = h2_shaped(
        [attr_indicator("weird", "value", operator="==", value_test="empty")]
    )
    with pytest.raises(SystemSpecError) as raised:
        engine.build_model(spec)
    assert "'weird'" in str(raised.value)
    assert "value_test='empty'" in str(raised.value)


def test_a_threshold_comparing_a_volume_to_a_boolean_is_refused_at_build():
    """Read by the seam, refused by the model: the kinds are checked where
    the attribute's declared kind is known, which is not here."""
    spec = h2_shaped(
        [attr_indicator("weird", "value", operator="==", value_test=True)]
    )
    with pytest.raises(pyraichu.ModelError) as raised:
        engine.build_model(spec)
    message = str(raised.value)
    assert "`weird`" in message
    assert "declared Float" in message and "Bool(true)" in message


def test_a_state_indicator_keeps_its_identity_threshold():
    """cod3s gives every `PycSTIndicator` an `operator` / `value_test` pair
    and defaults it to `== True`, which on a state is the state itself. The
    whole corpus falls through untouched, and that is worth a test because
    the alternative -- refusing on the mere presence of the pair -- would
    refuse every state indicator ever written."""
    spec = rbd(
        [
            {
                "name": "DEGRADED_occ",
                "kind": "PycSTIndicator",
                "component": "DEGRADED",
                "state": "occ",
                "measure": "value",
                "stats": ["mean"],
                "operator": "==",
                "value_test": True,
            }
        ]
    )
    spec["components"]["DEGRADED"] = an_event()
    entry = body_of(spec)["indicators"][0]
    assert entry["target"] == "state" and entry["state"] == "occ"


def test_a_state_indicator_read_any_other_way_is_refused_naming_the_other_state():
    """There is no negation of a state target to write `== False` with, and
    an event holds exactly two states, so the refusal names the remedy."""
    spec = rbd(
        [
            {
                "name": "DEGRADED_quiet",
                "kind": "PycSTIndicator",
                "component": "DEGRADED",
                "state": "occ",
                "measure": "value",
                "stats": ["mean"],
                "operator": "==",
                "value_test": False,
            }
        ]
    )
    spec["components"]["DEGRADED"] = an_event()
    with pytest.raises(SystemSpecError) as raised:
        engine.build_model(spec)
    message = str(raised.value)
    assert "'DEGRADED_quiet'" in message
    # The other state of the event, which is what to observe instead.
    assert "instead" in message and "not_occ" in message


# --- 5. one name, two observations -------------------------------------


def test_a_threshold_and_the_generated_observation_of_one_name_are_refused():
    """A name held by a different observation is refused, and a threshold
    IS a different observation of the same variable. Until the pair was
    read the two reconciled as one indicator in silence, and the estimate
    that came back was the one nobody asked for."""
    tank = a_tank(name="T_H2_LP", capacity="lp_tank", flows=("H2",))
    tank["capacities"][0]["content_init"] = {"H2": CONTENT}
    spec = {
        "version": "1.0.0",
        "name": "collide",
        # The generated set emits `{component}_{variable}`, which is the
        # name a modeller writing this declaration by hand reaches for.
        "generated_indicators": True,
        "components": {"T_H2_LP": tank},
        "connections": [],
        "indicators": [
            attr_indicator(
                "T_H2_LP_lp_tank_content_H2",
                "sojourn-time",
                operator=">",
                value_test=0.0,
            )
        ],
    }
    with pytest.raises(SystemSpecError) as raised:
        engine.build_model(spec)
    assert "T_H2_LP_lp_tank_content_H2" in str(raised.value)
