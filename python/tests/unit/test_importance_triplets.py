"""Basic events given as ``(component, automaton, state)`` triplets.

A basic event's name joins three names with dots, and component and mode
names may carry dots of their own (`RE.XY` is an ordinary label): the
name ``P.A.fm.1.fm.1__occ`` cannot be split back into its three parts. A caller
that already holds the parts passes them as a triplet, a tuple or a
mapping with the wire keys, and nothing is ever split.
"""

from __future__ import annotations

import pytest

import pyraichu

from test_importance_modes import _external, _leaf, _load


def dotted_model():
    """`Φ = A1 ∨ (B ∧ C)`, every component and mode name dotted."""
    objects = [
        _external("fm.1", "P.A", 0.01, "ok"),
        _external("fm.B", "P.B", 0.05, "ok"),
        _external("fm.C", "P.C", 0.08, "ok"),
        {
            "type": "ObjEvent",
            "name": "system_down",
            "target": True,
            "cond": [[_leaf("P.A", "ok")], [_leaf("P.B", "ok"), _leaf("P.C", "ok")]],
        },
    ]
    return _load(objects, {"P.A": ["ok"], "P.B": ["ok"], "P.C": ["ok"]})


def _importance(**kwargs):
    return pyraichu.importance(dotted_model(), nb_runs=2_000, t_max=40.0, instants=[40.0], seed=7, **kwargs)


def test_a_group_of_dotted_triplets_is_measured_as_its_component():
    analysis = _importance(
        groups={
            "pump": [("P.A", "fm.1", "fm.1__occ")],
            "pair": [{"obj": "P.B", "automaton": "fm.B", "attr": "fm.B__occ"}, ("P.C", "fm.C", "fm.C__occ")],
        }
    )
    assert analysis.groups["pump"].birnbaum == analysis.components["P.A"].birnbaum
    assert analysis.groups["pump"].events == ["P.A.fm.1.fm.1__occ"]
    assert analysis.groups["pair"].events == ["P.B.fm.B.fm.B__occ", "P.C.fm.C.fm.C__occ"]


def test_dotted_basic_events_are_selected_by_triplet():
    analysis = _importance(basic_events=[("P.A", "fm.1", "fm.1__occ"), ("P.B", "fm.B", "fm.B__occ"), ("P.C", "fm.C", "fm.C__occ")])
    assert set(analysis.basic_events) == {"P.A.fm.1.fm.1__occ", "P.B.fm.B.fm.B__occ", "P.C.fm.C.fm.C__occ"}


@pytest.mark.parametrize(
    "event",
    [("P.A", "fm.1"), ("P.A", "", "occ"), {"obj": "P.A", "automaton": "fm.1"}, 42],
)
def test_an_incomplete_triplet_is_refused_naming_it(event):
    with pytest.raises(ValueError, match="group `pump`"):
        _importance(groups={"pump": [event]})
