"""A boolean read through its identity threshold is the boolean itself.

cod3s gives every variable indicator a threshold pair, and a study written on
the platform often spells it: ``operator: "=="``, ``value_test: True`` on a
boolean variable. That is the identity, and it arrives here as a
``predicate`` entry (``eq``, ``true``) under the name the authoring layer
already generates for the plain ``attribute`` observation of that variable,
``{component}_{variable}``.

Measured 2026-09-25 on the internal instance: the study "MC DC P2" declares
``baie_1_air_fed_in`` that way, and the declaration route refused the whole
model on a name collision, while the reference engine ran it. The two entries
are not two observations: on a boolean the engine's value of the attribute
and the truth of ``== true`` are the same series, point for point. The
collision rule compares OBSERVATIONS, so it has to compare them in one
canonical form, and a threshold that is the identity on a boolean reduces to
the attribute. Any other threshold still collides, which the refusal tests
below pin.
"""

import pytest

import pyraichu
from pyraichu.indicators import merge_indicators, observation

ATTR = {"component": "c", "attribute": "up"}


def _predicate(name, cmp, value):
    return {
        "name": name,
        "target": "predicate",
        "attr": ATTR,
        "cmp": cmp,
        "value": {"kind": "bool", "value": value},
    }


def _attribute(name):
    return {"name": name, "target": "attribute", "attr": ATTR}


class Collision(Exception):
    pass


def _merge(held, incoming):
    return merge_indicators(
        [dict(entry) for entry in held],
        [incoming],
        collision=lambda name, wanted, already: Collision(name),
    )


@pytest.mark.parametrize(
    "identity", [("eq", True), ("ne", False)], ids=["eq-true", "ne-false"]
)
@pytest.mark.parametrize("order", ["attribute-first", "predicate-first"])
def test_the_identity_threshold_on_a_boolean_merges_with_the_attribute(
    identity, order
):
    attribute, predicate = _attribute("c_up"), _predicate("c_up", *identity)
    first, second = (
        (attribute, predicate) if order == "attribute-first" else (predicate, attribute)
    )

    merged = _merge([first], second)

    # One indicator, and the one already held: nothing is ever rewritten.
    assert merged == [first]


@pytest.mark.parametrize(
    "negation", [("eq", False), ("ne", True)], ids=["eq-false", "ne-true"]
)
def test_the_negation_is_another_observation_and_still_collides(negation):
    with pytest.raises(Collision):
        _merge([_attribute("c_up")], _predicate("c_up", *negation))


def test_a_numeric_threshold_still_collides_with_its_attribute():
    """The case the collision exists for: a level and a condition on it."""
    threshold = {
        "name": "c_up",
        "target": "predicate",
        "attr": ATTR,
        "cmp": "ge",
        "value": {"kind": "float", "value": 1.0},
    }
    with pytest.raises(Collision):
        _merge([_attribute("c_up")], threshold)


def test_observation_is_unchanged_for_what_it_already_returned():
    """`observation` stays the entry minus its name: the canonical form is
    the comparison's business, not a rewrite of what a caller reads."""
    entry = _predicate("c_up", "eq", True)
    assert observation(entry) == {k: v for k, v in entry.items() if k != "name"}


# --- the engine agrees that the two are one observation --------------------

#: A component that fails and is repaired at random, with `up` following its
#: state through a sensitive function: a boolean that actually moves.
FLICKER = {
    "name": "flicker",
    "components": [
        {
            "name": "c",
            "ports": [],
            "attributes": [
                {"name": "up", "kind": "bool", "init": {"kind": "bool", "value": True}}
            ],
            "automata": [
                {
                    "name": "aut",
                    "states": ["ok", "nok"],
                    "init": "ok",
                    "transitions": [
                        {"name": "fail", "source": "ok", "targets": ["nok"], "distrib": "exp", "rate": 0.3},
                        {"name": "fix", "source": "nok", "targets": ["ok"], "distrib": "exp", "rate": 0.5},
                    ],
                }
            ],
            "sensitive_functions": [
                {
                    "name": "follow",
                    "effects": [
                        {
                            "target": ATTR,
                            "value": {
                                "op": "state_active",
                                "state": {"component": "c", "automaton": "aut", "state": "ok"},
                            },
                        }
                    ],
                }
            ],
        }
    ],
    "indicators": [
        _attribute("as_attribute"),
        _predicate("as_eq_true", "eq", True),
        _predicate("as_ne_false", "ne", False),
    ],
}

MEASURES = (
    "mean",
    "std",
    "sojourn_mean",
    "sojourn_std",
    "nb_occurrences_mean",
    "nb_occurrences_std",
    "reached_mean",
    "reached_std",
)


def test_the_engine_estimates_the_identity_threshold_as_the_attribute():
    """Why the merge may keep either entry: every measure a launcher reads
    comes back identical, to the bit, on the same replicas."""
    estimates = pyraichu.monte_carlo(
        pyraichu.load_model(FLICKER),
        nb_runs=400,
        t_max=20.0,
        samples=[0.0, 2.5, 5.0, 10.0, 20.0],
        seed=7,
    ).indicators
    reference = estimates["as_attribute"]
    # The model moves: a comparison of two constant series proves nothing.
    assert 0.0 < reference.mean[-1] < 1.0
    for name in ("as_eq_true", "as_ne_false"):
        for measure in MEASURES:
            assert getattr(estimates[name], measure) == getattr(reference, measure), (
                name,
                measure,
            )
