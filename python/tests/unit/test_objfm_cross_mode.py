"""Several failure modes on one target: the composition, and the one
writer per reinitialized attribute that makes it possible.

A reinitialization effect is a **total** assignment: the target attribute
holds the failure value while the mode's gate is true and its rest value
otherwise. Written per mode, that rest branch is a mode asserting what
the attribute is when *it* has not failed it, which is a claim it may
only make while it is the attribute's sole writer. Two modes declared on
the same target each undid the other's failure, so no cut set ever paired
two modes: RAICHU reached the feared event only once ONE mode had failed
every target on its own, while it composed perfectly when the same two
modes declared one target each. That inconsistency with itself is what
made it a defect rather than a convention; a parity campaign running a
whole study on both engines is where it surfaced.

The expansion now merges those writers: one writer per reinitialized
attribute, gated by the OR over every mode impacting it. This suite pins
the invariant on the emitted model, the composition it restores, and the
**analytic** probability of the feared event, which is what says the
restored composition is the right one and not merely a different one.
"""

import itertools
import math

import pytest

import pyraichu
from pyraichu.plugins import expand_model

#: Order-1 rate and horizon of the analytic case: four independent
#: exponential clocks, so every closed form below is elementary.
RATE = 1e-3
T_MAX = 600.0


def _target(name):
    return {
        "name": name,
        "attributes": [
            {"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}
        ],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [],
        "equations": [],
    }


def _mode(name, targets, **extra):
    """One non-repairable ObjFM with an active order-1 law only: the
    orders above it are `null`, which the expansion skips."""
    spec = {
        "type": "ObjFM",
        "name": name,
        "targets": list(targets),
        "failure": [{"law": "exp", "rate": RATE}] + [None] * (len(targets) - 1),
        "repair": [None] * len(targets),
        "failure_effects": {"flow": False},
    }
    spec.update(extra)
    return spec


def _model(objects, targets=("A", "B")):
    return {
        "name": "cross_mode",
        "plugins": {
            "muscadet": {
                "objects": list(objects)
                + [
                    {
                        "type": "ObjEvent",
                        "name": "ER",
                        "target": True,
                        "cond": [
                            [
                                {
                                    "obj": name,
                                    "attr": "flow",
                                    "ope": "==",
                                    "value": False,
                                }
                                for name in targets
                            ]
                        ],
                    }
                ]
            }
        },
        "components": [_target(name) for name in targets],
        "connections": [],
        "indicators": [
            {
                "name": "ER_occ",
                "target": "state",
                "component": "ER",
                "automaton": "ev",
                "state": "occ",
            }
        ],
    }


def _writers(expanded):
    """`(component, attribute)` → the sensitive functions writing it."""
    writers: dict[tuple[str, str], list[str]] = {}
    for component in expanded["components"]:
        for function in component.get("sensitive_functions") or []:
            for effect in function.get("effects") or []:
                key = (effect["target"]["component"], effect["target"]["attribute"])
                writers.setdefault(key, []).append(
                    f"{component['name']}.{function['name']}"
                )
    return writers


def _gated_states(expression):
    """The `(component, automaton, state)` triples an expression reads."""
    if not isinstance(expression, dict):
        return set()
    if expression.get("op") == "state_active":
        state = expression["state"]
        return {(state["component"], state["automaton"], state["state"])}
    found = set()
    for key in ("args", "cond", "then", "otherwise", "lhs", "rhs"):
        value = expression.get(key)
        for item in value if isinstance(value, list) else [value]:
            found |= _gated_states(item)
    return found


def _cut_sets(model, nb_runs=20000, seed=7100):
    """Minimal cut sequences of the feared event, keyed by the ordered
    `(component, attribute)` signature."""
    sequences = pyraichu.analyse_sequences(
        pyraichu.load_model(model), nb_runs=nb_runs, t_max=T_MAX, seed=seed
    )
    return {
        tuple((event["obj"], event["attr"]) for event in sequence["events"]): sequence[
            "weight"
        ]
        for sequence in sequences
        if sequence["end_cause"] == "ER"
    }


# --- the invariant on the emitted model ---------------------------------------


def test_one_writer_per_reinitialized_attribute():
    """Two modes on one target set: each target attribute has exactly ONE
    writer, whose gate is the OR over both modes."""
    expanded = expand_model(
        _model([_mode("mode_a", ["A", "B"]), _mode("mode_b", ["A", "B"])])
    )
    writers = _writers(expanded)
    assert writers[("A", "flow")] == ["mode_a.apply_effects"]
    assert writers[("B", "flow")] == ["mode_a.apply_effects"]

    effects = {
        effect["target"]["component"]: effect["value"]
        for component in expanded["components"]
        if component["name"] == "mode_a"
        for function in component["sensitive_functions"]
        for effect in function["effects"]
    }
    # The rest branch is now reached only when NEITHER mode holds the
    # attribute failed, which is the whole of the fix. Each target keeps
    # its OWN combination: the merge is per attribute, not per mode.
    assert _gated_states(effects["A"]["cond"]) == {
        ("mode_a", "fm__cc_1", "occ__cc_1"),
        ("mode_b", "fm__cc_1", "occ__cc_1"),
    }
    assert _gated_states(effects["B"]["cond"]) == {
        ("mode_a", "fm__cc_2", "occ__cc_2"),
        ("mode_b", "fm__cc_2", "occ__cc_2"),
    }


def test_a_sole_writer_is_left_untouched():
    """A model where each attribute has one mode is emitted exactly as it
    was before the merge existed: the pass is a no-op on it."""
    objects = [_mode("mode_a", ["A"]), _mode("mode_b", ["B"])]
    writers = _writers(expand_model(_model(objects)))
    assert writers[("A", "flow")] == ["mode_a.apply_effects"]
    assert writers[("B", "flow")] == ["mode_b.apply_effects"]


def test_external_modes_sharing_a_target_merge_too():
    """`external` grafts its effects onto the target under its own
    function name, so two of them are two writers of one attribute: the
    same merge applies, on the target this time."""
    expanded = expand_model(
        _model(
            [
                _mode(
                    name,
                    ["A", "B"],
                    behaviour="external",
                    repair={"law": "exp", "rate": 0.5},
                )
                for name in ("mode_a", "mode_b")
            ]
        )
    )
    assert _writers(expanded)[("A", "flow")] == ["A.apply_mode_a"]
    assert _writers(expanded)[("B", "flow")] == ["B.apply_mode_a"]


def test_modes_disagreeing_on_the_rest_state_are_refused():
    """Two modes claiming different values for "nobody has failed this"
    is a contradiction about what the attribute IS, and it is refused
    rather than resolved by emission order."""
    objects = [
        _mode("mode_a", ["A", "B"]),
        _mode("mode_b", ["A", "B"], repair_effects={"flow": False}),
    ]
    with pytest.raises(ValueError, match="disagree on its value"):
        expand_model(_model(objects))


# --- the composition it restores ----------------------------------------------


def test_two_modes_on_one_target_set_compose_across_modes():
    """The feared event is reached by one mode failing one target and the
    OTHER failing the other one: the eight order-dependent cut sets of
    two modes over two targets, not the four of a single mode."""
    cuts = _cut_sets(_model([_mode("mode_a", ["A", "B"]), _mode("mode_b", ["A", "B"])]))
    expected = {
        ((first_mode, f"occ__cc_{first_target}"),
         (second_mode, f"occ__cc_{second_target}"),
         ("ER", "occ"))
        for first_mode, second_mode in itertools.product(("mode_a", "mode_b"), repeat=2)
        for first_target, second_target in ((1, 2), (2, 1))
    }
    assert set(cuts) == expected
    assert len([signature for signature in cuts
                if len({obj for obj, _ in signature if obj != "ER"}) >= 2]) == 4

    # Each of the eight carries real mass; four of them carried none at
    # all before the merge. Not each one's exact share: a trajectory
    # firing `a1`, `b1`, `a2` carries two of the eight as subsequences,
    # so the greedy minimal reduction attributes it to whichever it meets
    # first, and the split moves with the seed on both engines. The
    # probability that is NOT an attribution is the next test's.
    reached = sum(cuts.values())
    assert min(cuts.values()) > 0.05 * reached, cuts


def test_the_feared_event_reaches_its_analytic_probability():
    """Four independent order-1 clocks, no repair: the target set is lost
    when both targets are, so `P = (1 - exp(-2 λ t))²`.

    This is the statement the cut-set structure alone does not make. The
    defect's own probability was 0.298 against this 0.488, which is the
    understatement an analyst would have read as the study's result.
    """
    model = pyraichu.load_model(
        _model([_mode("mode_a", ["A", "B"]), _mode("mode_b", ["A", "B"])])
    )
    nb_runs = 200000
    result = pyraichu.monte_carlo(
        model, nb_runs=nb_runs, t_max=T_MAX, seed=11, samples=[T_MAX]
    )
    measured = result.indicators["ER_occ"].mean[0]
    expected = (1.0 - math.exp(-2.0 * RATE * T_MAX)) ** 2
    # Three Monte-Carlo standard errors of a proportion: seed-fixed, so
    # the decision is reproducible.
    tolerance = 3.0 * math.sqrt(expected * (1.0 - expected) / nb_runs)
    assert abs(measured - expected) < tolerance, (
        f"P(feared event) = {measured:.5f}, analytic {expected:.5f} "
        f"(± {tolerance:.5f})"
    )


def test_the_merge_survives_the_continuous_rebuild():
    """An `external` mode's writer lives on the TARGET, and a target
    carrying a continuous construct is REBUILT from its declaration once
    the whole model is known (`finalize_model`), the graft being carried
    across by difference. The merge runs before that rebuild, so what has
    to be carried is the merged writer and not the two it replaced.
    """

    def mode(name):
        return {
            "type": "ObjFM",
            "name": name,
            "targets": ["B1"],
            "behaviour": "external",
            "failure": {"law": "exp", "rate": RATE},
            "repair": {"law": "exp", "rate": 0.5},
            "failure_effects": {"Elec_fed_available_out": False},
        }

    expanded = expand_model(
        {
            "name": "graft_continuous",
            "plugins": {
                "muscadet": {
                    "objects": [
                        {
                            "type": "ObjFlow",
                            "name": "SourceElec",
                            "flows_continuous_out": [
                                {"name": "Elec", "var_fed_default": 10.0}
                            ],
                        },
                        {
                            "type": "ObjFlow",
                            "name": "B1",
                            "flows_continuous_in": [
                                {"name": "Elec", "var_demand_default": 3.0}
                            ],
                            "flows_continuous_out": [
                                {"name": "Elec", "var_fed_default": 5.0}
                            ],
                        },
                        mode("mode_a"),
                        mode("mode_b"),
                    ]
                }
            },
            "components": [],
            "connections": [
                {
                    "from": {"component": "SourceElec", "port": "Elec_out"},
                    "to": {"component": "B1", "port": "Elec_in"},
                }
            ],
            "indicators": [],
        }
    )
    target = next(c for c in expanded["components"] if c["name"] == "B1")
    # Both mirror automata survive; one writer, gated by both.
    assert {a["name"] for a in target["automata"]} >= {"mode_a", "mode_b"}
    assert _writers(expanded)[("B1", "Elec_fed_available_out")] == ["B1.apply_mode_a"]
    written = next(
        effect
        for function in target["sensitive_functions"]
        if function["name"] == "apply_mode_a"
        for effect in function["effects"]
    )
    assert _gated_states(written["value"]["cond"]) == {
        ("B1", "mode_a", "occ"),
        ("B1", "mode_b", "occ"),
    }
