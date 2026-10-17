"""The default guards of a muscadet external mode are no approximation.

An external mode locks its draw on its target's mirror being at rest and
its repair on the mirror being failed. The mirror follows the mode through
two transitions no draw governs, so with one mode on the target each lock
holds whenever the transition can fire: the tree is exact about it, and
says nothing. Reported before on every mode of every external model, these
two warnings buried the rare one that qualifies the number (COD3S #352).

What is pinned:

1. a block diagram of external modes, under the unavailability, warns of
   neither `guard_reads_state` nor `guarded_repair_assumed_available`, and
   its tree is the closed form;
2. a repair guarded by a real condition (another component being up) is
   still said to be guarded;
3. a lock that another mode on the same target can close (a common-cause
   combination of order two) still conditions the draw.
"""

from __future__ import annotations

import math

import pyraichu
import pytest

RATE, MU, MISSION = 1e-4, 0.1, 8760.0
STRUCTURAL = {"guard_reads_state", "guarded_repair_assumed_available"}


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


def _mode(name, targets, orders=1, **extra):
    unused = [None] * (len(targets) - orders)
    spec = {
        "type": "ObjFM",
        "name": name,
        "targets": list(targets),
        "behaviour": "external",
        "failure": [{"law": "exp", "rate": RATE}] * orders + unused,
        "repair": [{"law": "exp", "rate": MU}] * orders + unused,
        "failure_effects": {"flow": False},
    }
    spec.update(extra)
    return spec


def _lost(name):
    return {"obj": name, "attr": "flow", "ope": "==", "value": False}


def _model(modes):
    """`node` in series with the parallel pair `source_A`, `source_B`: the
    shape of the `MC RBD M2` study the defect was observed on."""
    return {
        "name": "rbd",
        "plugins": {
            "muscadet": {
                "objects": list(modes)
                + [
                    {
                        "type": "ObjEvent",
                        "name": "ER",
                        "target": True,
                        "cond": [[_lost("node")], [_lost("source_A"), _lost("source_B")]],
                    }
                ]
            }
        },
        "components": [_target(n) for n in ("node", "source_A", "source_B")],
        "connections": [],
        "indicators": [],
    }


def _warnings(modes):
    model = pyraichu.load_model(_model(modes))
    tree = pyraichu.fault_tree(model, targets=["ER"], measure="unavailability")
    envelope = tree.envelope([MISSION])
    return tree, envelope, envelope["generation"]["warnings"]


def unavailable(t=MISSION):
    return RATE / (RATE + MU) * (1.0 - math.exp(-(RATE + MU) * t))


def test_the_default_locks_of_external_modes_are_not_warned():
    tree, envelope, warnings = _warnings(
        [_mode("node_lost", ["node"]), _mode("source_lost", ["source_A", "source_B"])]
    )
    assert [w for w in warnings if w["kind"] in STRUCTURAL] == []
    # What remains is the common-cause reading, one per combination.
    assert sorted(w["kind"] for w in warnings) == ["common_cause_repair_approximation"] * 2
    assert all(not r["guarded"] for e in tree.basic_events for r in e["repairs"])
    q = unavailable()
    exact = q + q * q - q * q * q
    assert envelope["horizon"]["probability"] == pytest.approx(exact, rel=1e-12)


def test_a_repair_guarded_by_a_real_condition_is_still_guarded():
    """The node is repaired only while `source_A` is up (a crew it needs)."""
    _, _, warnings = _warnings(
        [
            _mode(
                "node_lost",
                ["node"],
                repair_cond=[[{"obj": "source_A", "attr": "flow", "value": True}]],
            ),
            _mode("source_lost", ["source_A", "source_B"]),
        ]
    )
    guarded = [w for w in warnings if w["kind"] == "guarded_repair_assumed_available"]
    assert [(w["event"], w["transition"]) for w in guarded] == [
        ("node_lost.fm.failure", "node_lost.fm.repair")
    ]


def test_a_lock_another_combination_can_close_still_conditions_the_draw():
    """Order two: `source_A` failed by its own combination blocks the
    combination failing both, so that lock is a real condition."""
    _, _, warnings = _warnings(
        [_mode("node_lost", ["node"]), _mode("source_lost", ["source_A", "source_B"], orders=2)]
    )
    reads = {w["transition"] for w in warnings if w["kind"] == "guard_reads_state"}
    assert "source_lost.fm__cc_1_2.occ__cc_1_2" in reads
    # The node's mode is still alone on its target.
    assert not any(t.startswith(("node_lost.", "node.")) for t in reads)
