"""Sequence-analysis annotation (F3a-A6): the muscadet plugin tags the
transitions the native Rust sequence analyser records: ObjFM occ/rep and
ObjEvent occ/not_occ get `monitored` + a `cycle_group` (the occ/rep partners
share it, for cycle filtering), and an ObjEvent marked `target: true` becomes
a model-level sequence target (feared event).

Pure expansion-structure tests (no simulation): they check the plugin emits
the core-schema metadata the engine (A1) and pipeline (A2) consume.
"""

import pyraichu


def _targets(names):
    return [
        {
            "name": n,
            "attributes": [{"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}],
            "ports": [],
            "interfaces": [],
            "automata": [],
            "sensitive_functions": [],
            "equations": [],
        }
        for n in names
    ]


def _spec():
    return {
        "name": "seq_annot",
        "plugins": {
            "muscadet": {
                "objects": [
                    {
                        "type": "ObjFM",
                        "name": "fm",
                        "targets": ["A", "B"],
                        "failure": [{"law": "exp", "rate": 0.1}, {"law": "exp", "rate": 0.05}],
                        "repair": [{"law": "exp", "rate": 0.2}, {"law": "exp", "rate": 0.2}],
                        "failure_effects": {"flow": False},
                    },
                    {
                        "type": "ObjEvent",
                        "name": "ER",
                        "target": True,
                        "cond": [[{"obj": "A", "attr": "flow", "ope": "==", "value": False}]],
                    },
                ]
            }
        },
        "components": _targets(["A", "B"]),
        "connections": [],
        "indicators": [],
    }


def test_objfm_ccf_occ_rep_are_monitored_cycle_pairs():
    """Every ObjFM combination automaton's occ and rep transitions are
    monitored and share the combination's cycle group (so the parity cycle
    filter can pair them)."""
    expanded = pyraichu.expand_model(_spec())
    fm = next(c for c in expanded["components"] if c["name"] == "fm")
    for aut in fm["automata"]:
        transitions = aut["transitions"]
        assert all(t["monitored"] for t in transitions)
        # occ and rep of one combination share the automaton's cycle group.
        assert {t["cycle_group"] for t in transitions} == {aut["name"]}


def test_objevent_occ_notocc_monitored_and_target_declared():
    """The ObjEvent occ/not_occ transitions are monitored (sharing the event
    as cycle group), and `target: true` adds a model-level sequence target on
    the occ state: the feared event's end cause."""
    expanded = pyraichu.expand_model(_spec())
    er = next(c for c in expanded["components"] if c["name"] == "ER")
    transitions = er["automata"][0]["transitions"]
    assert all(t["monitored"] for t in transitions)
    assert {t["cycle_group"] for t in transitions} == {"ER"}
    assert expanded["targets"] == [
        {"name": "ER", "component": "ER", "automaton": "ev", "state": "occ"}
    ]


def test_untargeted_objevent_declares_no_target():
    """An ObjEvent without `target: true` is monitored but is NOT a
    sequence target (no early stop / end cause)."""
    spec = _spec()
    del spec["plugins"]["muscadet"]["objects"][1]["target"]
    expanded = pyraichu.expand_model(spec)
    assert expanded["targets"] == []
    er = next(c for c in expanded["components"] if c["name"] == "ER")
    assert all(t["monitored"] for t in er["automata"][0]["transitions"])


def test_annotation_is_backward_compatible_loadable():
    """The annotated expansion still loads through the core (the new fields
    are optional/serde-default), so existing models are unaffected."""
    model = pyraichu.load_model(_spec())
    assert model is not None
