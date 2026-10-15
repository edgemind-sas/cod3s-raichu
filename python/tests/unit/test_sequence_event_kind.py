"""The reduced sequence levels name each event's automaton and declared kind.

The raw corpus records, for every event, the automaton it belongs to and the
reliability role its transition declares (``"failure"``, ``"repair"``,
``"observation"``, or ``None``). The cleaned and minimal levels a consumer
displays carry the same two fields, so a failure is told from a repair by
what the model declares, never by guessing from a state name. These tests
pin that contract on the Python surface: ``run_sequences``,
``analyse_raw_sequences`` and ``analyse_sequences``.
"""

import json

import pyraichu


def _unit(name, rate):
    """A repairable unit whose states are ``ok`` / ``ko``, not ``occ`` / ``rep``."""
    return {
        "name": name,
        "automata": [
            {
                "name": "life",
                "states": ["ok", "ko"],
                "init": "ok",
                "transitions": [
                    {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": rate,
                     "monitored": True, "cycle_group": "life", "kind": "failure"},
                    {"name": "fix", "source": "ko", "targets": ["ok"], "distrib": "exp", "rate": 1.0,
                     "monitored": True, "cycle_group": "life", "kind": "repair"},
                ],
            }
        ],
    }


def _down(name):
    return {"op": "state_active", "state": {"component": name, "automaton": "life", "state": "ko"}}


#: Two repairable units and a feared event the instant both are down. The
#: feared event's transition declares no kind.
MODEL = {
    "name": "pair",
    "components": [
        _unit("A", 0.3),
        _unit("B", 0.2),
        {
            "name": "ER",
            "automata": [
                {
                    "name": "ev",
                    "states": ["not_occ", "occ"],
                    "init": "not_occ",
                    "transitions": [
                        {"name": "occ", "source": "not_occ", "targets": ["occ"], "distrib": "delay", "time": 0.0,
                         "monitored": True,
                         "guard": {"op": "bool", "bool_op": "and", "args": [_down("A"), _down("B")]}},
                    ],
                }
            ],
        },
    ],
    "targets": [{"name": "both_down", "component": "ER", "automaton": "ev", "state": "occ"}],
}

NB_RUNS = 300
T_MAX = 20.0

#: What every event of the reduced levels holds, and what each event of
#: this model must say.
FIELDS = {"obj", "automaton", "attr", "time", "kind"}
EXPECTED = {
    ("A", "ko"): ("life", "failure"),
    ("B", "ko"): ("life", "failure"),
    ("A", "ok"): ("life", "repair"),
    ("B", "ok"): ("life", "repair"),
    ("ER", "occ"): ("ev", None),
}


def _campaign(tmp_path):
    return pyraichu.run_sequences(pyraichu.load_model(MODEL), NB_RUNS, T_MAX, seed=11, raw_path=tmp_path / "raw.jsonl")


def _events(level):
    return [event for sequence in level for event in sequence["events"]]


def test_every_reduced_event_names_its_automaton_and_declared_kind(tmp_path):
    campaign = _campaign(tmp_path)
    for level in (campaign.cleaned, campaign.minimal):
        events = _events(level)
        assert {e["obj"] for e in events} == {"A", "B", "ER"}
        for event in events:
            assert set(event) == FIELDS
            assert (event["automaton"], event["kind"]) == EXPECTED[(event["obj"], event["attr"])]


def test_the_older_entry_points_return_the_same_fields(tmp_path):
    campaign = _campaign(tmp_path)
    again = pyraichu.analyse_raw_sequences(campaign.raw_path)
    assert (again.cleaned, again.minimal) == (campaign.cleaned, campaign.minimal)
    assert pyraichu.analyse_sequences(pyraichu.load_model(MODEL), NB_RUNS, T_MAX, seed=11) == campaign.minimal


def test_a_corpus_written_without_automata_or_kinds_reads_null_for_both(tmp_path):
    campaign = _campaign(tmp_path)
    lines = campaign.raw_path.read_text().splitlines()
    old = [lines[0]]
    for line in lines[1:]:
        record = json.loads(line)
        record.pop("automata", None)
        record.pop("kinds", None)
        old.append(json.dumps(record))
    old_path = tmp_path / "old.jsonl"
    old_path.write_text("\n".join(old) + "\n")
    reduced = pyraichu.analyse_raw_sequences(old_path)
    assert [s["weight"] for s in reduced.cleaned] == [s["weight"] for s in campaign.cleaned]
    for level in (reduced.cleaned, reduced.minimal):
        for event in _events(level):
            assert set(event) == FIELDS
            assert (event["automaton"], event["kind"]) == (None, None)
