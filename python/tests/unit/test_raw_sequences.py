"""A sequence campaign kept whole: the raw corpus in the `raichu.sequences`
format, and the two reduced levels.

The engine records every trajectory's sequence and reduces them to the
minimal sequences a study reads. `run_sequences` keeps what the reduction
starts from, written straight from the engine as JSON Lines, so a campaign
can be audited, re-reduced or filtered without being run again. These tests
pin the Python surface; the Rust suite pins the format itself.
"""

import json

import pyraichu

#: The redundant pair of the sequence-analysis guide: one common-cause mode
#: over two targets, and a feared event when both are down.
MODEL = {
    "name": "redundant_pair",
    "plugins": {
        "muscadet": {
            "objects": [
                {
                    "type": "ObjFM",
                    "name": "fm",
                    "targets": ["A", "B"],
                    "failure": [{"law": "exp", "rate": 0.1}, {"law": "exp", "rate": 0.03}],
                    "repair": [{"law": "exp", "rate": 0.5}, {"law": "exp", "rate": 0.5}],
                    "failure_effects": {"flow": False},
                },
                {
                    "type": "ObjEvent",
                    "name": "system_down",
                    "target": True,
                    "cond": [
                        [
                            {"obj": "A", "attr": "flow", "ope": "==", "value": False},
                            {"obj": "B", "attr": "flow", "ope": "==", "value": False},
                        ]
                    ],
                },
            ]
        }
    },
    "components": [
        {"name": n, "attributes": [{"name": "flow", "kind": "bool", "init": {"kind": "bool", "value": True}}]}
        for n in ("A", "B")
    ],
}

NB_RUNS = 400
T_MAX = 100.0


def _campaign(tmp_path, seed=42):
    return pyraichu.run_sequences(
        pyraichu.load_model(MODEL), NB_RUNS, T_MAX, seed=seed, raw_path=tmp_path / "raw.jsonl"
    )


def test_the_minimal_level_is_what_analyse_sequences_returns(tmp_path):
    campaign = _campaign(tmp_path)
    assert campaign.minimal == pyraichu.analyse_sequences(pyraichu.load_model(MODEL), NB_RUNS, T_MAX, seed=42)


def test_the_raw_corpus_holds_a_header_and_one_line_per_replica(tmp_path):
    campaign = _campaign(tmp_path)
    lines = campaign.raw_path.read_text().splitlines()
    assert len(lines) == NB_RUNS + 1
    header = json.loads(lines[0])
    assert header["format"] == "raichu.sequences"
    assert header["version"] == 1
    assert header["engine_version"] == pyraichu.__version__
    assert header["model"] == "redundant_pair"
    assert (header["seed"], header["nb_runs"], header["t_max"]) == (42, NB_RUNS, T_MAX)
    assert header["targets"] == ["system_down"]
    assert [json.loads(line)["run"] for line in lines[1:]] == list(range(NB_RUNS))


def test_reading_it_back_gives_the_same_levels(tmp_path):
    campaign = _campaign(tmp_path)
    again = pyraichu.analyse_raw_sequences(campaign.raw_path)
    assert (again.cleaned, again.minimal) == (campaign.cleaned, campaign.minimal)
    assert again.header["nb_runs"] == NB_RUNS


def test_the_cleaned_level_accounts_for_every_trajectory_and_holds_more_paths(tmp_path):
    campaign = _campaign(tmp_path)
    assert sum(s["weight"] for s in campaign.cleaned) == NB_RUNS
    reaching = [s for s in campaign.cleaned if s["end_cause"] == "system_down"]
    minimal = [s for s in campaign.minimal if s["end_cause"] == "system_down"]
    assert len(reaching) >= len(minimal) > 0


def test_no_path_no_file(tmp_path):
    campaign = pyraichu.run_sequences(pyraichu.load_model(MODEL), 50, T_MAX, seed=1)
    assert campaign.raw_path is None
    assert list(tmp_path.iterdir()) == []
