"""Phase 10.5.1 dataset replay and independent label checks."""
import json

import pytest

from qwixx_ai.phase105_diagnose import validate


def test_old_dataset_requires_recollection(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "source": "candidate.pt",
        "samples": [{"observation": [0.0]}],
    }))
    with pytest.raises(ValueError, match="Recollect"):
        validate(path, "candidate.pt")


def test_empty_qualified_labels_do_not_evaluate(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({
        "source": "candidate.pt", "game_seed": 1,
        "games_per_seat": 1,
        "samples": [{
            "game_seed": 1, "seat": 1, "decision_index": 0,
            "observation": [0.0], "neural_action": 0,
            "estimates": [{"action": 0, "win_share": 0.3,
                           "mean_score": 40}],
        }],
    }))
    assert validate(path, "candidate.pt")["evaluated"] == 0


def test_rejects_wrong_source(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"source": "a.pt", "samples": []}))
    with pytest.raises(ValueError, match="source"):
        validate(path, "b.pt")
