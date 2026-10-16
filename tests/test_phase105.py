"""Phase 10.5 correction safeguards."""
import pytest

from qwixx_ai.phase105_correction import prepare


def test_only_accepts_clear_win_share_and_score_improvement():
    records = [
        {"observation": [0.0], "neural_action": 0,
         "estimates": [
             {"action": 0, "win_share": 0.3, "mean_score": 50},
             {"action": 1, "win_share": 0.45, "mean_score": 51},
             {"action": 2, "win_share": 0.7, "mean_score": 40}]},
        {"observation": [1.0], "neural_action": 0,
         "estimates": [
             {"action": 0, "win_share": 0.3, "mean_score": 50},
             {"action": 1, "win_share": 0.35, "mean_score": 55}]},
    ]
    assert prepare(records, min_gain=0.12) == [([0.0], 1)]


def test_empty_or_weak_evidence_never_creates_labels():
    records = [{"observation": [0.0], "neural_action": 0,
                "estimates": [{"action": 0, "win_share": 0.2,
                               "mean_score": 50}]}]
    assert prepare(records) == []


def test_evaluation_rejects_single_round():
    from qwixx_ai.phase105_evaluate import evaluate

    with pytest.raises(ValueError):
        evaluate("unused.pt", "unused.pt", rounds=1)
