"""Phase 12.5 feature extraction and state matching tests."""
import pytest

from qwixx_ai.phase125 import analyze, features, matched_pairs


def sample(seed, group, skips, progress=0, color=0):
    return {
        "game_seed": seed, "stage": "late", "actor": 0,
        "skip_group": group, "win_gap": 0.1, "score_gap": 1.0,
        "state_sha256": str(seed),
        "state": {"rows": [[[], [], [], []]],
                  "legal_actions": [1]},
        "source_move": {"color": color, "number": 2, "lock": False,
                        "existing_marks": progress, "skipped_cells": skips,
                        "remaining_after": 10},
        "champion_move": {"color": 1, "number": 3, "lock": False,
                          "existing_marks": 1, "skipped_cells": 1,
                          "remaining_after": 9},
    }


def test_features():
    result = features(sample(1, "3+", 3))
    assert result["developed_row_abandoned"]
    assert result["champion_immediate_gain"]
    assert result["progress_delta"] == 1


def test_exact_stratum_matching():
    rows = [features(sample(1, "3+", 3)),
            features(sample(2, "0-2", 1)),
            features(sample(3, "0-2", 1, progress=2))]
    pairs = matched_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["large_seed"] == 1
    assert pairs[0]["small_seed"] == 2


def test_reject_wrong_phase():
    with pytest.raises(ValueError, match="12.3"):
        analyze({"phase": "12.4", "decisions": []})


def test_empty_analysis():
    result = analyze({"phase": "12.3", "decisions": []})
    assert result["matched_pairs"] == 0
    assert result["matched_difference"] is None
    assert not result["promoted"]
