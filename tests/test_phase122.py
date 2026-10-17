"""Tests for Phase 12.2 subgroup analysis."""
import pytest

from qwixx_ai.phase122 import analyze, bins, group_summary


def record(seed, gap, skips=3, color=0):
    return {
        "game_seed": seed, "win_gap": gap, "score_gap": gap * 10,
        "stage": "late",
        "source_move": {"color": color, "skipped_cells": skips, "existing_marks": 2},
        "champion_move": {"color": (color + 1) % 4, "skipped_cells": 0, "existing_marks": 4},
    }


def test_binning():
    assert bins(record(1, 0.1))["skip_3plus"] == "3+"
    assert bins(record(1, 0.1, 2, 1))["source_red_green"] == "yellow_blue"


def test_game_weighting_differs_from_state_weighting():
    rows = [record(1, 1), record(1, 1), record(2, -1)]
    summary = group_summary(rows, 100, 7)
    assert summary["state_weighted_gap"] == pytest.approx(1 / 3)
    assert summary["game_weighted_gap"] == 0


def test_bootstrap_reproducible():
    rows = [record(1, 0.2), record(2, -0.1), record(3, 0.3)]
    assert group_summary(rows, 100, 9) == group_summary(rows, 100, 9)


def test_reject_wrong_phase():
    with pytest.raises(ValueError, match="12.1"):
        analyze({"phase": "11.9", "decisions": []})


def test_analysis_counts():
    result = analyze({"phase": "12.1", "decisions": [record(1, 0.2), record(2, 0.1)]}, 100)
    assert result["strata"]["skip_3plus"]["3+"]["states"] == 2
    assert result["promoted"] is False
