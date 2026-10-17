"""Tests for Phase 12.4 game-level diagnostics."""
import pytest

from qwixx_ai.phase124 import analyze, bootstrap_paired, describe, paired_games


def row(seed, group, gap, stage="late", actor=1, marks=2, color=0):
    return {
        "game_seed": seed, "skip_group": group, "win_gap": gap,
        "stage": stage, "actor": actor,
        "source_move": {"existing_marks": marks, "color": color},
    }


def test_effective_games_accounts_for_repeated_states():
    result = describe([row(1, "3+", 0.1), row(1, "3+", 0.2), row(2, "0-2", 0.0)])
    assert result["games"] == 2
    assert result["effective_games_kish"] == pytest.approx(9 / 5)


def test_paired_games_only_when_both_groups():
    rows = [row(1, "3+", 0.3), row(1, "0-2", 0.1), row(2, "3+", 0.2)]
    assert paired_games(rows) == [{"game_seed": 1, "difference": pytest.approx(0.2)}]
    assert not bootstrap_paired(paired_games(rows), 100)["estimable"]


def test_paired_bootstrap_reproducible():
    pairs = [{"game_seed": i, "difference": 0.1 * i} for i in range(5)]
    assert bootstrap_paired(pairs, 100, 7) == bootstrap_paired(pairs, 100, 7)


def test_stage_and_progress_strata():
    report = {"phase": "12.3", "decisions": [
        row(1, "3+", 0.2), row(1, "0-2", 0.1),
        row(2, "3+", 0.3, stage="middle", marks=4),
        row(2, "0-2", 0.0, stage="middle", marks=4),
    ]}
    result = analyze(report, 100)
    assert result["strata"]["progress_x_skip"]["3-5/3+"]["states"] == 1
    assert result["group_overlap"]["both_groups"] == 2
    assert result["paired_game_contrast"]["estimable"]
    assert not result["promoted"]


def test_rejects_wrong_phase():
    with pytest.raises(ValueError, match="12.3"):
        analyze({"phase": "12.2", "decisions": []})
