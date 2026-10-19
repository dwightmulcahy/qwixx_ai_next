"""Phase 11.7 confidence interval and input validation tests."""
import pytest

from qwixx_ai.phase117 import compare, late_game_diagnostics, summarize_blocks


def test_block_ci_constant():
    result = summarize_blocks([0.125, 0.125, 0.125])
    assert result["paired_advantage"] == 0.125
    assert result["advantage_ci_low"] == 0.125
    assert result["advantage_ci_high"] == 0.125


def test_block_ci_spans_zero():
    result = summarize_blocks([-0.1, 0.1])
    assert result["advantage_ci_low"] < 0 < result["advantage_ci_high"]


def test_invalid_block_count():
    with pytest.raises(ValueError):
        summarize_blocks([0.1])


def test_compare_rejects_duplicate_seeds():
    with pytest.raises(ValueError):
        compare(None, None, [1, 1], 2)


def test_diagnostics_rejects_invalid_games():
    with pytest.raises(ValueError):
        late_game_diagnostics(None, None, games=0)
