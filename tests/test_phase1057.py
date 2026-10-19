"""Phase 10.5.7 diagnostics tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase1057 import analyze, profile_rollouts


def test_empty_state_set():
    result = analyze([], validation_futures=(16, 32))
    assert result["states"] == 0
    assert result["validation_sensitivity"]["16"]["overrides"] == 0


def test_reject_invalid_audit_futures():
    with pytest.raises(ValueError):
        analyze([], audit_futures=1)


def test_same_state_diagnostics_reproducible():
    game = Game(seed=123)
    action = game.legal()[0]
    states = [(game, None, action, 0)]
    options = dict(seed=456, max_futures=4, initial_futures=2,
                   batch_size=2, validation_futures=(2, 4),
                   audit_futures=4)
    first = analyze(states, **options)
    second = analyze(states, **options)
    assert first["records"][0]["audit_win_gain"] == (
        second["records"][0]["audit_win_gain"])
    assert first["records"][0]["holdout"] == (
        second["records"][0]["holdout"])


def test_profiler_output():
    game = Game(seed=123)
    text = profile_rollouts([(game, None, game.legal()[0], 0)],
                            futures=1, top=5)
    assert "function calls" in text
