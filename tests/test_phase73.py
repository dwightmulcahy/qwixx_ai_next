"""Phase 7.3 threshold and evaluation regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase72_hybrid import large_jump
from qwixx_ai.phase73_validate import evaluate, selective


def test_jump_threshold_monotonic():
    game = Game(seed=51)
    for action in game.legal():
        if action is not None:
            assert not large_jump(game, action, 5) or large_jump(game, action, 3)
            assert not large_jump(game, action, 3) or large_jump(game, action, 2)


def test_selective_actions_legal():
    model = network(observe(Game()).size)
    game = Game(seed=52)
    for threshold in (2, 3, 4, 5):
        action = selective(model, threshold)(game)
        assert action is None or action in game.legal()


def test_ablation_smoke():
    model = network(observe(Game()).size)
    result = evaluate(model, rounds=2, seed=800, thresholds=(2, 3))
    assert set(result["results"]) == {"2", "3"}
    assert result["comparison_confidence"] == pytest.approx(0.9875)


def test_confirmation_smoke():
    model = network(observe(Game()).size)
    result = evaluate(model, rounds=2, seed=900, confirm=3)
    assert result["phase"] == "confirmation"
    assert list(result["results"]) == ["3"]
    assert result["comparison_confidence"] == pytest.approx(0.975)


def test_invalid_confirmation():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError, match="confirmation threshold"):
        evaluate(model, rounds=2, confirm=7)
