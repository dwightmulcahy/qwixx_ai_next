"""Phase 9 search regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase9_search import evaluate_actions, search, state_seed, validate


def test_state_seed_reproducible():
    assert state_seed(Game(seed=42), 7) == state_seed(Game(seed=42), 7)


def test_action_evaluation_covers_pass_and_legal():
    game = Game(seed=42)
    estimates = evaluate_actions(game, futures=2, seed=7)
    assert len(estimates) == len(game.legal()) + 1
    assert estimates[0]["action"] is None
    assert all(0 <= item["win_share"] <= 1 for item in estimates)


def test_search_returns_legal_action():
    game = Game(seed=43)
    action = search(game, futures=2, seed=7)
    assert action is None or action in game.legal()


def test_search_deterministic():
    game = Game(seed=44)
    assert search(game, futures=2, seed=7) == search(game, futures=2, seed=7)


def test_reject_invalid_futures():
    with pytest.raises(ValueError, match="positive"):
        evaluate_actions(Game(seed=42), futures=0)


def test_validation_smoke():
    result = validate(rounds=2, seed=80, futures=1)
    assert result["against_v2"]["rounds"] == 2
