"""Regression tests for the experimental strategy baseline and tournament."""
from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2, probability
from qwixx_ai.strategy_tournament import evaluate_policies


def test_two_dice_probability():
    assert probability(7) == 6 / 36
    assert probability(2) == 1 / 36
    assert probability(12) == 1 / 36
    assert probability(13) == 0


def test_heuristic_is_legal_and_deterministic():
    game = Game(seed=123)
    for _ in range(60):
        if game.done:
            break
        action = heuristic_v2(game)
        assert action is None or action in game.legal()
        assert heuristic_v2(game) == action
        game.step(action)


def test_tournament_is_repeatable():
    def pass_policy(game):
        return None

    result = evaluate_policies([("a", pass_policy), ("b", pass_policy)], rounds=2, seed=12)
    assert result == evaluate_policies([("a", pass_policy), ("b", pass_policy)], rounds=2, seed=12)
    assert result["results"]["a"]["games"] == 8
