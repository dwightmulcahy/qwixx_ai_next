"""Tests for paired strategy comparison and bounded rollout."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.controlled_strategy import paired_compare
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.rollout_agent import rollout


def test_paired_comparison_repeatable():
    result = paired_compare(heuristic_v2, heuristic_v2,
                            [heuristic_v2] * 3, rounds=2, seed=19)
    assert result["paired_advantage"] == 0
    assert result["games_per_policy"] == 8
    assert result["advantage_ci_low"] == 0
    assert result["advantage_ci_high"] == 0


def test_paired_comparison_rejects_bad_lineup():
    with pytest.raises(ValueError, match="exactly three"):
        paired_compare(heuristic_v2, heuristic_v2, [], rounds=2)


def test_rollout_returns_legal_move_without_mutating_game():
    game = Game(seed=21)
    before = (game.turn, game.phase, game.dice, game.scores())
    action = rollout(game, simulations=2, horizon=8)
    assert action is None or action in game.legal()
    assert before == (game.turn, game.phase, game.dice, game.scores())


def test_rollout_validates_budget():
    with pytest.raises(ValueError, match="positive"):
        rollout(Game(seed=2), simulations=0)
