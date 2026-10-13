"""Tests for paired strategy comparison and bounded rollout."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.controlled_strategy import paired_compare
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.rollout_agent import rollout, rollout_details


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


def test_rollout_diagnostics_are_deterministic():
    game = Game(seed=32)
    first = rollout_details(game, simulations=2, horizon=8, seed=444)
    second = rollout_details(game, simulations=2, horizon=8, seed=444)
    assert first == second
    assert first["action"] is None or first["action"] in game.legal()


def test_rollout_can_finish_full_games():
    game = Game(seed=12)
    detail = rollout_details(game, simulations=1, seed=55)
    assert detail["action"] is None or detail["action"] in game.legal()
    assert all(0 <= score <= 1 for score in detail["win_shares"].values())


def test_rollout_validates_opponent_count():
    with pytest.raises(ValueError, match="one policy per seat"):
        rollout(Game(seed=7), opponents=[heuristic_v2])
