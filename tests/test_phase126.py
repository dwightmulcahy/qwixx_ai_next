"""Phase 12.6 controlled policy tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase126 import adjusted_choice, evaluate, run


class FixedModel:
    def __call__(self, observation):
        import torch
        from qwixx_ai.env import ACTION_COUNT
        return torch.zeros(ACTION_COUNT)


def test_adjusted_choice_is_legal():
    game = Game(n=4, seed=42)
    assert adjusted_choice(FixedModel(), game, 0.2) in [None, *game.legal()]


def test_negative_bonus_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        adjusted_choice(FixedModel(), Game(seed=1), -1)


def test_evaluate_one_game():
    result, rows = evaluate(FixedModel(), (0.0,), 1, 100)
    assert len(rows) == 1
    assert result["0.0"]["games"] == 1


def test_run_disjoint_seed_validation():
    with pytest.raises(ValueError, match="overlap"):
        run("unused.pt", tune_games=2, eval_games=2, tune_seed=1, eval_seed=2)


def test_run_bonus_validation():
    with pytest.raises(ValueError, match="bonuses"):
        run("unused.pt", bonuses=(0.1,))
