"""Position-value evaluator correctness tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.position_value import choose, position_value, row_future_value


def test_locked_row_has_no_future_value():
    assert row_future_value([2, 3], 0, 12, {0}) == 0


def test_future_value_nonnegative():
    assert row_future_value([], 0, 10, set()) >= 0


def test_position_value_finite():
    game = Game(seed=7)
    assert isinstance(position_value(game, game.actor), float)


def test_choice_is_legal():
    game = Game(seed=2)
    assert choose(game) in [None, *game.legal()]


def test_invalid_weights():
    with pytest.raises(ValueError, match="nonnegative"):
        choose(Game(seed=2), future_weight=-1)


def test_fallback_for_no_moves():
    game = Game(seed=2)
    for color in range(4):
        game.locks.add(color)
    assert choose(game) == heuristic_v2(game)
