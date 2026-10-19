"""Adaptive search basic invariants."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.search_adaptive import select, evaluate


def test_invalid_settings():
    with pytest.raises(ValueError, match="invalid"):
        select(Game(seed=1), initial=1)


def test_no_legal_marks_returns_pass():
    game = Game(seed=1)
    game.legal = lambda: []
    assert select(game, initial=2, maximum=2) is None


def test_small_search_legal():
    game = Game(seed=2)
    action = select(game, initial=2, maximum=2)
    assert action in [None, *game.legal()]


def test_invalid_rounds():
    with pytest.raises(ValueError, match="rounds"):
        evaluate(rounds=1)
