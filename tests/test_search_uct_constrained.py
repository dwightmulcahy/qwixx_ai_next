"""Constrained UCT safeguards and chance sampling."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.search_uct_constrained import guarded_choice, root_estimates, select


def test_guard_blocks_unsupported_switch():
    actions = [None, (0, 2, False)]
    assert guarded_choice(actions, [5, 8], [200, 150], None, 20) is None
    assert guarded_choice(actions, [5, 8], [200, 210], None, 20) == actions[1]


def test_invalid_settings():
    with pytest.raises(ValueError, match="invalid"):
        select(Game(seed=1), depth=-1)


def test_early_skip():
    game = Game(seed=1)
    assert select(game) == heuristic_v2(game)


def test_root_samples_are_reproducible():
    game = Game(seed=2)
    actions = [None, *game.legal()]
    first = root_estimates(game, actions, 2, 1, 1.0, 8, 10)
    second = root_estimates(game, actions, 2, 1, 1.0, 8, 10)
    assert first == second
