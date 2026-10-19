"""Tactical rollout search tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.search_tactical import select, simulate, tactical_rollout


def test_early_rollout_matches_v2():
    game = Game(seed=5)
    assert tactical_rollout(game) == heuristic_v2(game)


def test_simulation_deterministic():
    game = Game(seed=4)
    a = simulate(game, None, 42, heuristic_v2)
    assert a == simulate(game, None, 42, heuristic_v2)


def test_invalid_rollout():
    with pytest.raises(ValueError, match="rollout"):
        select(Game(seed=1), rollout="unknown")


def test_noneligible_does_not_simulate():
    game = Game(seed=2)
    stats = dict(skipped=0)
    assert select(game, stats=stats) == heuristic_v2(game)
    assert stats["skipped"] == 1
