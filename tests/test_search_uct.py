"""Stochastic UCT invariants."""
import random

import pytest

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.search_uct import Node, choose_uct, observed_key, select, simulate


def test_key_does_not_include_rng():
    game = Game(seed=5)
    other = game.clone()
    other.rng = random.Random(123456)
    assert observed_key(game) == observed_key(other)


def test_root_simulation_updates_visits():
    game = Game(seed=7)
    tree = {}
    simulate(game, game.actor, tree, random.Random(42), 1.0, 100)
    root = tree[observed_key(game)]
    assert root.visits == 1
    assert sum(root.counts.values()) == 1


def test_uct_tries_unvisited():
    node = Node((None, (0, 2, False)))
    node.visits = 1
    node.counts[None] = 1
    node.rewards[None] = 0.0
    assert choose_uct(node, 1.0, random.Random(1)) == (0, 2, False)


def test_invalid_settings():
    with pytest.raises(ValueError, match="invalid"):
        select(Game(seed=1), simulations=0)


def test_early_position_skips_search():
    game = Game(seed=1)
    assert select(game) == heuristic_v2(game)
