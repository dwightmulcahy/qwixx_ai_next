"""Clone equivalence and rollout performance regression checks."""
import copy
import random

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.phase9_search import outcome


def snapshot(game):
    return (
        game.n, game.active, game.phase, game.turn, game.active_marks,
        game.dice, game.locks, game.white_locks,
        [(board.rows, board.penalties) for board in game.boards],
        game.rng.getstate(),
    )


def test_clone_matches_deepcopy_across_game_phases():
    for seed in (7, 23, 101):
        game = Game(seed=seed)
        for _ in range(150):
            clone = game.clone()
            reference = copy.deepcopy(game)
            assert snapshot(clone) == snapshot(reference)
            assert clone.legal() == reference.legal()
            assert clone.done == reference.done
            assert clone.scores() == reference.scores()
            if game.done:
                break
            game.step(heuristic_v2(game))


def test_clone_mutations_are_independent():
    original = Game(seed=19)
    cloned = original.clone()
    cloned.boards[0].rows[0].append(2)
    cloned.locks.add(1)
    cloned.white_locks.add(2)
    cloned.rng.random()
    assert original.boards[0].rows[0] == []
    assert original.locks == set()
    assert original.white_locks == set()
    assert cloned.rng.getstate() != original.rng.getstate()


def test_clone_and_deepcopy_same_continuations():
    for seed in (1, 2, 3, 4):
        game = Game(seed=seed)
        for _ in range(25):
            if game.done:
                break
            game.step(heuristic_v2(game))
        if game.done:
            continue
        for action in [None, *game.legal()]:
            clone = game.clone()
            reference = copy.deepcopy(game)
            for trial in (clone, reference):
                trial.rng = random.Random(seed + 100)
                trial.step(action)
                while not trial.done:
                    trial.step(heuristic_v2(trial))
            assert snapshot(clone) == snapshot(reference)


def test_outcome_does_not_mutate_game():
    game = Game(seed=20)
    before = snapshot(game)
    outcome(game, None, 1234)
    assert snapshot(game) == before
