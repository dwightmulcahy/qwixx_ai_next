"""Adaptive search basic invariants."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.search_adaptive import select, evaluate, search_budget, should_search


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


def test_budget_escalates_for_second_lock():
    game=Game(seed=4)
    game.turn=20
    game.locks={0}
    game.boards[game.actor].rows[1]=[2,3,4,5,6]
    game.dice=(6,6,6,6,6,6)
    assert any(a[2] for a in game.legal())
    assert search_budget(game,(1,12,True))==128
    assert should_search(game,(1,12,True))


def test_routine_early_choice_uses_frozen_incumbent_without_rollouts(monkeypatch):
    game=Game(seed=2)
    incumbent=game.legal()[0] if game.legal() else None
    monkeypatch.setattr("qwixx_ai.search_adaptive.outcome",
                        lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("search ran")))
    assert select(game,incumbent_policy=lambda g: incumbent)==incumbent
