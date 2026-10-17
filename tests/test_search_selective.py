"""Selective search correctness and efficiency checks."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.search_selective import choose_from_estimates, select


def test_no_search_outside_window():
    game = Game(seed=3)
    stats = {"skipped": 0, "searched": 0, "futures": 0,
             "refined": 0, "overrides": 0}
    assert select(game, stats=stats) == heuristic_v2(game)
    assert stats["skipped"] == 1
    assert stats["futures"] == 0


def test_safeguard():
    actions = [None, (0, 2, False)]
    assert choose_from_estimates(actions, [5, 6], [20, 25], 32, None) is None
    assert choose_from_estimates(actions, [5, 10], [20, 25], 32, None) == actions[1]


def test_invalid_budget():
    with pytest.raises(ValueError, match="invalid"):
        select(Game(seed=3), initial=0)


def test_does_not_call_champion(monkeypatch):
    import qwixx_ai.search_selective as module
    monkeypatch.setattr(module, "champion", lambda _: (_ for _ in ()).throw(AssertionError()))
    assert select(Game(seed=3)) in [None, *Game(seed=3).legal()]
