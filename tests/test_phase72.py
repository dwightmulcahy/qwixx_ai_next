"""Phase 7.2 hybrid and counterfactual regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase72_counterfactual import evaluate as counterfactual
from qwixx_ai.phase72_hybrid import select
from qwixx_ai.phase72_validate import evaluate as tournament


def test_hybrid_actions_are_legal():
    model = network(observe(Game()).size)
    game = Game(seed=52)
    for mode in ("student", "v2", "pass_correction", "large_jump", "selective"):
        action = select(game, model, mode)
        assert action is None or action in game.legal()


def test_pass_correction_when_teacher_passes(monkeypatch):
    from qwixx_ai import phase72_hybrid

    game = Game(seed=52)
    action = game.legal()[0] if game.legal() else None
    if action is None:
        pytest.skip("seed has no legal mark")
    monkeypatch.setattr(phase72_hybrid, "choose", lambda model, g: action)
    monkeypatch.setattr(phase72_hybrid, "heuristic_v2", lambda g: None)
    assert select(game, object(), "pass_correction") is None


def test_counterfactual_smoke():
    model = network(observe(Game()).size)
    report = counterfactual(model, games=2, futures=1,
                            per_category=2, bootstrap=10)
    assert report["games"] == 2


def test_tournament_smoke():
    model = network(observe(Game()).size)
    report = tournament(model, rounds=2, seed=80)
    assert len(report["results"]) == 3
    assert report["comparison_confidence"] > 0.95


def test_reject_unknown_mode():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError, match="unknown"):
        select(Game(seed=1), model, "unknown")
