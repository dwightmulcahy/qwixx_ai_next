"""Phase 12.8 confidence-gated hybrid tests."""
import pytest
import torch

from qwixx_ai.core import Game
from qwixx_ai.env import ACTION_COUNT
from qwixx_ai.phase128 import hybrid, neural_decision, run


class ConstantModel:
    def __call__(self, observation):
        return torch.zeros(ACTION_COUNT)


def test_neural_decision_is_legal():
    game = Game(seed=3)
    action, margin = neural_decision(ConstantModel(), game)
    assert action in [None, *game.legal()]
    assert margin >= 0


def test_hybrid_calls_fallback_on_tie(monkeypatch):
    import qwixx_ai.phase128 as phase128
    game = Game(seed=3)
    called = []
    monkeypatch.setattr(phase128, "champion", lambda g: called.append(True) or None)
    stats = {"decisions": 0, "fallbacks": 0, "changed": 0}
    assert hybrid(ConstantModel(), 0.1, stats)(game) is None
    assert called and stats["fallbacks"] == 1


def test_invalid_threshold():
    with pytest.raises(ValueError, match="nonnegative"):
        hybrid(ConstantModel(), -0.1)


def test_invalid_seed_overlap():
    with pytest.raises(ValueError, match="overlap"):
        run("unused.pt", tune_rounds=3, eval_rounds=3,
            tune_seed=10, eval_seed=11)


def test_invalid_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        run("unused.pt", thresholds=(0.1,))
