"""Confidence-gated Phase 10.5.2 action verification tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase105_verified import paired_evidence, VerifiedPolicy


def test_same_action_has_zero_paired_gain():
    game = Game(seed=123)
    evidence = paired_evidence(game, None, None, futures=4, seed=99)
    assert evidence["win_gain"] == 0
    assert evidence["win_ci_low"] == 0
    assert evidence["score_gain"] == 0


def test_paired_evidence_reproducible():
    game = Game(seed=123)
    action = game.legal()[0]
    a = paired_evidence(game, None, action, futures=4, seed=99)
    b = paired_evidence(game, None, action, futures=4, seed=99)
    assert a == b


def test_illegal_action_rejected():
    game = Game(seed=123)
    with pytest.raises(ValueError, match="illegal"):
        paired_evidence(game, None, (0, 999, False), futures=4)


def test_invalid_settings_rejected():
    game = Game(seed=123)
    with pytest.raises(ValueError):
        paired_evidence(game, None, None, futures=1)
    with pytest.raises(ValueError):
        paired_evidence(game, None, None, confidence=1)
    with pytest.raises(ValueError):
        VerifiedPolicy("unused.pt", futures=1)
