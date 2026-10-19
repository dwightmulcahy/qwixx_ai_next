"""Frozen candidate reproducibility and validation."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.decision_quality import features, policy
from qwixx_ai.decision_quality_heldout import (
    FROZEN_MARGIN, FROZEN_WEIGHTS, evaluate,
)


def test_weights_match_features():
    game = Game(seed=5)
    assert len(FROZEN_WEIGHTS) == len(features(game, None))
    assert FROZEN_MARGIN == 0.02


def test_frozen_policy_legal():
    game = Game(seed=5)
    assert policy(FROZEN_WEIGHTS, FROZEN_MARGIN)(game) in [
        None, *game.legal()
    ]


def test_reject_reused_seed():
    with pytest.raises(ValueError, match="fresh"):
        evaluate(rounds=2, seed=22000000)


def test_invalid_rounds():
    with pytest.raises(ValueError):
        evaluate(rounds=1)
