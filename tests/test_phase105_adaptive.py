"""Phase 10.5.3 adaptive verification tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase105_adaptive import (
    adaptive_evidence,
    decision_summary,
)


def test_summary_defaults_to_no_override_with_one_future():
    result = decision_summary([1.0], [10.0], 0.95, 0.05)
    assert not result["override"]


def test_same_action_skips_simulations():
    game = Game(seed=123)
    result = adaptive_evidence(game, None, None, initial_futures=4,
                               max_futures=8)
    assert result["futures"] == 0
    assert not result["override"]


def test_adaptive_reproducible_and_bounded():
    game = Game(seed=123)
    action = game.legal()[0]
    options = {"initial_futures": 4, "max_futures": 8,
               "batch_size": 2, "seed": 987}
    first = adaptive_evidence(game, None, action, **options)
    second = adaptive_evidence(game, None, action, **options)
    assert first == second
    assert 4 <= first["futures"] <= 8
    assert first["stop_reason"] in {
        "positive_evidence", "negative_evidence", "max_futures"}


def test_invalid_configuration_rejected():
    game = Game(seed=123)
    with pytest.raises(ValueError):
        adaptive_evidence(game, None, None, initial_futures=1)
    with pytest.raises(ValueError):
        adaptive_evidence(game, None, None, initial_futures=8,
                          max_futures=4)
    with pytest.raises(ValueError):
        adaptive_evidence(game, None, (0, 999, False))
