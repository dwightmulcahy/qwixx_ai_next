"""Phase 5.2 symmetric match and promotion regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase52 import symmetric_evaluate, evaluate_promotion


def test_identical_policies_have_zero_paired_advantage():
    model = network(observe(Game()).size)
    result = symmetric_evaluate(model, model, rounds=2, seed=123)
    assert result["games"] == 16
    assert result["paired_advantage"] == 0
    assert result["advantage_by_seat"] == [0.0] * 4
    assert result["advantage_ci_low"] == 0
    assert result == symmetric_evaluate(model, model, rounds=2, seed=123)


def test_identical_policies_cannot_promote(tmp_path):
    model = network(observe(Game()).size)
    destination = tmp_path / "champion.pt"
    result = evaluate_promotion(
        model, model, [model], rounds=2, seed=123,
        max_checks=10, destination=destination
    )
    assert not result["promoted"]
    assert result["comparison_confidence"] == pytest.approx(0.9975)
    assert not destination.exists()


def test_invalid_settings():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError):
        symmetric_evaluate(model, model, rounds=1)
    with pytest.raises(ValueError):
        evaluate_promotion(model, model, max_checks=0)
