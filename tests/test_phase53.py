"""Regression tests for controlled fixed-lineup evaluation."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase53 import controlled_evaluate, evaluate_promotion


def test_identical_focal_policies_have_zero_advantage():
    model = network(observe(Game()).size)
    lineup = [model, model, model]
    a = controlled_evaluate(model, model, lineup, rounds=2, seed=123)
    assert a == controlled_evaluate(model, model, lineup, rounds=2, seed=123)
    assert a["games"] == 16
    assert a["paired_advantage"] == 0
    assert a["advantage_by_seat"] == [0.0] * 4


def test_promotion_rejects_identical_models(tmp_path):
    model = network(observe(Game()).size)
    destination = tmp_path / "champion.pt"
    result = evaluate_promotion(model, model, [model], rounds=2,
                                seed=123, max_checks=10,
                                destination=destination)
    assert not result["promoted"]
    assert not destination.exists()
    assert result["comparison_confidence"] == pytest.approx(0.9975)


def test_lineup_and_rounds_validation():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError):
        controlled_evaluate(model, model, [model], rounds=2)
    with pytest.raises(ValueError):
        controlled_evaluate(model, model, [model] * 3, rounds=1)
