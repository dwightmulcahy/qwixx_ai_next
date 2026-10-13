"""Phase 5 paired evaluation regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase5 import paired_evaluate, evaluate_promotion


def test_paired_evaluation_is_reproducible():
    model = network(observe(Game()).size)
    a = paired_evaluate(model, [model], rounds=2, seed=42)
    b = paired_evaluate(model, [model], rounds=2, seed=42)
    assert a == b
    assert a["games"] == 8
    assert len(a["win_share_by_seat"]) == 4
    assert a["ci95_low"] <= a["win_share"] <= a["ci95_high"]


def test_paired_evaluation_rejects_empty_pool():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError):
        paired_evaluate(model, [], rounds=2)


def test_promotion_does_not_save_unproven_challenger(tmp_path):
    model = network(observe(Game()).size)
    path = tmp_path / "champion.pt"
    result = evaluate_promotion(model, model, rounds=2, seed=42,
                                minimum_share=0.99, destination=path)
    assert not result["promoted"]
    assert not path.exists()
