"""Champion league regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.league import head_to_head, promote
from qwixx_ai.model import network


def test_head_to_head_deterministic_and_rotates_seats():
    model = network(len(observe(Game())))
    first = head_to_head(model, model, games=4, seed=55)
    second = head_to_head(model, model, games=4, seed=55)
    assert first == second
    assert 0 <= first["win_share"] <= 1
    assert first["games"] == 4
    assert first["opponent"] == "three_champions"


def test_promotion_gate_preserves_champion_on_failure(tmp_path):
    model = network(len(observe(Game())))
    destination = tmp_path / "champion.pt"
    result = promote(model, model, destination, games=4, seed=55,
                     minimum_share=1.0)
    assert not result["promoted"]
    assert not destination.exists()


def test_head_to_head_rejects_zero_games():
    model = network(len(observe(Game())))
    with pytest.raises(ValueError):
        head_to_head(model, model, games=0)


def test_promotion_rejects_invalid_threshold(tmp_path):
    model = network(len(observe(Game())))
    with pytest.raises(ValueError):
        promote(model, model, tmp_path / "champion.pt", games=6,
                minimum_share=0.25)


def test_promotion_requires_multiple_nonempty_batches(tmp_path):
    model = network(len(observe(Game())))
    with pytest.raises(ValueError):
        promote(model, model, tmp_path / "champion.pt", games=2, batches=3)
