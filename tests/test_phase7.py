"""Phase 7 dataset collection and grouped merge regression tests."""
import numpy as np
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase7_disagreements import collect
from qwixx_ai.phase7_train import merge


def test_collection_legal_and_repeatable():
    model = network(observe(Game()).size)
    first, report = collect(model, games=2, seed=811, learner_fraction=0.5)
    second, other = collect(model, games=2, seed=811, learner_fraction=0.5)
    assert report == other
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["masks"][np.arange(len(first["actions"])),
                          first["actions"]].all()
    assert 0 <= report["disagreement_rate"] <= 1


def test_grouped_merge(tmp_path):
    sample = {
        "observations": np.zeros((2, 3), dtype=np.float32),
        "masks": np.ones((2, 4), dtype=np.bool_),
        "actions": np.array([0, 1], dtype=np.int64),
        "game_ids": np.array([0, 1], dtype=np.int64),
    }
    original = tmp_path / "original.npz"
    targeted = tmp_path / "targeted.npz"
    merged = tmp_path / "merged.npz"
    np.savez_compressed(original, **sample)
    np.savez_compressed(targeted, **sample)
    result = merge(original, targeted, merged, targeted_repeat=3)
    assert result["decisions"] == 8
    assert result["games"] == 4
    with np.load(merged) as data:
        assert set(data["game_ids"]) == {0, 1, 2, 3}
        assert list(data["game_ids"]).count(2) == 3


def test_collection_rejects_invalid_settings():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError, match="invalid"):
        collect(model, games=1)
