"""Smoke tests for Phase 6 data, imitation and checkpoint compatibility."""
import numpy as np

from qwixx_ai.env import ACTION_COUNT
from qwixx_ai.model import load
from qwixx_ai.phase6_dataset import generate
from qwixx_ai.phase6_train import train


def test_dataset_legal_labels_and_outcomes():
    data = generate(3, seed=501)
    count = len(data["actions"])
    assert data["observations"].shape[0] == count
    assert data["masks"].shape == (count, ACTION_COUNT)
    assert data["masks"][np.arange(count), data["actions"]].all()
    assert set(data["game_ids"]) == {0, 1, 2}
    assert np.all((data["win_shares"] >= 0) & (data["win_shares"] <= 1))


def test_dataset_repeatable():
    first = generate(2, seed=502)
    second = generate(2, seed=502)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_imitation_checkpoint_loads(tmp_path):
    data = generate(4, seed=503)
    path = tmp_path / "dataset.npz"
    checkpoint = tmp_path / "imitation.pt"
    np.savez_compressed(path, **data)
    result = train(path, checkpoint, epochs=1, batch_size=64, seed=4)
    assert result["train_games"] + result["validation_games"] == 4
    assert 0 <= result["best_validation_accuracy"] <= 1
    model = load(checkpoint)
    assert model is not None
