"""Phase 7.1 collection, weighted training, and counterfactual smoke tests."""
import numpy as np
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network, save
from qwixx_ai.phase71_collect import collect
from qwixx_ai.phase71_counterfactual import evaluate, finish
from qwixx_ai.phase71_train import fit


def test_collect_and_weighted_fit(tmp_path):
    model = network(observe(Game()).size)
    data, summary = collect(model, games=4, seed=811, learner_fraction=0.5)
    assert summary["decisions"] == len(data["actions"])
    assert set(np.unique(data["weights"])).issubset({1.0, 1.5, 2.0})
    assert data["masks"][np.arange(len(data["actions"])), data["actions"]].all()
    dataset = tmp_path / "new.npz"
    initial = tmp_path / "initial.pt"
    np.savez_compressed(dataset, **data)
    save(model, initial)
    result = fit(dataset, initial, tmp_path / "candidate.pt", epochs=1,
                 batch_size=256)
    assert result["train_games"] == 3
    assert result["holdout_games"] == 1


def test_counterfactual_same_action_same_future():
    game = Game(seed=9)
    from qwixx_ai.heuristic_v2 import heuristic_v2

    action = heuristic_v2(game)
    assert finish(game, action, 12) == finish(game, action, 12)


def test_counterfactual_smoke():
    model = network(observe(Game()).size)
    result = evaluate(model, games=1, futures=1, max_states=1)
    assert result["sampled_disagreements"] <= 1


def test_reject_invalid_collection():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError, match="invalid"):
        collect(model, games=1)
