"""Decision quality pipeline tests."""
import numpy as np
import pytest

from qwixx_ai.core import Game
from qwixx_ai.decision_quality import collect, diagnose, features, fit, policy


def test_feature_dimensions_and_rng_independence():
    game = Game(seed=3)
    other = game.clone()
    other.rng.seed(999)
    assert features(game, None) == features(other, None)
    assert len({len(features(game, a)) for a in [None, *game.legal()]}) == 1


def test_small_collection_and_fit():
    dataset = collect(games=2, max_states=2, futures=2)
    assert dataset["states"]
    weights = fit(dataset)
    assert np.isfinite(weights).all()
    assert diagnose(dataset, weights)["states"] == len(dataset["states"])


def test_policy_legal():
    game = Game(seed=3)
    weights = [0.0] * len(features(game, None))
    assert policy(weights)(game) in [None, *game.legal()]


def test_invalid_ridge():
    with pytest.raises(ValueError):
        fit({"states": []}, ridge=0)
