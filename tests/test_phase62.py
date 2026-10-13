"""Phase 6.2 GAE, legal sampling, checkpoint, and development selection tests."""
import numpy as np
import pytest
import torch

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network, save, load
from qwixx_ai.phase62_train import Critic, episode, gae, train
from qwixx_ai.heuristic_v2 import heuristic_v2


def test_gae_terminal_only():
    advantages, returns = gae([0.0, 1.0], [0.0, 0.0], gamma=1.0, lam=1.0)
    np.testing.assert_allclose(advantages.numpy(), [1.0, 1.0])
    np.testing.assert_allclose(returns.numpy(), [1.0, 1.0])


def test_gae_validates_lengths():
    with pytest.raises(ValueError, match="equal"):
        gae([1.0], [])


def test_episode_records_legal_actions():
    size = observe(Game()).size
    policy = network(size)
    critic = Critic(size)
    data = episode(policy, critic, 6500,
                   torch.Generator().manual_seed(9),
                   heuristic_v2, [])
    states, masks, actions = data[:3]
    assert states
    assert all(bool(m[a]) for m, a in zip(masks, actions))
    assert 0 <= data[7] <= 1


def test_train_smoke(tmp_path):
    initial = tmp_path / "initial.pt"
    save(network(observe(Game()).size), initial)
    result = train(initial, tmp_path / "experiment", episodes=2,
                   batch_games=2, eval_interval=2, eval_rounds=2, seed=6511)
    load(result["best_development_checkpoint"])
    load(result["latest_checkpoint"])


def test_reject_bad_settings(tmp_path):
    with pytest.raises(ValueError, match="invalid"):
        train(tmp_path / "missing.pt", episodes=0)
