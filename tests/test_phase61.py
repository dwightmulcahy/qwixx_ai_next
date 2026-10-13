"""Phase 6.1 smoke tests."""
import json

import pytest
import torch

from qwixx_ai.core import Game
from qwixx_ai.env import mask
from qwixx_ai.model import network, save, load
from qwixx_ai.phase61_train import play_episode, sample_action, train
from qwixx_ai.phase61_validate import evaluate


def test_sampled_action_is_legal():
    game = Game(seed=5)
    policy = network( Game_observation_size())
    generator = torch.Generator().manual_seed(1)
    for _ in range(8):
        action, log_prob = sample_action(policy, game, generator)
        assert action is None or action in game.legal()
        assert torch.isfinite(log_prob)
        game.step(action)


def Game_observation_size():
    from qwixx_ai.env import observe
    return observe(Game()).size


def test_training_smoke(tmp_path):
    initial = tmp_path / "initial.pt"
    save(network(Game_observation_size()), initial)
    result = train(initial, tmp_path / "out", episodes=2,
                   batch_games=2, eval_interval=2, seed=701)
    assert result["episodes"] == 2
    assert (tmp_path / "out" / "latest.pt").exists()
    assert len((tmp_path / "out" / "metrics.jsonl").read_text().splitlines()) == 1
    load(result["checkpoint"])


def test_episode_outcome_and_gradients():
    model = network(Game_observation_size())
    result = play_episode(model, 702, torch.Generator().manual_seed(1))
    reward, share, score, logs, imitation = result
    assert -1.1 <= reward <= 1.1
    assert 0 <= share <= 1
    assert isinstance(score, int)
    assert logs and imitation


def test_validation_gate_rejects_ties():
    model = network(Game_observation_size())
    result = evaluate(model, model, rounds=2, seed=900)
    assert result["promotion_eligible"] is False
    assert result["against_imitation"]["paired_advantage"] == 0


def test_invalid_training_configuration(tmp_path):
    with pytest.raises(ValueError, match="invalid"):
        train(tmp_path / "missing.pt", episodes=0)
