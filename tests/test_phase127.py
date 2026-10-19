"""Phase 12.7 diagnostic tests."""
import pytest
import torch

from qwixx_ai.core import Game
from qwixx_ai.env import ACTION_COUNT, encode, mask
from qwixx_ai.phase121 import snapshot
from qwixx_ai.phase123 import fingerprint
from qwixx_ai.phase127 import analyze, inspect, replay


class ConstantModel:
    def __call__(self, observation):
        return torch.zeros(ACTION_COUNT)


def test_inspect_requires_reproduced_challenger():
    game = Game(seed=1)
    source = encode(game.legal()[0])
    row = {"source_action": source, "champion_action": 0,
           "game_seed": 1, "state_sha256": "test", "stage": "middle",
           "actor": game.actor, "skip_group": "0-2", "win_gap": 0.1,
           "score_gap": 1.0}
    with pytest.raises(ValueError, match="does not reproduce"):
        inspect(ConstantModel(), row, game)


def test_replay_validates_snapshot():
    game = Game(seed=42)
    record = {"game_seed": 42, "turn": game.turn, "phase": game.phase,
              "actor": game.actor, "state_sha256": fingerprint(snapshot(game)),
              "source_action": 0, "champion_action": encode(game.legal()[0])}
    matches = list(replay({"decisions": [record]}))
    assert len(matches) == 1
    assert matches[0][1].actor == game.actor


def test_replay_rejects_bad_snapshot():
    game = Game(seed=42)
    record = {"game_seed": 42, "turn": 0, "phase": 0, "actor": 0,
              "state_sha256": "bad", "source_action": 0,
              "champion_action": encode(game.legal()[0])}
    with pytest.raises(ValueError, match="state mismatch"):
        list(replay({"decisions": [record]}))


def test_wrong_phase():
    with pytest.raises(ValueError, match="12.3"):
        analyze({"phase": "12.5"}, ConstantModel())


def test_legal_mask_includes_pass():
    assert mask(Game(seed=1))[0]
