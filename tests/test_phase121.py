"""Phase 12.1 deterministic replay and action diagnostics."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import encode
from qwixx_ai.phase121 import action_details, analyze, replay_decisions


def test_action_details_fresh_board():
    game = Game(n=4, seed=12)
    action = game.legal()[0]
    info = action_details(game, encode(action))
    assert info["existing_marks"] == 0
    assert info["skipped_cells"] >= 0
    assert info["remaining_after"] >= 0


def test_replay_empty_decisions():
    assert replay_decisions({"decisions": []}) == []


def test_replay_rejects_unmatched_position():
    with pytest.raises(ValueError, match="unmatched"):
        replay_decisions({"decisions": [{
            "game_seed": 42, "turn": 9999, "phase": 0, "actor": 1,
        }]})


def test_analyze_rejects_wrong_phase():
    with pytest.raises(ValueError, match="11.9"):
        analyze({"phase": "11.8", "decisions": []})
