"""Phase 8.1 diagnostic and policy regression tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase81 import (
    diagnostic,
    exact_four_policy,
    group_interval,
    tournament,
)


def test_exact_four_policy_legal():
    model = network(observe(Game()).size)
    game = Game(seed=51)
    action = exact_four_policy(model)(game)
    assert action is None or action in game.legal()


def test_group_interval_empty():
    result = group_interval([], bootstrap=10, seed=1)
    assert result["states"] == 0
    assert result["win_share_ci_95"] is None


def test_group_interval_clusters_games():
    rows = [
        {"game_id": 1, "win_share_delta": 1.0, "score_delta": 2.0},
        {"game_id": 1, "win_share_delta": 0.0, "score_delta": 4.0},
        {"game_id": 2, "win_share_delta": 0.0, "score_delta": 0.0},
    ]
    result = group_interval(rows, bootstrap=20, seed=1)
    assert result["games"] == 2
    assert result["win_share_delta"] == pytest.approx(0.25)
    assert result["score_delta"] == pytest.approx(1.5)


def test_diagnostic_smoke(tmp_path):
    model = network(observe(Game()).size)
    result = diagnostic(model, games=2, seed=100, futures=1,
                        max_overrides=2, bootstrap=10,
                        output=str(tmp_path / "records.jsonl"))
    assert set(result["subgroups"]) == {"exactly_four", "five_or_more"}


def test_tournament_smoke():
    model = network(observe(Game()).size)
    result = tournament(model, rounds=2, seed=100)
    assert result["comparison_confidence"] == pytest.approx(0.975)
    assert result["against_v2"]["rounds"] == 2
