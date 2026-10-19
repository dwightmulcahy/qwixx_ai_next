"""Tests for Phase 12.3 fingerprint and clustered contrast."""
from qwixx_ai.phase123 import bootstrap_contrast, fingerprint


def test_fingerprint_canonical():
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    assert fingerprint({"a": 2}) != fingerprint({"a": 3})


def test_cluster_bootstrap_positive():
    rows = [
        {"game_seed": game, "skip_group": group, "win_gap": gap}
        for game in range(10)
        for group, gap in (("3+", 0.2), ("0-2", -0.1))
    ]
    result = bootstrap_contrast(rows, iterations=200, seed=4)
    assert result["estimable"]
    assert result["ci95"][0] > 0


def test_bootstrap_insufficient_games():
    assert bootstrap_contrast([
        {"game_seed": 1, "skip_group": "3+", "win_gap": 0.2},
    ], iterations=100) is None


def test_bootstrap_one_group_not_estimable():
    rows = [{"game_seed": i, "skip_group": "3+", "win_gap": 0.2}
            for i in range(5)]
    assert not bootstrap_contrast(rows, iterations=100)["estimable"]
