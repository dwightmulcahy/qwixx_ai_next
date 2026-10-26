"""Regression coverage for player career persistence and result calculations."""
from types import SimpleNamespace

import pytest

from qwixx_ai.player_stats import PlayerStats


def completed(scores, penalties=0, turns=19):
    return SimpleNamespace(
        done=True,
        scores=lambda: scores,
        boards=[SimpleNamespace(penalties=penalties)],
        turn=turns,
    )


def test_profiles_and_completed_results_survive_restart(tmp_path):
    path = tmp_path / "players.sqlite3"
    store = PlayerStats(path)
    player = store.create_profile("Dice Master")
    with pytest.raises(ValueError, match="already exists"):
        store.create_profile("dice master")
    assert store.profiles() == [player]
    assert not store.record("unfinished", player["id"], "champion", SimpleNamespace(done=False))
    assert store.record("one", player["id"], "champion", completed([70, 60, 55, 42], 1))
    assert not store.record("one", player["id"], "champion", completed([70, 60, 55, 42], 1))
    assert store.record("two", player["id"], "beginner", completed([35, 80, 70, 60], 2))
    assert store.record("three", player["id"], "champion", completed([80, 80, 60, 40], 0))
    assert store.record("four", player["id"], "champion", completed([90, 55, 70, 65], 0))
    stats = store.stats(player["id"])
    assert (stats["games"], stats["wins"], stats["losses"], stats["ties"]) == (4, 2, 1, 1)
    assert stats["win_rate"] == 50.0
    assert stats["average_score"] == 68.8
    assert stats["highest_score"] == 90
    assert stats["lowest_score"] == 35
    assert stats["current_win_streak"] == 1
    assert stats["longest_win_streak"] == 1
    assert len(stats["recent"]) == 4
    assert store.stats(player["id"], difficulty="champion")["games"] == 3
    store.db.close()
    reopened = PlayerStats(path)
    assert reopened.stats(player["id"])["games"] == 4
    reopened.db.close()


def test_rejects_invalid_players_and_difficulties(tmp_path):
    store = PlayerStats(tmp_path / "players.sqlite3")
    with pytest.raises(ValueError):
        store.create_profile(" ")
    with pytest.raises(ValueError):
        store.stats("unknown")
    player = store.create_profile("Tester")
    with pytest.raises(ValueError):
        store.stats(player["id"], difficulty="impossible")
    store.db.close()


def test_leaderboard_only_verified_players_and_ranking_filters(tmp_path):
    store = PlayerStats(tmp_path / "players.sqlite3")
    registered = store.create_profile("Registered")
    second = store.create_profile("Second")
    guest = store.create_profile("Guest")
    with store.lock:
        store.db.execute("UPDATE players SET email=? WHERE id=?", ("registered@example.com", registered["id"]))
        store.db.execute("UPDATE players SET email=? WHERE id=?", ("second@example.com", second["id"]))
        store.db.commit()
    for i in range(5):
        assert store.record(f"r-{i}", registered["id"], "champion", completed([70+i, 30, 20, 10]))
    store.record("s-1", second["id"], "beginner", completed([100, 20, 10, 5]))
    store.record("guest", guest["id"], "champion", completed([150, 20, 10, 5]))
    result = store.leaderboard()
    assert result["registered_players"] == 2
    assert [p["name"] for p in result["rankings"]["most_wins"]] == ["Registered", "Second"]
    assert result["rankings"]["highest_score"][0]["name"] == "Second"
    assert [p["name"] for p in result["rankings"]["win_rate"]] == ["Registered"]
    assert result["rankings"]["average_score"][0]["name"] == "Registered"
    assert [p["name"] for p in store.leaderboard("champion")["rankings"]["most_wins"]] == ["Registered"]
    assert "email" not in str(result)
    with pytest.raises(ValueError):
        store.leaderboard("impossible")
    store.db.close()
