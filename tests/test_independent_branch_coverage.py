"""Targeted branch coverage for deterministic runtime fallbacks and validation."""
import hashlib
import time
from types import SimpleNamespace

import pytest
import torch

from qwixx_ai import champion as champion_module
from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import (
    DUELING_ARCH,
    LEGACY_ARCH,
    choose,
    export_policy,
    legacy_network,
    load,
    model_architecture,
    model_input_size,
    network,
)
from qwixx_ai.player_stats import PlayerStats


def completed(scores, penalties=0, turns=19):
    return SimpleNamespace(
        done=True,
        scores=lambda: scores,
        boards=[SimpleNamespace(penalties=penalties)],
        turn=turns,
    )


def test_model_runtime_fallback_and_serialization_branches(tmp_path, monkeypatch):
    dueling = network(3)
    legacy = legacy_network(3)

    assert dueling[0] is dueling.trunk[0]
    assert dueling[-1] is dueling.advantage
    with pytest.raises(IndexError):
        _ = dueling[1]

    assert model_architecture(dueling) == DUELING_ARCH
    assert model_architecture(legacy) == LEGACY_ARCH
    assert model_input_size(dueling) == 3
    assert model_input_size(legacy) == 3
    assert export_policy(dueling)["architecture"] == DUELING_ARCH
    assert export_policy(legacy)["architecture"] == LEGACY_ARCH

    legacy_path = tmp_path / "legacy.pt"
    torch.save({"model": legacy.state_dict(), "obs_size": 3}, legacy_path)
    assert model_architecture(load(legacy_path)) == LEGACY_ARCH

    dueling_path = tmp_path / "dueling.pt"
    torch.save({"model": dueling.state_dict(), "obs_size": 3}, dueling_path)
    assert model_architecture(load(dueling_path)) == DUELING_ARCH

    bad_path = tmp_path / "bad.pt"
    torch.save({"model": {}, "obs_size": 3, "architecture": "unknown-v1"}, bad_path)
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        load(bad_path)

    game = Game(seed=17)
    live_model = network(observe(game).size)
    monkeypatch.setattr("qwixx_ai.model.random.random", lambda: 0.0)
    monkeypatch.setattr("qwixx_ai.model.random.choice", lambda values: values[0])
    assert choose(live_model, game, epsilon=1.0) in game.legal()

    monkeypatch.setattr("qwixx_ai.model.random.random", lambda: 1.0)
    assert choose(live_model, game, epsilon=0.0) in game.legal()


def test_live_champion_lock_and_invalid_incumbent_fallbacks(monkeypatch):
    locking = Game(seed=51)
    monkeypatch.setattr(locking, "legal", lambda: [(0, 12, True)])
    assert champion_module._live_should_search(locking, None)

    game = Game(seed=52)
    legal = game.legal()
    assert legal
    monkeypatch.setattr(champion_module, "_live_incumbent", lambda selected: ("invalid",))
    monkeypatch.setattr(champion_module, "_live_should_search", lambda selected, incumbent: True)
    monkeypatch.setattr(champion_module, "heuristic_v2", lambda selected: ("still-invalid",))
    monkeypatch.setattr(champion_module, "outcome", lambda selected, action, seed: (0.0, 0.0))

    result = champion_module.live_champion(game, budget_ms=2000, max_futures=1)
    assert result is None


@pytest.mark.parametrize("value", [
    None,
    "missing-at-sign",
    "white space@example.com",
    "@example.com",
    "a@example",
    "a@.example.com",
    "a@example.com.",
])
def test_player_stats_email_validation_branches(value):
    with pytest.raises(ValueError, match="Invalid email"):
        PlayerStats.normalize_email(value)


def test_player_stats_auth_and_record_validation_branches(tmp_path, monkeypatch):
    store = PlayerStats(tmp_path / "players.sqlite3")
    try:
        with pytest.raises(ValueError, match="Name must be"):
            store.request_code("a@example.com", "")
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_PASS", raising=False)
        with pytest.raises(RuntimeError, match="not configured"):
            store.request_code("a@example.com")

        with pytest.raises(ValueError, match="Invalid verification code"):
            store.verify_code("a@example.com", "abc")
        with pytest.raises(ValueError, match="expired"):
            store.verify_code("a@example.com", "123456")

        now = time.time()
        email = "a@example.com"
        good = "123456"
        digest = hashlib.sha256((email + ":" + good).encode()).hexdigest()
        store.db.execute(
            "INSERT OR REPLACE INTO auth_codes VALUES (?,?,?,?,?,?)",
            (email, digest, now + 600, 0, now, "Tester"),
        )
        store.db.commit()
        with pytest.raises(ValueError, match="Incorrect code"):
            store.verify_code(email, "654321")

        with pytest.raises(ValueError, match="Player name must be text"):
            store.create_profile(123)
        with pytest.raises(ValueError, match="printable"):
            store.create_profile("bad\x01name")

        player = store.create_profile("Runtime Tester")
        with pytest.raises(ValueError, match="Invalid game mode"):
            store.record(
                "bad-mode", player["id"], "champion",
                completed([60, 50, 40, 30]), game_mode="bogus",
            )
        with pytest.raises(ValueError, match="Invalid player count"):
            store.record(
                "bad-count", player["id"], "champion",
                completed([60, 50, 40]), game_mode="standard",
            )
        with pytest.raises(ValueError, match="Invalid game mode"):
            store.leaderboard(game_mode="bogus")
    finally:
        store.db.close()


def test_ai_stats_skips_corrupt_rows_and_repairs_bad_seating(tmp_path):
    store = PlayerStats(tmp_path / "players.sqlite3")
    try:
        player = store.create_profile("AI Stats Branches")
        assert store.record("bad-json", player["id"], "champion", completed([50, 40, 30, 20]))
        assert store.record("short", player["id"], "champion", completed([60, 50, 40, 30]))
        assert store.record("bad-seating", player["id"], "champion", completed([70, 60, 50, 40]))

        store.db.execute("UPDATE results SET scores=? WHERE session_id='bad-json'", ("{",))
        store.db.execute("UPDATE results SET scores=? WHERE session_id='short'", ("[1,2]",))
        store.db.execute(
            "UPDATE results SET ai_names=? WHERE session_id='bad-seating'",
            ('["wrong","names","here"]',),
        )
        store.db.commit()

        stats = store.ai_stats()
        assert stats["games"] == 1
        assert all(player_stats["games"] == 1 for player_stats in stats["players"])
    finally:
        store.db.close()
