"""Email verification and session regression tests."""
import re
from unittest.mock import patch

import pytest

from qwixx_ai.player_stats import PlayerStats


class FakeSMTP:
    messages = []
    connections = []
    logins = []
    starttls_calls = 0

    def __init__(self, host, port, **kwargs):
        self.connections.append((host, port, kwargs.get("timeout")))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def starttls(self):
        type(self).starttls_calls += 1

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, message):
        self.messages.append(message)


def _reset_fake_smtp():
    FakeSMTP.messages.clear()
    FakeSMTP.connections.clear()
    FakeSMTP.logins.clear()
    FakeSMTP.starttls_calls = 0


def _configure_shared_smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "test-app-password")


def test_email_registration_signin_and_logout(tmp_path, monkeypatch):
    _configure_shared_smtp(monkeypatch)
    _reset_fake_smtp()
    store = PlayerStats(tmp_path / "players.sqlite3")
    with patch("qwixx_ai.player_stats.smtplib.SMTP", FakeSMTP):
        assert store.request_code("Player@Example.com", "Player One")["sent"]
        with pytest.raises(ValueError, match="60 seconds"):
            store.request_code("player@example.com")
        code = re.search(r"\b\d{6}\b", FakeSMTP.messages[-1].get_content()).group()
        with pytest.raises(ValueError, match="Incorrect"):
            store.verify_code("player@example.com", "000000" if code != "000000" else "111111")
        verified = store.verify_code("player@example.com", code)
        pid = verified["player"]["id"]
        assert store.authenticate(verified["token"]) == pid
        assert store.authenticate("invalid") is None
        store.logout(verified["token"])
        assert store.authenticate(verified["token"]) is None
        store.db.execute("DELETE FROM auth_codes")
        store.db.commit()
        store.request_code("player@example.com")
        code = re.search(r"\b\d{6}\b", FakeSMTP.messages[-1].get_content()).group()
        assert store.verify_code("player@example.com", code)["player"]["id"] == pid
    store.db.close()


def test_email_requires_smtp_credentials(tmp_path, monkeypatch):
    for key in (
        "QWIXX_SMTP_HOST",
        "QWIXX_SMTP_FROM",
        "QWIXX_SMTP_USER",
        "QWIXX_SMTP_PASSWORD",
        "QWIXX_SMTP_PORT",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASS",
        "GMAIL_ID",
        "GMAIL_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    store = PlayerStats(tmp_path / "players.sqlite3")
    with pytest.raises(RuntimeError, match="not configured"):
        store.request_code("person@example.com")
    store.db.close()


def test_shared_smtp_settings_are_supported(tmp_path, monkeypatch):
    _configure_shared_smtp(monkeypatch)
    _reset_fake_smtp()
    store = PlayerStats(tmp_path / "shared.sqlite3")
    with patch("qwixx_ai.player_stats.smtplib.SMTP", FakeSMTP):
        assert store.request_code("player@example.com")["sent"]
    assert FakeSMTP.connections == [("smtp.gmail.com", 587, 15)]
    assert FakeSMTP.starttls_calls == 1
    assert FakeSMTP.logins == [("sender@example.com", "test-app-password")]
    assert FakeSMTP.messages[-1]["From"] == "sender@example.com"
    store.db.close()


def test_legacy_smtp_settings_do_not_override_shared_settings(tmp_path, monkeypatch):
    _configure_shared_smtp(monkeypatch)
    monkeypatch.setenv("QWIXX_SMTP_HOST", "wrong.example.com")
    monkeypatch.setenv("QWIXX_SMTP_PORT", "465")
    monkeypatch.setenv("QWIXX_SMTP_USER", "wrong@example.com")
    monkeypatch.setenv("QWIXX_SMTP_PASSWORD", "wrong-password")
    monkeypatch.setenv("QWIXX_SMTP_FROM", "wrong@example.com")
    monkeypatch.setenv("QWIXX_SMTP_SSL", "true")
    monkeypatch.setenv("GMAIL_ID", "wrong@gmail.com")
    monkeypatch.setenv("GMAIL_TOKEN", "wrong-token")
    _reset_fake_smtp()
    store = PlayerStats(tmp_path / "legacy.sqlite3")
    with patch("qwixx_ai.player_stats.smtplib.SMTP", FakeSMTP):
        assert store.request_code("player@example.com")["sent"]
    assert FakeSMTP.connections == [("smtp.gmail.com", 587, 15)]
    assert FakeSMTP.starttls_calls == 1
    assert FakeSMTP.logins == [("sender@example.com", "test-app-password")]
    assert FakeSMTP.messages[-1]["From"] == "sender@example.com"
    store.db.close()
