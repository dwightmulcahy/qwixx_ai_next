"""Email verification and session regression tests."""
import re
from unittest.mock import patch
import pytest
from qwixx_ai.player_stats import PlayerStats


class FakeSMTP:
    messages = []
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def starttls(self):pass
    def login(self,*args):pass
    def send_message(self,message):self.messages.append(message)


def test_email_registration_signin_and_logout(tmp_path,monkeypatch):
    monkeypatch.setenv("QWIXX_SMTP_HOST","smtp.example.com")
    monkeypatch.setenv("QWIXX_SMTP_FROM","qwixx@example.com")
    FakeSMTP.messages.clear()
    store=PlayerStats(tmp_path/"players.sqlite3")
    with patch("qwixx_ai.player_stats.smtplib.SMTP",FakeSMTP):
        assert store.request_code("Player@Example.com","Player One")["sent"]
        with pytest.raises(ValueError,match="60 seconds"):
            store.request_code("player@example.com")
        code=re.search(r"\b\d{6}\b",FakeSMTP.messages[-1].get_content()).group()
        with pytest.raises(ValueError,match="Incorrect"):
            store.verify_code("player@example.com","000000" if code!="000000" else "111111")
        verified=store.verify_code("player@example.com",code)
        pid=verified["player"]["id"]
        assert store.authenticate(verified["token"])==pid
        assert store.authenticate("invalid") is None
        store.logout(verified["token"])
        assert store.authenticate(verified["token"]) is None
        store.db.execute("DELETE FROM auth_codes")
        store.db.commit()
        store.request_code("player@example.com")
        code=re.search(r"\b\d{6}\b",FakeSMTP.messages[-1].get_content()).group()
        assert store.verify_code("player@example.com",code)["player"]["id"]==pid
    store.db.close()


def test_email_requires_smtp(tmp_path,monkeypatch):
    monkeypatch.delenv("QWIXX_SMTP_HOST",raising=False)
    store=PlayerStats(tmp_path/"players.sqlite3")
    with pytest.raises(RuntimeError,match="not configured"):
        store.request_code("person@example.com")
    store.db.close()
