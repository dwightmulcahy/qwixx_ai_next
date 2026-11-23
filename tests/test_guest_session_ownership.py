"""Anonymous game sessions require possession of the private guest cookie."""
from qwixx_ai.distributed import Handler


def test_guest_session_requires_matching_cookie(monkeypatch):
    monkeypatch.setattr(Handler, "guest_session_tokens", {"session-a": "secret-token"})
    handler = object.__new__(Handler)
    handler.headers = {"Cookie": "qwixx_guest=secret-token"}
    assert handler.owns_session("session-a", (None, 0, None))
    handler.headers = {"Cookie": "qwixx_guest=other-token"}
    assert not handler.owns_session("session-a", (None, 0, None))
    handler.headers = {}
    assert not handler.owns_session("session-a", (None, 0, None))
    assert not handler.owns_session("unknown", (None, 0, None))


def test_registered_session_still_uses_player_identity(monkeypatch):
    handler = object.__new__(Handler)
    monkeypatch.setattr(Handler, "player_id", lambda self: "alice")
    assert handler.owns_session("registered", (None, 0, "alice"))
    assert not handler.owns_session("registered", (None, 0, "bob"))
