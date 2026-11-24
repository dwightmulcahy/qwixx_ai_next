"""Regression tests for shared throttling and malformed guest credentials."""
from qwixx_ai.distributed import Handler


def test_rate_pruning_preserves_shared_state(monkeypatch):
    monkeypatch.setattr(Handler, "rate_buckets", {(str(i), "game"): [-1000] for i in range(2050)})
    monkeypatch.setattr("qwixx_ai.distributed.time.monotonic", lambda: 100.0)
    monkeypatch.setattr(Handler, "game_limit_per_minute", 1)
    first = object.__new__(Handler)
    second = object.__new__(Handler)
    first.client_address = second.client_address = ("127.0.0.1", 1234)
    assert not first.game_limited()
    assert len(Handler.rate_buckets) == 1
    assert second.game_limited()


def test_non_ascii_guest_cookie_is_denied(monkeypatch):
    monkeypatch.setattr(Handler, "guest_session_tokens", {"session": "secret"})
    monkeypatch.setattr(Handler, "guest_token", lambda self: "é")
    handler = object.__new__(Handler)
    assert not handler.owns_session("session", (None, 0, None))
