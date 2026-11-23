"""Game request rate-limiting regression tests."""
from qwixx_ai.distributed import Handler


def test_game_limit_is_independent_by_ip(monkeypatch):
    monkeypatch.setattr(Handler, "rate_buckets", {})
    monkeypatch.setattr(Handler, "game_limit_per_minute", 3)
    first = object.__new__(Handler)
    first.client_address = ("192.0.2.1", 1234)
    second = object.__new__(Handler)
    second.client_address = ("192.0.2.2", 1234)
    assert [first.game_limited() for _ in range(4)] == [False, False, False, True]
    assert second.game_limited() is False
    assert ("192.0.2.1", "game") in Handler.rate_buckets


def test_game_limit_expires_old_requests(monkeypatch):
    monkeypatch.setattr(Handler, "rate_buckets", {("192.0.2.3", "game"): [1.0]})
    monkeypatch.setattr(Handler, "game_limit_per_minute", 1)
    monkeypatch.setattr("qwixx_ai.distributed.time.monotonic", lambda: 100.0)
    handler = object.__new__(Handler)
    handler.client_address = ("192.0.2.3", 1234)
    assert handler.game_limited() is False
    assert handler.game_limited() is True


def test_game_limit_prunes_stale_ip_buckets(monkeypatch):
    old = {(f"192.0.{i // 256}.{i % 256}", "game"): [1.0] for i in range(2049)}
    monkeypatch.setattr(Handler, "rate_buckets", old)
    monkeypatch.setattr("qwixx_ai.distributed.time.monotonic", lambda: 100.0)
    handler = object.__new__(Handler)
    handler.client_address = ("198.51.100.1", 1234)
    assert handler.game_limited() is False
    assert len(handler.rate_buckets) == 1
