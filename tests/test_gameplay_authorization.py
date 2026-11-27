"""Exercise gameplay authorization through real HTTP requests."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwixx_ai.distributed import Handler


@pytest.mark.parametrize("path", ["/move", "/undo-step1", "/api/session-progress"])
def test_gameplay_session_ownership(monkeypatch, path):
    game = SimpleNamespace(thinking_view={"private": True}, thinking_actor=1)
    coord = SimpleNamespace(
        sessions={"guest": (game, 0, None), "registered": (game, 0, "alice")},
        player_stats=SimpleNamespace(authenticate=lambda token: {
            "alice-token": "alice", "bob-token": "bob"
        }.get(token)),
        session_action=Mock(return_value={"ok": True}),
        session_undo_step1=Mock(return_value={"ok": True}),
    )
    monkeypatch.setattr(Handler, "coord", coord)
    monkeypatch.setattr(Handler, "guest_session_tokens", {"guest": "guest-secret"})
    monkeypatch.setattr(Handler, "game_limited", lambda self: False)
    monkeypatch.setattr(Handler, "inference_slots", threading.BoundedSemaphore(1))

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for sid, cookie, expected in [
                ("guest", "", 403),
                ("guest", "qwixx_guest=wrong", 403),
                ("guest", "qwixx_auth=alice-token", 403),
                ("registered", "", 403),
                ("registered", "qwixx_auth=bob-token", 403),
                ("registered", "qwixx_guest=guest-secret", 403),
                ("guest", "qwixx_guest=guest-secret", 200),
                ("registered", "qwixx_auth=alice-token", 200),
            ]:
                coord.session_action.reset_mock()
                coord.session_undo_step1.reset_mock()
                conn = http.client.HTTPConnection(*server.server_address, timeout=3)
                try:
                    if path.startswith("/api/"):
                        conn.request("GET", path + "?session_id=" + sid,
                                     headers={"Cookie": cookie})
                    else:
                        conn.request("POST", path, json.dumps({"session_id": sid}),
                                     {"Cookie": cookie, "Content-Type": "application/json"})
                    response = conn.getresponse()
                    body = json.loads(response.read())
                    assert response.status == expected
                    if expected == 403 or path.startswith("/api/"):
                        coord.session_action.assert_not_called()
                        coord.session_undo_step1.assert_not_called()
                    elif path == "/move":
                        coord.session_action.assert_called_once_with(sid, None, None, None)
                    else:
                        coord.session_undo_step1.assert_called_once_with(sid, None, None)
                    if path.startswith("/api/"):
                        assert ("state" in body) == (expected == 200)
                finally:
                    conn.close()
        finally:
            server.shutdown()
            thread.join(timeout=3)
