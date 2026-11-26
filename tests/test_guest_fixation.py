import http.client
import threading
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from qwixx_ai.distributed import Handler


def test_guest_cookie_issuance_and_reuse(monkeypatch):
    monkeypatch.setattr(Handler, "guest_session_tokens", {"stale": "expired"})
    coord = SimpleNamespace(lock=threading.RLock(), sessions={})

    class GuestHandler(Handler):
        def do_GET(self):
            sid = self.path[1:]
            with coord.lock:
                coord.sessions[sid] = (None, 0, None)
            self.guest_session_reply({"session_id": sid})

    GuestHandler.coord = coord
    with ThreadingHTTPServer(("127.0.0.1", 0), GuestHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(sid, candidate):
                conn = http.client.HTTPConnection(*server.server_address, timeout=3)
                try:
                    conn.request("GET", "/" + sid, headers={"Cookie": "qwixx_guest=" + candidate})
                    response = conn.getresponse()
                    assert response.status == 200
                    cookie = SimpleCookie(response.getheader("Set-Cookie"))
                    response.read()
                    return cookie["qwixx_guest"].value
                finally:
                    conn.close()

            token = request("first", "attacker-chosen")
            assert token != "attacker-chosen"
            assert request("second", token) == token
            assert request("third", "expired") != "expired"
            assert Handler.guest_session_tokens["first"] == token
            assert "stale" not in Handler.guest_session_tokens
        finally:
            server.shutdown()
            thread.join(timeout=3)
