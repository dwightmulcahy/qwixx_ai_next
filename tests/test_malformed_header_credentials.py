import http.client
import threading
from http.server import ThreadingHTTPServer

from qwixx_ai import distributed


def test_malformed_admin_header_is_forbidden(monkeypatch):
    monkeypatch.setattr(distributed, "ADMIN_TOKEN", "secret")
    with ThreadingHTTPServer(("127.0.0.1", 0), distributed.Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                conn.request("POST", "/api/admin", body="{}", headers={"X-Admin-Token": "é"})
                response = conn.getresponse()
                assert response.status == 403
                response.read()
            finally:
                conn.close()
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_worker_credentials(monkeypatch):
    monkeypatch.setattr(distributed, "WORKER_TOKEN", "secret")
    handler = object.__new__(distributed.Handler)
    for token, allowed in [("é", False), ("", False), ("wrong", False), ("secret", True)]:
        handler.headers = {"X-Worker-Token": token}
        assert handler.authorized_worker() is allowed
    monkeypatch.setattr(distributed, "WORKER_TOKEN", "")
    assert handler.authorized_worker()
