"""The "What's new" dialog: page wiring and the two static routes."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from qwixx_ai import distributed
from qwixx_ai.web import HTML


def test_page_includes_dialog_only_with_bundled_notes(tmp_path):
    assert distributed.with_whats_new("<head></head>", tmp_path / "missing.json") == "<head></head>"
    notes = tmp_path / "changelog.json"
    notes.write_text("{}")
    page = distributed.with_whats_new("<head></head>", notes)
    assert '<script src="/whats-new.js" data-src="/changelog.json" data-key="qwixx.seenVersion" defer></script></head>' in page
    # external script: allowed by the hash-based CSP via 'self'
    assert "script-src 'self'" in distributed.content_security_policy(page)
    assert '<button type="button" class="build-info-trigger" data-whats-new hidden>' in HTML


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed, "ROOT", tmp_path)
    monkeypatch.setattr(distributed, "CHANGELOG_JSON", tmp_path / "changelog.json")

    class TestHandler(distributed.Handler):
        def log_message(self, *args):
            pass

    TestHandler.coord = distributed.Coordinator()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    srv.shutdown()
    TestHandler.coord.db.close()


def test_routes_serve_script_and_notes(server):
    url, tmp = server
    with urllib.request.urlopen(f"{url}/whats-new.js", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("application/javascript")
        assert b"WhatsNew" in r.read()
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{url}/changelog.json", timeout=5)
    assert err.value.code == 404
    (tmp / "changelog.json").write_text(json.dumps({"current": "1.0.0", "releases": []}))
    with urllib.request.urlopen(f"{url}/changelog.json", timeout=5) as r:
        assert json.loads(r.read())["current"] == "1.0.0"
