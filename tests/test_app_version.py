"""The heading uses the release tag provided at build time."""
import importlib


def test_version_appears_next_to_heading(monkeypatch):
    import qwixx_ai.web as web

    monkeypatch.setenv("QWIXX_GIT_VERSION", "v1.2.3")
    try:
        module = importlib.reload(web)
        assert 'Qwixx AI <small id="app-version"' in module.HTML
        assert ">v1.2.3</small>" in module.HTML
    finally:
        monkeypatch.delenv("QWIXX_GIT_VERSION", raising=False)
        importlib.reload(web)


def test_version_escaped(monkeypatch):
    import qwixx_ai.web as web

    monkeypatch.setenv("QWIXX_GIT_VERSION", "<unsafe>")
    try:
        module = importlib.reload(web)
        assert "&lt;unsafe&gt;" in module.HTML
        assert "><unsafe></small>" not in module.HTML
    finally:
        monkeypatch.delenv("QWIXX_GIT_VERSION", raising=False)
        importlib.reload(web)
