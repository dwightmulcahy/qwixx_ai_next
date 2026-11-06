"""Version badge and build information dialog regression checks."""
import importlib


def test_version_appears_next_to_heading_and_is_clickable(monkeypatch):
    import qwixx_ai.web as web

    monkeypatch.setenv("QWIXX_GIT_VERSION", "v1.2.3")
    monkeypatch.setenv("QWIXX_GIT_SHA", "abcdef1234567890")
    monkeypatch.setenv("QWIXX_BUILD_DATE", "2026-11-06T15:30:00Z")
    try:
        module = importlib.reload(web)
        assert '<button type="button" id="app-version"' in module.HTML
        assert ">v1.2.3</button>" in module.HTML
        assert 'onclick="openBuildInfo()"' in module.HTML
        assert 'id="build-info-modal"' in module.HTML
        assert "Build information" in module.HTML
        assert "<code>abcdef1234567890</code>" in module.HTML
        assert "2026-11-06T15:30:00Z" in module.HTML
        assert "Python" in module.HTML
        assert "Architecture" in module.HTML
        assert "function openBuildInfo()" in module.HTML
        assert "function closeBuildInfo()" in module.HTML
    finally:
        monkeypatch.delenv("QWIXX_GIT_VERSION", raising=False)
        monkeypatch.delenv("QWIXX_GIT_SHA", raising=False)
        monkeypatch.delenv("QWIXX_BUILD_DATE", raising=False)
        importlib.reload(web)


def test_build_info_values_are_escaped(monkeypatch):
    import qwixx_ai.web as web

    monkeypatch.setenv("QWIXX_GIT_VERSION", "<unsafe>")
    monkeypatch.setenv("QWIXX_GIT_SHA", "<sha>")
    monkeypatch.setenv("QWIXX_BUILD_DATE", "<date>")
    try:
        module = importlib.reload(web)
        assert "&lt;unsafe&gt;" in module.HTML
        assert "&lt;sha&gt;" in module.HTML
        assert "&lt;date&gt;" in module.HTML
        assert "><unsafe></button>" not in module.HTML
        assert "<code><sha></code>" not in module.HTML
    finally:
        monkeypatch.delenv("QWIXX_GIT_VERSION", raising=False)
        monkeypatch.delenv("QWIXX_GIT_SHA", raising=False)
        monkeypatch.delenv("QWIXX_BUILD_DATE", raising=False)
        importlib.reload(web)
