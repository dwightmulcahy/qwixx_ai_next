from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


def test_published_image_carries_oci_version_labels():
    """Regression: images had no org.opencontainers.image.version label, so update tools
    (Updock) showed "no version label" for qwixx-ai."""
    text = RELEASE.read_text(encoding="utf-8")
    publish = text[text.index("Publish versioned Docker Hub image"):]
    for label in ("org.opencontainers.image.version=", "org.opencontainers.image.revision=",
                  "org.opencontainers.image.created=", "org.opencontainers.image.source="):
        assert label in publish, label
