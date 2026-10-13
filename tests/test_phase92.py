"""Phase 9.2 audit smoke and independence tests."""
import pytest

from qwixx_ai.phase92_audit import audit


def test_reject_same_seed(tmp_path):
    with pytest.raises(ValueError, match="differ"):
        audit(games=1, search_seed=4, audit_seed=4,
              output=str(tmp_path / "audit.jsonl"))


def test_audit_smoke(tmp_path):
    output = tmp_path / "audit.jsonl"
    result = audit(games=2, futures=1, audit_futures=2,
                   output=str(output))
    assert result["decisions"]["decisions"] > 0
    assert output.exists()
    assert result["decisions"]["overrides"] >= 0
