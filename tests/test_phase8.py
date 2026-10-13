"""Phase 8 override-audit smoke and determinism tests."""
import json

import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import observe
from qwixx_ai.model import network
from qwixx_ai.phase8_audit import audit


def test_audit_smoke(tmp_path):
    model = network(observe(Game()).size)
    output = tmp_path / "overrides.jsonl"
    result = audit(model, games=3, seed=810, futures=1,
                   max_overrides=2, bootstrap=10, output=output)
    assert result["games"] == 3
    assert result["sampled_overrides"] <= 2
    assert result["actual_overrides"] >= result["sampled_overrides"]
    assert 0 <= result["override_rate"] <= 1
    assert output.exists()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == result["sampled_overrides"]
    assert all(row["jump"] >= 4 for row in rows)


def test_invalid_sizes():
    model = network(observe(Game()).size)
    with pytest.raises(ValueError, match="positive"):
        audit(model, games=0)
