"""Regression coverage for Phase 11.6 collection and gate."""
import json

import pytest

from qwixx_ai.phase116 import collect, save_dataset
from qwixx_ai.phase116_gate import evaluate


def test_collection_requires_valid_arguments():
    with pytest.raises(ValueError):
        collect("unused.pt", games=0)
    with pytest.raises(ValueError):
        collect("unused.pt", max_records=0)


def test_save_dataset(tmp_path):
    path = tmp_path / "representative.jsonl"
    records = [{"split": "train", "agreement": True}]
    save_dataset(path, {"saved": 1}, records)
    assert json.loads(path.read_text().strip()) == records[0]
    assert json.loads(path.with_suffix(".summary.json").read_text())["saved"] == 1


def test_gate_rejects_invalid_rounds():
    with pytest.raises(ValueError):
        evaluate("unused.pt", "unused.pt", rounds=1)
