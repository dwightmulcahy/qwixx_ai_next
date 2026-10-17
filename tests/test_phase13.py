"""Phase 13 isolation and evaluation tests."""
import pytest

from qwixx_ai.phase13 import validate_seed_ranges, run_training, evaluate


def test_disjoint_seed_ranges():
    validate_seed_ranges(1, 5, 10, 5, 20, 5)


def test_overlapping_seeds_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_ranges(1, 5, 5, 5, 20, 5)


def test_bad_rounds_rejected():
    with pytest.raises(ValueError, match="rounds"):
        evaluate("unused", "unused", rounds=1)


def test_missing_training_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_training(tmp_path / "missing.pt", tmp_path / "out")
