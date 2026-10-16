"""Phase 11.3 dataset safety and supervised-loss tests."""
import json

import numpy as np
import pytest
import torch

from qwixx_ai.phase113 import batch_loss, read_dataset


def example(split, game_seed, label=0):
    return {
        "split": split, "game_seed": game_seed,
        "observation": [0.0, 1.0],
        "legal_actions": [0, 1],
        "champion_action": label,
    }


def test_test_split_not_loaded(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [example("train", 1), example("validation", 2),
               example("test", 3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    splits, counts = read_dataset(path)
    assert counts == {"train": 1, "validation": 1, "test": 1}
    assert sum(map(len, splits.values())) == 2


def test_rejects_leaked_game(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        example("train", 1), example("validation", 1)]))
    with pytest.raises(ValueError, match="leakage"):
        read_dataset(path)


def test_rejects_illegal_teacher(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        example("train", 1, 4), example("validation", 2)]))
    with pytest.raises(ValueError, match="teacher"):
        read_dataset(path)


def test_loss_is_finite():
    model = torch.nn.Linear(2, 89)
    records = [example("train", 1)]
    records[0]["observation"] = np.array(
        records[0]["observation"], dtype="float32")
    loss = batch_loss(model, records, "cpu", 0.25)
    assert torch.isfinite(loss)
    loss.backward()
