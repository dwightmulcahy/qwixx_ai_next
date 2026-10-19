"""Phase 11.5 recovery training safeguards."""
import json

import numpy as np
import pytest
import torch

from qwixx_ai.model import network
from qwixx_ai.phase115 import anchored_loss, train


def record(split, seed):
    return {
        "split": split, "game_seed": seed,
        "observation": [0.0, 1.0],
        "legal_actions": [0, 1],
        "champion_action": 1,
    }


def test_anchor_has_zero_initial_drift():
    model = network(2)
    anchor = network(2)
    anchor.load_state_dict(model.state_dict())
    item = record("train", 1)
    item["observation"] = np.asarray(item["observation"], dtype=np.float32)
    loss, drift, teacher = anchored_loss(model, anchor, [item])
    assert drift.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    assert torch.isfinite(teacher)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in anchor.parameters())


def test_anchor_penalizes_policy_drift():
    model = network(2)
    anchor = network(2)
    anchor.load_state_dict(model.state_dict())
    with torch.no_grad():
        model[-1].bias[0] += 2
    item = record("train", 1)
    item["observation"] = np.asarray(item["observation"], dtype=np.float32)
    _, drift, _ = anchored_loss(model, anchor, [item])
    assert drift.item() > 0


def test_recovery_never_reads_test_or_promotes(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("".join(json.dumps(record(split, seed)) + "\n" for split, seed
                               in [("train", 1), ("validation", 2), ("test", 3)]))
    source = tmp_path / "source.pt"
    torch.save({"model": network(2).state_dict(), "obs_size": 2}, source)
    output = tmp_path / "candidate.pt"
    result = train(dataset, source, output, epochs=2, batch_size=1, patience=1)
    assert result["split_counts"]["test"] == 1
    assert result["promoted"] is False
    assert result["best_epoch"] >= 0
    assert result["checkpoint_saved"] == output.exists()


def test_rejects_invalid_recovery_settings(tmp_path):
    with pytest.raises(ValueError, match="invalid recovery"):
        train(tmp_path / "missing", tmp_path / "missing", tmp_path / "out",
              anchor_weight=0)
