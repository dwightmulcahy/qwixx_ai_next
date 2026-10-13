"""Phase 5.5 training mode guardrails."""
import pytest

from qwixx_ai.train import train


def test_phase55_requires_phase54(tmp_path):
    with pytest.raises(ValueError, match="phase55 requires"):
        train(episodes=1, out=tmp_path / "a", phase55=True)


@pytest.mark.parametrize("weight,interval", [(-0.1, 4), (0.05, 0)])
def test_phase55_rejects_invalid_update_settings(tmp_path, weight, interval):
    with pytest.raises(ValueError, match="invalid Phase 5.5"):
        train(episodes=1, out=tmp_path / "b", phase55=True,
              phase54=True, anchor_weight=weight, update_every=interval)
