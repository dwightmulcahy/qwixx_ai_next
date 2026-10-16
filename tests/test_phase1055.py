"""Phase 10.5.5 verifier regression and reproducibility tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase1055 import evidence, interval


def test_empirical_bernstein_bounds_widen_with_looks():
    deltas = [0.0, 0.5, -0.5, 0.25] * 16
    a = interval(deltas, 0.95, looks=1)
    b = interval(deltas, 0.95, looks=4)
    assert b[0] < a[0]
    assert b[1] > a[1]


def test_interval_invalid_method():
    with pytest.raises(ValueError):
        interval([0.0, 1.0], 0.95, method="unknown")


def test_same_action_requires_no_rollouts():
    game = Game(seed=42)
    for method in ("holdout", "bernstein", "legacy"):
        result = evidence(game, None, None, method=method)
        assert result["futures"] == 0
        assert not result["override"]


def test_holdout_is_reproducible():
    game = Game(seed=42)
    action = game.legal()[0]
    kwargs = dict(initial_futures=4, max_futures=8,
                  batch_size=2, validation_futures=4,
                  seed=1234, method="holdout")
    a = evidence(game, None, action, **kwargs)
    b = evidence(game, None, action, **kwargs)
    assert a == b
    assert a["futures"] <= 12


def test_invalid_config():
    with pytest.raises(ValueError):
        evidence(Game(seed=42), None, None, validation_futures=1)


def test_benchmark_reports_fixed_policy_without_futures_used(monkeypatch):
    from qwixx_ai import phase1055

    class FakePolicy:
        def __init__(self, *args, **kwargs):
            self.stats = {
                "verified": 2, "overrides": 0,
                "verification_seconds": 0.1,
                "by_seat": {},
            }
            self.records = []

    def fake_compare(*args, **kwargs):
        return {"paired_advantage": 0.0}

    monkeypatch.setattr(phase1055, "ExperimentalPolicy", FakePolicy)
    monkeypatch.setattr(phase1055, "VerifiedPolicy", FakePolicy)
    monkeypatch.setattr(phase1055, "paired_compare", fake_compare)
    result = phase1055.benchmark("unused.pt", rounds=2)
    assert result["policy_stats"]["fixed64"]["mean_futures"] == 64
