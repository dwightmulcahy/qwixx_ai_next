"""Phase 10.5.6 replication and audit tests."""
import pytest

from qwixx_ai import phase1056


def test_audit_reports_truncation_and_confirmed_overrides():
    stats = {"verified": 5}
    records = [
        {"stop_reason": "positive_evidence", "override": True,
         "validation_gain": 0.3, "seat": 2},
        {"stop_reason": "negative_evidence", "override": False,
         "validation_gain": -0.1, "seat": 3},
    ]
    result = phase1056.audit(records, stats)
    assert result["records_truncated"]
    assert result["recorded_overrides"] == 1
    assert result["recorded_validation_rejections"] == 1
    assert result["accepted_by_seat"] == {2: 1}


def test_replication_report_and_gates(monkeypatch):
    class FakeHoldout:
        def __init__(self, *args, **kwargs):
            self.stats = {
                "verified": 2, "overrides": 1,
                "futures_used": 80, "verification_seconds": 0.08,
            }
            self.records = []

    class FakeFixed:
        def __init__(self, *args, **kwargs):
            self.stats = {
                "verified": 2, "overrides": 1,
                "verification_seconds": 0.16,
            }

    def compare(*args, **kwargs):
        return {
            "advantage_ci_low": -0.01,
            "challenger_mean_score": 58,
            "reference_mean_score": 57,
        }

    monkeypatch.setattr(phase1056, "ExperimentalPolicy", FakeHoldout)
    monkeypatch.setattr(phase1056, "VerifiedPolicy", FakeFixed)
    monkeypatch.setattr(phase1056, "paired_compare", compare)
    result = phase1056.replicate("unused.pt", rounds=2)
    assert result["efficiency"]["rollout_reduction_fraction"] == 0.375
    assert result["checks"]["lower_mean_verification_latency"]
    assert not result["eligible_for_champion_qualification"]
    assert not result["promoted"]


def test_rejects_invalid_rounds():
    with pytest.raises(ValueError):
        phase1056.replicate("unused.pt", rounds=1)
