"""Small deterministic Phase 10.1 diagnostics tests."""
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.phase10_diagnostics import compare_diagnostics


def test_diagnostics_report_all_seats_and_outcomes():
    result = compare_diagnostics(heuristic_v2, heuristic_v2, [heuristic_v2] * 3,
                                 rounds=2, seed=20261014)
    assert result["games_per_policy"] == 8
    for policy in ("challenger", "reference"):
        overall = result[policy]["overall"]
        assert overall["games"] == 8
        assert overall["wins"] + overall["ties"] + overall["losses"] == 8
        assert set(result[policy]["by_seat"]) == {"1", "2", "3", "4"}
        assert all(result[policy]["by_seat"][seat]["games"] == 2
                   for seat in ("1", "2", "3", "4"))
        assert 0 <= overall["mean_penalties"] <= 4
        assert overall["mean_marks"] >= 0
    assert result["challenger"] == result["reference"]


def test_diagnostics_reject_bad_inputs():
    import pytest
    with pytest.raises(ValueError):
        compare_diagnostics(heuristic_v2, heuristic_v2, [heuristic_v2] * 3,
                            rounds=0, seed=1)
