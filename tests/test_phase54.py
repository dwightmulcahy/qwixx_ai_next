"""Phase 5.4 sampling and selection regression tests."""
import pytest

from qwixx_ai.phase54 import exploration_rate, opponent_pool, select_checkpoint


def test_exploration_starts_low_and_decays():
    assert exploration_rate(0, 5000) == pytest.approx(0.08)
    assert exploration_rate(4000, 5000) == pytest.approx(0.015)
    assert exploration_rate(4999, 5000) == pytest.approx(0.015)
    with pytest.raises(ValueError):
        exploration_rate(5000, 5000)


def test_strong_opponents_dominate_sampling():
    tickets = opponent_pool(3, 4)
    strong = sum(isinstance(ticket, tuple) for ticket in tickets)
    assert strong > len(tickets) / 2
    assert ("permanent", 1) in tickets
    assert ("permanent", 2) in tickets
    with pytest.raises(ValueError):
        opponent_pool(2, 4)


def test_selection_rejects_negative_historical_advantage(monkeypatch):
    def fake_eval(challenger, reference, lineup, rounds, seed, confidence):
        advantage = {100: 0.03, 100100: -0.01, 200100: 0.02}[seed]
        return {"paired_advantage": advantage,
                "advantage_ci_low": advantage - 0.005,
                "advantage_ci_high": advantage + 0.005}

    monkeypatch.setattr("qwixx_ai.phase54.controlled_evaluate", fake_eval)
    result = select_checkpoint("new", "old", ["h1", "h2"],
                               rounds=2, seed=100, lineup=["old", "h1", "h2"])
    assert not result["selection"]["selected"]
    assert not result["promoted"]


def test_selection_is_distinct_from_promotion(monkeypatch):
    def fake_eval(challenger, reference, lineup, rounds, seed, confidence):
        return {"paired_advantage": 0.02,
                "advantage_ci_low": -0.001,
                "advantage_ci_high": 0.041}

    monkeypatch.setattr("qwixx_ai.phase54.controlled_evaluate", fake_eval)
    result = select_checkpoint("new", "old", ["h1", "h2"],
                               rounds=2, seed=100, lineup=["old", "h1", "h2"])
    assert result["selection"]["selected"]
    assert not result["promoted"]
