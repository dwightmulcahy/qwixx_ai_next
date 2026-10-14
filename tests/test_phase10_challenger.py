"""Phase 10 evaluation contract tests without expensive tournaments."""
import json
from unittest.mock import patch

from qwixx_ai import phase10_challenger as phase10


def _result(lower, challenger_score=40, reference_score=39):
    return {
        "advantage_ci_low": lower,
        "advantage_ci_high": lower + 0.02,
        "challenger_mean_score": challenger_score,
        "reference_mean_score": reference_score,
    }


def test_eligible_requires_both_positive_bounds_and_score():
    with patch.object(phase10, "load", return_value=object()):
        with patch.object(phase10, "paired_compare", side_effect=[
            _result(0.01), _result(0.02),
        ]) as compare:
            result = phase10.evaluate_checkpoint("runs/best.pt", rounds=10, seed=123)
    assert result["eligible_for_review"] is True
    assert result["promoted"] is False
    assert compare.call_count == 2
    assert compare.call_args_list[0].kwargs["seed"] == 123
    assert compare.call_args_list[1].kwargs["seed"] == 100123


def test_no_promotion_on_inconclusive_champion_comparison():
    with patch.object(phase10, "load", return_value=object()):
        with patch.object(phase10, "paired_compare", side_effect=[
            _result(-0.001), _result(0.02),
        ]):
            result = phase10.evaluate_checkpoint("runs/best.pt", rounds=10)
    assert result["eligible_for_review"] is False


def test_no_promotion_if_score_worse():
    with patch.object(phase10, "load", return_value=object()):
        with patch.object(phase10, "paired_compare", side_effect=[
            _result(0.01, 38, 39), _result(0.02),
        ]):
            result = phase10.evaluate_checkpoint("runs/best.pt", rounds=10)
    assert result["eligible_for_review"] is False


def test_history_is_append_only(tmp_path):
    history = tmp_path / "evaluations.jsonl"
    phase10.append_history(history, {"eligible_for_review": False})
    phase10.append_history(history, {"eligible_for_review": True})
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["eligible_for_review"] is False
    assert records[1]["eligible_for_review"] is True
    assert all("evaluated_at" in item for item in records)
