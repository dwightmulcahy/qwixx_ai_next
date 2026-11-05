"""Seat-balanced AI benchmark regressions."""
import pytest

from qwixx_ai import tournament


def test_evaluation_requires_complete_seat_rotations():
    assert tournament.evaluation_seeds(8,seed=100)==[100,101]
    with pytest.raises(ValueError):
        tournament.evaluation_seeds(6,seed=100)


def test_evaluate_replays_each_seed_in_every_seat(monkeypatch):
    calls=[]
    model=object()

    def fake_choose(selected_model,game):
        assert selected_model is model
        return None

    def fake_play(agents,seed):
        seat=next(i for i,agent in enumerate(agents) if agent is not tournament.heuristic)
        calls.append((seed,seat))
        scores=[10,20,30,40]
        shares=[0,0,0,1]
        return scores,shares

    monkeypatch.setattr(tournament,"choose",fake_choose)
    monkeypatch.setattr(tournament,"play",fake_play)
    result=tournament.evaluate(model,games=8,opponent="heuristic",seed=55)

    assert calls==[(55,0),(55,1),(55,2),(55,3),(56,0),(56,1),(56,2),(56,3)]
    assert result["seat_balanced"] is True
    assert result["rounds"]==2
    assert [row["games"] for row in result["by_seat"]]==[2,2,2,2]


def test_paired_compare_uses_identical_benchmark(monkeypatch):
    seen=[]
    def fake_evaluate(model,games,opponent,seed):
        seen.append((model,games,opponent,seed))
        return {"win_share":0.3 if model=="candidate" else 0.2,"mean_score":50 if model=="candidate" else 45}
    monkeypatch.setattr(tournament,"evaluate",fake_evaluate)
    result=tournament.paired_compare("candidate","incumbent",games=192,seed=123)
    assert seen==[
        ("candidate",192,"heuristic",123),
        ("incumbent",192,"heuristic",123),
    ]
    assert result["win_share_delta"]==pytest.approx(0.1)
    assert result["mean_score_delta"]==5
