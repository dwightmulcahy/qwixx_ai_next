"""Stronger rollout-policy regressions."""
from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.rollout_policy import rollout_policy


def test_rollout_policy_is_legal_for_four_player_game():
    game=Game(seed=17)
    action=rollout_policy(game)
    assert action in [None,*game.legal()]


def test_rollout_policy_falls_back_to_v2_for_other_player_counts():
    game=Game(n=3,seed=17)
    assert rollout_policy(game)==heuristic_v2(game)


def test_search_outcome_uses_rollout_policy(monkeypatch):
    from qwixx_ai import phase9_search
    game=Game(seed=19)
    calls={"count":0}
    original=phase9_search.rollout_policy

    def counted(trial):
        calls["count"]+=1
        return original(trial)

    monkeypatch.setattr(phase9_search,"rollout_policy",counted)
    phase9_search.outcome(game,None,12345)
    assert calls["count"]>0
