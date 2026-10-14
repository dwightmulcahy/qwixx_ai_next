"""Frozen champion regression checks."""
from qwixx_ai import champion as champion_module
from qwixx_ai.champion import SETTINGS, champion
from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.web import Session


def test_frozen_settings():
    assert SETTINGS == {
        "futures": 32, "seed": 900000, "min_gain": 0.10,
        "min_score_gain": 0.0, "start_turn": 12, "end_turn": 35,
    }


def test_champion_uses_v2_for_non_four_player_games():
    game = Game(n=3, seed=4)
    assert champion(game) == heuristic_v2(game)


def test_champion_routes_four_player_policy(monkeypatch):
    game = Game(seed=4)
    monkeypatch.setattr(champion_module, "_search", lambda g: ("called", g.n))
    assert champion(game) == ("called", 4)


def test_web_champion_routes_promoted_policy(monkeypatch):
    import qwixx_ai.web as web

    game = Game(seed=4)
    monkeypatch.setattr(web, "champion", lambda g: ("champion", g.n))
    assert Session(difficulty="champion").ai(game) == ("champion", 4)


def test_web_advanced_without_model_falls_back_to_v2():
    game = Game(seed=4)
    assert Session(difficulty="advanced").ai(game) == heuristic_v2(game)
