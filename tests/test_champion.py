"""Champion and Phase 9.1 regression checks."""
from qwixx_ai import champion as champion_module
from qwixx_ai.champion import SETTINGS, base_champion, champion, live_champion, phase91_champion
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
    assert phase91_champion(game) == ("called", 4)
    monkeypatch.setattr(
        champion_module, "row_change_only",
        lambda g, incumbent: ("hybrid", incumbent),
    )
    assert champion(game) == ("hybrid", ("called", 4))


def test_web_champion_routes_bounded_live_policy(monkeypatch):
    import qwixx_ai.web as web

    game = Game(seed=4)
    monkeypatch.setattr(web, "live_champion", lambda selected_game: ("live", selected_game.n))
    assert Session(model=object(),difficulty="champion").ai(game) == ("live", 4)


def test_web_advanced_without_model_falls_back_to_v2():
    game = Game(seed=4)
    assert Session(difficulty="advanced").ai(game) == heuristic_v2(game)


def test_base_champion_remains_frozen_strength_reference(monkeypatch):
    monkeypatch.setattr(champion_module, "_search", lambda game: ("phase91", game.n))
    monkeypatch.setattr(
        champion_module, "row_change_only",
        lambda game, incumbent: ("row", incumbent),
    )
    assert base_champion(Game(seed=5)) == ("row", ("phase91", 4))


def test_live_champion_respects_future_cap(monkeypatch):
    game = Game(seed=8)
    game.turn = 15
    calls = []

    monkeypatch.setattr(champion_module, "_live_incumbent", lambda game: None)
    monkeypatch.setattr(champion_module, "_live_should_search", lambda game, incumbent: True)

    def fake_outcome(game, action, seed):
        calls.append((action, seed))
        return (1.0 if action is not None else 0.0, 50 if action is not None else 20)

    monkeypatch.setattr(champion_module, "outcome", fake_outcome)
    stats = {}
    result = live_champion(game, budget_ms=2000, max_futures=3, stats=stats)

    assert result in game.legal()
    assert stats["futures"] == 3
    assert stats["searched"] is True
    assert stats["overridden"] is True
    assert len(calls) == 3 * (1 + len(game.legal()))


def test_live_champion_stops_after_budget_between_matched_futures(monkeypatch):
    game = Game(seed=9)
    game.turn = 15
    legal = game.legal()
    monkeypatch.setattr(champion_module, "_live_incumbent", lambda game: legal[0])
    monkeypatch.setattr(champion_module, "_live_should_search", lambda game, incumbent: True)

    clock = iter([0.0, 0.40, 0.40])
    monkeypatch.setattr(champion_module.time, "perf_counter", lambda: next(clock, 0.40))
    monkeypatch.setattr(champion_module, "outcome", lambda game, action, seed: (0.0, 0.0))

    stats = {}
    live_champion(game, budget_ms=350, max_futures=12, stats=stats)
    assert stats["futures"] == 1


def test_live_champion_requires_win_share_and_score_improvement(monkeypatch):
    game = Game(seed=10)
    game.turn = 15
    legal = game.legal()
    incumbent = legal[0]
    challenger = legal[-1]
    monkeypatch.setattr(champion_module, "_live_incumbent", lambda game: incumbent)
    monkeypatch.setattr(champion_module, "_live_should_search", lambda game, incumbent: True)

    def fake_outcome(game, action, seed):
        if action == incumbent:
            return (0.40, 60)
        if action == challenger:
            return (0.60, 50)
        return (0.10, 40)

    monkeypatch.setattr(champion_module, "outcome", fake_outcome)
    assert live_champion(game, budget_ms=2000, max_futures=2) == incumbent


def test_live_champion_skip_and_environment_clamps(monkeypatch):
    game = Game(seed=41)
    incumbent = game.legal()[0] if game.legal() else None
    monkeypatch.setattr(champion_module, "_live_incumbent", lambda game: incumbent)
    monkeypatch.setattr(champion_module, "_live_should_search", lambda game, incumbent: False)
    stats = {}
    assert live_champion(game, stats=stats) == incumbent
    assert stats == {"searched": False, "futures": 0, "elapsed_ms": 0.0, "overridden": False}

    monkeypatch.setattr(champion_module, "_live_should_search", lambda game, incumbent: True)
    monkeypatch.setenv("QWIXX_LIVE_CHAMPION_BUDGET_MS", "99999")
    monkeypatch.setenv("QWIXX_LIVE_CHAMPION_MAX_FUTURES", "999")
    monkeypatch.setattr(champion_module, "outcome", lambda game, action, seed: (0.0, 0.0))
    stats = {}
    live_champion(game, stats=stats)
    assert stats["budget_ms"] == 2000.0
    assert stats["max_futures"] == 32


def test_live_should_search_non_four_done_no_legal_and_penalty(monkeypatch):
    three = Game(n=3, seed=42)
    assert champion_module._live_should_search(three, None) is False

    done = Game(seed=43)
    done.locks = {0, 1}
    done.phase = done.n
    assert champion_module._live_should_search(done, None) is False

    game = Game(seed=44)
    monkeypatch.setattr(game, "legal", lambda: [])
    assert champion_module._live_should_search(game, None) is False

    penalty = Game(seed=45)
    penalty.phase = penalty.n
    penalty.active_marks = 0
    assert champion_module._live_should_search(penalty, heuristic_v2(penalty)) is True
