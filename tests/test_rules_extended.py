"""Additional independent Qwixx rule invariants and UI regressions."""
import random

import pytest

from qwixx_ai.core import Board, Game, ORDER
from qwixx_ai.web import HTML


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_phase_actor_order_and_turn_rotation(players):
    game = Game(n=players, seed=17)
    for turn in range(players * 2):
        active = game.active
        for phase in range(players + 1):
            assert game.phase == phase
            assert game.actor == (active if phase in (0, players) else (active + phase) % players)
            game.step(None)
        assert game.turn == turn + 1
        assert game.active == (active + 1) % players


@pytest.mark.parametrize("color", range(4))
@pytest.mark.parametrize("marks", range(0, 11))
def test_row_progression_and_lock_threshold(color, marks):
    board = Board()
    board.rows[color] = list(ORDER[color][:marks])
    for index, number in enumerate(ORDER[color]):
        expected = index > marks - 1 and (index != 10 or marks >= 5)
        assert board.can(color, number, set()) == expected
    assert not any(board.can(color, number, {color}) for number in ORDER[color])


@pytest.mark.parametrize("color", range(4))
@pytest.mark.parametrize("marks", range(0, 12))
@pytest.mark.parametrize("penalties", range(0, 5))
def test_score_formula_for_every_color(color, marks, penalties):
    board = Board()
    board.rows[color] = list(ORDER[color][:min(marks, 11)])
    board.penalties = penalties
    count = min(marks, 11)
    locked = count == 11
    expected = (count + int(locked)) * (count + int(locked) + 1) // 2 - 5 * penalties
    assert board.score({color} if locked else set()) == expected


@pytest.mark.parametrize("seed", range(25))
def test_random_legal_play_preserves_monotonic_rows_and_penalty_bounds(seed):
    game = Game(n=4, seed=seed)
    rng = random.Random(seed)
    for _ in range(1000):
        if game.done:
            break
        phase = game.phase
        active = game.active
        before = game.boards[active].penalties
        choices = game.legal()
        assert len(choices) == len(set(choices))
        for color, number, lock in choices:
            assert number in ORDER[color]
            assert lock == (number == ORDER[color][-1])
        choice = rng.choice([None, *choices])
        game.step(choice)
        for board in game.boards:
            for color, row in enumerate(board.rows):
                positions = [ORDER[color].index(number) for number in row]
                assert positions == sorted(set(positions))
        if phase == game.n:
            assert game.boards[active].penalties - before in (0, 1)
        else:
            assert game.boards[active].penalties == before
        assert all(board.penalties <= 4 for board in game.boards)
    else:
        pytest.fail("Game failed to terminate within 1000 legal actions")


def test_invalid_action_never_mutates_game():
    game = Game(seed=20)
    before = (game.phase, game.turn, game.actor, [list(row) for row in game.boards[0].rows])
    with pytest.raises(ValueError):
        game.step((0, 99, False))
    assert (game.phase, game.turn, game.actor, [list(row) for row in game.boards[0].rows]) == before


def test_no_strike_when_roller_marks_only_white_phase():
    game = Game(seed=21)
    game.dice = (1, 2, 3, 4, 5, 6)
    game.step((0, 3, False))
    for _ in range(game.n):
        game.step(None)
    assert game.boards[0].penalties == 0


def test_strike_when_roller_passes_both_phases_even_if_legal_moves_exist():
    game = Game(seed=22)
    game.dice = (1, 2, 3, 4, 5, 6)
    assert game.legal()
    for _ in range(game.n + 1):
        game.step(None)
    assert game.boards[0].penalties == 1


def test_white_dice_skip_has_no_confirmation():
    assert "Pass on the white dice total" not in HTML
    assert "function skipPhase(){if(!state||state.done||moving)return;act(null)}" in HTML


def test_training_rate_limit_is_separate_from_general_limit():
    from qwixx_ai.distributed import Handler
    from unittest.mock import patch

    class Client:
        client_address = ("192.0.2.10", 12345)
        rate_lock = Handler.rate_lock
        rate_buckets = {}

    with patch("qwixx_ai.distributed.time.monotonic", return_value=1000.0):
        for _ in range(120):
            assert Handler.limited(Client()) is False
        assert Handler.limited(Client()) is True
        for _ in range(600):
            assert Handler.limited(Client(), training=True) is False
        assert Handler.limited(Client(), training=True) is True
