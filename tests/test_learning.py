"""Regression tests for Phase 1 correctness and Phase 2 learning signals."""

import numpy as np
import pytest

from qwixx_ai.core import Game
from qwixx_ai.env import QwixxEnv, observe, mask
from qwixx_ai.learning import DISCOUNT, SHAPING_WEIGHT, potential, shaped_reward, terminal_reward


@pytest.mark.parametrize(("scores", "seat", "expected"), [
    ([20, 20, 10, 5], 0, 0.5),
    ([20, 20, 10, 5], 1, 0.5),
    ([20, 20, 10, 5], 2, -1.0),
    ([10, 10, 10, 10], 3, 0.25),
    ([30, 20, 10, 0], 0, 1.0),
])
def test_terminal_rewards_split_ties(scores, seat, expected):
    assert terminal_reward(scores, seat) == expected


def test_seeded_environment_reset_sequence():
    first = QwixxEnv(seed=101)
    second = QwixxEnv(seed=101)
    for _ in range(4):
        a, info_a = first.reset()
        b, info_b = second.reset()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])
    a, _ = first.reset(seed=404)
    b, _ = second.reset(seed=404)
    np.testing.assert_array_equal(a, b)


def test_strategic_observations_are_finite_bounded_and_stable():
    game = Game(seed=19)
    initial = observe(game)
    assert initial.dtype == np.float32
    assert np.all(np.isfinite(initial))
    assert np.all((initial >= 0) & (initial <= 1))
    game.boards[0].rows[0] = [2, 3, 4, 5, 6]
    updated = observe(game)
    assert updated.shape == initial.shape
    assert not np.array_equal(updated, initial)
    assert mask(game).shape == (89,)


def test_potential_zero_at_terminal_and_reward_telescopes():
    game = Game(seed=4)
    p0 = potential(game, 0)
    game.boards[0].rows[0] = [2, 3, 4, 5, 6]
    p1 = potential(game, 0)
    assert p1 != p0
    game.boards[0].penalties = 4
    assert game.done
    assert potential(game, 0) == 0.0
    first = shaped_reward(0, p0, p1, False)
    second = shaped_reward(0, p1, 0, True)
    assert first + DISCOUNT * second == pytest.approx(-SHAPING_WEIGHT * p0)


def test_terminal_scoring_and_no_extra_actions():
    game = Game(seed=22)
    game.boards[0].penalties = 3
    for _ in range(game.n + 1):
        game.step(None)
    assert game.done
    assert game.boards[0].penalties == 4
    with pytest.raises(RuntimeError):
        game.step(None)
