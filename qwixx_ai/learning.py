"""Shared learning rewards and outcome accounting."""

from .core import ORDER

DISCOUNT = 0.99
SHAPING_WEIGHT = 0.15


def terminal_reward(scores, seat):
    """Fractional win credit for ties; -1 for non-winning seats."""
    top = max(scores)
    winners = sum(score == top for score in scores)
    return 1.0 / winners if scores[seat] == top else -1.0


def potential(game, seat):
    """Bounded strategic potential: preserve row options and avoid penalties.

    Potential-based shaping uses a zero potential at terminal states.
    It does not award points directly for individual strategic features.
    """
    if game.done:
        return 0.0
    board = game.boards[seat]
    flexibility = 0.0
    readiness = 0.0
    for color, order in enumerate(ORDER):
        if color in game.locks:
            continue
        row = board.rows[color]
        last = order.index(row[-1]) if row else -1
        flexibility += (10 - last) / 11
        readiness += float(len(row) >= 5) / 4
    return (0.7 * flexibility / 4 + 0.3 * readiness
            - 0.2 * board.penalties / 4)


def shaped_reward(base, before, after, done, gamma=DISCOUNT):
    """Potential-based shaping with terminal potential fixed to zero."""
    return base + SHAPING_WEIGHT * ((0.0 if done else gamma * after) - before)
