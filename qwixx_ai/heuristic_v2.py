"""Probability-aware, deterministic Qwixx baseline.

This is an experimental heuristic, not a mathematically optimal policy.
It evaluates the marginal score, skipped spaces, reachable dice totals,
penalty exposure, and the effect of ending the game.
"""
from .core import ORDER

# Sum of two fair six-sided dice; available on passive turns.
SUM_WEIGHTS = (0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)


def probability(total):
    return SUM_WEIGHTS[total] / 36 if 2 <= total <= 12 else 0.0


def heuristic_v2(game):
    """Select a legal mark only when its estimated value exceeds passing."""
    legal = game.legal()
    if not legal:
        return None
    board = game.boards[game.actor]
    active = game.actor == game.active
    # On the active player's final action, passing without any mark incurs -5.
    penalty_if_pass = 5.0 if active and game.phase == game.n and game.active_marks == 0 else 0.0
    # Passing on the white action is not penalized yet: a colored action remains.
    pass_value = -penalty_if_pass

    def score(action):
        color, number, locking = action
        row = board.rows[color]
        order = ORDER[color]
        index = order.index(number)
        previous = order.index(row[-1]) if row else -1
        gap = index - previous - 1
        count = len(row)
        # Triangular scoring: the next mark earns count + 1 points.
        immediate = count + 1
        # Skipping a likely, playable number is costlier than skipping a rare one.
        lost = sum(0.9 + 4.5 * probability(order[j])
                   for j in range(previous + 1, index))
        # Reward future accessible totals without demanding uniform row progress.
        remaining = sum(probability(order[j]) for j in range(index + 1, 11))
        # Reduce the penalty for progressing when the game is already mature.
        pace = max(0.30, 1.0 - game.turn / 75.0)
        value = immediate - pace * lost + 1.8 * remaining
        # Locking scores a bonus X, but also ends that color for everyone.
        if locking:
            value += count + 2
            my_score = game.scores()[game.actor]
            opposing_best = max(game.scores()[i] for i in range(game.n)
                                if i != game.actor)
            value += 2.0 if my_score >= opposing_best else -3.0
            if len(game.locks) == 1:
                value += 3.0 if my_score >= opposing_best else -4.0
        # Late, wide jumps are less useful than their immediate points suggest.
        if gap >= 3 and game.turn < 35:
            value -= (gap - 2) * 0.8
        return value

    best = max(legal, key=lambda action: (score(action), -ORDER[action[0]].index(action[1]), -action[0]))
    return best if score(best) > pass_value else None
