"""Monte Carlo lookahead policy using V2 for simulated future decisions.

Rollouts reuse the current game's RNG state, so candidate moves see matched
future dice. This is a deterministic approximation, not an exact game solver.
"""
import copy

from .heuristic_v2 import heuristic_v2


def rollout(game, simulations=8, horizon=32):
    """Pick the action with the best approximate future outcome.

    A small horizon limits per-move CPU work for browser/server use.
    Set simulations=1 for deterministic single-future lookahead.
    """
    if simulations < 1 or horizon < 1:
        raise ValueError("simulations and horizon must be positive")
    choices = [None, *game.legal()]
    if len(choices) == 1:
        return None
    actor = game.actor
    initial_scores = game.scores()
    best_action = None
    best_value = float("-inf")
    for action in choices:
        total = 0.0
        for trial in range(simulations):
            future = copy.deepcopy(game)
            # Use distinct but matched future dice for each trial/action.
            if trial:
                future.rng.seed((future.rng.getrandbits(64) ^ (trial * 0x9E3779B97F4A7C15))
                                & ((1 << 64) - 1))
            future.step(action)
            steps = 0
            while not future.done and steps < horizon:
                future.step(heuristic_v2(future))
                steps += 1
            scores = future.scores()
            margin = scores[actor] - max(
                scores[i] for i in range(future.n) if i != actor
            )
            # Win-focused objective with a small score-difference tie-breaker.
            value = (1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)
            value += 0.015 * margin
            value += 0.002 * (scores[actor] - initial_scores[actor])
            total += value
        average = total / simulations
        if average > best_value:
            best_value = average
            best_action = action
    return best_action
