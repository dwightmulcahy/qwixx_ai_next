"""Matched-randomness Monte Carlo Qwixx policy with V2 fallback."""
import copy
import random

from .heuristic_v2 import heuristic_v2


def _share(scores, seat):
    top = max(scores)
    return (1 / sum(score == top for score in scores)
            if scores[seat] == top else 0.0)


def rollout_details(game, simulations=8, horizon=None, opponents=None,
                    seed=20261013, min_improvement=0.025):
    """Evaluate every legal action with matched future dice and full-game rollouts.

    horizon=None plays to game end; a finite horizon uses a V2 continuation
    estimate, not a premature terminal win/loss. Each candidate gets identical
    random states per trial. Opponents is a seat-indexed list of callables.
    """
    if simulations < 1 or (horizon is not None and horizon < 1):
        raise ValueError("simulations and horizon must be positive")
    if min_improvement < 0:
        raise ValueError("min_improvement must be nonnegative")
    if opponents is not None and len(opponents) != game.n:
        raise ValueError("opponents must have one policy per seat")
    policies = opponents if opponents is not None else [heuristic_v2] * game.n
    actor = game.actor
    choices = [None, *game.legal()]
    fallback = heuristic_v2(game)
    # All candidates receive the same future RNG state in each trial.
    generator = random.Random(seed)
    states = [random.Random(generator.getrandbits(64)).getstate()
              for _ in range(simulations)]
    totals = {a: 0.0 for a in choices}
    wins = {a: 0.0 for a in choices}
    for state in states:
        for action in choices:
            future = copy.deepcopy(game)
            future.rng.setstate(state)
            future.step(action)
            decisions = 0
            while not future.done and (horizon is None or decisions < horizon):
                future.step(policies[future.actor](future))
                decisions += 1
            scores = future.scores()
            if future.done:
                value = _share(scores, actor)
                wins[action] += value
            else:
                # Nonterminal evaluation avoids treating an early score lead as a win.
                margin = scores[actor] - max(
                    scores[i] for i in range(future.n) if i != actor)
                value = 0.5 + max(-0.4, min(0.4, margin / 100))
            totals[action] += value
    averages = {action: total / simulations for action, total in totals.items()}
    best = max(choices, key=lambda action: averages[action])
    # Small/noisy improvements defer to the proven deterministic baseline.
    selected = best if averages[best] > averages[fallback] + min_improvement else fallback
    return {
        "action": selected, "fallback": fallback,
        "changed": selected != fallback,
        "estimates": averages,
        "win_shares": {a: value / simulations for a, value in wins.items()},
        "simulations": simulations,
    }


def rollout(game, simulations=8, horizon=None, opponents=None,
            seed=20261013, min_improvement=0.025):
    return rollout_details(game, simulations, horizon, opponents,
                           seed, min_improvement)["action"]
