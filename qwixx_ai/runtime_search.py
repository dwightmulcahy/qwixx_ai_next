"""Runtime Monte Carlo helpers used by live Champion play.

This module contains only production decision logic. Historical phase scripts
import these primitives for reproducibility, but coverage for this module
reflects the deployed runtime path rather than experiment CLI wrappers.
"""
import hashlib
import random

from .heuristic_v2 import heuristic_v2
from .rollout_policy import rollout_policy


def state_seed(game, seed):
    """Stable seed independent of Python's randomized hash or game RNG state."""
    payload = (
        seed, game.turn, game.phase, game.active, game.active_marks,
        tuple(game.dice), tuple(sorted(game.locks)),
        tuple(sorted(game.white_locks)),
        tuple((tuple(tuple(row) for row in board.rows), board.penalties)
              for board in game.boards),
    )
    return int.from_bytes(
        hashlib.blake2b(repr(payload).encode(), digest_size=8).digest(),
        "big",
    )


def outcome(game, action, future_seed):
    """Simulate one matched future and return focal win share and score."""
    trial = game.clone()
    seat = trial.actor
    trial.rng = random.Random(future_seed)
    trial.step(action)
    while not trial.done:
        trial.step(rollout_policy(trial))
    scores = trial.scores()
    best = max(scores)
    share = (
        1 / sum(score == best for score in scores)
        if scores[seat] == best else 0.0
    )
    return share, scores[seat]


def evaluate_actions(game, futures=16, seed=900000):
    """Evaluate pass and every legal mark using matched future seeds."""
    if futures < 1:
        raise ValueError("futures must be positive")
    if game.done:
        raise ValueError("cannot search a finished game")
    actions = [None, *game.legal()]
    rng = random.Random(state_seed(game, seed))
    seeds = [rng.randrange(2**64) for _ in range(futures)]
    results = []
    for action in actions:
        samples = [outcome(game, action, future) for future in seeds]
        results.append({
            "action": action,
            "win_share": sum(sample[0] for sample in samples) / futures,
            "mean_score": sum(sample[1] for sample in samples) / futures,
        })
    return results


def eligible(game, teacher, start_turn=12, end_turn=35):
    """Return whether conservative runtime search is worth considering."""
    return (
        not game.done
        and start_turn <= game.turn < end_turn
        and teacher is not None
        and len(game.legal()) >= 2
    )


def decide(game, futures=32, seed=900000, min_gain=0.10,
           min_score_gain=0.0, start_turn=12, end_turn=35):
    """Conservative V2 override used by the frozen Champion baseline."""
    if futures < 1 or min_gain < 0 or min_score_gain < 0:
        raise ValueError("invalid search parameters")
    if not 0 <= start_turn < end_turn:
        raise ValueError("invalid turn window")
    teacher = heuristic_v2(game)
    if not eligible(game, teacher, start_turn, end_turn):
        return teacher, "v2_not_eligible"
    estimates = evaluate_actions(game, futures=futures, seed=seed)
    baseline = next(item for item in estimates if item["action"] == teacher)
    viable = [
        item for item in estimates
        if item["action"] != teacher
        and item["win_share"] >= baseline["win_share"] + min_gain
        and item["mean_score"] >= baseline["mean_score"] + min_score_gain
    ]
    if not viable:
        return teacher, "v2_safeguard"
    best = max(viable, key=lambda item: (
        item["win_share"], item["mean_score"],
    ))
    return best["action"], "override"


def policy(futures=32, seed=900000, min_gain=0.10,
           min_score_gain=0.0, start_turn=12, end_turn=35, stats=None):
    """Build the conservative runtime search policy."""
    def act(game):
        action, reason = decide(
            game, futures, seed, min_gain, min_score_gain,
            start_turn, end_turn,
        )
        if stats is not None:
            stats["decisions"] += 1
            stats[reason] += 1
        return action
    return act
