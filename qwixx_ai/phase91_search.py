"""Phase 9.1: conservative, selective Monte Carlo overrides of V2.

Frozen experimental defaults; exploratory only. Search is restricted to
multi-action middle-game decisions and requires both a win-share margin
and nonnegative expected score difference over V2.
"""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase9_search import evaluate_actions


def eligible(game, teacher, start_turn=12, end_turn=35):
    """Avoid low-choice and early/late states where search is costly or noisy."""
    return (
        not game.done
        and start_turn <= game.turn < end_turn
        and teacher is not None
        and len(game.legal()) >= 2
    )


def decide(game, futures=32, seed=900000, min_gain=0.10,
           min_score_gain=0.0, start_turn=12, end_turn=35):
    """Return (action, reason); never override without both safeguards."""
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


def validate(rounds=25, seed=992000, futures=32, search_seed=900000,
             min_gain=0.10, min_score_gain=0.0,
             start_turn=12, end_turn=35, confidence=0.95):
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid validation settings")
    stats = {
        "decisions": 0, "v2_not_eligible": 0,
        "v2_safeguard": 0, "override": 0,
    }
    candidate = policy(
        futures, search_seed, min_gain, min_score_gain,
        start_turn, end_turn, stats,
    )
    result = paired_compare(
        candidate, heuristic_v2, [heuristic_v2] * 3,
        rounds=rounds, seed=seed, confidence=confidence,
    )
    return {
        "phase": "9.1", "rounds": rounds, "seed": seed,
        "futures": futures, "search_seed": search_seed,
        "min_gain": min_gain, "min_score_gain": min_score_gain,
        "turn_window": [start_turn, end_turn],
        "decisions": stats,
        "override_rate": stats["override"] / stats["decisions"]
        if stats["decisions"] else 0.0,
        "eligible_rate": (
            (stats["v2_safeguard"] + stats["override"])
            / stats["decisions"] if stats["decisions"] else 0.0
        ),
        "against_v2": result,
        "warning": "Exploratory, not promotion evidence. Continuation RNG "
                   "streams may diverge. Freeze settings before a fresh "
                   "independent confirmation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=992000)
    parser.add_argument("--futures", type=int, default=32)
    parser.add_argument("--search-seed", type=int, default=900000)
    parser.add_argument("--min-gain", type=float, default=0.10)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--start-turn", type=int, default=12)
    parser.add_argument("--end-turn", type=int, default=35)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(validate(
        rounds=args.rounds, seed=args.seed, futures=args.futures,
        search_seed=args.search_seed, min_gain=args.min_gain,
        min_score_gain=args.min_score_gain,
        start_turn=args.start_turn, end_turn=args.end_turn,
        confidence=args.confidence,
    ), indent=2))


if __name__ == "__main__":
    main()
