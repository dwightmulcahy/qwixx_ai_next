"""Phase 9: outcome-guided Monte Carlo action search over a V2 rollout policy.

The search uses a fixed, state-derived set of future RNG seeds shared across
candidate actions. The focal player alone searches; simulated continuations
use a frozen non-recursive hybrid rollout policy for every seat. This is a search policy, not a trained checkpoint.
"""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .runtime_search import (evaluate_actions, outcome as outcome, state_seed as state_seed)


def search(game, futures=16, seed=900000, min_gain=0.0):
    """Only override V2 for a strictly better estimated win share.

    Mean score is a tie-breaker among alternatives, not an override criterion.
    A positive min_gain can reduce noise-induced overrides.
    """
    if min_gain < 0:
        raise ValueError("min_gain must be nonnegative")
    teacher = heuristic_v2(game)
    estimates = evaluate_actions(game, futures=futures, seed=seed)
    baseline = next(item for item in estimates if item["action"] == teacher)
    best = max(estimates, key=lambda item: (
        item["win_share"], item["mean_score"],
        item["action"] == teacher,
    ))
    if best["win_share"] > baseline["win_share"] + min_gain:
        return best["action"]
    return teacher


def policy(futures=16, seed=900000, min_gain=0.0):
    def act(game):
        return search(game, futures=futures, seed=seed, min_gain=min_gain)
    return act


def validate(rounds=100, seed=990000, futures=8,
             search_seed=900000, min_gain=0.0, confidence=0.95):
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid validation settings")
    result = paired_compare(
        policy(futures=futures, seed=search_seed, min_gain=min_gain),
        heuristic_v2, [heuristic_v2] * 3, rounds=rounds, seed=seed,
        confidence=confidence,
    )
    return {
        "policy": "monte_carlo_v2_rollout", "rounds": rounds,
        "seed": seed, "futures": futures, "search_seed": search_seed,
        "min_gain": min_gain, "against_v2": result,
        "warning": "Exploratory only. Common RNG seeds may diverge across "
                   "action-dependent game lengths. Search is expensive; "
                   "do not promote without fresh confirmation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=990000)
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=900000)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(validate(args.rounds, args.seed, args.futures,
                              args.search_seed, args.min_gain,
                              args.confidence), indent=2))


if __name__ == "__main__":
    main()
