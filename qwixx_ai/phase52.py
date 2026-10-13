"""Phase 5.2: symmetric paired matches and multiplicity-aware promotion.

Each seed is replayed with both policies in the focal seat, against three
copies of the other policy. The statistic is their paired win-share difference.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from .core import Game, ORDER
from .model import choose, load, save


def _play(focal, other, seed, seat):
    game = Game(seed=seed)
    while not game.done:
        game.step(choose(focal if game.actor == seat else other, game))
    scores = game.scores()
    winners = sum(score == max(scores) for score in scores)
    share = 1 / winners if scores[seat] == max(scores) else 0.0
    board = game.boards[seat]
    return share, scores[seat], board.penalties, sum(
        bool(row and row[-1] == ORDER[c][-1])
        for c, row in enumerate(board.rows)
    )


def symmetric_evaluate(challenger, champion, rounds=750, seed=700000,
                       confidence=0.95):
    """Paired policy swap, with all four seats for each dice seed.

    Confidence interval uses seed-block paired differences; each block has
    eight games. It is a normal-approximation interval, not an exact test.
    """
    if rounds < 2:
        raise ValueError("rounds must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    # Normal quantile using standard-library statistics.
    from statistics import NormalDist

    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    differences = []
    challenger_shares = []
    champion_shares = []
    scores = []
    penalties = []
    locks = []
    by_seat = [0.0] * 4
    try:
        for i in range(rounds):
            block = 0.0
            for seat in range(4):
                a = _play(challenger, champion, seed + i, seat)
                b = _play(champion, challenger, seed + i, seat)
                difference = a[0] - b[0]
                block += difference / 4
                by_seat[seat] += difference
                challenger_shares.append(a[0])
                champion_shares.append(b[0])
                scores.append(a[1])
                penalties.append(a[2])
                locks.append(a[3])
            differences.append(block)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    delta = float(np.mean(differences))
    stderr = float(np.std(differences, ddof=1) / math.sqrt(rounds))
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    return {
        "games": rounds * 8,
        "seed_blocks": rounds,
        "seed": seed,
        "confidence": confidence,
        "challenger_win_share": float(np.mean(challenger_shares)),
        "champion_win_share": float(np.mean(champion_shares)),
        "paired_advantage": delta,
        "advantage_ci_low": delta - critical * stderr,
        "advantage_ci_high": delta + critical * stderr,
        "advantage_by_seat": [value / rounds for value in by_seat],
        "challenger_mean_score": float(np.mean(scores)),
        "challenger_mean_penalties": float(np.mean(penalties)),
        "challenger_mean_completed_rows": float(np.mean(locks)),
    }


def evaluate_promotion(challenger, champion, historical=(), rounds=750,
                       seed=700000, alpha=0.05, max_checks=10,
                       destination=None):
    """Bonferroni-corrected gate for a prespecified evaluation budget.

    The per-comparison two-sided interval is 1-alpha/(max_checks*(1+history)).
    Callers must not exceed max_checks in a training run. History must pass
    the same positive paired-advantage criterion as the current champion.
    """
    history = list(historical)
    if not 0 < alpha < 1 or max_checks < 1:
        raise ValueError("invalid alpha or max_checks")
    confidence = 1 - alpha / (max_checks * (1 + len(history)))
    primary = symmetric_evaluate(challenger, champion, rounds, seed, confidence)
    previous = [
        symmetric_evaluate(challenger, model, rounds,
                           seed + 100000 * (i + 1), confidence)
        for i, model in enumerate(history)
    ]
    passed = primary["advantage_ci_low"] > 0 and all(
        item["advantage_ci_low"] > 0 for item in previous
    )
    result = {
        "promoted": passed,
        "alpha": alpha,
        "max_checks": max_checks,
        "comparison_confidence": confidence,
        "primary": primary,
        "historical": previous,
    }
    if passed and destination is not None:
        save(challenger, Path(destination), promotion_seed=seed,
             promotion_games=primary["games"],
             paired_advantage=primary["paired_advantage"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("challenger")
    parser.add_argument("champion")
    parser.add_argument("--historical", nargs="*", default=[])
    parser.add_argument("--rounds", type=int, default=750)
    parser.add_argument("--seed", type=int, default=987654)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-checks", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(evaluate_promotion(
        load(args.challenger), load(args.champion),
        [load(path) for path in args.historical],
        rounds=args.rounds, seed=args.seed, alpha=args.alpha,
        max_checks=args.max_checks,
    ), indent=2))


if __name__ == "__main__":
    main()
