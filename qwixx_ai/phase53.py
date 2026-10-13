"""Phase 5.3: controlled-seat evaluation against a fixed opponent lineup."""
import argparse
import json
import math
import random
from statistics import NormalDist

import numpy as np

from .core import Game, ORDER
from .model import choose, load


def _match(focal, lineup, seed, seat):
    game = Game(seed=seed)
    while not game.done:
        actor = game.actor
        opponent = lineup[actor if actor < seat else actor - 1] if actor != seat else None
        game.step(choose(focal if actor == seat else opponent, game))
    scores = game.scores()
    share = (1 / sum(score == max(scores) for score in scores)
             if scores[seat] == max(scores) else 0.0)
    board = game.boards[seat]
    return share, scores[seat], board.penalties, sum(
        bool(row and row[-1] == ORDER[c][-1])
        for c, row in enumerate(board.rows)
    )


def controlled_evaluate(challenger, reference, lineup, rounds=750,
                        seed=700000, confidence=0.95):
    """Change only the focal policy, holding other three policies fixed.

    Each dice seed and seat produces two games. Paired differences are
    aggregated by seed block before computing a normal-approximation CI.
    """
    if len(lineup) != 3:
        raise ValueError("lineup must contain exactly three opponents")
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid rounds or confidence")
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    blocks, by_seat = [], [0.0] * 4
    challenger_shares, reference_shares = [], []
    scores, penalties, completed = [], [], []
    try:
        for index in range(rounds):
            block = 0.0
            for seat in range(4):
                a = _match(challenger, lineup, seed + index, seat)
                b = _match(reference, lineup, seed + index, seat)
                delta = a[0] - b[0]
                block += delta / 4
                by_seat[seat] += delta
                challenger_shares.append(a[0])
                reference_shares.append(b[0])
                scores.append(a[1])
                penalties.append(a[2])
                completed.append(a[3])
            blocks.append(block)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    advantage = float(np.mean(blocks))
    stderr = float(np.std(blocks, ddof=1) / math.sqrt(rounds))
    margin = NormalDist().inv_cdf((1 + confidence) / 2) * stderr
    return {
        "games": rounds * 8, "seed_blocks": rounds, "seed": seed,
        "confidence": confidence,
        "challenger_win_share": float(np.mean(challenger_shares)),
        "reference_win_share": float(np.mean(reference_shares)),
        "paired_advantage": advantage,
        "advantage_ci_low": advantage - margin,
        "advantage_ci_high": advantage + margin,
        "advantage_by_seat": [value / rounds for value in by_seat],
        "challenger_mean_score": float(np.mean(scores)),
        "challenger_mean_penalties": float(np.mean(penalties)),
        "challenger_mean_completed_rows": float(np.mean(completed)),
    }


def evaluate_promotion(challenger, champion, historical=(), rounds=750,
                       seed=700000, alpha=0.05, max_checks=10,
                       lineup=None, destination=None):
    """Compare against current and historical champions on identical lineups.

    Bonferroni adjustment covers planned promotion checks and comparisons.
    The caller must respect max_checks; reserve independent seeds for final
    validation, rather than repeatedly selecting using the same holdout.
    """
    from pathlib import Path
    from .model import save

    history = list(historical)
    if not 0 < alpha < 1 or max_checks < 1:
        raise ValueError("invalid alpha or max_checks")
    fixed = list(lineup) if lineup is not None else [champion] + (
        history[:2] if history else [champion, champion]
    )
    while len(fixed) < 3:
        fixed.append(champion)
    if len(fixed) != 3:
        raise ValueError("lineup must contain exactly three opponents")
    confidence = 1 - alpha / (max_checks * (1 + len(history)))
    primary = controlled_evaluate(challenger, champion, fixed, rounds, seed, confidence)
    previous = [
        controlled_evaluate(challenger, opponent, fixed, rounds,
                            seed + 100000 * (i + 1), confidence)
        for i, opponent in enumerate(history)
    ]
    promoted = primary["advantage_ci_low"] > 0 and all(
        item["advantage_ci_low"] > 0 for item in previous
    )
    result = {"promoted": promoted, "alpha": alpha, "max_checks": max_checks,
              "comparison_confidence": confidence, "primary": primary,
              "historical": previous, "lineup": "fixed_three"}
    if promoted and destination is not None:
        save(challenger, Path(destination), promotion_seed=seed,
             promotion_games=primary["games"],
             paired_advantage=primary["paired_advantage"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("challenger")
    parser.add_argument("champion")
    parser.add_argument("--historical", nargs="*", default=[])
    parser.add_argument("--lineup", nargs=3, metavar=("SEAT_A", "SEAT_B", "SEAT_C"))
    parser.add_argument("--rounds", type=int, default=750)
    parser.add_argument("--seed", type=int, default=987654)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-checks", type=int, default=1,
                        help="Planned checks; use 1 for a single prespecified final test")
    args = parser.parse_args()
    champion = load(args.champion)
    history = [load(path) for path in args.historical]
    lineup = [load(path) for path in args.lineup] if args.lineup else None
    print(json.dumps(evaluate_promotion(
        load(args.challenger), champion, history, rounds=args.rounds,
        seed=args.seed, alpha=args.alpha, max_checks=args.max_checks,
        lineup=lineup,
    ), indent=2))


if __name__ == "__main__":
    main()
