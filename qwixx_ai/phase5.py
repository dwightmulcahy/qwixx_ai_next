"""Phase 5: paired-seat champion evaluation and uncertainty-aware promotion."""
import argparse
import json
import math
import random

import numpy as np

from .core import Game, ORDER
from .model import choose, load, save


def paired_evaluate(challenger, opponents, rounds=250, seed=700000):
    """Evaluate each dice seed in all four seats; one challenger per game.

    Each four-game block shares a seed. Confidence bounds are computed across
    independent seed blocks, not individual games.
    """
    if rounds < 2:
        raise ValueError("at least two seed blocks are required")
    if not opponents:
        raise ValueError("at least one opponent is required")
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    blocks = []
    scores = []
    penalties = []
    completed = []
    by_seat = [0.0] * 4
    try:
        for i in range(rounds):
            opponent = opponents[i % len(opponents)]
            block = 0.0
            for seat in range(4):
                game = Game(seed=seed + i)
                while not game.done:
                    game.step(choose(challenger if game.actor == seat else opponent, game))
                result = game.scores()
                winners = sum(s == max(result) for s in result)
                share = 1 / winners if result[seat] == max(result) else 0.0
                block += share / 4
                by_seat[seat] += share
                scores.append(result[seat])
                board = game.boards[seat]
                penalties.append(board.penalties)
                completed.append(sum(
                    bool(row and row[-1] == ORDER[c][-1])
                    for c, row in enumerate(board.rows)
                ))
            blocks.append(block)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    mean = float(np.mean(blocks))
    stderr = float(np.std(blocks, ddof=1) / math.sqrt(rounds))
    return {
        "games": rounds * 4,
        "seed_blocks": rounds,
        "seed": seed,
        "win_share": mean,
        "ci95_low": max(0.0, mean - 1.96 * stderr),
        "ci95_high": min(1.0, mean + 1.96 * stderr),
        "mean_score": float(np.mean(scores)),
        "mean_penalties": float(np.mean(penalties)),
        "mean_completed_rows": float(np.mean(completed)),
        "win_share_by_seat": [value / rounds for value in by_seat],
        "opponents": len(opponents),
    }


def evaluate_promotion(challenger, champion, historical=(), rounds=750,
                       seed=700000, minimum_share=0.25, destination=None):
    """Require lower confidence bound above parity against current champion.

    Historical champions are separately evaluated and reported as a regression
    guard: a challenger must also clear parity on average against each one.
    """
    if not 0.25 <= minimum_share < 1:
        raise ValueError("minimum_share must be between 0.25 and 1")
    primary = paired_evaluate(challenger, [champion], rounds, seed)
    history = [
        paired_evaluate(challenger, [model], rounds, seed + 100000 * (i + 1))
        for i, model in enumerate(historical)
    ]
    promoted = (primary["ci95_low"] > minimum_share
                and all(item["win_share"] >= 0.25 for item in history))
    result = {"promoted": promoted, "promotion_threshold": minimum_share,
              "primary": primary, "historical": history}
    if promoted and destination is not None:
        save(challenger, destination, promotion_seed=seed,
             promotion_games=primary["games"], win_share=primary["win_share"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("challenger")
    parser.add_argument("champion")
    parser.add_argument("--historical", nargs="*", default=[])
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=987654)
    parser.add_argument("--minimum-share", type=float, default=0.25)
    args = parser.parse_args()
    result = evaluate_promotion(
        load(args.challenger), load(args.champion),
        [load(path) for path in args.historical],
        rounds=args.rounds, seed=args.seed, minimum_share=args.minimum_share,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
