"""Champion league evaluation and conservative promotion."""
import copy
import random
from pathlib import Path

import numpy as np
from .core import Game, ORDER
from .model import choose, load, save


def freeze(model):
    return copy.deepcopy(model).eval()


def head_to_head(challenger, champion, games=1000, seed=700000):
    """One challenger versus three champion copies, with rotated seats.

    Equal-strength expectation is 0.25 win share, not 0.50.
    """
    if games < 1:
        raise ValueError("games must be positive")
    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    wins = score = penalties = locks = 0.0
    try:
        for i in range(games):
            game = Game(seed=seed + i)
            seat = i % 4
            while not game.done:
                model = challenger if game.actor == seat else champion
                game.step(choose(model, game))
            scores = game.scores()
            winners = sum(s == max(scores) for s in scores)
            if scores[seat] == max(scores):
                wins += 1 / winners
            score += scores[seat]
            penalties += game.boards[seat].penalties
            locks += sum(bool(game.boards[seat].rows[c] and
                              len(game.boards[seat].rows[c]) >= 6 and
                              game.boards[seat].rows[c][-1] ==
                              ORDER[c][-1])
                         for c in range(4))
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    return {
        "games": games, "seed": seed, "win_share": wins / games,
        "mean_score": score / games,
        "mean_penalties": penalties / games, "mean_completed_rows": locks / games,
        "opponent": "three_champions",
    }


def promote(challenger, champion, destination, games=1000, seed=700000,
            minimum_share=0.28, batches=3):
    """Promote only after consistently exceeding the equal-strength baseline.

    All batches must exceed 25% and their pooled share must reach minimum_share.
    This is a practical gate, not a statistical confidence test.
    """
    if batches < 2 or games < batches:
        raise ValueError("promotion requires at least two nonempty batches")
    if not 0.25 < minimum_share <= 1:
        raise ValueError("minimum_share must exceed the four-seat baseline")
    sizes = [games // batches + (i < games % batches) for i in range(batches)]
    results = [head_to_head(challenger, champion, size, seed + 100000 * i)
               for i, size in enumerate(sizes)]
    share = sum(r["win_share"] * r["games"] for r in results) / games
    result = {"games": games, "seed": seed, "win_share": share,
              "promotion_threshold": minimum_share,
              "batch_win_shares": [r["win_share"] for r in results],
              "mean_score": sum(r["mean_score"] * r["games"] for r in results) / games,
              "mean_penalties": sum(r["mean_penalties"] * r["games"] for r in results) / games,
              "mean_completed_rows": sum(r["mean_completed_rows"] * r["games"] for r in results) / games,
              "opponent": "three_champions"}
    result["promoted"] = share >= minimum_share and all(
        r["win_share"] > 0.25 for r in results)
    if result["promoted"]:
        save(challenger, Path(destination), win_share=share,
             promotion_games=games, promotion_seed=seed)
    return result


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("challenger")
    parser.add_argument("champion")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=700000)
    args = parser.parse_args()
    print(json.dumps(head_to_head(load(args.challenger), load(args.champion),
                                  args.games, args.seed), indent=2))


if __name__ == "__main__":
    main()
