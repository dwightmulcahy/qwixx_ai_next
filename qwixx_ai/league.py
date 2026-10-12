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
            minimum_share=0.30):
    result = head_to_head(challenger, champion, games, seed)
    result["promotion_threshold"] = minimum_share
    result["promoted"] = result["win_share"] >= minimum_share
    if result["promoted"]:
        save(challenger, Path(destination), win_share=result["win_share"],
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
