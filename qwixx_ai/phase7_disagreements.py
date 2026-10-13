"""Phase 7: measure V2/imitation disagreements on held-out and learner states."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .core import Game, ORDER
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load


def category(game, action):
    board = game.boards[game.actor]
    if action is None:
        return "pass"
    color, number, locking = action
    previous = ORDER[color].index(board.rows[color][-1]) if board.rows[color] else -1
    gap = ORDER[color].index(number) - previous - 1
    if locking:
        return "row_lock"
    if game.turn >= 35:
        return "late_game"
    if gap >= 3:
        return "large_jump"
    if board.penalties >= 2:
        return "penalty_pressure"
    return "ordinary"


def collect(model, games=500, seed=710000, learner_fraction=0.5):
    """Generate whole games with V2 labels; never mix game IDs across splits."""
    if games < 2 or not 0 <= learner_fraction <= 1:
        raise ValueError("invalid collection settings")
    observations, masks, labels, ids = [], [], [], []
    buckets = Counter()
    disagreements = Counter()
    decisions = 0
    mismatches = 0
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        learner_game = game_id < round(games * learner_fraction)
        while not game.done:
            teacher = heuristic_v2(game)
            student = choose(model, game)
            bucket = category(game, teacher)
            buckets[bucket] += 1
            decisions += 1
            if student != teacher:
                mismatches += 1
                disagreements[bucket] += 1
            observations.append(observe(game))
            masks.append(mask(game))
            labels.append(encode(teacher))
            ids.append(game_id)
            game.step(student if learner_game else teacher)
    summary = {
        "games": games, "seed": seed, "learner_games": round(games * learner_fraction),
        "decisions": decisions, "disagreements": mismatches,
        "disagreement_rate": mismatches / decisions,
        "categories": {
            key: {"decisions": count,
                  "disagreements": disagreements[key],
                  "rate": disagreements[key] / count}
            for key, count in sorted(buckets.items())
        },
    }
    return {"observations": np.stack(observations),
            "masks": np.stack(masks),
            "actions": np.asarray(labels, dtype=np.int64),
            "game_ids": np.asarray(ids, dtype=np.int64)}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase6/imitation.pt")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=710000)
    parser.add_argument("--learner-fraction", type=float, default=0.5)
    parser.add_argument("--output", default="runs/phase7/disagreements.npz")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data, report = collect(load(args.model), args.games, args.seed,
                           args.learner_fraction)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(json.dumps({"dataset": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
