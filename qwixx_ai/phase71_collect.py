"""Collect fresh learner-state examples with disagreement and jump weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import Game, ORDER
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load


def collect(model, games=1000, seed=730000, learner_fraction=0.75):
    if games < 2 or not 0 <= learner_fraction <= 1:
        raise ValueError("invalid collection settings")
    rows = {"observations": [], "masks": [], "actions": [],
            "game_ids": [], "weights": []}
    counts = {"decisions": 0, "disagreements": 0, "large_jumps": 0}
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        learner_game = game_id < round(games * learner_fraction)
        while not game.done:
            teacher = heuristic_v2(game)
            student = choose(model, game)
            jump = False
            if teacher is not None:
                color, number, _ = teacher
                row = game.boards[game.actor].rows[color]
                previous = ORDER[color].index(row[-1]) if row else -1
                jump = ORDER[color].index(number) - previous - 1 >= 3
            disagree = student != teacher
            counts["decisions"] += 1
            counts["disagreements"] += int(disagree)
            counts["large_jumps"] += int(jump)
            rows["observations"].append(observe(game))
            rows["masks"].append(mask(game))
            rows["actions"].append(encode(teacher))
            rows["game_ids"].append(game_id)
            rows["weights"].append(2.0 if disagree else (1.5 if jump else 1.0))
            game.step(student if learner_game else teacher)
    return {
        "observations": np.asarray(rows["observations"], dtype=np.float32),
        "masks": np.asarray(rows["masks"], dtype=np.bool_),
        "actions": np.asarray(rows["actions"], dtype=np.int64),
        "game_ids": np.asarray(rows["game_ids"], dtype=np.int64),
        "weights": np.asarray(rows["weights"], dtype=np.float32),
    }, {"games": games, "seed": seed, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=730000)
    parser.add_argument("--learner-fraction", type=float, default=0.75)
    parser.add_argument("--output", default="runs/phase7_1/weighted.npz")
    args = parser.parse_args()
    dataset, report = collect(load(args.model), args.games, args.seed,
                              args.learner_fraction)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    print(json.dumps({"dataset": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
