"""Matched-future-rollout diagnostics for V2 versus candidate disagreements.

This is an exploratory estimate, NOT an action oracle or promotion criterion.
Both actions start from an identical cloned state and identical future dice RNG.
"""
import argparse
import copy
import json
import random
from collections import defaultdict

from .core import Game
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase7_disagreements import category


def finish(game, action, future_seed):
    game = copy.deepcopy(game)
    seat = game.actor
    game.rng = random.Random(future_seed)
    game.step(action)
    while not game.done:
        game.step(heuristic_v2(game))
    scores = game.scores()
    highest = max(scores)
    share = 1 / sum(score == highest for score in scores) if scores[seat] == highest else 0
    return float(share), float(scores[seat])


def evaluate(model, games=100, seed=740000, futures=8, max_states=200):
    if games < 1 or futures < 1 or max_states < 1:
        raise ValueError("invalid evaluation settings")
    rng = random.Random(seed)
    samples = defaultdict(list)
    inspected = 0
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        while not game.done:
            teacher = heuristic_v2(game)
            student = choose(model, game)
            if teacher != student and inspected < max_states:
                group = category(game, teacher)
                deltas = []
                score_deltas = []
                for _ in range(futures):
                    future_seed = rng.randrange(2**32)
                    a_share, a_score = finish(game, student, future_seed)
                    b_share, b_score = finish(game, teacher, future_seed)
                    deltas.append(a_share - b_share)
                    score_deltas.append(a_score - b_score)
                samples[group].append((sum(deltas) / futures,
                                       sum(score_deltas) / futures))
                inspected += 1
            game.step(teacher)
    return {
        "games": games, "sampled_disagreements": inspected,
        "futures_per_action": futures,
        "categories": {
            name: {
                "states": len(values),
                "mean_student_minus_v2_win_share": sum(v[0] for v in values) / len(values),
                "mean_student_minus_v2_score": sum(v[1] for v in values) / len(values),
            }
            for name, values in sorted(samples.items())
        },
        "warning": "Exploratory; sequential RNG streams can diverge after actions "
                   "change game length. No automatic action changes.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=740000)
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(evaluate(load(args.model), args.games, args.seed,
                              args.futures, args.max_states), indent=2))


if __name__ == "__main__":
    main()
