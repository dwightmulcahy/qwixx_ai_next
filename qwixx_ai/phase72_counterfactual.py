"""Phase 7.2 stratified counterfactual diagnostics.

Sample distinct disagreement states across games; evaluate paired actions with
identical cloned states and future RNG seeds. Game-level bootstrap intervals
avoid treating multiple states in the same game as independent.
"""
import argparse
import copy
import json
import random
from collections import defaultdict

import numpy as np

from .core import Game
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase7_disagreements import category
from .phase71_counterfactual import finish


def group(game, teacher, student):
    if teacher is None and student is not None:
        return "v2_pass_student_mark"
    if student is None and teacher is not None:
        return "student_pass_v2_mark"
    return category(game, teacher)


def evaluate(model, games=1200, seed=762000, futures=12,
             per_category=150, bootstrap=1000):
    if min(games, futures, per_category, bootstrap) < 1:
        raise ValueError("invalid evaluation settings")
    rng = random.Random(seed)
    selected = defaultdict(list)
    # Reservoir sampling prevents early-game ordering from selecting all states.
    seen = defaultdict(int)
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        while not game.done:
            teacher = heuristic_v2(game)
            student = choose(model, game)
            if student != teacher:
                name = group(game, teacher, student)
                seen[name] += 1
                sample = (game_id, copy.deepcopy(game), student, teacher)
                if len(selected[name]) < per_category:
                    selected[name].append(sample)
                else:
                    index = rng.randrange(seen[name])
                    if index < per_category:
                        selected[name][index] = sample
            game.step(teacher)
    results = {}
    for name, states in sorted(selected.items()):
        game_effects = defaultdict(list)
        for game_id, game, student, teacher in states:
            diffs = []
            for _ in range(futures):
                future_seed = rng.randrange(2**32)
                a_share, a_score = finish(game, student, future_seed)
                b_share, b_score = finish(game, teacher, future_seed)
                diffs.append((a_share - b_share, a_score - b_score))
            game_effects[game_id].append(np.mean(diffs, axis=0))
        # Resample games, preserving any within-game dependence.
        by_game = [np.mean(values, axis=0) for values in game_effects.values()]
        values = np.asarray(by_game)
        generator = np.random.default_rng(seed + sum(map(ord, name)))
        estimates = np.empty((bootstrap, 2))
        for i in range(bootstrap):
            draw = generator.integers(0, len(values), len(values))
            estimates[i] = values[draw].mean(axis=0)
        lo, hi = np.quantile(estimates, [0.025, 0.975], axis=0)
        avg = values.mean(axis=0)
        results[name] = {
            "eligible_states": seen[name], "sampled_states": len(states),
            "distinct_games": len(game_effects),
            "student_minus_v2_win_share": float(avg[0]),
            "win_share_ci_95": [float(lo[0]), float(hi[0])],
            "student_minus_v2_score": float(avg[1]),
            "score_ci_95": [float(lo[1]), float(hi[1])],
        }
    return {
        "games": games, "seed": seed, "futures": futures,
        "max_states_per_category": per_category,
        "results": results,
        "warning": "Exploratory paired RNG rollouts: action-dependent game "
                   "length can desynchronize future dice. CIs are per category "
                   "and not multiplicity-adjusted. No automatic policy changes.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    parser.add_argument("--games", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=762000)
    parser.add_argument("--futures", type=int, default=12)
    parser.add_argument("--per-category", type=int, default=150)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(evaluate(load(args.model), args.games, args.seed,
                              args.futures, args.per_category,
                              args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
