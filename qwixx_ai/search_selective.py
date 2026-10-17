"""Efficient selective search: reuse the Phase 9.1 sample budget.

Only search Phase 9.1-eligible positions. The first 32 futures reproduce its
baseline decision without calling champion(); extra futures are spent only
when action comparisons are close. Optional neural continuation experiments
are separate from the default V2 continuation.
"""
import argparse
import json
import random
from pathlib import Path

from .champion import SETTINGS, champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase9_search import outcome, state_seed
from .phase91_search import eligible


def choose_from_estimates(actions, wins, scores, count, teacher,
                          min_gain=0.10, min_score_gain=0.0):
    base = actions.index(teacher)
    viable = [
        i for i, action in enumerate(actions)
        if action != teacher
        and wins[i] >= wins[base] + min_gain * count
        and scores[i] >= scores[base] + min_score_gain * count
    ]
    return (actions[max(viable, key=lambda i: (wins[i], scores[i]))]
            if viable else teacher)


def select(game, initial=32, maximum=64, seed=900000,
           min_gain=0.10, min_score_gain=0.0, ambiguity=0.06, stats=None):
    if initial < 1 or maximum < initial or min_gain < 0 or min_score_gain < 0 or ambiguity < 0:
        raise ValueError("invalid search settings")
    teacher = heuristic_v2(game)
    if game.n != 4 or not eligible(game, teacher, SETTINGS["start_turn"], SETTINGS["end_turn"]):
        if stats is not None:
            stats["skipped"] += 1
        return teacher
    actions = [None, *game.legal()]
    rng = random.Random(state_seed(game, seed))
    wins = [0.0] * len(actions)
    scores = [0.0] * len(actions)

    def sample(amount):
        for _ in range(amount):
            future = rng.randrange(2**64)
            for i, action in enumerate(actions):
                share, score = outcome(game, action, future)
                wins[i] += share
                scores[i] += score

    sample(initial)
    base = actions.index(teacher)
    # Refine only when an alternative is near the champion override threshold.
    near = any(
        abs((wins[i] - wins[base]) / initial - min_gain) <= ambiguity
        and scores[i] >= scores[base] - ambiguity * initial * 25
        for i in range(len(actions)) if i != base
    )
    used = initial
    if near and maximum > initial:
        sample(maximum - initial)
        used = maximum
    selected = choose_from_estimates(
        actions, wins, scores, used, teacher, min_gain, min_score_gain)
    if stats is not None:
        stats["searched"] += 1
        stats["futures"] += used
        stats["refined"] += used > initial
        stats["overrides"] += selected != teacher
    return selected


def policy(**kwargs):
    return lambda game: select(game, **kwargs)


def evaluate(rounds=12, seed=15020000, initial=32, maximum=64,
             ambiguity=0.06):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    stats = {"skipped": 0, "searched": 0, "futures": 0,
             "refined": 0, "overrides": 0}
    candidate = policy(initial=initial, maximum=maximum,
                       ambiguity=ambiguity, stats=stats)
    comparison = paired_compare(candidate, champion, [heuristic_v2] * 3,
                                rounds=rounds, seed=seed)
    return {
        "experiment": "selective_search", "settings": {
            "initial": initial, "maximum": maximum, "ambiguity": ambiguity},
        "versus_phase91": comparison, "search_stats": stats,
        "mean_futures_per_searched_decision": (
            stats["futures"] / stats["searched"] if stats["searched"] else 0),
        "promoted": False,
        "note": "V2 continuations retained. This optimizes search allocation; "
                "it does not yet improve rollout policy or reuse transpositions.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=15020000)
    parser.add_argument("--initial", type=int, default=32)
    parser.add_argument("--maximum", type=int, default=64)
    parser.add_argument("--ambiguity", type=float, default=0.06)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/search/selective_exploratory.json"))
    args = parser.parse_args(argv)
    result = evaluate(args.rounds, args.seed, args.initial,
                      args.maximum, args.ambiguity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
