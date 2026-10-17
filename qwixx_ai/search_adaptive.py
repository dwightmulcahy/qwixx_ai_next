"""Adaptive Monte Carlo Qwixx search with paired racing and a frozen incumbent.

Search compares pass and every legal mark on common future seeds, allocates
extra rollouts to unresolved moves, and uses the frozen champion as a safe
fallback. No model training, promotion or checkpoint mutation.
"""
import argparse
import json
import math
import random
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase9_search import outcome, state_seed


def select(game, initial=16, maximum=128, batch=16, seed=14000000,
           min_gain=0.03, confidence_z=1.96, stats=None):
    if initial < 2 or maximum < initial or batch < 1 or min_gain < 0 or confidence_z < 0:
        raise ValueError("invalid search settings")
    if game.n != 4 or game.done:
        return heuristic_v2(game)
    actions = [None, *game.legal()]
    if len(actions) == 1:
        return actions[0]
    incumbent = champion(game)
    rng = random.Random(state_seed(game, seed))
    totals = [0.0] * len(actions)
    score_totals = [0.0] * len(actions)
    # Paired differences against incumbent, for conservative selection.
    base_index = actions.index(incumbent)
    diffs = [[] for _ in actions]
    n = 0
    while True:
        target = min(maximum, initial if n == 0 else n + batch)
        for _ in range(n, target):
            future = rng.randrange(2**64)
            samples = [outcome(game, action, future) for action in actions]
            base_share = samples[base_index][0]
            for i, (share, score) in enumerate(samples):
                totals[i] += share
                score_totals[i] += score
                diffs[i].append(share - base_share)
        n = target
        viable = []
        for i in range(len(actions)):
            if i == base_index:
                continue
            avg = sum(diffs[i]) / n
            variance = (sum((d - avg) ** 2 for d in diffs[i]) / (n - 1))
            lower = avg - confidence_z * math.sqrt(variance / n)
            if lower > min_gain:
                viable.append(i)
        if viable:
            best = max(viable, key=lambda i: (totals[i] / n, score_totals[i] / n))
            chosen = actions[best]
            break
        if n >= maximum:
            chosen = incumbent
            break
        # If all candidates are convincingly inferior, stop early.
        if all(
            (sum(diffs[i]) / n + confidence_z * math.sqrt(
                (sum((d - sum(diffs[i]) / n) ** 2 for d in diffs[i]) / (n - 1)) / n
            ) < min_gain)
            for i in range(len(actions)) if i != base_index
        ):
            chosen = incumbent
            break
    if stats is not None:
        stats["decisions"] += 1
        stats["futures"] += n
        stats["overrides"] += chosen != incumbent
    return chosen


def policy(**kwargs):
    return lambda game: select(game, **kwargs)


def evaluate(rounds=12, seed=14020000, initial=16, maximum=64,
             batch=16, min_gain=0.03):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    settings = dict(initial=initial, maximum=maximum, batch=batch,
                    min_gain=min_gain)
    stats = {"decisions": 0, "futures": 0, "overrides": 0}
    candidate = policy(**settings, stats=stats)
    result = paired_compare(candidate, champion, [heuristic_v2] * 3,
                            rounds=rounds, seed=seed)
    return {
        "experiment": "adaptive_search", "settings": settings,
        "versus_phase91": result,
        "search_stats": stats,
        "mean_futures_per_decision": stats["futures"] / stats["decisions"]
        if stats["decisions"] else 0.0,
        "override_rate": stats["overrides"] / stats["decisions"]
        if stats["decisions"] else 0.0,
        "promoted": False,
        "note": "Exploratory only; paired normal CI is not a sequential-search "
                "guarantee. Test fresh seeds before promotion.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=14020000)
    parser.add_argument("--initial", type=int, default=16)
    parser.add_argument("--maximum", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--min-gain", type=float, default=0.03)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/search/adaptive_exploratory.json"))
    args = parser.parse_args(argv)
    report = evaluate(args.rounds, args.seed, args.initial, args.maximum,
                      args.batch, args.min_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
