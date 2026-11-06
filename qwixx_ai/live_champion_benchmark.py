"""Benchmark bounded live Champion against the frozen 0.8-era Champion."""
import argparse
import json

from .champion import base_champion, live_champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark(rounds=8, seed=18000000, budget_ms=None, max_futures=None):
    latencies = []
    futures = []

    def candidate(game):
        stats = {}
        action = live_champion(
            game,
            budget_ms=budget_ms,
            max_futures=max_futures,
            stats=stats,
        )
        if stats.get("searched"):
            latencies.append(stats["elapsed_ms"])
            futures.append(stats["futures"])
        return action

    result = paired_compare(
        candidate,
        base_champion,
        [heuristic_v2] * 3,
        rounds=rounds,
        seed=seed,
    )
    return {
        "candidate": "bounded_live_champion",
        "reference": "frozen_v0.8_champion",
        "comparison": result,
        "searched_decisions": len(latencies),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "mean_futures": sum(futures) / len(futures) if futures else 0.0,
        "settings": {
            "budget_ms": budget_ms,
            "max_futures": max_futures,
        },
        "note": (
            "Use enough rounds for strength conclusions. The reference policy is "
            "intentionally slow and is for offline validation only."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=18000000)
    parser.add_argument("--budget-ms", type=float)
    parser.add_argument("--max-futures", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(
        benchmark(args.rounds, args.seed, args.budget_ms, args.max_futures),
        indent=2,
    ))


if __name__ == "__main__":
    main()
