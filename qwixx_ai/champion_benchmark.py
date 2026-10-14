"""Opt-in independent champion comparison against V2."""
import argparse
import json

from .champion import SETTINGS, champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2


def benchmark(rounds=1000, seed=996000, confidence=0.95):
    result = paired_compare(champion, heuristic_v2, [heuristic_v2] * 3, rounds=rounds, seed=seed, confidence=confidence)
    return {"champion": "phase91_search", "settings": SETTINGS, "against_v2": result, "passed": result["advantage_ci_low"] > 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=996000)
    args = parser.parse_args()
    report = benchmark(args.rounds, args.seed)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
