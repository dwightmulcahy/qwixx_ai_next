"""Held-out Phase 10.5 correction evaluation; never promotes a model."""
import argparse
import json

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase105_correction import corrected_policy


def evaluate(base, correction, rounds=250, seed=2801000, margin=1.5):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    baseline = load(base)
    frozen = lambda game: choose(baseline, game, epsilon=0)
    candidate = corrected_policy(base, correction, margin=margin)
    opponents = [heuristic_v2] * 3
    against_original = paired_compare(candidate, frozen, opponents,
                                      rounds=rounds, seed=seed)
    against_champion = paired_compare(candidate, champion, opponents,
                                      rounds=rounds, seed=seed+100000)
    against_v2 = paired_compare(candidate, heuristic_v2, opponents,
                                rounds=rounds, seed=seed+200000)
    eligible = (against_original["advantage_ci_low"] > 0
                and against_champion["advantage_ci_low"] > 0
                and against_v2["advantage_ci_low"] > 0
                and against_champion["challenger_mean_score"] >=
                against_champion["reference_mean_score"])
    return {"phase": "10.5", "versus_original": against_original,
            "versus_champion": against_champion, "versus_v2": against_v2,
            "eligible_for_review": eligible, "promoted": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--correction", required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seed", type=int, default=2801000)
    p.add_argument("--margin", type=float, default=1.5)
    args = p.parse_args(argv)
    print(json.dumps(evaluate(args.base, args.correction, args.rounds,
                              args.seed, args.margin), indent=2))


if __name__ == "__main__":
    main()
