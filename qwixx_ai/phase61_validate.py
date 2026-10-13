"""Independent Phase 6.1 validation and conservative promotion recommendation."""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load


def evaluate(candidate, imitation, rounds=1250, seed=876543,
             alpha=0.05, planned_checks=1):
    if not 0 < alpha < 1 or planned_checks < 1:
        raise ValueError("invalid promotion parameters")
    confidence = 1 - alpha / (2 * planned_checks)
    challenger = lambda g: choose(candidate, g)
    reference = lambda g: choose(imitation, g)
    lineup = [heuristic_v2] * 3
    against_imitation = paired_compare(
        challenger, reference, lineup, rounds, seed, confidence)
    against_v2 = paired_compare(
        challenger, heuristic_v2, lineup, rounds, seed + 100000, confidence)
    # Both comparisons must have a strictly positive lower bound.
    promoted = (against_imitation["advantage_ci_low"] > 0
                and against_v2["advantage_ci_low"] > 0)
    return {"promotion_eligible": promoted,
            "comparison_confidence": confidence,
            "against_imitation": against_imitation,
            "against_v2": against_v2,
            "note": "Advisory only. No automatic checkpoint replacement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="runs/phase6_1/latest.pt")
    parser.add_argument("--imitation", default="runs/phase6/imitation.pt")
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=876543)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--planned-checks", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(evaluate(load(args.candidate), load(args.imitation),
                              args.rounds, args.seed, args.alpha,
                              args.planned_checks), indent=2))


if __name__ == "__main__":
    main()
