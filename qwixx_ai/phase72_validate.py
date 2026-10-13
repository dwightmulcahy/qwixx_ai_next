"""Phase 7.2 controlled hybrid tournament with multiplicity-adjusted intervals."""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import load
from .phase72_hybrid import policy


def evaluate(model, rounds=1250, seed=752000, confidence=0.95):
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid evaluation settings")
    modes = ("pass_correction", "large_jump", "selective")
    # Two references per mode; Bonferroni family-wise coverage.
    adjusted = 1 - (1 - confidence) / (2 * len(modes))
    student = policy(model, "student")
    results = {}
    for index, mode in enumerate(modes):
        challenger = policy(model, mode)
        results[mode] = {
            "against_student": paired_compare(
                challenger, student, [heuristic_v2] * 3,
                rounds=rounds, seed=seed + index * 100000,
                confidence=adjusted),
            "against_v2": paired_compare(
                challenger, heuristic_v2, [heuristic_v2] * 3,
                rounds=rounds, seed=seed + index * 100000 + 50000,
                confidence=adjusted),
        }
    return {
        "rounds": rounds, "seed": seed,
        "family_confidence": confidence,
        "comparison_confidence": adjusted,
        "results": results,
        "note": "Advisory only. No automatic promotion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=752000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(evaluate(load(args.model), args.rounds, args.seed,
                              args.confidence), indent=2))


if __name__ == "__main__":
    main()
