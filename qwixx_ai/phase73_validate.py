"""Phase 7.3: frozen-threshold ablations and independent paired validation.

Thresholds 2, 3, 4, 5 are exploratory. Choose a threshold using the
ablation results, then run a NEW seed for confirmation. Never treat
the threshold-selection sample as confirmatory evidence.
"""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase72_hybrid import large_jump


THRESHOLDS = (2, 3, 4, 5)


def selective(model, threshold=3):
    if threshold < 1:
        raise ValueError("threshold must be positive")

    def act(game):
        student = choose(model, game)
        teacher = heuristic_v2(game)
        if student == teacher or student is None or teacher is None:
            return teacher
        return student if large_jump(game, student, threshold=threshold) else teacher

    return act


def evaluate(model, rounds=2500, seed=773000, confidence=0.95,
             thresholds=THRESHOLDS, confirm=None):
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid evaluation settings")
    if confirm is not None and confirm not in THRESHOLDS:
        raise ValueError("confirmation threshold must be 2, 3, 4 or 5")
    if confirm is not None:
        # Exactly two prespecified comparisons; family-wise coverage.
        selected = (confirm,)
        phase = "confirmation"
    else:
        selected = tuple(thresholds)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("thresholds must be distinct and nonempty")
        if any(t not in THRESHOLDS for t in selected):
            raise ValueError("thresholds must be 2, 3, 4 or 5")
        phase = "exploratory_ablation"
    adjusted = 1 - (1 - confidence) / (2 * len(selected))
    student = lambda game: choose(model, game)
    results = {}
    for threshold in selected:
        candidate = selective(model, threshold)
        # Identical seed blocks across thresholds for interpretable ablations.
        results[str(threshold)] = {
            "against_v2": paired_compare(
                candidate, heuristic_v2, [heuristic_v2] * 3,
                rounds=rounds, seed=seed, confidence=adjusted),
            "against_phase7": paired_compare(
                candidate, student, [heuristic_v2] * 3,
                rounds=rounds, seed=seed, confidence=adjusted),
        }
    return {
        "phase": phase, "rounds": rounds, "seed": seed,
        "family_confidence": confidence, "comparison_confidence": adjusted,
        "results": results,
        "note": ("Exploratory selection only: use --confirm with a new seed "
                 "for an independently tested threshold."
                 if confirm is None else
                 "Confirmation is valid only if threshold and seed were "
                 "prespecified before inspecting these games. No promotion."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    parser.add_argument("--rounds", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=773000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--confirm", type=int, choices=THRESHOLDS)
    args = parser.parse_args()
    print(json.dumps(evaluate(load(args.model), args.rounds, args.seed,
                              args.confidence, confirm=args.confirm), indent=2))


if __name__ == "__main__":
    main()
