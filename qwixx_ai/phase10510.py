"""Phase 10.5.10: compare frozen champion, neural, and fixed-64.

Fresh, seat-rotated paired tournaments against the same Heuristic V2 lineup.
No automatic promotion.
"""
import argparse
import json
import time
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase105_verified import VerifiedPolicy


def evaluate(checkpoint, rounds=250, seed=10510001, confidence=0.95):
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("rounds >= 2 and confidence between 0 and 1 required")
    model = load(checkpoint)
    neural = lambda game: choose(model, game, epsilon=0)
    fixed = VerifiedPolicy(checkpoint, futures=64, seed=seed + 900000,
                           min_gain=0.05, confidence=confidence,
                           seats=(2, 3))
    opponents = [heuristic_v2] * 3
    started = time.perf_counter()
    # Distinct tournament seeds; each comparison remains paired within itself.
    comparisons = {
        "fixed64_vs_champion": paired_compare(
            fixed, champion, opponents, rounds, seed, confidence),
        "neural_vs_champion": paired_compare(
            neural, champion, opponents, rounds, seed + 100000, confidence),
        "fixed64_vs_neural": paired_compare(
            fixed, neural, opponents, rounds, seed + 200000, confidence),
    }
    s = fixed.stats
    n = s["verified"]
    stats = {
        "decisions": s["decisions"],
        "disagreements": s["disagreements"],
        "verified": n,
        "overrides": s["overrides"],
        "mean_verification_seconds": (
            s["verification_seconds"] / n if n else 0.0),
        "mean_futures": 64 if n else 0,
        "by_seat": s["by_seat"],
    }
    primary = comparisons["fixed64_vs_champion"]
    qualifies = (
        primary["advantage_ci_low"] > 0
        and primary["challenger_mean_score"]
        >= primary["reference_mean_score"])
    return {
        "phase": "10.5.10", "checkpoint": str(checkpoint),
        "rounds": rounds, "seed": seed, "confidence": confidence,
        "comparisons": comparisons, "fixed64_stats": stats,
        "elapsed_seconds": time.perf_counter() - started,
        "eligible_for_review": qualifies,
        "promoted": False,
        "note": "Exploratory paired comparisons, not automatic promotion. "
                "Normal-approximation confidence intervals and three "
                "comparisons do not establish familywise error control.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seed", type=int, default=10510001)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase105/phase10510_champion_comparison.json"))
    args = p.parse_args(argv)
    result = evaluate(args.candidate, args.rounds, args.seed,
                      args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
