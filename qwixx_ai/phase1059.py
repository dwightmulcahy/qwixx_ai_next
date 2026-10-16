"""Phase 10.5.9: paired holdout-64 versus fixed-64 evaluation."""
import argparse
import json
from pathlib import Path

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase1055 import ExperimentalPolicy
from .phase105_verified import VerifiedPolicy


def evaluate(checkpoint, rounds=250, seed=10590001, confidence=0.95):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    common = dict(seed=seed + 900000, confidence=confidence,
                  min_gain=0.05, seats=(2, 3))
    holdout = ExperimentalPolicy(
        checkpoint, method="holdout", validation_futures=64,
        initial_futures=16, max_futures=64, batch_size=16, **common)
    fixed = VerifiedPolicy(checkpoint, futures=64, **common)
    comparison = paired_compare(
        holdout, fixed, [heuristic_v2] * 3, rounds=rounds,
        seed=seed, confidence=confidence)
    def stats(policy):
        s = policy.stats
        n = s["verified"]
        return dict(verified=n, overrides=s["overrides"],
                    mean_seconds=s["verification_seconds"] / n if n else 0,
                    mean_futures=(s.get("futures_used", n * 64) / n
                                  if n else 0))
    h, f = stats(holdout), stats(fixed)
    return dict(phase="10.5.9", checkpoint=str(checkpoint),
                rounds=rounds, seed=seed, comparison=comparison,
                holdout_stats=h, fixed_stats=f,
                eligible_for_review=(
                    comparison["advantage_ci_low"] > 0
                    and comparison["challenger_mean_score"]
                    >= comparison["reference_mean_score"]
                    and h["mean_seconds"] < f["mean_seconds"]),
                promoted=False,
                note="Exploratory comparison only; champion frozen.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seed", type=int, default=10590001)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase105/phase1059_tournament.json"))
    args = p.parse_args(argv)
    result = evaluate(args.candidate, args.rounds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
