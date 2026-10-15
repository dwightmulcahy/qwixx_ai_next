"""Phase 10.5.6: independent holdout replication and decision audit.

No checkpoint, champion or automatic promotion is changed.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase1055 import ExperimentalPolicy
from .phase105_verified import VerifiedPolicy


def audit(records, stats):
    """Summarize saved verification decisions; records may be truncated."""
    reasons = Counter(record["stop_reason"] for record in records)
    selected = [r for r in records if r.get("override")]
    rejected_validation = [
        r for r in records
        if "validation_gain" in r and not r.get("override")
    ]
    return {
        "recorded": len(records),
        "total_verified": stats["verified"],
        "records_truncated": len(records) < stats["verified"],
        "recorded_stop_reasons": dict(reasons),
        "recorded_overrides": len(selected),
        "recorded_validation_rejections": len(rejected_validation),
        "accepted_validation_gain_mean": (
            sum(r["validation_gain"] for r in selected) / len(selected)
            if selected else None),
        "accepted_validation_score_gain_mean": None,
        "accepted_by_seat": dict(Counter(r["seat"] for r in selected)),
        "note": "Rollout evidence is not a causal estimate of actual game gains. "
                "The saved records are not necessarily a random sample.",
    }


def replicate(checkpoint, rounds=250, seed=10560001, *,
              initial_futures=16, max_futures=64, batch_size=16,
              validation_futures=32, confidence=0.95, min_gain=0.05,
              seats=(2, 3), log_limit=1000,
              latency_target_seconds=0.05):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    if latency_target_seconds <= 0:
        raise ValueError("latency target must be positive")
    holdout = ExperimentalPolicy(
        checkpoint, method="holdout",
        validation_futures=validation_futures,
        initial_futures=initial_futures, max_futures=max_futures,
        batch_size=batch_size, seed=seed + 900000,
        min_gain=min_gain, confidence=confidence,
        seats=seats, log_limit=log_limit)
    fixed = VerifiedPolicy(
        checkpoint, futures=max_futures, seed=seed + 900000,
        min_gain=min_gain, confidence=confidence, seats=seats)
    started = time.perf_counter()
    paired = paired_compare(holdout, fixed, [heuristic_v2] * 3,
                            rounds, seed, confidence)
    elapsed = time.perf_counter() - started
    h, f = holdout.stats, fixed.stats
    h_n, f_n = h["verified"], f["verified"]
    h_latency = h["verification_seconds"] / h_n if h_n else 0.0
    f_latency = f["verification_seconds"] / f_n if f_n else 0.0
    h_futures = h["futures_used"] / h_n if h_n else 0.0
    checks = {
        "positive_win_share_advantage": paired["advantage_ci_low"] > 0,
        "mean_score_nonregression": (
            paired["challenger_mean_score"] >=
            paired["reference_mean_score"]),
        "lower_mean_verification_latency": (
            h_n > 0 and f_n > 0 and h_latency < f_latency),
        "lower_mean_rollout_count": (
            h_n > 0 and h_futures < max_futures),
        "latency_below_target": h_n > 0 and
        h_latency < latency_target_seconds,
    }
    return {
        "phase": "10.5.6",
        "comparison": "holdout_vs_fixed64_independent_replication",
        "checkpoint": str(checkpoint),
        "rounds": rounds, "seed": seed, "confidence": confidence,
        "configuration": {
            "initial_futures": initial_futures,
            "max_futures": max_futures,
            "batch_size": batch_size,
            "validation_futures": validation_futures,
            "min_gain": min_gain, "seats": list(seats),
            "latency_target_seconds": latency_target_seconds,
        },
        "paired": paired,
        "holdout_stats": h,
        "fixed_stats": f,
        "efficiency": {
            "holdout_mean_futures": h_futures,
            "fixed_mean_futures": float(max_futures),
            "rollout_reduction_fraction": (
                1 - h_futures / max_futures if h_n else None),
            "holdout_mean_verification_seconds": h_latency,
            "fixed_mean_verification_seconds": f_latency,
            "latency_reduction_fraction": (
                1 - h_latency / f_latency if f_latency else None),
        },
        "audit": audit(holdout.records, h),
        "checks": checks,
        "eligible_for_champion_qualification": (
            checks["positive_win_share_advantage"]
            and checks["mean_score_nonregression"]
            and checks["lower_mean_rollout_count"]),
        "elapsed_seconds": elapsed,
        "decision_records": holdout.records,
        "promoted": False,
        "note": "Independent game seeds from Phase 10.5.5; still a single "
                "replication set. Confidence interval is exploratory. "
                "No champion promotion.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seed", type=int, default=10560001)
    p.add_argument("--initial-futures", type=int, default=16)
    p.add_argument("--max-futures", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-futures", type=int, default=32)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--seats", type=int, nargs="+", default=[2, 3])
    p.add_argument("--log-limit", type=int, default=1000)
    p.add_argument("--latency-target-seconds", type=float, default=0.05)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase105/phase1056_replication.json"))
    args = p.parse_args(argv)
    result = replicate(
        args.candidate, rounds=args.rounds, seed=args.seed,
        initial_futures=args.initial_futures,
        max_futures=args.max_futures, batch_size=args.batch_size,
        validation_futures=args.validation_futures,
        confidence=args.confidence, min_gain=args.min_gain,
        seats=tuple(args.seats), log_limit=args.log_limit,
        latency_target_seconds=args.latency_target_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "decision_records"}, indent=2))


if __name__ == "__main__":
    main()
