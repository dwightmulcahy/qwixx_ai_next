"""Phase 10.5.4 controlled verification benchmark and manual qualification.

Runs both policies on identical seeds and seats, with fixed V2 opponents.
The champion is never changed by this command.
"""
import argparse
import json
import time
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase105_adaptive import AdaptiveVerifiedPolicy
from .phase105_verified import VerifiedPolicy


def benchmark(checkpoint, rounds=100, seed=7501000, confidence=0.95,
              max_futures=64, initial_futures=16, batch_size=16,
              min_gain=0.05, seats=(2, 3), log_limit=1000):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    fixed = VerifiedPolicy(checkpoint, max_futures, seed + 900000,
                           min_gain, confidence, seats)
    adaptive = AdaptiveVerifiedPolicy(
        checkpoint, initial_futures, max_futures, batch_size,
        seed + 900000, min_gain, confidence, seats, log_limit)
    opponents = [heuristic_v2] * 3
    start = time.perf_counter()
    direct = paired_compare(adaptive, fixed, opponents, rounds, seed,
                            confidence)
    elapsed = time.perf_counter() - start
    fixed_stats = dict(fixed.stats)
    adaptive_stats = dict(adaptive.stats)
    for stats in (fixed_stats, adaptive_stats):
        count = stats["verified"]
        stats["mean_verification_seconds"] = (
            stats["verification_seconds"] / count if count else 0.0)
    adaptive_stats["mean_futures_per_verification"] = (
        adaptive_stats["futures_used"] / adaptive_stats["verified"]
        if adaptive_stats["verified"] else 0.0)
    return {
        "phase": "10.5.4", "comparison": "adaptive_vs_fixed64",
        "checkpoint": str(checkpoint), "seed": seed,
        "rounds": rounds, "confidence": confidence,
        "max_futures": max_futures, "initial_futures": initial_futures,
        "batch_size": batch_size, "min_gain": min_gain,
        "seats": list(seats), "paired": direct,
        "elapsed_seconds": elapsed, "adaptive_stats": adaptive_stats,
        "fixed_stats": fixed_stats, "decision_records": adaptive.records,
        "promoted": False,
        "note": "Comparison uses the same game seeds and opponent lineup; "
                "rollout verification remains policy-dependent."}


def qualify(checkpoint, rounds=500, seed=8501000, confidence=0.95,
            max_futures=64, initial_futures=16, batch_size=16,
            min_gain=0.05, seats=(2, 3), log_limit=1000,
            max_mean_verification_seconds=0.1):
    if rounds < 500:
        raise ValueError("qualification requires at least 500 rounds")
    candidate = AdaptiveVerifiedPolicy(
        checkpoint, initial_futures, max_futures, batch_size,
        seed + 900000, min_gain, confidence, seats, log_limit)
    start = time.perf_counter()
    result = paired_compare(candidate, champion, [heuristic_v2] * 3,
                            rounds, seed, confidence)
    stats = candidate.stats
    mean_time = (stats["verification_seconds"] / stats["verified"]
                 if stats["verified"] else 0.0)
    checks = {
        "positive_champion_advantage": result["advantage_ci_low"] > 0,
        "no_mean_score_regression": (
            result["challenger_mean_score"] >= result["reference_mean_score"]),
        "verification_latency_acceptable": (
            mean_time <= max_mean_verification_seconds),
    }
    return {
        "phase": "10.5.4", "comparison": "adaptive_vs_champion",
        "checkpoint": str(checkpoint), "seed": seed,
        "rounds": rounds, "confidence": confidence,
        "configuration": {
            "initial_futures": initial_futures,
            "max_futures": max_futures, "batch_size": batch_size,
            "min_gain": min_gain, "seats": list(seats)},
        "paired": result, "verification_stats": stats,
        "mean_verification_seconds": mean_time,
        "max_mean_verification_seconds": max_mean_verification_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "checks": checks, "eligible_for_manual_review": all(checks.values()),
        "decision_records": candidate.records, "promoted": False,
        "note": "Single-seed qualification is not independent replication. "
                "No champion or checkpoint is changed."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--mode", choices=("benchmark", "qualify"),
                   default="benchmark")
    p.add_argument("--rounds", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--max-futures", type=int, default=64)
    p.add_argument("--initial-futures", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--seats", type=int, nargs="+", default=[2, 3])
    p.add_argument("--log-limit", type=int, default=1000)
    p.add_argument("--max-mean-verification-seconds", type=float, default=0.1)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    options = dict(
        rounds=args.rounds or (500 if args.mode == "qualify" else 100),
        seed=args.seed if args.seed is not None
        else (8501000 if args.mode == "qualify" else 7501000),
        confidence=args.confidence, max_futures=args.max_futures,
        initial_futures=args.initial_futures, batch_size=args.batch_size,
        min_gain=args.min_gain, seats=tuple(args.seats),
        log_limit=args.log_limit)
    if args.mode == "qualify":
        result = qualify(
            args.candidate, **options,
            max_mean_verification_seconds=args.max_mean_verification_seconds)
    else:
        result = benchmark(args.candidate, **options)
    output = args.output or Path(
        "runs/phase105/phase1054_" + args.mode + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "decision_records"}, indent=2))


if __name__ == "__main__":
    main()
