"""Phase 10.5.5: variance-aware and independent-holdout verification.

The two-stage option is exploratory: independent confirmation reduces selection
bias but an ordinary normal interval is not a finite-sample guarantee.
The empirical-Bernstein option uses a conservative union bound over looks.
No model or champion is modified.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import NormalDist, mean, variance

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase105_adaptive import AdaptiveVerifiedPolicy
from .phase105_verified import VerifiedPolicy
from .phase9_search import outcome, state_seed


def interval(deltas, confidence, looks=1, method="bernstein"):
    n = len(deltas)
    if n < 2:
        return float("-inf"), float("inf")
    avg = mean(deltas)
    if method == "bernstein":
        # Maurer-Pontil empirical Bernstein bound, range = 2,
        # with Bonferroni allocation across scheduled looks.
        log_term = math.log(2 * looks / (1 - confidence))
        radius = math.sqrt(2 * variance(deltas) * log_term / n)
        radius += 14 * log_term / (3 * (n - 1))
    elif method == "normal":
        radius = NormalDist().inv_cdf((1 + confidence) / 2) * (
            math.sqrt(variance(deltas) / n))
    else:
        raise ValueError("unknown interval method")
    return avg - radius, avg + radius


def evidence(game, baseline, alternative, *, initial_futures=16,
             max_futures=64, batch_size=16, seed=3301000,
             confidence=0.95, min_gain=0.05, method="holdout",
             validation_futures=32):
    if (initial_futures < 2 or max_futures < initial_futures
            or batch_size < 1 or validation_futures < 2
            or min_gain < 0 or not 0 < confidence < 1
            or method not in ("holdout", "bernstein", "legacy")):
        raise ValueError("invalid verifier configuration")
    legal = [None, *game.legal()]
    if baseline not in legal or alternative not in legal:
        raise ValueError("illegal action")
    if baseline == alternative:
        return {"win_gain": 0.0, "win_ci_low": 0.0,
                "win_ci_high": 0.0, "score_gain": 0.0,
                "futures": 0, "override": False, "stop_reason": "same_action"}
    rng = random.Random(state_seed(game, seed))
    wins, scores = [], []
    looks = 1 + math.ceil((max_futures - initial_futures) / batch_size)
    while len(wins) < max_futures:
        target = min(max_futures, initial_futures if not wins
                     else len(wins) + batch_size)
        for _ in range(target - len(wins)):
            future = rng.randrange(2**64)
            aw, a_score = outcome(game, alternative, future)
            bw, b_score = outcome(game, baseline, future)
            wins.append(aw - bw)
            scores.append(a_score - b_score)
        bounds_method = "bernstein" if method == "bernstein" else "normal"
        low, high = interval(wins, confidence,
                             looks if method == "bernstein" else 1,
                             bounds_method)
        positive = low > min_gain and mean(scores) >= 0
        if positive:
            break
        if high <= min_gain:
            return {"win_gain": mean(wins), "win_ci_low": low,
                    "win_ci_high": high, "score_gain": mean(scores),
                    "futures": len(wins), "override": False,
                    "stop_reason": "negative_evidence"}
    if method == "holdout" and positive:
        # Selection and confirmation draws are disjoint. This remains an
        # exploratory two-stage rule, not a guaranteed 95% test.
        confirm_wins, confirm_scores = [], []
        for _ in range(validation_futures):
            future = rng.randrange(2**64)
            aw, a_score = outcome(game, alternative, future)
            bw, b_score = outcome(game, baseline, future)
            confirm_wins.append(aw - bw)
            confirm_scores.append(a_score - b_score)
        confirm_low, confirm_high = interval(confirm_wins, confidence,
                                            method="normal")
        accepted = confirm_low > 0 and mean(confirm_scores) >= 0
        return {"win_gain": mean(wins), "win_ci_low": low,
                "win_ci_high": high, "score_gain": mean(scores),
                "validation_gain": mean(confirm_wins),
                "validation_ci_low": confirm_low,
                "validation_ci_high": confirm_high,
                "futures": len(wins) + validation_futures,
                "override": accepted,
                "stop_reason": "positive_evidence" if accepted
                else "negative_evidence"}
    accepted = positive
    return {"win_gain": mean(wins), "win_ci_low": low,
            "win_ci_high": high, "score_gain": mean(scores),
            "futures": len(wins), "override": accepted,
            "stop_reason": "positive_evidence" if accepted
            else "max_futures"}


class ExperimentalPolicy(AdaptiveVerifiedPolicy):
    def __init__(self, checkpoint, *, method="holdout",
                 validation_futures=32, **kwargs):
        super().__init__(checkpoint, **kwargs)
        if method not in ("holdout", "bernstein", "legacy"):
            raise ValueError("unknown method")
        self.method = method
        self.validation_futures = validation_futures
        self.stats["stops"]["same_action"] = 0

    def evidence(self, game, baseline, alternative, **kwargs):
        return evidence(game, baseline, alternative, method=self.method,
                        validation_futures=self.validation_futures, **kwargs)


def benchmark(checkpoint, rounds=100, seed=9501000, *,
              initial_futures=16, max_futures=64, batch_size=16,
              validation_futures=32, confidence=0.95, min_gain=0.05,
              seats=(2, 3), log_limit=1000):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    common = dict(initial_futures=initial_futures, max_futures=max_futures,
                  batch_size=batch_size, seed=seed + 900000,
                  confidence=confidence, min_gain=min_gain,
                  seats=seats, log_limit=log_limit)
    policies = {
        "holdout": ExperimentalPolicy(
            checkpoint, method="holdout",
            validation_futures=validation_futures, **common),
        "bernstein": ExperimentalPolicy(
            checkpoint, method="bernstein", **common),
        "legacy": ExperimentalPolicy(
            checkpoint, method="legacy", **common),
        "fixed64": VerifiedPolicy(
            checkpoint, futures=max_futures, seed=seed + 900000,
            min_gain=min_gain, confidence=confidence, seats=seats),
    }
    results = {}
    started = time.perf_counter()
    # Every pair uses identical seeds, seat rotations and V2 opponents.
    for name in ("holdout", "bernstein", "legacy"):
        results[name + "_vs_fixed64"] = paired_compare(
            policies[name], policies["fixed64"], [heuristic_v2] * 3,
            rounds, seed, confidence)
    stats = {}
    for name, policy in policies.items():
        st = policy.stats
        verified = st["verified"]
        stats[name] = {
            "verified": verified, "overrides": st["overrides"],
            "mean_verification_seconds": (
                st["verification_seconds"] / verified if verified else 0),
            "mean_futures": (
                st["futures_used"] / verified if verified else max_futures),
            "stops": st.get("stops", {}),
            "by_seat": st["by_seat"],
        }
    return {"phase": "10.5.5", "checkpoint": str(checkpoint),
            "seed": seed, "rounds": rounds, "comparisons": results,
            "policy_stats": stats,
            "elapsed_seconds": time.perf_counter() - started,
            "decision_records": {
                name: policies[name].records
                for name in ("holdout", "bernstein", "legacy")},
            "promoted": False,
            "note": "Holdout and legacy methods are exploratory. "
                    "No champion qualification or promotion is implied."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=9501000)
    p.add_argument("--initial-futures", type=int, default=16)
    p.add_argument("--max-futures", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-futures", type=int, default=32)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--seats", type=int, nargs="+", default=[2, 3])
    p.add_argument("--log-limit", type=int, default=1000)
    p.add_argument("--output", type=Path,
                   default=Path("runs/phase105/phase1055_benchmark.json"))
    args = p.parse_args(argv)
    result = benchmark(
        args.candidate, rounds=args.rounds, seed=args.seed,
        initial_futures=args.initial_futures,
        max_futures=args.max_futures, batch_size=args.batch_size,
        validation_futures=args.validation_futures,
        confidence=args.confidence, min_gain=args.min_gain,
        seats=tuple(args.seats), log_limit=args.log_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "decision_records"}, indent=2))


if __name__ == "__main__":
    main()
