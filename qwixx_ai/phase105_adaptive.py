"""Phase 10.5.3: adaptive, audited verification with independent confirmation.

The original checkpoint and champion remain frozen; promotion is never automatic.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean

from .champion import champion
from .controlled_strategy import paired_compare
from .env import encode
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase9_search import outcome, state_seed


def decision_summary(win_deltas, score_deltas, confidence, min_gain, looks=4):
    """Simultaneous Hoeffding bounds using a union bound across all looks.

    Each paired win-share difference is in [-1, 1]. The resulting bounds
    cover all scheduled looks with probability at least confidence.
    """
    n = len(win_deltas)
    if n < 2:
        return {"win_gain": 0.0, "win_ci_low": float("-inf"),
                "win_ci_high": float("inf"), "score_gain": 0.0,
                "futures": n, "override": False}
    if not 0 < confidence < 1 or looks < 1:
        raise ValueError("invalid confidence or looks")
    avg = mean(win_deltas)
    margin = math.sqrt(2 * math.log(2 * looks / (1 - confidence)) / n)
    score_gain = mean(score_deltas)
    return {"win_gain": avg, "win_ci_low": avg - margin,
            "win_ci_high": avg + margin, "score_gain": score_gain,
            "futures": n,
            "override": avg - margin > min_gain and score_gain >= 0}


def adaptive_evidence(game, baseline, alternative, *,
                      initial_futures=16, max_futures=64, batch_size=16,
                      seed=3301000, confidence=0.95, min_gain=0.05):
    """Stop early only when interval resolves above/below gain threshold.

    An early positive interval also requires nonnegative observed score gain.
    Negative/ambiguous evidence defaults to the frozen neural action.
    """
    if (initial_futures < 2 or max_futures < initial_futures
            or batch_size < 1 or min_gain < 0 or not 0 < confidence < 1):
        raise ValueError("invalid adaptive verification settings")
    legal = [None, *game.legal()]
    if baseline not in legal or alternative not in legal:
        raise ValueError("illegal action")
    if baseline == alternative:
        return {"win_gain": 0.0, "win_ci_low": 0.0, "win_ci_high": 0.0,
                "score_gain": 0.0, "futures": 0, "override": False,
                "stop_reason": "same_action"}
    looks = 1 + math.ceil((max_futures - initial_futures) / batch_size)
    rng = random.Random(state_seed(game, seed))
    wins, scores = [], []
    while len(wins) < max_futures:
        target = min(max_futures, initial_futures if not wins
                     else len(wins) + batch_size)
        for _ in range(target - len(wins)):
            future = rng.randrange(2**64)
            aw, ass = outcome(game, alternative, future)
            bw, bs = outcome(game, baseline, future)
            wins.append(aw - bw)
            scores.append(ass - bs)
        result = decision_summary(wins, scores, confidence, min_gain, looks)
        # Repeated peeking at ordinary CIs does not preserve nominal coverage.
        # These are exploratory stopping heuristics, not formal guarantees.
        if result["win_ci_low"] > min_gain and result["score_gain"] >= 0:
            result["stop_reason"] = "positive_evidence"
            return result
        if result["win_ci_high"] <= min_gain:
            result["stop_reason"] = "negative_evidence"
            return result
    result["stop_reason"] = "max_futures"
    return result


class AdaptiveVerifiedPolicy:
    def __init__(self, checkpoint, initial_futures=16, max_futures=64,
                 batch_size=16, seed=3301000, min_gain=0.05,
                 confidence=0.95, seats=(2, 3), log_limit=1000):
        if not seats or any(s not in (1, 2, 3, 4) for s in seats):
            raise ValueError("invalid seats")
        if log_limit < 0:
            raise ValueError("log_limit must be nonnegative")
        if (initial_futures < 2 or max_futures < initial_futures
                or batch_size < 1 or min_gain < 0
                or not 0 < confidence < 1):
            raise ValueError("invalid adaptive verification settings")
        self.model = load(checkpoint)
        self.initial_futures, self.max_futures = initial_futures, max_futures
        self.batch_size, self.seed = batch_size, seed
        self.min_gain, self.confidence = min_gain, confidence
        self.seats, self.log_limit = frozenset(seats), log_limit
        self.records = []
        self.stats = {"decisions": 0, "disagreements": 0, "verified": 0,
                      "overrides": 0, "futures_used": 0,
                      "verification_seconds": 0.0,
                      "stops": {"positive_evidence": 0,
                                "negative_evidence": 0, "max_futures": 0},
                      "by_seat": {str(s): {"verified": 0, "overrides": 0}
                                  for s in range(1, 5)}}

    def __call__(self, game):
        self.stats["decisions"] += 1
        baseline = choose(self.model, game, epsilon=0)
        seat = game.actor + 1
        if seat not in self.seats:
            return baseline
        alternative = champion(game)
        if alternative == baseline:
            return baseline
        self.stats["disagreements"] += 1
        start = time.perf_counter()
        result = adaptive_evidence(
            game, baseline, alternative,
            initial_futures=self.initial_futures,
            max_futures=self.max_futures, batch_size=self.batch_size,
            seed=self.seed, confidence=self.confidence,
            min_gain=self.min_gain)
        self.stats["verification_seconds"] += time.perf_counter() - start
        self.stats["verified"] += 1
        self.stats["futures_used"] += result["futures"]
        self.stats["stops"][result["stop_reason"]] += 1
        self.stats["by_seat"][str(seat)]["verified"] += 1
        if result["override"]:
            self.stats["overrides"] += 1
            self.stats["by_seat"][str(seat)]["overrides"] += 1
        if result["override"] or len(self.records) < self.log_limit:
            self.records.append({
                "seat": seat, "turn": game.turn, "phase": game.phase,
                "active": game.active + 1, "dice": list(game.dice),
                "neural_action": encode(baseline),
                "champion_action": encode(alternative),
                **result})
        return alternative if result["override"] else baseline


def evaluate(checkpoint, rounds=250, seed=4501000, *,
             initial_futures=16, max_futures=64, batch_size=16,
             min_gain=0.05, confidence=0.95, seats=(2, 3),
             log_limit=1000):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    candidate = AdaptiveVerifiedPolicy(
        checkpoint, initial_futures, max_futures, batch_size,
        seed + 900000, min_gain, confidence, seats, log_limit)
    base = load(checkpoint)
    frozen = lambda game: choose(base, game, epsilon=0)
    opponents = [heuristic_v2] * 3
    start = time.perf_counter()
    comparisons = {
        "versus_original": paired_compare(
            candidate, frozen, opponents, rounds, seed, confidence),
        "versus_champion": paired_compare(
            candidate, champion, opponents, rounds, seed + 100000, confidence),
        "versus_v2": paired_compare(
            candidate, heuristic_v2, opponents, rounds, seed + 200000,
            confidence),
    }
    stats = candidate.stats
    stats["mean_futures_per_verification"] = (
        stats["futures_used"] / stats["verified"] if stats["verified"] else 0)
    stats["mean_verification_seconds"] = (
        stats["verification_seconds"] / stats["verified"]
        if stats["verified"] else 0)
    eligible = (all(c["advantage_ci_low"] > 0 for c in comparisons.values())
                and comparisons["versus_champion"]["challenger_mean_score"]
                >= comparisons["versus_champion"]["reference_mean_score"])
    return {"phase": "10.5.3", "checkpoint": str(checkpoint),
            "rounds": rounds, "seed": seed,
            "initial_futures": initial_futures,
            "max_futures": max_futures, "batch_size": batch_size,
            "min_gain": min_gain, "confidence": confidence,
            "seats": list(seats), **comparisons,
            "timing_seconds": time.perf_counter() - start,
            "verification_stats": stats,
            "decision_records": candidate.records,
            "decision_records_truncated": stats["verified"] > len(candidate.records),
            "eligible_for_review": eligible, "promoted": False,
            "note": "Adaptive stopping uses repeated ordinary CIs: nominal "
                    "coverage is not guaranteed. Confirm with fresh seeds."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seed", type=int, default=4501000)
    p.add_argument("--initial-futures", type=int, default=16)
    p.add_argument("--max-futures", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seats", type=int, nargs="+", default=[2, 3])
    p.add_argument("--log-limit", type=int, default=1000)
    p.add_argument("--output", type=Path,
                   default=Path("runs/phase105/adaptive_confirmation.json"))
    args = p.parse_args(argv)
    result = evaluate(
        args.candidate, args.rounds, args.seed,
        initial_futures=args.initial_futures,
        max_futures=args.max_futures, batch_size=args.batch_size,
        min_gain=args.min_gain, confidence=args.confidence,
        seats=tuple(args.seats), log_limit=args.log_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "
")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "decision_records"}, indent=2))


if __name__ == "__main__":
    main()
