"""Phase 10.5.2: conservative on-demand paired-rollout neural corrections.

Only compare the frozen neural action with the frozen Phase 9.1 proposal.
No learned correction head, checkpoint changes, or automatic promotion.
"""
import argparse
import json
import math
import random
import time
from statistics import NormalDist, mean, stdev

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase9_search import outcome, state_seed


def paired_evidence(game, baseline, alternative, futures=96,
                    seed=3301000, confidence=0.95):
    """Use identical future RNG seeds for both legal actions.

    Confidence intervals are exploratory normal approximations over future
    draws; game trajectories can diverge after the initial action.
    """
    if futures < 2 or not 0 < confidence < 1:
        raise ValueError("futures >= 2 and 0 < confidence < 1 required")
    if baseline != alternative and alternative is not None and alternative not in game.legal():
        raise ValueError("illegal alternative")
    if baseline is not None and baseline not in game.legal():
        raise ValueError("illegal baseline")
    rng = random.Random(state_seed(game, seed))
    deltas, score_deltas = [], []
    for _ in range(futures):
        future = rng.randrange(2**64)
        a_win, a_score = outcome(game, alternative, future)
        b_win, b_score = outcome(game, baseline, future)
        deltas.append(a_win - b_win)
        score_deltas.append(a_score - b_score)
    avg = mean(deltas)
    stderr = stdev(deltas) / math.sqrt(futures)
    margin = NormalDist().inv_cdf((1 + confidence) / 2) * stderr
    return {"win_gain": avg, "win_ci_low": avg - margin,
            "win_ci_high": avg + margin,
            "score_gain": mean(score_deltas), "futures": futures}


class VerifiedPolicy:
    def __init__(self, checkpoint, futures=96, seed=3301000,
                 min_gain=0.05, confidence=0.95, seats=(2, 3)):
        if futures < 2 or min_gain < 0 or not 0 < confidence < 1:
            raise ValueError("invalid verification settings")
        if not seats or any(s not in (1, 2, 3, 4) for s in seats):
            raise ValueError("seats must be a nonempty subset of 1..4")
        self.model = load(checkpoint)
        self.futures, self.seed = futures, seed
        self.min_gain, self.confidence = min_gain, confidence
        self.seats = frozenset(seats)
        self.stats = {"decisions": 0, "disagreements": 0,
                      "verified": 0, "overrides": 0,
                      "verification_seconds": 0.0,
                      "by_seat": {str(s): {"verified": 0, "overrides": 0}
                                  for s in range(1, 5)}}

    def __call__(self, game):
        self.stats["decisions"] += 1
        baseline = choose(self.model, game, epsilon=0)
        if game.actor + 1 not in self.seats:
            return baseline
        alternative = champion(game)
        if baseline == alternative:
            return baseline
        self.stats["disagreements"] += 1
        start = time.perf_counter()
        evidence = paired_evidence(
            game, baseline, alternative, self.futures, self.seed,
            self.confidence)
        self.stats["verification_seconds"] += time.perf_counter() - start
        self.stats["verified"] += 1
        self.stats["by_seat"][str(game.actor + 1)]["verified"] += 1
        if (evidence["win_ci_low"] > self.min_gain
                and evidence["score_gain"] >= 0):
            self.stats["overrides"] += 1
            self.stats["by_seat"][str(game.actor + 1)]["overrides"] += 1
            return alternative
        return baseline


def evaluate(checkpoint, rounds=100, seed=3401000, futures=96,
             min_gain=0.05, confidence=0.95, seats=(2, 3)):
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    candidate = VerifiedPolicy(checkpoint, futures, seed + 900000,
                               min_gain, confidence, seats)
    original = load(checkpoint)
    frozen = lambda game: choose(original, game, epsilon=0)
    opponents = [heuristic_v2] * 3
    started = time.perf_counter()
    against_original = paired_compare(candidate, frozen, opponents,
                                      rounds=rounds, seed=seed,
                                      confidence=confidence)
    against_champion = paired_compare(candidate, champion, opponents,
                                      rounds=rounds, seed=seed + 100000,
                                      confidence=confidence)
    against_v2 = paired_compare(candidate, heuristic_v2, opponents,
                                rounds=rounds, seed=seed + 200000,
                                confidence=confidence)
    eligible = (against_original["advantage_ci_low"] > 0
                and against_champion["advantage_ci_low"] > 0
                and against_v2["advantage_ci_low"] > 0
                and against_champion["challenger_mean_score"]
                >= against_champion["reference_mean_score"])
    stats = candidate.stats
    stats["mean_verification_seconds"] = (
        stats["verification_seconds"] / stats["verified"]
        if stats["verified"] else 0.0)
    return {"phase": "10.5.2", "checkpoint": str(checkpoint),
            "futures": futures, "min_gain": min_gain,
            "confidence": confidence, "seats": list(seats),
            "versus_original": against_original,
            "versus_champion": against_champion,
            "versus_v2": against_v2,
            "timing_seconds": time.perf_counter() - started,
            "verification_stats": stats,
            "eligible_for_review": eligible, "promoted": False,
            "note": "Exploratory: in-game V2 rollouts and normal-approximation "
                    "intervals are not a guarantee of real policy improvement."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=3401000)
    p.add_argument("--futures", type=int, default=96)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seats", type=int, nargs="+", default=[2, 3])
    p.add_argument("--output")
    args = p.parse_args(argv)
    result = evaluate(args.candidate, args.rounds, args.seed, args.futures,
                      args.min_gain, args.confidence, tuple(args.seats))
    if args.output:
        from pathlib import Path
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
