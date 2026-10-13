"""Phase 8.1: exact-four override audit and fresh paired tournament.

Audit uses Phase 8 collection, with game-cluster bootstrap subgroup intervals.
Tournament evaluates a frozen exact-four-only rule against V2 on new seeds.
No automatic promotion.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase72_hybrid import large_jump
from .phase8_audit import audit


def exact_four_policy(model):
    def act(game):
        teacher = heuristic_v2(game)
        student = choose(model, game)
        if (student is not None and teacher is not None
                and student != teacher
                and large_jump(game, student, threshold=4)
                and not large_jump(game, student, threshold=5)):
            return student
        return teacher

    return act


def group_interval(records, bootstrap, seed):
    if not records:
        return {"states": 0, "games": 0, "win_share_delta": None,
                "score_delta": None, "win_share_ci_95": None,
                "score_ci_95": None}
    groups = defaultdict(list)
    for record in records:
        groups[record["game_id"]].append(
            (record["win_share_delta"], record["score_delta"])
        )
    values = np.asarray([np.mean(group, axis=0) for group in groups.values()])
    mean = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty((bootstrap, 2))
    for i in range(bootstrap):
        indices = rng.integers(0, len(values), len(values))
        samples[i] = values[indices].mean(axis=0)
    ci = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "states": len(records), "games": len(groups),
        "win_share_delta": float(mean[0]), "score_delta": float(mean[1]),
        "win_share_ci_95": [float(ci[0, 0]), float(ci[1, 0])],
        "score_ci_95": [float(ci[0, 1]), float(ci[1, 1])],
    }


def diagnostic(model, games=10000, seed=820000, futures=48,
               max_overrides=1000, bootstrap=2000,
               output="runs/phase81/overrides.jsonl"):
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    report = audit(model, games=games, seed=seed, futures=futures,
                   max_overrides=max_overrides, bootstrap=bootstrap,
                   output=output)
    records = [json.loads(line) for line in
               Path(output).read_text(encoding="utf-8").splitlines()]
    report["subgroups"] = {
        "exactly_four": group_interval(
            [r for r in records if r["jump"] == 4], bootstrap, seed + 1),
        "five_or_more": group_interval(
            [r for r in records if r["jump"] >= 5], bootstrap, seed + 2),
    }
    report["note"] = ("Exploratory subgroup analysis, not independent evidence "
                      "of improvement. Cluster bootstrap is by game.")
    return report


def tournament(model, rounds=5000, seed=930000, confidence=0.95):
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid tournament settings")
    # Single prespecified policy vs two references, Bonferroni adjusted.
    adjusted = 1 - (1 - confidence) / 2
    candidate = exact_four_policy(model)
    student = lambda game: choose(model, game)
    return {
        "rounds": rounds, "seed": seed,
        "family_confidence": confidence,
        "comparison_confidence": adjusted,
        "against_v2": paired_compare(
            candidate, heuristic_v2, [heuristic_v2] * 3,
            rounds=rounds, seed=seed, confidence=adjusted),
        "against_phase7": paired_compare(
            candidate, student, [heuristic_v2] * 3,
            rounds=rounds, seed=seed, confidence=adjusted),
        "note": ("Fresh seeds required, chosen before inspecting tournament "
                 "results. No automatic promotion."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("audit")
    scan.add_argument("--games", type=int, default=10000)
    scan.add_argument("--seed", type=int, default=820000)
    scan.add_argument("--futures", type=int, default=48)
    scan.add_argument("--max-overrides", type=int, default=1000)
    scan.add_argument("--bootstrap", type=int, default=2000)
    scan.add_argument("--output", default="runs/phase81/overrides.jsonl")
    trial = sub.add_parser("tournament")
    trial.add_argument("--rounds", type=int, default=5000)
    trial.add_argument("--seed", type=int, default=930000)
    trial.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    model = load(args.model)
    if args.command == "audit":
        result = diagnostic(model, args.games, args.seed, args.futures,
                            args.max_overrides, args.bootstrap, args.output)
    else:
        result = tournament(model, args.rounds, args.seed, args.confidence)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
