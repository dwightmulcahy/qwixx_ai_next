"""Controlled paired comparisons of Qwixx policies against fixed opponent lineups."""
import argparse
import json
import math
from statistics import NormalDist, mean, stdev

from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .tournament import play


def paired_compare(challenger, reference, opponents, rounds=1250, seed=932811,
                   confidence=0.95):
    """Same dice seeds, focal seats and three fixed opponents for both policies."""
    if len(opponents) != 3:
        raise ValueError("exactly three opponents required")
    if rounds < 2 or not 0 < confidence < 1:
        raise ValueError("invalid rounds or confidence")
    blocks, challenger_scores, reference_scores = [], [], []
    challenger_wins, reference_wins = [], []
    for i in range(rounds):
        deltas = []
        for seat in range(4):
            def trial(policy):
                agents = list(opponents)
                agents.insert(seat, policy)
                scores, shares = play(agents, seed + i)
                return scores[seat], shares[seat]
            a_score, a_win = trial(challenger)
            b_score, b_win = trial(reference)
            challenger_scores.append(a_score)
            reference_scores.append(b_score)
            challenger_wins.append(a_win)
            reference_wins.append(b_win)
            deltas.append(a_win - b_win)
        blocks.append(mean(deltas))
    advantage = mean(blocks)
    margin = NormalDist().inv_cdf((1 + confidence) / 2) * stdev(blocks) / math.sqrt(rounds)
    return {
        "rounds": rounds, "games_per_policy": rounds * 4, "seed": seed,
        "confidence": confidence, "challenger_win_share": mean(challenger_wins),
        "reference_win_share": mean(reference_wins),
        "paired_advantage": advantage, "advantage_ci_low": advantage - margin,
        "advantage_ci_high": advantage + margin,
        "challenger_mean_score": mean(challenger_scores),
        "reference_mean_score": mean(reference_scores),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=932811)
    args = parser.parse_args()
    champion = load(args.champion)
    champion_policy = lambda g: choose(champion, g)
    candidate_policy = None
    if args.candidate:
        candidate = load(args.candidate)
        candidate_policy = lambda g: choose(candidate, g)
    # Avoid the very weak original heuristic in all competitive lineups.
    lineups = {"champions": [champion_policy] * 3,
               "mixed": [champion_policy, candidate_policy or champion_policy,
                         heuristic_v2]}
    results = {}
    for name, lineup in lineups.items():
        results[name] = {
            "v2_vs_champion": paired_compare(
                heuristic_v2, champion_policy, lineup, args.rounds, args.seed),
        }
        if candidate_policy:
            results[name]["v2_vs_candidate"] = paired_compare(
                heuristic_v2, candidate_policy, lineup, args.rounds,
                args.seed + 100000)
    print(json.dumps({"lineups": list(lineups), "results": results,
                      "note": "Exploratory paired comparison; not automatic promotion."},
                     indent=2))


if __name__ == "__main__":
    main()
