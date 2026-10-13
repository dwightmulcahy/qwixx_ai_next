"""Compare Qwixx policies using fixed seeds and balanced seat assignments.

Example:
python -m qwixx_ai.strategy_tournament --champion runs/phase5_1/best_competitive.pt \\
    --candidate runs/phase5_6/neither/best_candidate.pt --rounds 500 --seed 842119
"""
import argparse
import json
from statistics import mean

from .agents import heuristic
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .tournament import play


def evaluate_policies(policies, rounds=500, seed=842119):
    """Each policy occupies every seat against a fixed three-opponent lineup."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if len(policies) < 2:
        raise ValueError("at least two policies required")
    # Hold opponents fixed, avoiding policy-dependent changes in matchup.
    baseline = policies[:3]
    while len(baseline) < 3:
        baseline.append(policies[0])
    report = {}
    for name, focal in policies:
        wins, scores = [], []
        for i in range(rounds):
            for seat in range(4):
                agents = [fn for _, fn in baseline]
                agents.insert(seat, focal)
                result, shares = play(agents, seed + i)
                wins.append(shares[seat])
                scores.append(result[seat])
        report[name] = {
            "games": len(wins),
            "win_share": mean(wins),
            "mean_score": mean(scores),
        }
    return {"seed": seed, "rounds": rounds, "lineup": [n for n, _ in baseline],
            "results": report,
            "note": "Exploratory common-seed seat-balanced comparison; not a promotion test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion")
    parser.add_argument("--candidate")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=842119)
    args = parser.parse_args()
    policies = [("heuristic_v1", heuristic), ("heuristic_v2", heuristic_v2)]
    if args.champion:
        model = load(args.champion)
        policies.append(("champion", lambda game: choose(model, game)))
    if args.candidate:
        model = load(args.candidate)
        policies.append(("candidate", lambda game: choose(model, game)))
    print(json.dumps(evaluate_policies(policies, args.rounds, args.seed), indent=2))


if __name__ == "__main__":
    main()
