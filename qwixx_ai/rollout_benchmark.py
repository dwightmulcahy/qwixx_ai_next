"""Compare rollout decisions against the underlying V2 policy."""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .rollout_agent import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=953711)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    champion = load(args.champion)
    baseline = lambda g: choose(champion, g)
    rollout_policy = lambda g: rollout(g, args.simulations, args.horizon)
    results = paired_compare(rollout_policy, heuristic_v2,
                             [baseline, baseline, heuristic_v2],
                             args.rounds, args.seed)
    print(json.dumps({"comparison": "rollout_vs_v2", **results}, indent=2))


if __name__ == "__main__":
    main()
