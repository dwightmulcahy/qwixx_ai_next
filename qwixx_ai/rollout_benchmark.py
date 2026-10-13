"""Benchmark matched-dice rollout against V2, with optional decision diagnostics."""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .rollout_agent import rollout, rollout_details
from .core import Game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=953711)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=0,
                        help="0 means simulate to game end")
    parser.add_argument("--min-improvement", type=float, default=0.025)
    parser.add_argument("--opponents", choices=["v2", "champion", "mixed"],
                        default="mixed")
    parser.add_argument("--diagnostics", type=int, default=0,
                        help="Sample N V2 game decisions and report overrides")
    args = parser.parse_args()
    champion = load(args.champion)
    baseline = lambda g: choose(champion, g)
    opponent_policies = {
        "v2": [heuristic_v2] * 4,
        "champion": [baseline] * 4,
        "mixed": [baseline, baseline, heuristic_v2, heuristic_v2],
    }[args.opponents]
    horizon = args.horizon or None
    rollout_policy = lambda g: rollout(
        g, args.simulations, horizon, opponent_policies,
        args.seed, args.min_improvement)
    results = paired_compare(rollout_policy, heuristic_v2,
                             [baseline, baseline, heuristic_v2],
                             args.rounds, args.seed)
    output = {"comparison": "rollout_v2_vs_heuristic_v2",
              "opponent_model": args.opponents,
              "horizon": "terminal" if horizon is None else horizon,
              **results}
    if args.diagnostics:
        game = Game(seed=args.seed + 300000)
        changes, samples = 0, []
        while not game.done and len(samples) < args.diagnostics:
            detail = rollout_details(game, args.simulations, horizon,
                                     opponent_policies, args.seed,
                                     args.min_improvement)
            changes += detail["changed"]
            samples.append({"turn": game.turn, "actor": game.actor,
                            "v2": detail["fallback"], "rollout": detail["action"],
                            "changed": detail["changed"]})
            game.step(heuristic_v2(game))
        output["diagnostics"] = {"sampled": len(samples), "changed": changes,
                                 "decisions": samples}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
