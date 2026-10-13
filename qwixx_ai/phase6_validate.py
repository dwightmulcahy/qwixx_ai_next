"""Phase 6: evaluate imitation against V2 and historical neural policies."""
import argparse
import json

from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imitation", default="runs/phase6/imitation.pt")
    parser.add_argument("--champion", default="runs/phase5_1/best_competitive.pt")
    parser.add_argument("--candidate", default="runs/phase5_6/neither/best_candidate.pt")
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=6211)
    args = parser.parse_args()
    imitation = load(args.imitation)
    champion = load(args.champion)
    candidate = load(args.candidate) if args.candidate else None
    im = lambda game: choose(imitation, game)
    ch = lambda game: choose(champion, game)
    lineup = [heuristic_v2] * 3
    comparisons = {
        "imitation_vs_v2": paired_compare(im, heuristic_v2, lineup,
                                          args.rounds, args.seed),
        "imitation_vs_phase5_1": paired_compare(im, ch, lineup,
                                                args.rounds, args.seed + 100000),
    }
    if candidate is not None:
        ca = lambda game: choose(candidate, game)
        comparisons["imitation_vs_phase5_6"] = paired_compare(
            im, ca, lineup, args.rounds, args.seed + 200000)
    print(json.dumps({"lineup": "three V2 opponents", "results": comparisons,
                      "note": "No automatic promotion or checkpoint replacement."},
                     indent=2))


if __name__ == "__main__":
    main()
