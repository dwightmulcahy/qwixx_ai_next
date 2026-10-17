"""Phase 11.6: independent paired comparison against source neural and champion."""
import argparse
import json
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load


def evaluate(candidate, source, rounds=250, seed=11060003):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    candidate_model = load(candidate)
    source_model = load(source)
    challenger = lambda game: choose(candidate_model, game, epsilon=0)
    baseline = lambda game: choose(source_model, game, epsilon=0)
    opponents = [heuristic_v2] * 3
    versus_source = paired_compare(challenger, baseline, opponents, rounds, seed)
    versus_champion = paired_compare(
        challenger, champion, opponents, rounds, seed + 100000)
    versus_v2 = paired_compare(
        challenger, heuristic_v2, opponents, rounds, seed + 200000)
    eligible = (
        versus_source["advantage_ci_low"] > 0
        and versus_champion["advantage_ci_low"] > 0
        and versus_v2["advantage_ci_low"] > 0
        and versus_champion["challenger_mean_score"]
        >= versus_champion["reference_mean_score"]
    )
    return {
        "phase": "11.6", "candidate": str(candidate),
        "source": str(source), "rounds": rounds, "seed": seed,
        "versus_source": versus_source,
        "versus_champion": versus_champion,
        "versus_v2": versus_v2,
        "eligible_for_review": eligible, "promoted": False,
        "note": "Independent exploratory gate; no automatic promotion.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seed", type=int, default=11060003)
    p.add_argument("--history", type=Path, default=Path(
        "runs/phase11/phase116_evaluations.jsonl"))
    args = p.parse_args(argv)
    for path in (args.candidate, args.source):
        if not path.is_file():
            p.error(f"checkpoint does not exist: {path}")
    result = evaluate(args.candidate, args.source, args.rounds, args.seed)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
