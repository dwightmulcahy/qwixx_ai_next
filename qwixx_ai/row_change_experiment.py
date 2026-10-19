"""Fresh-seed comparison of row-change-only frozen learner vs Phase 9.1.

Diagnostic candidate only. Does not modify or promote champion.
"""
import argparse
import json
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .decision_disagreement_audit import model_choice
from .heuristic_v2 import heuristic_v2

DEFAULT_SEED = 26000000


def row_change_only(game, incumbent):
    """Change only the row of an existing non-locking mark."""
    proposed = model_choice(game, incumbent)
    return row_change_only_from_choices(incumbent, proposed)


def evaluate(rounds=100, seed=DEFAULT_SEED):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    if seed in {20000000, 21000000, 22000000, 23000000,
                24000000, 25000000}:
        raise ValueError("choose a fresh evaluation seed")
    stats = {"decisions": 0, "row_overrides": 0,
             "blocked_pass_to_mark": 0, "blocked_other": 0}

    def challenger(game):
        incumbent = champion(game)
        proposed = model_choice(game, incumbent)
        chosen = row_change_only_from_choices(incumbent, proposed)
        stats["decisions"] += 1
        if chosen != incumbent:
            stats["row_overrides"] += 1
        elif proposed != incumbent:
            if incumbent is None and proposed is not None:
                stats["blocked_pass_to_mark"] += 1
            else:
                stats["blocked_other"] += 1
        return chosen

    result = paired_compare(
        challenger, champion, [heuristic_v2] * 3,
        rounds=rounds, seed=seed
    )
    stats["override_rate"] = (
        stats["row_overrides"] / stats["decisions"]
        if stats["decisions"] else 0
    )
    return {
        "experiment": "frozen_row_change_only",
        "settings": {"rounds": rounds, "seed": seed,
                     "frozen_weights": True, "margin": 0.02},
        "versus_phase91": result,
        "decision_stats": stats,
        "promoted": False,
        "note": "Fresh-seed exploratory test; compare win share, score and "
                "confidence interval before considering further validation.",
    }


def row_change_only_from_choices(incumbent, proposed):
    if (incumbent is not None and proposed is not None
            and not incumbent[2] and not proposed[2]
            and incumbent[0] != proposed[0]):
        return proposed
    return incumbent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path(
        "runs/decision_quality/row_change_only_26000000.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = evaluate(args.rounds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
