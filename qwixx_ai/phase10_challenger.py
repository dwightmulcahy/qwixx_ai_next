"""Phase 10: held-out, paired neural challenger gate against the frozen champion.

This is an offline evaluation tool. It never replaces the deployed champion.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase10_diagnostics import compare_diagnostics
from .model import choose, load


def evaluate_checkpoint(checkpoint, rounds=250, seed=1001000, confidence=0.95, diagnostics=False):
    """Compare the candidate with Phase 9.1 and V2 under paired four-seat trials."""
    model = load(checkpoint)

    def challenger(game):
        return choose(model, game, epsilon=0)

    versus_champion = paired_compare(
        challenger, champion, [heuristic_v2] * 3,
        rounds=rounds, seed=seed, confidence=confidence,
    )
    versus_v2 = paired_compare(
        challenger, heuristic_v2, [heuristic_v2] * 3,
        rounds=rounds, seed=seed + 100000, confidence=confidence,
    )
    # The same opponent lineup is used for each paired candidate/reference trial.
    # A positive lower bound is necessary, not sufficient, for a promotion review.
    eligible = (
        versus_champion["advantage_ci_low"] > 0
        and versus_champion["challenger_mean_score"]
        >= versus_champion["reference_mean_score"]
        and versus_v2["advantage_ci_low"] > 0
    )
    result = {
        "phase": 10,
        "candidate": str(checkpoint),
        "reference": "phase91_search (frozen)",
        "opponents": "three heuristic_v2 policies",
        "exploration": 0,
        "rounds": rounds,
        "confidence": confidence,
        "versus_champion": versus_champion,
        "versus_v2": versus_v2,
        "eligible_for_review": eligible,
        "promoted": False,
        "note": "Exploratory evaluation; no automatic deployment or promotion.",
    }
    if diagnostics:
        result["diagnostics"] = {
            "versus_champion": compare_diagnostics(
                challenger, champion, [heuristic_v2] * 3, rounds, seed),
            "versus_v2": compare_diagnostics(
                challenger, heuristic_v2, [heuristic_v2] * 3,
                rounds, seed + 100000),
        }
    return result


def append_history(path, result):
    """Append a complete, machine-readable evaluation record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"evaluated_at": datetime.now(timezone.utc).isoformat(), **result}
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1001000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run additional per-seat, penalties, locks and outcome analysis")
    parser.add_argument("--history", type=Path, default=Path("runs/phase10_evaluations.jsonl"))
    args = parser.parse_args(argv)
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if not args.candidate.is_file():
        parser.error(f"checkpoint does not exist: {args.candidate}")
    result = evaluate_checkpoint(args.candidate, args.rounds, args.seed, args.confidence,
                                 diagnostics=args.diagnostics)
    append_history(args.history, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
