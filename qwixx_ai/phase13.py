"""Phase 13: frozen-baseline fine-tuning and independent multi-lineup evaluation.

Train via --mode train, then evaluate with --mode evaluate. No promotion.
Never use final confirmation seeds for training or candidate selection.
"""
import argparse
import json
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .train import train


def validate_seed_ranges(train_seed, episodes, tune_seed, tune_rounds,
                         confirm_seed, confirm_rounds):
    ranges = [
        set(range(train_seed, train_seed + episodes)),
        set(range(tune_seed, tune_seed + tune_rounds)),
        set(range(confirm_seed, confirm_seed + confirm_rounds)),
    ]
    if any(ranges[i] & ranges[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training, tuning and confirmation seed ranges overlap")


def evaluate(candidate, baseline, rounds=24, seed=13020001):
    if rounds < 2:
        raise ValueError("rounds must be at least 2")
    candidate_model = load(candidate)
    baseline_model = load(baseline)
    candidate_policy = lambda game: choose(candidate_model, game, epsilon=0)
    baseline_policy = lambda game: choose(baseline_model, game, epsilon=0)
    lineups = {
        "v2": [heuristic_v2] * 3,
        "mixed": [heuristic_v2, champion, baseline_policy],
        "champion_heavy": [champion] * 3,
    }
    results = {}
    for index, (name, opponents) in enumerate(lineups.items()):
        # Each lineup has its own seed block, shared between candidate and reference.
        results[name] = {
            "versus_neural_baseline": paired_compare(
                candidate_policy, baseline_policy, opponents,
                rounds=rounds, seed=seed + index * rounds),
            "versus_phase91_champion": paired_compare(
                candidate_policy, champion, opponents,
                rounds=rounds, seed=seed + index * rounds),
        }
    return results


def run_training(source, out, episodes=250, seed=13000001,
                 learning_rate=0.00001, anchor_weight=0.2,
                 win_weight=2.0, relative_weight=0.5):
    source, out = Path(source), Path(out)
    if not source.is_file():
        raise FileNotFoundError(source)
    if episodes < 1 or learning_rate <= 0 or min(anchor_weight, win_weight, relative_weight) < 0:
        raise ValueError("invalid training parameters")
    if out.resolve() == source.parent.resolve() or source.resolve().is_relative_to(out.resolve()):
        raise ValueError("output may not contain or overwrite source checkpoint")
    train(
        episodes=episodes, seed=seed, out=str(out),
        init_checkpoint=source, learning_rate=learning_rate,
        eval_games=40, eval_interval=max(1, episodes // 5),
        terminal_win_weight=win_weight,
        relative_score_weight=relative_weight,
        finetune_anchor_weight=anchor_weight,
        competitive_rounds=0,
    )
    return out / "latest.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "confirm"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/phase13"))
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--train-seed", type=int, default=13000001)
    parser.add_argument("--tune-seed", type=int, default=13020001)
    parser.add_argument("--confirm-seed", type=int, default=13100001)
    parser.add_argument("--tune-rounds", type=int, default=24)
    parser.add_argument("--confirm-rounds", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--win-weight", type=float, default=2.0)
    parser.add_argument("--relative-weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not args.baseline.is_file():
        parser.error("baseline checkpoint does not exist")
    if args.tune_rounds < 2 or args.confirm_rounds < 2 or args.episodes < 1:
        parser.error("episodes must be positive and rounds at least 2")
    try:
        validate_seed_ranges(args.train_seed, args.episodes, args.tune_seed,
                             args.tune_rounds * 3, args.confirm_seed,
                             args.confirm_rounds * 3)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "train":
        candidate = run_training(
            args.baseline, args.out / "training", args.episodes,
            args.train_seed, args.learning_rate, args.anchor_weight,
            args.win_weight, args.relative_weight)
        print(json.dumps({"phase": 13, "mode": "train",
                          "candidate": str(candidate), "promoted": False}, indent=2))
        return
    if args.candidate is None or not args.candidate.is_file():
        parser.error("--candidate must point to an existing checkpoint")
    if args.candidate.resolve() == args.baseline.resolve():
        parser.error("candidate and baseline must differ")
    rounds = args.tune_rounds if args.mode == "evaluate" else args.confirm_rounds
    seed = args.tune_seed if args.mode == "evaluate" else args.confirm_seed
    report = {
        "phase": 13, "mode": args.mode, "baseline": str(args.baseline),
        "candidate": str(args.candidate), "rounds_per_lineup": rounds,
        "seed": seed, "results": evaluate(args.candidate, args.baseline, rounds, seed),
        "promoted": False,
        "note": "Exploratory paired four-seat seed-cluster comparisons; "
                "no automatic selection or promotion. Confirmation seeds must "
                "remain untouched until a candidate is frozen.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / ("confirmation.json" if args.mode == "confirm" else "evaluation.json")
    if destination.exists():
        parser.error(f"refusing to overwrite existing evaluation: {destination}")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
