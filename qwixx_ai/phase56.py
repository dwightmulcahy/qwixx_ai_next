"""Phase 5.6: reproducible two-factor ablation of Phase 5.5 stabilizers.

Runs a 2x2 factorial design. All arms share the same training seed, initial
champion, historical opponents, and evaluation seed blocks. No model is
promoted by this study; its evaluations are exploratory comparisons.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .model import load
from .phase53 import evaluate_promotion

ARMS = (
    ("both", 0.05, 4),
    ("anchor_only", 0.05, 1),
    ("slow_updates_only", 0.0, 4),
    ("neither", 0.0, 1),
)


def training_command(args, name, weight, interval):
    """Build one arm without modifying shared training settings."""
    return [
        sys.executable, "-m", "qwixx_ai.train",
        "--episodes", str(args.episodes),
        "--eval-interval", str(args.eval_interval),
        "--eval-games", str(args.eval_games),
        "--target-interval", str(args.target_interval),
        "--champion", args.champion,
        "--historical", *args.historical,
        "--warm-start", "--phase5", "--phase52", "--phase53",
        "--phase54", "--phase55",
        "--learning-rate", str(args.learning_rate),
        "--anchor-weight", str(weight),
        "--update-every", str(interval),
        "--promotion-games", str(args.promotion_games),
        "--promotion-alpha", str(args.alpha),
        "--rollback-patience", str(args.rollback_patience),
        "--seed", str(args.seed),
        "--out", str(Path(args.out) / name),
    ]


def run(args):
    if args.episodes < 1 or args.eval_interval < 1 or args.rounds < 2:
        raise ValueError("episodes, eval interval and evaluation rounds must be positive")
    if len(args.historical) != 2:
        raise ValueError("exactly two historical checkpoints are required")
    if args.seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for name, weight, interval in ARMS:
        checkpoint = root / name / "best_candidate.pt"
        if not args.evaluate_only:
            command = training_command(args, name, weight, interval)
            print("TRAIN", name, " ".join(command), flush=True)
            subprocess.run(command, check=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint}: arm not trained or no candidate saved")

    champion = load(args.champion)
    history = [load(path) for path in args.historical]
    lineup = [champion, *history]
    results = {}
    for name, weight, interval in ARMS:
        candidate = load(root / name / "best_candidate.pt")
        evaluation = evaluate_promotion(
            candidate, champion, history, rounds=args.rounds,
            seed=args.validation_seed, alpha=args.alpha,
            max_checks=1, lineup=lineup,
        )
        results[name] = {
            "anchor_weight": weight,
            "update_every": interval,
            "checkpoint": str(root / name / "best_candidate.pt"),
            "evaluation": evaluation,
        }
        print("EVALUATED", name, evaluation["primary"]["paired_advantage"], flush=True)

    # Factorial contrasts are descriptive. The models share seeds and
    # opponents, but these contrasts do not have formal confidence intervals.
    advantage = {name: result["evaluation"]["primary"]["paired_advantage"]
                 for name, result in results.items()}
    contrasts = {
        "anchor_effect": ((advantage["both"] - advantage["slow_updates_only"])
                          + (advantage["anchor_only"] - advantage["neither"])) / 2,
        "slow_update_effect": ((advantage["both"] - advantage["anchor_only"])
                              + (advantage["slow_updates_only"] - advantage["neither"])) / 2,
        "interaction": (advantage["both"] - advantage["anchor_only"]
                        - advantage["slow_updates_only"] + advantage["neither"]),
    }
    report = {
        "design": "2x2_factorial",
        "training_seed": args.seed,
        "validation_seed": args.validation_seed,
        "rounds_per_comparison": args.rounds,
        "note": "Exploratory ablation. Do not treat this validation set as an untouched promotion holdout.",
        "arms": results,
        "primary_factorial_contrasts": contrasts,
    }
    destination = root / "ablation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(destination),
                      "primary_factorial_contrasts": contrasts}, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--champion", required=True)
    p.add_argument("--historical", nargs=2, required=True)
    p.add_argument("--out", default="runs/phase5_6")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--target-interval", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.00005)
    p.add_argument("--promotion-games", type=int, default=3000)
    p.add_argument("--rollback-patience", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--validation-seed", type=int, default=7890123)
    p.add_argument("--rounds", type=int, default=1250)
    p.add_argument("--evaluate-only", action="store_true",
                   help="Reuse trained arm checkpoints without retraining")
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
